# -*- coding: utf-8 -*-
"""
gemini_v3.py

目标：
1) 使用豆包 OpenSpeech 端到端语音（SAUC bigmodel / realtime dialogue）做语音识别（ASR）。
2) 将识别到的“用户问题”喂给本项目既有的教练链路：RAG 快报 -> Coach LLM（OpenRouter/Google/Gemini）-> TTS。
3) TTS 使用你配置的 speaker 音色包（沿用 gemini_v2 / TTSPlayer 的 --doubao-tts-speaker）。

说明（重要）：
- 本实现只把端到端模型用于“语音识别+事件流解析”，避免直接依赖 volcengine-audio SDK（你当前 Python 3.10 下该 SDK 会因 StrEnum 导致 import 失败）。
- 你需要正确配置 e2e 的鉴权参数（app_key/access_key/resource_id），这些字段会从环境变量或命令行读取。
- e2e 的鉴权方式可能因你控制台配置略有差异：如果连不上，把 `api_access_key` 改成文档要求的形式即可（通常是 `Jwt; <token>`）。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import struct
import threading
import time
import uuid
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import urlopen

import gemini_v1 as gv
import gemini_v2 as gv2


# ========= e2e（端到端实时语音识别）协议常量 =========
HOST = "openspeech.bytedance.com"

# protocol.py（volcengine_audio）里这几个是固定值：
PROTOCOL_VERSION_V1 = 0b0001
HEADER_SIZE_4 = 0b0001  # 4 bytes words

MESSAGE_TYPE_FULL_CLIENT_REQUEST = 0b0001
MESSAGE_TYPE_AUDIO_ONLY_REQUEST = 0b0010

MESSAGE_TYPE_SPECIFIC_FLAG_CARRY_EVENT_ID = 0b0100

SERIALIZATION_JSON_NO_COMP = 0b00010000
RESERVED = 0b00000000

# EventSend（端到端实时语音对话）
EVENT_START_CONNECTION = 1
EVENT_FINISH_CONNECTION = 2
EVENT_START_SESSION = 100
EVENT_TASK_REQUEST = 200
EVENT_FINISH_SESSION = 102

# EventReceive（用于解析响应事件）
EVENT_CONNECTION_STARTED = 50
EVENT_CONNECTION_FAILED = 51
EVENT_SESSION_STARTED = 150
EVENT_SESSION_FAILED = 153
EVENT_ASR_RESPONSE = 451
EVENT_ASR_ENDED = 459


def _coach_system_role_short() -> str:
    # 这里不要直接复用 gemini_v1 的巨长 prompt，避免触发 SDK 端到端配置校验上限（system_role+speaking_style <= 4000）。
    return "你是一个游戏教练，回答要简洁可执行。"


def _maybe_strip_sensevoice_tags(text: str) -> str:
    # gemini_v2 里对 SenseVoice 的后处理类似：去掉 <|zh|><|NEUTRAL|> 等控制标签
    text = (text or "").strip()
    text = re.sub(r"<\|[^|>]+\|>", "", text).strip()
    return text


def _asr_postprocess(text: str, *, simplify_zh: bool, glossary: Dict[str, str]) -> str:
    text = _maybe_strip_sensevoice_tags(text)
    if simplify_zh:
        try:
            import zhconv  # type: ignore
        except ImportError:
            pass
        else:
            text = zhconv.convert(text, "zh-cn")
    text = gv2._apply_asr_glossary(text, glossary)
    return text


def _build_full_client_payload(
    *,
    message_type: int,
    event_number: Optional[int],
    session_id: Optional[str],
    request_meta: Optional[Dict[str, Any]],
) -> bytes:
    """
    对应 volcengine_audio.realtime.RealtimeDialogueFunctions._calculate_payload
    """
    # 4 bytes header
    b0 = (PROTOCOL_VERSION_V1 << 4) | HEADER_SIZE_4
    b1 = (message_type << 4) | MESSAGE_TYPE_SPECIFIC_FLAG_CARRY_EVENT_ID
    b2 = SERIALIZATION_JSON_NO_COMP
    b3 = RESERVED
    payload = bytes([b0, b1, b2, b3])

    if event_number is not None:
        payload += struct.pack(">I", int(event_number))

    if session_id:
        sid_b = session_id.encode("utf-8")
        payload += struct.pack(">I", len(sid_b))
        payload += sid_b

    meta_dict = request_meta or {}
    meta_b = json.dumps(meta_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload += struct.pack(">I", len(meta_b))
    payload += meta_b

    return payload


def _build_task_request_payload(*, session_id: str, audio_data: bytes) -> bytes:
    """
    对应 volcengine_audio.realtime.RealtimeDialogueFunctions.task_request_payload
    """
    b0 = (PROTOCOL_VERSION_V1 << 4) | HEADER_SIZE_4
    b1 = (MESSAGE_TYPE_AUDIO_ONLY_REQUEST << 4) | MESSAGE_TYPE_SPECIFIC_FLAG_CARRY_EVENT_ID
    b2 = SERIALIZATION_JSON_NO_COMP
    b3 = RESERVED
    payload = bytes([b0, b1, b2, b3])

    payload += struct.pack(">I", EVENT_TASK_REQUEST)

    sid_b = session_id.encode("utf-8")
    payload += struct.pack(">I", len(sid_b))
    payload += sid_b

    payload += struct.pack(">I", len(audio_data))
    payload += audio_data
    return payload


def _parse_ws_server_packet(data: bytes) -> Tuple[int, str, Optional[Any]]:
    """
    解析 volcengine_audio.tts.VolcengineTTSFunctions.extract_response_payload 同风格的协议帧，
    但我们只关心 event_number、session_id 和 JSON payload。
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 12:
        raise ValueError("invalid packet")

    header_byte0 = data[0]
    _protocol_version = (header_byte0 >> 4) & 0b1111
    _header_size_words = header_byte0 & 0b1111

    message_type_byte = data[1]
    _message_type = (message_type_byte >> 4) & 0b1111

    serialization_byte = data[2]
    serialization_method = (serialization_byte >> 4) & 0b1111  # 1 = JSON in our protocol

    optional_data = data[4:8]
    if len(optional_data) != 4:
        raise ValueError("packet too short for optional_data")
    event_number = struct.unpack(">I", optional_data)[0]

    session_len = struct.unpack(">I", data[8:12])[0]
    session_end = 12 + session_len
    if session_end + 4 > len(data):
        raise ValueError("packet too short for session_end")
    session_id = data[12:session_end].decode("utf-8", errors="ignore")

    # 跳过 payload_size(4 bytes) 后面的 payload
    payload = data[session_end + 4 :]
    if not payload:
        return event_number, session_id, None

    if serialization_method == 0b0001:  # JSON
        try:
            obj = json.loads(payload.decode("utf-8", errors="ignore"))
        except Exception:
            obj = None
        return event_number, session_id, obj

    # RAW：不做解析
    return event_number, session_id, None


class E2EAsrClient:
    """
    使用端到端实时语音识别的 websocket（SAUC bigmodel）做“整段语音 -> 最终文本”。
    """

    def __init__(
        self,
        *,
        app_key: str,
        access_key: str,
        resource_id: str = "volc.bigasr.sauc.duration",
        ws_endpoint: str = "/api/v3/sauc/bigmodel_nostream",
        # 你也可以在这里直接填已经带 Jwt 前缀的 api_access_key：
        # 示例： "Jwt; <jwt_token>"
        api_access_key: str = "",
        api_app_key: str = "",
        api_resource_id: str = "",
        jwt_sts_appid: Optional[str] = None,
        jwt_sts_access_key: Optional[str] = None,
        jwt_sts_duration_sec: int = 300,
        sts_url: str = "https://openspeech.bytedance.com/api/v1/sts/token",
        timeout_sec: float = 60.0,
    ) -> None:
        self.app_key = (app_key or "").strip()
        self.access_key = (access_key or "").strip()
        self.resource_id = (resource_id or "").strip()
        self.ws_endpoint = (ws_endpoint or "").strip()

        self.api_access_key = (api_access_key or "").strip()
        self.api_app_key = (api_app_key or "").strip()
        self.api_resource_id = (api_resource_id or "").strip()

        self.jwt_sts_appid = jwt_sts_appid
        self.jwt_sts_access_key = jwt_sts_access_key
        self.jwt_sts_duration_sec = int(jwt_sts_duration_sec or 300)
        self.sts_url = sts_url

        self.timeout_sec = max(10.0, float(timeout_sec or 60.0))

    def _resolve_jwt_api_access_key(self) -> str:
        # 优先使用用户显式提供的 api_access_key
        if self.api_access_key:
            return self.api_access_key

        # 若未提供，则尝试把 access_key 当作 JWT（或 accessKey 本身就是所需 token）
        # 常见格式要求：Jwt; <token>（注意空格）
        ak = self.access_key
        if not ak:
            return ""
        if ak.startswith("Jwt;"):
            return ak
        if " " in ak and ak.lower().startswith("jwt"):
            return ak
        return f"Jwt; {ak}"

    def _resolve_api_app_key(self) -> str:
        return self.api_app_key or self.app_key

    def _resolve_api_resource_id(self) -> str:
        return self.api_resource_id or self.resource_id

    def _maybe_get_jwt_token_via_sts(self) -> Optional[str]:
        # 若用户提供 sts 需要的字段，则走 STS 获取 jwt_token，再组装 api_access_key。
        if not self.jwt_sts_appid or not self.jwt_sts_access_key:
            return None

        try:
            import requests  # type: ignore
        except ImportError as e:
            raise SystemExit("未安装 requests。请执行： pip install requests") from e

        headers = {
            "Authorization": f"Bearer; {self.jwt_sts_access_key}",
            "Content-Type": "application/json",
        }
        body = {"appid": self.jwt_sts_appid, "duration": self.jwt_sts_duration_sec}
        r = requests.post(self.sts_url, headers=headers, json=body, timeout=min(20.0, self.timeout_sec))
        r.raise_for_status()
        obj = r.json()
        # 返回字段在不同文档可能略有差异
        return obj.get("jwt_token") or obj.get("token") or obj.get("access_token")

    def _build_ws_url(self) -> str:
        # 如果 sts 能拿到 jwt_token，则优先拼接。
        jwt = self._maybe_get_jwt_token_via_sts()
        api_access_key = self.api_access_key
        if not api_access_key and jwt:
            api_access_key = f"Jwt; {jwt}"

        if not api_access_key:
            api_access_key = self._resolve_jwt_api_access_key()

        api_app_key = self._resolve_api_app_key()
        api_resource_id = self._resolve_api_resource_id()

        if not api_app_key or not api_access_key or not api_resource_id:
            raise RuntimeError(
                "[gemini_v3][e2e] 缺少鉴权信息：需要 api_app_key/api_access_key/api_resource_id。"
            )

        # query 参数（WebSocket 无法自定义 header，所以用 query）
        # 参考：文档/示例里使用 api_resource_id, api_app_key, api_access_key
        return (
            f"wss://{HOST}{self.ws_endpoint}"
            f"?api_resource_id={quote(api_resource_id)}"
            f"&api_app_key={quote(api_app_key)}"
            f"&api_access_key={quote(api_access_key)}"
        )

    def transcribe_pcm_s16le(
        self,
        *,
        pcm_s16le_bytes: bytes,
        session_id: Optional[str] = None,
        dialog_system_role: str = "",
        tts_speaker: str = "",
    ) -> str:
        """
        返回：识别到的最终文本（尽量选择非 interim 的结果）
        """
        try:
            import websocket  # type: ignore
        except ImportError as e:
            raise SystemExit("流式 e2e ASR 需要 websocket-client： pip install websocket-client") from e

        ws_url = self._build_ws_url()
        sid = session_id or str(uuid.uuid4())

        # StartConnection
        payload_start_conn = _build_full_client_payload(
            message_type=MESSAGE_TYPE_FULL_CLIENT_REQUEST,
            event_number=EVENT_START_CONNECTION,
            session_id=None,
            request_meta={},
        )

        # StartSession config（只要能通过校验即可）
        asr_audio_info = {"format": "pcm", "sample_rate": 16000, "channel": 1}
        config: Dict[str, Any] = {
            "dialog": {"bot_name": "豆包"},
            "asr": {"audio_info": asr_audio_info},
        }
        if dialog_system_role:
            config["dialog"]["system_role"] = dialog_system_role
        if tts_speaker:
            config["tts"] = {"speaker": tts_speaker}

        payload_start_session = _build_full_client_payload(
            message_type=MESSAGE_TYPE_FULL_CLIENT_REQUEST,
            event_number=EVENT_START_SESSION,
            session_id=sid,
            request_meta=config,
        )

        payload_task = _build_task_request_payload(session_id=sid, audio_data=pcm_s16le_bytes)

        # 用于捕获最终文本
        latest_interim = ""
        latest_final = ""

        ws = websocket.create_connection(ws_url, timeout=self.timeout_sec)
        ws.settimeout(self.timeout_sec)
        try:
            # 连接建立后按序发送
            ws.send_binary(payload_start_conn)
            ws.send_binary(payload_start_session)
            # 尽量等待 SessionStarted 再发送音频（提升稳定性）
            try:
                t0_wait = time.perf_counter()
                while time.perf_counter() - t0_wait < min(10.0, self.timeout_sec):
                    raw0 = ws.recv()
                    if not isinstance(raw0, (bytes, bytearray)) or not raw0:
                        continue
                    try:
                        evn0, _sid0, _obj0 = _parse_ws_server_packet(raw0)
                    except Exception:
                        continue
                    if evn0 == EVENT_SESSION_STARTED:
                        break
                # 再发送任务
            except Exception:
                # 不强制：服务端可能直接就绪/不发 SessionStarted
                pass
            ws.send_binary(payload_task)

            # 读回事件流
            # 由于服务端细节可能有差异：我们直到收到 ASR 结束事件或超过时间。
            t0 = time.perf_counter()
            got_end = False
            while time.perf_counter() - t0 < self.timeout_sec:
                try:
                    raw = ws.recv()
                except Exception:
                    break

                if not isinstance(raw, (bytes, bytearray)) or not raw:
                    continue
                try:
                    event_number, _sess_id, obj = _parse_ws_server_packet(raw)
                except Exception:
                    continue

                if event_number == EVENT_CONNECTION_FAILED:
                    raise RuntimeError(f"[gemini_v3][e2e] 连接失败（event={event_number}）")

                if event_number == EVENT_ASR_RESPONSE and isinstance(obj, dict):
                    results = obj.get("results")
                    if isinstance(results, list):
                        # results: [{text, is_interim}, ...]
                        finals: List[str] = []
                        interims: List[str] = []
                        for r in results:
                            if not isinstance(r, dict):
                                continue
                            txt = str(r.get("text") or "").strip()
                            if not txt:
                                continue
                            if bool(r.get("is_interim")):
                                interims.append(txt)
                            else:
                                finals.append(txt)
                        if finals:
                            latest_final = " ".join(finals).strip()
                        if interims:
                            latest_interim = " ".join(interims).strip()

                if event_number == EVENT_ASR_ENDED:
                    got_end = True
                    break

            # 结束时优先 final
            return (latest_final or latest_interim or "").strip()
        finally:
            try:
                ws.send_binary(
                    _build_full_client_payload(
                        message_type=MESSAGE_TYPE_FULL_CLIENT_REQUEST,
                        event_number=EVENT_FINISH_SESSION,
                        session_id=sid,
                        request_meta={},
                    )
                )
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass


class E2EVoiceSession:
    """
    端到端 e2e ASR：按空格开始录音，回车结束并识别（不显示实时转写）。
    """

    SR = 16000

    def __init__(
        self,
        *,
        simplify_zh: bool,
        asr_glossary: Dict[str, str],
        e2e_asr_app_key: str,
        e2e_asr_access_key: str,
        e2e_asr_resource_id: str,
        e2e_ws_endpoint: str,
        tts_speaker: str,
        timeout_sec: float,
        mic_device: Optional[int] = None,
        jwt_sts_appid: Optional[str] = None,
        jwt_sts_access_key: Optional[str] = None,
    ) -> None:
        self.simplify_zh = bool(simplify_zh)
        self.asr_glossary = dict(asr_glossary or {})
        self.tts_speaker = (tts_speaker or "").strip()
        self.mic_device = mic_device
        self._client = E2EAsrClient(
            app_key=e2e_asr_app_key,
            access_key=e2e_asr_access_key,
            resource_id=e2e_asr_resource_id,
            ws_endpoint=e2e_ws_endpoint,
            timeout_sec=timeout_sec,
            jwt_sts_appid=jwt_sts_appid,
            jwt_sts_access_key=jwt_sts_access_key,
        )

        self._lock = threading.Lock()
        self._recording = False
        self._chunks: List[bytes] = []
        self._submit = threading.Event()

        self._listener = None
        self._stream = None
        self._audio_bytes: bytes = b""

    def run(self, banner: str) -> str:
        try:
            import numpy as np
            import sounddevice as sd
            from pynput import keyboard
        except ImportError as e:
            raise SystemExit(
                "语音依赖未就绪。请执行： pip install sounddevice numpy pynput websocket-client requests"
                + "\n"
                + str(e)
            ) from e

        # 为了避免框架/操作系统对回调/线程的限制：录音与识别分离。
        self._chunks = []
        self._submit.clear()

        def audio_cb(indata, frames, tcb, status) -> None:
            if not self._recording:
                return
            # indata: (frames, 1) int16
            if hasattr(indata, "copy"):
                buf = indata.copy().tobytes()
            else:
                buf = np.asarray(indata).astype("<i2").tobytes()
            self._chunks.append(buf)

        def _resolve_input_device() -> Optional[int]:
            """
            部分机器 sounddevice 的默认输入 device=-1 会直接抛 PortAudioError。
            这里优先用用户指定，其次用默认输入，最后扫描第一个可输入的设备。
            """
            # 1) 用户强制指定优先
            if self.mic_device is not None:
                try:
                    info = sd.query_devices(self.mic_device)
                    if (info.get("max_input_channels") or 0) > 0:
                        return int(self.mic_device)
                except Exception:
                    pass

            # 2) 默认输入设备
            try:
                default_dev = sd.default.device
                default_in = (
                    default_dev[0]
                    if isinstance(default_dev, (tuple, list)) and len(default_dev) >= 1
                    else default_dev
                )
                if isinstance(default_in, int) and default_in >= 0:
                    info = sd.query_devices(default_in)
                    if (info.get("max_input_channels") or 0) > 0:
                        return default_in
            except Exception:
                pass

            # 3) 扫描所有设备：找第一个 max_input_channels>0 的
            try:
                devices = sd.query_devices()
            except Exception:
                return None

            for idx, dev in enumerate(devices):
                try:
                    if (dev.get("max_input_channels") or 0) > 0:
                        return idx
                except Exception:
                    continue
            return None

        input_device = _resolve_input_device()
        if input_device is None:
            # 兜底：给一点可操作的诊断信息
            detail_lines: List[str] = []
            try:
                for idx, dev in enumerate(sd.query_devices()):
                    mic = dev.get("max_input_channels") or 0
                    if mic:
                        name = str(dev.get("name") or "").strip()
                        detail_lines.append(f"{idx}: {name} (max_input_channels={mic})")
            except Exception:
                detail_lines = []

            detail = "\n".join(detail_lines[:20])
            raise SystemExit(
                "[gemini_v3][语音] 未找到可用麦克风输入设备（max_input_channels>0）。"
                + ("\n可用输入设备（前 20 个）：\n" + detail if detail else "\n请检查麦克风驱动/权限，然后用 --mic-device 指定设备索引。")
            )

        self._stream = sd.InputStream(
            samplerate=self.SR,
            channels=1,
            dtype="int16",
            callback=audio_cb,
            blocksize=1024,
            device=input_device,
        )
        self._stream.start()

        def on_press(key) -> None:
            if key == keyboard.Key.space:
                with self._lock:
                    if self._recording:
                        return
                    self._recording = True
                    self._chunks = []
            elif key == keyboard.Key.enter:
                with self._lock:
                    if not self._recording:
                        self._submit.set()
                        return
                    self._recording = False
                    frames_bytes = b"".join(self._chunks)
                    self._audio_bytes = frames_bytes
                    self._submit.set()
                    # 立刻停掉采集线程
                    try:
                        if self._listener is not None:
                            self._listener.stop()
                    except Exception:
                        pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

        print(banner, flush=True)
        print(
            "操作：按一次【空格】开始录音；【回车】结束录音并提交（语音识别整段完成，不显示实时转写）。",
            flush=True,
        )

        try:
            self._submit.wait()
        except KeyboardInterrupt:
            self._submit.set()
        finally:
            try:
                if self._listener is not None:
                    self._listener.stop()
            except Exception:
                pass
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass

        if not self._audio_bytes:
            return ""

        # 端到端 ASR：整段发送
        raw_text = self._client.transcribe_pcm_s16le(
            pcm_s16le_bytes=self._audio_bytes,
            session_id=str(uuid.uuid4()),
            dialog_system_role="",  # ASR 用不到系统角色；保守起见留空
            tts_speaker=self.tts_speaker,
        )
        out = _asr_postprocess(
            raw_text,
            simplify_zh=self.simplify_zh,
            glossary=self.asr_glossary,
        )
        if out:
            print(f"[识别] {out}", flush=True)
        return out


def _build_argparser() -> argparse.ArgumentParser:
    ap = gv2._build_argparser()
    ap.description = "（v3：e2e 端到端语音 ASR -> Coach LLM -> 豆包 TTS）"

    # e2e（端到端语音识别）鉴权参数（按文档使用 api_app_key/api_access_key/api_resource_id）
    ap.add_argument(
        "--doubao-e2e-app-key",
        default="",
        help="端到端 ASR：api_app_key（不填则读 DOUBAO_E2E_APP_KEY）",
    )
    ap.add_argument(
        "--doubao-e2e-access-key",
        default="",
        help="端到端 ASR：api_access_key 的原始输入（不填则读 DOUBAO_E2E_ACCESS_KEY）",
    )
    ap.add_argument(
        "--doubao-e2e-resource-id",
        default="volc.bigasr.sauc.duration",
        help="端到端 ASR：api_resource_id（默认 volc.bigasr.sauc.duration）",
    )
    ap.add_argument(
        "--doubao-e2e-ws-endpoint",
        default="/api/v3/sauc/bigmodel_nostream",
        help="端到端 ASR：WebSocket endpoint（默认 /api/v3/sauc/bigmodel_nostream）",
    )
    ap.add_argument(
        "--doubao-e2e-timeout",
        type=float,
        default=60.0,
        help="端到端 ASR 超时秒数（默认 60）",
    )

    # 可选：如果你手里有最终 api_access_key（例如已是 Jwt; xxx），可直接通过环境变量/命令行提供
    ap.add_argument(
        "--doubao-e2e-access-key-jwt",
        default="",
        help="端到端 ASR：直接提供完整 api_access_key（例如 'Jwt; <jwt_token>'），覆盖前面的 access-key 输入",
    )

    ap.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="输入设备索引（sounddevice 的 device index）。不填则自动选择可用麦克风。",
    )

    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()
    io_lock = threading.Lock()

    # 合并 ASR glossary（复用 gemini_v2 逻辑）
    asr_glossary = gv2._merge_asr_glossaries(
        args.asr_glossary.resolve() if getattr(args, "asr_glossary", None) else None
    )

    # 是否走语音输入（沿用 gemini_v2 的判定）
    use_voice = bool(
        os.isatty(0)  # stdin
        and not bool(getattr(args, "no_voice", False))
        and not (getattr(args, "question", "") or "").strip()
    )

    tts = gv2.TTSPlayer(args)

    voice: Optional[E2EVoiceSession] = None
    if use_voice:
        # TTS speaker 直接复用（你的 speaker 音色包）
        speaker = getattr(args, "doubao_tts_speaker", "") or os.getenv("DOUBAO_TTS_SPEAKER", "")
        speaker = (speaker or "").strip()

        e2e_app_key = getattr(args, "doubao_e2e_app_key", "") or os.getenv("DOUBAO_E2E_APP_KEY", "")
        e2e_access_key = (
            getattr(args, "doubao_e2e_access_key", "") or os.getenv("DOUBAO_E2E_ACCESS_KEY", "")
        )
        # 可选：直接提供完整 Jwt; token
        jwt_api_access_key = getattr(args, "doubao_e2e_access_key_jwt", "") or os.getenv(
            "DOUBAO_E2E_API_ACCESS_KEY", ""
        )

        # 如果用户提供了完整 jwt api_access_key，就把它塞给 E2EAsrClient 的 access_key 输入。
        # E2EAsrClient 内部会识别前缀 Jwt;，不会重复加。
        use_direct_jwt = bool(jwt_api_access_key)
        if use_direct_jwt:
            e2e_access_key = jwt_api_access_key

        e2e_resource_id = getattr(args, "doubao_e2e_resource_id", "") or os.getenv(
            "DOUBAO_E2E_RESOURCE_ID", "volc.bigasr.sauc.duration"
        )
        e2e_ws_endpoint = getattr(args, "doubao_e2e_ws_endpoint", "") or os.getenv(
            "DOUBAO_E2E_WS_ENDPOINT", "/api/v3/sauc/bigmodel_nostream"
        )
        timeout_sec = float(getattr(args, "doubao_e2e_timeout", 60.0) or 60.0)

        mic_device = getattr(args, "mic_device", None)
        if mic_device is None:
            env_md = (os.getenv("DOUBAO_MIC_DEVICE", "") or "").strip()
            if env_md:
                try:
                    mic_device = int(env_md)
                except Exception:
                    mic_device = None

        if not e2e_app_key or not e2e_access_key:
            raise SystemExit(
                "[gemini_v3][e2e] 缺少 e2e 鉴权信息：请设置 DOUBAO_E2E_APP_KEY / DOUBAO_E2E_ACCESS_KEY（或用命令行 --doubao-e2e-*）。"
            )

        voice = E2EVoiceSession(
            simplify_zh=not bool(getattr(args, "no_zh_simplify", False)),
            asr_glossary=asr_glossary,
            e2e_asr_app_key=str(e2e_app_key),
            e2e_asr_access_key=str(e2e_access_key),
            e2e_asr_resource_id=str(e2e_resource_id),
            e2e_ws_endpoint=str(e2e_ws_endpoint),
            tts_speaker=speaker,
            timeout_sec=timeout_sec,
            mic_device=mic_device,
            # 若用户没直接提供 Jwt;token，则用 appid/accessKey 去 STS 拿临时 jwt_token，
            # 避免把 accessKey 当成 jwt_token 直接拼 Jwt; 前缀。
            jwt_sts_appid=None if use_direct_jwt else str(e2e_app_key),
            jwt_sts_access_key=None if use_direct_jwt else str(e2e_access_key),
        )

    # 读取问题：复用 gemini_v2 的判定方式（语音优先，否则 stdin）
    def read_question_from_voice_or_text(hold_io_lock: bool) -> str:
        q0 = (getattr(args, "question", "") or "").strip()
        if q0:
            return q0
        if not os.isatty(0):
            return sys.stdin.read().strip()
        if use_voice and voice is not None:
            if hold_io_lock:
                io_lock.acquire()
            try:
                return voice.run("【语音】要问随风听笛什么？")
            finally:
                if hold_io_lock:
                    io_lock.release()
        return input("你有什么要问的：").strip()

    question = (getattr(args, "question", "") or "").strip()
    summary_path: Optional[Path] = None
    bundle_for_coach: Optional[Dict[str, Any]] = None

    # ====== summary_json / pipeline 分支（复用 gemini_v2 的核心逻辑）======
    if getattr(args, "summary_json", None):
        summary_path = Path(args.summary_json).resolve()
        if not summary_path.is_file():
            raise SystemExit(f"找不到文件: {summary_path}")
        if not question:
            question = read_question_from_voice_or_text(hold_io_lock=False)
        if not question:
            raise SystemExit("未提供问题：请使用 --question / -q，或语音/键盘输入。")
    else:
        img_dir = getattr(args, "img_dir", None)
        img_dir = img_dir.resolve() if img_dir else gv.DEFAULT_IMG_DIR
        if not img_dir.is_dir():
            raise SystemExit(f"截图目录不存在: {img_dir}")
        if not gv._dir_has_screenshot(img_dir):
            raise SystemExit(
                f"目录内无截图: {img_dir}\n"
                f"请放入主图 -a / 辅图 -b，或使用 --summary-json 指定已有 *_summary.json"
            )

        cache_dir = gv.PIPELINE_CACHE_DIR.resolve()
        use_cache = (
            not bool(getattr(args, "force_pipeline", False))
            and cache_dir.is_dir()
            and any(cache_dir.glob("*_summary.json"))
        )

        if use_cache:
            summary_path = gv._find_first_summary_json(cache_dir)
            print(f"[gemini_v3] 使用缓存: {cache_dir}（跳过 pipeline）", flush=True)
            if not question:
                question = read_question_from_voice_or_text(hold_io_lock=False)
            if not question:
                raise SystemExit("未提供问题：请使用 --question / -q，或语音/键盘输入。")
        else:
            out_dir = getattr(args, "pipeline_out", None).resolve()
            pipe_err: List[Optional[Exception]] = [None]
            pipe_data: Dict[str, Any] = {"cap": None, "wall": 0.0}
            done = threading.Event()
            quiet = not bool(getattr(args, "pipeline_verbose", False))

            def _pipeline_worker() -> None:
                t0 = time.perf_counter()
                try:
                    pipe_data["cap"] = gv2.run_pipeline(img_dir, out_dir, quiet=quiet)
                except Exception as e:
                    pipe_err[0] = e
                finally:
                    pipe_data["wall"] = time.perf_counter() - t0
                    done.set()

            th = threading.Thread(target=_pipeline_worker, name="pipeline", daemon=True)
            th.start()

            st: Optional[threading.Thread] = None
            if quiet:
                st = threading.Thread(
                    target=gv._pipeline_quiet_progress_until,
                    args=(done, "对局 JSON 分析中", io_lock),
                    name="pipeline_progress",
                    daemon=True,
                )
                st.start()

            if os.isatty(1):
                print(
                    "-" * 60
                    + "\n【提示】对局识别已在后台运行；语音：空格开始录，回车结束并识别。\n"
                    + "-" * 60,
                    flush=True,
                )

            if not question:
                question = read_question_from_voice_or_text(hold_io_lock=True) if os.isatty(0) else input().strip()

            if not question:
                raise SystemExit("未提供问题：请使用 --question / -q，或语音/键盘输入。")

            th.join()
            if st is not None:
                st.join(timeout=5.0)
            if pipe_err[0] is not None:
                raise pipe_err[0]

            if quiet and pipe_data.get("cap") is not None:
                cap = pipe_data["cap"]
                merged = (cap[0] or "") + "\n" + (cap[1] or "")
                print("\n[gemini_v3] 对局分析耗时（摘自子进程捕获日志）")
                for ln in gv._pipeline_timing_report_lines(merged, float(pipe_data["wall"])):
                    print(ln)

            summary_path = gv._find_first_summary_json(out_dir)

    assert summary_path is not None
    if not question:
        raise SystemExit("未提供问题：请使用 --question / -q，或语音/键盘输入。")

    # ====== 追问逻辑 ======
    follow_up: Optional[Callable[[], str]] = None
    if use_voice and voice is not None and os.isatty(0):

        def follow_up() -> str:
            print(
                "\n—— 追问：【空格】开始录（无实时转写），【回车】结束并识别；"
                "若随风听笛已答完你想结束，回车空提交或说 q / quit。",
                flush=True,
            )
            return voice.run("【语音追问】")

    gv2.run_coach_v2_after_summary(
        args,
        summary_path,
        question,
        io_lock,
        follow_up_reader=follow_up,
        log_prefix="gemini_v3",
        coach_bundle=bundle_for_coach,
        skip_bundle_preview_print=bundle_for_coach is not None,
        on_answer=tts.speak,
    )


if __name__ == "__main__":
    main()

