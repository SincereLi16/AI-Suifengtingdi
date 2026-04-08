# -*- coding: utf-8 -*-
"""
gemini_v3.py

目标：
1) 使用豆包 OpenSpeech 端到端语音（SAUC bigmodel / realtime dialogue）做语音识别（ASR）。
2) 将识别到的“用户问题”喂给本项目既有的教练链路：RAG 快报 -> Coach LLM（OpenRouter/Google/Gemini）-> TTS。
3) TTS 使用你配置的 speaker 音色包（沿用 gemini_v2 / TTSPlayer 的 --doubao-tts-speaker）。

说明（重要）：
- 本实现只把端到端模型用于“语音识别+事件流解析”，避免直接依赖 volcengine-audio SDK（你当前 Python 3.10 下该 SDK 会因 StrEnum 导致 import 失败）。
- e2e 与 eg/ 一致：`wss://.../api/v3/realtime/dialogue` + Header 鉴权。
- `X-Api-App-Key` 为官方固定产品密钥（默认 `PlgvMymc7f3tQnJ6`），勿把控制台数字应用 ID 当作 App-Key；应用 ID 填 `DOUBAO_E2E_APP_ID` 或把纯数字写在 `DOUBAO_E2E_APP_KEY` 时会自动作为 `X-Api-App-ID`。
- `DOUBAO_E2E_ACCESS_KEY` + `DOUBAO_E2E_APP_KEY`（STS 的 appid）换 Jwt；`DOUBAO_E2E_RESOURCE_ID` 默认 `volc.speech.dialog`。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import queue
import re
import socket
import struct
import threading
import time
import uuid
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import urlopen

import gemini_v1 as gv
import gemini_v2 as gv2


# ========= e2e（端到端实时语音识别）协议常量 =========
HOST = "openspeech.bytedance.com"
DEFAULT_E2E_STS_URL = "https://openspeech.bytedance.com/api/v1/sts/token"
# 与 eg/config.py 一致：实时对话 WebSocket（Header 鉴权，不用 query）
DIALOGUE_WS_PATH = "/api/v3/realtime/dialogue"
DEFAULT_DIALOGUE_RESOURCE_ID = "volc.speech.dialog"
# eg/config.py 里 X-Api-App-Key 固定值；与控制台「应用 ID」数字不是同一字段
DEFAULT_DIALOGUE_X_API_APP_KEY = "PlgvMymc7f3tQnJ6"

# protocol.py（volcengine_audio）里这几个是固定值：
PROTOCOL_VERSION_V1 = 0b0001
HEADER_SIZE_4 = 0b0001  # 4 bytes words

MESSAGE_TYPE_FULL_CLIENT_REQUEST = 0b0001
MESSAGE_TYPE_AUDIO_ONLY_REQUEST = 0b0010

MESSAGE_TYPE_SPECIFIC_FLAG_CARRY_EVENT_ID = 0b0100

RESERVED = 0b00000000

# 与 eg/protocol.py 官方示例一致：全量请求为 JSON+GZIP；音频帧为「无序列化+GZIP（pcm 先 gzip）」
_PROTO_SERIAL_JSON = 0b0001
_PROTO_SERIAL_RAW = 0b0000
_PROTO_COMP_GZIP = 0b0001
_PROTO_COMP_NONE = 0b0000

# 服务端消息类型（parse_response）
_SERVER_FULL_RESPONSE = 0b1001
_SERVER_ACK = 0b1011
_SERVER_ERROR_RESPONSE = 0b1111
_SERVER_MSG_FLAG_EVENT = 0b0100
_SERVER_MSG_FLAG_SEQ = 0b0010

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
# 与 eg/audio_manager.receive_loop 一致：152/153 表示会话侧结束（152 常为正常收尾）
EVENT_SESSION_FINISHED = 152
EVENT_SESSION_FAILED = 153
EVENT_ASR_RESPONSE = 451
EVENT_ASR_ENDED = 459
EVENT_CHAT_RESPONSE = 550  # 部分链路 ASR/对话文本在此事件
# 与 eg/realtime_dialog_client 一致
EVENT_SAY_HELLO = 300
EVENT_CHAT_TEXT_QUERY = 501
# eg/audio_manager：非 audio_file 时，359 表示开场 TTS 结束，之后才能正常对话
EVENT_TTS_PLAY_DONE = 359

# 与 eg/config.py input_audio_config["chunk"] 一致：每块 3200 帧；16-bit 单声道 -> 6400 字节/块
E2E_DIALOGUE_PCM_CHUNK_FRAMES = 3200
E2E_DIALOGUE_INPUT_SAMPLE_RATE = 16000


def _e2e_ws_exc_is_timeout(e: BaseException) -> bool:
    en = type(e).__name__
    return "timeout" in en.lower() or "Timeout" in en


def _extract_e2e_reply_text(obj: Any) -> str:
    """从服务端 JSON payload 中尽量抽出可读回复（ASR/对话混用）。"""
    if isinstance(obj, str) and obj.strip():
        return obj.strip()
    if not isinstance(obj, dict):
        return ""
    for k in ("content", "text", "reply", "answer", "asr_text", "result", "message"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    results = obj.get("results")
    if isinstance(results, list):
        parts: List[str] = []
        for r in results:
            if isinstance(r, dict):
                t = str(r.get("text") or "").strip()
                if t:
                    parts.append(t)
        if parts:
            return " ".join(parts).strip()
    return ""


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


def _hint_e2e_resource_speaker_mismatch(payload_msg: Any) -> str:
    """服务端报 resource 与 speaker 不匹配时的排查说明。"""
    s = repr(payload_msg).lower()
    if "mismatch" not in s or "speaker" not in s:
        return ""
    return (
        "\n  说明：连接上的 X-Api-Resource-Id（如 DOUBAO_E2E_RESOURCE_ID）必须与当前 "
        "tts.speaker 所属资源一致。默认 volc.speech.dialog 对应官方 Jupiter 等内置音色；"
        "若使用复刻音色 S_xxx，请在控制台查看该音色绑定的 Resource-Id 并写入环境变量。"
        "文档里「上传克隆音频」用的 Resource-Id（如 seed-icl-1.0 / seed-icl-2.0）是另一条 HTTP 接口，"
        "与实时对话 WebSocket 的 Resource-Id 不一定相同，需以控制台为准。"
    )


def _build_full_client_payload_gzip(
    *,
    event_number: int,
    session_id: Optional[str],
    request_meta: Optional[Dict[str, Any]],
) -> bytes:
    """
    与 eg/realtime_dialog_client.py 一致：FULL_CLIENT + JSON 元数据经 gzip 后再发。
    """
    b0 = (PROTOCOL_VERSION_V1 << 4) | HEADER_SIZE_4
    b1 = (MESSAGE_TYPE_FULL_CLIENT_REQUEST << 4) | MESSAGE_TYPE_SPECIFIC_FLAG_CARRY_EVENT_ID
    b2 = (_PROTO_SERIAL_JSON << 4) | _PROTO_COMP_GZIP
    b3 = RESERVED
    out = bytearray([b0, b1, b2, b3])
    out += struct.pack(">I", int(event_number))
    if session_id:
        sid_b = session_id.encode("utf-8")
        out += struct.pack(">I", len(sid_b))
        out += sid_b
    meta_b = json.dumps(request_meta or {}, ensure_ascii=False).encode("utf-8")
    gz = gzip.compress(meta_b)
    out += struct.pack(">I", len(gz))
    out += gz
    return bytes(out)


def _build_task_request_payload_gzip(*, session_id: str, audio_data: bytes) -> bytes:
    """
    与 eg/realtime_dialog_client.task_request 一致：AUDIO_ONLY + 无序列化 + gzip(pcm)。
    """
    b0 = (PROTOCOL_VERSION_V1 << 4) | HEADER_SIZE_4
    b1 = (MESSAGE_TYPE_AUDIO_ONLY_REQUEST << 4) | MESSAGE_TYPE_SPECIFIC_FLAG_CARRY_EVENT_ID
    b2 = (_PROTO_SERIAL_RAW << 4) | _PROTO_COMP_GZIP
    b3 = RESERVED
    out = bytearray([b0, b1, b2, b3])
    out += struct.pack(">I", EVENT_TASK_REQUEST)
    sid_b = session_id.encode("utf-8")
    out += struct.pack(">I", len(sid_b))
    out += sid_b
    gz = gzip.compress(audio_data)
    out += struct.pack(">I", len(gz))
    out += gz
    return bytes(out)


def _build_chat_text_query_payload_gzip(*, session_id: str, content: str) -> bytes:
    """
    与 eg/realtime_dialog_client.chat_text_query 一致：FULL_CLIENT + JSON + GZIP，event=501。
    """
    return _build_full_client_payload_gzip(
        event_number=EVENT_CHAT_TEXT_QUERY,
        session_id=session_id,
        request_meta={"content": content},
    )


def _parse_openspeech_ws_response(res: bytes) -> Dict[str, Any]:
    """
    与 eg/protocol.py parse_response 对齐：变长头、session 长度、gzip、JSON。
    """
    if isinstance(res, str) or not res or len(res) < 4:
        return {}
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F
    if len(res) < header_size * 4:
        return {}
    payload = res[header_size * 4 :]
    result: Dict[str, Any] = {}
    payload_msg: Any = None
    payload_size = 0
    start = 0

    if message_type == _SERVER_FULL_RESPONSE or message_type == _SERVER_ACK:
        result["message_type"] = "SERVER_ACK" if message_type == _SERVER_ACK else "SERVER_FULL_RESPONSE"
        # 与 eg/protocol.parse_response 一致：先 seq 再 event
        if message_type_specific_flags & _SERVER_MSG_FLAG_SEQ:
            start += 4
        if message_type_specific_flags & _SERVER_MSG_FLAG_EVENT:
            if len(payload) < start + 4:
                return result
            result["event"] = int.from_bytes(payload[start : start + 4], "big", signed=False)
            start += 4
        payload = payload[start:]
        if len(payload) < 4:
            return result
        # 官方用 signed=True；非法长度时回退 unsigned
        session_id_size = int.from_bytes(payload[:4], "big", signed=True)
        if session_id_size < 0 or session_id_size > len(payload) - 4:
            session_id_size = int.from_bytes(payload[:4], "big", signed=False)
        if len(payload) < 4 + session_id_size:
            return result
        result["session_id"] = payload[4 : 4 + session_id_size].decode("utf-8", errors="ignore")
        payload = payload[4 + session_id_size :]
        if len(payload) < 4:
            return result
        payload_size = int.from_bytes(payload[:4], "big", signed=False)
        payload_msg = payload[4:]
    elif message_type == _SERVER_ERROR_RESPONSE:
        result["message_type"] = "SERVER_ERROR"
        if len(payload) < 4:
            return result
        result["code"] = int.from_bytes(payload[:4], "big", signed=False)
        if len(payload) < 8:
            return result
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]

    if payload_msg is None:
        return result
    if message_compression == _PROTO_COMP_GZIP:
        try:
            payload_msg = gzip.decompress(payload_msg)
        except Exception:
            # 少数帧 compression 标记与内容不一致时，尝试按未压缩 JSON 解析
            if serialization_method == _PROTO_SERIAL_JSON:
                try:
                    payload_msg = json.loads(str(payload_msg, "utf-8"))
                except Exception:
                    return result
            else:
                return result
    if serialization_method == _PROTO_SERIAL_JSON:
        try:
            payload_msg = json.loads(str(payload_msg, "utf-8"))
        except Exception:
            payload_msg = None
    elif serialization_method != _PROTO_SERIAL_RAW:
        payload_msg = str(payload_msg, "utf-8")
    result["payload_msg"] = payload_msg
    result["payload_size"] = payload_size
    return result


def _e2e_apply_socket_timeout(ws: Any, seconds: float) -> None:
    """websocket-client 在 SSL 上有时需同时设 WebSocket 与底层 sock 超时，否则 recv 可能长时间无返回。"""
    try:
        ws.settimeout(seconds)
    except Exception:
        pass
    try:
        sk = getattr(ws, "sock", None)
        if sk is not None:
            sk.settimeout(seconds)
    except Exception:
        pass


def _post_openspeech_sts_token(
    *,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """
    请求 OpenSpeech STS 获取 jwt_token。
    部分 Windows/网络环境下会出现 SSLEOFError：
    - 环境变量 HTTP(S)_PROXY 被 requests 默认使用（trust_env=True）时，错误代理会导致握手直接 EOF；
    - 或 IPv6 路由异常、握手被重置等。
    这里使用 certifi、显式 User-Agent；SSL/连接失败时先忽略环境代理再试，再用 urllib3 仅 IPv4。
    """
    try:
        import requests
        from requests import exceptions as req_exc
    except ImportError as e:
        raise SystemExit("未安装 requests。请执行： pip install requests") from e

    merged: Dict[str, str] = {
        "User-Agent": "gemini_v3 (OpenSpeech STS; Python requests)",
        **headers,
    }
    verify: Any = True
    try:
        import certifi

        verify = certifi.where()
    except ImportError:
        pass

    def _post_session(*, trust_env: bool) -> Any:
        with requests.Session() as s:
            s.trust_env = trust_env
            r = s.post(url, headers=merged, json=body, timeout=timeout, verify=verify)
            r.raise_for_status()
            return r.json()

    err_with_proxy: Optional[BaseException] = None
    try:
        return _post_session(trust_env=True)
    except (req_exc.SSLError, req_exc.ConnectionError, OSError) as e_first:
        # HTTPError 继承 OSError，不能当作传输层失败去重试
        if isinstance(e_first, req_exc.HTTPError):
            raise
        err_with_proxy = e_first
        try:
            return _post_session(trust_env=False)
        except (req_exc.SSLError, req_exc.ConnectionError, OSError) as e_np:
            if isinstance(e_np, req_exc.HTTPError):
                raise
            pass
        except Exception:
            raise

    try:
        import urllib3.util.connection as u3c
    except ImportError:
        raise RuntimeError(
            "[gemini_v3][e2e] STS HTTPS 失败。可尝试：pip install -U certifi urllib3 requests；"
            "或设置 DOUBAO_E2E_API_ACCESS_KEY 为完整 Jwt; token 以跳过 STS。"
        ) from err_with_proxy
    if not hasattr(u3c, "allowed_gai_family"):
        raise RuntimeError(
            "[gemini_v3][e2e] STS HTTPS 失败且当前 urllib3 不支持 IPv4 回退。"
            "请升级 urllib3>=2，或设置 DOUBAO_E2E_API_ACCESS_KEY 跳过 STS。\n"
            f"原因: {err_with_proxy!r}"
        ) from err_with_proxy
    _orig = u3c.allowed_gai_family

    def _only_ipv4() -> int:
        return socket.AF_INET

    u3c.allowed_gai_family = _only_ipv4  # type: ignore[assignment]
    try:
        return _post_session(trust_env=False)
    except req_exc.HTTPError:
        raise
    except Exception as e_second:
        raise RuntimeError(
            "[gemini_v3][e2e] STS 请求失败（已尝试忽略环境代理与仅 IPv4）。"
            "请检查：代理/VPN/防火墙/公司网络；若必须用系统代理访问外网，请修正 HTTP(S)_PROXY；"
            "或设置 DOUBAO_E2E_API_ACCESS_KEY 跳过 STS。\n"
            f"首次错误(含系统代理): {err_with_proxy!r}\nIPv4+无环境代理: {e_second!r}"
        ) from e_second
    finally:
        u3c.allowed_gai_family = _orig  # type: ignore[assignment]


class E2EAsrClient:
    """
    与 eg/realtime_dialog_client 一致：wss://.../api/v3/realtime/dialogue + X-Api-* Header 鉴权，
    gzip 二进制帧；不再使用 query 的 SAUC bigmodel 地址。
    """

    def __init__(
        self,
        *,
        app_key: str,
        access_key: str,
        resource_id: str = DEFAULT_DIALOGUE_RESOURCE_ID,
        dialogue_app_id: str = "",
        # 你也可以在这里直接填已经带 Jwt 前缀的 api_access_key：
        # 示例： "Jwt; <jwt_token>"
        api_access_key: str = "",
        api_app_key: str = "",
        api_resource_id: str = "",
        jwt_sts_appid: Optional[str] = None,
        jwt_sts_access_key: Optional[str] = None,
        jwt_sts_duration_sec: int = 300,
        sts_url: str = DEFAULT_E2E_STS_URL,
        timeout_sec: float = 60.0,
    ) -> None:
        self.app_key = (app_key or "").strip()
        self.access_key = (access_key or "").strip()
        self.resource_id = (resource_id or "").strip()
        self.dialogue_app_id = (dialogue_app_id or "").strip()

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

    def _resolve_api_resource_id(self) -> str:
        return self.api_resource_id or self.resource_id

    def _maybe_get_jwt_token_via_sts(self) -> Optional[str]:
        # 若用户提供 sts 需要的字段，则走 STS 获取 jwt_token，再组装 api_access_key。
        if not self.jwt_sts_appid or not self.jwt_sts_access_key:
            return None

        print("[gemini_v3] e2e：正在请求 OpenSpeech STS（获取临时 Jwt）…", flush=True)

        headers = {
            "Authorization": f"Bearer; {self.jwt_sts_access_key}",
            "Content-Type": "application/json",
        }
        body = {"appid": self.jwt_sts_appid, "duration": self.jwt_sts_duration_sec}
        obj = _post_openspeech_sts_token(
            url=self.sts_url,
            headers=headers,
            body=body,
            timeout=min(20.0, self.timeout_sec),
        )
        # 返回字段在不同文档可能略有差异
        return obj.get("jwt_token") or obj.get("token") or obj.get("access_token")

    def _dialogue_ws_url(self) -> str:
        return f"wss://{HOST}{DIALOGUE_WS_PATH}"

    def _effective_access_key(self) -> str:
        jwt = self._maybe_get_jwt_token_via_sts()
        api_access_key = self.api_access_key
        if not api_access_key and jwt:
            api_access_key = f"Jwt; {jwt}"
        if not api_access_key:
            api_access_key = self._resolve_jwt_api_access_key()
        if not api_access_key.strip():
            raise RuntimeError(
                "[gemini_v3][e2e] 缺少鉴权：需要 access_key / STS，或 DOUBAO_E2E_API_ACCESS_KEY（Jwt;…）。"
            )
        return api_access_key.strip()

    def _dialogue_x_api_app_key(self) -> str:
        """实时对话 WebSocket 要求固定产品 App-Key（见 eg/config）；勿把控制台数字 appid 填在这里。"""
        v = (os.getenv("DOUBAO_E2E_DIALOG_X_API_APP_KEY", "") or "").strip()
        return v or DEFAULT_DIALOGUE_X_API_APP_KEY

    def _dialogue_x_api_app_id(self) -> str:
        """控制台「应用 ID」走 X-Api-App-ID；优先 DOUBAO_E2E_APP_ID / --doubao-e2e-app-id。"""
        if self.dialogue_app_id:
            return self.dialogue_app_id
        ak = self.app_key.strip()
        if ak.isdigit():
            return ak
        return ""

    def _dialogue_ws_headers(self, access_key: str) -> List[str]:
        """与 eg/config.py ws_connect_config['headers'] 一致（websocket-client 用 header 列表）。"""
        api_resource_id = self._resolve_api_resource_id()
        if not api_resource_id:
            raise RuntimeError(
                "[gemini_v3][e2e] 缺少鉴权：需要 resource_id（X-Api-Resource-Id，如 volc.speech.dialog）。"
            )
        x_app_key = self._dialogue_x_api_app_key()
        x_app_id = self._dialogue_x_api_app_id()
        return [
            f"X-Api-App-ID: {x_app_id}",
            f"X-Api-Access-Key: {access_key}",
            f"X-Api-Resource-Id: {api_resource_id}",
            f"X-Api-App-Key: {x_app_key}",
            f"X-Api-Connect-Id: {str(uuid.uuid4())}",
        ]

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

        if not (self.jwt_sts_appid and self.jwt_sts_access_key):
            print("[gemini_v3] e2e：使用环境/参数中的 access key（未走 STS）…", flush=True)
        try:
            access_key = self._effective_access_key()
        except RuntimeError as e:
            raise RuntimeError(f"[gemini_v3][e2e] 鉴权失败: {e}") from e

        ws_url = self._dialogue_ws_url()
        ws_headers = self._dialogue_ws_headers(access_key)
        print(
            f"[gemini_v3] e2e：正在连接 WebSocket（Dialogue + Header，最长约 {int(self.timeout_sec)}s）…",
            flush=True,
        )
        sid = session_id or str(uuid.uuid4())

        # 与 eg/config.py start_session_req 结构一致（含 tts.audio_config、dialog.location 等）
        # recv_timeout 与 eg 保持 10；过大会让服务端在静音后等待很久才处理
        recv_to = 10
        spk = (tts_speaker or "zh_male_yunzhou_jupiter_bigtts").strip()
        if (dialog_system_role or "").strip():
            sr = dialog_system_role.strip()
            ss = "你的说话风格简洁明了，语速适中，语调自然。"
        else:
            sr = "你使用活泼灵动的女声，性格开朗，热爱生活。"
            ss = "你的说话风格简洁明了，语速适中，语调自然。"
        config: Dict[str, Any] = {
            "asr": {
                "extra": {"end_smooth_window_ms": 1500},
            },
            "tts": {
                "speaker": spk,
                "audio_config": {
                    "channel": 1,
                    "format": "pcm",
                    "sample_rate": 24000,
                },
            },
            "dialog": {
                "bot_name": "豆包",
                "system_role": sr,
                "speaking_style": ss,
                "location": {"city": "北京"},
                "extra": {
                    "strict_audit": False,
                    "audit_response": "支持客户自定义安全审核回复话术。",
                    "recv_timeout": recv_to,
                    "input_mod": "audio",
                },
            },
        }

        payload_start_conn = _build_full_client_payload_gzip(
            event_number=EVENT_START_CONNECTION,
            session_id=None,
            request_meta={},
        )
        payload_start_session = _build_full_client_payload_gzip(
            event_number=EVENT_START_SESSION,
            session_id=sid,
            request_meta=config,
        )

        # 与 eg 麦克风模式一致：SayHello 先触发开场 TTS，等 359 后再发音频
        payload_say_hello = _build_full_client_payload_gzip(
            event_number=EVENT_SAY_HELLO,
            session_id=sid,
            request_meta={"content": "你好，我是豆包，有什么可以帮助你的？"},
        )
        payload_finish_session = _build_full_client_payload_gzip(
            event_number=EVENT_FINISH_SESSION,
            session_id=sid,
            request_meta={},
        )

        # 用于捕获最终文本
        latest_interim = ""
        latest_final = ""

        ws = None  # type: ignore
        finish_session_sent = False
        try:
            ws = websocket.create_connection(
                ws_url,
                header=ws_headers,
                timeout=self.timeout_sec,
                enable_multithread=True,
            )
        except Exception as e:
            raise RuntimeError(
                "[gemini_v3][e2e] WebSocket 连接失败（Dialogue + Header；请检查 AppID/AppKey/ResourceId/Jwt 与网络）。"
                f"\n  详情: {e}"
            ) from e

        # 等 SESSION_STARTED 阶段用较短 recv 超时，避免单次 recv 阻塞整段 timeout_sec（像「卡死」）
        wait_sess_sec = min(4.0, max(1.0, self.timeout_sec * 0.25))
        ws.settimeout(wait_sess_sec)
        try:
            # 官方示例：StartConnection → recv；再 StartSession → recv（不可两帧连发后再收）
            ws.settimeout(wait_sess_sec)
            ws.send_binary(payload_start_conn)
            raw_conn = ws.recv()
            p_conn = _parse_openspeech_ws_response(
                raw_conn if isinstance(raw_conn, (bytes, bytearray)) else b""
            )
            if p_conn.get("message_type") == "SERVER_ERROR":
                raise RuntimeError(f"[gemini_v3][e2e] StartConnection 错误: {p_conn.get('payload_msg')!r}")
            if p_conn.get("event") == EVENT_CONNECTION_FAILED:
                raise RuntimeError(
                    "[gemini_v3][e2e] 服务端拒绝连接（EVENT_CONNECTION_FAILED），请核对 resource_id / app_key / token。"
                )
            _pm_conn = p_conn.get("payload_msg")
            if isinstance(_pm_conn, dict) and _pm_conn.get("error"):
                raise RuntimeError(f"[gemini_v3][e2e] StartConnection 失败: {_pm_conn!r}")

            ws.send_binary(payload_start_session)
            raw_sess = ws.recv()
            p_sess = _parse_openspeech_ws_response(
                raw_sess if isinstance(raw_sess, (bytes, bytearray)) else b""
            )
            if p_sess.get("message_type") == "SERVER_ERROR":
                raise RuntimeError(
                    f"[gemini_v3][e2e] StartSession 错误: {p_sess.get('payload_msg')!r}"
                    + _hint_e2e_resource_speaker_mismatch(p_sess.get("payload_msg"))
                )
            if p_sess.get("event") == EVENT_SESSION_FAILED:
                pm = p_sess.get("payload_msg")
                err = ""
                if isinstance(pm, dict):
                    err = str(pm.get("error") or pm.get("message") or pm)
                raise RuntimeError(f"[gemini_v3][e2e] 会话启动失败（SESSION_FAILED）：{err or pm!r}")
            _pm_sess = p_sess.get("payload_msg")
            if isinstance(_pm_sess, dict) and _pm_sess.get("error"):
                raise RuntimeError(
                    f"[gemini_v3][e2e] StartSession 失败: {_pm_sess!r}"
                    + _hint_e2e_resource_speaker_mismatch(_pm_sess)
                )

            # ── 与 eg 麦克风模式一致：SayHello → 等 359 → 发音频 ──
            # audio_file 模式下服务端不触发 VAD；必须用 "audio" + SayHello 开场才行。
            _e2e_apply_socket_timeout(ws, min(30.0, self.timeout_sec))

            print("[gemini_v3] e2e：发送 SayHello(300)，等待开场结束(359)…", flush=True)
            ws.send_binary(payload_say_hello)
            hello_deadline = time.perf_counter() + min(30.0, self.timeout_sec)
            hello_done = False
            while time.perf_counter() < hello_deadline:
                _e2e_apply_socket_timeout(ws, min(3.0, hello_deadline - time.perf_counter()))
                try:
                    raw = ws.recv()
                except Exception as e:
                    if _e2e_ws_exc_is_timeout(e):
                        continue
                    break
                p = _parse_openspeech_ws_response(
                    raw if isinstance(raw, (bytes, bytearray)) else b""
                )
                ev = p.get("event")
                if ev is not None:
                    print(
                        f"[gemini_v3] e2e：开场下行 event={ev} mt={p.get('message_type')!r}",
                        flush=True,
                    )
                if ev == EVENT_TTS_PLAY_DONE:
                    hello_done = True
                    print("[gemini_v3] e2e：已收到 359（开场结束），开始发送音频…", flush=True)
                    break
            if not hello_done:
                print(
                    "[gemini_v3] e2e：未在限时内收到 359，仍将继续发送音频…",
                    flush=True,
                )

            # ── 音频预处理：归一化 + 尾部静音 ──
            import numpy as np
            _pcm = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).copy()
            _pk = int(np.max(np.abs(_pcm))) if len(_pcm) > 0 else 0
            _E2E_NORM_TARGET = 20000
            _E2E_NORM_THRESHOLD = 10000
            if 0 < _pk < _E2E_NORM_THRESHOLD:
                _gain = _E2E_NORM_TARGET / float(_pk)
                _pcm = np.clip(_pcm.astype(np.float32) * _gain, -32768, 32767).astype(np.int16)
                _new_pk = int(np.max(np.abs(_pcm)))
                print(
                    f"[gemini_v3] e2e：音频归一化 peak {_pk}→{_new_pk}（gain={_gain:.2f}x）",
                    flush=True,
                )
            _E2E_TRAILING_SILENCE_SEC = 2.0
            _silence = np.zeros(int(E2E_DIALOGUE_INPUT_SAMPLE_RATE * _E2E_TRAILING_SILENCE_SEC), dtype=np.int16)
            _pcm = np.concatenate([_pcm, _silence])
            pcm_s16le_bytes = _pcm.tobytes()
            print(
                f"[gemini_v3] e2e：已追加 {_E2E_TRAILING_SILENCE_SEC:.1f}s 尾部静音（总时长 {len(_pcm)/E2E_DIALOGUE_INPUT_SAMPLE_RATE:.1f}s）",
                flush=True,
            )

            pcm_len = len(pcm_s16le_bytes)
            if pcm_len == 0:
                raise RuntimeError("[gemini_v3][e2e] 无 PCM 数据")
            chunk_bytes = E2E_DIALOGUE_PCM_CHUNK_FRAMES * 2
            n_chunks = (pcm_len + chunk_bytes - 1) // chunk_bytes
            _rtc = (os.getenv("GEMINI_V3_E2E_REALTIME_CHUNK", "1") or "").strip().lower()
            rt_chunk = _rtc not in ("0", "false", "no")

            dbg = (os.getenv("GEMINI_V3_E2E_DEBUG", "") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            )

            def _merge_asr_obj(obj: Any) -> None:
                nonlocal latest_final, latest_interim
                if not isinstance(obj, dict):
                    return
                results = obj.get("results")
                if isinstance(results, list):
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
                    return
                for k in ("text", "asr_text", "result", "content"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        latest_final = v.strip()
                        return

            asr_ended = False
            _recv_lock = threading.Lock()

            def _handle_e2e_server_parsed(parsed: Dict[str, Any]) -> None:
                nonlocal latest_final, latest_interim, asr_ended
                if dbg and parsed:
                    pm_preview = parsed.get("payload_msg")
                    if isinstance(pm_preview, bytes):
                        pm_preview = f"<{len(pm_preview)} bytes>"
                    print(
                        f"[gemini_v3][e2e][debug] event={parsed.get('event')!r} "
                        f"mt={parsed.get('message_type')!r} payload={pm_preview!r}",
                        flush=True,
                    )
                if parsed.get("message_type") == "SERVER_ERROR":
                    print(
                        f"[gemini_v3][e2e] 服务端错误: code={parsed.get('code')!r} msg={parsed.get('payload_msg')!r}",
                        flush=True,
                    )
                    asr_ended = True
                    return
                event_number = parsed.get("event")
                obj = parsed.get("payload_msg")
                if event_number is None and isinstance(obj, dict):
                    if obj.get("results") is not None or any(
                        obj.get(k) for k in ("text", "asr_text", "result")
                    ):
                        event_number = EVENT_ASR_RESPONSE

                if event_number is None:
                    return

                if event_number == EVENT_CONNECTION_FAILED:
                    print(f"[gemini_v3][e2e] 连接失败（event={event_number}）", flush=True)
                    asr_ended = True
                    return

                if event_number == EVENT_SESSION_FINISHED:
                    if isinstance(obj, dict):
                        _merge_asr_obj(obj)
                    asr_ended = True
                    return

                if event_number == EVENT_SESSION_FAILED:
                    err = ""
                    if isinstance(obj, dict):
                        err = str(obj.get("error") or obj.get("message") or obj)
                    print(
                        f"[gemini_v3][e2e] 会话失败（SESSION_FAILED）：{err or obj}",
                        flush=True,
                    )
                    asr_ended = True
                    return

                if event_number in (EVENT_ASR_RESPONSE, EVENT_CHAT_RESPONSE) and isinstance(
                    obj, dict
                ):
                    if event_number == EVENT_CHAT_RESPONSE and isinstance(obj.get("content"), str):
                        c = str(obj.get("content") or "")
                        if c:
                            with _recv_lock:
                                latest_final = (latest_final or "") + c
                    else:
                        with _recv_lock:
                            _merge_asr_obj(obj)

                if event_number == EVENT_ASR_ENDED:
                    asr_ended = True

                if event_number == EVENT_TTS_PLAY_DONE:
                    if isinstance(obj, dict) and obj.get("no_content"):
                        asr_ended = True

            def _process_raw_e2e_frame(raw: bytes) -> None:
                if not isinstance(raw, (bytes, bytearray)) or not raw:
                    return
                parsed = _parse_openspeech_ws_response(raw)
                _handle_e2e_server_parsed(parsed)

            # ── recv 线程：与 eg/audio_manager.receive_loop 对应 ──
            t0 = time.perf_counter()
            recv_done_ev = threading.Event()
            recv_thread_err: List[Optional[BaseException]] = [None]

            def _recv_worker() -> None:
                try:
                    while not asr_ended and time.perf_counter() - t0 < self.timeout_sec:
                        rem = self.timeout_sec - (time.perf_counter() - t0)
                        if rem <= 0:
                            break
                        _e2e_apply_socket_timeout(ws, min(2.0, max(0.3, rem)))
                        try:
                            raw = ws.recv()
                        except Exception as e:
                            if _e2e_ws_exc_is_timeout(e):
                                continue
                            if asr_ended or (latest_final or latest_interim):
                                break
                            recv_thread_err[0] = e
                            break
                        _process_raw_e2e_frame(
                            raw if isinstance(raw, (bytes, bytearray)) else b""
                        )
                except Exception as e:
                    recv_thread_err[0] = e
                finally:
                    recv_done_ev.set()

            recv_t = threading.Thread(target=_recv_worker, name="e2e_recv", daemon=True)
            recv_t.start()

            # ── 发送音频（主线程） ──
            print(
                f"[gemini_v3] e2e：正在上传音频（{n_chunks} 块，每块最多 "
                f"{E2E_DIALOGUE_PCM_CHUNK_FRAMES} 帧 @ 16kHz），recv 线程已并发启动…",
                flush=True,
            )
            for off in range(0, pcm_len, chunk_bytes):
                if asr_ended:
                    break
                chunk = pcm_s16le_bytes[off : off + chunk_bytes]
                if not chunk:
                    break
                payload_task = _build_task_request_payload_gzip(session_id=sid, audio_data=chunk)
                ws.send_binary(payload_task)
                if rt_chunk and len(chunk) >= 2:
                    time.sleep((len(chunk) // 2) / float(E2E_DIALOGUE_INPUT_SAMPLE_RATE))

            # 与 eg/audio_manager 的 audio_file 模式一致：发完最后一块音频后 **不** 立即发
            # FinishSession(102)。服务端会在检测到输入结束（VAD / recv_timeout）后自行
            # 处理 ASR → Dialog → TTS，下发 350/550/352/351/359。客户端只在收到 359 后
            # 才发 FinishSession（或 finally 兜底补发）。若在此处就发 102，服务端把它当
            # 作"立即关闭"，直接返回空 152。
            print(
                "[gemini_v3] e2e：音频已全部发送，等待服务端处理（与 eg 一致：不先发 FinishSession）…",
                flush=True,
            )

            # ── 等待 recv 线程结束（它会在收到 152/359 等时自行退出） ──
            remaining = max(1.0, self.timeout_sec - (time.perf_counter() - t0))
            print(f"[gemini_v3] e2e：等待识别结果（recv 线程，最多 {remaining:.0f}s）…", flush=True)

            def _heartbeat_loop() -> None:
                while not recv_done_ev.wait(3.0):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[gemini_v3] e2e：识别中… {elapsed:.0f}s / {self.timeout_sec:.0f}s"
                        + (f" 已拼接: {(latest_final or '')[:40]!r}" if latest_final else ""),
                        flush=True,
                    )

            hb_thread = threading.Thread(target=_heartbeat_loop, name="e2e_hb", daemon=True)
            hb_thread.start()

            recv_t.join(timeout=remaining)
            recv_done_ev.set()

            # 与 eg 一致：recv 结束（收到 359/152）后再发 FinishSession + FinishConnection
            if not finish_session_sent:
                try:
                    ws.send_binary(payload_finish_session)
                    finish_session_sent = True
                    _e2e_apply_socket_timeout(ws, 5.0)
                    try:
                        raw_fs = ws.recv()
                        if dbg:
                            p = _parse_openspeech_ws_response(
                                raw_fs if isinstance(raw_fs, (bytes, bytearray)) else b""
                            )
                            print(
                                f"[gemini_v3][e2e][debug] FinishSession 回包 event={p.get('event')!r}",
                                flush=True,
                            )
                    except Exception:
                        pass
                except Exception:
                    pass

            if recv_thread_err[0] is not None and not (latest_final or latest_interim):
                print(
                    f"[gemini_v3][e2e] recv 线程异常: {recv_thread_err[0]!r}",
                    flush=True,
                )

            result_text = (latest_final or latest_interim or "").strip()
            if not result_text:
                print(
                    "[gemini_v3] e2e：未获取到识别文本。请检查：\n"
                    "  1) 麦克风是否有声音（录音时长/音量）\n"
                    "  2) 设 GEMINI_V3_E2E_DEBUG=1 查看所有 event\n"
                    "  3) 文本 probe 是否正常（--e2e-probe-text 你好）",
                    flush=True,
                )
            return result_text
        finally:
            if ws is not None:
                if not finish_session_sent:
                    try:
                        ws.send_binary(payload_finish_session)
                    except Exception:
                        pass
                try:
                    ws.close()
                except Exception:
                    pass

    def probe_text_dialogue(
        self,
        *,
        query: str,
        tts_speaker: str = "",
        dialog_system_role: str = "",
        say_hello_first: bool = True,
    ) -> str:
        """
        不经过音频/ASR：与 eg 相同鉴权与 WebSocket，StartSession 使用 input_mod=text，
        再发 ChatTextQuery(501)。用于验证「豆包实时对话」链路是否下行内容，与 ASR 解耦。
        """
        try:
            import websocket  # type: ignore
        except ImportError as e:
            raise SystemExit("需要 websocket-client： pip install websocket-client") from e

        q = (query or "").strip()
        if not q:
            raise RuntimeError("[gemini_v3][e2e] probe_text_dialogue：query 为空")

        access_key = self._effective_access_key()
        ws_url = self._dialogue_ws_url()
        ws_headers = self._dialogue_ws_headers(access_key)
        sid = str(uuid.uuid4())
        recv_to = int(min(120, max(10, int(self.timeout_sec))))
        spk = (tts_speaker or "zh_male_yunzhou_jupiter_bigtts").strip()
        if (dialog_system_role or "").strip():
            sr = dialog_system_role.strip()
            ss = "你的说话风格简洁明了，语速适中，语调自然。"
        else:
            sr = "你使用活泼灵动的女声，性格开朗，热爱生活。"
            ss = "你的说话风格简洁明了，语速适中，语调自然。"
        config: Dict[str, Any] = {
            "asr": {"extra": {"end_smooth_window_ms": 1500}},
            "tts": {
                "speaker": spk,
                "audio_config": {"channel": 1, "format": "pcm", "sample_rate": 24000},
            },
            "dialog": {
                "bot_name": "豆包",
                "system_role": sr,
                "speaking_style": ss,
                "location": {"city": "北京"},
                "extra": {
                    "strict_audit": False,
                    "audit_response": "支持客户自定义安全审核回复话术。",
                    "recv_timeout": recv_to,
                    "input_mod": "text",
                },
            },
        }

        payload_start_conn = _build_full_client_payload_gzip(
            event_number=EVENT_START_CONNECTION,
            session_id=None,
            request_meta={},
        )
        payload_start_session = _build_full_client_payload_gzip(
            event_number=EVENT_START_SESSION,
            session_id=sid,
            request_meta=config,
        )
        payload_say_hello = _build_full_client_payload_gzip(
            event_number=EVENT_SAY_HELLO,
            session_id=sid,
            request_meta={"content": "你好，我是豆包，有什么可以帮助你的？"},
        )
        payload_chat = _build_full_client_payload_gzip(
            event_number=EVENT_CHAT_TEXT_QUERY,
            session_id=sid,
            request_meta={"content": q},
        )
        payload_finish_session = _build_full_client_payload_gzip(
            event_number=EVENT_FINISH_SESSION,
            session_id=sid,
            request_meta={},
        )
        payload_finish_conn = _build_full_client_payload_gzip(
            event_number=EVENT_FINISH_CONNECTION,
            session_id=None,
            request_meta={},
        )

        dbg = (os.getenv("GEMINI_V3_E2E_DEBUG", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        collected: List[str] = []
        ws = None  # type: ignore
        try:
            print(
                "[gemini_v3] e2e probe：文本直连（input_mod=text + event 501），不走 ASR…",
                flush=True,
            )
            print("[gemini_v3] e2e probe：正在建立 WebSocket（若长时间无下一条，多为网络/TLS 卡住）…", flush=True)
            ws = websocket.create_connection(
                ws_url,
                header=ws_headers,
                timeout=self.timeout_sec,
            )
            print("[gemini_v3] e2e probe：WebSocket 已建立", flush=True)
            wait_sess_sec = min(4.0, max(1.0, self.timeout_sec * 0.25))
            ws.settimeout(wait_sess_sec)
            print("[gemini_v3] e2e probe：发送 StartConnection…", flush=True)
            ws.send_binary(payload_start_conn)
            raw_conn = ws.recv()
            p_conn = _parse_openspeech_ws_response(
                raw_conn if isinstance(raw_conn, (bytes, bytearray)) else b""
            )
            print(
                f"[gemini_v3] e2e probe：StartConnection 回包 event={p_conn.get('event')!r} "
                f"mt={p_conn.get('message_type')!r}",
                flush=True,
            )
            if p_conn.get("message_type") == "SERVER_ERROR":
                raise RuntimeError(
                    f"[gemini_v3][e2e] probe StartConnection: {p_conn.get('payload_msg')!r}"
                )
            if p_conn.get("event") == EVENT_CONNECTION_FAILED:
                raise RuntimeError("[gemini_v3][e2e] probe：EVENT_CONNECTION_FAILED")

            print("[gemini_v3] e2e probe：发送 StartSession（input_mod=text）…", flush=True)
            ws.send_binary(payload_start_session)
            raw_sess = ws.recv()
            p_sess = _parse_openspeech_ws_response(
                raw_sess if isinstance(raw_sess, (bytes, bytearray)) else b""
            )
            print(
                f"[gemini_v3] e2e probe：StartSession 回包 event={p_sess.get('event')!r} "
                f"mt={p_sess.get('message_type')!r}",
                flush=True,
            )
            if p_sess.get("message_type") == "SERVER_ERROR":
                raise RuntimeError(
                    f"[gemini_v3][e2e] probe StartSession: {p_sess.get('payload_msg')!r}"
                    + _hint_e2e_resource_speaker_mismatch(p_sess.get("payload_msg"))
                )
            if p_sess.get("event") == EVENT_SESSION_FAILED:
                pm = p_sess.get("payload_msg")
                err = ""
                if isinstance(pm, dict):
                    err = str(pm.get("error") or pm.get("message") or pm)
                raise RuntimeError(f"[gemini_v3][e2e] probe StartSession 失败：{err or pm!r}")

            # 与 eg 文本模式一致：先发 SayHello，再等到开场 TTS 结束（359），否则 501 可能被忽略
            if say_hello_first:
                print(
                    "[gemini_v3] e2e probe：发送 SayHello(300)，并等待开场结束（event≈359）…",
                    flush=True,
                )
                ws.send_binary(payload_say_hello)
                t_h0 = time.perf_counter()
                hello_cap = min(45.0, max(15.0, self.timeout_sec * 0.75))
                saw_359 = False
                while time.perf_counter() - t_h0 < hello_cap:
                    rem_h = hello_cap - (time.perf_counter() - t_h0)
                    _e2e_apply_socket_timeout(ws, min(8.0, max(0.5, rem_h)))
                    try:
                        raw_h = ws.recv()
                    except Exception as e:
                        if _e2e_ws_exc_is_timeout(e):
                            print(
                                "[gemini_v3] e2e probe：开场阶段 recv 超时，继续等下一帧…",
                                flush=True,
                            )
                            continue
                        raise
                    if not isinstance(raw_h, (bytes, bytearray)) or not raw_h:
                        continue
                    ph = _parse_openspeech_ws_response(raw_h)
                    evh = ph.get("event")
                    mth = ph.get("message_type")
                    print(
                        f"[gemini_v3] e2e probe：开场下行 event={evh!r} mt={mth!r} len={len(raw_h)}",
                        flush=True,
                    )
                    if dbg and ph:
                        print(f"[gemini_v3][e2e][debug] hello payload_msg type={type(ph.get('payload_msg'))!r}", flush=True)
                    obj_h = ph.get("payload_msg")
                    th = _extract_e2e_reply_text(obj_h)
                    if th:
                        collected.append(th)
                    if evh == EVENT_TTS_PLAY_DONE:
                        saw_359 = True
                        print("[gemini_v3] e2e probe：已收到 359（开场 TTS 结束）", flush=True)
                        break
                    if evh == EVENT_SESSION_FINISHED:
                        print("[gemini_v3] e2e probe：开场阶段收到 152，会话结束", flush=True)
                        break
                if not saw_359:
                    print(
                        "[gemini_v3] e2e probe：未在限时内收到 359，仍将发送 501（若仍无文本请试 GEMINI_V3_E2E_DEBUG=1）",
                        flush=True,
                    )
            else:
                print("[gemini_v3] e2e probe：已跳过 SayHello（--e2e-probe-no-hello）", flush=True)

            print("[gemini_v3] e2e probe：发送 ChatTextQuery(501)…", flush=True)
            ws.send_binary(payload_chat)
            t0 = time.perf_counter()
            session_done = False
            recv_done_ev = threading.Event()

            def _probe_hb() -> None:
                while not recv_done_ev.wait(3.0):
                    print(
                        f"[gemini_v3] e2e probe：等待 501 回复… {time.perf_counter() - t0:.0f}s / {self.timeout_sec:.0f}s",
                        flush=True,
                    )

            hb_thread = threading.Thread(target=_probe_hb, name="e2e_probe_hb", daemon=True)
            hb_thread.start()
            try:
                while time.perf_counter() - t0 < self.timeout_sec and not session_done:
                    rem = self.timeout_sec - (time.perf_counter() - t0)
                    if rem <= 0:
                        break
                    _e2e_apply_socket_timeout(ws, min(8.0, max(0.5, rem)))
                    try:
                        raw = ws.recv()
                    except Exception as e:
                        if _e2e_ws_exc_is_timeout(e) and rem > 1.0:
                            continue
                        break
                    if not isinstance(raw, (bytes, bytearray)) or not raw:
                        continue
                    parsed = _parse_openspeech_ws_response(raw)
                    ev = parsed.get("event")
                    obj = parsed.get("payload_msg")
                    print(
                        f"[gemini_v3] e2e probe：收包 event={ev!r} mt={parsed.get('message_type')!r} "
                        f"bytes={len(raw)}",
                        flush=True,
                    )
                    if dbg and parsed:
                        print(
                            f"[gemini_v3][e2e][debug] payload_msg preview={repr(obj)[:500]}",
                            flush=True,
                        )
                    if parsed.get("message_type") == "SERVER_ERROR":
                        raise RuntimeError(
                            f"[gemini_v3][e2e] probe 服务端错误: {parsed.get('payload_msg')!r}"
                        )
                    if ev in (EVENT_ASR_RESPONSE, EVENT_CHAT_RESPONSE) or (
                        ev is None and isinstance(obj, dict)
                    ):
                        t = _extract_e2e_reply_text(obj)
                        if t:
                            collected.append(t)
                    if ev == EVENT_SESSION_FINISHED:
                        t2 = _extract_e2e_reply_text(obj)
                        if t2:
                            collected.append(t2)
                        session_done = True
                        break
                    if ev == EVENT_ASR_ENDED:
                        session_done = True
                        break
                    # 文本 501 回复流式 550 之后，常以 359（可带 no_content）表示本轮播报结束；若不 break 会空转等超时
                    if ev == EVENT_TTS_PLAY_DONE:
                        session_done = True
                        break
            finally:
                recv_done_ev.set()

            try:
                ws.send_binary(payload_finish_session)
            except Exception:
                pass
            _e2e_apply_socket_timeout(ws, min(5.0, self.timeout_sec))
            try:
                raw_fs = ws.recv()
                p_fs = _parse_openspeech_ws_response(
                    raw_fs if isinstance(raw_fs, (bytes, bytearray)) else b""
                )
                print(
                    f"[gemini_v3] e2e probe：FinishSession 后收包 event={p_fs.get('event')!r}",
                    flush=True,
                )
            except Exception:
                pass

            try:
                ws.send_binary(payload_finish_conn)
                _e2e_apply_socket_timeout(ws, 5.0)
                raw_fc = ws.recv()
                pf = _parse_openspeech_ws_response(
                    raw_fc if isinstance(raw_fc, (bytes, bytearray)) else b""
                )
                print(
                    f"[gemini_v3] e2e probe：FinishConnection 回包 event={pf.get('event')!r}",
                    flush=True,
                )
            except Exception:
                pass

            # 550 为逐字/逐段流式，勿用空格拼接
            return "".join(collected).strip()
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# E2EDialogueSession -- 豆包全端到端对话（ASR + LLM + TTS 全在一条 WebSocket）
# ---------------------------------------------------------------------------

E2E_TTS_OUTPUT_SAMPLE_RATE = 24000


class E2EDialogueSession:
    """持久 WebSocket 会话：豆包 ASR → LLM → TTS 全端到端，支持多轮语音对话。"""

    def __init__(
        self,
        *,
        client: E2EAsrClient,
        tts_speaker: str = "",
        timeout_sec: float = 60.0,
    ) -> None:
        self._client = client
        self._tts_speaker = (tts_speaker or "zh_male_yunzhou_jupiter_bigtts").strip()
        self._timeout = max(10.0, float(timeout_sec))
        self._ws: Any = None
        self._sid = ""
        self._opened = False

    # ------------------------------------------------------------------ open
    def open(
        self,
        system_role: str,
        battle_brief: str,
        *,
        speaking_style: str = "你的说话风格简洁明了，语速适中，语调自然。",
    ) -> None:
        """建立 WebSocket → StartConnection → StartSession → SayHello → 注入快报。"""
        import websocket  # type: ignore

        access_key = self._client._effective_access_key()
        ws_url = self._client._dialogue_ws_url()
        ws_headers = self._client._dialogue_ws_headers(access_key)
        self._sid = str(uuid.uuid4())
        recv_to = 10

        config: Dict[str, Any] = {
            "asr": {"extra": {"end_smooth_window_ms": 1500}},
            "tts": {
                "speaker": self._tts_speaker,
                "audio_config": {"channel": 1, "format": "pcm_s16le", "sample_rate": E2E_TTS_OUTPUT_SAMPLE_RATE},
            },
            "dialog": {
                "bot_name": "豆包",
                "system_role": system_role,
                "speaking_style": speaking_style,
                "location": {"city": "北京"},
                "extra": {
                    "strict_audit": False,
                    "audit_response": "支持客户自定义安全审核回复话术。",
                    "recv_timeout": recv_to,
                    "input_mod": "audio",
                },
            },
        }

        print("[gemini_v3] e2e-dialogue：正在连接 WebSocket…", flush=True)
        self._ws = websocket.create_connection(
            ws_url, header=ws_headers, timeout=self._timeout, enable_multithread=True,
        )
        ws = self._ws

        # -- StartConnection -----------------------------------------------
        ws.send_binary(_build_full_client_payload_gzip(
            event_number=EVENT_START_CONNECTION, session_id=None, request_meta={},
        ))
        _e2e_apply_socket_timeout(ws, 10.0)
        raw = ws.recv()
        p = _parse_openspeech_ws_response(raw if isinstance(raw, (bytes, bytearray)) else b"")
        if p.get("message_type") == "SERVER_ERROR":
            raise RuntimeError(f"[e2e-dialogue] StartConnection 错误: {p.get('payload_msg')!r}")

        # -- StartSession --------------------------------------------------
        ws.send_binary(_build_full_client_payload_gzip(
            event_number=EVENT_START_SESSION, session_id=self._sid, request_meta=config,
        ))
        raw = ws.recv()
        p = _parse_openspeech_ws_response(raw if isinstance(raw, (bytes, bytearray)) else b"")
        if p.get("message_type") == "SERVER_ERROR" or p.get("event") == EVENT_SESSION_FAILED:
            raise RuntimeError(
                f"[e2e-dialogue] StartSession 错误: {p.get('payload_msg')!r}"
                + _hint_e2e_resource_speaker_mismatch(p.get("payload_msg"))
            )

        # -- SayHello → 等 359 --------------------------------------------
        print("[gemini_v3] e2e-dialogue：SayHello，等待开场结束…", flush=True)
        ws.send_binary(_build_full_client_payload_gzip(
            event_number=EVENT_SAY_HELLO, session_id=self._sid,
            request_meta={"content": "你好，我是豆包，有什么可以帮助你的？"},
        ))
        self._recv_until_359(label="开场")

        # -- 注入战术快报 via ChatTextQuery(501) ----------------------------
        if battle_brief.strip():
            print("[gemini_v3] e2e-dialogue：正在注入对局情报（ChatTextQuery）…", flush=True)
            ws.send_binary(_build_chat_text_query_payload_gzip(
                session_id=self._sid, content=battle_brief.strip(),
            ))
            self._recv_until_359(label="快报注入", play_tts=True)

        self._opened = True
        print("[gemini_v3] e2e-dialogue：会话就绪，等待语音输入。", flush=True)

    # ----------------------------------------------------------- voice_turn
    def voice_turn(self, pcm_s16le_bytes: bytes) -> Tuple[str, str]:
        """发送一段录音，返回 (asr_text, llm_response_text)，同时播放 TTS 音频。"""
        import numpy as np

        if not self._opened or self._ws is None:
            raise RuntimeError("[e2e-dialogue] 会话尚未 open")
        ws = self._ws

        # 归一化 + 尾部静音（复用之前的逻辑）
        _pcm = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).copy()
        _pk = int(np.max(np.abs(_pcm))) if len(_pcm) > 0 else 0
        if 0 < _pk < 10000:
            _gain = 20000.0 / float(_pk)
            _pcm = np.clip(_pcm.astype(np.float32) * _gain, -32768, 32767).astype(np.int16)
            print(f"[gemini_v3] e2e：音频归一化 peak {_pk}→{int(np.max(np.abs(_pcm)))}（gain={_gain:.2f}x）", flush=True)
        _silence = np.zeros(int(E2E_DIALOGUE_INPUT_SAMPLE_RATE * 2.0), dtype=np.int16)
        _pcm = np.concatenate([_pcm, _silence])
        pcm_bytes = _pcm.tobytes()
        print(f"[gemini_v3] e2e：音频总时长 {len(_pcm)/E2E_DIALOGUE_INPUT_SAMPLE_RATE:.1f}s（含 2s 尾部静音）", flush=True)

        chunk_bytes = E2E_DIALOGUE_PCM_CHUNK_FRAMES * 2
        n_chunks = (len(pcm_bytes) + chunk_bytes - 1) // chunk_bytes

        asr_text = ""
        llm_text_parts: List[str] = []
        turn_done = False
        tts_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        recv_done = threading.Event()
        recv_err: List[Optional[BaseException]] = [None]
        t0 = time.perf_counter()

        dbg = (os.getenv("GEMINI_V3_E2E_DEBUG", "") or "").strip().lower() in ("1", "true", "yes")

        def _on_parsed(parsed: Dict[str, Any]) -> None:
            nonlocal asr_text, turn_done
            if dbg and parsed:
                pm = parsed.get("payload_msg")
                if isinstance(pm, bytes):
                    pm = f"<{len(pm)} bytes>"
                print(f"[e2e-dialogue][debug] event={parsed.get('event')!r} mt={parsed.get('message_type')!r} payload={pm!r}", flush=True)

            mt = parsed.get("message_type")
            ev = parsed.get("event")
            obj = parsed.get("payload_msg")

            if mt == "SERVER_ERROR":
                print(f"[e2e-dialogue] 服务端错误: code={parsed.get('code')!r} msg={obj!r}", flush=True)
                turn_done = True
                return
            if ev in (EVENT_SESSION_FINISHED, EVENT_SESSION_FAILED):
                turn_done = True
                return

            # TTS 音频（SERVER_ACK 的二进制 payload）
            if mt == "SERVER_ACK" and isinstance(obj, bytes) and len(obj) > 0:
                tts_queue.put(obj)
                return

            # ASR 结果
            if ev == EVENT_ASR_RESPONSE and isinstance(obj, dict):
                results = obj.get("results")
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict):
                            txt = str(r.get("text") or "").strip()
                            if txt and not r.get("is_interim"):
                                asr_text = txt
                return

            if ev == EVENT_ASR_ENDED:
                if isinstance(obj, dict):
                    results = obj.get("results")
                    if isinstance(results, list):
                        for r in results:
                            if isinstance(r, dict):
                                txt = str(r.get("text") or "").strip()
                                if txt:
                                    asr_text = txt
                return

            # LLM 回复文本
            if ev == EVENT_CHAT_RESPONSE and isinstance(obj, dict):
                c = str(obj.get("content") or "")
                if c:
                    llm_text_parts.append(c)
                return

            if ev == EVENT_TTS_PLAY_DONE:
                turn_done = True
                return

        def _recv_worker() -> None:
            try:
                while not turn_done and time.perf_counter() - t0 < self._timeout:
                    rem = self._timeout - (time.perf_counter() - t0)
                    if rem <= 0:
                        break
                    _e2e_apply_socket_timeout(ws, min(2.0, max(0.3, rem)))
                    try:
                        raw = ws.recv()
                    except Exception as e:
                        if _e2e_ws_exc_is_timeout(e):
                            continue
                        if turn_done:
                            break
                        recv_err[0] = e
                        break
                    if isinstance(raw, (bytes, bytearray)) and raw:
                        _on_parsed(_parse_openspeech_ws_response(raw))
            except Exception as e:
                recv_err[0] = e
            finally:
                tts_queue.put(None)
                recv_done.set()

        def _tts_player() -> None:
            try:
                import sounddevice as sd
                import numpy as np
                with sd.RawOutputStream(samplerate=E2E_TTS_OUTPUT_SAMPLE_RATE, channels=1, dtype="int16") as out:
                    # 播放 0.3 秒的 440Hz 柔和提示音 + 0.7 秒静音，作为蓝牙设备的绝对预热唤醒
                    sr = E2E_TTS_OUTPUT_SAMPLE_RATE
                    t = np.linspace(0, 0.3, int(sr * 0.3), False)
                    beep = np.sin(440 * 2 * np.pi * t) * 4000
                    beep_bytes = beep.astype(np.int16).tobytes()
                    silence_bytes = b'\x00' * int(sr * 0.7 * 2)
                    try:
                        out.write(beep_bytes)
                        out.write(silence_bytes)
                    except Exception:
                        pass
                    
                    while True:
                        chunk = tts_queue.get()
                        if chunk is None:
                            break
                        try:
                            out.write(chunk)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[e2e-dialogue] TTS 播放异常: {e}", flush=True)

        recv_t = threading.Thread(target=_recv_worker, name="e2e_dial_recv", daemon=True)
        play_t = threading.Thread(target=_tts_player, name="e2e_dial_play", daemon=True)
        recv_t.start()
        play_t.start()

        # 发送音频
        print(f"[gemini_v3] e2e：正在上传音频（{n_chunks} 块）…", flush=True)
        pcm_len = len(pcm_bytes)
        for off in range(0, pcm_len, chunk_bytes):
            if turn_done:
                break
            chunk = pcm_bytes[off: off + chunk_bytes]
            if not chunk:
                break
            ws.send_binary(_build_task_request_payload_gzip(session_id=self._sid, audio_data=chunk))
            if len(chunk) >= 2:
                time.sleep((len(chunk) // 2) / float(E2E_DIALOGUE_INPUT_SAMPLE_RATE))
        print("[gemini_v3] e2e：音频已全部发送，等待回复…", flush=True)

        recv_t.join(timeout=max(1.0, self._timeout - (time.perf_counter() - t0)))
        recv_done.set()
        tts_queue.put(None)
        play_t.join(timeout=10.0)

        final_asr = asr_text.strip()
        final_llm = "".join(llm_text_parts).strip()

        if final_asr:
            print(f"[识别] {final_asr}", flush=True)
        if final_llm:
            print(f"\n【随风听笛说】\n{final_llm}\n", flush=True)
        elif not final_asr:
            print("[e2e-dialogue] 本轮未获取到 ASR 或 LLM 回复。", flush=True)

        return final_asr, final_llm

    # ----------------------------------------------------------- close
    def close(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.send_binary(_build_full_client_payload_gzip(
                event_number=EVENT_FINISH_SESSION, session_id=self._sid, request_meta={},
            ))
            _e2e_apply_socket_timeout(self._ws, 5.0)
            try:
                self._ws.recv()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._ws.send_binary(_build_full_client_payload_gzip(
                event_number=EVENT_FINISH_CONNECTION, session_id=None, request_meta={},
            ))
            _e2e_apply_socket_timeout(self._ws, 3.0)
            try:
                self._ws.recv()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws = None
        self._opened = False
        print("[gemini_v3] e2e-dialogue：会话已关闭。", flush=True)

    # ------------------------------------------------ internal helpers
    def _recv_until_359(self, *, label: str = "", play_tts: bool = False) -> None:
        """阻塞接收直到 event=359（TTS 播放结束），可选播放 TTS 音频。"""
        ws = self._ws
        if ws is None:
            return
        tts_chunks: List[bytes] = []
        deadline = time.perf_counter() + min(60.0, self._timeout)
        while time.perf_counter() < deadline:
            _e2e_apply_socket_timeout(ws, min(3.0, deadline - time.perf_counter()))
            try:
                raw = ws.recv()
            except Exception as e:
                if _e2e_ws_exc_is_timeout(e):
                    continue
                break
            if not isinstance(raw, (bytes, bytearray)) or not raw:
                continue
            p = _parse_openspeech_ws_response(raw)
            ev = p.get("event")
            mt = p.get("message_type")
            if mt == "SERVER_ACK" and isinstance(p.get("payload_msg"), bytes):
                if play_tts:
                    tts_chunks.append(p["payload_msg"])
            if mt == "SERVER_ERROR":
                print(f"[e2e-dialogue][{label}] 服务端错误: {p.get('payload_msg')!r}", flush=True)
                break
            if ev == EVENT_TTS_PLAY_DONE:
                break
            if ev in (EVENT_SESSION_FINISHED, EVENT_SESSION_FAILED):
                break
        if play_tts and tts_chunks:
            self._play_tts_sync(b"".join(tts_chunks))

    def _play_tts_sync(self, pcm_data: bytes) -> None:
        """同步播放一段 TTS PCM 音频。"""
        if not pcm_data:
            return
        try:
            import sounddevice as sd
            import numpy as np
            arr = np.frombuffer(pcm_data, dtype=np.int16)
            sd.play(arr, samplerate=E2E_TTS_OUTPUT_SAMPLE_RATE, blocking=True)
        except Exception as e:
            print(f"[e2e-dialogue] TTS 播放失败: {e}", flush=True)


# ---------------------------------------------------------------------------
# E2EVoiceSession -- 麦克风录音 + 端到端对话
# ---------------------------------------------------------------------------

class E2EVoiceSession:
    """
    端到端 e2e 对话：按空格开始录音，回车结束，发送给豆包全端到端模型。
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
        e2e_dialog_app_id: str = "",
        tts_speaker: str,
        timeout_sec: float,
        mic_device: Optional[int] = None,
        jwt_sts_appid: Optional[str] = None,
        jwt_sts_access_key: Optional[str] = None,
        sts_url: str = "",
    ) -> None:
        self.simplify_zh = bool(simplify_zh)
        self.asr_glossary = dict(asr_glossary or {})
        self.tts_speaker = (tts_speaker or "").strip()
        self.mic_device = mic_device
        self._client = E2EAsrClient(
            app_key=e2e_asr_app_key,
            access_key=e2e_asr_access_key,
            resource_id=e2e_asr_resource_id,
            dialogue_app_id=e2e_dialog_app_id,
            timeout_sec=timeout_sec,
            jwt_sts_appid=jwt_sts_appid,
            jwt_sts_access_key=jwt_sts_access_key,
            sts_url=(sts_url or "").strip() or DEFAULT_E2E_STS_URL,
        )
        self._dialogue: Optional[E2EDialogueSession] = None

        self._lock = threading.Lock()
        self._recording = False
        self._chunks: List[bytes] = []
        self._submit = threading.Event()

        self._listener = None
        self._stream = None
        self._audio_bytes: bytes = b""

    # ---- 对话生命周期 ----

    def open_dialogue(self, system_role: str, battle_brief: str) -> None:
        """建立持久 WebSocket 会话（含注入快报），后续 run() 走端到端对话。"""
        self._dialogue = E2EDialogueSession(
            client=self._client,
            tts_speaker=self.tts_speaker,
            timeout_sec=self._client.timeout_sec,
        )
        self._dialogue.open(system_role, battle_brief)

    def close_dialogue(self) -> None:
        if self._dialogue is not None:
            self._dialogue.close()
            self._dialogue = None

    # ---- 录音 + 对话 ----

    def run(self, banner: str) -> Tuple[str, str]:
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

        # Windows：主键盘 Enter 与小键盘 Enter 可能同为 VK=0x0D，但 extended 标志不同，
        # pynput 的 KeyCode.__eq__ 会判为「不等于」Key.enter，导致仅 `key == Key.enter` 时收不到回车。
        def _is_enter(key: Any) -> bool:
            try:
                if key == keyboard.Key.enter:
                    return True
            except Exception:
                pass
            return getattr(key, "vk", None) == 13  # VK_RETURN

        def _on_enter_submit() -> None:
            with self._lock:
                if not self._recording:
                    self._submit.set()
                    return
                self._recording = False
                frames_bytes = b"".join(self._chunks)
                self._audio_bytes = frames_bytes
                self._submit.set()
                try:
                    if self._listener is not None:
                        self._listener.stop()
                except Exception:
                    pass

        _rec_start_time: List[Optional[float]] = [None]

        def on_press(key: Any) -> None:
            if key == keyboard.Key.space:
                with self._lock:
                    if self._recording:
                        return
                    self._recording = True
                    self._chunks = []
                    _rec_start_time[0] = time.perf_counter()
                print("[gemini_v3] >>> 录音已开始，请对着麦克风说话… 说完后按【回车】提交。", flush=True)
            elif _is_enter(key):
                _on_enter_submit()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        time.sleep(0.2)
        if not getattr(self._listener, "running", False):
            raise SystemExit(
                "[gemini_v3][语音] 键盘监听未能启动（pynput）。常见于 Cursor/VS Code 内置终端或权限限制。\n"
                "请改用「独立」窗口运行：Windows Terminal / PowerShell / cmd；必要时以管理员身份运行。\n"
                "或跳过语音：加参数 --no-voice，并用 -q \"你的问题\" 传入文字。"
            )

        print(banner, flush=True)
        print(
            "操作：按一次【空格】开始录音；【回车】结束录音并提交（语音识别整段完成，不显示实时转写）。",
            flush=True,
        )
        print(
            "提示：若按键完全无反应，多半是内置终端收不到全局键；请换外部终端运行，或 --no-voice -q \"…\"。",
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
            print(
                "[gemini_v3] 未采集到有效音频（需先按【空格】再按【回车】；或录音过短/麦克风未写入）。",
                flush=True,
            )
            return ("", "")

        # ── 录音质量诊断 ──
        _pcm_arr = np.frombuffer(self._audio_bytes, dtype=np.int16)
        _dur_sec = len(_pcm_arr) / float(self.SR)
        _peak = int(np.max(np.abs(_pcm_arr))) if len(_pcm_arr) > 0 else 0
        _rms = int(np.sqrt(np.mean(_pcm_arr.astype(np.float64) ** 2))) if len(_pcm_arr) > 0 else 0
        print(
            f"[gemini_v3] 录音统计: 时长={_dur_sec:.1f}s  采样={len(_pcm_arr)}  "
            f"峰值={_peak}/32767  RMS={_rms}",
            flush=True,
        )
        if _peak < 500:
            print(
                "[gemini_v3] ⚠ 警告：录音峰值极低（<500），疑似静音！"
                "请确认麦克风未静音且对着麦克风说话。",
                flush=True,
            )
        elif _peak < 2000:
            print(
                "[gemini_v3] ⚠ 提示：录音音量偏低（峰值<2000），服务端 VAD 可能检测不到语音。",
                flush=True,
            )
        if _dur_sec < 1.0:
            print(
                "[gemini_v3] ⚠ 提示：录音时长不足 1 秒，可能太短导致识别失败。",
                flush=True,
            )

        # ── 端到端对话 or ASR-only ──
        if self._dialogue is not None:
            print("[gemini_v3] 录音已结束，正在发送给端到端对话…", flush=True)
            asr_text, llm_text = self._dialogue.voice_turn(self._audio_bytes)
            return (asr_text.strip(), llm_text.strip())

        # 旧路径：仅 ASR（无 open_dialogue 时的回退）
        print("[gemini_v3] 录音已结束，正在连接云端识别，请稍候…", flush=True)
        raw_text = self._client.transcribe_pcm_s16le(
            pcm_s16le_bytes=self._audio_bytes,
            session_id=str(uuid.uuid4()),
            dialog_system_role="",
            tts_speaker=self.tts_speaker,
        )
        out = _asr_postprocess(
            raw_text,
            simplify_zh=self.simplify_zh,
            glossary=self.asr_glossary,
        )
        stripped = (out or "").strip()
        if not stripped:
            print(
                "[识别] （空）云端未返回有效 ASR 文本；"
                "请检查麦克风与讲话时长，或改用 --no-voice -q \"你的问题\"。",
                flush=True,
            )
        return (stripped, "")


# 主流程在「无 -q」且语音/键盘为空时退出；与 E2EVoiceSession.run 中 [识别] 行配合排查
_GEMINI_V3_EXIT_NO_QUESTION = (
    "未提供问题或语音识别为空：请使用 --question / -q，或重新语音/键盘输入。\n"
    "若已录音，请看终端中的 [识别] 行：（空）表示未得到有效文本，可重试或改用 --no-voice -q \"…\"。"
)


def _build_argparser() -> argparse.ArgumentParser:
    ap = gv2._build_argparser()
    ap.description = "（v3：e2e 端到端语音 ASR -> Coach LLM -> 豆包 TTS）"
    _v3_note = (
        "\n\n[v3 说明] 入口仍是 gemini_v3.py；日志里若出现 [gemini_v2]，多为复用了 v2 的 TTS/公用逻辑，并非改跑 v2 脚本。\n"
        "测「端到端麦克风 ASR」时不要加 --no-voice（否则会跳过 e2e，仅用 -q/键盘文字当问题）。\n"
        "跳过语音时请写完整：  python gemini_v3.py --no-voice -q \"你的问题\""
    )
    ap.epilog = (ap.epilog or "") + _v3_note
    ap.formatter_class = argparse.RawDescriptionHelpFormatter

    # e2e：STS 用 app_key 作 appid；实时对话 WebSocket 的 X-Api-App-Key 为固定产品密钥（见代码常量），勿混用
    ap.add_argument(
        "--doubao-e2e-app-key",
        default="",
        help="端到端：STS 的 appid，且若为纯数字则作为 X-Api-App-ID（不填则读 DOUBAO_E2E_APP_KEY）",
    )
    ap.add_argument(
        "--doubao-e2e-access-key",
        default="",
        help="端到端 ASR：api_access_key 的原始输入（不填则读 DOUBAO_E2E_ACCESS_KEY）",
    )
    ap.add_argument(
        "--doubao-e2e-app-id",
        default="",
        help="端到端：X-Api-App-ID（不填则读 DOUBAO_E2E_APP_ID；可与官方示例一样留空）",
    )
    ap.add_argument(
        "--doubao-e2e-resource-id",
        default=DEFAULT_DIALOGUE_RESOURCE_ID,
        help=f"端到端：X-Api-Resource-Id（默认 {DEFAULT_DIALOGUE_RESOURCE_ID}，与 eg/config 一致）",
    )
    ap.add_argument(
        "--doubao-e2e-ws-endpoint",
        default="",
        help="已弃用：现固定为 wss://openspeech.../api/v3/realtime/dialogue + Header 鉴权",
    )
    ap.add_argument(
        "--doubao-e2e-timeout",
        type=float,
        default=60.0,
        help="端到端 ASR 超时秒数（默认 60）",
    )
    ap.add_argument(
        "--doubao-e2e-sts-url",
        default="",
        help="STS 获取 jwt 的 HTTPS 地址（不填则读 DOUBAO_E2E_STS_URL，默认 openspeech.bytedance.com）",
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
    ap.add_argument(
        "--e2e-probe-text",
        default="",
        help="仅调试：不走 ASR，用 input_mod=text + ChatTextQuery(501) 直连豆包实时对话，验证鉴权/下行；执行后退出",
    )
    ap.add_argument(
        "--e2e-probe-no-hello",
        action="store_true",
        help="与 --e2e-probe-text 联用：跳过 SayHello(300) 与等待 359（默认与 eg 文本模式一致：先发开场再发 501）",
    )

    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()

    probe_txt = (getattr(args, "e2e_probe_text", "") or "").strip()
    if probe_txt:
        e2e_app_key = getattr(args, "doubao_e2e_app_key", "") or os.getenv("DOUBAO_E2E_APP_KEY", "")
        e2e_access_key = (
            getattr(args, "doubao_e2e_access_key", "") or os.getenv("DOUBAO_E2E_ACCESS_KEY", "")
        )
        jwt_api_access_key = getattr(args, "doubao_e2e_access_key_jwt", "") or os.getenv(
            "DOUBAO_E2E_API_ACCESS_KEY", ""
        )
        use_direct_jwt = bool(jwt_api_access_key)
        if use_direct_jwt:
            e2e_access_key = jwt_api_access_key
        e2e_resource_id = getattr(args, "doubao_e2e_resource_id", "") or os.getenv(
            "DOUBAO_E2E_RESOURCE_ID", DEFAULT_DIALOGUE_RESOURCE_ID
        )
        e2e_app_id = getattr(args, "doubao_e2e_app_id", "") or os.getenv("DOUBAO_E2E_APP_ID", "")
        e2e_sts_url = (
            (getattr(args, "doubao_e2e_sts_url", "") or "").strip()
            or (os.getenv("DOUBAO_E2E_STS_URL", "") or "").strip()
        )
        timeout_sec = float(getattr(args, "doubao_e2e_timeout", 60.0) or 60.0)
        speaker = getattr(args, "doubao_tts_speaker", "") or os.getenv("DOUBAO_TTS_SPEAKER", "")
        speaker = (speaker or "").strip()
        if not e2e_app_key or not e2e_access_key:
            raise SystemExit(
                "[gemini_v3][e2e] probe 需要 DOUBAO_E2E_APP_KEY / DOUBAO_E2E_ACCESS_KEY"
                "（或 --doubao-e2e-access-key-jwt / DOUBAO_E2E_API_ACCESS_KEY）"
            )
        client = E2EAsrClient(
            app_key=str(e2e_app_key),
            access_key=str(e2e_access_key),
            resource_id=str(e2e_resource_id),
            dialogue_app_id=str(e2e_app_id),
            timeout_sec=timeout_sec,
            jwt_sts_appid=None if use_direct_jwt else str(e2e_app_key),
            jwt_sts_access_key=None if use_direct_jwt else str(e2e_access_key),
            sts_url=(e2e_sts_url or "").strip() or DEFAULT_E2E_STS_URL,
        )
        out = client.probe_text_dialogue(
            query=probe_txt,
            tts_speaker=speaker,
            say_hello_first=not bool(getattr(args, "e2e_probe_no_hello", False)),
        )
        if out:
            print(f"[gemini_v3] e2e probe 收到文本：{out}", flush=True)
        else:
            print(
                "[gemini_v3] e2e probe：未解析到文本回复。可设 GEMINI_V3_E2E_DEBUG=1 查看各帧 event。",
                flush=True,
            )
        raise SystemExit(0)

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
            "DOUBAO_E2E_RESOURCE_ID", DEFAULT_DIALOGUE_RESOURCE_ID
        )
        e2e_app_id = getattr(args, "doubao_e2e_app_id", "") or os.getenv("DOUBAO_E2E_APP_ID", "")
        e2e_sts_url = (
            (getattr(args, "doubao_e2e_sts_url", "") or "").strip()
            or (os.getenv("DOUBAO_E2E_STS_URL", "") or "").strip()
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
            e2e_dialog_app_id=str(e2e_app_id),
            tts_speaker=speaker,
            timeout_sec=timeout_sec,
            mic_device=mic_device,
            # 若用户没直接提供 Jwt;token，则用 appid/accessKey 去 STS 拿临时 jwt_token，
            # 避免把 accessKey 当成 jwt_token 直接拼 Jwt; 前缀。
            jwt_sts_appid=None if use_direct_jwt else str(e2e_app_key),
            jwt_sts_access_key=None if use_direct_jwt else str(e2e_access_key),
            sts_url=e2e_sts_url,
        )

    question = (getattr(args, "question", "") or "").strip()
    summary_path: Optional[Path] = None

    # ====== summary_json / pipeline 分支 ======
    if getattr(args, "summary_json", None):
        summary_path = Path(args.summary_json).resolve()
        if not summary_path.is_file():
            raise SystemExit(f"找不到文件: {summary_path}")
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
                    + "\n【提示】对局识别已在后台运行，请稍候…\n"
                    + "-" * 60,
                    flush=True,
                )

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

    # ====== 构建上下文 ======
    bundle = gv.build_coach_bundle(args, summary_path)
    battle_brief = str(bundle.get("brief") or "")
    gv.print_coach_bundle_preview(args, bundle)
    system_role = gv._coach_system_prompt()

    # ====== 端到端对话 or 回退到旧 Coach LLM ======
    if use_voice and voice is not None:
        try:
            voice.open_dialogue(system_role, battle_brief)
        except Exception as e:
            print(f"[gemini_v3] 端到端对话连接失败: {e}", flush=True)
            raise SystemExit(1)

        try:
            turn = 0
            while True:
                turn += 1
                banner = "【语音】要问随风听笛什么？" if turn == 1 else "【语音追问】"
                asr_text, llm_text = voice.run(banner)
                if not asr_text and not llm_text:
                    break
                if not sys.stdin.isatty():
                    break
                print(
                    "\n—— 追问：【空格】开始录，【回车】结束并发送；"
                    "直接回车空提交或说 q / quit 结束。",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("\n[gemini_v3] 用户中断。", flush=True)
        finally:
            voice.close_dialogue()
    else:
        # --no-voice 或无麦克风：回退到旧的 Coach LLM 流程
        if not question:
            question = input("你有什么要问的：").strip()
        if not question:
            raise SystemExit(_GEMINI_V3_EXIT_NO_QUESTION)
        gv2.run_coach_v2_after_summary(
            args, summary_path, question, io_lock,
            log_prefix="gemini_v3",
            on_answer=tts.speak,
        )


if __name__ == "__main__":
    main()

