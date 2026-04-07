 
   
   
   
    
    # -*- coding: utf-8 -*-
"""
gemini_v1 同款流水线与教练对话，交互问题改为：**按一次空格** 开始录音（仅提示录音中，不显示实时文字）→ **回车** 结束录音，
录音时 **边录边识别**（WebSocket v3 流式协议），按【回车】结束并发送尾包后立即取最终结果。

流式 ASR 仅使用豆包流式语音识别模型2.0（v3 API WebSocket），不再使用HTTP Flash方法。

仍可编辑 data/gemini_v2_asr_glossary.json 做误识别→正写替换（长短语优先匹配）。

LLM：默认**流式**输出（OpenRouter 与 Google 直连均走流式接口）；TTS：豆包 v3 流式拉取音频块并边下边播。

当前模型版本（v2，按当前可跑通配置）：
  - ASR：豆包流式 ASR 2.0（v3 协议，model_name=bigmodel）
  - TTS：声音复刻模型 2.0（字符版）
  - Speaker：S_XL8NxUsY1（复刻音色）

依赖（需额外安装）：
  pip install sounddevice numpy pynput requests websocket-client
  # 可选：ASR 繁体 → 简体（推荐）
  pip install zhconv

并行说明（v2）：在「读 summary_json / 用缓存跳过 pipeline」时，会在你**录音同时**后台构建快报+RAG，
说完按回车后再请求 LLM，从而把 RAG 耗时叠进说话时间里（pipeline 分支仍须等识别完才有 summary，无法与 RAG 并行）。
仅发「战报+RAG」而不带问题的**额外一次** LLM 请求在没有「上下文缓存」API 时一般**更慢**，故未实现。

豆包 ASR 凭证（与 TTS 的 App Id / Access Key 一般为同一套控制台凭证）：
  DOUBAO_ASR_APP_KEY（或回退 DOUBAO_TTS_APP_ID）
  DOUBAO_ASR_ACCESS_KEY（或回退 DOUBAO_TTS_ACCESS_KEY）
  DOUBAO_ASR_WS_CLUSTER（WebSocket 流式 cluster，默认 volcengine_streaming_common）

示例：
  python gemini_v2.py
  python gemini_v2.py --no-voice              # 与 v1 相同，键盘输入
  python gemini_v2.py -q "文字问题"           # 跳过语音
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import mimetypes
import os
import queue
import re
import sys
import threading
import time
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gemini_v1 as gv
from gemini_v1 import (
    DEFAULT_IMG_DIR,
    PIPELINE_CACHE_DIR,
    _coach_first_user_message,
    _coach_followup_user_message,
    _coach_system_prompt,
    _google_gemini_key,
    _google_hist_to_contents,
    _openrouter_chat_completion,
    _openrouter_env,
    _resolve_chat_backend,
    build_coach_argparser,
    build_coach_bundle,
    print_coach_bundle_preview,
    run_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ASR_GLOSSARY_PATH = REPO_ROOT / "data" / "gemini_v2_asr_glossary.json"
DEFAULT_TTS_OUT_DIR = REPO_ROOT / "runs" / "tts"


def _load_asr_glossary_file(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        ks = str(k).strip()
        vs = str(v).strip()
        if ks and vs:
            out[ks] = vs
    return out


def _merge_asr_glossaries(override: Optional[Path]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    merged.update(_load_asr_glossary_file(DEFAULT_ASR_GLOSSARY_PATH))
    if override is not None and override.resolve() != DEFAULT_ASR_GLOSSARY_PATH.resolve():
        merged.update(_load_asr_glossary_file(override))
    return merged


def _apply_asr_glossary(text: str, glossary: Dict[str, str]) -> str:
    if not text or not glossary:
        return text
    out = text
    for wrong, right in sorted(glossary.items(), key=lambda kv: len(kv[0]), reverse=True):
        if wrong in out:
            out = out.replace(wrong, right)
    return out


def _pcm_float32_to_wav_bytes(pcm_f32: Any, sample_rate: int) -> bytes:
    import numpy as np

    wav = np.asarray(pcm_f32, dtype=np.float32).ravel()
    wav = np.clip(wav, -1.0, 1.0)
    pcm_i16 = (wav * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


# ---- 豆包流式 ASR（WebSocket，v3 API，二进制协议）----

_ASR_WS_PROTO_VER = 0b0001
_ASR_WS_CLIENT_FULL = 0b0001
_ASR_WS_CLIENT_AUDIO = 0b0010
_ASR_WS_SERVER_FULL = 0b1001
_ASR_WS_SERVER_ACK = 0b1011
_ASR_WS_SERVER_ERR = 0b1111
_ASR_WS_FLAG_NONE = 0b0000
_ASR_WS_FLAG_LAST = 0b0010
_ASR_WS_JSON = 0b0001
_ASR_WS_NONE = 0b0000
_ASR_WS_GZIP = 0b0001
_ASR_WS_SUCCESS = 1000


def _asr_ws_header(
    message_type: int,
    flags: int = _ASR_WS_FLAG_NONE,
    serialization: int = _ASR_WS_JSON,
    compression: int = _ASR_WS_GZIP,
) -> bytearray:
    hb = bytearray()
    hb.append((_ASR_WS_PROTO_VER << 4) | 1)
    hb.append((message_type << 4) | flags)
    hb.append((serialization << 4) | compression)
    hb.append(0x00)
    return hb


_ASR_VERBOSE = True  # 设为 False 可关闭 ASR 调试日志（提升性能）


def _asr_ws_parse_message(res: bytes, verbose: bool = False) -> Dict[str, Any]:
    if len(res) < 4:
        return {}
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F
    payload = res[header_size * 4 :]
    out: Dict[str, Any] = {"message_type": message_type}
    payload_msg = None
    payload_size = 0
    if verbose:
        print(f"[ASR_PARSE] header_size={header_size}, message_type={message_type}, serialization={serialization_method}, compression={message_compression}, payload_len={len(payload)}", flush=True)
    if message_type == _ASR_WS_SERVER_FULL:
        if len(payload) < 8:  # 需要 sequence (4B) + payload_size (4B)
            if verbose:
                print(f"[ASR_PARSE] SERVER_FULL payload 太短: {len(payload)}", flush=True)
            return out
        seq = int.from_bytes(payload[:4], "big", signed=True)
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        if verbose:
            print(f"[ASR_PARSE] SERVER_FULL seq={seq}, payload_size={payload_size}", flush=True)
        payload_msg = payload[8:]
    elif message_type == _ASR_WS_SERVER_ACK:
        if len(payload) >= 12:
            seq = int.from_bytes(payload[:4], "big", signed=True)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
    elif message_type == _ASR_WS_SERVER_ERR:
        if len(payload) >= 8:
            code = int.from_bytes(payload[:4], "big", signed=False)
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            out["error_code"] = code
            payload_msg = payload[8 : 8 + payload_size]
    if payload_msg is None:
        if verbose:
            print(f"[ASR_PARSE] payload_msg 为 None", flush=True)
        return out
    if verbose:
        print(f"[ASR_PARSE] payload_msg len={len(payload_msg)}", flush=True)
    if message_compression == _ASR_WS_GZIP:
        try:
            payload_msg = gzip.decompress(payload_msg)
            if verbose:
                print(f"[ASR_PARSE] 解压后长度={len(payload_msg)}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[ASR_PARSE] 解压失败: {e}", flush=True)
            return out
    if serialization_method == _ASR_WS_JSON:
        try:
            out["payload_msg"] = json.loads(str(payload_msg, "utf-8"))
            if verbose:
                print(f"[ASR_PARSE] JSON 解析成功，code={out['payload_msg'].get('code')}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[ASR_PARSE] JSON 解析失败: {e}", flush=True)
    return out


def _asr_ws_text_from_payload(pm: Any) -> str:
    if not isinstance(pm, dict):
        return ""
    
    # 如果有 code 字段，检查是否成功（旧API兼容）
    code = pm.get("code")
    if code is not None and code != _ASR_WS_SUCCESS:
        return ""
    
    res = pm.get("result")
    if not isinstance(res, dict):
        # 旧API可能是列表格式
        if isinstance(res, list):
            parts = []
            for item in res:
                if isinstance(item, dict):
                    t = str(item.get("text") or "").strip()
                    if t:
                        parts.append(t)
            return " ".join(parts).strip()
        return ""
    
    # v3 API 是字典格式，直接取 text 字段
    t = str(res.get("text") or "").strip()
    return t


def _asr_ws_stream_run(
    *,
    audio_q: "queue.Queue[Any]",
    stop_ev: threading.Event,
    appid: str,
    token: str,
    cluster: str,
    uid: str,
    sample_rate: int,
    chunk_ms: int,
    workflow: str,
    live_log: Callable[[str], None],
    verbose: bool = False,
) -> str:
    try:
        import websocket  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "流式 ASR 需要 websocket-client： pip install websocket-client"
        ) from e

    # 使用 v3 API with hardcoded credentials
    ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    connect_id = str(uuid.uuid4())
    seq = 1  # 序列号从1开始
    
    req_body: Dict[str, Any] = {
        "user": {"uid": uid},
        "audio": {
            "format": "pcm",
            "rate": sample_rate,
            "language": "zh-CN",
            "bits": 16,
            "channel": 1,
            "codec": "raw",
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "enable_ddc": True,
            "show_utterances": True,
            "result_type": "full",
        },
    }
    payload_b = gzip.compress(json.dumps(req_body).encode("utf-8"))
    # Full client request: header + seq (4B) + payload_size (4B) + payload
    full_req = bytearray(_asr_ws_header(_ASR_WS_CLIENT_FULL, flags=0b0001))  # POS_SEQUENCE flag
    full_req.extend(seq.to_bytes(4, "big", signed=True))  # 序列号（正数）
    full_req.extend(len(payload_b).to_bytes(4, "big"))
    full_req.extend(payload_b)
    seq += 1  # 发送完full request后递增

    # v3 API uses header-based authentication
    if verbose:
        print(f"[ASR] 连接到 {ws_url}", flush=True)
    ws = websocket.create_connection(
        ws_url,
        header=[
            "X-Api-App-Key: 3491963725",
            "X-Api-Access-Key: dqbpiqoB26POmsIca81QFzhWX25N6rdS",
            "X-Api-Resource-Id: volc.bigasr.sauc.duration",
            f"X-Api-Connect-Id: {connect_id}",
        ],
        timeout=30,
    )
    if verbose:
        print(f"[ASR] WebSocket 连接成功", flush=True)
    latest = ""
    try:
        if verbose:
            print(f"[ASR] 发送 full client request，seq={seq-1}，payload_size={len(payload_b)}", flush=True)
        ws.send_binary(bytes(full_req))
        pm = None
        for i in range(4):
            try:
                raw = ws.recv()
                if verbose:
                    print(f"[ASR] 收到响应包 #{i}, 大小 {len(raw)} 字节", flush=True)
                pr = _asr_ws_parse_message(raw, verbose=verbose)
                if verbose:
                    print(f"[ASR] 解析响应: message_type={pr.get('message_type')}", flush=True)
                pm = pr.get("payload_msg")
                if isinstance(pm, dict):
                    if verbose:
                        print(f"[ASR] Full request 成功，code={pm.get('code')}", flush=True)
                    break
            except Exception as e:
                if verbose:
                    print(f"[ASR] 接收/解析响应 #{i} 出错: {e}", flush=True)
                raise
        # v3 API 初始化成功后直接返回识别结果，无 code 字段；检查 result 字段
        if not isinstance(pm, dict) or "result" not in pm:
            raise RuntimeError(
                f"ASR 初始化失败: {pm!r}"
            )

        bytes_per_ms = int(sample_rate * 2 / 1000)
        chunk_bytes = max(3200, int(chunk_ms * bytes_per_ms))
        buf = bytearray()

        def _drain_queue_into_buf() -> None:
            while True:
                try:
                    arr = audio_q.get_nowait()
                except queue.Empty:
                    break
                if arr is None:
                    continue
                import numpy as np

                x = np.clip(np.asarray(arr, dtype=np.float32).ravel(), -1.0, 1.0)
                buf.extend((x * 32767.0).astype("<i2").tobytes())

        while not stop_ev.is_set() or len(buf) > 0 or not audio_q.empty():
            _drain_queue_into_buf()
            if len(buf) < chunk_bytes and not stop_ev.is_set():
                time.sleep(0.02)
                continue
            if len(buf) >= chunk_bytes:
                take = chunk_bytes
            elif stop_ev.is_set():
                take = len(buf)
                if take == 0:
                    break
            else:
                time.sleep(0.02)
                continue
            piece = bytes(buf[:take])
            del buf[:take]
            zip_b = gzip.compress(piece)
            pkt = bytearray(_asr_ws_header(_ASR_WS_CLIENT_AUDIO, flags=0b0001, serialization=_ASR_WS_NONE, compression=_ASR_WS_GZIP))  # POS_SEQUENCE
            pkt.extend(seq.to_bytes(4, "big", signed=True))  # 序列号（正数）
            pkt.extend(len(zip_b).to_bytes(4, "big"))
            pkt.extend(zip_b)
            print(f"[ASR] 发送音频包 seq={seq}, 原始={len(piece)}B 压缩={len(zip_b)}B", flush=True)
            seq += 1
            ws.send_binary(bytes(pkt))
            try:
                raw2 = ws.recv()
                if verbose:
                    print(f"[ASR] 收到音频响应，大小 {len(raw2)} 字节", flush=True)
                pr2 = _asr_ws_parse_message(raw2, verbose=verbose)
                pm2 = pr2.get("payload_msg")
                t = _asr_ws_text_from_payload(pm2)
                if t:
                    if verbose:
                        print(f"[ASR] 识别结果: {t}", flush=True)
                    latest = t
                    live_log(t)
            except Exception as e:
                if verbose:
                    print(f"[ASR] 接收音频响应出错: {e}", flush=True)
                raise

        zip_last = gzip.compress(bytes(buf)) if buf else gzip.compress(b"")
        pkt_l = bytearray(
            _asr_ws_header(_ASR_WS_CLIENT_AUDIO, flags=0b0011, serialization=_ASR_WS_NONE, compression=_ASR_WS_GZIP)  # NEG_WITH_SEQUENCE
        )
        pkt_l.extend((-seq).to_bytes(4, "big", signed=True))  # 序列号（负数表示最后一包）
        pkt_l.extend(len(zip_last).to_bytes(4, "big"))
        pkt_l.extend(zip_last)
        if verbose:
            print(f"[ASR] 发送最后一包，seq={-seq}, 大小={len(zip_last)}B", flush=True)
        ws.send_binary(bytes(pkt_l))
        if verbose:
            print(f"[ASR] 等待最终响应...", flush=True)
        try:
            ws.settimeout(3.0)
        except Exception:
            pass
        for i in range(5):
            try:
                raw_l = ws.recv()
                if verbose:
                    print(f"[ASR] 收到最终响应 #{i}, 大小 {len(raw_l)} 字节", flush=True)
                pr_l = _asr_ws_parse_message(raw_l, verbose=verbose)
                pm_l = pr_l.get("payload_msg")
                t2 = _asr_ws_text_from_payload(pm_l)
                if t2:
                    if verbose:
                        print(f"[ASR] 最终识别结果: {t2}", flush=True)
                    latest = t2
            except Exception as e:
                if verbose:
                    print(f"[ASR] 接收最终响应出错: {e}", flush=True)
                break
    finally:
        try:
            if verbose:
                print(f"[ASR] 关闭 WebSocket 连接，最终识别结果: {latest}", flush=True)
            ws.close()
        except Exception:
            pass
    return latest


_llm_session = None

def _google_gemini_chat_stream(
    *,
    system_prompt: str,
    chat_hist: List[Dict[str, str]],
    model: str,
    temperature: float = 0.35,
    timeout_s: float = 120.0,
    on_stream_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    try:
        import requests
    except ImportError as e:
        raise SystemExit("未安装 requests。请执行： pip install requests") from e

    global _llm_session
    if _llm_session is None:
        _llm_session = requests.Session()

    key = _google_gemini_key()
    if not key:
        raise RuntimeError("缺少 GEMINI_API_KEY / GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    contents = _google_hist_to_contents(chat_hist)
    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    r = _llm_session.post(
        url,
        params={"key": key, "alt": "sse"},
        json=body,
        timeout=timeout_s,
        stream=True,
    )
    if not r.ok:
        try:
            err = r.json().get("error") or {}
            msg = err.get("message", r.text[:600])
        except Exception:
            msg = r.text[:600]
        raise RuntimeError(f"Gemini stream HTTP {r.status_code}: {msg}")
    r.encoding = "utf-8"
    acc: List[str] = []
    for raw in r.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, (bytes, bytearray)) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        for cand in evt.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content") or {}
            for p in content.get("parts") or []:
                if not isinstance(p, dict):
                    continue
                t = p.get("text")
                if isinstance(t, str) and t:
                    acc.append(t)
                    if callable(on_stream_chunk):
                        on_stream_chunk(t)
    return "".join(acc).strip()


def _coach_chat_turn_streaming(
    chat_hist: List[Dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.35,
    on_stream_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    if not chat_hist or chat_hist[0].get("role") != "user":
        raise ValueError("chat_hist 必须以 user 开头")
    backend = _resolve_chat_backend()
    sys_p = _coach_system_prompt()
    if backend == "google":
        gm = (model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")).strip()
        print(
            f"[gemini_v2] LLM 后端: google（流式） 模型: {gm}  上下文: {len(chat_hist)} 条",
            flush=True,
        )
        return _google_gemini_chat_stream(
            system_prompt=sys_p,
            chat_hist=list(chat_hist),
            model=gm,
            temperature=temperature,
            on_stream_chunk=on_stream_chunk,
        )
    om = model or _openrouter_env()[2]
    print(
        f"[gemini_v2] LLM 后端: openrouter（流式） 模型: {om}  上下文: {len(chat_hist)} 条",
        flush=True,
    )
    return _openrouter_chat_completion(
        messages=[{"role": "system", "content": sys_p}, *chat_hist],
        model=om,
        temperature=temperature,
        stream_output=True,
        on_stream_chunk=on_stream_chunk,
    )


def run_coach_v2_after_summary(
    args: argparse.Namespace,
    summary_path: Path,
    question: str,
    io_lock: threading.Lock,
    *,
    follow_up_reader: Optional[Callable[[], str]] = None,
    log_prefix: str = "gemini_v2",
    coach_bundle: Optional[Dict[str, Any]] = None,
    skip_bundle_preview_print: bool = False,
    on_answer: Optional[Callable[[str, int], None]] = None,
    max_history_turns: int = 3,
) -> None:
    """同 gemini_v1.run_coach_after_summary，但 LLM 固定走流式接口。"""
    if coach_bundle is None:
        coach_bundle = build_coach_bundle(args, summary_path)
        if not skip_bundle_preview_print:
            print_coach_bundle_preview(args, coach_bundle)
    elif not skip_bundle_preview_print:
        print_coach_bundle_preview(args, coach_bundle)

    brief = str(coach_bundle.get("brief") or "")
    rag_block = str(coach_bundle.get("rag_block") or "")
    chess_block = str(coach_bundle.get("chess_block") or "")
    t_prep0 = float(coach_bundle.get("t_prep0") or 0.0)
    t_prep1 = float(coach_bundle.get("t_prep1") or 0.0)

    chat_hist: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": _coach_first_user_message(
                brief, rag_block, chess_block, question
            ),
        }
    ]
    max_history_turns = max(1, int(max_history_turns or 3))
    ll_total = 0.0
    turn = 0
    answer = ""

    def _trim_chat_hist() -> None:
        """保留首轮 user 消息 + 最近 max_history_turns 轮追问。"""
        if len(chat_hist) <= 1 + max_history_turns * 2:
            return
        first = chat_hist[0]
        trimmed = chat_hist[-max_history_turns * 2:]
        chat_hist.clear()
        chat_hist.append(first)
        chat_hist.extend(trimmed)

    def _default_follow_up() -> str:
        io_lock.acquire()
        try:
            return input("哈基星：").strip()
        finally:
            io_lock.release()

    reader = follow_up_reader if follow_up_reader is not None else _default_follow_up

    while True:
        turn += 1
        print(f"\n正在请求 LLM（第 {turn} 轮，流式）…\n", flush=True)
        print_to_tty = bool(sys.stdout.isatty())
        if print_to_tty:
            print("【随风听笛说】\n", flush=True)
        t_ll0 = time.perf_counter()
        first_chunk_ts: List[Optional[float]] = [None]
        flush_ts: List[float] = [t_ll0]
        stream_buf: List[str] = []

        def _on_chunk(chunk: str) -> None:
            now = time.perf_counter()
            if first_chunk_ts[0] is None:
                first_chunk_ts[0] = now
            stream_buf.append(chunk)
            if print_to_tty and (now - flush_ts[0] >= 0.2):
                out = "".join(stream_buf)
                stream_buf.clear()
                print(out, end="", flush=True)
                flush_ts[0] = now

        answer_raw = _coach_chat_turn_streaming(
            chat_hist,
            temperature=0.35,
            on_stream_chunk=_on_chunk,
        )
        answer = str(answer_raw or "").strip()
        t_ll1 = time.perf_counter()

        if print_to_tty and stream_buf:
            print("".join(stream_buf), end="", flush=True)
            stream_buf.clear()
            print(flush=True)

        total_sec = t_ll1 - t_ll0
        if first_chunk_ts[0] is not None:
            ttft = first_chunk_ts[0] - t_ll0
            stream_sec = t_ll1 - first_chunk_ts[0]
            ll_total += total_sec
            print(
                f"[{log_prefix}] 第 {turn} 轮 LLM：首包 {ttft:.2f}s（排队/网络/模型）"
                f" + 流式输出 {stream_sec:.2f}s = 合计 {total_sec:.2f}s",
                flush=True,
            )
        else:
            ll_total += total_sec
            print(
                f"[{log_prefix}] 第 {turn} 轮 LLM 合计: {total_sec:.2f}s",
                flush=True,
            )

        if not print_to_tty:
            print("\n【随风听笛说】\n", flush=True)
            print(answer, flush=True)
            print(flush=True)
        if on_answer is not None:
            try:
                on_answer(answer, turn)
            except Exception as e:
                print(f"[{log_prefix}] TTS 回调失败（已忽略）: {e}", flush=True)
        if not sys.stdin.isatty():
            break
        print(
            "—— 首轮已含完整战术快报，追问不必重跑检索；"
            "模型依赖对话历史继续推理（上下文过长被截断时需新开一局或重喂快报）。"
        )
        print("—— 继续提问直接输入；空行或 q / quit / exit 结束。\n")
        try:
            nxt = reader()
        except EOFError:
            break
        if not nxt or nxt.lower() in ("q", "quit", "exit", "bye", "再见"):
            break
        chat_hist.append({"role": "assistant", "content": answer})
        chat_hist.append(
            {"role": "user", "content": _coach_followup_user_message(nxt)}
        )
        _trim_chat_hist()

    print(f"[{log_prefix}] 本回合步骤耗时")
    if args.no_rag:
        print(f"  读 JSON + 构建快报（无 RAG）: {t_prep1 - t_prep0:.2f}s")
    else:
        print(f"  读 JSON + 构建快报 + 阵容检索与棋子智库: {t_prep1 - t_prep0:.2f}s")
    print(f"  LLM 合计（{turn} 轮）: {ll_total:.2f}s")
    print()


def _voice_with_parallel_coach_bundle(
    args: argparse.Namespace,
    summary_path: Path,
    read_question: Callable[[], str],
) -> tuple[str, Dict[str, Any]]:
    """
    录音同时后台 build_coach_bundle；适合已有 summary_path 的分支（--summary-json / 缓存跳过 pipeline）。
    返回 (用户问题, bundle)；打印预览后再由 run_coach_v2_after_summary(..., coach_bundle=) 走 LLM。
    """
    holder: Dict[str, Any] = {}
    err: List[Optional[BaseException]] = [None]
    done = threading.Event()

    def _prep() -> None:
        try:
            holder["b"] = build_coach_bundle(args, summary_path)
        except BaseException as e:
            err[0] = e
        finally:
            done.set()

    threading.Thread(target=_prep, daemon=True, name="coach_bundle_prep").start()
    q = read_question()
    done.wait()
    if err[0] is not None:
        raise err[0]
    bundle = holder["b"]
    print_coach_bundle_preview(args, bundle)
    return q.strip(), bundle


class VoiceSession:
    """按一次空格开始录音；流式 ASR 边录边识别，回车发尾包并结束。"""

    SR = 16000

    def __init__(
        self,
        *,
        simplify_zh: bool = True,
        asr_glossary: Optional[Dict[str, str]] = None,
        doubao_asr_app_key: str = "",
        doubao_asr_access_key: str = "",
        doubao_asr_resource_id: str = "volc.bigasr.sauc.duration",
        asr_timeout_sec: float = 120.0,
        doubao_asr_ws_cluster: str = "volcengine_streaming_common",
        asr_chunk_ms: int = 200,
        asr_stream: bool = True,
        asr_workflow: str = "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
        asr_verbose: bool = False,
    ) -> None:
        self.simplify_zh = bool(simplify_zh)
        self.asr_glossary = dict(asr_glossary or {})
        self.doubao_asr_app_key = str(doubao_asr_app_key or "").strip()
        self.doubao_asr_access_key = str(doubao_asr_access_key or "").strip()
        self.doubao_asr_resource_id = str(doubao_asr_resource_id or "").strip()
        self.asr_timeout_sec = max(10.0, float(asr_timeout_sec or 120.0))
        self.doubao_asr_ws_cluster = str(doubao_asr_ws_cluster or "").strip()
        self.asr_chunk_ms = max(40, min(500, int(asr_chunk_ms or 200)))
        self.asr_stream = bool(asr_stream)
        self.asr_workflow = str(asr_workflow or "").strip()
        self.asr_verbose = bool(asr_verbose)
        self._lock = threading.Lock()
        self._stdout_lock = threading.Lock()
        self.segments: List[str] = []
        self.recording = False
        self._recording_started_at = 0.0
        self._chunks: List[Any] = []
        self._submit = threading.Event()
        self._stream = None
        self._listener = None
        self._audio_q: queue.Queue[Any] = queue.Queue(maxsize=512)
        self._asr_stop = threading.Event()
        self._asr_ws_thread: Optional[threading.Thread] = None
        self._asr_ws_holder: Dict[str, Any] = {}
        self._live_log_ts = 0.0

    def warmup(self) -> None:
        """豆包 ASR 无本地权重加载；预留钩子便于以后做连通性预检。"""

    def _resolve_asr_creds(self) -> tuple[str, str]:
        # Hardcoded credentials for v3 API
        return "3491963725", "dqbpiqoB26POmsIca81QFzhWX25N6rdS"

    def _live_asr_log(self, text: str) -> None:
        now = time.perf_counter()
        if now - self._live_log_ts < 0.25:
            return
        self._live_log_ts = now
        t = (text or "").replace("\n", " ")
        if len(t) > 100:
            t = t[:100] + "…"

        def _safe_write(six: str) -> None:
            with self._stdout_lock:
                sys.stdout.write(six)
                sys.stdout.flush()

        _safe_write(f"\r\x1b[2K[听写] {t}")

    def _asr_postprocess(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
        # SenseVoice 结果常带控制标签，如 <|zh|><|NEUTRAL|>，在终端问句中去掉。
        text = re.sub(r"<\|[^|>]+\|>", "", text).strip()
        if self.simplify_zh:
            try:
                import zhconv
            except ImportError:
                pass
            else:
                text = zhconv.convert(text, "zh-cn")
        text = _apply_asr_glossary(text, self.asr_glossary)
        return text

    def _transcribe(self, audio) -> str:
        """
        备用整段识别方法（已改为只使用流式识别）。
        如果流式识别失败，建议重新录音而不是尝试其他方法。
        """
        return ""

    def run(self, banner: str) -> str:
        try:
            import numpy as np
            import sounddevice as sd
            from pynput import keyboard
        except ImportError as e:
            raise SystemExit(
                "语音依赖未就绪。请执行： pip install sounddevice numpy pynput requests\n"
                + str(e)
            ) from e

        def _safe_write(s: str, *, flush: bool = True) -> None:
            with self._stdout_lock:
                sys.stdout.write(s)
                if flush:
                    sys.stdout.flush()

        self._submit.clear()
        self.segments = []
        self.recording = False
        with self._lock:
            self._chunks.clear()

        print(banner, flush=True)
        print(
            "操作：按一次【空格】开始录音；"
            "【回车】结束录音并提交（未按空格直接回车视为空问题）。Ctrl+C 退出。\n",
            flush=True,
        )

        def audio_cb(indata, frames, tcb, status):
            if not self.recording:
                return
            chunk = indata.copy().ravel()
            with self._lock:
                self._chunks.append(chunk)
            if (
                self.asr_stream
                and self._asr_ws_thread is not None
                and self._asr_ws_thread.is_alive()
            ):
                try:
                    self._audio_q.put_nowait(chunk)
                except queue.Full:
                    try:
                        _ = self._audio_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._audio_q.put_nowait(chunk)
                    except queue.Full:
                        pass

        self._stream = sd.InputStream(
            samplerate=self.SR,
            channels=1,
            dtype="float32",
            callback=audio_cb,
            blocksize=1024,
        )
        self._stream.start()

        def on_press(key):
            if key == keyboard.Key.space:
                with self._lock:
                    if self.recording:
                        return
                    self.recording = True
                    self._recording_started_at = time.perf_counter()
                    self._chunks.clear()
                    self._live_log_ts = 0.0
                while True:
                    try:
                        self._audio_q.get_nowait()
                    except queue.Empty:
                        break
                self._asr_stop.clear()
                self._asr_ws_holder.clear()
                self._asr_ws_thread = None
                app_key, access = self._resolve_asr_creds()
                cluster = (
                    self.doubao_asr_ws_cluster
                    or os.getenv("DOUBAO_ASR_WS_CLUSTER", "").strip()
                    or "volcengine_streaming_common"
                )
                stream_on = self.asr_stream and bool(app_key and access)
                if stream_on:
                    holder = self._asr_ws_holder

                    def _run_ws() -> None:
                        try:
                            holder["t"] = _asr_ws_stream_run(
                                audio_q=self._audio_q,
                                stop_ev=self._asr_stop,
                                appid=app_key,
                                token=access,
                                cluster=cluster,
                                uid=app_key[:32],
                                sample_rate=self.SR,
                                chunk_ms=self.asr_chunk_ms,
                                workflow=self.asr_workflow
                                or "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
                                live_log=self._live_asr_log,
                            )
                        except Exception as e:
                            holder["e"] = e

                    th = threading.Thread(target=_run_ws, daemon=True, name="asr_ws")
                    th.start()
                    self._asr_ws_thread = th
                _safe_write("\r\x1b[2K")
                msg = "[语音] 录音中"
                if stream_on:
                    msg += "（流式识别，回车即出结果）"
                msg += "…"
                print(msg, flush=True)
            elif key == keyboard.Key.enter:
                with self._lock:
                    was_rec = self.recording
                    now_t = time.perf_counter()
                    if was_rec and (now_t - float(self._recording_started_at or 0.0)) < 0.35:
                        return
                    self.recording = False
                    self._recording_started_at = 0.0
                    if self._chunks:
                        hold = [x.copy() for x in self._chunks]
                    else:
                        hold = []
                    self._chunks.clear()
                text_final = ""
                self._asr_stop.set()
                ws_th = self._asr_ws_thread
                if ws_th is not None and ws_th.is_alive():
                    ws_th.join(timeout=min(30.0, self.asr_timeout_sec))
                self._asr_ws_thread = None
                err = self._asr_ws_holder.get("e")
                raw_ws = str(self._asr_ws_holder.get("t") or "").strip()
                _safe_write("\r\x1b[2K")
                if err is not None and not raw_ws:
                    print(f"[语音] 流式识别异常，回退整段识别：{err}", flush=True)
                if raw_ws:
                    text_final = self._asr_postprocess(raw_ws)
                if not text_final and was_rec and hold:
                    print("[语音] 整段识别中…", flush=True)
                    text_final = self._transcribe(np.concatenate(hold))
                with self._lock:
                    self.segments.clear()
                    if text_final:
                        self.segments.append(text_final)
                self._submit.set()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

        try:
            self._submit.wait()
        except KeyboardInterrupt:
            self._submit.set()
        finally:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._stream.stop()
            self._stream.close()
            _safe_write("\n")

        out = "".join(self.segments).strip()
        if out:
            print(f"[识别] {out}", flush=True)
        return out


def _doubao_tts_speech_rate_from_speed(speed: float) -> int:
    """火山文档：实际语速 ≈ 0.05 * speech_rate配置 + 0.5。"""
    v = (float(speed) - 0.5) / 0.05
    return max(-50, min(100, int(round(v))))


def _infer_doubao_resource_id_from_speaker(speaker: str) -> Optional[str]:
    s = (speaker or "").strip().lower()
    if not s:
        return None
    # 声音复刻音色（S_xxx / ICL_xxx）走 seed-icl 通道，默认优先 2.0 字符版。
    if s.startswith("s_") or s.startswith("icl_"):
        return "seed-icl-2.0"
    # 官方内置音色：saturn_* / *_jupiter_bigtts 归到 TTS 2.0。
    if s.startswith("saturn_") or "_jupiter_bigtts" in s:
        return "seed-tts-2.0"
    # 其余常见内置 bigtts 音色默认按 TTS 1.0。
    return "seed-tts-1.0"


def _tts_model_label_from_resource_id(resource_id: str) -> str:
    rid = (resource_id or "").strip().lower()
    mapping = {
        "seed-tts-1.0": "豆包语音合成模型 1.0（字符版）",
        "volc.service_type.10029": "豆包语音合成模型 1.0（字符版）",
        "seed-tts-1.0-concurr": "豆包语音合成模型 1.0（并发版）",
        "volc.service_type.10048": "豆包语音合成模型 1.0（并发版）",
        "seed-tts-2.0": "豆包语音合成模型 2.0（字符版）",
        "seed-icl-1.0": "声音复刻模型 1.0（字符版）",
        "seed-icl-1.0-concurr": "声音复刻模型 1.0（并发版）",
        "seed-icl-2.0": "声音复刻模型 2.0（字符版）",
    }
    return mapping.get(rid, f"未知资源模型（{resource_id}）")


class TTSPlayer:
    """将 LLM 文本调用豆包 TTS 合成，并可选本地播放。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.backend = "doubao"
        self.timeout_sec = max(5.0, float(getattr(args, "tts_timeout", 45.0) or 45.0))
        self.play = bool(getattr(args, "tts_play", True))
        self.save = bool(getattr(args, "tts_save", False))
        self.out_dir = Path(getattr(args, "tts_out_dir", DEFAULT_TTS_OUT_DIR)).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.speed = float(getattr(args, "tts_speed", 1.25) or 1.25)
        emo = str(getattr(args, "doubao_tts_emotion", "") or "").strip()
        if not emo:
            emo = os.getenv("DOUBAO_TTS_EMOTION", "").strip()
        self.doubao_emotion = emo
        emo_sc = getattr(args, "doubao_tts_emotion_scale", None)
        if emo_sc is None:
            self.doubao_emotion_scale: Optional[int] = 4 if emo else None
        else:
            self.doubao_emotion_scale = max(1, min(5, int(emo_sc)))
        self.doubao_loudness = float(
            getattr(args, "doubao_tts_loudness", 1.15) or 1.15
        )
        # doubao tts (volcengine) config
        self.doubao_app_id = str(getattr(args, "doubao_tts_app_id", "") or "").strip()
        self.doubao_access_key = str(getattr(args, "doubao_tts_access_key", "") or "").strip()
        self.doubao_resource_id = str(
            getattr(args, "doubao_tts_resource_id", "") or ""
        ).strip()
        self.doubao_speaker = str(getattr(args, "doubao_tts_speaker", "") or "").strip()
        self.doubao_uid = str(getattr(args, "doubao_tts_uid", "gemini_v2") or "gemini_v2").strip()
        self.doubao_audio_format = str(
            getattr(args, "doubao_tts_audio_format", "pcm") or "pcm"
        ).strip().lower()
        self.doubao_sample_rate = int(getattr(args, "doubao_tts_sample_rate", 24000) or 24000)
        
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            self.session = None

    def _write_audio_bytes(self, raw: bytes, turn: int, ext_hint: str = ".wav") -> Path:
        ext = ext_hint if ext_hint.startswith(".") else f".{ext_hint}"
        p = self.out_dir / f"answer_turn_{turn}_{int(time.time())}{ext}"
        p.write_bytes(raw)
        return p

    def _extract_audio_payload_from_json(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 常见返回：base64、本地 path、可下载 url
        for k in ("audio_base64", "wav_base64", "audio"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                try:
                    raw = base64.b64decode(v)
                except Exception:
                    continue
                return {"raw": raw, "ext": ".wav"}

        for k in ("path", "audio_path"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                p = Path(v).expanduser()
                if p.is_file():
                    return {"path": p.resolve()}

        for k in ("url", "audio_url"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                with urllib.request.urlopen(v, timeout=self.timeout_sec) as resp:
                    raw = resp.read()
                    ct = str(resp.headers.get("Content-Type") or "")
                ext = mimetypes.guess_extension(ct.split(";")[0].strip()) or ".wav"
                return {"raw": raw, "ext": ext}
        return None


    def _pcm_to_wav_bytes(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def _doubao_tts_request(self, text: str, audio_format: str):
        try:
            import requests
        except ImportError as e:
            raise SystemExit("未安装 requests。请执行： pip install requests") from e

        app_id = self.doubao_app_id or os.getenv("DOUBAO_TTS_APP_ID", "").strip()
        access_key = self.doubao_access_key or os.getenv("DOUBAO_TTS_ACCESS_KEY", "").strip()
        resource_id = self.doubao_resource_id or os.getenv("DOUBAO_TTS_RESOURCE_ID", "seed-tts-1.0").strip()
        speaker = self.doubao_speaker or os.getenv("DOUBAO_TTS_SPEAKER", "").strip()
        uid = self.doubao_uid or os.getenv("DOUBAO_TTS_UID", "gemini_v2").strip()
        if not app_id or not access_key or not speaker:
            print(
                "[gemini_v2][TTS] 豆包 TTS 缺少配置：需要 APP_ID / ACCESS_KEY / SPEAKER。",
                flush=True,
            )
            return None

        inferred_resource = _infer_doubao_resource_id_from_speaker(speaker)
        if inferred_resource and resource_id != inferred_resource:
            print(
                f"[gemini_v2][TTS] 检测到 speaker={speaker} 更匹配 {inferred_resource}，"
                f"当前 resource_id={resource_id}，已自动改用 {inferred_resource}"
                f"（{_tts_model_label_from_resource_id(inferred_resource)}）。",
                flush=True,
            )
            resource_id = inferred_resource
        else:
            print(
                f"[gemini_v2][TTS] 当前配置：speaker={speaker}  resource_id={resource_id}"
                f"（{_tts_model_label_from_resource_id(resource_id)}）",
                flush=True,
            )

        url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
        headers = {
            "X-Api-App-Id": app_id,
            "X-Api-Access-Key": access_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }
        aparam: Dict[str, Any] = {
            "format": audio_format,
            "sample_rate": self.doubao_sample_rate,
            "speech_rate": _doubao_tts_speech_rate_from_speed(self.speed),
            "enable_timestamp": True,
        }
        if abs(self.doubao_loudness - 1.0) > 0.01:
            aparam["loudness_ratio"] = float(self.doubao_loudness)
        req: Dict[str, Any] = {
            "text": text,
            "speaker": speaker,
            "audio_params": aparam,
        }
        if self.doubao_emotion:
            req["emotion"] = self.doubao_emotion
            if self.doubao_emotion_scale is not None:
                req["emotion_scale"] = int(self.doubao_emotion_scale)

        payload: Dict[str, Any] = {"user": {"uid": uid}, "req_params": req}
        if self.session is not None:
            return self.session.post(
                url, headers=headers, json=payload, stream=True, timeout=self.timeout_sec, verify=True
            )
        return requests.post(
            url, headers=headers, json=payload, stream=True, timeout=self.timeout_sec, verify=True
        )

    def _consume_doubao_tts_stream(
        self,
        resp: Any,
        *,
        on_pcm_chunk: Optional[Callable[[bytes], None]] = None,
        collect: Optional[List[bytes]] = None,
    ) -> bool:
        """解析豆包 TTS 流式 JSON 行；有音频块则回调或追加到 collect。有有效载荷返回 True。"""
        got = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            s = str(line).strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            code = int(obj.get("code", -1))
            if code == 0:
                # 处理音频数据
                b64 = obj.get("data")
                if isinstance(b64, str) and b64:
                    try:
                        chunk = base64.b64decode(b64)
                    except Exception:
                        continue
                    got = True
                    if collect is not None:
                        collect.append(chunk)
                    if on_pcm_chunk is not None:
                        on_pcm_chunk(chunk)
                    continue
                # 处理时间戳数据（sentence_data）
                if "sentence" in obj and obj.get("sentence"):
                    continue
            elif code == 20000000:
                # 合成完成，可包含usage统计信息
                break
            else:
                msg = str(obj.get("message") or "unknown")
                hint = ""
                if code == 55000000 and "resource" in msg.lower() and "speaker" in msg.lower():
                    hint = "\n  【诊断】Resource ID 与 Speaker 不匹配。检查：\n" \
                           "   • 豆包语音合成模型1.0：seed-tts-1.0 / volc.service_type.10029（字符版）\n" \
                           "   • 豆包语音合成模型1.0并发：seed-tts-1.0-concurr / volc.service_type.10048\n" \
                           "   • 豆包语音合成模型2.0：seed-tts-2.0（字符版）\n" \
                           "   • 声音复刻1.0：seed-icl-1.0（字符版）/ seed-icl-1.0-concurr（并发版）\n" \
                           "   • 声音复刻2.0：seed-icl-2.0（字符版）\n" \
                           "   • 使用 --doubao-tts-resource-id 或环境变量 DOUBAO_TTS_RESOURCE_ID 修改"
                raise RuntimeError(f"豆包 TTS 返回异常 code={code} message={msg}{hint}")
        return got

    def _call_doubao_tts_buffered(self, text: str) -> Optional[Dict[str, Any]]:
        """mp3/ogg 等非 pcm：整段缓冲后再播放/落盘。"""
        audio_format = (
            self.doubao_audio_format
            if self.doubao_audio_format in {"pcm", "mp3", "ogg_opus"}
            else "pcm"
        )
        resp = self._doubao_tts_request(text, audio_format)
        if resp is None:
            return None
        if resp.status_code >= 400:
            raise RuntimeError(f"豆包 TTS HTTP {resp.status_code}: {(resp.text or '')[:300]}")
        parts: List[bytes] = []
        if not self._consume_doubao_tts_stream(resp, collect=parts):
            return None
        raw = b"".join(parts)
        if audio_format == "pcm":
            wav_bytes = self._pcm_to_wav_bytes(raw, self.doubao_sample_rate)
            return {"raw": wav_bytes, "ext": ".wav"}
        if audio_format == "mp3":
            return {"raw": raw, "ext": ".mp3"}
        return {"raw": raw, "ext": ".ogg"}

    def _play_wav_payload(self, payload: Dict[str, Any]) -> None:
        if not self.play:
            return
        raw = payload.get("raw")
        ext = str(payload.get("ext") or "").lower()
        p = payload.get("path")

        if raw and os.name == "nt" and ext == ".wav":
            import winsound

            winsound.PlaySound(raw, winsound.SND_MEMORY)
            return
        if isinstance(p, Path) and p.is_file() and os.name == "nt" and p.suffix.lower() == ".wav":
            import winsound

            winsound.PlaySound(str(p), winsound.SND_FILENAME)
            return
        print("[gemini_v2][TTS] 自动播放：非 wav 或未在 Windows 上时请改用 --doubao-tts-audio-format pcm + sounddevice。")

    def speak(self, text: str, turn: int) -> None:
        t = (text or "").strip()
        if not t:
            return
        audio_format = (
            self.doubao_audio_format
            if self.doubao_audio_format in {"pcm", "mp3", "ogg_opus"}
            else "pcm"
        )
        print(f"[gemini_v2][TTS] 第 {turn} 轮流式合成中…", flush=True)

        if audio_format != "pcm":
            payload = self._call_doubao_tts_buffered(t)
            if payload is None:
                print("[gemini_v2][TTS] doubao 未返回可用音频。")
                return
            if self.play:
                self._play_wav_payload(payload)
                print(f"[gemini_v2][TTS] 第 {turn} 轮语音已播放。")
            if self.save:
                raw = payload.get("raw")
                ext = str(payload.get("ext") or ".wav")
                if raw:
                    sp = self._write_audio_bytes(raw, turn, ext)
                    print(f"[gemini_v2][TTS] 第 {turn} 轮语音已保存: {sp}")
            if not self.play and not self.save:
                print("[gemini_v2][TTS] 已合成语音（未播放、未保存）。")
            return

        resp = self._doubao_tts_request(t, "pcm")
        if resp is None:
            return
        if resp.status_code >= 400:
            print(f"[gemini_v2][TTS] HTTP {resp.status_code}: {(resp.text or '')[:300]}", flush=True)
            return

        pcm_chunks: List[bytes] = []
        out_stream = None

        if self.play:
            try:
                import sounddevice as sd
            except ImportError:
                print(
                    "[gemini_v2][TTS] 流式播放需要 sounddevice（pip install sounddevice）"
                    "，将改为整段缓冲后用内存 wav 播放。",
                    flush=True,
                )
                self._consume_doubao_tts_stream(resp, collect=pcm_chunks)
                raw_pcm = b"".join(pcm_chunks)
                if raw_pcm:
                    wav_b = self._pcm_to_wav_bytes(raw_pcm, self.doubao_sample_rate)
                    self._play_wav_payload({"raw": wav_b, "ext": ".wav"})
                    print(f"[gemini_v2][TTS] 第 {turn} 轮语音已播放。", flush=True)
                    if self.save:
                        sp = self._write_audio_bytes(wav_b, turn, ".wav")
                        print(f"[gemini_v2][TTS] 第 {turn} 轮语音已保存: {sp}", flush=True)
                return

            out_stream = sd.RawOutputStream(
                samplerate=self.doubao_sample_rate,
                channels=1,
                dtype="int16",
                blocksize=4096,
            )
            out_stream.start()

            def _on_chunk(b: bytes) -> None:
                pcm_chunks.append(b)
                if out_stream is not None:
                    out_stream.write(b)

            try:
                if not self._consume_doubao_tts_stream(resp, on_pcm_chunk=_on_chunk):
                    print("[gemini_v2][TTS] doubao 未返回可用音频。", flush=True)
            finally:
                out_stream.stop()
                out_stream.close()
        else:
            self._consume_doubao_tts_stream(resp, collect=pcm_chunks)
            if not pcm_chunks:
                print("[gemini_v2][TTS] doubao 未返回可用音频。", flush=True)

        raw_pcm = b"".join(pcm_chunks)
        if self.play and raw_pcm:
            print(f"[gemini_v2][TTS] 第 {turn} 轮语音已播放。", flush=True)
        if self.save and raw_pcm:
            wav_b = self._pcm_to_wav_bytes(raw_pcm, self.doubao_sample_rate)
            sp = self._write_audio_bytes(wav_b, turn, ".wav")
            print(f"[gemini_v2][TTS] 第 {turn} 轮语音已保存: {sp}", flush=True)
        if not self.play and not self.save and raw_pcm:
            print("[gemini_v2][TTS] 已拉取流式语音（未播放、未保存）。", flush=True)


def _build_argparser() -> argparse.ArgumentParser:
    ap = build_coach_argparser()
    ap.description = (ap.description or "") + "（v2：语音空格录制 + 回车提交）"
    ap.add_argument(
        "--no-voice",
        action="store_true",
        help="不用语音，终端键盘输入（与 gemini_v1 一致）",
    )
    ap.add_argument(
        "--doubao-asr-app-key",
        default="",
        help="豆包ASR: X-Api-App-Key（默认读 DOUBAO_ASR_APP_KEY，或回退 DOUBAO_TTS_APP_ID）",
    )
    ap.add_argument(
        "--doubao-asr-access-key",
        default="",
        help="豆包ASR: X-Api-Access-Key（默认读 DOUBAO_ASR_ACCESS_KEY，或回退 DOUBAO_TTS_ACCESS_KEY）",
    )
    ap.add_argument(
        "--doubao-asr-resource-id",
        default="volc.bigasr.sauc.duration",
        help="豆包ASR: X-Api-Resource-Id（默认 volc.bigasr.sauc.duration）",
    )
    ap.add_argument(
        "--asr-timeout",
        type=float,
        default=120.0,
        help="豆包 ASR 请求超时秒数（默认 120）",
    )
    ap.add_argument(
        "--doubao-asr-ws-cluster",
        default="",
        help="流式 ASR 的 cluster（默认读 DOUBAO_ASR_WS_CLUSTER 或 volcengine_streaming_common）",
    )
    ap.add_argument(
        "--asr-chunk-ms",
        type=int,
        default=200,
        help="流式 ASR 每包时长约多少毫秒音频（40～500，默认 200）",
    )
    ap.add_argument(
        "--asr-no-stream",
        action="store_true",
        help="关闭边录边识别，仅回车后用 HTTP flash 整段识别",
    )
    ap.add_argument(
        "--asr-verbose",
        action="store_true",
        help="开启 ASR 调试日志（默认关闭，开启后会打印每个音频包的收发细节）",
    )
    ap.add_argument(
        "--asr-workflow",
        default="audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
        help="流式 ASR 的 workflow（默认含逆规整与智能标点）",
    )
    ap.add_argument(
        "--no-zh-simplify",
        action="store_true",
        help="不对 ASR 结果做繁转简（默认安装 zhconv 时转为大陆简体）",
    )
    ap.add_argument(
        "--max-history-turns",
        type=int,
        default=3,
        help="追问时保留最近多少轮对话历史（默认 3，防止越问越慢）",
    )
    ap.add_argument(
        "--asr-glossary",
        type=Path,
        default=None,
        help="可选 JSON：{\"常误听\":\"正写\"}，会与 data/gemini_v2_asr_glossary.json 合并（键越长越先替换）",
    )
    ap.add_argument(
        "--tts-timeout",
        type=float,
        default=45.0,
        help="TTS 接口超时秒数（默认 45）",
    )
    ap.add_argument(
        "--tts-play",
        action="store_true",
        default=True,
        help="收到 TTS 音频后尝试本地播放（默认开启，内存直播放，不落盘）",
    )
    ap.add_argument(
        "--no-tts-play",
        dest="tts_play",
        action="store_false",
        help="关闭 TTS 自动播放（仍可保留 TTS 合成能力）",
    )
    ap.add_argument(
        "--tts-save",
        action="store_true",
        help="将 TTS 音频落盘保存到 --tts-out-dir（默认不保存）",
    )
    ap.add_argument(
        "--tts-out-dir",
        type=Path,
        default=DEFAULT_TTS_OUT_DIR,
        help="TTS 音频保存目录（默认 runs/tts）",
    )
    ap.add_argument(
        "--tts-speed",
        type=float,
        default=1.8,
        help="豆包TTS语速倍率（默认 1.8；按官方映射写入 speech_rate）",
    )
    ap.add_argument(
        "--doubao-tts-emotion",
        default="angry",
        help="TTS 情感（如 happy、angry；不填则不带该字段，按音色能力为准）",
    )
    ap.add_argument(
        "--doubao-tts-emotion-scale",
        type=int,
        default=None,
        help="情感强度 1～5（需同时配置情感；不设则用 4）",
    )
    ap.add_argument(
        "--doubao-tts-loudness",
        type=float,
        default=1.15,
        help="TTS 响度比例 loudness_ratio（默认 1.15；1.0 为平常）",
    )
    ap.add_argument(
        "--doubao-tts-app-id",
        default="",
        help="豆包TTS: APP ID（不填则读环境变量 DOUBAO_TTS_APP_ID）",
    )
    ap.add_argument(
        "--doubao-tts-access-key",
        default="",
        help="豆包TTS: Access Key（不填则读环境变量 DOUBAO_TTS_ACCESS_KEY）",
    )
    ap.add_argument(
        "--doubao-tts-resource-id",
        default="seed-tts-1.0",  # Doubao TTS 1.0 model
        help="豆包TTS: 资源ID（默认 seed-tts-1.0，用于模型1.0音色；若使用模型2.0音色需改为 seed-tts-2.0）",
    )
    ap.add_argument(
        "--doubao-tts-speaker",
        default="",
        help="豆包TTS: 音色 speaker（必填；需与 resource-id 配对：1.0音色用 seed-tts-1.0，2.0音色用 seed-tts-2.0）",
    )
    ap.add_argument(
        "--doubao-tts-uid",
        default="gemini_v2",
        help="豆包TTS: 用户uid（默认 gemini_v2）",
    )
    ap.add_argument(
        "--doubao-tts-audio-format",
        default="pcm",
        choices=("pcm", "mp3", "ogg_opus"),
        help="豆包TTS: 输出音频格式（默认 pcm；会封装为 wav 播放）",
    )
    ap.add_argument(
        "--doubao-tts-sample-rate",
        type=int,
        default=24000,
        help="豆包TTS: 采样率（默认 24000）",
    )
    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()
    io_lock = threading.Lock()

    use_voice = bool(
        sys.stdin.isatty()
        and not args.no_voice
        and not (args.question or "").strip()
    )
    asr_glossary = _merge_asr_glossaries(
        args.asr_glossary.resolve() if args.asr_glossary else None
    )
    voice: Optional[VoiceSession] = None
    if use_voice:
        ws_cl = str(args.doubao_asr_ws_cluster or "").strip()
        if not ws_cl:
            ws_cl = os.getenv("DOUBAO_ASR_WS_CLUSTER", "").strip()
        if not ws_cl:
            ws_cl = "volcengine_streaming_common"
        voice = VoiceSession(
            simplify_zh=not bool(args.no_zh_simplify),
            asr_glossary=asr_glossary,
            doubao_asr_app_key=str(args.doubao_asr_app_key or ""),
            doubao_asr_access_key=str(args.doubao_asr_access_key or ""),
            doubao_asr_resource_id=str(args.doubao_asr_resource_id or ""),
            asr_timeout_sec=float(args.asr_timeout or 120.0),
            doubao_asr_ws_cluster=ws_cl,
            asr_chunk_ms=int(args.asr_chunk_ms or 200),
            asr_stream=not bool(args.asr_no_stream),
            asr_workflow=str(args.asr_workflow or ""),
            asr_verbose=bool(args.asr_verbose),
        )
        voice.warmup()

    bundle_for_coach: Optional[Dict[str, Any]] = None
    tts = TTSPlayer(args)

    def read_q_voice_or_text(hold_io_lock: bool) -> str:
        q0 = (args.question or "").strip()
        if q0:
            return q0
        if not sys.stdin.isatty():
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

    question = (args.question or "").strip()
    summary_path: Optional[Path] = None

    if args.summary_json:
        summary_path = args.summary_json.resolve()
        if not summary_path.is_file():
            raise SystemExit(f"找不到文件: {summary_path}")
        if not question:
            if use_voice and voice is not None and sys.stdin.isatty():
                question, bundle_for_coach = _voice_with_parallel_coach_bundle(
                    args,
                    summary_path,
                    read_question=lambda: read_q_voice_or_text(hold_io_lock=False),
                )
            else:
                question = read_q_voice_or_text(hold_io_lock=False)
        if not question:
            raise SystemExit("未提供问题：请使用 --question / -q，或语音/键盘输入。")
    else:
        img_dir = args.img_dir.resolve() if args.img_dir else DEFAULT_IMG_DIR
        if not img_dir.is_dir():
            raise SystemExit(f"截图目录不存在: {img_dir}")
        if not gv._dir_has_screenshot(img_dir):
            raise SystemExit(
                f"目录内无截图: {img_dir}\n"
                f"请放入主图 -a / 辅图 -b，或使用 --summary-json 指定已有 *_summary.json"
            )

        cache_dir = PIPELINE_CACHE_DIR.resolve()
        use_cache = (
            not bool(args.force_pipeline)
            and cache_dir.is_dir()
            and any(cache_dir.glob("*_summary.json"))
        )

        if use_cache:
            summary_path = gv._find_first_summary_json(cache_dir)
            print(f"[gemini_v2] 使用缓存: {cache_dir}（跳过 pipeline）", flush=True)
            if not question:
                if sys.stdin.isatty():
                    print(
                        "-" * 60 + "\n【提示】已使用本地缓存的识别结果，跳过 pipeline。\n" + "-" * 60,
                        flush=True,
                    )
                if use_voice and voice is not None and sys.stdin.isatty():
                    question, bundle_for_coach = _voice_with_parallel_coach_bundle(
                        args,
                        summary_path,
                        read_question=lambda: read_q_voice_or_text(hold_io_lock=False),
                    )
                else:
                    question = read_q_voice_or_text(hold_io_lock=False)
            if not question:
                raise SystemExit("未提供问题：请使用 --question / -q，或语音/键盘输入。")
        else:
            out_dir = args.pipeline_out.resolve()
            pipe_err: List[Optional[Exception]] = [None]
            pipe_data: Dict[str, Any] = {"cap": None, "wall": 0.0}
            done = threading.Event()
            quiet = not args.pipeline_verbose

            def _pipeline_worker() -> None:
                t0 = time.perf_counter()
                try:
                    pipe_data["cap"] = run_pipeline(img_dir, out_dir, quiet=quiet)
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

            if sys.stdout.isatty():
                print(
                    "-" * 60
                    + "\n【提示】对局识别已在后台运行；语音：空格开始录（无实时转写），回车结束并识别。\n"
                    + "-" * 60,
                    flush=True,
                )

            if not question:
                if sys.stdin.isatty():
                    question = read_q_voice_or_text(hold_io_lock=True)
                else:
                    question = sys.stdin.read().strip()
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
                print("\n[gemini_v2] 对局分析耗时（摘自子进程捕获日志）")
                for ln in gv._pipeline_timing_report_lines(
                    merged, float(pipe_data["wall"])
                ):
                    print(ln)
            summary_path = gv._find_first_summary_json(out_dir)

    assert summary_path is not None
    follow_up = None
    if use_voice and voice is not None and sys.stdin.isatty():

        def follow_up():
            print(
                "\n—— 追问：【空格】开始录（无实时转写），【回车】结束并识别；"
                "若随风听笛已答完你想结束，回车空提交或说 q / quit。",
                flush=True,
            )
            return voice.run("【语音追问】")

    run_coach_v2_after_summary(
        args,
        summary_path,
        question,
        io_lock,
        follow_up_reader=follow_up,
        log_prefix="gemini_v2",
        coach_bundle=bundle_for_coach,
        skip_bundle_preview_print=bundle_for_coach is not None,
        on_answer=tts.speak,
        max_history_turns=getattr(args, "max_history_turns", 3),
    )


if __name__ == "__main__":
    main()
