# -*- coding: utf-8 -*-
"""
端到端麦克风 ASR 自动化自检（与 gemini_v3 的 transcribe_pcm_s16le 一致）。

流程（默认）：
1) 可选：先跑 gemini_v3.py --e2e-probe-text，验证文本链路（与 ASR 无关但可快速排鉴权）。
2) 固定秒数录音（16kHz / mono / int16，与 E2EVoiceSession 一致）。
3) 同一段 PCM 依次测试（与 gemini_v3 默认一致：未设 FINISH 时先发 102）：
   - A：FINISH_AFTER=1（默认行为）
   - B：FINISH_AFTER=0（延后 102，对照）

依赖：与 gemini_v3 语音相同 — sounddevice、numpy、websocket-client、requests。

用法（在项目根目录，已配置 .env 或环境变量）：
  python e2e_mic_asr_autotest.py
  python e2e_mic_asr_autotest.py --seconds 8 --mic-device 1
  python e2e_mic_asr_autotest.py --skip-probe --full-matrix
"""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent

# 与 gemini_v3.E2EVoiceSession 一致
SR = 16000


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(ROOT / ".env")
    except Exception:
        pass


def _resolve_mic_device(sd: Any, explicit: Optional[int]) -> Optional[int]:
    if explicit is not None:
        try:
            info = sd.query_devices(explicit)
            if (info.get("max_input_channels") or 0) > 0:
                return int(explicit)
        except Exception:
            pass
        return None
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
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if (dev.get("max_input_channels") or 0) > 0:
                return idx
    except Exception:
        pass
    return None


def record_pcm_s16le(
    seconds: float,
    mic_device: Optional[int],
) -> Tuple[bytes, int]:
    import numpy as np
    import sounddevice as sd

    if seconds <= 0:
        raise SystemExit("--seconds 必须 > 0")

    dev = _resolve_mic_device(sd, mic_device)
    if dev is None:
        raise SystemExit(
            "[e2e_mic_asr_autotest] 未找到可用麦克风；请检查设备或传 --mic-device N"
        )

    n = int(seconds * SR)
    print(
        f"[e2e_mic_asr_autotest] 使用输入设备 index={dev}，将录 {seconds:g}s（{n} 帧）…",
        flush=True,
    )
    for i in range(3, 0, -1):
        print(f"  {i}…", flush=True)
        time.sleep(1.0)
    print("[e2e_mic_asr_autotest] 录音中，请对着麦克风说话…", flush=True)
    buf = sd.rec(n, samplerate=SR, channels=1, dtype="int16", device=dev)
    sd.wait()
    pcm = buf.copy().tobytes()
    print(f"[e2e_mic_asr_autotest] 录音结束，共 {len(pcm)} 字节 PCM。", flush=True)
    return pcm, dev


def _env_snapshot(keys: Tuple[str, ...]) -> Dict[str, Optional[str]]:
    return {k: os.environ.get(k) for k in keys}


def _env_restore(snapshot: Dict[str, Optional[str]]) -> None:
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def analyze_transcribe_log(log: str) -> Dict[str, Any]:
    """从 transcribe 打印的日志中粗判是否出现 ASR 相关事件。"""
    # debug 行形如: [gemini_v3][e2e][debug] event=451 mt=...
    has_451 = bool(re.search(r"\[e2e\]\[debug\].*event=451\b", log))
    has_550 = bool(re.search(r"\[e2e\]\[debug\].*event=550\b", log))
    # 兜底：未开 debug 时整段里出现 event=451 字样
    if not has_451:
        has_451 = "event=451" in log or "event='451'" in log
    if not has_550:
        has_550 = "event=550" in log or "event='550'" in log
    content_hints = "content" in log and ("550" in log or "451" in log)
    return {
        "has_451": has_451,
        "has_550": has_550,
        "content_keyword": content_hints,
    }


def run_probe(
    args: argparse.Namespace,
    env: Dict[str, str],
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(ROOT / "gemini_v3.py"),
        "--e2e-probe-text",
        (args.probe_text or "你好").strip(),
    ]
    if getattr(args, "e2e_probe_no_hello", False):
        cmd.append("--e2e-probe-no-hello")
    for attr, flag in (
        ("doubao_e2e_app_key", "--doubao-e2e-app-key"),
        ("doubao_e2e_access_key", "--doubao-e2e-access-key"),
        ("doubao_e2e_app_id", "--doubao-e2e-app-id"),
        ("doubao_e2e_resource_id", "--doubao-e2e-resource-id"),
        ("doubao_e2e_access_key_jwt", "--doubao-e2e-access-key-jwt"),
        ("doubao_e2e_sts_url", "--doubao-e2e-sts-url"),
    ):
        v = (getattr(args, attr, "") or "").strip()
        if v:
            cmd.extend([flag, v])
    to = getattr(args, "doubao_e2e_timeout", None)
    if to is not None:
        cmd.extend(["--doubao-e2e-timeout", str(to)])
    sp = (getattr(args, "doubao_tts_speaker", "") or "").strip()
    if sp:
        cmd.extend(["--doubao-tts-speaker", sp])

    print("\n" + "=" * 60, flush=True)
    print("[e2e_mic_asr_autotest] 步骤 1/2：文本 probe（可选，验证鉴权/下行）", flush=True)
    print("  " + " ".join(cmd), flush=True)
    print("=" * 60 + "\n", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    return int(r.returncode)


def build_client(args: argparse.Namespace):
    from gemini_v3 import DEFAULT_E2E_STS_URL, DEFAULT_DIALOGUE_RESOURCE_ID, E2EAsrClient

    app_key = (args.doubao_e2e_app_key or os.getenv("DOUBAO_E2E_APP_KEY", "")).strip()
    access_key = (args.doubao_e2e_access_key or os.getenv("DOUBAO_E2E_ACCESS_KEY", "")).strip()
    jwt_raw = (args.doubao_e2e_access_key_jwt or os.getenv("DOUBAO_E2E_API_ACCESS_KEY", "")).strip()
    use_direct_jwt = bool(jwt_raw)
    if use_direct_jwt:
        access_key = jwt_raw
    resource_id = (
        (args.doubao_e2e_resource_id or "").strip()
        or os.getenv("DOUBAO_E2E_RESOURCE_ID", "")
        or DEFAULT_DIALOGUE_RESOURCE_ID
    )
    app_id = (args.doubao_e2e_app_id or os.getenv("DOUBAO_E2E_APP_ID", "")).strip()
    sts_url = (args.doubao_e2e_sts_url or os.getenv("DOUBAO_E2E_STS_URL", "") or "").strip() or DEFAULT_E2E_STS_URL
    timeout_sec = float(args.doubao_e2e_timeout or 60.0)

    if not app_key or not access_key:
        raise SystemExit(
            "缺少 DOUBAO_E2E_APP_KEY / DOUBAO_E2E_ACCESS_KEY "
            "（或 --doubao-e2e-access-key-jwt / DOUBAO_E2E_API_ACCESS_KEY）"
        )

    return E2EAsrClient(
        app_key=str(app_key),
        access_key=str(access_key),
        resource_id=str(resource_id),
        dialogue_app_id=str(app_id),
        timeout_sec=timeout_sec,
        jwt_sts_appid=None if use_direct_jwt else str(app_key),
        jwt_sts_access_key=None if use_direct_jwt else str(access_key),
        sts_url=sts_url,
    )


def main() -> None:
    _try_load_dotenv()

    ap = argparse.ArgumentParser(
        description="自动录音 + 多组 GEMINI_V3_E2E_* 环境变量跑 transcribe_pcm_s16le",
    )
    ap.add_argument("--seconds", type=float, default=5.0, help="录音时长（秒），默认 5")
    ap.add_argument("--mic-device", type=int, default=None, help="sounddevice 输入设备索引")
    ap.add_argument("--timeout", type=float, default=None, help="覆盖 e2e 超时（秒），默认 60")
    ap.add_argument("--skip-probe", action="store_true", help="跳过 gemini_v3 文本 probe")
    ap.add_argument("--probe-text", default="你好", help="--e2e-probe-text 的文案")
    ap.add_argument(
        "--e2e-probe-no-hello",
        dest="e2e_probe_no_hello",
        action="store_true",
        help="probe 时加 --e2e-probe-no-hello",
    )
    ap.add_argument(
        "--full-matrix",
        action="store_true",
        help="跑四组：DEBUG 0/1 × FINISH_AFTER 0/1（同一段 PCM），替代默认的两组（DEBUG=1 × FINISH 0/1）",
    )
    ap.add_argument(
        "--doubao-e2e-app-key",
        default=os.getenv("DOUBAO_E2E_APP_KEY", ""),
    )
    ap.add_argument(
        "--doubao-e2e-access-key",
        default=os.getenv("DOUBAO_E2E_ACCESS_KEY", ""),
    )
    ap.add_argument(
        "--doubao-e2e-access-key-jwt",
        default=os.getenv("DOUBAO_E2E_API_ACCESS_KEY", ""),
    )
    ap.add_argument(
        "--doubao-e2e-app-id",
        default=os.getenv("DOUBAO_E2E_APP_ID", ""),
    )
    ap.add_argument(
        "--doubao-e2e-resource-id",
        default=os.getenv("DOUBAO_E2E_RESOURCE_ID", ""),
    )
    ap.add_argument(
        "--doubao-e2e-sts-url",
        default=os.getenv("DOUBAO_E2E_STS_URL", ""),
    )
    ap.add_argument(
        "--doubao-e2e-timeout",
        type=float,
        default=60.0,
    )
    ap.add_argument(
        "--doubao-tts-speaker",
        default=os.getenv("DOUBAO_TTS_SPEAKER", ""),
        help="传给 transcribe 的 tts speaker（与 gemini_v3 一致）",
    )
    args = ap.parse_args()
    if args.timeout is not None:
        args.doubao_e2e_timeout = float(args.timeout)

    env_keys = ("GEMINI_V3_E2E_DEBUG", "GEMINI_V3_E2E_FINISH_AFTER_AUDIO")
    base_snapshot = _env_snapshot(env_keys)

    child_env = os.environ.copy()

    if not args.skip_probe:
        rc = run_probe(args, child_env)
        if rc != 0:
            print(
                f"[e2e_mic_asr_autotest] probe 退出码 {rc}，仍继续 ASR 测试…",
                flush=True,
            )

    try:
        pcm, dev_used = record_pcm_s16le(args.seconds, args.mic_device)
    except Exception as e:
        raise SystemExit(str(e)) from e

    client = build_client(args)
    speaker = (args.doubao_tts_speaker or "").strip()

    runs: List[Tuple[str, str, Optional[str]]] = [
        ("A: DEBUG=1（并发 recv+send）", "1", None),
    ]
    if args.full_matrix:
        runs = [
            ("1: DEBUG=0", "0", None),
            ("2: DEBUG=1", "1", None),
        ]

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 60, flush=True)
    print("[e2e_mic_asr_autotest] 步骤 2/2：同一段 PCM 多次 transcribe_pcm_s16le", flush=True)
    print("=" * 60 + "\n", flush=True)

    for label, dbg, fin in runs:
        os.environ["GEMINI_V3_E2E_DEBUG"] = dbg
        if fin is None:
            os.environ.pop("GEMINI_V3_E2E_FINISH_AFTER_AUDIO", None)
        else:
            os.environ["GEMINI_V3_E2E_FINISH_AFTER_AUDIO"] = fin

        buf = io.StringIO()
        tee = _Tee(sys.stdout, buf)
        old_out = sys.stdout
        text_out = ""
        err: Optional[BaseException] = None
        try:
            sys.stdout = tee  # type: ignore[assignment]
            text_out = client.transcribe_pcm_s16le(
                pcm_s16le_bytes=pcm,
                session_id=str(uuid.uuid4()),
                dialog_system_role="",
                tts_speaker=speaker,
            )
        except BaseException as e:
            err = e
        finally:
            sys.stdout = old_out
            _env_restore(base_snapshot)

        log = buf.getvalue()
        ana = analyze_transcribe_log(log)
        row = {
            "label": label,
            "recognized": (text_out or "").strip(),
            "error": repr(err) if err else "",
            **ana,
        }
        results.append(row)

        print(
            f"\n--- 小结 {label} ---\n"
            f"  识别文本: {row['recognized']!r}\n"
            f"  日志中 event=451: {row['has_451']}\n"
            f"  日志中 event=550: {row['has_550']}\n"
            f"  日志含 content 线索: {row.get('content_keyword', False)}\n"
            f"  异常: {row['error'] or '无'}\n",
            flush=True,
        )

    print("\n" + "=" * 60, flush=True)
    print("[e2e_mic_asr_autotest] 汇总表", flush=True)
    print(
        f"  设备: index={dev_used}，PCM: {len(pcm)} bytes，{args.seconds:g}s @ {SR}Hz mono int16",
        flush=True,
    )
    for r in results:
        ok = bool(r["recognized"]) or r["has_451"] or r["has_550"]
        flag = "OK" if ok else "无字/无451/550"
        print(
            f"  [{r['label']}] {flag} | 451={r['has_451']} 550={r['has_550']} | 文本={r['recognized'][:80]!r}"
            + ("…" if len(r["recognized"]) > 80 else ""),
            flush=True,
        )
    print("=" * 60, flush=True)
    print(
        "说明：若 probe 正常而 ASR 仍无 451/550，请检查音量、环境安静度、"
        "或尝试延长 --seconds；必要时对比 FINISH_AFTER 两种结果。",
        flush=True,
    )


if __name__ == "__main__":
    main()
