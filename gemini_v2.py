# -*- coding: utf-8 -*-
"""
gemini_v1 同款流水线与教练对话，交互问题改为：**按一次空格** 开始录音 → 录音过程中 ASR
周期解码实时刷新本行 → **回车** 结束录音并提交（单段问题）。

说明：所谓「实时」是每隔 --asr-interval 秒对当前累积音频整段重解码，不是云端流式 ASR。
首次运行会从 Hugging Face 拉权重；可设置环境变量 HF_TOKEN 提高限额（见 HF 文档）。

识别准确率：Whisper / SenseVoice 都可能把金铲铲黑话（如「D 牌」）或英雄简称识别错。
可尝试：① whisper 用更大模型 --whisper-model small|medium；② whisper 调 --whisper-beam-size 3～5；③ 编辑 data/gemini_v2_asr_glossary.json 做误识别→正写替换（长短语优先匹配）。
 
 
依赖（需额外安装）：
  pip install sounddevice numpy pynput faster-whisper
  # 若使用 SenseVoice 后端（--asr-backend sensevoice）
  pip install funasr
  # 可选：ASR 繁体 → 简体（推荐）
  pip install zhconv

CPU 可用；有 NVIDIA CUDA 时可加 --whisper-device cuda（对 whisper/sensevoice 都会透传 device 字段）。

并行说明（v2）：在「读 summary_json / 用缓存跳过 pipeline」时，会在你**录音同时**后台构建快报+RAG，
说完按回车后再请求 LLM，从而把 RAG 耗时叠进说话时间里（pipeline 分支仍须等识别完才有 summary，无法与 RAG 并行）。
仅发「战报+RAG」而不带问题的**额外一次** LLM 请求在没有「上下文缓存」API 时一般**更慢**，故未实现。

示例：
  python gemini_v2.py
  python gemini_v2.py --no-voice              # 与 v1 相同，键盘输入
  python gemini_v2.py --whisper-model tiny              # whisper：更快、略不准
  python gemini_v2.py --asr-backend sensevoice          # 切到 SenseVoice
  python gemini_v2.py -q "文字问题"          # 跳过语音
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import mimetypes
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gemini_v1 as gv
from gemini_v1 import (
    DEFAULT_IMG_DIR,
    PIPELINE_CACHE_DIR,
    build_coach_argparser,
    build_coach_bundle,
    print_coach_bundle_preview,
    run_coach_after_summary,
    run_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ASR_GLOSSARY_PATH = REPO_ROOT / "data" / "gemini_v2_asr_glossary.json"
DEFAULT_TTS_OUT_DIR = REPO_ROOT / "runs" / "tts"
DEFAULT_TTS_REF_AUDIO = REPO_ROOT / "语音包.m4a"
DEFAULT_TTS_PROMPT_TEXT = (
    "哈基星你这个蠢货，又他妈存50块钱买棺材板？赶紧全D了找 2 星瑞兹，"
    "找不到你赶紧卸载金铲铲回去玩你那泳装蓝梦吧。"
)


def _is_tcp_port_open(host: str, port: int, timeout_sec: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _ensure_tts_service_started(args: argparse.Namespace) -> None:
    if str(getattr(args, "tts_backend", "none") or "none").strip().lower() != "gpt-sovits":
        return
    if not bool(getattr(args, "tts_autostart", False)):
        return

    api_url = str(getattr(args, "tts_api_url", "http://127.0.0.1:9880/tts") or "").strip()
    parsed = urllib.parse.urlparse(api_url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    wait_sec = max(1.0, float(getattr(args, "tts_autostart_wait", 25.0) or 25.0))
    start_cmd = str(getattr(args, "tts_start_cmd", "") or "").strip()
    start_dir_raw = str(getattr(args, "tts_start_dir", "") or "").strip()
    start_dir = Path(start_dir_raw).resolve() if start_dir_raw else REPO_ROOT

    if _is_tcp_port_open(host, port, timeout_sec=0.5):
        print(f"[gemini_v2][TTS] 检测到服务已在运行：{host}:{port}", flush=True)
        return
    if not start_cmd:
        raise SystemExit(
            f"TTS 自动启动已开启，但未提供 --tts-start-cmd，且 {host}:{port} 未监听。"
        )

    print(f"[gemini_v2][TTS] 未检测到服务，准备自动启动：{start_cmd}", flush=True)
    subprocess.Popen(
        start_cmd,
        cwd=str(start_dir),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < wait_sec:
        if _is_tcp_port_open(host, port, timeout_sec=0.5):
            print(f"[gemini_v2][TTS] 服务已就绪：{host}:{port}", flush=True)
            return
        time.sleep(0.4)
    raise SystemExit(
        f"TTS 自动启动后仍未就绪（{host}:{port}）。请检查启动命令或端口。"
    )


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


def _voice_with_parallel_coach_bundle(
    args: argparse.Namespace,
    summary_path: Path,
    read_question: Callable[[], str],
) -> tuple[str, Dict[str, Any]]:
    """
    录音同时后台 build_coach_bundle；适合已有 summary_path 的分支（--summary-json / 缓存跳过 pipeline）。
    返回 (用户问题, bundle)；打印预览后再由 run_coach_after_summary(..., coach_bundle=) 走 LLM。
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
    """按一次空格开始录音；录音中按间隔刷新识别预览；回车结束本段并提交。"""

    SR = 16000

    def __init__(
        self,
        *,
        asr_backend: str = "sensevoice",
        model_size: str = "base",
        device: str = "cpu",
        asr_interval: float = 0.45,
        simplify_zh: bool = True,
        beam_size: int = 1,
        asr_glossary: Optional[Dict[str, str]] = None,
    ) -> None:
        self.asr_backend = (asr_backend or "whisper").strip().lower()
        if self.asr_backend not in {"whisper", "sensevoice"}:
            raise SystemExit(
                f"不支持的 --asr-backend: {self.asr_backend}（仅支持 whisper / sensevoice）"
            )
        self.model_size = model_size
        self.device = device
        self.asr_interval = max(0.2, float(asr_interval))
        self.simplify_zh = bool(simplify_zh)
        self.beam_size = max(1, int(beam_size))
        self.asr_glossary = dict(asr_glossary or {})
        self._model = None
        self._model_init_lock = threading.Lock()
        self._lock = threading.Lock()
        self.segments: List[str] = []
        self.live_partial = ""
        self.recording = False
        self._chunks: List[Any] = []
        self._submit = threading.Event()
        self._stream = None
        self._listener = None

    def warmup(self) -> None:
        """在首次语音 UI 前调用，避免下载/加载与终端 \\r 状态行混在一起。"""
        self._get_model()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_init_lock:
            if self._model is not None:
                return self._model
            if self.asr_backend == "whisper":
                try:
                    from faster_whisper import WhisperModel
                except ImportError as e:
                    raise SystemExit(
                        "未安装 faster-whisper。请执行： pip install faster-whisper\n"
                        + str(e)
                    ) from e
                ct = "int8" if self.device == "cpu" else "float16"
                sys.stderr.write(
                    f"[gemini_v2] 加载 Whisper（{self.model_size} / {self.device} / {ct}）…\n"
                )
                sys.stderr.flush()
                t0 = time.perf_counter()
                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=ct
                )
                sys.stderr.write(
                    f"[gemini_v2] Whisper 就绪，耗时 {time.perf_counter() - t0:.1f}s\n"
                )
                sys.stderr.flush()
            else:
                try:
                    from funasr import AutoModel  # type: ignore[reportMissingImports]
                except ImportError as e:
                    raise SystemExit(
                        "未安装 funasr。若要启用 SenseVoice，请执行： pip install funasr\n"
                        + str(e)
                    ) from e
                model_name = self.model_size
                if not model_name or model_name.strip().lower() == "base":
                    model_name = "iic/SenseVoiceSmall"
                sys.stderr.write(
                    f"[gemini_v2] 加载 SenseVoice（{model_name} / {self.device}）…\n"
                )
                sys.stderr.flush()
                t0 = time.perf_counter()
                try:
                    self._model = AutoModel(
                        model=model_name,
                        device=self.device,
                        disable_update=True,
                    )
                except Exception as e:
                    raise SystemExit(
                        "SenseVoice 初始化失败。可尝试：\n"
                        "1) pip install funasr\n"
                        "2) --whisper-device cpu\n"
                        "3) --whisper-model iic/SenseVoiceSmall\n"
                        f"原始错误: {e}"
                    ) from e
                sys.stderr.write(
                    f"[gemini_v2] SenseVoice 就绪，耗时 {time.perf_counter() - t0:.1f}s\n"
                )
                sys.stderr.flush()
        return self._model

    def _asr_postprocess(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
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
        import numpy as np

        if audio is None or audio.size < int(self.SR * 0.22):
            return ""
        wav = np.asarray(audio, dtype=np.float32).ravel()
        model = self._get_model()
        if self.asr_backend == "whisper":
            segs, _info = model.transcribe(
                wav,
                language="zh",
                beam_size=self.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return self._asr_postprocess("".join(s.text for s in segs))

        # SenseVoice（funasr）接口返回格式可能因版本有差异，这里做宽松兼容解析。
        # funasr 会向 stdout 打印 rtf_avg 进度，这里静默掉，避免刷屏干扰录音 UI。
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            try:
                out = model.generate(input=wav, language="zh", use_itn=True)
            except TypeError:
                out = model.generate(input=wav)
        text = ""
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                text = str(first.get("text") or "")
            else:
                text = str(first)
        elif isinstance(out, dict):
            text = str(out.get("text") or "")
        else:
            text = str(out or "")
        return self._asr_postprocess(text)

    def _live_worker(self) -> None:
        import numpy as np

        while not self._submit.is_set():
            time.sleep(0.02)
            if not self.recording:
                continue
            time.sleep(self.asr_interval)
            if not self.recording or self._submit.is_set():
                continue
            with self._lock:
                if not self._chunks:
                    continue
                audio = np.concatenate([x.copy() for x in self._chunks])
            text = self._transcribe(audio)
            with self._lock:
                if self.recording:
                    self.live_partial = text

    def run(self, banner: str) -> str:
        try:
            import numpy as np
            import sounddevice as sd
            from pynput import keyboard
        except ImportError as e:
            raise SystemExit(
                "语音依赖未就绪。请执行： pip install sounddevice numpy pynput faster-whisper\n"
                + str(e)
            ) from e

        self._submit.clear()
        self.segments = []
        self.live_partial = ""
        self.recording = False
        with self._lock:
            self._chunks.clear()

        print(banner, flush=True)
        print(
            "操作：按一次【空格】开始录音，本行会定时刷新识别文字；"
            "【回车】结束录音并提交（未按空格直接回车视为空问题）。Ctrl+C 退出。\n",
            flush=True,
        )

        def audio_cb(indata, frames, tcb, status):
            if self.recording:
                with self._lock:
                    self._chunks.append(indata.copy().ravel())

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
                    self._chunks.clear()
                    self.live_partial = ""
            elif key == keyboard.Key.enter:
                with self._lock:
                    was_rec = self.recording
                    self.recording = False
                    if self._chunks:
                        hold = [x.copy() for x in self._chunks]
                    else:
                        hold = []
                    self._chunks.clear()
                    self.live_partial = ""
                text_final = ""
                if was_rec and hold:
                    text_final = self._transcribe(np.concatenate(hold))
                with self._lock:
                    self.segments.clear()
                    if text_final:
                        self.segments.append(text_final)
                self._submit.set()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        th = threading.Thread(target=self._live_worker, daemon=True)
        th.start()

        try:
            while not self._submit.wait(0.05):
                with self._lock:
                    rec = self.recording
                    live = self.live_partial if rec else ""
                if rec:
                    vis = f"【录音中】{live}" if live else "【录音中】…"
                else:
                    vis = "…按【空格】开始录音，【回车】结束并提交…"
                if len(vis) > 100:
                    vis = vis[:97] + "..."
                sys.stdout.write("\r" + vis.ljust(108))
                sys.stdout.flush()
        except KeyboardInterrupt:
            self._submit.set()
        finally:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._stream.stop()
            self._stream.close()
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

        out = "".join(self.segments).strip()
        if out:
            print(out, flush=True)
        return out


class TTSPlayer:
    """将 LLM 文本调用 GPT-SoVITS 合成，并可选本地播放。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.backend = str(
            getattr(args, "tts_backend", "gpt-sovits") or "gpt-sovits"
        ).strip().lower()
        self.api_url = str(
            getattr(args, "tts_api_url", "http://127.0.0.1:9880/tts")
            or "http://127.0.0.1:9880/tts"
        ).strip()
        self.timeout_sec = max(3.0, float(getattr(args, "tts_timeout", 20.0) or 20.0))
        self.play = bool(getattr(args, "tts_play", True))
        self.save = bool(getattr(args, "tts_save", False))
        self.out_dir = Path(getattr(args, "tts_out_dir", DEFAULT_TTS_OUT_DIR)).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.voice = str(getattr(args, "tts_voice", "") or "").strip()
        ref_audio_cli = str(getattr(args, "tts_ref_audio", "") or "").strip()
        prompt_text_cli = str(getattr(args, "tts_prompt_text", "") or "").strip()
        if ref_audio_cli:
            self.ref_audio = ref_audio_cli
        elif DEFAULT_TTS_REF_AUDIO.is_file():
            self.ref_audio = str(DEFAULT_TTS_REF_AUDIO)
        else:
            self.ref_audio = ""
        self.prompt_text = prompt_text_cli or DEFAULT_TTS_PROMPT_TEXT
        self.prompt_lang = str(getattr(args, "tts_prompt_lang", "zh") or "zh").strip()
        self.text_lang = str(getattr(args, "tts_text_lang", "zh") or "zh").strip()
        self.speed = float(getattr(args, "tts_speed", 1.0) or 1.0)

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

    def _call_gpt_sovits(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            import requests
        except ImportError as e:
            raise SystemExit(
                "未安装 requests。请执行： pip install requests"
            ) from e

        payload: Dict[str, Any] = {
            "text": text,
            "text_lang": self.text_lang,
            "prompt_lang": self.prompt_lang,
            "speed": self.speed,
        }
        if self.voice:
            payload["voice"] = self.voice
        if self.ref_audio:
            payload["ref_audio_path"] = self.ref_audio
        if self.prompt_text:
            payload["prompt_text"] = self.prompt_text

        r = requests.post(self.api_url, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if ctype.startswith("audio/"):
            ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ".wav"
            return {"raw": r.content, "ext": ext}
        try:
            obj = r.json()
        except Exception:
            if r.content:
                return {"raw": r.content, "ext": ".wav"}
            return None
        if isinstance(obj, dict):
            return self._extract_audio_payload_from_json(obj)
        return None

    def _play(self, payload: Dict[str, Any]) -> None:
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
        print("[gemini_v2][TTS] 自动播放仅支持 Windows 下 wav（内存或本地路径）。")

    def speak(self, text: str, turn: int) -> None:
        if self.backend != "gpt-sovits":
            return
        t = (text or "").strip()
        if not t:
            return
        payload = self._call_gpt_sovits(t)
        if payload is None:
            print("[gemini_v2][TTS] GPT-SoVITS 未返回可用音频。")
            return
        if self.play:
            self._play(payload)
            print(f"[gemini_v2][TTS] 第 {turn} 轮语音已播放。")
        if self.save:
            raw = payload.get("raw")
            p = payload.get("path")
            ext = str(payload.get("ext") or ".wav")
            if raw:
                sp = self._write_audio_bytes(raw, turn, ext)
                print(f"[gemini_v2][TTS] 第 {turn} 轮语音已保存: {sp}")
            elif isinstance(p, Path):
                print(f"[gemini_v2][TTS] 第 {turn} 轮语音路径: {p}")
        if not self.play and not self.save:
            print("[gemini_v2][TTS] 已合成语音（未播放、未保存）。")


def _build_argparser() -> argparse.ArgumentParser:
    ap = build_coach_argparser()
    ap.description = (ap.description or "") + "（v2：语音空格录制 + 回车提交）"
    ap.add_argument(
        "--no-voice",
        action="store_true",
        help="不用语音，终端键盘输入（与 gemini_v1 一致）",
    )
    ap.add_argument(
        "--asr-backend",
        default="sensevoice",
        choices=("whisper", "sensevoice"),
        help="语音识别后端：whisper / sensevoice（默认 sensevoice）",
    )
    ap.add_argument(
        "--whisper-model",
        default="",
        help="ASR 模型名。whisper 默认 base；sensevoice 默认 iic/SenseVoiceSmall",
    )
    ap.add_argument(
        "--whisper-device",
        default="cpu",
        help="Whisper 推理设备：cpu / cuda（默认 cpu）",
    )
    ap.add_argument(
        "--asr-interval",
        type=float,
        default=0.45,
        help="录音中刷新实时识别的间隔秒数（默认 0.45；越小越跟手但 CPU 更忙，建议≥0.2）",
    )
    ap.add_argument(
        "--no-zh-simplify",
        action="store_true",
        help="不对 ASR 结果做繁转简（默认安装 zhconv 时转为大陆简体）",
    )
    ap.add_argument(
        "--whisper-beam-size",
        type=int,
        default=1,
        help="Whisper beam search，1 最快；3～5 略准但更慢（默认 1）",
    )
    ap.add_argument(
        "--asr-glossary",
        type=Path,
        default=None,
        help="可选 JSON：{\"常误听\":\"正写\"}，会与 data/gemini_v2_asr_glossary.json 合并（键越长越先替换）",
    )
    ap.add_argument(
        "--tts-backend",
        default="gpt-sovits",
        choices=("none", "gpt-sovits"),
        help="TTS 后端：none / gpt-sovits（默认 gpt-sovits）",
    )
    ap.add_argument(
        "--tts-api-url",
        default="http://127.0.0.1:9880/tts",
        help="GPT-SoVITS HTTP 接口地址",
    )
    ap.add_argument(
        "--tts-autostart",
        action="store_true",
        help="运行 gemini_v2 时自动检测并尝试拉起 GPT-SoVITS 服务",
    )
    ap.add_argument(
        "--tts-start-cmd",
        default="",
        help="自动拉起 GPT-SoVITS 的命令（建议填完整命令）",
    )
    ap.add_argument(
        "--tts-start-dir",
        default="",
        help="执行 --tts-start-cmd 的工作目录（默认项目根目录）",
    )
    ap.add_argument(
        "--tts-autostart-wait",
        type=float,
        default=25.0,
        help="自动启动后等待服务就绪的秒数（默认 25）",
    )
    ap.add_argument(
        "--tts-timeout",
        type=float,
        default=20.0,
        help="TTS 接口超时秒数（默认 20）",
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
        "--tts-voice",
        default="",
        help="可选：传给 GPT-SoVITS 的 voice 字段",
    )
    ap.add_argument(
        "--tts-ref-audio",
        default="",
        help="可选：传给 GPT-SoVITS 的 ref_audio_path；留空时自动尝试项目根目录的 语音包.m4a",
    )
    ap.add_argument(
        "--tts-prompt-text",
        default="",
        help="可选：传给 GPT-SoVITS 的 prompt_text；留空时使用内置默认参考文本",
    )
    ap.add_argument(
        "--tts-prompt-lang",
        default="zh",
        help="可选：传给 GPT-SoVITS 的 prompt_lang（默认 zh）",
    )
    ap.add_argument(
        "--tts-text-lang",
        default="zh",
        help="可选：传给 GPT-SoVITS 的 text_lang（默认 zh）",
    )
    ap.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="可选：传给 GPT-SoVITS 的 speed（默认 1.0）",
    )
    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()
    io_lock = threading.Lock()
    _ensure_tts_service_started(args)

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
        voice = VoiceSession(
            asr_backend=str(args.asr_backend).strip(),
            model_size=str(args.whisper_model).strip(),
            device=str(args.whisper_device).strip(),
            asr_interval=float(args.asr_interval),
            simplify_zh=not bool(args.no_zh_simplify),
            beam_size=int(args.whisper_beam_size),
            asr_glossary=asr_glossary,
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
                    + "\n【提示】对局识别已在后台运行；语音：空格开始录，回车结束并提交。\n"
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
                "\n—— 追问：【空格】开始录，【回车】结束并提交；"
                "若随风听笛已答完你想结束，回车空提交或说 q / quit。",
                flush=True,
            )
            return voice.run("【语音追问】")

    run_coach_after_summary(
        args,
        summary_path,
        question,
        io_lock,
        follow_up_reader=follow_up,
        log_prefix="gemini_v2",
        coach_bundle=bundle_for_coach,
        skip_bundle_preview_print=bundle_for_coach is not None,
        on_answer=tts.speak if tts.backend == "gpt-sovits" else None,
    )


if __name__ == "__main__":
    main()
