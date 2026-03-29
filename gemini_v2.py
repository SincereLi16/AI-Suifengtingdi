# -*- coding: utf-8 -*-
"""
gemini_v1 同款流水线与教练对话，交互问题改为：**按一次空格** 开始录音 → 录音过程中 **faster-whisper**
周期解码实时刷新本行 → **回车** 结束录音并提交（单段问题）。

说明：所谓「实时」是每隔 --asr-interval 秒对当前累积音频整段重解码，不是云端流式 ASR。
首次运行会从 Hugging Face 拉权重；可设置环境变量 HF_TOKEN 提高限额（见 HF 文档）。

识别准确率：Whisper 对金铲铲黑话（如「D 牌」听成「低」）、英雄简称/英文名混读（瑞兹→Rid）容易错。
可尝试：① larger 模型 --whisper-model small|medium；② --whisper-beam-size 3～5；③ 编辑 data/gemini_v2_asr_glossary.json 做误识别→正写替换（长短语优先匹配）。
 
 
依赖（需额外安装）：
  pip install sounddevice numpy pynput faster-whisper
  # 可选：ASR 繁体 → 简体（推荐）
  pip install zhconv

CPU 可用；有 NVIDIA CUDA 时可加 --whisper-device cuda。

并行说明（v2）：在「读 summary_json / 用缓存跳过 pipeline」时，会在你**录音同时**后台构建快报+RAG，
说完按回车后再请求 LLM，从而把 RAG 耗时叠进说话时间里（pipeline 分支仍须等识别完才有 summary，无法与 RAG 并行）。
仅发「战报+RAG」而不带问题的**额外一次** LLM 请求在没有「上下文缓存」API 时一般**更慢**，故未实现。

示例：
  python gemini_v2.py
  python gemini_v2.py --no-voice              # 与 v1 相同，键盘输入
  python gemini_v2.py --whisper-model tiny    # 更快、略不准
  python gemini_v2.py -q "文字问题"          # 跳过语音
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
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
        model_size: str = "base",
        device: str = "cpu",
        asr_interval: float = 0.45,
        simplify_zh: bool = True,
        beam_size: int = 1,
        asr_glossary: Optional[Dict[str, str]] = None,
    ) -> None:
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
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise SystemExit(
                    "未安装 faster-whisper。请执行： pip install faster-whisper\n" + str(e)
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
        segs, _info = model.transcribe(
            wav,
            language="zh",
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return self._asr_postprocess("".join(s.text for s in segs))

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


def _build_argparser() -> argparse.ArgumentParser:
    ap = build_coach_argparser()
    ap.description = (ap.description or "") + "（v2：语音空格录制 + 回车提交）"
    ap.add_argument(
        "--no-voice",
        action="store_true",
        help="不用语音，终端键盘输入（与 gemini_v1 一致）",
    )
    ap.add_argument(
        "--whisper-model",
        default="base",
        help="faster-whisper 模型：tiny base small medium large-v3 等（默认 base）",
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
        voice = VoiceSession(
            model_size=str(args.whisper_model).strip(),
            device=str(args.whisper_device).strip(),
            asr_interval=float(args.asr_interval),
            simplify_zh=not bool(args.no_zh_simplify),
            beam_size=int(args.whisper_beam_size),
            asr_glossary=asr_glossary,
        )
        voice.warmup()

    bundle_for_coach: Optional[Dict[str, Any]] = None

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
    )


if __name__ == "__main__":
    main()
