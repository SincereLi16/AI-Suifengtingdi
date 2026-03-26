# -*- coding: utf-8 -*-
"""
player_onnx：对局截图固定 ROI 的玩家信息识别（ONNX Runtime + 可选 DirectML）。

特点：
- OCR 引擎使用 rapidocr_onnxruntime（PP-OCR ONNX）
- 设备策略支持 auto/dml/cuda/cpu（不可用时自动降级）
- 主流程替换为 one-pass ROI 拼图识别（与 player_onepass_stitch_test 一致）
- 支持批量输出 stitched 图与 summary.json
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import onnxruntime as ort

import player_recog as pr
import player_onepass_stitch_test as p1

_thread_local = threading.local()


def _resolve_device(requested: str) -> Tuple[str, Dict[str, bool]]:
    req = (requested or "auto").strip().lower()
    providers = set(ort.get_available_providers())
    has_dml = "DmlExecutionProvider" in providers
    has_cuda = "CUDAExecutionProvider" in providers

    # 统一为 det/cls/rec 三路开关；未命中即走 CPUExecutionProvider。
    flags = {"det_use_dml": False, "cls_use_dml": False, "rec_use_dml": False}
    cflags = {"det_use_cuda": False, "cls_use_cuda": False, "rec_use_cuda": False}

    if req == "dml":
        if has_dml:
            flags = {k: True for k in flags}
            return "dml", {**flags, **cflags}
        return "cpu", {**flags, **cflags}
    if req == "cuda":
        if has_cuda:
            cflags = {k: True for k in cflags}
            return "cuda", {**flags, **cflags}
        return "cpu", {**flags, **cflags}
    if req == "cpu":
        return "cpu", {**flags, **cflags}

    # auto: dml > cuda > cpu
    if has_dml:
        flags = {k: True for k in flags}
        return "dml", {**flags, **cflags}
    if has_cuda:
        cflags = {k: True for k in cflags}
        return "cuda", {**flags, **cflags}
    return "cpu", {**flags, **cflags}


def _get_thread_ocr(device_kwargs: Dict[str, bool]):
    eng = getattr(_thread_local, "ocr_engine", None)
    if eng is None:
        eng = p1.create_ocr_engine(device_kwargs)
        _thread_local.ocr_engine = eng
    return eng


def _process_one(
    img_path: Path,
    device_kwargs: Dict[str, bool],
    out_dir: Path,
    write_json: bool,
) -> Tuple[str, Optional[str]]:
    ocr_engine = _get_thread_ocr(device_kwargs)
    stem = img_path.stem
    out_png = out_dir / f"{stem}_player_onnx_stitched.png"
    out_json = out_dir / f"{stem}_player_onnx_summary.json"
    p1.run_once(img_path, out_json, out_png, ocr_engine=ocr_engine)
    if write_json:
        return out_png.name, out_json.name
    try:
        out_json.unlink(missing_ok=True)
    except Exception:
        pass
    return out_png.name, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="player_onnx：ONNX Runtime + DML/CUDA/CPU 的玩家信息识别"
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=str(pr.DEFAULT_INPUT),
        help=f"对局截图目录或单张 PNG（默认: {pr.DEFAULT_INPUT}）",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=str(pr.DEFAULT_OUT.with_name("player_info_onnx")),
        help="输出目录",
    )
    ap.add_argument("--json", action="store_true", help="写出每图 summary.json")
    ap.add_argument("--no-clear", action="store_true", help="不清空输出目录（追加写入）")
    ap.add_argument(
        "--device",
        choices=["auto", "dml", "cuda", "cpu"],
        default="auto",
        help="OCR 设备策略（默认 auto：dml>cuda>cpu）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4))),
        help="图片并发数（默认 1~4 自动）",
    )
    ap.add_argument(
        "--allow-dml-multi",
        action="store_true",
        help="允许 DML 模式下 workers>1（实验开关，可能触发驱动/运行时崩溃）",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    if not args.no_clear:
        pr._clear_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    images = pr._iter_pngs(in_path)
    device_name, device_kwargs = _resolve_device(args.device)
    workers = max(1, int(args.workers))
    if device_name == "dml" and workers > 1 and not bool(args.allow_dml_multi):
        print(
            "[player_onnx] 检测到 DML + workers>1，已自动降级为 workers=1 以避免底层崩溃。"
            "若需强制并发，请显式加 --allow-dml-multi。"
        )
        workers = 1
    print(f"[player_onnx] device={device_name}, workers={workers}")

    # one-pass 方案默认关闭 cls 子模型，减少不必要调用。
    device_kwargs["cls_use_dml"] = False
    device_kwargs["cls_use_cuda"] = False

    if workers == 1:
        for img_path in images:
            out_png_name, jname = _process_one(
                img_path, device_kwargs, out_dir, bool(args.json)
            )
            print(f"[OK] {img_path.name} -> {out_png_name}")
            if jname:
                print(f"     json -> {jname}")
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _process_one,
                p,
                device_kwargs,
                out_dir,
                bool(args.json),
            ): p
            for p in images
        }
        for fut in as_completed(futs):
            img_path = futs[fut]
            out_png_name, jname = fut.result()
            print(f"[OK] {img_path.name} -> {out_png_name}")
            if jname:
                print(f"     json -> {jname}")


if __name__ == "__main__":
    main()

