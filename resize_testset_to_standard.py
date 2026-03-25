# -*- coding: utf-8 -*-
"""
批量将测试集图片缩放到标准分辨率（默认 2196x1253）。

默认行为：
- 输入目录：项目根目录/测试集/交叉验证组
- 输出目录：项目根目录/测试集/交叉验证组_2196x1253
- 保留原文件名

可选：
- --in-place：原地覆盖输入目录图片
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "测试集" / "交叉验证组"
DEFAULT_OUT = PROJECT_DIR / "测试集" / "交叉验证组_2196x1253"


def _iter_images(d: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _read_image(path: Path) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读取失败: {path}")
    return img


def _write_image(path: Path, img: np.ndarray) -> None:
    ext = path.suffix.lower() if path.suffix else ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    path.write_bytes(buf.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="批量缩放测试集图片到标准分辨率")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"输入目录（默认 {DEFAULT_INPUT}）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"输出目录（默认 {DEFAULT_OUT}）")
    ap.add_argument("--width", type=int, default=2196, help="目标宽度，默认 2196")
    ap.add_argument("--height", type=int, default=1253, help="目标高度，默认 1253")
    ap.add_argument("--in-place", action="store_true", help="原地覆盖输入目录图片")
    args = ap.parse_args()

    in_dir = args.input.resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {in_dir}")

    target_w = int(args.width)
    target_h = int(args.height)
    if target_w <= 0 or target_h <= 0:
        raise SystemExit("宽高必须为正整数")

    out_dir = in_dir if args.in_place else args.out.resolve()
    if not args.in_place:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = list(_iter_images(in_dir))
    if not files:
        raise SystemExit(f"目录内没有图片: {in_dir}")

    print(f"[RESIZE] 输入: {in_dir}")
    print(f"[RESIZE] 输出: {out_dir} {'(in-place)' if args.in_place else ''}")
    print(f"[RESIZE] 目标尺寸: {target_w}x{target_h}")

    ok_cnt = 0
    skip_cnt = 0
    for p in files:
        img = _read_image(p)
        h, w = img.shape[:2]
        dst = out_dir / p.name
        if w == target_w and h == target_h:
            if args.in_place:
                print(f"[SKIP] {p.name}: already {w}x{h}")
                skip_cnt += 1
                continue
            # 非 in-place 模式下也复制一份，保证输出目录完整
            resized = img
        else:
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        _write_image(dst, resized)
        print(f"[OK] {p.name}: {w}x{h} -> {target_w}x{target_h}")
        ok_cnt += 1

    print(f"完成。处理成功: {ok_cnt}，跳过: {skip_cnt}")


if __name__ == "__main__":
    main()

