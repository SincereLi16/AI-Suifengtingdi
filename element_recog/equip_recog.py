# -*- coding: utf-8 -*-
"""
equip_recog：合并原 crop_below_healthbars + match_below_roi_to_gallery。
批量读取「对局截图」下全部图片，在每条血条下方 ROI 内与 equip_gallery 做多尺度模板匹配，
输出带装备标注的整图至 ``equip_recog/``。运行开始时清空 ``equip_recog/``。
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .bars_recog import (
        iter_input_images,
        parse_roi_string,
        resolve_bars_for_one_image,
    )
    from .chess_recog import (
        DEFAULT_BATCH_EMBED_OUT_DIRNAME,
        ROI as DEFAULT_ROI,
        _draw_chinese_text,
        _load_image,
    )
except ImportError:
    from bars_recog import (
        iter_input_images,
        parse_roi_string,
        resolve_bars_for_one_image,
    )
    from chess_recog import (
        DEFAULT_BATCH_EMBED_OUT_DIRNAME,
        ROI as DEFAULT_ROI,
        _draw_chinese_text,
        _load_image,
    )

from project_paths import DEFAULT_OUT_EQUIP_RECOG, PROJECT_ROOT

_PROJECT_ROOT = PROJECT_ROOT
_ELEMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = _ELEMENT_DIR  # 兼容旧名：element_recog 包目录
DEFAULT_INPUT = _PROJECT_ROOT / "对局截图"
DEFAULT_GALLERY = _PROJECT_ROOT / "equip_gallery"
DEFAULT_OUT = DEFAULT_OUT_EQUIP_RECOG

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _clear_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def _save_bgr(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"imencode 失败: {path}")
    path.write_bytes(buf.tobytes())


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _iter_images(folder: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def _iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def _nms_xywh(
    boxes: Sequence[Tuple[int, int, int, int]],
    scores: Sequence[float],
    iou_thresh: float,
) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou_xywh(boxes[i], boxes[j]) < iou_thresh]
    return keep


def _build_templates(gallery_dir: Path) -> List[Tuple[str, np.ndarray]]:
    templates: List[Tuple[str, np.ndarray]] = []
    for p in _iter_images(gallery_dir):
        bgr = _load_image(p)
        g = _to_gray(bgr)
        if g.shape[0] < 2 or g.shape[1] < 2:
            continue
        templates.append((p.name, g))
    return templates


def _find_local_maxima(
    res: np.ndarray, threshold: float
) -> List[Tuple[float, int, int]]:
    h, w = res.shape
    if h < 2 or w < 2:
        return []
    padded = np.pad(res, 1, mode="edge")
    mask = np.ones((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            mask &= res > padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    mask &= res >= threshold
    ys, xs = np.where(mask)
    peaks = [(float(res[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]
    peaks.sort(key=lambda t: t[0], reverse=True)
    return peaks


def _collect_peaks_for_template(
    roi_g: np.ndarray,
    tmpl_gray: np.ndarray,
    scales: Sequence[int],
    method: int,
    threshold: float,
    max_peaks_per_scale: int,
) -> List[Tuple[float, int, int, int]]:
    h, w = roi_g.shape[:2]
    out: List[Tuple[float, int, int, int]] = []
    for s in scales:
        if s < 2 or h < s or w < s:
            continue
        t = cv2.resize(tmpl_gray, (s, s), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(roi_g, t, method)
        peaks = _find_local_maxima(res, threshold)
        for score, x, y in peaks[:max_peaks_per_scale]:
            out.append((score, x, y, s))
    return out


def _equip_label_stem(template_filename: str) -> str:
    return Path(template_filename).stem


def _apply_blue_buff_special_case(
    picked: List[Tuple[float, int, int, str, int]],
    *,
    gap_min: float,
) -> List[Tuple[float, int, int, str, int]]:
    """
    蓝霸符专项规则（特例）：
    - 背景：13.png 出现“无装备区域误检蓝霸符（score≈0.81）”；
    - 目标：不改全局阈值，尽量不影响其它装备/其它蓝霸符真阳性。
    规则仅在 top1=蓝霸符 时触发：
      1) 若无 top2，直接判为不可靠并过滤本 bar 的所有候选；
      2) 若有 top2 但 top1-top2 分差 < gap_min，也过滤本 bar 的所有候选。
    """
    if not picked:
        return picked
    top1 = picked[0]
    if Path(top1[3]).stem != "蓝霸符":
        return picked
    if len(picked) < 2:
        return []
    top2 = picked[1]
    gap = float(top1[0]) - float(top2[0])
    if gap < float(gap_min):
        return []
    return picked


def _apply_spatula_special_case(
    picked: List[Tuple[float, int, int, str, int]],
) -> List[Tuple[float, int, int, str, int]]:
    """
    金铲铲机制（特例）：
    如果本 bar 的装备区识别出了“秘法手套”，则该棋子的其它装备都是由秘法手套变换来的。
    因此本条装备区只输出最高分的“秘法手套”，过滤掉其它装备候选。
    """
    if not picked:
        return picked
    for cand in picked:
        # cand[3] 是 template 文件名（例如“秘法手套.png”），取 stem 做匹配
        if Path(cand[3]).stem == "秘法手套":
            return [cand]
    return picked


def crop_below_bar_center(
    scene: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    crop_w: int,
    crop_h: int,
    below_px: int = 2,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x, y, w, h = map(int, box)
    H, W = scene.shape[:2]
    cx = x + w // 2
    top = y + h + int(below_px)
    left = cx - crop_w // 2
    left = max(0, min(left, W - 1))
    top = max(0, min(top, H - 1))
    right = min(W, left + crop_w)
    bottom = min(H, top + crop_h)
    actual_w = max(0, right - left)
    actual_h = max(0, bottom - top)
    if actual_w <= 0 or actual_h <= 0:
        raise ValueError(f"裁剪区域无效: box={box}, left={left}, top={top}")
    crop = scene[top : top + actual_h, left : left + actual_w].copy()
    return crop, (left, top, actual_w, actual_h)


def main() -> None:
    ap = argparse.ArgumentParser(description="对局截图批量：血条下 ROI 与 equip_gallery 装备匹配，输出至 equip_recog/")
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT, help="对局截图目录（默认 对局截图/）")
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="输出根目录（默认 equip_recog/，运行前会清空）",
    )
    ap.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY, help="装备图鉴目录")
    ap.add_argument("--template", default=None, help="血条模板路径；默认自动查找")
    ap.add_argument(
        "--roi",
        default=None,
        help=f"血条检测 ROI x1,y1,x2,y2；默认 {DEFAULT_ROI}",
    )
    ap.add_argument("--width", type=int, default=120, help="血条下 ROI 宽度")
    ap.add_argument("--height", type=int, default=50, help="血条下 ROI 高度")
    ap.add_argument("--below-px", type=int, default=2, help="相对血条底边向下偏移像素")
    ap.add_argument(
        "--bars-from",
        choices=("auto", "detect", "result-json"),
        default="auto",
        help="血条框来源（与 bars_recog 一致）",
    )
    ap.add_argument("--result-json", default=None, help="强制指定 result.json")
    ap.add_argument(
        "--scales",
        type=str,
        default="24,25,26,27,28",
        help="装备模板边长列表",
    )
    ap.add_argument("--max-peaks-per-scale", type=int, default=4)
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.78,
        help="模板匹配峰值下限；与 fightboard_recog 默认同步（更好召回弱但真实装备；仍依赖特例过滤压误检）",
    )
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--nms-iou", type=float, default=0.35)
    ap.add_argument(
        "--blue-buff-gap-min",
        type=float,
        default=0.05,
        help="蓝霸符专项规则：当 top1=蓝霸符 时要求 top1-top2>=该阈值，否则过滤该 bar（特例规则）",
    )
    ap.add_argument("--label-font-size", type=int, default=12)
    ap.add_argument("--max-label-chars", type=int, default=14)
    args = ap.parse_args()

    root = _PROJECT_ROOT
    img_dir = args.img_dir if args.img_dir.is_absolute() else root / args.img_dir
    if not img_dir.is_dir():
        raise SystemExit(f"img-dir 不存在: {img_dir}")

    gallery_dir = args.gallery if args.gallery.is_absolute() else root / args.gallery
    if not gallery_dir.is_dir():
        raise SystemExit(f"图鉴目录不存在: {gallery_dir}")

    out_dir = args.out if args.out.is_absolute() else root / args.out
    print(f"[清空输出] {out_dir}")
    _clear_output_dir(out_dir)

    scales = tuple(int(x.strip()) for x in args.scales.split(",") if x.strip())
    if not scales or any(s < 2 for s in scales):
        raise SystemExit("--scales 无效")
    thr = float(args.threshold)
    top_k = int(args.top_k)
    nms_iou = float(args.nms_iou)
    max_peaks = int(args.max_peaks_per_scale)
    label_font_size = max(8, int(args.label_font_size))
    max_label_chars = max(4, int(args.max_label_chars))
    crop_w, crop_h = int(args.width), int(args.height)
    below_px = int(args.below_px)

    templates = _build_templates(gallery_dir)
    if not templates:
        raise SystemExit(f"图鉴无可用图片: {gallery_dir}")

    method = cv2.TM_CCOEFF_NORMED
    min_roi = max(scales)

    images = iter_input_images(img_dir)
    roi_user = bool(args.roi)
    roi_base = parse_roi_string(args.roi) if args.roi else DEFAULT_ROI

    csv_rows: List[List[str]] = []
    for image_path in images:
        stem = image_path.stem
        scene = _load_image(image_path)
        boxes, _tpl, roi, _resolved, _rj = resolve_bars_for_one_image(
            image_path,
            project_dir=root,
            bars_from=str(args.bars_from),
            result_json_opt=args.result_json,
            template_opt=args.template,
            roi_user=roi_user,
            roi_start=roi_base,
            batch_embed_dirname=DEFAULT_BATCH_EMBED_OUT_DIRNAME,
        )
        vis = scene.copy()
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (128, 128, 255), 2)

        bi = 0
        for box in boxes:
            bi += 1
            try:
                crop_bgr, (left, top, aw, ah) = crop_below_bar_center(
                    scene, box, crop_w=crop_w, crop_h=crop_h, below_px=below_px
                )
            except ValueError:
                continue
            roi_g = _to_gray(crop_bgr)
            h, w = roi_g.shape[:2]
            if w < min_roi or h < min_roi:
                continue

            candidates: List[Tuple[float, int, int, str, int]] = []
            for name, tmpl_gray in templates:
                peaks = _collect_peaks_for_template(
                    roi_g, tmpl_gray, scales, method, thr, max_peaks
                )
                for score, x, y, side_win in peaks:
                    candidates.append((score, x, y, name, side_win))

            candidates.sort(key=lambda t: t[0], reverse=True)
            boxes_xy = [(c[1], c[2], c[4], c[4]) for c in candidates]
            scores = [c[0] for c in candidates]
            keep_idx = _nms_xywh(boxes_xy, scores, nms_iou) if candidates else []
            picked = [candidates[i] for i in keep_idx[:top_k]]
            picked = _apply_blue_buff_special_case(
                picked,
                gap_min=float(args.blue_buff_gap_min),
            )
            picked = _apply_spatula_special_case(picked)

            Hv, Wv = vis.shape[:2]
            for rank, (score, x, y, tname, side_win) in enumerate(picked, start=1):
                gx1, gy1 = left + x, top + y
                cv2.rectangle(
                    vis,
                    (gx1, gy1),
                    (gx1 + side_win, gy1 + side_win),
                    (0, 255, 0),
                    1,
                )
                stem_l = _equip_label_stem(tname)
                if len(stem_l) > max_label_chars:
                    stem_l = stem_l[: max_label_chars - 1] + "…"
                label = f"{stem_l} {score:.2f}"
                text_y = gy1 - 4
                if text_y < label_font_size + 2:
                    text_y = min(Hv - 2, gy1 + side_win + label_font_size + 2)
                text_x = max(0, min(gx1, Wv - 1))
                _draw_chinese_text(
                    vis, label, (text_x, text_y), font_size=label_font_size, color=(0, 255, 0)
                )
                csv_rows.append(
                    [
                        image_path.name,
                        str(bi),
                        tname,
                        f"{score:.6f}",
                        str(gx1),
                        str(gy1),
                        str(side_win),
                        str(rank),
                    ]
                )

        out_img = out_dir / f"{stem}_equip_annotated.png"
        _save_bgr(out_img, vis)
        print(f"[OK] {image_path.name} -> {out_img.name}  血条={len(boxes)}")

    csv_path = out_dir / "matches_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image",
                "bar_index",
                "template_file",
                "score",
                "global_x",
                "global_y",
                "side",
                "rank",
            ]
        )
        w.writerows(csv_rows)

    print(f"完成。输出: {out_dir}")


if __name__ == "__main__":
    main()
