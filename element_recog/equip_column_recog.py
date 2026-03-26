# -*- coding: utf-8 -*-
"""
equip_column_recog：装备栏（左侧竖条 ROI）模板匹配识别。

与现有 `equip_recog.py` 匹配方法保持一致：
- 多尺度 `cv2.matchTemplate(T M_CCOEFF_NORMED)` + 局部峰值筛选
- 结果 NMS 去重
- 蓝霸符 / 金铲铲（秘法手套）特例过滤
- 叠加中文标注并输出带框图片

差异点：不再检测血条并裁剪「血条下方 ROI」，而是直接在“左侧装备栏竖条 ROI”
上做模板匹配；可将 ROI 切为网格，每格单独匹配。

蓝霸符特例：equip_recog 中「无第二候选则清空」用于多装备 ROI；切块后单格常只有
一条蓝霸符，本脚本对「仅 1 条蓝霸符」改为按分数保留（见 --blue-buff-single-min-score），
避免误删真阳性。
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from element_recog import equip_recog as er
except ImportError:
    import equip_recog as er  # type: ignore

try:
    from .bars_recog import iter_input_images
except ImportError:
    from bars_recog import iter_input_images  # type: ignore

# 本文件位于 element_recog/ 时，工程根为上一级（对局截图、equip_gallery、默认可写输出仍在项目根）
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "element_recog" else _SCRIPT_DIR
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_GALLERY = PROJECT_DIR / "equip_gallery"
DEFAULT_OUT = PROJECT_DIR / "equip_column_recog"

DEFAULT_EQUIP_ROI = (10, 120, 190, 920)  # x1,y1,x2,y2（按你给的“左侧竖条”默认推断）

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _parse_roi_string(s: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("equip-roi 需为 x1,y1,x2,y2 四个整数")
    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


def _split_roi_grid(
    rx1: int,
    ry1: int,
    rx2: int,
    ry2: int,
    *,
    cols: int,
    rows: int,
) -> List[Tuple[int, int, int, int, int]]:
    """
    把 ROI 等分成 cols*rows 个块，返回 [(block_idx, bx1, by1, bx2, by2), ...]
    使用“整除 + 余数摊分”的方式，保证覆盖完整且不越界。
    """
    if cols <= 0 or rows <= 0:
        raise ValueError("cols/rows 必须为正数")
    if rx2 <= rx1 or ry2 <= ry1:
        return []

    total_w = rx2 - rx1
    total_h = ry2 - ry1
    base_w = total_w // cols
    rem_w = total_w % cols
    base_h = total_h // rows
    rem_h = total_h % rows

    blocks: List[Tuple[int, int, int, int, int]] = []
    block_idx = 0
    for r in range(rows):
        by1 = ry1 + r * base_h + min(r, rem_h)
        bh = base_h + (1 if r < rem_h else 0)
        by2 = by1 + bh
        for c in range(cols):
            bx1 = rx1 + c * base_w + min(c, rem_w)
            bw = base_w + (1 if c < rem_w else 0)
            bx2 = bx1 + bw
            blocks.append((block_idx, bx1, by1, bx2, by2))
            block_idx += 1
    return blocks


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
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou_xywh(boxes[i], boxes[j]) < iou_thresh]
    return keep


def _build_templates(gallery_dir: Path) -> List[Tuple[str, np.ndarray]]:
    """
    读取图鉴图片，并统一转为灰度模板：
    返回 [(template_filename, template_gray), ...]
    """
    templates: List[Tuple[str, np.ndarray]] = []
    for p in sorted(gallery_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        bgr = er._load_image(p)
        g = _to_gray(bgr)
        if g.shape[0] < 2 or g.shape[1] < 2:
            continue
        templates.append((p.name, g))
    return templates


def _find_local_maxima(res: np.ndarray, threshold: float) -> List[Tuple[float, int, int]]:
    """
    在 matchTemplate 结果图 res 上找局部极大值，返回 [(score,x,y), ...]。
    """
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
    """
    对单个模板（灰度）在 roi_g 内做多尺度匹配：
    返回 [(score, x, y, side_win), ...]
    其中 x/y 是 ROI 内坐标（与 matchTemplate 的左上角一致）。
    """
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
    single_min_score: float,
) -> List[Tuple[float, int, int, str, int]]:
    """
    蓝霸符专项规则（equip_column 切块版，与 equip_recog 略有不同）：

    equip_recog 血条下 ROI 往往多装备，可用「top1-top2 分差」压误检。
    切块后单格常只有一个装备，NMS 后可能只剩一条「蓝霸符」候选；
    若仍沿用「无 top2 则整格清空」，会把真阳性蓝霸符误删。

    - 若 top1≠蓝霸符：原样返回
    - 若仅 1 条且为蓝霸符：分数 >= single_min_score 则保留，否则清空（仍防空白区低分误检）
    - 若≥2 条且 top1=蓝霸符：保持 top1-top2 分差 >= gap_min，否则清空
    """
    if not picked:
        return picked
    top1 = picked[0]
    if Path(top1[3]).stem != "蓝霸符":
        return picked
    if len(picked) < 2:
        if float(top1[0]) >= float(single_min_score):
            return picked
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
    金铲铲（秘法手套）专项规则（特例，保持与 equip_recog.py 一致）：
    如果识别到“秘法手套”，则只输出最高分的“秘法手套”，过滤掉其它装备候选。
    """
    if not picked:
        return picked
    for cand in picked:
        if Path(cand[3]).stem == "秘法手套":
            return [cand]
    return picked


def compute_equip_column_matches(
    scene_bgr: np.ndarray,
    *,
    templates: List[Tuple[str, np.ndarray]],
    equip_roi: Tuple[int, int, int, int] = DEFAULT_EQUIP_ROI,
    scales: Sequence[int] = (60, 61, 62, 63, 64),
    threshold: float = 0.6,
    max_peaks_per_scale: int = 4,
    top_k: int = 15,
    nms_iou: float = 0.35,
    blue_buff_gap_min: float = 0.05,
    blue_buff_single_min_score: float = 0.78,
    grid_cols: int = 2,
    grid_rows: int = 10,
) -> Dict[str, Any]:
    """
    在整图 BGR 上识别左侧装备栏竖条 ROI，返回结构化结果（供 pipeline 写 JSON / 再叠加绘制）。
    """
    x1, y1, x2, y2 = equip_roi
    H, W = scene_bgr.shape[:2]
    rx1 = max(0, min(int(x1), W))
    ry1 = max(0, min(int(y1), H))
    rx2 = max(0, min(int(x2), W))
    ry2 = max(0, min(int(y2), H))
    out_matches: List[Dict[str, Any]] = []
    if rx2 <= rx1 or ry2 <= ry1:
        return {"equip_roi": [rx1, ry1, rx2, ry2], "matches": [], "match_count": 0}

    scales_t = tuple(int(s) for s in scales if int(s) >= 2)
    if not scales_t:
        return {"equip_roi": [rx1, ry1, rx2, ry2], "matches": [], "match_count": 0}
    min_roi = max(scales_t)
    thr = float(threshold)
    max_peaks = int(max_peaks_per_scale)
    tk = int(top_k)
    method = cv2.TM_CCOEFF_NORMED
    gc = max(1, int(grid_cols))
    gr = max(1, int(grid_rows))

    blocks = _split_roi_grid(rx1, ry1, rx2, ry2, cols=gc, rows=gr)
    for block_idx, bx1, by1, bx2, by2 in blocks:
        roi_crop = scene_bgr[by1:by2, bx1:bx2].copy()
        roi_g = er._to_gray(roi_crop)
        if roi_g.shape[1] < min_roi or roi_g.shape[0] < min_roi:
            continue

        candidates: List[Tuple[float, int, int, str, int]] = []
        for name, tmpl_gray in templates:
            peaks = er._collect_peaks_for_template(
                roi_g,
                tmpl_gray,
                scales_t,
                method,
                thr,
                max_peaks,
            )
            for score, x, y, side_win in peaks:
                candidates.append((score, x, y, name, side_win))

        if not candidates:
            continue

        candidates.sort(key=lambda t: float(t[0]), reverse=True)
        boxes_xy = [(c[1], c[2], c[4], c[4]) for c in candidates]
        scores = [float(c[0]) for c in candidates]
        keep_idx = er._nms_xywh(boxes_xy, scores, nms_iou) if candidates else []
        picked = [candidates[i] for i in keep_idx[:tk]]
        picked = _apply_blue_buff_special_case(
            picked,
            gap_min=blue_buff_gap_min,
            single_min_score=blue_buff_single_min_score,
        )
        picked = er._apply_spatula_special_case(picked)

        region_index = block_idx + 1
        for rank, (score, x, y, tname, side_win) in enumerate(picked, start=1):
            gx1 = bx1 + int(x)
            gy1 = by1 + int(y)
            stem_l = er._equip_label_stem(tname)
            out_matches.append(
                {
                    "template_file": str(tname),
                    "name_stem": str(stem_l),
                    "score": float(score),
                    "global_x": int(gx1),
                    "global_y": int(gy1),
                    "side": int(side_win),
                    "roi_index": int(region_index),
                    "rank_in_block": int(rank),
                }
            )

    return {
        "equip_roi": [rx1, ry1, rx2, ry2],
        "matches": out_matches,
        "match_count": len(out_matches),
    }


def draw_equip_column_matches_on_bgr(
    base_bgr: np.ndarray,
    matches: Sequence[Dict[str, Any]],
    *,
    equip_roi: Tuple[int, int, int, int],
    draw_roi_box: bool = True,
    label_font_size: int = 12,
    max_label_chars: int = 14,
) -> np.ndarray:
    """在已与 scene 对齐的底图（如 fightboard 叠加图）上绘制装备栏匹配框与标签。"""
    vis = base_bgr.copy()
    Hv, Wv = vis.shape[:2]
    x1, y1, x2, y2 = equip_roi
    rx1 = max(0, min(int(x1), Wv))
    ry1 = max(0, min(int(y1), Hv))
    rx2 = max(0, min(int(x2), Wv))
    ry2 = max(0, min(int(y2), Hv))
    if draw_roi_box and rx2 > rx1 and ry2 > ry1:
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (128, 128, 255), 2)

    for m in matches:
        gx1 = int(m.get("global_x") or 0)
        gy1 = int(m.get("global_y") or 0)
        side_win = int(m.get("side") or 0)
        score = float(m.get("score") or 0.0)
        stem_l = str(m.get("name_stem") or "").strip() or "?"
        if len(stem_l) > max_label_chars:
            stem_l = stem_l[: max_label_chars - 1] + "…"
        if side_win <= 0:
            continue
        cv2.rectangle(vis, (gx1, gy1), (gx1 + side_win, gy1 + side_win), (0, 255, 0), 1)
        label = f"{stem_l} {score:.2f}"
        text_y = gy1 - 4
        if text_y < label_font_size + 2:
            text_y = min(Hv - 2, gy1 + side_win + label_font_size + 2)
        text_x = max(0, min(gx1, Wv - 1))
        er._draw_chinese_text(
            vis,
            label,
            (text_x, text_y),
            font_size=max(8, int(label_font_size)),
            color=(0, 255, 0),
        )
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description="对局截图批量：左侧装备栏 ROI 模板匹配，输出带标注图片")
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT, help="对局截图目录（默认 对局截图/）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出根目录（默认 equip_column_recog/，运行前会清空）")
    ap.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY, help="装备图鉴目录（默认 equip_gallery/）")
    ap.add_argument("--equip-roi", default=f"{DEFAULT_EQUIP_ROI[0]},{DEFAULT_EQUIP_ROI[1]},{DEFAULT_EQUIP_ROI[2]},{DEFAULT_EQUIP_ROI[3]}", help="装备栏 ROI：x1,y1,x2,y2（全图坐标）")

    # 装备栏竖条内图标：默认用边长 60~64 像素的正方形窗口做多尺度匹配（与 equip_recog 流程一致，仅尺度列表不同）。
    ap.add_argument(
        "--scales",
        type=str,
        default="60,61,62,63,64",
        help="装备模板边长列表（匹配正方形：s*s 像素；默认 60~64）",
    )
    # 与 equip_recog 默认 0.78 不同：小窗口尺度下峰值分布会变，需按图调 --threshold。
    ap.add_argument("--threshold", type=float, default=0.6, help="模板匹配峰值下限（60~64 窗口可从 0.6 起试）")
    ap.add_argument("--max-peaks-per-scale", type=int, default=4, help="每个模板每个尺度最多取峰值数量")
    ap.add_argument("--top-k", type=int, default=15, help="最终每张图最多输出的匹配数量（按分数排序）")
    ap.add_argument("--nms-iou", type=float, default=0.35, help="NMS IoU 阈值")

    ap.add_argument("--blue-buff-gap-min", type=float, default=0.05, help="蓝霸符专项：top1-top2 分差下限；小于则过滤本 ROI")
    ap.add_argument(
        "--blue-buff-single-min-score",
        type=float,
        default=0.78,
        help="切块模式：某格仅识别到 1 条蓝霸符时，分数≥该值才保留（防空白区误检；默认同 equip_recog 常见阈值附近）",
    )
    ap.add_argument("--label-font-size", type=int, default=12, help="中文标签字体大小")
    ap.add_argument("--max-label-chars", type=int, default=14, help="标签最大字符数（超过截断）")
    ap.add_argument("--grid-cols", type=int, default=2, help="装备栏 ROI 切块列数（默认 2）")
    ap.add_argument("--grid-rows", type=int, default=10, help="装备栏 ROI 切块行数（默认 10）")
    args = ap.parse_args()

    root = PROJECT_DIR
    img_dir = args.img_dir if args.img_dir.is_absolute() else (root / args.img_dir)
    gallery_dir = args.gallery if args.gallery.is_absolute() else (root / args.gallery)
    out_dir = args.out if args.out.is_absolute() else (root / args.out)

    if not img_dir.is_dir():
        raise SystemExit(f"img-dir 不存在: {img_dir}")
    if not gallery_dir.is_dir():
        raise SystemExit(f"图鉴目录不存在: {gallery_dir}")

    equip_roi = _parse_roi_string(str(args.equip_roi))
    x1, y1, x2, y2 = equip_roi

    _clear_output_dir(out_dir)

    scales = tuple(int(x.strip()) for x in str(args.scales).split(",") if x.strip())
    if not scales or any(s < 2 for s in scales):
        raise SystemExit("--scales 无效")
    thr = float(args.threshold)
    max_peaks = int(args.max_peaks_per_scale)
    top_k = int(args.top_k)
    nms_iou = float(args.nms_iou)
    gap_min = float(args.blue_buff_gap_min)
    blue_single_min = float(args.blue_buff_single_min_score)
    label_font_size = max(8, int(args.label_font_size))
    max_label_chars = max(4, int(args.max_label_chars))
    grid_cols = max(1, int(args.grid_cols))
    grid_rows = max(1, int(args.grid_rows))

    # 复用 equip_recog.py 的图鉴读取/模板预处理，确保识别机制与原脚本一致
    templates = er._build_templates(gallery_dir)
    if not templates:
        raise SystemExit(f"图鉴无可用图片: {gallery_dir}")

    method = cv2.TM_CCOEFF_NORMED

    csv_rows: List[List[str]] = []
    images = iter_input_images(img_dir, image_exts=IMAGE_EXTS)
    for image_path in images:
        stem = image_path.stem
        scene = er._load_image(image_path)
        H, W = scene.shape[:2]

        rx1 = max(0, min(int(x1), W))
        ry1 = max(0, min(int(y1), H))
        rx2 = max(0, min(int(x2), W))
        ry2 = max(0, min(int(y2), H))
        if rx2 <= rx1 or ry2 <= ry1:
            print(f"[跳过] {image_path.name}: equip-roi 裁剪为空")
            continue

        vis = scene.copy()
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (128, 128, 255), 2)

        min_roi = max(scales)
        Hv, Wv = vis.shape[:2]

        blocks = _split_roi_grid(rx1, ry1, rx2, ry2, cols=grid_cols, rows=grid_rows)
        if not blocks:
            print(f"[跳过] {image_path.name}: equip-roi 裁剪为空")
            continue

        total_picked = 0
        for block_idx, bx1, by1, bx2, by2 in blocks:
            # 每个切块内部做模板匹配：局部的 x/y 再映射回全局坐标
            roi_crop = scene[by1:by2, bx1:bx2].copy()
            roi_g = er._to_gray(roi_crop)
            if roi_g.shape[1] < min_roi or roi_g.shape[0] < min_roi:
                continue

            candidates: List[Tuple[float, int, int, str, int]] = []
            for name, tmpl_gray in templates:
                peaks = er._collect_peaks_for_template(
                    roi_g,
                    tmpl_gray,
                    scales,
                    method,
                    thr,
                    max_peaks,
                )
                for score, x, y, side_win in peaks:
                    candidates.append((score, x, y, name, side_win))

            if not candidates:
                continue

            candidates.sort(key=lambda t: float(t[0]), reverse=True)
            boxes_xy = [(c[1], c[2], c[4], c[4]) for c in candidates]
            scores = [float(c[0]) for c in candidates]
            keep_idx = er._nms_xywh(boxes_xy, scores, nms_iou) if candidates else []
            picked = [candidates[i] for i in keep_idx[:top_k]]
            picked = _apply_blue_buff_special_case(
                picked,
                gap_min=gap_min,
                single_min_score=blue_single_min,
            )
            picked = er._apply_spatula_special_case(picked)

            total_picked += len(picked)
            region_index = block_idx + 1  # 块索引从 1 开始
            for rank, (score, x, y, tname, side_win) in enumerate(picked, start=1):
                gx1 = bx1 + int(x)
                gy1 = by1 + int(y)

                cv2.rectangle(vis, (gx1, gy1), (gx1 + side_win, gy1 + side_win), (0, 255, 0), 1)

                stem_l = er._equip_label_stem(tname)
                if len(stem_l) > max_label_chars:
                    stem_l = stem_l[: max_label_chars - 1] + "…"

                label = f"{stem_l} {float(score):.2f}"
                text_y = gy1 - 4
                if text_y < label_font_size + 2:
                    text_y = min(Hv - 2, gy1 + side_win + label_font_size + 2)
                text_x = max(0, min(gx1, Wv - 1))

                er._draw_chinese_text(
                    vis,
                    label,
                    (text_x, text_y),
                    font_size=label_font_size,
                    color=(0, 255, 0),
                )

                csv_rows.append(
                    [
                        image_path.name,
                        str(region_index),
                        tname,
                        f"{float(score):.6f}",
                        str(gx1),
                        str(gy1),
                        str(side_win),
                        str(rank),
                    ]
                )

        out_img = out_dir / f"{stem}_equipcol_annotated.png"
        _save_bgr(out_img, vis)
        print(f"[OK] {image_path.name} -> {out_img.name}  匹配数={total_picked}")

    csv_path = out_dir / "matches_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "roi_index", "template_file", "score", "global_x", "global_y", "side", "rank"])
        w.writerows(csv_rows)

    print(f"完成。输出: {out_dir}")


if __name__ == "__main__":
    main()

