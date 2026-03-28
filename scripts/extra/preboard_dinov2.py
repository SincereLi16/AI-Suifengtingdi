# -*- coding: utf-8 -*-
"""
preboard_info：备战席（单行 9 格）棋子 + 装备综合识别。

与 `fightboard_recog` 的逻辑一致，但不同点：
1) 取样区域是备战席：只有一排 9 个格子，中心坐标已知（默认第1格中心(600,880)，第9格中心(1580,880)）。
   备战席格子的血条可能会向上超出格子范围，因此会在棋子检测 ROI 上增加“向上额外像素”（默认 70px），并在棋子检测 ROI 上向左额外覆盖像素（默认左侧额外 50px，总计 50），向右额外覆盖像素（默认 30px）。
2) 输出信息：棋子名称 + 装备（不输出棋盘位置）。
   标注仍会根据 chess_recog 的 `confidence`：`low` → 棋子名后加 "?"，且使用黄色。
"""

from __future__ import annotations

import sys
from pathlib import Path

for _d in Path(__file__).resolve().parents:
    if (_d / "repo_sys_path.py").exists():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
import repo_sys_path  # noqa: F401

import argparse
import json
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from element_recog import bars_recog as br
from element_recog import chess_recog as cr
from element_recog import equip_recog as er


from project_paths import DEFAULT_OUT_PREBOARD, PROJECT_ROOT

PROJECT_DIR = PROJECT_ROOT
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_PIECE_DIR = PROJECT_DIR / "chess_gallery"
DEFAULT_EQUIP_GALLERY = PROJECT_DIR / "equip_gallery"
DEFAULT_OUT = DEFAULT_OUT_PREBOARD


def _cell_left_top(cx: float, cy: float, cell_size: int) -> Tuple[int, int]:
    half = cell_size / 2.0
    return int(round(cx - half)), int(round(cy - half))


def _cell_index_from_x(cx: float, first_cx: float, step_x: float, n_cells: int) -> int:
    if step_x == 0:
        return 0
    idx = int(round((cx - first_cx) / step_x))
    return max(0, min(n_cells - 1, idx))


def _confidence_rank(conf: str) -> int:
    # 与 chess_recog 的定义保持一致
    c = (conf or "").lower()
    if c == "high":
        return 2
    if c == "medium":
        return 1
    if c == "low":
        return 0
    return 0


def _assign_piece_bars_to_cells(
    results: List[Dict[str, Any]],
    *,
    n_cells: int,
    first_cx: float,
    step_x: float,
) -> Dict[int, Dict[str, Any]]:
    """
    把 chess_recog 输出的 bar_box 按 x 坐标映射到备战席第 i 格。
    如果多个 bar 命中同一格，取证据更强者（confidence > used_samples > best_score）。
    """
    out: Dict[int, Dict[str, Any]] = {}
    for r in results:
        bar_box = r.get("bar_box") or [0, 0, 0, 0]
        bx, by, bw, bh = map(int, bar_box)
        cx = bx + bw / 2.0
        idx = _cell_index_from_x(cx, first_cx=first_cx, step_x=step_x, n_cells=n_cells)

        prev = out.get(idx)
        if prev is None:
            out[idx] = r
            continue

        prev_rank = _confidence_rank(str(prev.get("confidence") or ""))
        cur_rank = _confidence_rank(str(r.get("confidence") or ""))
        if cur_rank != prev_rank:
            if cur_rank > prev_rank:
                out[idx] = r
            continue

        prev_used = int(prev.get("used_samples") or 0)
        cur_used = int(r.get("used_samples") or 0)
        if cur_used != prev_used:
            if cur_used > prev_used:
                out[idx] = r
            continue

        if float(r.get("best_score") or 0.0) > float(prev.get("best_score") or 0.0):
            out[idx] = r
    return out


def _detect_one_bar_equip_top1(
    scene_bgr: "cv2.Mat",
    *,
    bar_box_xywh: Sequence[int],
    templates: List[Tuple[str, Any]],
    scales: Sequence[int],
    method: int,
    threshold: float,
    max_peaks_per_scale: int,
    top_k: int,
    nms_iou: float,
    below_px: int,
    crop_w: int,
    crop_h: int,
    min_roi: int,
    blue_buff_gap_min: float,
) -> Optional[Dict[str, Any]]:
    x, y, w, h = map(int, bar_box_xywh)
    crop_bgr, (crop_left, crop_top, _cw, _ch) = er.crop_below_bar_center(
        scene_bgr,
        (x, y, w, h),
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        below_px=int(below_px),
    )
    roi_g = er._to_gray(crop_bgr)
    rh, rw = roi_g.shape[:2]
    if rw < int(min_roi) or rh < int(min_roi):
        return None

    candidates: List[Tuple[float, int, int, str, int]] = []
    for name, tmpl_gray in templates:
        peaks = er._collect_peaks_for_template(
            roi_g,
            tmpl_gray,
            scales,
            method,
            float(threshold),
            int(max_peaks_per_scale),
        )
        for score, px, py, side_win in peaks:
            candidates.append((float(score), int(px), int(py), str(name), int(side_win)))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    boxes_xy = [(c[1], c[2], c[4], c[4]) for c in candidates]
    scores = [c[0] for c in candidates]
    keep_idx = er._nms_xywh(boxes_xy, scores, float(nms_iou)) if candidates else []
    picked = [candidates[i] for i in keep_idx[: int(top_k)]]

    # 蓝霸符专项规则（特例）
    picked = er._apply_blue_buff_special_case(picked, gap_min=float(blue_buff_gap_min))
    # 金铲铲机制（特例）：若识别出秘法手套，只保留秘法手套
    picked = er._apply_spatula_special_case(picked)

    if not picked:
        return None

    top1 = picked[0]
    equip_name = er._equip_label_stem(top1[3])
    return {"name": equip_name, "score": float(top1[0]), "gx1": int(crop_left + top1[1]), "gy1": int(crop_top + top1[2])}


def _draw_preboard_overlay(
    scene_bgr: "cv2.Mat",
    *,
    coverage_roi: Tuple[int, int, int, int],
    n_cells: int,
    cell_size: int,
    pieces_by_cell: Dict[int, Dict[str, Any]],
    equip_by_cell: Dict[int, Optional[Dict[str, Any]]],
    font_size: int,
) -> "cv2.Mat":
    vis = scene_bgr.copy()

    # 框出 chess_recog 实际使用的覆盖区域（便于你确认识别范围是否过大）
    rx1, ry1, rx2, ry2 = map(int, coverage_roi)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)

    for i in range(n_cells):
        piece_obj = pieces_by_cell.get(i)
        if piece_obj is None:
            # 若无棋子则不标注
            continue

        bar_box = piece_obj.get("bar_box") or [0, 0, 0, 0]
        bx, by, bw, bh = map(int, bar_box)
        if bw <= 0 or bh <= 0:
            continue

        # 与 fightboard 一致：框出血条
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)

        piece = str(piece_obj.get("best") or "?")
        conf = str(piece_obj.get("confidence") or "")
        piece_label = f"{piece}?" if conf.lower() == "low" else piece
        piece_color = (0, 200, 255) if conf.lower() == "low" else (0, 255, 0)

        # 棋子名：放在血条上方一点
        piece_xy = (bx, max(0, by - (int(font_size) + 8)))
        cr._draw_chinese_text(
            vis,
            piece_label,
            (int(piece_xy[0]), int(piece_xy[1])),
            font_size=int(font_size),
            color=piece_color,
        )

        # 若无装备则不标注装备
        equip_obj = equip_by_cell.get(i)
        if equip_obj is None:
            continue

        equip_name = str(equip_obj.get("name") or "?")
        gx1 = int(equip_obj.get("gx1") or bx)
        gy1 = int(equip_obj.get("gy1") or (by + bh + 2))
        equip_xy = (gx1, min(vis.shape[0] - 1, gy1 + 12))

        # 装备：仅输出装备名
        cr._draw_chinese_text(
            vis,
            f"装:{equip_name}",
            (int(equip_xy[0]), int(equip_xy[1])),
            font_size=max(12, int(font_size) - 4),
            color=(255, 255, 0),
        )

    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description="preboard_info：备战席棋子+装备综合标注")
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT, help="对局截图目录/单张图")
    ap.add_argument("--piece-dir", type=Path, default=DEFAULT_PIECE_DIR, help="chess_gallery 目录")
    ap.add_argument("--equip-gallery", type=Path, default=DEFAULT_EQUIP_GALLERY, help="equip_gallery 目录")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录（会清空）")

    # 备战席几何参数（来自用户给定）
    ap.add_argument("--first-cx", type=float, default=600.0)
    ap.add_argument("--last-cx", type=float, default=1580.0)
    ap.add_argument("--cy", type=float, default=880.0)
    ap.add_argument("--n-cells", type=int, default=9)
    ap.add_argument("--cell-size", type=int, default=120)

    # 血条可能向上超出格子：扩展检测 ROI
    ap.add_argument("--seat-up-extra-px", type=int, default=70, help="棋子检测 ROI 向上额外覆盖像素")
    ap.add_argument("--seat-down-extra-px", type=int, default=30, help="棋子检测 ROI 向下额外覆盖像素")
    ap.add_argument(
        "--seat-lr-extra-px",
        type=int,
        default=30,
        help="棋子检测 ROI 向右额外覆盖像素（默认右侧 30px；左侧还会叠加 seat-left-extra-px）",
    )
    ap.add_argument(
        "--seat-left-extra-px",
        type=int,
        default=20,
        help="棋子检测 ROI 向左额外覆盖像素（在 seat-lr-extra-px 的基础上再加）",
    )

    # equip_recog 参数（只取 top1，所以 top-k/nms 仍用于候选稳定）
    ap.add_argument("--equip-threshold", type=float, default=0.78, help="equip_recog 模板匹配峰值阈值")
    ap.add_argument("--equip-scales", type=str, default="24,25,26,27,28", help="equip_recog 模板多尺度边长列表")
    ap.add_argument("--equip-max-peaks-per-scale", type=int, default=4)
    ap.add_argument("--equip-top-k", type=int, default=15)
    ap.add_argument("--equip-nms-iou", type=float, default=0.35)
    ap.add_argument("--equip-width", type=int, default=120, help="血条下 ROI 宽")
    ap.add_argument("--equip-height", type=int, default=50, help="血条下 ROI 高")
    ap.add_argument("--equip-below-px", type=int, default=2, help="血条底边向下偏移像素")
    ap.add_argument("--blue-buff-gap-min", type=float, default=0.05, help="蓝霸符专项规则：top1-top2 >= 该值才保留（特例规则）")

    ap.add_argument("--font-size", type=int, default=16, help="综合标注字体大小")
    ap.add_argument("--json", action="store_true", help="每张图额外输出 preboard summary json")

    args = ap.parse_args()

    img_dir = args.img_dir.resolve()
    if not img_dir.exists():
        raise SystemExit(f"img-dir 不存在: {img_dir}")
    piece_dir = args.piece_dir.resolve()
    if not piece_dir.exists():
        raise SystemExit(f"piece-dir 不存在: {piece_dir}")
    equip_gallery_dir = args.equip_gallery.resolve()
    if not equip_gallery_dir.exists():
        raise SystemExit(f"equip-gallery 不存在: {equip_gallery_dir}")

    out_root = args.out.resolve()
    if out_root.exists():
        import shutil

        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # templates/equip 匹配准备
    templates = er._build_templates(equip_gallery_dir)
    if not templates:
        raise SystemExit(f"equip-gallery 无可用图片: {equip_gallery_dir}")
    method = cv2.TM_CCOEFF_NORMED
    scales = tuple(int(x.strip()) for x in args.equip_scales.split(",") if x.strip())
    min_roi = max(scales)

    # chess_recog embedding/model 准备（固定使用 DINOv2）
    template = br.find_healthbar_template(PROJECT_DIR)
    model, device, transform = cr._get_embedding_model("dinov2_vits14")
    piece_db, _ = cr.load_or_build_piece_embedding_db(
        piece_dir,
        model,
        device,
        transform,
        embed_backbone="dinov2_vits14",
        root=PROJECT_DIR,
        force_rebuild=False,
        verbose=True,
    )

    # preboard 运行时动态 ROI：单行 9 格 + 向上扩展
    half = args.cell_size / 2.0
    # 左侧在右侧同样基础 padding 上额外再加 20px（seat-left-extra-px）
    roi_left = int(
        round(min(args.first_cx, args.last_cx) - half - float(args.seat_lr_extra_px) - float(args.seat_left_extra_px))
    )
    roi_right = int(round(max(args.first_cx, args.last_cx) + half + float(args.seat_lr_extra_px)))
    roi_top = int(round(args.cy - half - float(args.seat_up_extra_px)))
    roi_bottom = int(round(args.cy + half + float(args.seat_down_extra_px)))
    custom_roi = (roi_left, roi_top, roi_right, roi_bottom)

    images = br.iter_input_images(img_dir)
    for image_path in images:
        stem = image_path.stem
        scene_bgr = cr._load_image(image_path)

        # 1) chess_recog：在备战席 ROI 里识别棋子（忽略位置字段，只用 best/confidence）
        old_roi = cr.ROI
        try:
            cr.ROI = custom_roi
            with tempfile.TemporaryDirectory(prefix="preboard_recog_chess_") as td:
                chess_out_dir = Path(td) / "chess"
                chess_out_dir.mkdir(parents=True, exist_ok=True)
                out_json = cr.run_recognition(
                    image_path=image_path,
                    template_path=template,
                    piece_dir=piece_dir,
                    output_dir=chess_out_dir,
                    circle_diameter=84,
                    embed_backbone="dinov2_vits14",
                    model=model,
                    device=device,
                    transform=transform,
                    piece_db=piece_db,
                    alpha_tight=True,
                    # 其它逻辑仍按 chess_recog 的内部实现走；我们主要依赖 ROI 裁剪带来的检测范围限制
                )
                results: List[Dict[str, Any]] = out_json.get("results") or []
        finally:
            cr.ROI = old_roi

        pieces_by_cell = _assign_piece_bars_to_cells(
            results,
            n_cells=int(args.n_cells),
            first_cx=float(args.first_cx),
            step_x=(float(args.last_cx) - float(args.first_cx)) / max(1, int(args.n_cells) - 1),
        )

        # 2) equip_recog：对每个 cell 命中的血条，识别 top1 装备
        equip_by_cell: Dict[int, Optional[Dict[str, Any]]] = {}
        for idx, piece_obj in pieces_by_cell.items():
            bar_box = piece_obj.get("bar_box") or [0, 0, 0, 0]
            equip_by_cell[idx] = _detect_one_bar_equip_top1(
                scene_bgr,
                bar_box_xywh=bar_box,
                templates=templates,
                scales=scales,
                method=method,
                threshold=float(args.equip_threshold),
                max_peaks_per_scale=int(args.equip_max_peaks_per_scale),
                top_k=int(args.equip_top_k),
                nms_iou=float(args.equip_nms_iou),
                below_px=int(args.equip_below_px),
                crop_w=int(args.equip_width),
                crop_h=int(args.equip_height),
                min_roi=int(min_roi),
                blue_buff_gap_min=float(args.blue_buff_gap_min),
            )

        # 3) 输出综合标注图
        vis = _draw_preboard_overlay(
            scene_bgr,
            coverage_roi=custom_roi,
            n_cells=int(args.n_cells),
            cell_size=int(args.cell_size),
            pieces_by_cell=pieces_by_cell,
            equip_by_cell=equip_by_cell,
            font_size=int(args.font_size),
        )
        out_path = out_root / f"{stem}_preboard_综合标注.png"
        cr._save_image(vis, out_path)
        if args.json:
            summary = {
                "file": image_path.name,
                "annotated_image": out_path.name,
                "coverage_roi": list(custom_roi),
                "pieces_by_cell": pieces_by_cell,
                "equip_by_cell": equip_by_cell,
            }
            out_json_path = out_root / f"{stem}_preboard_summary.json"
            out_json_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"[OK] {image_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()

