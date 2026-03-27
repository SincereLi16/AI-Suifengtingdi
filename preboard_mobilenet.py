# -*- coding: utf-8 -*-
"""
preboard_mobilenet：备战席识别（MobileNet + DML/CPU；血条为 fightboard V3 细切块 + OpenCV TM）
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from element_recog import bars_recog as br
from element_recog import chess_recog as cr
from element_recog import equip_recog as er

import fightboard_mobilenet as f3
from preboard_dinov2 import _assign_piece_bars_to_cells, _draw_preboard_overlay

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_PIECE_DIR = PROJECT_DIR / "chess_gallery"
DEFAULT_EQUIP_GALLERY = PROJECT_DIR / "equip_gallery"
DEFAULT_OUT = PROJECT_DIR / "preboard_info_v3"


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
    picked = er._apply_blue_buff_special_case(picked, gap_min=float(blue_buff_gap_min))
    picked = er._apply_spatula_special_case(picked)
    if not picked:
        return None
    top1 = picked[0]
    equip_name = er._equip_label_stem(top1[3])
    return {"name": equip_name, "score": float(top1[0]), "gx1": int(crop_left + top1[1]), "gy1": int(crop_top + top1[2])}


def main() -> None:
    ap = argparse.ArgumentParser(description="preboard_mobilenet：备战席棋子+装备综合标注（MobileNet+DML；血条 OpenCV TM）")
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--piece-dir", type=Path, default=DEFAULT_PIECE_DIR)
    ap.add_argument("--equip-gallery", type=Path, default=DEFAULT_EQUIP_GALLERY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)

    ap.add_argument("--first-cx", type=float, default=600.0)
    ap.add_argument("--last-cx", type=float, default=1580.0)
    ap.add_argument("--cy", type=float, default=880.0)
    ap.add_argument("--n-cells", type=int, default=9)
    ap.add_argument("--cell-size", type=int, default=120)

    ap.add_argument("--seat-up-extra-px", type=int, default=70)
    ap.add_argument("--seat-down-extra-px", type=int, default=30)
    ap.add_argument("--seat-lr-extra-px", type=int, default=30)
    ap.add_argument("--seat-left-extra-px", type=int, default=20)

    ap.add_argument("--torch-device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
    ap.add_argument("--torch-auto-priority", type=str, default="dml,cpu")
    ap.add_argument("--workers", type=int, default=1)

    ap.add_argument("--equip-threshold", type=float, default=0.78)
    ap.add_argument("--equip-scales", type=str, default="24,25,26,27,28")
    ap.add_argument("--equip-max-peaks-per-scale", type=int, default=4)
    ap.add_argument("--equip-top-k", type=int, default=15)
    ap.add_argument("--equip-nms-iou", type=float, default=0.35)
    ap.add_argument("--equip-width", type=int, default=120)
    ap.add_argument("--equip-height", type=int, default=50)
    ap.add_argument("--equip-below-px", type=int, default=2)
    ap.add_argument("--blue-buff-gap-min", type=float, default=0.05)
    ap.add_argument("--equip-workers", type=int, default=1)

    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    img_dir = args.img_dir.resolve()
    piece_dir = args.piece_dir.resolve()
    equip_gallery_dir = args.equip_gallery.resolve()
    out_root = args.out.resolve()

    if out_root.exists():
        import shutil
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    templates = er._build_templates(equip_gallery_dir)
    method = cv2.TM_CCOEFF_NORMED
    scales = tuple(int(x.strip()) for x in args.equip_scales.split(",") if x.strip())
    min_roi = max(scales)

    template = br.find_healthbar_template(PROJECT_DIR)
    mobilenet_bundle = f3._get_worker_mobilenet_bundle(
        piece_dir=piece_dir,
        torch_device=str(args.torch_device),
        prefer_rocm=False,
        torch_auto_priority=str(args.torch_auto_priority),
        torch_fallback_to_cpu=True,
    )

    half = args.cell_size / 2.0
    custom_roi = (
        int(round(min(args.first_cx, args.last_cx) - half - float(args.seat_lr_extra_px) - float(args.seat_left_extra_px))),
        int(round(args.cy - half - float(args.seat_up_extra_px))),
        int(round(max(args.first_cx, args.last_cx) + half + float(args.seat_lr_extra_px))),
        int(round(args.cy + half + float(args.seat_down_extra_px))),
    )

    # 血条检测函数替换为 fightboard V3（细切块 + OpenCV TM）
    legacy_detect = f3._detect_healthbars_in_roi
    f3._detect_healthbars_in_roi = lambda scene, templates, roi, strategy="simple_tiled", simple_threshold=0.58: f3._v3_detect_healthbars_in_roi(
        scene, templates, roi, simple_threshold=float(simple_threshold)
    )

    # preboard 只处理主图 -a
    images = [p for p in br.iter_input_images(img_dir) if p.stem.lower().endswith("-a")]

    for image_path in images:
        stem = image_path.stem
        scene_bgr = cr._load_image(image_path)

        old_roi = f3.CR_ROI
        try:
            f3.CR_ROI = custom_roi
            with tempfile.TemporaryDirectory(prefix="preboard_mobilenet_chess_") as td:
                chess_out_dir = Path(td) / "chess"
                chess_out_dir.mkdir(parents=True, exist_ok=True)
                out_json = f3.run_recognition_chess(
                    image_path=image_path,
                    template_path=template,
                    piece_dir=piece_dir,
                    output_dir=chess_out_dir,
                    circle_diameter=84,
                    alpha_tight=True,
                    save_debug_crops=False,
                    batch_embed=True,
                    mobilenet_bundle=mobilenet_bundle,
                    torch_device=str(args.torch_device),
                    torch_auto_priority=str(args.torch_auto_priority),
                )
                results: List[Dict[str, Any]] = out_json.get("results") or []
        finally:
            f3.CR_ROI = old_roi

        pieces_by_cell = _assign_piece_bars_to_cells(
            results,
            n_cells=int(args.n_cells),
            first_cx=float(args.first_cx),
            step_x=(float(args.last_cx) - float(args.first_cx)) / max(1, int(args.n_cells) - 1),
        )

        equip_by_cell: Dict[int, Optional[Dict[str, Any]]] = {}
        ew = max(1, int(args.equip_workers))

        def _job(idx: int, piece_obj: Dict[str, Any]):
            bar_box = piece_obj.get("bar_box") or [0, 0, 0, 0]
            return idx, _detect_one_bar_equip_top1(
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

        if ew <= 1:
            for idx, piece_obj in pieces_by_cell.items():
                i, o = _job(idx, piece_obj)
                equip_by_cell[i] = o
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=ew) as ex:
                futs = [ex.submit(_job, idx, piece_obj) for idx, piece_obj in pieces_by_cell.items()]
                for f in concurrent.futures.as_completed(futs):
                    i, o = f.result()
                    equip_by_cell[i] = o

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
            (out_root / f"{stem}_preboard_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {image_path.name} -> {out_path.name}")

    f3._detect_healthbars_in_roi = legacy_detect


if __name__ == "__main__":
    main()
