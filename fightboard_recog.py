# -*- coding: utf-8 -*-
"""
fightboard_recog：把 chess_recog + equip_recog 串起来。

输入：对局截图文件夹（或单张截图文件）
输出：每张图一张“综合标记图”，标注：
  1) 棋子名称
  2) 棋盘格位置（来自 chess_recog 的 position）
  3) 对应血条下方识别到的装备（来自 equip_recog 的模板匹配逻辑）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import tempfile

from element_recog import chess_recog as cr
from element_recog import equip_recog as er
from element_recog import bars_recog as br


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_PIECE_DIR = PROJECT_DIR / "chess_gallery"
DEFAULT_EQUIP_GALLERY = PROJECT_DIR / "equip_gallery"
DEFAULT_OUT = PROJECT_DIR / "fightboard_info"


def _draw_two_line_label(
    bgr: "cv2.Mat",
    *,
    line1: str,
    line2: str,
    xy: Tuple[int, int],
    font_size: int,
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
) -> None:
    x, y = int(xy[0]), int(xy[1])
    cr._draw_chinese_text(bgr, line1, (x, y), font_size=font_size, color=c1)
    dy = max(10, font_size + 4)
    cr._draw_chinese_text(bgr, line2, (x, y + dy), font_size=max(10, font_size - 2), color=c2)


def _detect_one_bar_equip(
    scene_bgr: "cv2.Mat",
    *,
    bar_box_xywh: Sequence[int],
    templates: List[Tuple[str, "Any"]],
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
    label_topn: int,
) -> List[Dict[str, Any]]:
    # 复用 equip_recog 的“血条中心下方裁剪 + 多尺度模板匹配 + NMS + 蓝霸符特例过滤”
    x, y, w, h = map(int, bar_box_xywh)
    crop_bgr, (crop_left, crop_top, _crop_aw, _crop_ah) = er.crop_below_bar_center(
        scene_bgr,
        (x, y, w, h),
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        below_px=int(below_px),
    )
    roi_g = er._to_gray(crop_bgr)
    rh, rw = roi_g.shape[:2]
    if rw < int(min_roi) or rh < int(min_roi):
        return []

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
            candidates.append((score, int(px), int(py), str(name), int(side_win)))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)
    boxes_xy = [(c[1], c[2], c[4], c[4]) for c in candidates]
    scores = [c[0] for c in candidates]
    keep_idx = er._nms_xywh(boxes_xy, scores, float(nms_iou)) if candidates else []
    picked = [candidates[i] for i in keep_idx[: int(top_k)]]

    # 蓝霸符专项规则（特例）
    picked = er._apply_blue_buff_special_case(picked, gap_min=float(blue_buff_gap_min))
    if not picked:
        return []

    # 金铲铲机制（特例）：若识别出“秘法手套”，则该装备区只输出秘法手套
    picked = er._apply_spatula_special_case(picked)
    if not picked:
        return []

    # picked 结构：(score, x, y, name, side_win)
    out: List[Dict[str, Any]] = []
    for rank, (score, px, py, tname, side_win) in enumerate(picked[: int(label_topn)], start=1):
        gx1 = int(crop_left + int(px))
        gy1 = int(crop_top + int(py))
        out.append(
            {
                "name": er._equip_label_stem(tname),
                "score": float(score),
                "gx1": gx1,
                "gy1": gy1,
                "side": int(side_win),
                "rank": int(rank),
            }
        )
    return out


def _overlay_fightboard(
    *,
    scene_bgr: "cv2.Mat",
    results: List[Dict[str, Any]],
    equip_by_bar: Dict[int, List[Dict[str, Any]]],
    font_size: int,
) -> "cv2.Mat":
    vis = scene_bgr.copy()
    for r in results:
        bar_index = int(r.get("bar_index") or 0)
        bar_box = r.get("bar_box") or [0, 0, 0, 0]
        x, y, w, h = map(int, bar_box)
        piece = r.get("best") or "?"
        conf = r.get("confidence") or ""
        # 与 chess_recog 一致：low 置信度 → 黄色 + "?" 后缀
        piece_label = f"{piece}?" if str(conf) == "low" else str(piece)
        text_color = (0, 200, 255) if str(conf) == "low" else (0, 255, 0)
        pos = "?"
        pos_obj = r.get("position") or {}
        if isinstance(pos_obj, dict):
            pos = pos_obj.get("label") or "?"

        # 文字分层：棋子/位置放在血条上方；装备放到装备候选框附近（避免都挤在血条上）
        line1 = f"{piece_label} {pos}".strip()
        tx, ty = x, max(0, y - (int(font_size) + 8))
        cr._draw_chinese_text(
            vis,
            line1,
            (int(tx), int(ty)),
            font_size=int(font_size),
            color=text_color,
        )

        equip_list = equip_by_bar.get(bar_index) or []
        if equip_list:
            for eq in equip_list:
                gx1 = int(eq.get("gx1") or x)
                gy1 = int(eq.get("gy1") or (y + h + 2))
                side = int(eq.get("side") or 0)
                rank = int(eq.get("rank") or 1)
                eq_name = str(eq.get("name") or "?")
                score = float(eq.get("score") or 0.0)

                # 可视化候选框（帮助你确认装备来源）
                cv2.rectangle(
                    vis,
                    (gx1, gy1),
                    (min(vis.shape[1] - 1, gx1 + side), min(vis.shape[0] - 1, gy1 + side)),
                    (0, 200, 255),
                    1,
                )

                # 轻微错开，减少多件装备文字重叠
                text_x = max(0, min(gx1, vis.shape[1] - 1))
                text_y = max(0, min(gy1 + (rank - 1) * (int(font_size) + 3), vis.shape[0] - 1))
                cr._draw_chinese_text(
                    vis,
                    f"{eq_name} {score:.2f}",
                    (int(text_x), int(text_y)),
                    font_size=max(10, int(font_size) - 2),
                    color=(255, 255, 0),
                )
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description="fightboard_recog：棋子+位置+装备综合标注")
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT, help="对局截图文件夹/单张图")
    ap.add_argument("--piece-dir", type=Path, default=DEFAULT_PIECE_DIR, help="chess_recog 特征库目录")
    ap.add_argument("--equip-gallery", type=Path, default=DEFAULT_EQUIP_GALLERY, help="equip_recog 装备图鉴目录")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出根目录（会清空）")
    ap.add_argument("--circle-diameter", type=int, default=84, help="棋子采样圆直径（chess_recog）")
    ap.add_argument("--alpha-tight", action="store_true", default=True, help="chess_recog 采样裁剪 alpha 紧裁（默认开启）")

    # equip_recog 参数
    # 0.80 对某些真实装备会略偏严格；例如 7.png 的“大亨之铠”分数约 0.783。
    ap.add_argument("--equip-threshold", type=float, default=0.78, help="模板匹配峰值阈值（equip_recog）；更低可召回弱匹配但需依赖特例过滤减少误检")
    ap.add_argument("--equip-scales", type=str, default="24,25,26,27,28", help="模板多尺度边长列表")
    ap.add_argument("--equip-max-peaks-per-scale", type=int, default=4, help="每个尺度最多保留峰值数（equip_recog）")
    ap.add_argument("--equip-top-k", type=int, default=15, help="每个血条 NMS 后最多保留候选（equip_recog）")
    ap.add_argument("--equip-nms-iou", type=float, default=0.35, help="equip_recog NMS iou 阈值")
    ap.add_argument("--equip-width", type=int, default=120, help="血条下 ROI 宽（equip_recog）")
    ap.add_argument("--equip-height", type=int, default=50, help="血条下 ROI 高（equip_recog）")
    ap.add_argument("--equip-below-px", type=int, default=2, help="血条底边向下偏移（equip_recog）")
    ap.add_argument(
        "--blue-buff-gap-min",
        type=float,
        default=0.05,
        help="蓝霸符专项规则：top1-top2 必须 >= 该值，否则过滤该血条全部候选（特例规则）",
    )
    ap.add_argument("--equip-label-topn", type=int, default=3, help="每个血条最多标注前 N 件装备")
    ap.add_argument("--label-font-size", type=int, default=16, help="综合标注字体大小")
    ap.add_argument("--json", action="store_true", help="每张图额外输出 fightboard summary json")

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
    # 清空输出根目录
    if out_root.exists():
        import shutil

        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # templates: 装备图鉴只需构建一次
    templates = er._build_templates(equip_gallery_dir)
    if not templates:
        raise SystemExit(f"equip-gallery 无可用图片: {equip_gallery_dir}")
    method = cv2.TM_CCOEFF_NORMED
    scales = tuple(int(x.strip()) for x in args.equip_scales.split(",") if x.strip())
    if not scales or any(s < 2 for s in scales):
        raise SystemExit("--equip-scales 无效")
    min_roi = max(scales)

    # chess_recog 模型/特征库只构建一次
    template = br.find_healthbar_template(PROJECT_DIR)
    # 固定使用 DINOv2-ViT-S/14，避免误用实验性 backbone
    model, device, transform = cr._get_embedding_model("dinov2_vits14")
    piece_db, _ = cr.load_or_build_piece_embedding_db(
        piece_dir,
        model,
        device,
        transform,
        embed_backbone="dinov2_vits14",
        root=PROJECT_DIR,
        force_rebuild=False,
    )

    images = br.iter_input_images(img_dir)
    for image_path in images:
        stem = image_path.stem
        # 1) chess_recog：棋子名 + 格子位置
        # 为了满足“fightboard_info 里只保留最终图片”，中间输出放入临时目录并自动清理。
        scene_bgr = cr._load_image(image_path)
        with tempfile.TemporaryDirectory(prefix="fightboard_recog_chess_") as td:
            chess_out_dir = Path(td) / "chess"
            chess_out_dir.mkdir(parents=True, exist_ok=True)
            out_json = cr.run_recognition(
                image_path=image_path,
                template_path=template,
                piece_dir=piece_dir,
                output_dir=chess_out_dir,
                circle_diameter=int(args.circle_diameter),
                model=model,
                device=device,
                transform=transform,
                piece_db=piece_db,
                alpha_tight=bool(args.alpha_tight),
            )
            results: List[Dict[str, Any]] = out_json.get("results") or []

        # 2) equip_recog：对每个 bar_box 在其下方 ROI 做装备模板匹配
        equip_by_bar: Dict[int, List[Dict[str, Any]]] = {}
        for r in results:
            bar_index = int(r.get("bar_index") or 0)
            bar_box = r.get("bar_box") or [0, 0, 0, 0]
            equip_by_bar[bar_index] = _detect_one_bar_equip(
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
                label_topn=int(args.equip_label_topn),
            )

        # 3) 综合叠加输出
        vis = _overlay_fightboard(
            scene_bgr=scene_bgr,
            results=results,
            equip_by_bar=equip_by_bar,
            font_size=int(args.label_font_size),
        )
        out_path = out_root / f"{stem}_fightboard_综合标注.png"
        cr._save_image(vis, out_path)
        if args.json:
            summary = {
                "file": image_path.name,
                "annotated_image": out_path.name,
                "results": results,
                "equip_by_bar": equip_by_bar,
            }
            out_json_path = out_root / f"{stem}_fightboard_summary.json"
            out_json_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"[OK] {image_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()

