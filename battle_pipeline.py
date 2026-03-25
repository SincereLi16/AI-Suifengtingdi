# -*- coding: utf-8 -*-
"""
battle_pipeline（内存一体化版本）：
- 不再通过 subprocess 调 fight/pre/player 子脚本
- 单进程内初始化一次模型与 OCR，然后逐图完成整条链路
- 每张图直接产出：*_annotated.png + *_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import fightboard_recog as fb
import preboard_recog as pb
import player_recog as pr


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_OUT = PROJECT_DIR / "battle_pipeline_out"
CACHE_SCHEMA_VERSION = "bp_cache_v1"


def _iter_images(path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"输入路径不存在: {path}")
    files = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"输入目录内无图像: {path}")
    return files


def _save_image(path: Path, bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    path.write_bytes(buf.tobytes())


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _cache_key(module: str, image_sha1: str, params_key: str) -> str:
    raw = f"{CACHE_SCHEMA_VERSION}|{module}|{image_sha1}|{params_key}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_module_cache(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    p = cache_dir / f"{key}.pkl"
    if not p.is_file():
        return None
    try:
        obj = pickle.loads(p.read_bytes())
        if isinstance(obj, dict) and "vis" in obj and "summary" in obj:
            return obj
    except Exception:
        return None
    return None


def _save_module_cache(cache_dir: Path, key: str, data: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.pkl"
    p.write_bytes(pickle.dumps(data))


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _summary_lines_from_modules(
    fight_json: Optional[Dict[str, Any]],
    pre_json: Optional[Dict[str, Any]],
    player_json: Optional[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []

    fr = (fight_json or {}).get("results") or []
    equip_by_bar = (fight_json or {}).get("equip_by_bar") or {}
    lines.append("【Fightboard】")
    if not fr:
        lines.append("(无棋子)")
    for r in fr:
        bi = int((r or {}).get("bar_index") or 0)
        name = str((r or {}).get("best") or "?")
        conf = str((r or {}).get("confidence") or "")
        pos = "?"
        pos_obj = (r or {}).get("position") or {}
        if isinstance(pos_obj, dict):
            pos = str(pos_obj.get("label") or "?")
        eq_list = equip_by_bar.get(str(bi)) if isinstance(equip_by_bar, dict) else None
        if eq_list is None and isinstance(equip_by_bar, dict):
            eq_list = equip_by_bar.get(bi)
        equip_names: List[str] = []
        if isinstance(eq_list, list):
            for e in eq_list:
                if not isinstance(e, dict):
                    continue
                en = str(e.get("name") or "").strip()
                if en:
                    equip_names.append(en)
        if equip_names:
            lines.append(f"- {name} ({conf}) 位置:{pos} 装备:{'/'.join(equip_names)}")
        else:
            lines.append(f"- {name} ({conf}) 位置:{pos}")

    pc = (pre_json or {}).get("pieces_by_cell") or {}
    eqc = (pre_json or {}).get("equip_by_cell") or {}
    lines.append("【Preboard】")
    pre_keys = sorted(pc.keys(), key=lambda x: _safe_int(x, 0))
    if not pre_keys:
        lines.append("(无棋子)")
    for k in pre_keys:
        o = pc.get(k) or {}
        nm = str(o.get("best") or "").strip()
        if not nm:
            continue
        eo = eqc.get(k) if isinstance(eqc, dict) else None
        if eo is None and isinstance(eqc, dict):
            eo = eqc.get(str(k))
        en = ""
        if isinstance(eo, dict):
            en = str(eo.get("name") or "").strip()
        if en:
            lines.append(f"- 格{k}: {nm} 装备:{en}")
        else:
            lines.append(f"- 格{k}: {nm}")

    fields = ((player_json or {}).get("fields") or {}) if isinstance(player_json, dict) else {}
    bonds = fields.get("bonds") if isinstance(fields, dict) else None
    bsum = str((bonds or {}).get("bond_summary") or (bonds or {}).get("parsed") or "").strip() if isinstance(bonds, dict) else ""
    lines.append("【Player】")
    lines.append("羁绊:")
    lines.append(bsum if bsum else "(empty)")
    for k in ("phase", "level", "exp", "gold", "streak"):
        d = fields.get(k) if isinstance(fields, dict) else None
        if isinstance(d, dict):
            pv = str(d.get("parsed") or "").strip()
            if pv:
                lines.append(f"{d.get('name') or k}: {pv}")
    # 装备栏（若 player 输出里存在该字段则展示）
    equip_field = fields.get("equip") if isinstance(fields, dict) else None
    if isinstance(equip_field, dict):
        ev = str(equip_field.get("parsed") or "").strip()
        if ev:
            lines.append(f"装备栏: {ev}")
    return lines


def _draw_text_panel(base: np.ndarray, lines: List[str]) -> np.ndarray:
    h, w = base.shape[:2]
    panel_w = max(620, int(w * 0.42))
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, h - 1), (90, 90, 90), 1)
    cv2.putText(panel, "Unified Summary", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = None
        for fp in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
            if fp.exists():
                try:
                    font = ImageFont.truetype(str(fp), 18)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        dr = ImageDraw.Draw(pil)
        y = 58
        for ln in lines:
            if y > h - 10:
                break
            dr.text((12, y), ln, font=font, fill=(255, 255, 255))
            y += 23
        panel = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        y = 58
        for ln in lines:
            if y > h - 10:
                break
            cv2.putText(panel, ln[:120], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 23
    return np.hstack([base, panel])


def _group_key(stem: str, pattern: str) -> str:
    m = re.match(pattern, stem)
    return m.group(1) if m else stem


def _run_one_fightboard(
    *,
    image_path: Path,
    template,
    model,
    device,
    transform,
    piece_db,
    equip_templates,
    scales: List[int],
    min_roi: int,
    temp_out: Path,
) -> Dict[str, Any]:
    scene_bgr = fb.cr._load_image(image_path)
    out_json = fb.cr.run_recognition(
        image_path=image_path,
        template_path=template,
        piece_dir=PROJECT_DIR / "chess_gallery",
        output_dir=temp_out,
        circle_diameter=84,
        model=model,
        device=device,
        transform=transform,
        piece_db=piece_db,
        alpha_tight=True,
    )
    results = out_json.get("results") or []
    equip_by_bar: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        bi = int(r.get("bar_index") or 0)
        bar_box = r.get("bar_box") or [0, 0, 0, 0]
        equip_by_bar[bi] = fb._detect_one_bar_equip(
            scene_bgr,
            bar_box_xywh=bar_box,
            templates=equip_templates,
            scales=scales,
            method=cv2.TM_CCOEFF_NORMED,
            threshold=0.78,
            max_peaks_per_scale=4,
            top_k=15,
            nms_iou=0.35,
            below_px=2,
            crop_w=120,
            crop_h=50,
            min_roi=min_roi,
            blue_buff_gap_min=0.05,
            label_topn=3,
        )
    vis = fb._overlay_fightboard(scene_bgr=scene_bgr, results=results, equip_by_bar=equip_by_bar, font_size=16)
    return {"vis": vis, "summary": {"file": image_path.name, "results": results, "equip_by_bar": equip_by_bar}}


def _run_one_preboard(
    *,
    image_path: Path,
    template,
    model,
    device,
    transform,
    piece_db,
    equip_templates,
    scales: List[int],
    min_roi: int,
    temp_out: Path,
) -> Dict[str, Any]:
    scene_bgr = pb.cr._load_image(image_path)
    first_cx, last_cx, cy, n_cells, cell_size = 600.0, 1580.0, 880.0, 9, 120
    seat_up_extra_px, seat_down_extra_px, seat_lr_extra_px, seat_left_extra_px = 70, 30, 30, 20
    half = cell_size / 2.0
    custom_roi = (
        int(round(min(first_cx, last_cx) - half - seat_lr_extra_px - seat_left_extra_px)),
        int(round(cy - half - seat_up_extra_px)),
        int(round(max(first_cx, last_cx) + half + seat_lr_extra_px)),
        int(round(cy + half + seat_down_extra_px)),
    )
    old_roi = pb.cr.ROI
    try:
        pb.cr.ROI = custom_roi
        out_json = pb.cr.run_recognition(
            image_path=image_path,
            template_path=template,
            piece_dir=PROJECT_DIR / "chess_gallery",
            output_dir=temp_out,
            circle_diameter=84,
            embed_backbone="dinov2_vits14",
            model=model,
            device=device,
            transform=transform,
            piece_db=piece_db,
            alpha_tight=True,
        )
        results = out_json.get("results") or []
    finally:
        pb.cr.ROI = old_roi
    pieces_by_cell = pb._assign_piece_bars_to_cells(
        results,
        n_cells=n_cells,
        first_cx=first_cx,
        step_x=(last_cx - first_cx) / max(1, n_cells - 1),
    )
    equip_by_cell: Dict[int, Optional[Dict[str, Any]]] = {}
    for idx, piece_obj in pieces_by_cell.items():
        bar_box = piece_obj.get("bar_box") or [0, 0, 0, 0]
        equip_by_cell[idx] = pb._detect_one_bar_equip_top1(
            scene_bgr,
            bar_box_xywh=bar_box,
            templates=equip_templates,
            scales=scales,
            method=cv2.TM_CCOEFF_NORMED,
            threshold=0.78,
            max_peaks_per_scale=4,
            top_k=15,
            nms_iou=0.35,
            below_px=2,
            crop_w=120,
            crop_h=50,
            min_roi=min_roi,
            blue_buff_gap_min=0.05,
        )
    vis = pb._draw_preboard_overlay(
        scene_bgr,
        coverage_roi=custom_roi,
        n_cells=n_cells,
        cell_size=cell_size,
        pieces_by_cell=pieces_by_cell,
        equip_by_cell=equip_by_cell,
        font_size=16,
    )
    return {
        "vis": vis,
        "summary": {
            "file": image_path.name,
            "coverage_roi": list(custom_roi),
            "pieces_by_cell": pieces_by_cell,
            "equip_by_cell": equip_by_cell,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="一体化 battle pipeline：单进程逐图识别")
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mode", type=str, choices=["A", "B"], default="B")
    ap.add_argument(
        "--group-pattern",
        type=str,
        default=r"^(\d+)[-_]",
        help="stem 提取组号；01-a 与 01_b 同属组 01；纯数字 stem 不匹配时用整 stem 作为组键",
    )
    ap.add_argument("--no-cache", action="store_true", help="禁用模块缓存（默认开启）")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / ".pipeline_cache", help="缓存目录")
    args = ap.parse_args()

    images = _iter_images(args.img_dir.resolve())
    out_root = args.out.resolve().parent / f"{args.out.resolve().name}_{args.mode}"
    if out_root.exists():
        import shutil

        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_dir.resolve()
    use_cache = not bool(args.no_cache)
    if use_cache:
        (cache_root / "fight").mkdir(parents=True, exist_ok=True)
        (cache_root / "pre").mkdir(parents=True, exist_ok=True)
        (cache_root / "player").mkdir(parents=True, exist_ok=True)

    # 一次性初始化（关键：避免每图重复准备）
    template = fb.br.find_healthbar_template(PROJECT_DIR)
    model, device, transform = fb.cr._get_embedding_model("dinov2_vits14")
    piece_db, _ = fb.cr.load_or_build_piece_embedding_db(
        PROJECT_DIR / "chess_gallery",
        model,
        device,
        transform,
        embed_backbone="dinov2_vits14",
        root=PROJECT_DIR,
        force_rebuild=False,
        verbose=True,
    )
    equip_templates = fb.er._build_templates(PROJECT_DIR / "equip_gallery")
    scales = [24, 25, 26, 27, 28]
    min_roi = max(scales)

    # player OCR 引擎只创建一次
    centers, rects = pr._default_layout()
    ocr_engine = pr._create_ocr_engine(verbose=False)

    # A 模式聚合容器
    fight_by_stem: Dict[str, Dict[str, Any]] = {}
    player_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="battle_pipeline_unified_tmp_") as td:
        tmp = Path(td)
        t0_init = time.perf_counter()
        print("[TIME] 初始化模型与资源...")
        print("[TIME]  - 初始化开始")
        # 已在前面初始化
        t1_init = time.perf_counter()
        print(f"[TIME]  - 初始化完成: {t1_init - t0_init:.2f}s")
        for image_path in images:
            stem = image_path.stem
            t0_img = time.perf_counter()
            print(f"[TIME] [{image_path.name}] 开始")
            t0 = time.perf_counter()
            img_sha1 = _file_sha1(image_path)
            t1 = time.perf_counter()
            print(f"[TIME] [{image_path.name}] 计算hash: {t1 - t0:.2f}s")

            t0 = time.perf_counter()
            fight_params_key = "fight:cd84:eq0.78:sc24-28:nms0.35"
            fight_cache_key = _cache_key("fight", img_sha1, fight_params_key)
            fight = _load_module_cache(cache_root / "fight", fight_cache_key) if use_cache else None
            if fight is not None:
                print(f"[CACHE] [{image_path.name}] fight hit")
            else:
                print(f"[CACHE] [{image_path.name}] fight miss")
                fight = _run_one_fightboard(
                    image_path=image_path,
                    template=template,
                    model=model,
                    device=device,
                    transform=transform,
                    piece_db=piece_db,
                    equip_templates=equip_templates,
                    scales=scales,
                    min_roi=min_roi,
                    temp_out=tmp / f"{stem}_fight",
                )
                if use_cache:
                    _save_module_cache(cache_root / "fight", fight_cache_key, fight)
            t1 = time.perf_counter()
            print(f"[TIME] [{image_path.name}] fightboard: {t1 - t0:.2f}s")

            t0 = time.perf_counter()
            pre_params_key = "pre:cd84:eq0.78:sc24-28:nms0.35:seat-default"
            pre_cache_key = _cache_key("pre", img_sha1, pre_params_key)
            pre = _load_module_cache(cache_root / "pre", pre_cache_key) if use_cache else None
            if pre is not None:
                print(f"[CACHE] [{image_path.name}] pre hit")
            else:
                print(f"[CACHE] [{image_path.name}] pre miss")
                pre = _run_one_preboard(
                    image_path=image_path,
                    template=template,
                    model=model,
                    device=device,
                    transform=transform,
                    piece_db=piece_db,
                    equip_templates=equip_templates,
                    scales=scales,
                    min_roi=min_roi,
                    temp_out=tmp / f"{stem}_pre",
                )
                if use_cache:
                    _save_module_cache(cache_root / "pre", pre_cache_key, pre)
            t1 = time.perf_counter()
            print(f"[TIME] [{image_path.name}] preboard: {t1 - t0:.2f}s")

            t0 = time.perf_counter()
            player_params_key = "player:font18:layout-default"
            player_cache_key = _cache_key("player", img_sha1, player_params_key)
            player_cached = _load_module_cache(cache_root / "player", player_cache_key) if use_cache else None
            if player_cached is not None:
                print(f"[CACHE] [{image_path.name}] player hit")
                player_vis = player_cached["vis"]
                player_summary = player_cached["summary"]
            else:
                print(f"[CACHE] [{image_path.name}] player miss")
                player_vis, player_summary = pr.process_image(ocr_engine, image_path, centers, rects, 18)
                if use_cache:
                    _save_module_cache(
                        cache_root / "player",
                        player_cache_key,
                        {"vis": player_vis, "summary": player_summary},
                    )
            t1 = time.perf_counter()
            print(f"[TIME] [{image_path.name}] player: {t1 - t0:.2f}s")

            t0 = time.perf_counter()
            final_img = _draw_text_panel(
                fight["vis"],
                _summary_lines_from_modules(fight["summary"], pre["summary"], player_summary),
            )
            final_img_path = out_root / f"{stem}_annotated.png"
            _save_image(final_img_path, final_img)
            t1 = time.perf_counter()
            print(f"[TIME] [{image_path.name}] 绘制与保存图片: {t1 - t0:.2f}s")

            merged = {
                "mode": args.mode,
                "file": image_path.name,
                "annotated_image": final_img_path.name,
                "modules": {
                    "fightboard": fight["summary"],
                    "preboard": pre["summary"],
                    "player": player_summary,
                },
                "analysis": {"cross_validation": None},
                "confirmed_fightboard_results": (fight["summary"] or {}).get("results", []),
            }
            summary_path = out_root / f"{stem}_summary.json"
            t0 = time.perf_counter()
            summary_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
            t1 = time.perf_counter()
            print(f"[TIME] [{image_path.name}] 写summary: {t1 - t0:.2f}s")
            print(f"[OK] final -> {final_img_path.name}, {summary_path.name}")
            t1_img = time.perf_counter()
            print(f"[TIME] [{image_path.name}] 总耗时: {t1_img - t0_img:.2f}s")

            if args.mode == "A":
                gk = _group_key(stem, args.group_pattern)
                fight_by_stem[stem] = fight["summary"]
                player_by_group[gk].append(player_summary)

    if args.mode == "A":
        import trait_cross_validate as tcv

        # 交叉验证：与 trait_cross_validate 相同，bond_items 羁绊名按图鉴归一后再聚合
        legend_trait_names = tcv._load_legend_trait_names(PROJECT_DIR / "data" / "rag_legend_traits.jsonl")
        group_traits_map: Dict[str, Dict[str, Any]] = {}
        for gk, plist in player_by_group.items():
            entries = [(str(p.get("file") or "unknown"), p) for p in sorted(plist, key=lambda x: str(x.get("file") or ""))]
            group_traits_map[gk] = tcv._aggregate_bonds_from_player_entries(
                entries,
                legend_trait_names=legend_trait_names,
            )

        for image_path in images:
            stem = image_path.stem
            gk = _group_key(stem, args.group_pattern)
            fsum = fight_by_stem.get(stem, {})
            gtraits = group_traits_map.get(
                gk,
                {
                    "trait_count_max": {},
                    "trait_sources": {},
                    "raw_items": [],
                    "trait_canonicalization_log": [],
                },
            )
            cv = tcv._apply_cross_validation_rules(
                fightboard_summary=fsum,
                group_player_traits=gtraits,
                legend_chess_path=(PROJECT_DIR / "data" / "rag_legend_chess.jsonl"),
                legend_traits_path=(PROJECT_DIR / "data" / "rag_legend_traits.jsonl"),
                legend_equip_path=(PROJECT_DIR / "data" / "rag_legend_equip.jsonl"),
            )
            sp = out_root / f"{stem}_summary.json"
            js = _read_json_if_exists(sp) or {}
            js["analysis"] = {"cross_validation": cv}
            js["confirmed_fightboard_results"] = cv.get("confirmed_results") or js.get("confirmed_fightboard_results") or []
            sp.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成。输出目录: {out_root}")


if __name__ == "__main__":
    main()

