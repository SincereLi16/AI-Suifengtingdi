# -*- coding: utf-8 -*-
"""
battle_pipeline_v3：在 v2 基础上加速 chess_recog 的 torch_bar_loop，并优先使用 CUDA 推理。

相对 battle_pipeline_v2：
- chess_recog.run_recognition：``batch_embed=True`` 时 Stage1/Stage2 各用一次 batched ViT 前向（同模型、同逻辑）；
  默认 ``save_debug_crops=False`` 跳过 crops PNG 落盘。
- 初始化时 ``torch.backends.cudnn.benchmark = True``（224 固定输入），并打印当前 CUDA 设备名。
- 其余流程（equip_column、player OCR、TCV、主图出图）与 v2 一致。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import battle_pipeline as bp
import fightboard_recog as fb
import player_recog as pr
import preboard_recog as pb
import trait_cross_validate as tcv
from element_recog import equip_column_recog as ecr

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_OUT = PROJECT_DIR / "battle_pipeline_v3_out"


def _configure_torch_cuda_for_v3() -> None:
    """固定输入尺寸下启用 cuDNN autotune，并打印推理设备（无 CUDA 时仍可用 CPU + batch）。"""
    import torch

    if not torch.cuda.is_available():
        print("[V3] CUDA 不可用，将使用 CPU（batch 嵌入仍可减少 ViT 前向次数）")
        return
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        print(f"[V3] CUDA 推理: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("[V3] CUDA 可用")


def _pil_text_width(font: Any, text: str) -> float:
    try:
        return float(font.getlength(text))
    except Exception:
        bbox = font.getbbox(text)
        return float(bbox[2] - bbox[0])


def _wrap_line_to_width(text: str, font: Any, max_width: float) -> List[str]:
    """按像素宽度折行（中文逐字累加），避免单行过长溢出面板。"""
    if not text:
        return [""]
    if _pil_text_width(font, text) <= max_width:
        return [text]
    out: List[str] = []
    cur = ""
    for ch in text:
        trial = cur + ch
        if _pil_text_width(font, trial) <= max_width:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out if out else [text]


def _draw_summary_panel(base: np.ndarray, lines: List[str], *, title: str = "对局汇总") -> np.ndarray:
    """右侧汇总面板：标题与正文均用 PIL 绘中文（勿用 cv2.putText 写中文，否则会乱码）。"""
    h, w = base.shape[:2]
    panel_w = max(620, int(w * 0.42))
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, h - 1), (90, 90, 90), 1)
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = None
        font_title = None
        for fp in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
            if fp.exists():
                try:
                    font = ImageFont.truetype(str(fp), 18)
                    font_title = ImageFont.truetype(str(fp), 20)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        if font_title is None:
            font_title = font

        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        dr = ImageDraw.Draw(pil)
        max_text_w = float(panel_w - 24)
        line_height = 22
        title_y = 12
        dr.text((12, title_y), title, font=font_title, fill=(255, 255, 0))
        y = title_y + 34
        for ln in lines:
            if y > h - line_height:
                break
            for sub in _wrap_line_to_width(ln, font, max_text_w):
                if y > h - line_height:
                    break
                dr.text((12, y), sub, font=font, fill=(255, 255, 255))
                y += line_height
        panel = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        y = 12
        cv2.putText(panel, "Summary", (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        y = 50
        for ln in lines:
            if y > h - 10:
                break
            cv2.putText(panel, ln[:120], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 23
    return np.hstack([base, panel])


def _format_board_position(pos_obj: Any) -> str:
    if isinstance(pos_obj, dict):
        cr = pos_obj.get("cell_row")
        cc = pos_obj.get("cell_col")
        if cr is not None and cc is not None:
            return f"{int(cr)},{int(cc)}"
        lab = pos_obj.get("label")
        if lab:
            return str(lab)
    return "?"


def _fightboard_pipe_lines(fight_json: Optional[Dict[str, Any]]) -> List[str]:
    """棋盘：棋子名称 | 位置 | 所携带装备"""
    lines: List[str] = []
    fr = (fight_json or {}).get("results") or []
    equip_by_bar = (fight_json or {}).get("equip_by_bar") or {}
    if not fr:
        lines.append("(无棋子)")
        return lines
    for r in fr:
        bi = int((r or {}).get("bar_index") or 0)
        name = str((r or {}).get("best") or "?")
        if str((r or {}).get("confidence") or "") == "low":
            name = f"{name}?"
        pos = _format_board_position((r or {}).get("position"))
        eq_list = equip_by_bar.get(str(bi)) if isinstance(equip_by_bar, dict) else None
        if eq_list is None and isinstance(equip_by_bar, dict):
            eq_list = equip_by_bar.get(bi)
        equip_names: List[str] = []
        if isinstance(eq_list, list):
            for e in eq_list:
                if isinstance(e, dict):
                    en = str(e.get("name") or "").strip()
                    if en:
                        equip_names.append(en)
        eq_str = "、".join(equip_names) if equip_names else "无"
        lines.append(f"{name} | {pos} | {eq_str}")
    return lines


def _preboard_lines(pre_json: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    pc = (pre_json or {}).get("pieces_by_cell") or {}
    eqc = (pre_json or {}).get("equip_by_cell") or {}
    pre_keys = sorted(pc.keys(), key=lambda x: bp._safe_int(x, 0))
    if not pre_keys:
        lines.append("(无棋子)")
        return lines
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
        eq_str = en if en else "无"
        lines.append(f"{nm} | {k} | {eq_str}")
    return lines


def _equip_column_text_lines(equip_col: Optional[Dict[str, Any]]) -> List[str]:
    m = (equip_col or {}).get("matches") if isinstance(equip_col, dict) else None
    if not isinstance(m, list) or not m:
        return ["(无)"]
    names: List[str] = []
    for x in m:
        if not isinstance(x, dict):
            continue
        n = str(x.get("name_stem") or "").strip()
        if n:
            names.append(n)
    if not names:
        return ["(无)"]
    return ["、".join(names)]


def _clean_phase_display(parsed: str) -> str:
    """阶段展示：去掉 OCR 里多余的「总」字，保留如 2-1。"""
    s = (parsed or "").strip()
    if not s:
        return ""
    return re.sub(r"总", "", s)


def _format_winloss_streak(parsed: str) -> str:
    """胜负：无 / n连胜 / n连败（由 player streak 解析结果转换）。"""
    s = (parsed or "").strip()
    if not s or s == "无连胜/连败" or ("无" in s and "连胜" not in s and "连败" not in s):
        return "无"
    m = re.search(r"(\d+)\s*连胜|连胜\s*(\d+)", s)
    if m:
        n = m.group(1) or m.group(2)
        return f"{n}连胜"
    m = re.search(r"(\d+)\s*连败|连败\s*(\d+)", s)
    if m:
        n = m.group(1) or m.group(2)
        return f"{n}连败"
    if re.search(r"连胜", s):
        m2 = re.search(r"(\d+)", s)
        return f"{m2.group(1)}连胜" if m2 else "无"
    if re.search(r"连败", s):
        m2 = re.search(r"(\d+)", s)
        return f"{m2.group(1)}连败" if m2 else "无"
    return "无"


def _situation_lines(player_json: Optional[Dict[str, Any]]) -> List[str]:
    """局势：阶段、等级、经验、金币、胜负、玩家血量。"""
    lines: List[str] = []
    fields = ((player_json or {}).get("fields") or {}) if isinstance(player_json, dict) else {}
    if not isinstance(fields, dict):
        return lines
    for k in ("phase", "level", "exp", "gold"):
        d = fields.get(k)
        if isinstance(d, dict):
            pv = str(d.get("parsed") or "").strip()
            if not pv:
                continue
            if k == "phase":
                pv = _clean_phase_display(pv)
            nm = str(d.get("name") or k)
            lines.append(f"{nm}: {pv}")
    streak_d = fields.get("streak")
    if isinstance(streak_d, dict):
        sp = str(streak_d.get("parsed") or "").strip()
        lines.append(f"胜负: {_format_winloss_streak(sp)}")
    hp_nick = fields.get("hp_nick")
    if isinstance(hp_nick, dict):
        cells = hp_nick.get("player_cells") or []
        if isinstance(cells, list) and cells:
            lines.append("玩家与血量:")
            for c in cells:
                if not isinstance(c, dict):
                    continue
                pid = str(c.get("id_text") or "").strip() or "?"
                hpv = str(c.get("hp") or "").strip() or "?"
                lines.append(f"  {pid} 血量 {hpv}")
        else:
            parsed = hp_nick.get("parsed")
            if isinstance(parsed, list) and parsed:
                lines.append("玩家与血量:")
                for ln in parsed:
                    lines.append(f"  {ln}")
            elif isinstance(parsed, str) and parsed.strip():
                lines.append(f"玩家与血量: {parsed.strip()}")
    return lines


def _cv_analysis_lines_v2_slim(cv: Dict[str, Any]) -> List[str]:
    """TCV 摘要：不含「棋盘羁绊·英雄/装备赐/合计」三行。"""
    lines: List[str] = []
    lines.append(f"trait_loss={cv.get('trait_loss')}  consistent={cv.get('consistent')}  status={cv.get('status')}")
    diff = cv.get("trait_diff") or {}
    if diff:
        lines.append("差异(仅不一致):")
        for t, d in sorted(diff.items(), key=lambda kv: kv[0]):
            lines.append(
                f"  {t}: 期望={d.get('expected')} 盘={d.get('board')} Δ={d.get('delta_board_minus_expected')}"
            )
    for ln in cv.get("inference_notes") or []:
        lines.append(str(ln))
    for c in cv.get("changes") or []:
        lines.append(f"修正: bar{c.get('bar_index')} {c.get('from')}=>{c.get('to')}")
    return lines


def _bond_lines_and_tcv(
    group_traits: Dict[str, Any],
    cv: Dict[str, Any],
) -> List[str]:
    """羁绊：种类数、n羁绊名（如 1弗雷尔卓德）、TCV（精简）。"""
    lines: List[str] = []
    tm = group_traits.get("trait_count_max") or {}
    if isinstance(tm, dict) and tm:
        lines.append(f"种类数: {len(tm)}")
        bond_parts = [f"{int(n)}{t}" for t, n in sorted(tm.items(), key=lambda kv: kv[0])]
        lines.append("羁绊: " + "、".join(bond_parts))
    else:
        lines.append("种类数: 0")
        lines.append("羁绊: (无)")
    lines.append("")
    lines.extend(_cv_analysis_lines_v2_slim(cv))
    return lines


def _summary_lines_v3_primary(
    *,
    fight_json: Optional[Dict[str, Any]],
    pre_json: Optional[Dict[str, Any]],
    player_json_main: Optional[Dict[str, Any]],
    equip_col_json: Optional[Dict[str, Any]],
    group_traits: Dict[str, Any],
    cv: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    lines.append("battle_pipeline_v3 · 主图")
    lines.append("")
    lines.append("【羁绊】")
    lines.extend(_bond_lines_and_tcv(group_traits, cv))
    lines.append("")
    lines.append("【棋盘】")
    lines.extend(_fightboard_pipe_lines(fight_json))
    lines.append("")
    lines.append("【备战】")
    lines.extend(_preboard_lines(pre_json))
    lines.append("")
    lines.append("【装备】")
    lines.extend(_equip_column_text_lines(equip_col_json))
    lines.append("")
    lines.append("【局势】")
    lines.extend(_situation_lines(player_json_main))
    return lines


def _run_one_fightboard_profiled(
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
    tag: str,
    batch_embed: bool = True,
    save_debug_crops: bool = False,
    cross_bar_stage1_batch_embed: bool = True,
) -> Dict[str, Any]:
    """与 battle_pipeline._run_one_fightboard 等价，终端输出分步耗时与统计。"""
    t0 = time.perf_counter()
    scene_bgr = fb.cr._load_image(image_path)
    t1 = time.perf_counter()
    print(f"  [{tag}] fightboard-1 加载图像: {t1 - t0:.4f}s  size={scene_bgr.shape[1]}x{scene_bgr.shape[0]}")

    t0 = time.perf_counter()
    cr_profile: Dict[str, float] = {}
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
        profile_timings=cr_profile,
        batch_embed=batch_embed,
        save_debug_crops=save_debug_crops,
        cross_bar_stage1_batch_embed=cross_bar_stage1_batch_embed,
    )
    t1 = time.perf_counter()
    results = out_json.get("results") or []
    n_bars = len(results)
    samples = out_json.get("samples")
    n_samp = len(samples) if isinstance(samples, list) else 0
    if not batch_embed:
        be = "逐枚前向"
    elif cross_bar_stage1_batch_embed:
        be = "batch_embed+跨血条Stage1"
    else:
        be = "batch_embed"
    print(
        f"  [{tag}] fightboard-2 chess_recog.run_recognition（{be}；血条检测+嵌入匹配）: "
        f"{t1 - t0:.4f}s  血条数={n_bars}  采样圆点数={n_samp}"
    )
    _cr_order = [
        "load_scene_templates",
        "detect_healthbars",
        "prepare_model_and_db",
        "mark_roi_tiles",
        "prefetch_cross_bar_stage1",
        "torch_bar_loop",
        "position_mapping",
        "save_outputs",
        "total",
    ]
    for _k in _cr_order:
        if _k in cr_profile:
            print(f"      └ {_k}: {cr_profile[_k]:.4f}s")

    equip_by_bar: Dict[int, List[Dict[str, Any]]] = {}
    t_eq0 = time.perf_counter()
    total_eq = 0
    for i, r in enumerate(results):
        t_b0 = time.perf_counter()
        bi = int(r.get("bar_index") or 0)
        bar_box = r.get("bar_box") or [0, 0, 0, 0]
        eq_list = fb._detect_one_bar_equip(
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
        equip_by_bar[bi] = eq_list
        total_eq += len(eq_list)
        t_b1 = time.perf_counter()
        print(
            f"  [{tag}] fightboard-3 血条下方装备ROI 条{i + 1}/{n_bars} bar_index={bi} "
            f"裁剪+模板匹配 装备数={len(eq_list)} 耗时={t_b1 - t_b0:.4f}s"
        )
    t_eq1 = time.perf_counter()
    print(f"  [{tag}] fightboard-3 小计（全部血条装备）: {t_eq1 - t_eq0:.4f}s  装备条目合计={total_eq}")

    t0 = time.perf_counter()
    vis = fb._overlay_fightboard(scene_bgr=scene_bgr, results=results, equip_by_bar=equip_by_bar, font_size=16)
    t1 = time.perf_counter()
    print(f"  [{tag}] fightboard-4 叠加棋子/血条/装备标注: {t1 - t0:.4f}s")

    return {"vis": vis, "summary": {"file": image_path.name, "results": results, "equip_by_bar": equip_by_bar}}


def _run_one_preboard_profiled(
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
    tag: str,
    batch_embed: bool = True,
    save_debug_crops: bool = False,
    cross_bar_stage1_batch_embed: bool = True,
) -> Dict[str, Any]:
    """与 battle_pipeline._run_one_preboard 等价，终端输出分步耗时。"""
    t0 = time.perf_counter()
    scene_bgr = pb.cr._load_image(image_path)
    t1 = time.perf_counter()
    print(f"  [{tag}] preboard-1 加载图像: {t1 - t0:.4f}s")

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
        t0 = time.perf_counter()
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
            batch_embed=batch_embed,
            save_debug_crops=save_debug_crops,
            cross_bar_stage1_batch_embed=cross_bar_stage1_batch_embed,
        )
        t1 = time.perf_counter()
        results = out_json.get("results") or []
        n_bars = len(results)
        samples = out_json.get("samples")
        n_samp = len(samples) if isinstance(samples, list) else 0
        if not batch_embed:
            pre_be = "逐枚前向"
        elif cross_bar_stage1_batch_embed:
            pre_be = "batch+跨血条Stage1"
        else:
            pre_be = "batch"
        print(
            f"  [{tag}] preboard-2 chess_recog.run_recognition（备战区 ROI · {pre_be}）: "
            f"{t1 - t0:.4f}s  检测血条数={n_bars}  采样圆点={n_samp}"
        )
    finally:
        pb.cr.ROI = old_roi

    t0 = time.perf_counter()
    pieces_by_cell = pb._assign_piece_bars_to_cells(
        results,
        n_cells=n_cells,
        first_cx=first_cx,
        step_x=(last_cx - first_cx) / max(1, n_cells - 1),
    )
    t1 = time.perf_counter()
    n_occupied = sum(1 for v in pieces_by_cell.values() if str((v or {}).get("best") or "").strip())
    print(f"  [{tag}] preboard-3 映射到备战格: {t1 - t0:.4f}s  有棋子格数={n_occupied}/{n_cells}")

    equip_by_cell: Dict[int, Optional[Dict[str, Any]]] = {}
    t_eq0 = time.perf_counter()
    for idx, piece_obj in sorted(pieces_by_cell.items(), key=lambda kv: bp._safe_int(kv[0], 0)):
        t_b0 = time.perf_counter()
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
        t_b1 = time.perf_counter()
        eq_one = equip_by_cell[idx]
        en = str((eq_one or {}).get("name") or "").strip() if isinstance(eq_one, dict) else ""
        print(f"  [{tag}] preboard-4 格{idx} 装备top1: {t_b1 - t_b0:.4f}s  name={en or '(无)'}")
    t_eq1 = time.perf_counter()
    print(f"  [{tag}] preboard-4 小计（全部格）: {t_eq1 - t_eq0:.4f}s")

    t0 = time.perf_counter()
    vis = pb._draw_preboard_overlay(
        scene_bgr,
        coverage_roi=custom_roi,
        n_cells=n_cells,
        cell_size=cell_size,
        pieces_by_cell=pieces_by_cell,
        equip_by_cell=equip_by_cell,
        font_size=16,
    )
    t1 = time.perf_counter()
    print(f"  [{tag}] preboard-5 叠加绘制（仅用于 JSON/调试，主图输出不用）: {t1 - t0:.4f}s")

    return {
        "vis": vis,
        "summary": {
            "file": image_path.name,
            "coverage_roi": list(custom_roi),
            "pieces_by_cell": pieces_by_cell,
            "equip_by_cell": equip_by_cell,
        },
    }


def _ensure_fight_pre_runtime(
    template_holder: List[Any],
    model_holder: List[Any],
    device_holder: List[Any],
    transform_holder: List[Any],
    piece_db_holder: List[Any],
    equip_templates_holder: List[Any],
) -> None:
    if model_holder[0] is not None:
        return
    print("[V3] 初始化 fightboard / preboard 模型…")
    _configure_torch_cuda_for_v3()
    template_holder[0] = fb.br.find_healthbar_template(PROJECT_DIR)
    try:
        model, device, transform = fb.cr._get_embedding_model("dinov2_vits14")
    except OSError as ex:
        msg = str(ex)
        if "shm.dll" in msg or "WinError 127" in msg or "找不到指定的程序" in msg:
            raise SystemExit(
                "无法加载 PyTorch（fightboard 依赖 DINOv2/torch）。\n"
                "常见原因：缺少 VC++ 运行库或 torch 与系统不匹配。\n"
                "可尝试：\n"
                "  1) 安装 Microsoft Visual C++ 2015–2022 Redistributable (x64)\n"
                "  2) 按 pytorch.org 选择与当前 Python 版本一致的 CPU/CUDA 轮子重装 torch\n"
                "  3) 在项目专用 venv 中重新 pip install torch torchvision\n"
                f"\n原始错误: {msg}"
            ) from ex
        raise
    model_holder[0] = model
    device_holder[0] = device
    transform_holder[0] = transform
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
    piece_db_holder[0] = piece_db
    equip_templates_holder[0] = fb.er._build_templates(PROJECT_DIR / "equip_gallery")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="battle_pipeline_v3：batch 嵌入 + CUDA 优先；fight+pre+player+equip_column + trait cross validate（仅主图出图）"
    )
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--group-pattern",
        type=str,
        default=r"^(\d+)[-_]",
        help="stem 提取组号；01-a 与 01-b 同属组 01",
    )
    ap.add_argument(
        "--fightboard-stem-suffix",
        type=str,
        default="a",
        help="仅主图 stem（-a/_a 或纯数字）跑 fight/pre/equip_column 全链并出图",
    )
    ap.add_argument(
        "--no-batch-embed",
        action="store_true",
        help="关闭 chess_recog 批量前向（与 v2 逐枚 forward 行为接近，用于对比）",
    )
    ap.add_argument(
        "--save-debug-crops",
        action="store_true",
        help="写出 chess_recog crops 目录下的调试图 PNG（较慢）",
    )
    ap.add_argument(
        "--no-cross-bar-stage1",
        action="store_true",
        help="关闭 Stage1 跨血条合并前向（仍可用逐条 batch_embed，用于对比速度与结果）",
    )
    args = ap.parse_args()
    batch_embed = not bool(args.no_batch_embed)
    save_debug_crops = bool(args.save_debug_crops)
    cross_bar_stage1 = not bool(args.no_cross_bar_stage1)

    img_dir = args.img_dir.resolve()
    out_root = args.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    images = bp._iter_images(img_dir)
    if not images:
        raise SystemExit(f"输入目录无图像: {img_dir}")

    target_w, target_h = tcv._infer_reference_resolution(PROJECT_DIR)
    suffix = (args.fightboard_stem_suffix or "").strip()

    scales_bar = [24, 25, 26, 27, 28]
    min_roi_bar = max(scales_bar)
    scales_col = (60, 61, 62, 63, 64)

    template_box: List[Any] = [None]
    model_box: List[Any] = [None]
    device_box: List[Any] = [None]
    transform_box: List[Any] = [None]
    piece_db_box: List[Any] = [None]
    equip_templates_box: List[Any] = [None]

    need_fight = any(tcv._stem_matches_fightboard_suffix(p.stem, suffix) for p in images)
    if not need_fight:
        print("[V3] 当前输入无主图（如 -a），将仅跑 player/equip_column，不进行 fightboard 交叉验证。")

    if need_fight:
        print("[V3] 预加载 PyTorch（先于 Paddle OCR，避免 Windows DLL 冲突）…")
        _ensure_fight_pre_runtime(template_box, model_box, device_box, transform_box, piece_db_box, equip_templates_box)

    print("[V3] 初始化 OCR…")
    centers, rects = pr._default_layout()
    ocr_engine = pr._create_ocr_engine(verbose=False)

    col_templates = fb.er._build_templates(PROJECT_DIR / "equip_gallery")
    if not col_templates:
        raise SystemExit(f"装备图鉴为空: {PROJECT_DIR / 'equip_gallery'}")

    t0_all = time.perf_counter()
    print(f"[V3] 初始化完成 ({t0_all:.1f}s 起算)")

    fight_by_stem: Dict[str, Dict[str, Any]] = {}
    pre_by_stem: Dict[str, Dict[str, Any]] = {}
    primary_fight_vis_by_stem: Dict[str, Any] = {}
    equip_column_by_stem: Dict[str, Dict[str, Any]] = {}
    player_by_stem: Dict[str, Dict[str, Any]] = {}
    player_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="bp_v3_tmp_") as td:
        tmp = Path(td)
        for image_path in images:
            stem = image_path.stem
            src_w, src_h = tcv._image_size(image_path)
            used_path = image_path
            if (src_w, src_h) != (target_w, target_h):
                used_path = tmp / f"{stem}__norm_{target_w}x{target_h}.png"
                t_n0 = time.perf_counter()
                tcv._normalize_image_to_size(image_path, used_path, target_w, target_h)
                t_n1 = time.perf_counter()
                print(
                    f"[V3] 尺寸对齐 {image_path.name}: {src_w}x{src_h} -> {target_w}x{target_h} "
                    f"耗时={t_n1 - t_n0:.4f}s"
                )

            run_fight = tcv._stem_matches_fightboard_suffix(stem, suffix)
            if run_fight:
                print(f"[V3] 处理 {image_path.name}（主图 · fight + pre + player + equip_column）…")
            else:
                print(f"[V3] 处理 {image_path.name}（辅图 · player + equip_column）…")

            t_img = time.perf_counter()
            t_h0 = time.perf_counter()
            fp_digest = tcv._file_sha1(image_path) + f"|norm:{target_w}x{target_h}|src:{src_w}x{src_h}"
            print(f"  [{image_path.name}] 文件指纹(SHA1): {time.perf_counter() - t_h0:.4f}s  digest={fp_digest[:16]}…")

            fight_summary: Optional[Dict[str, Any]] = None
            pre_summary: Optional[Dict[str, Any]] = None
            fight_vis = None

            if run_fight:
                _ensure_fight_pre_runtime(template_box, model_box, device_box, transform_box, piece_db_box, equip_templates_box)
                assert template_box[0] is not None and piece_db_box[0] is not None and equip_templates_box[0] is not None

                fight = _run_one_fightboard_profiled(
                    image_path=used_path,
                    template=template_box[0],
                    model=model_box[0],
                    device=device_box[0],
                    transform=transform_box[0],
                    piece_db=piece_db_box[0],
                    equip_templates=equip_templates_box[0],
                    scales=scales_bar,
                    min_roi=min_roi_bar,
                    temp_out=tmp / f"{stem}_fight",
                    tag=image_path.name,
                    batch_embed=batch_embed,
                    save_debug_crops=save_debug_crops,
                    cross_bar_stage1_batch_embed=cross_bar_stage1,
                )
                if isinstance(fight.get("summary"), dict):
                    fight["summary"]["file"] = image_path.name
                fight_summary = fight["summary"]
                fight_vis = fight["vis"]
                fight_by_stem[stem] = fight_summary
                primary_fight_vis_by_stem[stem] = fight_vis

                pre = _run_one_preboard_profiled(
                    image_path=used_path,
                    template=template_box[0],
                    model=model_box[0],
                    device=device_box[0],
                    transform=transform_box[0],
                    piece_db=piece_db_box[0],
                    equip_templates=equip_templates_box[0],
                    scales=scales_bar,
                    min_roi=min_roi_bar,
                    temp_out=tmp / f"{stem}_pre",
                    tag=image_path.name,
                    batch_embed=batch_embed,
                    save_debug_crops=save_debug_crops,
                    cross_bar_stage1_batch_embed=cross_bar_stage1,
                )
                if isinstance(pre.get("summary"), dict):
                    pre["summary"]["file"] = image_path.name
                pre_summary = pre["summary"]
                pre_by_stem[stem] = pre_summary

            t_sc0 = time.perf_counter()
            scene_bgr = fb.cr._load_image(used_path)
            t_sc1 = time.perf_counter()
            print(f"  [{image_path.name}] 加载整图（equip_column 用）: {t_sc1 - t_sc0:.4f}s")
            t_eq0 = time.perf_counter()
            eq_js = ecr.compute_equip_column_matches(
                scene_bgr,
                templates=col_templates,
                equip_roi=ecr.DEFAULT_EQUIP_ROI,
                scales=scales_col,
                threshold=0.6,
                max_peaks_per_scale=4,
                top_k=15,
                nms_iou=0.35,
                blue_buff_gap_min=0.05,
                blue_buff_single_min_score=0.78,
                grid_cols=2,
                grid_rows=10,
            )
            t_eq1 = time.perf_counter()
            mc = int((eq_js or {}).get("match_count") or 0)
            print(
                f"  [{image_path.name}] equip_column 竖条ROI切块 grid=2×10 匹配条目数={mc} "
                f"耗时={t_eq1 - t_eq0:.4f}s"
            )
            equip_column_by_stem[stem] = eq_js

            t_pl0 = time.perf_counter()
            _, player_summary = pr.process_image(ocr_engine, used_path, centers, rects, 18)
            t_pl1 = time.perf_counter()
            print(f"  [{image_path.name}] player 全 ROI OCR/解析: {t_pl1 - t_pl0:.4f}s")

            player_summary = dict(player_summary)
            player_summary["file"] = image_path.name
            player_by_stem[stem] = player_summary
            gk = tcv._extract_group_key(stem, args.group_pattern)
            player_by_group[gk].append(player_summary)

            print(f"  [V3] 本图合计耗时: {time.perf_counter() - t_img:.4f}s  ({image_path.name})")

    legend_trait_names = tcv._load_legend_trait_names(PROJECT_DIR / "data" / "rag_legend_traits.jsonl")
    group_traits_map: Dict[str, Dict[str, Any]] = {}
    for gk, plist in player_by_group.items():
        entries = [(str(p.get("file") or "unknown"), p) for p in sorted(plist, key=lambda x: str(x.get("file") or ""))]
        group_traits_map[gk] = tcv._aggregate_bonds_from_player_entries(entries, legend_trait_names=legend_trait_names)

    legend_chess = PROJECT_DIR / "data" / "rag_legend_chess.jsonl"
    legend_traits = PROJECT_DIR / "data" / "rag_legend_traits.jsonl"
    legend_equip = PROJECT_DIR / "data" / "rag_legend_equip.jsonl"

    manifest = {
        "img_dir": str(img_dir),
        "out_root": str(out_root),
        "images": [p.name for p in images],
        "groups": sorted(group_traits_map.keys()),
        "fightboard_stem_suffix": suffix,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for stem in sorted(fight_by_stem.keys()):
        if not tcv._stem_matches_fightboard_suffix(stem, suffix):
            continue
        gk = tcv._extract_group_key(stem, args.group_pattern)
        fight_js = fight_by_stem.get(stem) or {}
        pre_js = pre_by_stem.get(stem) or {}
        player_main = player_by_stem.get(stem) or {}
        eq_col = equip_column_by_stem.get(stem) or {}
        group_traits = group_traits_map.get(
            gk,
            {
                "trait_count_max": {},
                "trait_sources": {},
                "raw_items": [],
                "per_file_bonds": [],
                "merged_bonds_one_line": "",
                "trait_canonicalization_log": [],
            },
        )

        cv = tcv._apply_cross_validation_rules(
            fightboard_summary=fight_js,
            group_player_traits=group_traits,
            legend_chess_path=legend_chess.resolve(),
            legend_traits_path=legend_traits.resolve(),
            legend_equip_path=legend_equip.resolve(),
        )

        fight_vis = primary_fight_vis_by_stem.get(stem)
        if fight_vis is None:
            continue

        lines = _summary_lines_v3_primary(
            fight_json=fight_js,
            pre_json=pre_js,
            player_json_main=player_main,
            equip_col_json=eq_col,
            group_traits=group_traits,
            cv=cv,
        )
        # 左侧仅保留 Fightboard 可视化（棋子/血条/棋盘侧装备），不叠 pre/player/equip_column
        final_img = _draw_summary_panel(fight_vis, lines)
        out_png = out_root / f"{stem}_annotated.png"
        bp._save_image(out_png, final_img)

        merged_json: Dict[str, Any] = {
            "pipeline": "battle_pipeline_v3",
            "file": str(fight_js.get("file") or f"{stem}.png"),
            "group": gk,
            "annotated_image": out_png.name,
            "modules": {
                "fightboard": fight_js,
                "preboard": pre_js,
                "player": player_main,
                "equip_column": eq_col,
            },
            "analysis": {
                "group_traits_merged": group_traits,
                "cross_validation": cv,
            },
            "confirmed_fightboard_results": cv.get("confirmed_results") or fight_js.get("results") or [],
        }
        (out_root / f"{stem}_summary.json").write_text(
            json.dumps(merged_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] 主图输出 -> {out_png.name}, {stem}_summary.json")

    for gk, agg in group_traits_map.items():
        stems_in_group = {s for s in fight_by_stem if tcv._extract_group_key(s, args.group_pattern) == gk}
        if (agg.get("per_file_bonds") or []) and not stems_in_group:
            print(f"[WARN] 组 {gk} 有 player 结果但无主图 fightboard（缺 -{suffix or 'a'} 等），未写出主图汇总")

    print(f"完成。输出目录: {out_root}  （仅主图 *_annotated.png 与 *_summary.json；辅图不落盘）")


if __name__ == "__main__":
    main()
