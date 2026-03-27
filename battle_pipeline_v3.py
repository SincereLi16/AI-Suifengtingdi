# -*- coding: utf-8 -*-
"""
battle_pipeline_v3：单链路对局分析（与当前子模块选型一致）。

流程：
1. 主图 / 辅图：按 stem 后缀（如 -a 为主图）区分；
2. 主图：fightboard_mobilenet（棋子 MobileNet + 血条下装备）+ player_onnx 全 ROI + equip_column 全量，输出 JSON 结构；
3. 辅图：player_onnx 仅羁绊栏（bonds_only），合并进同组主图的 player JSON；
4. trait_cross_validate：用合并后的羁绊与棋盘结果校验/校准棋子；
5. 仅主图输出 PNG + JSON：左侧为**原始截图**（不在画面上叠字/框），右侧为汇总面板。

设备策略：
- fightboard 棋子嵌入：torch，auto 优先 DirectML → CUDA → CPU（与 fightboard_mobilenet 一致）；
- player_onnx：ONNX Runtime，auto 优先 DirectML → CUDA → CPU（与 player_onnx 一致）。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import fightboard_mobilenet as fm
import player_onnx as pon
import trait_cross_validate as tcv
from element_recog import bars_recog as br
from element_recog import chess_recog as cr
from element_recog import equip_recog as er
from element_recog import equip_column_recog as ecr

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_OUT = PROJECT_DIR / "battle_pipeline_v3_out"


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


def _configure_pipeline_devices() -> Tuple[str, Dict[str, bool]]:
    """打印 MobileNet 所用 torch 设备；返回 player_onnx 的 OCR 设备名与 kwargs。"""
    dev, backend = fm._select_torch_device(
        torch_device="auto",
        prefer_rocm=False,
        auto_priority="dml,cuda,cpu",
        fallback_to_cpu=True,
    )
    print(f"[V3] fightboard MobileNet torch device={dev} ({backend})")
    name, kwargs = pon._resolve_device("auto")
    print(f"[V3] player_onnx OCR ort device={name}")
    return name, kwargs


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
        cr_ = pos_obj.get("cell_row")
        cc = pos_obj.get("cell_col")
        if cr_ is not None and cc is not None:
            return f"{int(cr_)},{int(cc)}"
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


def _slim_fight_result_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """写入 summary.json 用：去掉每条棋子上的大字段 samples/stage2，并截断候选列表。"""
    if not isinstance(r, dict):
        return r
    out: Dict[str, Any] = {k: v for k, v in r.items() if k not in ("samples", "stage2")}
    at = out.get("agg_top")
    if isinstance(at, list):
        out["agg_top"] = at[:3]
    vt = out.get("vote_top")
    if isinstance(vt, list):
        out["vote_top"] = vt[:3]
    return out


def _slim_fightboard_module(fj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(fj)
    rs = out.get("results")
    if isinstance(rs, list):
        out["results"] = [_slim_fight_result_row(x) if isinstance(x, dict) else x for x in rs]
    out.pop("timings_s", None)
    return out


def _slim_group_traits_for_export(gt: Dict[str, Any]) -> Dict[str, Any]:
    """羁绊聚合：summary 中只保留计数与一行汇总，去掉 per_file 长表。"""
    return {
        "trait_count_max": dict(gt.get("trait_count_max") or {}),
        "merged_bonds_one_line": str(gt.get("merged_bonds_one_line") or ""),
    }


def _slim_cross_validation_for_export(cv: Dict[str, Any]) -> Dict[str, Any]:
    """
    TCV 精简：保留状态、损失、差异、说明与修正；去掉 legend 路径、冗长分项与可编辑条列表。
    装备纹章只保留「有条目且 granted_traits 非空」的简短列表。
    """
    keys = (
        "status",
        "greedy_fix_mode",
        "rules_version",
        "trait_loss",
        "trait_loss_after",
        "consistent",
        "trait_diff",
        "player_traits_from_bonds",
        "emblem_bonus_applied_to_expected",
        "player_traits_used",
        "board_trait_counts",
        "inference_notes",
        "changes",
        "suspect_bar_indices",
    )
    slim: Dict[str, Any] = {k: cv[k] for k in keys if k in cv}
    em = cv.get("emblem_contributions") or []
    grants: List[Dict[str, Any]] = []
    if isinstance(em, list):
        for x in em:
            if not isinstance(x, dict):
                continue
            tr = x.get("granted_traits") or []
            if not tr:
                continue
            grants.append(
                {
                    "bar_index": x.get("bar_index"),
                    "equip": x.get("equip_name"),
                    "granted_traits": list(tr),
                }
            )
    if grants:
        slim["emblem_grants"] = grants
    return slim


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
    lines.append("【装备栏】")
    lines.extend(_equip_column_text_lines(equip_col_json))
    lines.append("")
    lines.append("【局势】")
    lines.extend(_situation_lines(player_json_main))
    return lines


def _run_equip_column_on_path(
    image_path: Path,
    col_templates: Sequence[Tuple[str, Any]],
    scales_col: Sequence[int],
) -> Dict[str, Any]:
    scene_bgr = cr._load_image(image_path)
    return ecr.compute_equip_column_matches(
        scene_bgr,
        templates=list(col_templates),
        equip_roi=ecr.DEFAULT_EQUIP_ROI,
        scales=tuple(scales_col),
        threshold=0.6,
        max_peaks_per_scale=4,
        top_k=15,
        nms_iou=0.35,
        blue_buff_gap_min=0.05,
        blue_buff_single_min_score=0.78,
        grid_cols=2,
        grid_rows=10,
        skip_block_min_std=0.0,
        block_workers=1,
    )


def _run_fightboard_mobilenet_one(
    *,
    image_path: Path,
    chess_tmp: Path,
    mobilenet_bundle: Tuple[Any, Any, Any, List[str], np.ndarray],
    bar_tpl: Path,
    equip_templates: List[Tuple[str, Any]],
    scales_bar: Sequence[int],
    min_roi: int,
    batch_embed: bool,
    save_debug_crops: bool,
    tag: str,
) -> Dict[str, Any]:
    """
    fightboard_mobilenet 单图：chess + 血条下装备，返回与 TCV 兼容的 summary（含 results、equip_by_bar）。
    """
    t0 = time.perf_counter()
    scene_bgr = cr._load_image(image_path)
    t1 = time.perf_counter()
    print(f"  [{tag}] fightboard 加载图像: {t1 - t0:.4f}s  size={scene_bgr.shape[1]}x{scene_bgr.shape[0]}")

    chess_out_dir = chess_tmp
    chess_out_dir.mkdir(parents=True, exist_ok=True)
    t_c0 = time.perf_counter()
    out_json = fm.run_recognition_chess(
        image_path=image_path,
        template_path=bar_tpl,
        piece_dir=PROJECT_DIR / "chess_gallery",
        output_dir=chess_out_dir,
        circle_diameter=84,
        alpha_tight=True,
        save_debug_crops=save_debug_crops,
        batch_embed=batch_embed,
        mobilenet_bundle=mobilenet_bundle,
        torch_device="auto",
        prefer_rocm=False,
        torch_auto_priority="dml,cuda,cpu",
        torch_fallback_to_cpu=True,
        print_timings=True,
    )
    print(f"  [{tag}] fightboard chess_recog(MobileNet): {time.perf_counter() - t_c0:.4f}s")

    results: List[Dict[str, Any]] = list(out_json.get("results") or [])
    method = cv2.TM_CCOEFF_NORMED
    equip_by_bar: Dict[int, List[Dict[str, Any]]] = {}
    t_eq0 = time.perf_counter()
    for i, r in enumerate(results):
        t_b0 = time.perf_counter()
        bi = int(r.get("bar_index") or 0)
        bar_box = r.get("bar_box") or [0, 0, 0, 0]
        eq_list = fm._detect_one_bar_equip(
            scene_bgr,
            bar_box_xywh=bar_box,
            templates=equip_templates,
            scales=scales_bar,
            method=method,
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
        print(
            f"  [{tag}] fightboard 装备 bar{i + 1}/{len(results)} idx={bi} n={len(eq_list)} "
            f"{time.perf_counter() - t_b0:.4f}s"
        )
    print(f"  [{tag}] fightboard 装备小计: {time.perf_counter() - t_eq0:.4f}s")

    summary: Dict[str, Any] = {
        "file": image_path.name,
        "pipeline": "fightboard_mobilenet",
        "chess_backend": "mobilenet_v3_small",
        "results": results,
        "equip_by_bar": equip_by_bar,
    }
    return summary


_mobilenet_bundle_box: List[Any] = [None]
_bar_tpl_box: List[Any] = [None]
_equip_templates_box: List[Any] = [None]


def _ensure_fightboard_mobilenet_runtime() -> None:
    if _mobilenet_bundle_box[0] is not None:
        return
    print("[V3] 初始化 fightboard_mobilenet（MobileNet + 装备模板）…")
    try:
        _mobilenet_bundle_box[0] = fm._get_worker_mobilenet_bundle(
            piece_dir=PROJECT_DIR / "chess_gallery",
            torch_device="auto",
            prefer_rocm=False,
            torch_auto_priority="dml,cuda,cpu",
            torch_fallback_to_cpu=True,
        )
    except OSError as ex:
        msg = str(ex)
        if "shm.dll" in msg or "WinError 127" in msg or "找不到指定的程序" in msg:
            raise SystemExit(
                "无法加载 PyTorch（fightboard_mobilenet 依赖 torch）。\n"
                "可尝试安装 VC++ 运行库或按 pytorch.org 重装与 Python 版本一致的 torch。\n"
                f"\n原始错误: {msg}"
            ) from ex
        raise
    _bar_tpl_box[0] = br.find_healthbar_template(PROJECT_DIR)
    _equip_templates_box[0] = fm._get_worker_equip_templates(PROJECT_DIR / "equip_gallery")


def _merge_player_group(
    primary_stem: Optional[str],
    stems_in_group: List[str],
    player_partial_by_stem: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """同组主图全量 + 辅图羁绊合并，结构与 player_onnx 成对输出一致。"""
    if primary_stem and primary_stem in player_partial_by_stem:
        final = copy.deepcopy(player_partial_by_stem[primary_stem])
        mb = final.get("fields", {}).get("bonds") or pon._empty_bonds_field()
        for st in stems_in_group:
            if st == primary_stem:
                continue
            o = player_partial_by_stem.get(st)
            if not o:
                continue
            b2 = (o.get("fields") or {}).get("bonds")
            if isinstance(b2, dict):
                mb = pon._merge_bonds_fields(mb, b2)
        if isinstance(final.get("fields"), dict):
            final["fields"]["bonds"] = mb
        final["mode"] = "onepass_stitch_ocr_pair_merged"
        final["pair_stems"] = list(stems_in_group)
        return final

    merged_b = pon._empty_bonds_field()
    for st in stems_in_group:
        o = player_partial_by_stem.get(st)
        if not o:
            continue
        b2 = (o.get("fields") or {}).get("bonds")
        if isinstance(b2, dict):
            merged_b = pon._merge_bonds_fields(merged_b, b2)
    if not stems_in_group:
        return None
    return {
        "file": f"group-{'-'.join(sorted(stems_in_group))}",
        "mode": "bonds_only_group_no_primary",
        "fields": pon._empty_fields_shell(merged_b),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="battle_pipeline_v3：fightboard_mobilenet + player_onnx + equip_column + TCV（仅主图出图）"
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
        help="主图 stem（-a/_a 或纯数字）跑 fightboard 全链并出图",
    )
    ap.add_argument(
        "--no-batch-embed",
        action="store_true",
        help="关闭棋子采样批量前向（MobileNet）",
    )
    ap.add_argument(
        "--save-debug-crops",
        action="store_true",
        help="写出 chess_recog 调试图 crops（较慢）",
    )
    args = ap.parse_args()
    batch_embed = not bool(args.no_batch_embed)
    save_debug_crops = bool(args.save_debug_crops)

    img_dir = args.img_dir.resolve()
    out_root = args.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    images = _iter_images(img_dir)
    if not images:
        raise SystemExit(f"输入目录无图像: {img_dir}")

    target_w, target_h = tcv._infer_reference_resolution(PROJECT_DIR)
    suffix = (args.fightboard_stem_suffix or "").strip()

    scales_bar = [24, 25, 26, 27, 28]
    min_roi_bar = max(scales_bar)
    scales_col = (60, 61, 62, 63, 64)

    need_fight = any(tcv._stem_matches_fightboard_suffix(p.stem, suffix) for p in images)
    if not need_fight:
        print("[V3] 当前输入无主图（如 -a），将仅跑 player/equip_column，不进行 fightboard 交叉验证。")
    else:
        print("[V3] 预加载 fightboard MobileNet（先于 player_onnx OCR 初始化）…")
        _ensure_fightboard_mobilenet_runtime()

    _pon_name, pon_kwargs = _configure_pipeline_devices()
    ocr_engine = pon.create_ocr_engine(pon_kwargs)

    col_templates = er._build_templates(PROJECT_DIR / "equip_gallery")
    if not col_templates:
        raise SystemExit(f"装备图鉴为空: {PROJECT_DIR / 'equip_gallery'}")

    t0_all = time.perf_counter()
    print(f"[V3] 初始化完成 ({t0_all:.1f}s 起算)")

    fight_by_stem: Dict[str, Dict[str, Any]] = {}
    equip_column_by_stem: Dict[str, Dict[str, Any]] = {}
    player_partial_by_stem: Dict[str, Dict[str, Any]] = {}
    raw_scene_by_stem: Dict[str, np.ndarray] = {}
    stems_by_group: Dict[str, List[str]] = defaultdict(list)

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
            gk = tcv._extract_group_key(stem, args.group_pattern)
            stems_by_group[gk].append(stem)

            if run_fight:
                print(f"[V3] 处理 {image_path.name}（主图 · fightboard + player + equip_column）…")
            else:
                print(f"[V3] 处理 {image_path.name}（辅图 · 羁绊栏 player_onnx）…")

            t_img = time.perf_counter()

            if run_fight:
                assert _mobilenet_bundle_box[0] is not None
                assert _bar_tpl_box[0] is not None and _equip_templates_box[0] is not None

                chess_tmp = tmp / f"{stem}_fight_chess"
                stitch_png = tmp / f"{stem}_player_stitch.png"

                def _job_fight() -> Dict[str, Any]:
                    return _run_fightboard_mobilenet_one(
                        image_path=used_path,
                        chess_tmp=chess_tmp,
                        mobilenet_bundle=_mobilenet_bundle_box[0],
                        bar_tpl=_bar_tpl_box[0],
                        equip_templates=_equip_templates_box[0],
                        scales_bar=scales_bar,
                        min_roi=min_roi_bar,
                        batch_embed=batch_embed,
                        save_debug_crops=save_debug_crops,
                        tag=image_path.name,
                    )

                def _job_eq() -> Dict[str, Any]:
                    return _run_equip_column_on_path(used_path, col_templates, scales_col)

                with ThreadPoolExecutor(max_workers=2) as ex:
                    fut_f = ex.submit(_job_fight)
                    fut_e = ex.submit(_job_eq)
                    fight_summary = fut_f.result()
                    eq_js = fut_e.result()

                fight_summary["file"] = image_path.name
                fight_by_stem[stem] = fight_summary

                t_pl0 = time.perf_counter()
                pl = pon.run_once(used_path, None, stitch_png, ocr_engine=ocr_engine, bonds_only=False)
                t_pl1 = time.perf_counter()
                print(f"  [{image_path.name}] player_onnx 全 ROI: {t_pl1 - t_pl0:.4f}s")
                pl = dict(pl)
                pl["file"] = image_path.name
                player_partial_by_stem[stem] = pl

                mc = int((eq_js or {}).get("match_count") or 0)
                print(f"  [{image_path.name}] equip_column 匹配条目={mc}")
                equip_column_by_stem[stem] = eq_js
                raw_scene_by_stem[stem] = cr._load_image(used_path)

            else:
                stitch_png = tmp / f"{stem}_player_stitch.png"
                t_pl0 = time.perf_counter()
                pl = pon.run_once(used_path, None, stitch_png, ocr_engine=ocr_engine, bonds_only=True)
                t_pl1 = time.perf_counter()
                print(f"  [{image_path.name}] player_onnx 羁绊栏: {t_pl1 - t_pl0:.4f}s")
                pl = dict(pl)
                pl["file"] = image_path.name
                player_partial_by_stem[stem] = pl

            print(f"  [V3] 本图合计耗时: {time.perf_counter() - t_img:.4f}s  ({image_path.name})")

    legend_trait_names = tcv._load_legend_trait_names(PROJECT_DIR / "data" / "rag_legend_traits.jsonl")
    group_traits_map: Dict[str, Dict[str, Any]] = {}
    merged_player_by_primary: Dict[str, Dict[str, Any]] = {}

    for gk, stem_list in stems_by_group.items():
        stems_sorted = sorted(set(stem_list), key=lambda s: s)
        primary = next((s for s in stems_sorted if tcv._stem_matches_fightboard_suffix(s, suffix)), None)
        merged = _merge_player_group(primary, stems_sorted, player_partial_by_stem)
        if merged is None:
            continue
        if primary:
            merged_player_by_primary[primary] = merged
        entries: List[Tuple[str, Dict[str, Any]]] = []
        if primary and primary in merged_player_by_primary:
            entries.append((str(merged_player_by_primary[primary].get("file") or primary), merged_player_by_primary[primary]))
        else:
            for s in stems_sorted:
                if s in player_partial_by_stem:
                    entries.append(
                        (str(player_partial_by_stem[s].get("file") or s), player_partial_by_stem[s])
                    )
        if entries:
            group_traits_map[gk] = tcv._aggregate_bonds_from_player_entries(
                entries, legend_trait_names=legend_trait_names
            )

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
        player_main = merged_player_by_primary.get(stem) or player_partial_by_stem.get(stem) or {}
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

        raw_scene = raw_scene_by_stem.get(stem)
        if raw_scene is None:
            raw_scene = np.zeros((max(1, target_h), max(1, target_w), 3), dtype=np.uint8)

        lines = _summary_lines_v3_primary(
            fight_json=fight_js,
            player_json_main=player_main,
            equip_col_json=eq_col,
            group_traits=group_traits,
            cv=cv,
        )
        final_img = _draw_summary_panel(raw_scene, lines)
        out_png = out_root / f"{stem}_annotated.png"
        _save_image(out_png, final_img)

        conf_raw = cv.get("confirmed_results") or fight_js.get("results") or []
        conf_slim = [_slim_fight_result_row(x) if isinstance(x, dict) else x for x in (conf_raw if isinstance(conf_raw, list) else [])]

        merged_json: Dict[str, Any] = {
            "pipeline": "battle_pipeline_v3",
            "file": str(fight_js.get("file") or f"{stem}.png"),
            "group": gk,
            "annotated_image": out_png.name,
            "modules": {
                "fightboard": _slim_fightboard_module(fight_js),
                "player": player_main,
                "equip_column": eq_col,
            },
            "analysis": {
                "group_traits_merged": _slim_group_traits_for_export(group_traits),
                "cross_validation": _slim_cross_validation_for_export(cv),
            },
            "confirmed_fightboard_results": conf_slim,
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
