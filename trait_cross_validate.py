# -*- coding: utf-8 -*-
"""
trait_cross_validate：按对局编号聚合 player 羁绊，并与 fightboard 做交叉验证。

默认单链路：对 --img-dir 内图片依次跑 fightboard + player（不跑备战席），再自动比对输出；也可 --from-json。

1) 01-a / 01-b 等同组聚合羁绊；--fightboard-stem-suffix a 时对主图做棋盘验证。
2) 棋盘羁绊 = 英雄固有（rag_legend_chess）+ 装备赐羁绊（rag_legend_equip）；低置信棋子可贪心替换。
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CV_IMG_DIR = PROJECT_DIR / "测试集" / "交叉验证组"
DEFAULT_CV_OUT_DIR = PROJECT_DIR / "测试集" / "交叉验证组_cv_out"
CACHE_SCHEMA_VERSION = "tcv_recog_v1"

_CHESS_TRAITS_BY_PATH: Dict[str, Dict[str, List[str]]] = {}
_EQUIP_GRANTS_BY_PATH: Dict[str, Dict[str, List[str]]] = {}
_EQUIP_ALL_NAMES_BY_PATH: Dict[str, Set[str]] = {}
_LEGEND_TRAIT_NAMES_BY_PATH: Dict[str, Set[str]] = {}

# 装备描述中「赐予」额外羁绊的片段（转职纹章、暗裔神器等）
_GRANT_TRAIT_FROM_EQUIP_RE = re.compile(r"(?:携带者)?获得【([^】]+)】羁绊")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_player_module(path: Path) -> Optional[Dict[str, Any]]:
    """*_player_recog_summary.json 或 *_summary.json 内的 modules.player。"""
    js = _read_json(path)
    if not js:
        return None
    if isinstance(js.get("fields"), dict):
        return js
    mods = js.get("modules") or {}
    pl = mods.get("player")
    if isinstance(pl, dict) and isinstance(pl.get("fields"), dict):
        return pl
    return None


def _load_fightboard_module(path: Path) -> Optional[Dict[str, Any]]:
    """*_fightboard_summary.json 或 *_summary.json 内的 modules.fightboard。"""
    js = _read_json(path)
    if not js:
        return None
    if isinstance(js.get("results"), list):
        return js
    mods = js.get("modules") or {}
    fb = mods.get("fightboard")
    if isinstance(fb, dict) and isinstance(fb.get("results"), list):
        return fb
    return None


def _player_stem_from_path(p: Path) -> str:
    n = p.name
    if n.endswith("_player_recog_summary.json"):
        return n.replace("_player_recog_summary.json", "")
    if n.endswith("_summary.json"):
        return n.replace("_summary.json", "")
    return p.stem


def _fightboard_stem_from_path(p: Path) -> str:
    n = p.name
    if n.endswith("_fightboard_summary.json"):
        return n.replace("_fightboard_summary.json", "")
    if n.endswith("_summary.json"):
        return n.replace("_summary.json", "")
    return p.stem


def _iter_player_json_paths(player_dir: Path) -> List[Path]:
    direct = sorted(player_dir.glob("*_player_recog_summary.json"))
    if direct:
        return direct
    return sorted(player_dir.glob("*_summary.json"))


def _iter_fightboard_json_paths(fight_dir: Path) -> List[Path]:
    direct = sorted(fight_dir.glob("*_fightboard_summary.json"))
    if direct:
        return direct
    return sorted(fight_dir.glob("*_summary.json"))


def _extract_group_key(stem: str, pattern: str) -> str:
    m = re.match(pattern, stem)
    if m:
        return m.group(1)
    return stem


def _stem_matches_fightboard_suffix(stem: str, suffix: str) -> bool:
    """
    suffix 非空时：处理以 -a / _a 结尾的 stem（主图），或纯数字 stem（旧命名 1 / 12 等仍跑交叉验证）。
    suffix 为空则全部处理。
    """
    if not suffix:
        return True
    s = stem.lower()
    su = suffix.lower()
    if s.endswith(f"-{su}") or s.endswith(f"_{su}"):
        return True
    if re.fullmatch(r"\d+", s or ""):
        return True
    return False


def _parse_trait_piece(piece: str) -> Optional[Tuple[int, str]]:
    s = re.sub(r"\s+", "", str(piece or ""))
    m = re.match(r"^(\d+)([\u4e00-\u9fffA-Za-z·]+)$", s)
    if not m:
        return None
    try:
        return int(m.group(1)), m.group(2)
    except Exception:
        return None


def _load_legend_trait_names(path: Path) -> Set[str]:
    """从 rag_legend_traits.jsonl 读取标准羁绊名（type=trait），供 player 侧 OCR 名归一。"""
    key = str(path.resolve())
    if key in _LEGEND_TRAIT_NAMES_BY_PATH:
        return _LEGEND_TRAIT_NAMES_BY_PATH[key]
    names: Set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("type") != "trait":
                continue
            n = str(o.get("name") or "").strip()
            if n:
                names.add(n)
    _LEGEND_TRAIT_NAMES_BY_PATH[key] = names
    return names


def _canonicalize_player_trait_name(raw: str, legend_names: Set[str]) -> Tuple[str, str]:
    """
    将 bond_items 解析出的羁绊名对齐到图鉴标准名。
    顺序：完全匹配 → 唯一前缀扩展（截断 OCR）→ difflib 近邻（错字）。
    返回 (用于聚合的名, 方式标签)。
    """
    if not raw or not legend_names:
        return raw, "as_is"
    if raw in legend_names:
        return raw, "exact"
    prefs = sorted({n for n in legend_names if n.startswith(raw) and len(n) > len(raw)})
    if len(prefs) == 1:
        return prefs[0], "prefix_extend"
    if len(prefs) > 1:
        pick = difflib.get_close_matches(raw, prefs, n=1, cutoff=0.35)
        if pick:
            return pick[0], "prefix_pick"
        prefs.sort(key=len, reverse=True)
        return prefs[0], "prefix_longest"
    fuzzy = difflib.get_close_matches(raw, list(legend_names), n=1, cutoff=0.5)
    if fuzzy:
        return fuzzy[0], "fuzzy"
    return raw, "unresolved"


def _load_chess_name_to_traits(path: Path) -> Dict[str, List[str]]:
    key = str(path.resolve())
    if key in _CHESS_TRAITS_BY_PATH:
        return _CHESS_TRAITS_BY_PATH[key]
    m: Dict[str, List[str]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("type") != "chess":
                continue
            name = str(o.get("name") or "").strip()
            traits = o.get("traits") or []
            if name and isinstance(traits, list):
                m[name] = [str(t) for t in traits]
    _CHESS_TRAITS_BY_PATH[key] = m
    return m


def _equip_record_granted_traits(o: Dict[str, Any]) -> List[str]:
    """从单条 legend_equip 的 desc/text 抽出「获得【某】羁绊」中的羁绊名（有序去重）。"""
    order: List[str] = []
    seen: Set[str] = set()
    for blob in (str(o.get("desc") or ""), str(o.get("text") or "")):
        for m in _GRANT_TRAIT_FROM_EQUIP_RE.finditer(blob):
            t = m.group(1).strip()
            if t and t not in seen:
                seen.add(t)
                order.append(t)
    return order


def _load_equip_name_to_granted_traits(path: Path) -> Dict[str, List[str]]:
    """装备显示名 -> 该装备额外提供的羁绊列表（每人每件装备各计一次，可与英雄固有叠加）。"""
    key = str(path.resolve())
    if key in _EQUIP_GRANTS_BY_PATH:
        return _EQUIP_GRANTS_BY_PATH[key]
    m: Dict[str, List[str]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("type") != "equip":
                continue
            name = str(o.get("name") or "").strip()
            traits = _equip_record_granted_traits(o)
            if name and traits:
                m[name] = traits
    _EQUIP_GRANTS_BY_PATH[key] = m
    return m


def _load_equip_all_names(path: Path) -> Set[str]:
    """图鉴中出现的全部装备名（用于判断识别结果是否在库）。"""
    k = str(path.resolve())
    if k in _EQUIP_ALL_NAMES_BY_PATH:
        return _EQUIP_ALL_NAMES_BY_PATH[k]
    names: Set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("type") != "equip":
                continue
            n = str(o.get("name") or "").strip()
            if n:
                names.add(n)
    _EQUIP_ALL_NAMES_BY_PATH[k] = names
    return names


def _equips_for_bar(equip_by_bar: Any, bar_index: Any) -> List[Dict[str, Any]]:
    if not isinstance(equip_by_bar, dict):
        return []
    candidates: List[Any] = []
    if bar_index is not None:
        candidates.append(bar_index)
        s = str(bar_index).strip()
        candidates.append(s)
        if s.isdigit():
            candidates.append(int(s))
    tried: Set[Any] = set()
    for k in candidates:
        if k in tried:
            continue
        tried.add(k)
        if k in equip_by_bar:
            raw = equip_by_bar[k]
            if isinstance(raw, list):
                return [x for x in raw if isinstance(x, dict)]
            return []
    return []


def _trait_counts_from_board(
    names: List[str],
    results: List[Dict[str, Any]],
    equip_by_bar: Any,
    chess: Dict[str, List[str]],
    equip_grants: Dict[str, List[str]],
) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for i, r in enumerate(results or []):
        nm = names[i] if i < len(names) else ""
        if nm:
            for t in chess.get(nm, []):
                c[t] += 1
        bi = (r or {}).get("bar_index")
        for eq in _equips_for_bar(equip_by_bar, bi):
            en = str(eq.get("name") or "").strip()
            for t in equip_grants.get(en, []):
                c[t] += 1
    return dict(c)


def _trait_counts_emblems_only(
    results: List[Dict[str, Any]],
    equip_by_bar: Any,
    equip_grants: Dict[str, List[str]],
) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in results or []:
        bi = (r or {}).get("bar_index")
        for eq in _equips_for_bar(equip_by_bar, bi):
            en = str(eq.get("name") or "").strip()
            for t in equip_grants.get(en, []):
                c[t] += 1
    return dict(c)


def _emblem_audit_rows(
    results: List[Dict[str, Any]],
    equip_by_bar: Any,
    equip_grants: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """逐格装备赐羁绊明细。"""
    rows: List[Dict[str, Any]] = []
    for r in results or []:
        bi = (r or {}).get("bar_index")
        for eq in _equips_for_bar(equip_by_bar, bi):
            en = str(eq.get("name") or "").strip()
            if not en:
                continue
            gt = equip_grants.get(en, [])
            rows.append({"bar_index": bi, "equip_name": en, "granted_traits": list(gt)})
    return rows


def _format_merged_bonds_line(trait_max: Dict[str, int]) -> str:
    if not trait_max:
        return "(无有效羁绊条目)"
    parts = [f"{cnt}{trait}" for trait, cnt in sorted(trait_max.items(), key=lambda kv: kv[0])]
    return " / ".join(parts)


def _field_parsed_from_player(fields: Any, key: str) -> str:
    if not isinstance(fields, dict):
        return ""
    d = fields.get(key)
    if isinstance(d, dict):
        return str(d.get("parsed") or "").strip()
    return ""


def _aggregate_bonds_from_player_entries(
    entries: List[Tuple[str, Dict[str, Any]]],
    legend_trait_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """entries: (来源标签如文件名, player 模块 dict，含 fields)。
    legend_trait_names 非空时：将 bond_items 中的羁绊名按图鉴归一后再做 max 聚合。"""
    trait_max: Dict[str, int] = {}
    trait_sources: Dict[str, List[str]] = defaultdict(list)
    raw_items: List[Dict[str, Any]] = []
    per_file: List[Dict[str, Any]] = []
    canon_log: Set[str] = set()

    for label, js in sorted(entries, key=lambda x: x[0]):
        fields = js.get("fields") or {}
        bonds = fields.get("bonds") if isinstance(fields, dict) else None
        bond_items = bonds.get("bond_items") if isinstance(bonds, dict) else []
        bsum = (
            str((bonds or {}).get("bond_summary") or (bonds or {}).get("parsed") or "").strip()
            if isinstance(bonds, dict)
            else ""
        )
        bond_cells = list((bonds or {}).get("bond_cells") or []) if isinstance(bonds, dict) else []
        bonds_raw_full = str((bonds or {}).get("raw") or "").strip() if isinstance(bonds, dict) else ""
        per_file.append(
            {
                "file": label,
                "bond_summary": bsum,
                "bond_items": list(bond_items) if isinstance(bond_items, list) else [],
                "bond_cells": bond_cells,
                "bonds_raw_concat": bonds_raw_full,
                "phase_parsed": _field_parsed_from_player(fields, "phase"),
                "level_parsed": _field_parsed_from_player(fields, "level"),
            }
        )
        for item in bond_items or []:
            parsed = _parse_trait_piece(str(item))
            if not parsed:
                continue
            cnt, trait_ocr = parsed
            trait = trait_ocr
            resolution = "exact"
            if legend_trait_names:
                trait, resolution = _canonicalize_player_trait_name(trait_ocr, legend_trait_names)
                if trait != trait_ocr:
                    canon_log.add(f"{trait_ocr}→{trait} ({resolution})")
            if trait not in trait_max or cnt > trait_max[trait]:
                trait_max[trait] = cnt
            trait_sources[trait].append(label)
            row: Dict[str, Any] = {
                "file": label,
                "piece": item,
                "parsed_count": cnt,
                "trait_ocr": trait_ocr,
                "parsed_trait": trait,
                "trait_resolution": resolution,
            }
            raw_items.append(row)

    return {
        "trait_count_max": dict(sorted(trait_max.items(), key=lambda kv: kv[0])),
        "trait_sources": {k: sorted(set(v)) for k, v in sorted(trait_sources.items(), key=lambda kv: kv[0])},
        "raw_items": raw_items,
        "per_file_bonds": per_file,
        "merged_bonds_one_line": _format_merged_bonds_line(trait_max),
        "trait_canonicalization_log": sorted(canon_log),
    }


def _collect_group_player_traits(group_files: List[Path], legend_traits: Path) -> Dict[str, Any]:
    entries: List[Tuple[str, Dict[str, Any]]] = []
    for p in group_files:
        js = _load_player_module(p)
        if js:
            entries.append((p.name, js))
    return _aggregate_bonds_from_player_entries(
        entries,
        legend_trait_names=_load_legend_trait_names(legend_traits),
    )


def _per_file_bond_report_lines(row: Dict[str, Any]) -> List[str]:
    """与 player_recog 一致：分块 bond_cells + 阶段/等级 + 汇总。"""
    lines: List[str] = []
    fn = row.get("file") or "?"
    lines.append(f"--- {fn} ---")
    ph = str(row.get("phase_parsed") or "").strip()
    lv = str(row.get("level_parsed") or "").strip()
    if ph:
        lines.append(f"  阶段: {ph}")
    if lv:
        lines.append(f"  等级: {lv}")
    lines.append("  羁绊栏分块（ROI 网格逐格 OCR，与 player_recog 相同结构）:")
    cells = row.get("bond_cells") or []
    if not cells:
        lines.append("    （本图无 bond_cells 或羁绊区未检出）")
        br = str(row.get("bonds_raw_concat") or "").strip()
        if br:
            lines.append(f"    bonds.raw: {br[:160]}{'…' if len(br) > 160 else ''}")
    else:
        for c in cells:
            if not isinstance(c, dict):
                continue
            idx = c.get("index")
            piece = c.get("piece")
            ocr_rows = c.get("ocr_rows") or ""
            raw_cell = str(c.get("raw") or "").strip()
            piece_s = str(piece).strip() if piece else "（未解析为 N羁绊名）"
            lines.append(f"    格{idx}  piece: {piece_s}")
            lines.append(f"            ocr_rows: {ocr_rows}")
            if raw_cell and raw_cell != str(ocr_rows).strip():
                lines.append(f"            raw: {raw_cell[:120]}")
    bsum = str(row.get("bond_summary") or "").strip()
    if bsum:
        lines.append(f"  汇总 bond_summary/parsed: {bsum}")
    items = row.get("bond_items") or []
    if items:
        lines.append(f"  可解析 bond_items: {', '.join(str(x) for x in items)}")
    return lines


def _fightboard_result_lines(fj: Dict[str, Any]) -> List[str]:
    """与 battle_pipeline fight 段一致的棋盘文本行。"""
    fr = (fj or {}).get("results") or []
    equip_by_bar = (fj or {}).get("equip_by_bar") or {}
    lines: List[str] = []
    if not fr:
        lines.append("(无棋子)")
        return lines
    for r in fr:
        try:
            bi = int((r or {}).get("bar_index") or 0)
        except Exception:
            bi = 0
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
                if isinstance(e, dict):
                    en = str(e.get("name") or "").strip()
                    if en:
                        equip_names.append(en)
        if equip_names:
            lines.append(f"- {name} ({conf}) 位置:{pos} 装备:{'/'.join(equip_names)}")
        else:
            lines.append(f"- {name} ({conf}) 位置:{pos}")
    return lines


def _cv_analysis_lines(cv: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append(f"trait_loss={cv.get('trait_loss')}  consistent={cv.get('consistent')}  status={cv.get('status')}")
    ho = cv.get("board_trait_counts_heroes_only") or {}
    eo = cv.get("board_trait_counts_emblems_only") or {}
    bt = cv.get("board_trait_counts") or {}

    def _ft(d: Dict[str, Any]) -> str:
        if not d:
            return "(空)"
        return " / ".join(f"{n}{t}" for t, n in sorted(d.items(), key=lambda kv: kv[0]))

    lines.append(f"棋盘羁绊·英雄: {_ft(ho)}")
    lines.append(f"棋盘羁绊·装备赐: {_ft(eo)}")
    lines.append(f"棋盘羁绊·合计: {_ft(bt)}")
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


def _save_primary_annotated_png(
    *,
    out_dir: Path,
    stem: str,
    fight_vis: Any,
    group_traits: Dict[str, Any],
    fight_js: Dict[str, Any],
    cv: Dict[str, Any],
) -> None:
    """主图 a：左侧 fightboard 标注 + 右侧文字（聚合羁绊、分图 player 结构、棋盘列表、交叉验证）。"""
    import battle_pipeline as bp

    lines: List[str] = []
    lines.append("trait_cross_validate · 主图汇总")
    lines.append("")
    lines.append("【聚合羁绊 a+b · trait_count_max】")
    lines.append(str(group_traits.get("merged_bonds_one_line") or ""))
    tm = group_traits.get("trait_count_max") or {}
    if tm:
        lines.append("计数: " + " / ".join(f"{n}{t}" for t, n in sorted(tm.items(), key=lambda kv: kv[0])))
    lines.append("")
    lines.append("【各图 player 羁绊栏 · 与 player_recog 相同字段】")
    for row in group_traits.get("per_file_bonds") or []:
        lines.extend(_per_file_bond_report_lines(row if isinstance(row, dict) else {}))
        lines.append("")
    lines.append("【主图 Fightboard 识别】")
    lines.extend(_fightboard_result_lines(fight_js))
    lines.append("")
    lines.append("【羁绊 vs 棋盘一致性】")
    lines.extend(_cv_analysis_lines(cv))

    panel = bp._draw_text_panel(fight_vis, lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    bp._save_image(out_dir / f"{stem}_tcv_annotated.png", panel)


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


def _image_size(path: Path) -> Tuple[int, int]:
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"读取图片失败: {path}")
    h, w = im.shape[:2]
    return w, h


def _infer_reference_resolution(project_root: Path) -> Tuple[int, int]:
    """
    返回标准对局分辨率（固定 2196x1253）。
    说明：player/fightboard 的固定 ROI 是按该坐标系调过的，测试集需先映射到这个尺寸。
    """
    _ = project_root
    return (2196, 1253)


def _normalize_image_to_size(src: Path, dst: Path, target_w: int, target_h: int) -> None:
    """直接缩放到目标尺寸，保证与固定 ROI 坐标系一致。"""
    im = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"读取图片失败: {src}")
    resized = cv2.resize(im, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".png", resized)
    if not ok:
        raise RuntimeError(f"编码图片失败: {src}")
    dst.write_bytes(buf.tobytes())


def _run_cross_validate_outputs(
    *,
    group_traits_map: Dict[str, Dict[str, Any]],
    fight_by_stem: Dict[str, Dict[str, Any]],
    out_root: Path,
    group_pattern: str,
    fightboard_stem_suffix: str,
    legend_chess: Path,
    legend_equip: Path,
    legend_traits: Path,
    fightboard_source: str = "json",
    primary_fight_vis_by_stem: Optional[Dict[str, Any]] = None,
) -> None:
    """对 fight_by_stem 中主图 stem 写 group_traits / confirmed_fightboard；recognition 模式可附主图 fight vis 写 PNG。"""
    out_root.mkdir(parents=True, exist_ok=True)
    group_out = out_root / "group_traits"
    confirmed_out = out_root / "confirmed_fightboard"
    annotated_dir = out_root / "primary_annotated"
    group_out.mkdir(parents=True, exist_ok=True)
    confirmed_out.mkdir(parents=True, exist_ok=True)

    for gk, agg in sorted(group_traits_map.items(), key=lambda kv: kv[0]):
        (group_out / f"{gk}_group_traits.json").write_text(
            json.dumps(
                {
                    "group": gk,
                    "player_files": [row.get("file") for row in (agg.get("per_file_bonds") or [])],
                    "merged_player_traits": agg,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    suffix = (fightboard_stem_suffix or "").strip()
    for stem in sorted(fight_by_stem.keys(), key=lambda s: s):
        if not _stem_matches_fightboard_suffix(stem, suffix):
            continue
        gk = _extract_group_key(stem, group_pattern)
        fight_js = fight_by_stem[stem]
        if not fight_js or not isinstance(fight_js.get("results"), list):
            print(f"[SKIP] 无 fightboard 结果: {stem}")
            continue
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
        cv = _apply_cross_validation_rules(
            fightboard_summary=fight_js,
            group_player_traits=group_traits,
            legend_chess_path=legend_chess.resolve(),
            legend_traits_path=legend_traits.resolve(),
            legend_equip_path=legend_equip.resolve(),
        )
        _print_terminal_report(
            group=gk,
            merged=group_traits,
            fight_stem=stem,
            fight_file=f"{stem}.png",
            fightboard_summary=fight_js,
            cv=cv,
        )
        src_name = str(fight_js.get("file") or f"{stem}.png")
        out_js = {
            "file": src_name,
            "group": gk,
            "fightboard_source": fightboard_source,
            "group_traits_file": f"{gk}_group_traits.json" if (group_out / f"{gk}_group_traits.json").is_file() else None,
            "cross_validation": cv,
            "confirmed_results": cv.get("confirmed_results") or [],
        }
        (confirmed_out / f"{stem}_confirmed_fightboard.json").write_text(
            json.dumps(out_js, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] confirmed -> {stem}_confirmed_fightboard.json")
        if primary_fight_vis_by_stem and stem in primary_fight_vis_by_stem:
            vis = primary_fight_vis_by_stem.get(stem)
            if vis is not None:
                try:
                    _save_primary_annotated_png(
                        out_dir=annotated_dir,
                        stem=stem,
                        fight_vis=vis,
                        group_traits=group_traits,
                        fight_js=fight_js,
                        cv=cv,
                    )
                    print(f"[OK] 主图标注 -> primary_annotated/{stem}_tcv_annotated.png")
                except Exception as ex:
                    print(f"[WARN] 主图标注保存失败 {stem}: {ex}")

    # 组内仅有辅图、无主图 fight 时提示
    for gk, agg in group_traits_map.items():
        stems_in_group = {s for s in fight_by_stem if _extract_group_key(s, group_pattern) == gk}
        if (agg.get("per_file_bonds") or []) and not stems_in_group:
            print(f"[WARN] 组 {gk} 有 player 结果但无主图 fightboard（缺 -{fightboard_stem_suffix or 'a'} 等），未做交叉验证")

def run_recognition_then_validate(
    *,
    img_dir: Path,
    out_root: Path,
    group_pattern: str,
    fightboard_stem_suffix: str,
    legend_chess: Path,
    legend_equip: Path,
    legend_traits: Path,
    use_cache: bool,
    cache_dir: Path,
) -> None:
    """单进程：主图（-a / 纯数字）跑 fightboard + player；辅图（如 -b）仅 player；再合并羁绊并与主图棋盘比对。"""
    import pickle

    import battle_pipeline as bp
    import fightboard_recog as fb
    import player_recog as pr

    project = bp.PROJECT_DIR
    images = bp._iter_images(img_dir.resolve())
    if not images:
        raise SystemExit(f"输入目录无图像: {img_dir}")

    cache_root = cache_dir.resolve()
    if use_cache:
        (cache_root / "fight").mkdir(parents=True, exist_ok=True)
        (cache_root / "player").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    template: Optional[Any] = None
    model: Optional[Any] = None
    device: Optional[Any] = None
    transform: Optional[Any] = None
    piece_db: Optional[Any] = None
    equip_templates: Optional[Any] = None
    scales = [24, 25, 26, 27, 28]
    min_roi = max(scales)
    centers, rects = pr._default_layout()

    target_w, target_h = _infer_reference_resolution(project)
    print(f"[TCV] 参考分辨率: {target_w}x{target_h}")

    need_fight = any(_stem_matches_fightboard_suffix(p.stem, fightboard_stem_suffix) for p in images)
    if not need_fight:
        print("[TCV] 当前输入无主图（如 -a），将仅跑 player，不进行 fightboard 交叉验证。")
    fight_runtime_ready = False

    def _ensure_fight_runtime() -> None:
        nonlocal template, model, device, transform, piece_db, equip_templates, fight_runtime_ready
        if fight_runtime_ready:
            return
        print("[TCV] 初始化 fightboard 模型…")
        try:
            template = fb.br.find_healthbar_template(project)
            model, device, transform = fb.cr._get_embedding_model("dinov2_vits14")
            piece_db, _ = fb.cr.load_or_build_piece_embedding_db(
                project / "chess_gallery",
                model,
                device,
                transform,
                embed_backbone="dinov2_vits14",
                root=project,
                force_rebuild=False,
                verbose=True,
            )
            equip_templates = fb.er._build_templates(project / "equip_gallery")
            fight_runtime_ready = True
        except OSError as ex:
            msg = str(ex)
            if "shm.dll" in msg or "WinError 127" in msg:
                raise SystemExit(
                    "fightboard 模型初始化失败：PyTorch DLL 依赖未满足（shm.dll / WinError 127）。\n"
                    "可先用已有缓存继续：保持主图缓存命中时不会触发模型加载；\n"
                    "若需重新识别主图，请修复当前 Python 的 torch 运行库后重试。"
                ) from ex
            raise

    fight_params_key = "fight:cd84:eq0.78:sc24-28:nms0.35"

    def _will_need_fight_runtime_this_run() -> bool:
        """与主循环内 fight 缓存键一致；若即将加载 torch，则须在 Paddle OCR 之前加载（Windows DLL 顺序）。"""
        if not need_fight:
            return False
        if not use_cache:
            return True
        for image_path in images:
            stem = image_path.stem
            if not _stem_matches_fightboard_suffix(stem, fightboard_stem_suffix):
                continue
            src_w, src_h = _image_size(image_path)
            img_sha1 = _file_sha1(image_path) + f"|norm:{target_w}x{target_h}|src:{src_w}x{src_h}"
            fight_cache_key = _cache_key("fight", img_sha1, fight_params_key)
            p_cache = cache_root / "fight" / f"{fight_cache_key}.pkl"
            if not p_cache.is_file():
                return True
            try:
                data = pickle.loads(p_cache.read_bytes())
                if not (isinstance(data, dict) and "summary" in data):
                    return True
            except Exception:
                return True
        return False

    if _will_need_fight_runtime_this_run():
        print("[TCV] 预加载 fightboard（先于 Paddle OCR，避免 Windows 上 torch DLL 与 paddle 冲突）…")
        _ensure_fight_runtime()

    print("[TCV] 初始化 OCR…")
    ocr_engine = pr._create_ocr_engine(verbose=False)

    print(f"[TCV] 初始化完成 ({time.perf_counter() - t0:.1f}s)")

    fight_by_stem: Dict[str, Dict[str, Any]] = {}
    primary_fight_vis_by_stem: Dict[str, Any] = {}
    player_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="tcv_fight_tmp_") as td:
        tmp = Path(td)
        for image_path in images:
            stem = image_path.stem
            src_w, src_h = _image_size(image_path)
            used_path = image_path
            if (src_w, src_h) != (target_w, target_h):
                used_path = tmp / f"{stem}__norm_{target_w}x{target_h}.png"
                _normalize_image_to_size(image_path, used_path, target_w, target_h)
                print(f"[TCV] 尺寸对齐 {image_path.name}: {src_w}x{src_h} -> {target_w}x{target_h}")
            run_fight = _stem_matches_fightboard_suffix(stem, fightboard_stem_suffix)
            if run_fight:
                print(f"[TCV] 识别 {image_path.name}（主图 · fightboard + player）…")
            else:
                print(f"[TCV] 识别 {image_path.name}（辅图 · 仅 player）…")
            t_img = time.perf_counter()
            img_sha1 = _file_sha1(image_path) + f"|norm:{target_w}x{target_h}|src:{src_w}x{src_h}"

            if run_fight:
                fight_cache_key = _cache_key("fight", img_sha1, fight_params_key)
                fight = None
                if use_cache:
                    p_cache = cache_root / "fight" / f"{fight_cache_key}.pkl"
                    if p_cache.is_file():
                        try:
                            fight = pickle.loads(p_cache.read_bytes())
                            if isinstance(fight, dict) and "summary" in fight:
                                print(f"  [CACHE] fight hit")
                        except Exception:
                            fight = None
                if fight is None:
                    _ensure_fight_runtime()
                    assert template is not None and piece_db is not None and equip_templates is not None
                    fight = bp._run_one_fightboard(
                        image_path=used_path,
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
                    if isinstance(fight.get("summary"), dict):
                        fight["summary"]["file"] = image_path.name
                    if use_cache:
                        p_cache = cache_root / "fight" / f"{fight_cache_key}.pkl"
                        p_cache.parent.mkdir(parents=True, exist_ok=True)
                        p_cache.write_bytes(pickle.dumps(fight))
                else:
                    if isinstance(fight.get("summary"), dict):
                        fight["summary"]["file"] = image_path.name
                fight_by_stem[stem] = fight["summary"]
                primary_fight_vis_by_stem[stem] = fight["vis"]

            player_params_key = "player:font18:layout-default"
            player_cache_key = _cache_key("player", img_sha1, player_params_key)
            player_summary: Optional[Dict[str, Any]] = None
            pp = cache_root / "player" / f"{player_cache_key}.pkl"
            if use_cache and pp.is_file():
                try:
                    pc = pickle.loads(pp.read_bytes())
                    if isinstance(pc, dict) and isinstance(pc.get("summary"), dict):
                        player_summary = pc["summary"]
                        print(f"  [CACHE] player hit")
                except Exception:
                    player_summary = None
            if player_summary is None:
                _, player_summary = pr.process_image(ocr_engine, used_path, centers, rects, 18)
                if use_cache:
                    pp.parent.mkdir(parents=True, exist_ok=True)
                    pp.write_bytes(pickle.dumps({"summary": player_summary}))

            player_summary = dict(player_summary)
            player_summary["file"] = image_path.name
            gk = _extract_group_key(stem, group_pattern)
            player_by_group[gk].append(player_summary)
            print(f"  [TCV] 完成 ({time.perf_counter() - t_img:.1f}s)")

    legend_trait_names = _load_legend_trait_names(legend_traits)
    group_traits_map: Dict[str, Dict[str, Any]] = {}
    for gk, plist in player_by_group.items():
        entries = [(str(p.get("file") or "unknown"), p) for p in sorted(plist, key=lambda x: str(x.get("file") or ""))]
        group_traits_map[gk] = _aggregate_bonds_from_player_entries(
            entries,
            legend_trait_names=legend_trait_names,
        )

    _run_cross_validate_outputs(
        group_traits_map=group_traits_map,
        fight_by_stem=fight_by_stem,
        out_root=out_root,
        group_pattern=group_pattern,
        fightboard_stem_suffix=fightboard_stem_suffix,
        legend_chess=legend_chess,
        legend_equip=legend_equip,
        legend_traits=legend_traits,
        fightboard_source="recognition",
        primary_fight_vis_by_stem=primary_fight_vis_by_stem or None,
    )

    manifest = {
        "img_dir": str(img_dir.resolve()),
        "out_root": str(out_root.resolve()),
        "images": [p.name for p in images],
        "groups": sorted(group_traits_map.keys()),
    }
    (out_root / "tcv_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[TCV] 全部完成。输出: {out_root.resolve()}")


def _hero_names_from_results(results: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for r in results or []:
        out.append(str((r or {}).get("best") or "").strip())
    return out


def _trait_counts_from_names(names: List[str], chess: Dict[str, List[str]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for n in names:
        if not n:
            continue
        ts = chess.get(n)
        if not ts:
            continue
        for t in ts:
            c[t] += 1
    return dict(c)


def _trait_loss(expected: Dict[str, int], actual: Dict[str, int]) -> int:
    keys: Set[str] = set(expected.keys()) | set(actual.keys())
    return sum(abs(int(expected.get(k, 0)) - int(actual.get(k, 0))) for k in keys)


def _trait_diff_detail(expected: Dict[str, int], actual: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    keys: Set[str] = set(expected.keys()) | set(actual.keys())
    detail: Dict[str, Dict[str, int]] = {}
    for k in sorted(keys):
        e = int(expected.get(k, 0))
        b = int(actual.get(k, 0))
        if e != b:
            detail[k] = {"expected": e, "board": b, "delta_board_minus_expected": b - e}
    return detail


def _candidate_names_from_result(r: Dict[str, Any], *, limit: int = 10) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    cur = str(r.get("best") or "").strip()
    if cur:
        seen.add(cur)
        out.append(cur)
    cap = max(limit, 8)
    for key in ("agg_top", "vote_top"):
        for it in (r.get(key) or [])[:cap]:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("name") or "").strip()
            if nm and nm not in seen:
                seen.add(nm)
                out.append(nm)
    return out[:cap]


def _greedy_fix_low_confidence(
    results: List[Dict[str, Any]],
    expected: Dict[str, int],
    chess: Dict[str, List[str]],
    equip_by_bar: Any,
    equip_grants: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    """
    仅调整 confidence==low 的条目；贪心多轮直到无法降低 loss。
    返回 (新 results 深拷贝, changes, loss_before, loss_after)。
    """
    out = copy.deepcopy(results or [])
    names = _hero_names_from_results(out)
    eb = equip_by_bar
    loss_before = _trait_loss(expected, _trait_counts_from_board(names, out, eb, chess, equip_grants))
    low_idx = [i for i, r in enumerate(out) if str((r or {}).get("confidence") or "") == "low"]
    changes: List[Dict[str, Any]] = []
    if not low_idx:
        return out, changes, loss_before, loss_before

    loss_cur = loss_before
    improved = True
    while improved:
        improved = False
        for i in low_idx:
            r = out[i]
            cands = _candidate_names_from_result(r)
            best_cand = names[i]
            best_l = _trait_loss(
                expected,
                _trait_counts_from_board(names, out, eb, chess, equip_grants),
            )
            for cand in cands:
                trial = names.copy()
                trial[i] = cand
                l = _trait_loss(expected, _trait_counts_from_board(trial, out, eb, chess, equip_grants))
                if l < best_l:
                    best_l = l
                    best_cand = cand
            if best_cand != names[i] and best_l < loss_cur:
                pos_lbl = ""
                po = r.get("position") if isinstance(r.get("position"), dict) else {}
                if isinstance(po, dict):
                    pos_lbl = str(po.get("label") or "")
                changes.append(
                    {
                        "bar_index": r.get("bar_index"),
                        "position_label": pos_lbl,
                        "from": names[i],
                        "to": best_cand,
                        "confidence": "low",
                        "trait_loss_before": loss_cur,
                        "trait_loss_after": best_l,
                    }
                )
                names[i] = best_cand
                out[i]["best"] = best_cand
                out[i]["trait_cv_corrected"] = True
                loss_cur = best_l
                improved = True
    return out, changes, loss_before, loss_cur


def _apply_cross_validation_rules(
    *,
    fightboard_summary: Dict[str, Any],
    group_player_traits: Dict[str, Any],
    legend_chess_path: Path,
    legend_traits_path: Path,
    legend_equip_path: Optional[Path] = None,
) -> Dict[str, Any]:
    _ = legend_traits_path  # 预留：羁绊白名单/加权
    chess = _load_chess_name_to_traits(legend_chess_path)
    equip_grants: Dict[str, List[str]] = {}
    if legend_equip_path is not None:
        equip_grants = _load_equip_name_to_granted_traits(legend_equip_path)
    original_results = fightboard_summary.get("results") or []
    equip_by_bar = fightboard_summary.get("equip_by_bar")
    expected = dict(group_player_traits.get("trait_count_max") or {})
    names0 = _hero_names_from_results(original_results)
    hero_only_traits = _trait_counts_from_names(names0, chess)
    emblem_only_traits = _trait_counts_emblems_only(original_results, equip_by_bar, equip_grants)
    board_traits = _trait_counts_from_board(names0, original_results, equip_by_bar, chess, equip_grants)
    emblem_audit = _emblem_audit_rows(original_results, equip_by_bar, equip_grants)
    equip_on_board = {str(x.get("equip_name") or "").strip() for x in emblem_audit if x.get("equip_name")}
    equip_on_board.discard("")
    if legend_equip_path and legend_equip_path.is_file():
        all_equip_names = _load_equip_all_names(legend_equip_path)
        unknown_equips = sorted(e for e in equip_on_board if e not in all_equip_names)
    else:
        unknown_equips = []
    diff = _trait_diff_detail(expected, board_traits)
    loss = _trait_loss(expected, board_traits)
    consistent = loss == 0

    low_indices = [
        int((r or {}).get("bar_index") or -1)
        for r in original_results
        if str((r or {}).get("confidence") or "") == "low"
    ]
    low_indices = [x for x in low_indices if x >= 0]

    unknown_heroes = sorted({n for n in names0 if n and n not in chess})

    inference_notes: List[str] = []
    confirmed = copy.deepcopy(original_results)
    changes: List[Dict[str, Any]] = []
    loss_after = loss
    status = "ok"

    if consistent:
        status = "ok"
    else:
        if not low_indices:
            status = "mismatch_no_low_confidence"
            inference_notes.append(
                "棋盘推导羁绊与聚合羁绊不一致，且无非 low 置信度格子可供按候选替换推断；请检查高置信识别、羁绊 OCR 或装备/纹章识别。"
            )
        else:
            confirmed, changes, loss_after, _ = _greedy_fix_low_confidence(
                original_results,
                expected,
                chess,
                equip_by_bar,
                equip_grants,
            )
            if changes:
                status = "mismatch_inferred" if loss_after > 0 else "ok_after_inference"
                for ch in changes:
                    inference_notes.append(
                        f"低置信 bar#{ch.get('bar_index')} {ch.get('position_label') or ''} : "
                        f"「{ch.get('from')}」→ 推断「{ch.get('to')}」（trait_loss {ch.get('trait_loss_before')}→{ch.get('trait_loss_after')}）"
                    )
            else:
                status = "mismatch_low_not_helped"
                inference_notes.append(
                    "存在 low 置信度棋子，但 agg_top/vote 候选无法降低与羁绊表的差异；请扩大候选或检查羁绊 OCR、纹章与装备识别。"
                )

    return {
        "status": status,
        "rules_version": "v1_trait_board_emblem",
        "legend_chess_path": str(legend_chess_path),
        "legend_traits_path": str(legend_traits_path),
        "legend_equip_path": str(legend_equip_path) if legend_equip_path else None,
        "player_traits_used": expected,
        "board_trait_counts": board_traits,
        "board_trait_counts_heroes_only": hero_only_traits,
        "board_trait_counts_emblems_only": emblem_only_traits,
        "emblem_contributions": emblem_audit,
        "equip_names_not_in_legend": unknown_equips,
        "trait_diff": diff,
        "trait_loss": loss,
        "trait_loss_after": loss_after,
        "consistent": consistent,
        "low_confidence_bar_indices": low_indices,
        "unknown_hero_names": unknown_heroes,
        "changes": changes,
        "inference_notes": inference_notes,
        "confirmed_results": confirmed,
    }


def _fmt_terminal_trait_counts(d: Optional[Dict[str, Any]]) -> str:
    """羁绊名 -> 展示为「人数+羁绊名」与 merged_bonds_one_line 一致。"""
    if not d:
        return "（无）"
    parts = [f"{int(n)}{t}" for t, n in sorted(d.items(), key=lambda kv: kv[0])]
    return " / ".join(parts)


def _fightboard_piece_names_in_order(fightboard_summary: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for r in fightboard_summary.get("results") or []:
        if not isinstance(r, dict):
            continue
        n = str(r.get("best") or "").strip()
        if n:
            out.append(n)
    return out


def _emblem_grant_equip_names(cv: Dict[str, Any]) -> List[str]:
    """图鉴中「获得【X】羁绊」的装备（转职/赐羁绊），来自 emblem_contributions。"""
    seen: Set[str] = set()
    names: List[str] = []
    for x in cv.get("emblem_contributions") or []:
        if not isinstance(x, dict):
            continue
        gt = x.get("granted_traits") or []
        en = str(x.get("equip_name") or "").strip()
        if not en or not gt:
            continue
        if en not in seen:
            seen.add(en)
            names.append(en)
    return sorted(names)


def _print_terminal_report(
    *,
    group: str,
    merged: Dict[str, Any],
    fight_stem: str,
    fight_file: str,
    fightboard_summary: Dict[str, Any],
    cv: Dict[str, Any],
) -> None:
    """
    终端仅输出五类核心信息。
    说明（1）：与 player_recog 一致的数据源为各图 fields.bonds.bond_items（已解析的「人数+羁绊」条目）；
    同组多图按羁绊取人数 max 聚合，不是 bond_summary 原始 OCR 长串的直接合并。
    羁绊名将按 data/rag_legend_traits.jsonl 标准名做归一（截断/近邻纠错）后再与棋盘比对。
    """
    _ = fight_file
    exp = dict(merged.get("trait_count_max") or {})
    bt = cv.get("board_trait_counts") or {}
    ho = cv.get("board_trait_counts_heroes_only") or {}
    eo = cv.get("board_trait_counts_emblems_only") or {}
    pieces = _fightboard_piece_names_in_order(fightboard_summary)
    em_equips = _emblem_grant_equip_names(cv)
    ok = bool(cv.get("consistent"))
    loss = cv.get("trait_loss")

    print("\n" + "=" * 72)
    print(f"[交叉验证] 组={group}  |  主图 fightboard stem={fight_stem}")
    print("（1）玩家羁绊栏聚合（同组主/副图 bond_items 按羁绊取 max 人数；羁绊名按图鉴 rag_legend_traits 归一）")
    print(f"     {_fmt_terminal_trait_counts(exp)}")
    fixes = merged.get("trait_canonicalization_log") or []
    if fixes:
        print(f"     羁绊 OCR→图鉴: {', '.join(fixes)}")
    print("     注：与单图 bond_summary 一行 OCR 不同；聚合仅以 bond_items 为准。")
    print("（2）棋盘棋子名称（fightboard results）")
    print(f"     {', '.join(pieces) if pieces else '（无）'}")
    print("（3）转职/赐羁绊装备（图鉴「获得【X】羁绊」且本局识别到）")
    print(f"     {', '.join(em_equips) if em_equips else '无'}")
    print("（4）棋盘推导羁绊（棋子固有 + 装备赐羁绊，合计）")
    print(f"     {_fmt_terminal_trait_counts(bt if isinstance(bt, dict) else {})}")
    print("     └ 分项 · 仅棋子: " + _fmt_terminal_trait_counts(ho if isinstance(ho, dict) else {}))
    print("     └ 分项 · 仅装备赐: " + _fmt_terminal_trait_counts(eo if isinstance(eo, dict) else {}))
    print("（5）验证（(4) 与 (1) 是否一致）")
    print(f"     {'一致' if ok else '不一致'}  |  trait_loss={loss}  |  status={cv.get('status')}")
    diff = cv.get("trait_diff") or {}
    if diff and not ok:
        print("     差异（羁绊 → 期望人数 / 棋盘人数）:")
        for t, d in sorted(diff.items(), key=lambda kv: kv[0]):
            if not isinstance(d, dict):
                continue
            print(
                f"       · {t}: 期望={d.get('expected')}  棋盘={d.get('board')}  "
                f"Δ(盘−期)={d.get('delta_board_minus_expected')}"
            )
    notes = cv.get("inference_notes") or []
    chg = cv.get("changes") or []
    if notes:
        for ln in notes:
            print(f"     说明: {ln}")
    if chg:
        print("     低置信修正（已写入 confirmed）:")
        for c in chg:
            print(
                f"       bar#{c.get('bar_index')} {c.get('position_label') or ''} "
                f"{c.get('from')} => {c.get('to')}"
            )
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="交叉验证：默认对图片目录先跑 fightboard + player 识别再比对；可用 --from-json 仅读 JSON。"
    )
    ap.add_argument(
        "--img-dir",
        type=Path,
        default=DEFAULT_CV_IMG_DIR,
        help=f"图片输入目录（默认 {DEFAULT_CV_IMG_DIR}）；与 --from-json 互斥",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"输出目录（默认同级下 {{img-dir名}}_cv_out，或 {DEFAULT_CV_OUT_DIR} 在仅 JSON 模式且无 img-dir 时）",
    )
    ap.add_argument(
        "--from-json",
        action="store_true",
        help="不跑识别，改为从 --fightboard-dir / --player-dir 读 summary（旧流程）",
    )
    ap.add_argument("--fightboard-dir", type=Path, default=None, help="--from-json 时：fightboard / 合并 summary 目录")
    ap.add_argument("--player-dir", type=Path, default=None, help="--from-json 时：player / 合并 summary 目录，可与 fightboard-dir 相同")
    ap.add_argument("--no-cache", action="store_true", help="识别模式下禁用 fight/player 缓存")
    ap.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / ".pipeline_cache", help="识别模式缓存目录")
    ap.add_argument(
        "--group-pattern",
        type=str,
        default=r"^(\d+)[-_]",
        help="从文件 stem 提取组号的正则（group(1) 为组号），默认 01-a -> 01",
    )
    ap.add_argument(
        "--fightboard-stem-suffix",
        type=str,
        default="a",
        help="仅对主图 stem（-a/_a 或纯数字）做棋盘交叉验证；设为空字符串则每张 fight 都跑",
    )
    ap.add_argument("--legend-chess", type=Path, default=Path("data/rag_legend_chess.jsonl"))
    ap.add_argument("--legend-equip", type=Path, default=Path("data/rag_legend_equip.jsonl"), help="转职纹章等装备赐羁绊")
    ap.add_argument("--legend-traits", type=Path, default=Path("data/rag_legend_traits.jsonl"))
    args = ap.parse_args()

    project_root = PROJECT_DIR
    legend_chess = args.legend_chess if args.legend_chess.is_absolute() else (project_root / args.legend_chess)
    legend_equip = args.legend_equip if args.legend_equip.is_absolute() else (project_root / args.legend_equip)
    legend_traits = args.legend_traits if args.legend_traits.is_absolute() else (project_root / args.legend_traits)

    if args.from_json:
        fd = args.fightboard_dir
        pd = args.player_dir
        if fd is None or pd is None:
            raise SystemExit("--from-json 时必须同时指定 --fightboard-dir 与 --player-dir")
        fight_dir = fd.resolve()
        player_dir = pd.resolve()
        out_root = (args.out or DEFAULT_CV_OUT_DIR).resolve()
        player_files = _iter_player_json_paths(player_dir)
        fight_files = _iter_fightboard_json_paths(fight_dir)
        if not fight_files:
            raise SystemExit(f"未找到 fightboard 相关 json: {fight_dir}")
        if not player_files:
            raise SystemExit(f"未找到 player 相关 json: {player_dir}")

        group_to_player_files: Dict[str, List[Path]] = defaultdict(list)
        for p in player_files:
            stem = _player_stem_from_path(p)
            gk = _extract_group_key(stem, args.group_pattern)
            group_to_player_files[gk].append(p)

        group_traits_map: Dict[str, Dict[str, Any]] = {}
        for gk, files in sorted(group_to_player_files.items(), key=lambda kv: kv[0]):
            group_traits_map[gk] = _collect_group_player_traits(files, legend_traits)

        fight_by_stem: Dict[str, Dict[str, Any]] = {}
        for f in fight_files:
            stem = _fightboard_stem_from_path(f)
            js = _load_fightboard_module(f)
            if js:
                fight_by_stem[stem] = js

        _run_cross_validate_outputs(
            group_traits_map=group_traits_map,
            fight_by_stem=fight_by_stem,
            out_root=out_root,
            group_pattern=args.group_pattern,
            fightboard_stem_suffix=args.fightboard_stem_suffix,
            legend_chess=legend_chess,
            legend_equip=legend_equip,
            legend_traits=legend_traits,
            fightboard_source="json",
        )
        print(f"完成。输出目录: {out_root}")
        return

    img_dir = args.img_dir.resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"--img-dir 不是目录: {img_dir}")

    if args.out is not None:
        out_root = args.out.resolve()
    else:
        out_root = (img_dir.parent / f"{img_dir.name}_cv_out").resolve()

    run_recognition_then_validate(
        img_dir=img_dir,
        out_root=out_root,
        group_pattern=args.group_pattern,
        fightboard_stem_suffix=args.fightboard_stem_suffix,
        legend_chess=legend_chess,
        legend_equip=legend_equip,
        legend_traits=legend_traits,
        use_cache=not bool(args.no_cache),
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
