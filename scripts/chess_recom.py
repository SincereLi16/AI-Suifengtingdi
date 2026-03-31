# -*- coding: utf-8 -*-
"""
棋子智库结构化块：Top1 基准、目标高费缺口、可替挂件、装备继承（挂件成装 → 万金油可佩戴）。
供 gemini_v1 等入口调用；逻辑与数据与战术快报解耦，便于单独维护。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAG_LEGEND_CHESS = _PROJECT_ROOT / "data" / "rag_legend_chess.jsonl"

_CORE_CHESS_LIST_CACHE: Optional[Tuple[Tuple[str, float], List[Dict[str, Any]]]] = None
_LEGEND_CHESS_NAME_CACHE: Optional[Tuple[Tuple[str, float], Dict[str, Dict[str, Any]]]] = None


def _load_lineup_docs(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_core_chess_list(path: Path) -> List[Dict[str, Any]]:
    """rag_core_chess.jsonl 全量 core_chess 行（带缓存；随文件 mtime 失效）。"""
    global _CORE_CHESS_LIST_CACHE
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    ck: Tuple[str, float] = (key, mtime)
    if _CORE_CHESS_LIST_CACHE and _CORE_CHESS_LIST_CACHE[0] == ck:
        return _CORE_CHESS_LIST_CACHE[1]
    docs = _load_lineup_docs(path)
    core = [
        d
        for d in docs
        if str(d.get("type") or "") == "core_chess"
        or str(d.get("id") or "").startswith("core_chess:")
    ]
    _CORE_CHESS_LIST_CACHE = (ck, core)
    return core


def load_legend_chess_name_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """rag_legend_chess.jsonl：棋子名 → 整行（费用等）。"""
    global _LEGEND_CHESS_NAME_CACHE
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    ck: Tuple[str, float] = (key, mtime)
    if _LEGEND_CHESS_NAME_CACHE and _LEGEND_CHESS_NAME_CACHE[0] == ck:
        return _LEGEND_CHESS_NAME_CACHE[1]
    out: Dict[str, Dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(o.get("type") or "") != "chess":
                    continue
                nm = str(o.get("name") or "").strip()
                if nm:
                    out[nm] = o
    _LEGEND_CHESS_NAME_CACHE = (ck, out)
    return out


def lookup_core_chess_row(name_key: str, core: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """按棋盘名匹配棋子库一条（精确 > 子串）。"""
    nk = (name_key or "").strip().rstrip("?").rstrip("？").strip()
    if len(nk) < 1:
        return None
    best_sc = 0
    best_d: Optional[Dict[str, Any]] = None
    for d in core:
        cn = str(d.get("chess_name") or "")
        if cn == nk:
            sc = 10
        elif len(nk) >= 2 and (nk in cn or cn in nk):
            sc = 5
        else:
            sc = 0
        if sc > best_sc:
            best_sc = sc
            best_d = d
    return best_d if best_sc > 0 else None


def chess_cost_from_core_or_legend(
    lookup_key: str,
    core: List[Dict[str, Any]],
    legend_map: Dict[str, Dict[str, Any]],
) -> Optional[int]:
    """棋子费用：优先 core_chess，否则 rag_legend_chess。"""
    nk = lookup_key.rstrip("?").rstrip("？").strip()
    dr = lookup_core_chess_row(nk, core)
    if dr is not None and dr.get("cost") is not None:
        try:
            return int(dr.get("cost"))
        except (TypeError, ValueError):
            pass
    leg = legend_map.get(nk)
    if isinstance(leg, dict) and leg.get("cost") is not None:
        try:
            return int(leg.get("cost"))
        except (TypeError, ValueError):
            pass
    return None


def slot_role_and_cost_for_name(
    lookup_key: str,
    core: List[Dict[str, Any]],
    legend_map: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[int]]:
    """与战术快报棋盘行一致：core 有 meta.slot_role；仅图鉴时视为挂件。"""
    nk = lookup_key.rstrip("?").rstrip("？").strip()
    cost = chess_cost_from_core_or_legend(nk, core, legend_map)
    dr = lookup_core_chess_row(nk, core)
    if isinstance(dr, dict):
        mm = dr.get("meta") if isinstance(dr.get("meta"), dict) else {}
        rl = str(mm.get("slot_role") or "").strip()
        if rl:
            return rl, cost
    if nk in legend_map:
        return "挂件", cost
    return "?", cost


def board_line_hero_display_name(r: Dict[str, Any]) -> str:
    """单格棋子展示名：confirmed_fightboard_results 每行 best；低置信度加 ?。"""
    if not isinstance(r, dict):
        return "?"
    name = str(r.get("best") or "?")
    conf = str(r.get("confidence") or "")
    if conf == "low":
        name = f"{name}?"
    return name


def tcv_board_hero_names_for_rag(summary: Dict[str, Any]) -> List[str]:
    """与战术快报棋盘行同源的棋子名列表（去重保序）。"""
    rows = summary.get("confirmed_fightboard_results")
    if not isinstance(rows, list):
        return []
    out: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        disp = board_line_hero_display_name(r)
        key = disp.rstrip("?").rstrip("？").strip()
        if key and key not in ("?", "？"):
            out.append(key)
    return list(dict.fromkeys(out))


def parse_lineup_final_hero_names(text: str) -> List[str]:
    """从阵容攻略 text 的【成型站位与出装】段解析棋子名（去重保序）。"""
    if "【成型站位与出装】" not in text:
        return []
    rest = text.split("【成型站位与出装】", 1)[1]
    for sep in ("【主C装备替换】", "【棋子替换】", "【可追三星】"):
        if sep in rest:
            rest = rest.split(sep, 1)[0]
    rest = rest.split("【", 1)[0]
    out: List[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[；;]", rest):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("【"):
            break
        if "位置" not in chunk:
            continue
        head = chunk.split("位置", 1)[0].strip()
        head = re.sub(r"（[^）]*）\s*$", "", head).strip()
        if not head or head.startswith("英雄"):
            continue
        if head not in seen:
            seen.add(head)
            out.append(head)
    return out


def hero_name_matches_board(hero: str, board_names: List[str]) -> bool:
    h = hero.strip()
    if not h:
        return False
    for b in board_names:
        bk = b.strip().rstrip("?").rstrip("？").strip()
        if not bk:
            continue
        if h == bk or (len(h) >= 2 and (h in bk or bk in h)):
            return True
    return False


def equip_names_for_bar(summary: Dict[str, Any], bar_index: int) -> List[str]:
    """战术快报 fightboard.equip_by_bar 中该 bar 的成装名列表。"""
    fight = (summary.get("modules") or {}).get("fightboard") or {}
    eq_by = fight.get("equip_by_bar") or {}
    raw = eq_by.get(str(bar_index))
    if raw is None:
        raw = eq_by.get(bar_index)
    names: List[str] = []
    if isinstance(raw, list):
        for e in raw:
            if isinstance(e, dict):
                en = str(e.get("name") or "").strip()
                if en:
                    names.append(en)
    return names


def core_wanyou_rows(core: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """meta.slot_role == 万金油 的 core 行。"""
    out: List[Dict[str, Any]] = []
    for d in core:
        mm = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        if str(mm.get("slot_role") or "").strip() != "万金油":
            continue
        out.append(d)
    return out


def board_worker_support_with_equips(
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    core: List[Dict[str, Any]],
    legend_map: Dict[str, Dict[str, Any]],
    star_by_bar: Dict[int, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    场上定位 ∈ {打工仔, 挂件} 的棋子，并拆成「有成装」与「无成装」两组。
    """
    with_eq: List[Dict[str, Any]] = []
    no_eq: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        disp = board_line_hero_display_name(r)
        lk = disp.rstrip("?").rstrip("？").strip()
        slot_tag, cost = slot_role_and_cost_for_name(lk, core, legend_map)
        if slot_tag not in ("打工仔", "挂件"):
            continue
        try:
            bi = int(r.get("bar_index") or 0)
        except (TypeError, ValueError):
            bi = 0
        eqs = equip_names_for_bar(summary, bi)
        pos = r.get("position") or {}
        loc = "?"
        if isinstance(pos, dict):
            cr, cc = pos.get("cell_row"), pos.get("cell_col")
            if cr is not None and cc is not None:
                loc = f"R{int(cr)}C{int(cc)}"
            else:
                lab = str(pos.get("label") or "").strip()
                loc = lab if lab else "?"
        star_seg = star_display_segment(r, star_by_bar)
        cs = f"{int(cost)}费" if cost is not None else "?费"
        entry: Dict[str, Any] = {
            "display": disp,
            "lookup_key": lk,
            "slot_role": slot_tag,
            "cost": cost,
            "cost_str": cs,
            "star_seg": star_seg,
            "loc": loc,
            "bar_index": bi,
            "equips": eqs,
        }
        if eqs:
            with_eq.append(entry)
        else:
            no_eq.append(entry)
    return with_eq, no_eq


def equipment_inheritance_block(
    guajia_entries: List[Dict[str, Any]],
    core: List[Dict[str, Any]],
    board_name_list: List[str],
    *,
    board_carries: Optional[List[Tuple[str, str]]] = None,
    missing_ge4: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    三、装备继承：仅列出「主力核心或目标缺口」在万金油库中可承接的挂件成装；一行一件：成装名 + 当前佩戴者 → 预期佩戴者。
    """
    lines: List[str] = []
    wanyou = core_wanyou_rows(core)
    board_carries = board_carries or []
    missing_ge4 = missing_ge4 or []
    missing_map: Dict[str, str] = {}
    for e in missing_ge4:
        if not isinstance(e, dict):
            continue
        h = str(e.get("chess_name") or "").strip()
        if h:
            missing_map[h] = str(e.get("slot_role") or "?")

    meta: Dict[str, Any] = {
        "guajia_carried_equips": [],
        "by_equip_wearers": [],
        "flows": [],
    }

    lines.append("三、装备继承")
    if not guajia_entries:
        lines.append("  （场上无挂件位。）")
        return lines, meta

    for gu in guajia_entries:
        eqs = gu.get("equips") or []
        meta["guajia_carried_equips"].append({"display": gu["display"], "equips": eqs})

    all_eq: List[str] = []
    for gu in guajia_entries:
        for e in gu.get("equips") or []:
            es = str(e).strip()
            if es and es not in all_eq:
                all_eq.append(es)

    if not wanyou:
        lines.append("  （core 无万金油条目，无法匹配可佩戴棋子。）")
        return lines, meta

    if not all_eq:
        lines.append("  （挂件均未携带成装。）")
        return lines, meta

    by_eq: Dict[str, List[str]] = {}
    for eq in all_eq:
        names: List[str] = []
        for d in wanyou:
            cn = str(d.get("chess_name") or "").strip()
            if not cn:
                continue
            mm = d.get("meta") if isinstance(d.get("meta"), dict) else {}
            rec = mm.get("recommended_equips") if isinstance(mm.get("recommended_equips"), list) else []
            rec_set = {str(x).strip() for x in rec if str(x).strip()}
            if eq in rec_set:
                tag = " [已在场]" if hero_name_matches_board(cn, board_name_list) else ""
                names.append(f"{cn}{tag}")
        if names:
            by_eq[eq] = names

    if not by_eq:
        lines.append("  （上述成装与万金油库内推荐装无交集。）")
        return lines, meta

    def _strip_in_board(s: str) -> str:
        return s.replace(" [已在场]", "").strip()

    def _label_receiver(raw_name: str) -> Tuple[int, str]:
        """(排序键, 展示标签)；3=仅万金油且非场上主C/主坦、非目标缺口。"""
        rn = _strip_in_board(raw_name)
        for disp, role in board_carries:
            if hero_name_matches_board(rn, [disp]):
                tier = 0 if role == "主C" else 1
                return (tier, f"{rn}({role})")
        for mk, mrole in missing_map.items():
            if rn == mk or hero_name_matches_board(rn, [mk]):
                return (2, f"{rn}(目标·{mrole})")
        on_board = hero_name_matches_board(rn, board_name_list)
        ob = "已在场" if on_board else "未上场"
        return (3, f"{rn}(万金油·{ob})")

    for eq in all_eq:
        if eq not in by_eq:
            continue
        cand_labels: List[Tuple[int, str]] = [_label_receiver(x) for x in by_eq[eq]]
        inheritable = [x for x in cand_labels if x[0] <= 2]
        if not inheritable:
            continue
        inheritable.sort(key=lambda x: (x[0], x[1]))
        to_best = inheritable[0][1]

        holders: List[str] = []
        for gu in guajia_entries:
            eqs = gu.get("equips") or []
            if eq in eqs:
                holders.append(str(gu.get("display") or "").strip())
        from_s = "、".join(dict.fromkeys(holders)) if holders else "（未知）"

        flow_line = f"  - {eq}：{from_s} → {to_best}"
        lines.append(flow_line)
        meta["flows"].append(
            {
                "equip": eq,
                "from": from_s,
                "to": [x[1] for x in inheritable],
                "to_primary": to_best,
                "line": f"{eq}：{from_s} → {to_best}",
            }
        )

    if not meta["flows"]:
        lines.append("  （无可继承成装：场上主力核心与目标棋子均无法承接挂件当前成装。）")

    meta["by_equip_wearers"] = [{"equip": k, "chess_names": v} for k, v in by_eq.items()]
    return lines, meta


def _star_level_from_star_obj(st: Any) -> Optional[int]:
    if not isinstance(st, dict):
        return None
    for key in ("pred", "pred_raw"):
        pr = st.get(key)
        if pr is None:
            continue
        try:
            n = int(float(pr))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= 3:
            return n
    return None


def _fightboard_star_index_from_results(rs: Any) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(rs, list):
        return out
    for r in rs:
        if not isinstance(r, dict):
            continue
        try:
            bi = int(r.get("bar_index") or -1)
        except (TypeError, ValueError):
            continue
        st = r.get("star")
        if bi >= 0 and isinstance(st, dict):
            out[bi] = st
    return out


def _load_fightboard_sidecar_star_index(
    summary: Dict[str, Any],
    summary_json_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    fn = str(summary.get("file") or "").strip()
    if not fn:
        return {}
    stem = Path(fn).stem
    name = f"{stem}_fightboard_summary.json"
    candidates: List[Path] = []
    if summary_json_path is not None:
        candidates.append(summary_json_path.resolve().parent / name)
    candidates.append(_PROJECT_ROOT / "runs" / "fightboard_info_v2" / name)
    for p in candidates:
        if not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fb = (raw.get("modules") or {}).get("fightboard") or raw
        return _fightboard_star_index_from_results(fb.get("results"))
    return {}


def fightboard_star_by_bar(
    summary: Dict[str, Any],
    *,
    summary_json_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    fb = (summary.get("modules") or {}).get("fightboard") or {}
    out = _fightboard_star_index_from_results(fb.get("results"))
    side = _load_fightboard_sidecar_star_index(summary, summary_json_path=summary_json_path)
    for bi, st in side.items():
        cur = out.get(bi)
        if _star_level_from_star_obj(cur) is not None:
            continue
        if _star_level_from_star_obj(st) is not None:
            out[bi] = st
    return out


def star_display_segment(r: Dict[str, Any], star_by_bar: Dict[int, Dict[str, Any]]) -> str:
    st: Any = r.get("star")
    if not isinstance(st, dict):
        try:
            bi = int(r.get("bar_index") or -1)
        except (TypeError, ValueError):
            bi = -1
        if bi >= 0:
            st = star_by_bar.get(bi)
    n = _star_level_from_star_obj(st)
    if n is not None:
        return f"{n}星"
    return "?星"


def retrieve_core_chess_rag(
    summary: Dict[str, Any],
    rag_path: Path,
    top_k: int = 8,
    *,
    lineup_top_doc: Optional[Dict[str, Any]] = None,
    legend_chess_path: Optional[Path] = None,
    summary_json_path: Optional[Path] = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    棋子智库：不逐子复述快报已有棋子与推荐装；仅输出与「阵容 Top1」对齐的结构化块。
    返回 (文本块, id 列表占位, meta 列表)。
    """
    _ = top_k
    core = load_core_chess_list(rag_path)
    if not core:
        return ("（棋子智库 RAG 库未加载或为空。）", [], [])

    leg_path = legend_chess_path or DEFAULT_RAG_LEGEND_CHESS
    legend_map = load_legend_chess_name_map(leg_path)
    board_name_list = tcv_board_hero_names_for_rag(summary)
    star_by_bar = fightboard_star_by_bar(summary, summary_json_path=summary_json_path)

    rows = summary.get("confirmed_fightboard_results")
    if not isinstance(rows, list):
        rows = []

    blocks: List[str] = []
    ids: List[str] = []
    meta: List[Dict[str, Any]] = []

    has_top1 = bool(lineup_top_doc and isinstance(lineup_top_doc, dict))
    missing_ge4: List[Dict[str, Any]] = []
    missing_lt4: List[Dict[str, Any]] = []
    missing_unknown_cost: List[Dict[str, Any]] = []
    guajia_entries: List[Dict[str, Any]] = []
    guajia_names: List[str] = []
    ltid = ""
    lname = ""

    for r in rows:
        if not isinstance(r, dict):
            continue
        disp = board_line_hero_display_name(r)
        lk = disp.rstrip("?").rstrip("？").strip()
        slot_tag, cost = slot_role_and_cost_for_name(lk, core, legend_map)
        if slot_tag != "挂件":
            continue
        try:
            bi = int(r.get("bar_index") or 0)
        except (TypeError, ValueError):
            bi = 0
        eqs = equip_names_for_bar(summary, bi)
        pos = r.get("position") or {}
        loc = "?"
        if isinstance(pos, dict):
            cr, cc = pos.get("cell_row"), pos.get("cell_col")
            if cr is not None and cc is not None:
                loc = f"R{int(cr)}C{int(cc)}"
            else:
                lab = str(pos.get("label") or "").strip()
                loc = lab if lab else "?"
        star_seg = star_display_segment(r, star_by_bar)
        cs = f"{cost}费" if cost is not None else "?费"
        guajia_entries.append(
            {
                "display": disp,
                "equips": eqs,
                "star_seg": star_seg,
                "cost_str": cs,
                "loc": loc,
                "bar_index": bi,
            }
        )
        guajia_names.append(disp)

    guajia_meta_rows: List[Dict[str, Any]] = [
        {
            "display": gu["display"],
            "star_seg": gu["star_seg"],
            "cost_str": gu["cost_str"],
            "slot_role": "挂件",
        }
        for gu in guajia_entries
    ]

    if has_top1:
        doc = lineup_top_doc or {}
        ltid = str(doc.get("lineup_id") or "")
        lname = str(doc.get("name") or ltid)
        ltext = str(doc.get("text") or "")
        need_names = parse_lineup_final_hero_names(ltext)
        missing = [h for h in need_names if not hero_name_matches_board(h, board_name_list)]

        for h in missing:
            c = chess_cost_from_core_or_legend(h, core, legend_map)
            rl, _ = slot_role_and_cost_for_name(h, core, legend_map)
            entry = {"chess_name": h, "cost": c, "slot_role": rl}
            if c is not None and c >= 4:
                missing_ge4.append(entry)
            elif c is not None and c < 4:
                missing_lt4.append(entry)
            else:
                missing_unknown_cost.append(entry)

        missing_ge4.sort(key=lambda x: (-(x.get("cost") or 0), str(x.get("chess_name") or "")))
        missing_lt4.sort(key=lambda x: (-(x.get("cost") or 0), str(x.get("chess_name") or "")))

        blocks.append("【棋子智库】")
        blocks.append("")
        blocks.append("基准阵容（阵容智库 Top1）")
        blocks.append(f"  · lineup_id: {ltid}")
        blocks.append(f"  · 名称: {lname}")
        blocks.append("")
        blocks.append("一、目标棋子")
        blocks.append("  说明：目标阵容所缺乏的高费棋子 / 万金油棋子。")
        if missing_ge4:
            for e in missing_ge4:
                h = str(e.get("chess_name") or "")
                c = e.get("cost")
                rl = str(e.get("slot_role") or "?")
                cs = f"{int(c)}费" if c is not None else "?费"
                blocks.append(f"  - {h} | {cs} | {rl}")
        else:
            blocks.append(
                "  （无：均已上场，或缺口仅为低费/未解析费用，或无法解析【成型站位与出装】）"
            )
        if missing_unknown_cost:
            blocks.append("  · 费用未在库中解析的缺口棋子（请对照图鉴或阵容原文）：")
            for e in missing_unknown_cost:
                h = str(e.get("chess_name") or "")
                rl = str(e.get("slot_role") or "?")
                blocks.append(f"    - {h} | {rl}")
        blocks.append("")
        blocks.append("二、可替挂件")
        if guajia_entries:
            for gu in guajia_entries:
                lines = f"  - {gu['display']} | {gu['star_seg']} | {gu['cost_str']} | {gu['loc']}"
                blocks.append(lines)
        else:
            blocks.append("  （无：场上未识别出挂件定位，或均为核心/打工等）")
        blocks.append("")

        if missing_lt4:
            blocks.append("（参考）其他未上场且费用<4 的棋子")
            for e in missing_lt4:
                h = str(e.get("chess_name") or "")
                c = e.get("cost")
                rl = str(e.get("slot_role") or "?")
                cs = f"{int(c)}费" if c is not None else "?费"
                blocks.append(f"  - {h} | {cs} | {rl}")
            blocks.append("")
    else:
        blocks.append("【棋子智库】")
        blocks.append("")
        blocks.append("未命中阵容智库 Top1，一、二及目标缺口从略。")
        blocks.append("")

    board_carries: List[Tuple[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        disp = board_line_hero_display_name(r)
        lk = disp.rstrip("?").rstrip("？").strip()
        slot_tag, _ = slot_role_and_cost_for_name(lk, core, legend_map)
        if slot_tag in ("主C", "主坦"):
            board_carries.append((disp, slot_tag))

    w_lines, w_meta = equipment_inheritance_block(
        guajia_entries,
        core,
        board_name_list,
        board_carries=board_carries,
        missing_ge4=missing_ge4,
    )
    blocks.extend(w_lines)

    meta.append(
        {
            "kind": "chess_structured",
            "top1_lineup_id": ltid or None,
            "top1_name": lname or None,
            "missing_ge4": missing_ge4,
            "missing_lt4": missing_lt4,
            "missing_unknown_cost": missing_unknown_cost,
            "board_guajia": guajia_meta_rows,
            "equipment_inheritance": w_meta,
        }
    )

    return ("\n".join(blocks), ids, meta)


__all__ = (
    "DEFAULT_RAG_LEGEND_CHESS",
    "board_line_hero_display_name",
    "board_worker_support_with_equips",
    "chess_cost_from_core_or_legend",
    "core_wanyou_rows",
    "equip_names_for_bar",
    "fightboard_star_by_bar",
    "hero_name_matches_board",
    "load_core_chess_list",
    "load_legend_chess_name_map",
    "lookup_core_chess_row",
    "parse_lineup_final_hero_names",
    "retrieve_core_chess_rag",
    "slot_role_and_cost_for_name",
    "star_display_segment",
    "tcv_board_hero_names_for_rag",
    "equipment_inheritance_block",
)
