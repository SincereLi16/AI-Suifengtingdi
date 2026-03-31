# -*- coding: utf-8 -*-
"""
棋盘装备相性审计：基于 rag_legend_equip、role_tags、棋子图鉴文本。
仅输出「严重」「警告」。

本节上半为图鉴/主词条等共用逻辑（供 equip_recom 同包引用，避免再拆 equip_common）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_RAG_LEGEND_EQUIP_PATH = _PROJECT_ROOT / "data" / "rag_legend_equip.jsonl"

EQUIP_AUDIT_TEAM_ITEMS: frozenset = frozenset(
    {"能量圣杯", "基克的先驱", "静止法衣", "灵风"}
)

# 棋盘审计行默认间距；gemini 传入 BOARD_EMOJI_TEXT_GAP（多为 2 空格）以减轻终端里贴字感
_DEFAULT_AUDIT_EMOJI_GAP = "  "

_LEGEND_EQUIP_FULL_CACHE: Optional[Tuple[Tuple[str, float], Dict[str, Dict[str, Any]]]] = None


def load_legend_equip_full_map(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    global _LEGEND_EQUIP_FULL_CACHE
    p = path or DEFAULT_RAG_LEGEND_EQUIP_PATH
    key = str(p.resolve())
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    ck: Tuple[str, float] = (key, mtime)
    if _LEGEND_EQUIP_FULL_CACHE and _LEGEND_EQUIP_FULL_CACHE[0] == ck:
        return _LEGEND_EQUIP_FULL_CACHE[1]
    out: Dict[str, Dict[str, Any]] = {}
    if p.is_file():
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(o.get("type") or "") != "equip":
                    continue
                nm = str(o.get("name") or "").strip()
                if nm:
                    out[nm] = o
    _LEGEND_EQUIP_FULL_CACHE = (ck, out)
    return out


def parse_game_phase_tuple(phase_raw: str) -> Optional[Tuple[int, int]]:
    s = (phase_raw or "").strip().replace("总", "")
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None


def phase_is_at_least(cur: Optional[Tuple[int, int]], target: Tuple[int, int]) -> bool:
    if cur is None:
        return False
    return cur[0] > target[0] or (cur[0] == target[0] and cur[1] >= target[1])


def equip_audit_excluded(name: str, row: Optional[Dict[str, Any]]) -> bool:
    if not name:
        return True
    if name in EQUIP_AUDIT_TEAM_ITEMS:
        return True
    if "纹章" in name:
        return True
    et = str((row or {}).get("equip_type") or "")
    if et == "转职纹章":
        return True
    return False


def equip_primary_stat_kind(basic_desc: str) -> str:
    s = basic_desc or ""
    for pat, kind in (
        (r"\+(\d+)\s*物理加成", "AD"),
        (r"\+(\d+)\s*法术加成", "AP"),
        (r"\+(\d+)\s*护甲", "TANK"),
        (r"\+(\d+)\s*魔法抗性", "TANK"),
        (r"\+(\d+)\s*生命上限", "TANK"),
        (r"\+(\d+)%\s*攻击速度", "AS"),
        (r"\+(\d+)%\s*暴击率", "CRIT"),
        (r"\+(\d+)\s*法力回复", "MANA"),
        (r"\+(\d+)%\s*伤害增幅", "MIX"),
        (r"\+(\d+)%\s*全能吸血", "MIX"),
        (r"\+(\d+)%\s*伤害减免", "TANK"),
    ):
        if re.search(pat, s):
            return kind
    return "UNK"


def equip_numeric_scores(basic_desc: str) -> Dict[str, float]:
    s = basic_desc or ""
    sc: Dict[str, float] = {"AD": 0.0, "AP": 0.0, "TANK": 0.0, "AS": 0.0, "MANA": 0.0}
    for m in re.finditer(r"\+(\d+)\s*物理加成", s):
        sc["AD"] += float(m.group(1))
    for m in re.finditer(r"\+(\d+)\s*法术加成", s):
        sc["AP"] += float(m.group(1))
    for m in re.finditer(r"\+(\d+)\s*护甲", s):
        sc["TANK"] += float(m.group(1))
    for m in re.finditer(r"\+(\d+)\s*魔法抗性", s):
        sc["TANK"] += float(m.group(1))
    for m in re.finditer(r"\+(\d+)\s*生命上限", s):
        sc["TANK"] += float(m.group(1)) * 0.02
    if re.search(r"法力回复", s):
        sc["MANA"] += 5.0
    return sc


def traits_to_role_tags(traits: Any) -> List[str]:
    if not isinstance(traits, list):
        return []
    tags: List[str] = []
    for t in traits:
        ts = str(t).strip()
        if not ts:
            continue
        if "法师" in ts or ts == "法师":
            tags.append("法师")
        if "射手" in ts or ts == "射手" or "狙神" in ts:
            tags.append("射手")
        if "枪手" in ts or ts == "枪手":
            tags.append("枪手")
        if "刺客" in ts or ts == "刺客":
            tags.append("刺客")
        if "护卫" in ts or "斗士" in ts or "神盾" in ts:
            tags.append("坦克")
    return list(dict.fromkeys(tags))


def role_tags_expect_damage(tags: List[str]) -> Optional[str]:
    if not tags:
        return None
    if "法师" in tags:
        return "AP"
    if "射手" in tags or "刺客" in tags:
        return "AD"
    if "坦克" in tags and "法师" not in tags and "射手" not in tags:
        return "TANK"
    return None


def item_is_penetration(basic_desc: str, desc: str) -> bool:
    blob = f"{basic_desc or ''} {desc or ''}"
    keys = (
        "护甲击碎",
        "魔抗击碎",
        "护甲削减",
        "法术穿透",
        "破甲",
        "法穿",
        "降低护甲值",
        "降低魔抗",
    )
    return any(k in blob for k in keys)


# 描述漏标时仍视为穿透/减抗（与 item_is_penetration 一起用于审计与推荐，避免两套名单分叉）
PENETRATION_ITEM_NAMES: frozenset = frozenset(
    {
        "最后的轻语",
        "虚空之杖",
        "离子火花",
        "薄暮法袍",
        "黑曜石切割者",
        "光明版最后的轻语",
        "光明版虚空之杖",
        "光明版离子火花",
        "光明版薄暮法袍",
    }
)


def item_counts_as_shred(name: str, basic_desc: str, desc: str) -> bool:
    """是否计为破甲/法穿/减抗类（名称或描述）。"""
    if name in PENETRATION_ITEM_NAMES:
        return True
    return item_is_penetration(basic_desc, desc)


def main_c_ap_ad_flags(role_tags: List[str]) -> Tuple[bool, bool, bool]:
    """
    与审计侧一致：主C(AP)、主C(AD)、法射混合（混合时不做属性偏离审计）。
    供 equip_recom 与 audit 使用同一套羁绊判定。
    """
    hybrid = "法师" in role_tags and ("射手" in role_tags or "刺客" in role_tags)
    ap_c = "法师" in role_tags and not hybrid
    ad_c = ("射手" in role_tags or "刺客" in role_tags) and "法师" not in role_tags
    return ap_c, ad_c, hybrid


def primary_is_carry_output(pk: str) -> bool:
    """主词条是否纯输出向（不含 MIX）；用于主坦「纯输出」严重、主C 全防御等严重。"""
    return pk in ("AD", "AP", "AS", "CRIT")


def classify_main_c_piece_kind(name: str, basic_desc: str, desc: str, pk: str) -> str:
    """
    主 C 装备槽语义分类（与 equip_recom 原 _carry_piece_kind 一致，供审计「缺乏启动」与推荐共用）。
    返回：penetration | startup | violent | neutral

    注意：纳什之牙等「普攻叠法力」在 basic_desc 里也会匹配到 AP，若仅按 pk 会误判为暴力；
    鬼索/烁刃等同理（法强主词条 + 攻速叠层）；水银主词条常为魔抗，兜底 violent 会误判双暴力。
    """
    name = (name or "").strip()
    b, d = basic_desc or "", desc or ""
    if item_counts_as_shred(name, b, d):
        return "penetration"
    if name in ("水银", "光明版水银"):
        return "neutral"
    # basic_desc 常以「生命上限」抢先匹配成 TANK，但玩法上属于普攻叠蓝/攻速启动（如纳什之牙）
    if name in ("纳什之牙", "光明版纳什之牙"):
        return "startup"
    if "普攻" in d and ("法力" in d or "法力值" in d):
        return "startup"
    if name in ("鬼索的狂暴之刃", "光明版鬼索的狂暴之刃", "烁刃"):
        return "startup"
    if "每秒" in d and "攻击速度" in d and ("可叠加" in d or "叠加" in d):
        return "startup"
    if name == "黎明核心" or ("最大法力值" in d and "缩减" in d):
        return "startup"
    if pk == "MANA" or "法力回复" in b:
        return "startup"
    if pk in ("AD", "AP", "CRIT", "AS"):
        return "violent"
    if pk == "MIX":
        if "法力回复" in b:
            return "startup"
        return "violent"
    return "violent"


def tank_loadout_is_pure_output(eq_names: List[str], equip_map: Dict[str, Dict[str, Any]]) -> bool:
    """
    是否与审计「主坦 + 至少 2 件参与装备且主词条均为 AD/AP/AS/CRIT」一致（职业错位：纯输出）。
    推荐侧用于：命中该情形时优先推坦装，与严重结论对齐。
    """
    kinds: List[str] = []
    for en in eq_names:
        row = equip_map.get(en)
        if equip_audit_excluded(en, row):
            continue
        bd = str((row or {}).get("basic_desc") or "")
        kinds.append(equip_primary_stat_kind(bd))
    if len(kinds) < 2:
        return False
    return all(primary_is_carry_output(k) for k in kinds)


__all__ = (
    "DEFAULT_RAG_LEGEND_EQUIP_PATH",
    "EQUIP_AUDIT_TEAM_ITEMS",
    "PENETRATION_ITEM_NAMES",
    "load_legend_equip_full_map",
    "traits_to_role_tags",
    "role_tags_expect_damage",
    "parse_game_phase_tuple",
    "phase_is_at_least",
    "item_is_penetration",
    "item_counts_as_shred",
    "main_c_ap_ad_flags",
    "primary_is_carry_output",
    "classify_main_c_piece_kind",
    "tank_loadout_is_pure_output",
    "format_equipment_audit_suffix",
    "format_equipment_audit_terminal_lines",
)


def _parse_mana_pool_max_from_legend_text(text: str) -> Optional[int]:
    if not text:
        return None
    best: Optional[int] = None
    for m in re.finditer(
        r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)",
        text,
    ):
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d, e, f = int(m.group(4)), int(m.group(5)), int(m.group(6))
        left_max = max(a, b, c)
        right_min = min(d, e, f)
        if left_max <= 120 and right_min >= 80:
            cand = max(a, b, c)
            if best is None or cand > best:
                best = cand
    return best


def _carry_is_short_range(role_tags: List[str], legend_text: str) -> bool:
    lt = legend_text or ""
    if "无限射程" in lt:
        return False
    if "狙神" in role_tags or "射手" in role_tags or "枪手" in role_tags:
        return False
    if "法师" in role_tags:
        return False
    return True


def _item_has_mana_or_as(basic_desc: str) -> bool:
    s = basic_desc or ""
    if "法力回复" in s:
        return True
    if re.search(r"%\s*攻击速度", s) or "攻击速度" in s:
        return True
    return False


def _item_has_lifesteal_or_sustain(basic_desc: str, desc: str) -> bool:
    blob = f"{basic_desc or ''} {desc or ''}"
    if re.search(r"吸血|护盾|救主|不可被选取|免疫伤害|伤害减免", blob):
        return True
    return False


def _unique_passive_conflict_key(desc: str) -> Optional[str]:
    d = desc or ""
    if "救主灵刃" in d:
        return "救主灵刃"
    if "技能可以暴击" in d:
        return "技能暴击类"
    return None


def _team_has_penetration_or_shred(equip_map: Dict[str, Dict[str, Any]], all_names: Sequence[str]) -> bool:
    for en in all_names:
        row = equip_map.get(en)
        bd = str((row or {}).get("basic_desc") or "") if row else ""
        dsc = str((row or {}).get("desc") or "") if row else ""
        if item_counts_as_shred(en, bd, dsc):
            return True
    return False


def _audit_equipment_line(
    *,
    hero_display_name: str,
    role_tags: List[str],
    slot_role: str,
    eq_names: List[str],
    equip_map: Dict[str, Dict[str, Any]],
    phase_tuple: Optional[Tuple[int, int]],
    team_all_equip_names: Sequence[str],
    legend_chess_text: str,
) -> List[Tuple[str, str]]:
    hn = (hero_display_name or "").strip() or "该棋子"
    findings: List[Tuple[str, str]] = []
    participating: List[Tuple[str, str, str, Dict[str, float], str]] = []
    for en in eq_names:
        row = equip_map.get(en)
        bd = str((row or {}).get("basic_desc") or "")
        desc = str((row or {}).get("desc") or "")
        if equip_audit_excluded(en, row):
            continue
        pk = equip_primary_stat_kind(bd)
        participating.append((en, bd, desc, equip_numeric_scores(bd), pk))

    ap_f, ad_f, hybrid_f = main_c_ap_ad_flags(role_tags)
    is_hybrid_ap_ad = slot_role == "主C" and hybrid_f
    is_ap_carry = slot_role == "主C" and ap_f
    is_ad_carry = slot_role == "主C" and ad_f

    def _all_primary_kinds() -> List[str]:
        return [p[-1] for p in participating]

    kinds = _all_primary_kinds()
    n_part = len(participating)

    if not is_hybrid_ap_ad:
        if is_ap_carry and n_part >= 2:
            if kinds and all(k == "AD" for k in kinds):
                ad_names = [p[0] for p in participating if p[4] == "AD"]
                if ad_names:
                    findings.append(
                        ("严重", f"属性偏离：{hn}携带{'、'.join(ad_names)}（纯物理向，与法师主C不符）")
                    )
        if is_ad_carry and n_part >= 2:
            if kinds and all(k == "AP" for k in kinds):
                ap_names = [p[0] for p in participating if p[4] == "AP"]
                if ap_names:
                    findings.append(
                        ("严重", f"属性偏离：{hn}携带{'、'.join(ap_names)}（纯法术向，与物理主C不符）")
                    )

    if slot_role == "主坦" and n_part >= 2:
        if kinds and all(primary_is_carry_output(k) for k in kinds):
            out_names = [p[0] for p in participating if primary_is_carry_output(p[4])]
            if out_names:
                findings.append(("严重", f"职业错位：{hn}携带{'、'.join(out_names)}（纯输出向）"))
    if slot_role == "主C" and n_part >= 2:
        if kinds and all(k == "TANK" for k in kinds):
            tank_names = [p[0] for p in participating if p[4] == "TANK"]
            if tank_names:
                findings.append(("严重", f"职业错位：{hn}携带{'、'.join(tank_names)}（纯防御向）"))

    if slot_role != "主C":
        return findings

    violent_count = sum(
        1
        for en, bd, dsc, _, pk in participating
        if classify_main_c_piece_kind(en, bd, dsc, pk) == "violent"
    )
    mana_pool = _parse_mana_pool_max_from_legend_text(legend_chess_text)

    if violent_count >= 2:
        any_mana_as = any(
            _item_has_mana_or_as(bd)
            or classify_main_c_piece_kind(en, bd, dsc, pk) == "startup"
            for en, bd, dsc, _, pk in participating
        )
        if not any_mana_as and mana_pool is not None and mana_pool > 60:
            v_names = [
                en
                for en, bd, dsc, _, pk in participating
                if classify_main_c_piece_kind(en, bd, dsc, pk) == "violent"
            ]
            findings.append(
                ("警告", f"{hn}携带{'、'.join(v_names)}等为输出向且无回蓝|攻速，图鉴蓝条偏高")
            )

    if len(eq_names) >= 3 and phase_is_at_least(phase_tuple, (5, 1)):
        if not _team_has_penetration_or_shred(equip_map, team_all_equip_names):
            findings.append(("警告", f"{hn}满装但全队无破甲|法穿|减抗类装备"))

    if _carry_is_short_range(role_tags, legend_chess_text):
        any_sus = any(
            _item_has_lifesteal_or_sustain(bd, dsc) for _, bd, dsc, _, _ in participating
        )
        if not any_sus and n_part >= 1:
            findings.append(("警告", f"{hn}无吸血|护盾|保命类词条（短射程）"))

    uk_to_names: Dict[str, List[str]] = {}
    for en, bd, dsc, _, _ in participating:
        uk = _unique_passive_conflict_key(dsc)
        if uk:
            uk_to_names.setdefault(uk, []).append(en)
    dup_parts = [f"{k}（{'、'.join(v)}）" for k, v in uk_to_names.items() if len(v) >= 2]
    if dup_parts:
        findings.append(("警告", f"{hn}重复同类唯一被动装备：{' | '.join(dup_parts)}"))

    return findings


def format_equipment_audit_suffix(
    *,
    role_tags: List[str],
    slot_role: str,
    cell_row: Optional[int],
    eq_names: List[str],
    equip_map: Dict[str, Dict[str, Any]],
    phase_raw: str = "",
    team_all_equip_names: Optional[Sequence[str]] = None,
    legend_chess_text: str = "",
    hero_display_name: str = "",
) -> str:
    _ = cell_row
    pt = parse_game_phase_tuple(phase_raw) if phase_raw else None
    team = list(team_all_equip_names) if team_all_equip_names is not None else list(eq_names)
    findings = _audit_equipment_line(
        hero_display_name=hero_display_name,
        role_tags=role_tags,
        slot_role=slot_role,
        eq_names=eq_names,
        equip_map=equip_map,
        phase_tuple=pt,
        team_all_equip_names=team,
        legend_chess_text=legend_chess_text,
    )
    if findings:
        parts = [f"[{lvl}]{msg}" for lvl, msg in findings]
        return "审计:" + "；".join(parts)
    return ""


def format_equipment_audit_terminal_lines(
    *,
    hero_display_name: str,
    role_tags: List[str],
    slot_role: str,
    eq_names: List[str],
    equip_map: Dict[str, Dict[str, Any]],
    phase_raw: str = "",
    team_all_equip_names: Optional[Sequence[str]] = None,
    legend_chess_text: str = "",
    emoji_text_gap: Optional[str] = None,
) -> List[str]:
    """战报第三行：※ 装备佩戴正常… 或 ❗ 严重 / ⚠ 警告（可多行），文案含棋子名与具体装备。"""
    gap = emoji_text_gap if emoji_text_gap is not None else _DEFAULT_AUDIT_EMOJI_GAP
    pt = parse_game_phase_tuple(phase_raw) if phase_raw else None
    team = list(team_all_equip_names) if team_all_equip_names is not None else list(eq_names)
    findings = _audit_equipment_line(
        hero_display_name=hero_display_name,
        role_tags=role_tags,
        slot_role=slot_role,
        eq_names=eq_names,
        equip_map=equip_map,
        phase_tuple=pt,
        team_all_equip_names=team,
        legend_chess_text=legend_chess_text,
    )
    if not findings:
        return [
            f"※{gap}装备佩戴正常（与棋子定位相符，未见明显错配）"
        ]
    out: List[str] = []
    for lvl, msg in findings:
        if lvl == "严重":
            out.append(f"❗{gap}严重：{msg}")
        else:
            out.append(f"⚠{gap}警告：{msg}")
    return out
