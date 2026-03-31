# -*- coding: utf-8 -*-
"""
基于当前已携带成装的下一件 / 多件推荐（与 equip_audit 职责分离）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from scripts.extra.equip_audit import (
    classify_main_c_piece_kind,
    equip_audit_excluded,
    equip_primary_stat_kind,
    item_counts_as_shred,
    main_c_ap_ad_flags,
    role_tags_expect_damage,
    tank_loadout_is_pure_output,
)

# 名称级「启动/叠蓝/减最大法力」：勿再当作纯 AD/AP 暴力输出推荐。
# 与 need==startup 中「法力回复 / MANA / 描述减蓝」互补；朔极/蓝霸符等 pk 常为 AD 首位，violent_ad 必须排除。
_STARTUP_ITEM_NAMES: frozenset = frozenset(
    {
        "纳什之牙",
        "光明版纳什之牙",
        "鬼索的狂暴之刃",
        "光明版鬼索的狂暴之刃",
        "烁刃",
        "朔极之矛",
        "光明版朔极之矛",
        "蓝霸符",
        "光明版蓝霸符",
        "大天使之杖",
        "光明版大天使之杖",
        "适应性头盔",
        "光明版适应性头盔",
        "黎明核心",
        "海克斯科技枪刃",
        "光明版海克斯科技枪刃",
        "正义之手",
        "光明版正义之手",
    }
)


def _worn_has_startup_item(eq_names: List[str]) -> bool:
    for en in eq_names:
        n = (en or "").strip()
        if n in _STARTUP_ITEM_NAMES:
            return True
    return False


__all__ = ("pick_next_finished_recommendations",)


# ---------- 主 C：分类（与 equip_audit.classify_main_c_piece_kind 同源） ----------
def _carry_worn_kinds(eq_names: List[str], equip_map: Dict[str, Dict[str, Any]]) -> List[str]:
    kinds: List[str] = []
    for en in eq_names:
        row = equip_map.get(en)
        if equip_audit_excluded(en, row):
            continue
        bd = str((row or {}).get("basic_desc") or "")
        desc = str((row or {}).get("desc") or "")
        pk = equip_primary_stat_kind(bd)
        kinds.append(classify_main_c_piece_kind(en, bd, desc, pk))
    return kinds


# ---------- 主坦：分类 ----------
_TANK_FUNCTIONAL_NAMES: Set[str] = {
    "救赎",
    "日炎斗篷",
    "离子火花",
    "圣盾使的誓约",
    "钢铁烈阳之匣",
    "冕卫",
}


def _hp_from_bd(bd: str) -> int:
    s = bd or ""
    m = re.search(r"\+(\d+)\s*生命上限", s)
    return int(m.group(1)) if m else 0


def _armor_from_bd(bd: str) -> int:
    s = bd or ""
    m = re.search(r"\+(\d+)\s*护甲", s)
    return int(m.group(1)) if m else 0


def _mr_from_bd(bd: str) -> int:
    s = bd or ""
    m = re.search(r"\+(\d+)\s*魔法抗性", s)
    return int(m.group(1)) if m else 0


def _tank_piece_kind(name: str, bd: str, desc: str, pk: str) -> str:
    """
    hp | resist | functional | mixed_hp_resist
    mixed：单件上同时有显著生命+护甲/魔抗（如冕卫），用于与另一件组合时避免误判为「纯血」。
    """
    if name in _TANK_FUNCTIONAL_NAMES:
        return "functional"
    hp = _hp_from_bd(bd)
    ar = _armor_from_bd(bd)
    mr = _mr_from_bd(bd)
    blob = f"{bd} {desc}"
    if "救赎" in name or "日炎" in name or "离子" in name:
        return "functional"
    if "护盾" in blob and ("邻格" in blob or "友军" in blob):
        return "functional"

    if hp >= 200 and ar + mr <= 40:
        return "hp"
    if hp >= 80 and (ar >= 15 or mr >= 15):
        return "mixed_hp_resist"
    if ar >= 20 or mr >= 20:
        return "resist"
    if hp >= 100:
        return "hp"
    return "resist"


def _tank_worn_tags(eq_names: List[str], equip_map: Dict[str, Dict[str, Any]]) -> List[str]:
    tags: List[str] = []
    for en in eq_names:
        row = equip_map.get(en)
        if equip_audit_excluded(en, row):
            continue
        bd = str((row or {}).get("basic_desc") or "")
        desc = str((row or {}).get("desc") or "")
        pk = equip_primary_stat_kind(bd)
        tags.append(_tank_piece_kind(en, bd, desc, pk))
    return tags


# ---------- 候选池（与 meta_rec 交集优先） ----------
def _item_match_carry_need(
    name: str,
    row: Optional[Dict[str, Any]],
    need: str,
    *,
    ap_carry: bool,
) -> bool:
    if not row or equip_audit_excluded(name, row):
        return need == "any"
    bd = str(row.get("basic_desc") or "")
    desc = str(row.get("desc") or "")
    pk = equip_primary_stat_kind(bd)
    if need == "startup":
        nm = (name or "").strip()
        if nm in _STARTUP_ITEM_NAMES:
            return True
        if pk == "MANA" or "法力回复" in bd:
            return True
        # 黎明核心等：basic_desc 无双回蓝词条，仅靠描述标减最大法力（与 equip_audit.classify 一致）
        blob = f"{bd} {desc}"
        if "最大法力值" in blob and "缩减" in blob:
            return True
        return False
    if need == "as":
        return pk == "AS" or "攻击速度" in bd
    if need == "shred":
        return item_counts_as_shred(name, bd, desc)
    if need == "violent_ap":
        if not ap_carry:
            return False
        # 朔极之矛等同时带物/法小件词条，仅靠「法术加成」子串会误判为暴力输出
        if _item_match_carry_need(name, row, "startup", ap_carry=ap_carry):
            return False
        return pk in ("AP", "CRIT") or "法术加成" in bd
    if need == "violent_ad":
        if ap_carry:
            return False
        # 朔极/蓝霸符/正义之手等 pk 常为 AD 或 CRIT 首位，易被当成「补伤害」；须与 startup 一致排除
        if _item_match_carry_need(name, row, "startup", ap_carry=ap_carry):
            return False
        return pk in ("AD", "CRIT", "AS")
    if need == "violent":
        if _item_match_carry_need(name, row, "startup", ap_carry=ap_carry):
            return False
        if ap_carry:
            return pk in ("AP", "CRIT", "AS", "MIX") or "法术加成" in bd
        return pk in ("AD", "CRIT", "AS", "MIX") or "物理加成" in bd
    if need == "violent_or_shred":
        return _item_match_carry_need(name, row, "shred", ap_carry=ap_carry) or _item_match_carry_need(
            name, row, "violent_ap" if ap_carry else "violent_ad", ap_carry=ap_carry
        )
    if need == "startup_or_as":
        return _item_match_carry_need(name, row, "startup", ap_carry=ap_carry) or _item_match_carry_need(
            name, row, "as", ap_carry=ap_carry
        )
    return True


def _item_match_tank_need(name: str, row: Optional[Dict[str, Any]], need: str) -> bool:
    if not row or equip_audit_excluded(name, row):
        return need == "any"
    bd = str(row.get("basic_desc") or "")
    desc = str(row.get("desc") or "")
    pk = equip_primary_stat_kind(bd)
    if need == "hp":
        return "生命上限" in bd or pk == "TANK" and _hp_from_bd(bd) >= 150
    if need == "resist":
        return "护甲" in bd or "魔法抗性" in bd or "魔抗" in desc
    if need == "resist_or_dr":
        return (
            _item_match_tank_need(name, row, "resist")
            or "伤害减免" in bd
            or "坚定之心" in name
            or "振奋盔甲" in name
        )
    if need == "func_or_grievous":
        if name in ("救赎", "日炎斗篷", "莫雷洛秘典", "离子火花", "圣盾使的誓约"):
            return True
        return "重伤" in desc or "灼烧" in desc
    return True


def _fallback_carry(ap_carry: bool) -> Dict[str, List[str]]:
    return {
        "startup": ["朔极之矛", "蓝霸符", "大天使之杖"],
        "as": ["鬼索的狂暴之刃", "红霸符", "纳什之牙"],
        "startup_or_as": ["朔极之矛", "蓝霸符", "鬼索的狂暴之刃"],
        "violent_ap": ["珠光护手", "班克斯的魔法帽", "大天使之杖"],
        "violent_ad": ["无尽之刃", "巨人捕手", "锐利之刃"],
        "shred": ["最后的轻语", "虚空之杖", "薄暮法袍"],
        "violent": ["珠光护手", "无尽之刃", "巨人捕手"] if not ap_carry else ["珠光护手", "班克斯的魔法帽", "大天使之杖"],
        "violent_or_shred": ["最后的轻语", "虚空之杖", "珠光护手", "无尽之刃"],
    }


def _fallback_tank() -> Dict[str, List[str]]:
    return {
        "hp": ["狂徒铠甲"],
        "resist": ["石像鬼石板甲", "巨龙之爪", "棘刺背心"],
        "resist_or_dr": ["石像鬼石板甲", "巨龙之爪", "坚定之心", "振奋盔甲"],
        "func_or_grievous": ["救赎", "日炎斗篷", "莫雷洛秘典"],
    }


def _collect_candidates(
    *,
    pool: List[str],
    worn: Set[str],
    equip_map: Dict[str, Dict[str, Any]],
    needs: List[str],
    ap_carry: bool,
    tank: bool,
) -> List[str]:
    out: List[str] = []
    for x in pool:
        if x in worn or x in out:
            continue
        row = equip_map.get(x)
        for need in needs:
            ok = (
                _item_match_tank_need(x, row, need)
                if tank
                else _item_match_carry_need(x, row, need, ap_carry=ap_carry)
            )
            if ok:
                out.append(x)
                break
    return out


def _fill_from_fallback(
    worn: Set[str],
    equip_map: Dict[str, Dict[str, Any]],
    needs: List[str],
    ap_carry: bool,
    tank: bool,
    cap: int,
    existing: List[str],
) -> List[str]:
    out = list(existing)
    fb = _fallback_tank() if tank else _fallback_carry(ap_carry)
    for need in needs:
        for name in fb.get(need, []):
            if len(out) >= cap:
                return out
            if name in worn or name in out:
                continue
            row = equip_map.get(name)
            if tank:
                ok = _item_match_tank_need(name, row, need)
            else:
                ok = _item_match_carry_need(name, row, need, ap_carry=ap_carry)
            if ok:
                out.append(name)
    return out[:cap]


def pick_next_finished_recommendations(
    *,
    role_tags: List[str],
    slot_role: str,
    eq_names: List[str],
    n_eq: int,
    meta_rec: List[str],
    equip_map: Dict[str, Dict[str, Any]],
) -> List[str]:
    if n_eq >= 3:
        return []
    need = max(0, 3 - n_eq)
    worn = set(eq_names)
    pool = [str(x).strip() for x in meta_rec if str(x).strip()]

    ap_f, ad_f, hybrid_f = main_c_ap_ad_flags(role_tags)
    ap_carry = ap_f or hybrid_f
    ad_carry = ad_f
    exp = role_tags_expect_damage(role_tags)

    # ---------- 主坦 ----------
    if slot_role == "主坦" or "坦克" in role_tags:
        # 与审计「职业错位：主坦纯输出」一致时，优先补坦度（避免审计报严重而推荐仍在走功能/节奏）
        if tank_loadout_is_pure_output(eq_names, equip_map):
            needs_list: List[str] = ["resist", "hp"]
            cand = _collect_candidates(
                pool=pool, worn=worn, equip_map=equip_map, needs=needs_list, ap_carry=False, tank=True
            )
            cand = _fill_from_fallback(worn, equip_map, needs_list, False, True, need, cand)
            return cand[:need]

        tags = _tank_worn_tags(eq_names, equip_map)
        needs_list = []
        if n_eq == 1 and len(tags) == 1:
            t = tags[0]
            if t == "hp":
                needs_list = ["resist"]
            elif t == "resist":
                needs_list = ["hp"]
            elif t == "functional":
                needs_list = ["hp"]
            elif t == "mixed_hp_resist":
                needs_list = ["resist_or_dr"]
            else:
                needs_list = ["hp"]
        elif n_eq == 2 and len(tags) == 2:
            a, b = tags[0], tags[1]
            if a == "resist" and b == "resist":
                needs_list = ["hp"]
            elif a == "hp" and b == "hp":
                needs_list = ["resist_or_dr"]
            elif {a, b} == {"resist", "hp"}:
                needs_list = ["func_or_grievous"]
            elif a == "functional" or b == "functional":
                needs_list = ["hp"]
            elif a == "mixed_hp_resist" and b == "mixed_hp_resist":
                needs_list = ["hp"]
            else:
                needs_list = ["func_or_grievous"]
        else:
            needs_list = ["hp", "resist"]

        cand = _collect_candidates(
            pool=pool, worn=worn, equip_map=equip_map, needs=needs_list, ap_carry=False, tank=True
        )
        cand = _fill_from_fallback(worn, equip_map, needs_list, False, True, need, cand)
        return cand[:need]

    # ---------- 主 C ----------
    if slot_role == "主C":
        kinds = _carry_worn_kinds(eq_names, equip_map)
        needs2: List[str] = []
        if n_eq == 0:
            needs2 = ["startup", "violent"] if ap_carry else ["violent_ad", "startup"]
        elif n_eq == 1 and len(kinds) == 1:
            k = kinds[0]
            if k == "violent":
                needs2 = ["startup"]
            elif k == "startup":
                needs2 = ["violent_ap" if ap_carry else "violent_ad"]
            elif k == "penetration":
                needs2 = ["violent_ap" if ap_carry else "violent_ad"]
            elif k == "neutral":
                needs2 = ["violent_ap" if ap_carry else "violent_ad", "startup"]
            else:
                needs2 = ["startup"]
        elif n_eq == 2 and len(kinds) == 2:
            a, b = kinds[0], kinds[1]
            ss = {a, b}
            if a == "violent" and b == "violent":
                # 纳什等偶发被判成双暴力时，用名称兜底避免再推朔极
                needs2 = (
                    ["violent_or_shred"]
                    if _worn_has_startup_item(eq_names)
                    else ["startup_or_as"]
                )
            elif ss == {"violent", "startup"}:
                needs2 = ["violent_or_shred"]
            elif ss == {"startup", "startup"}:
                needs2 = ["violent_ap" if ap_carry else "violent_ad"]
            elif ss == {"penetration", "violent"}:
                needs2 = (
                    ["violent_or_shred"]
                    if _worn_has_startup_item(eq_names)
                    else ["startup"]
                )
            elif ss == {"penetration", "startup"}:
                needs2 = ["violent_ap" if ap_carry else "violent_ad"]
            elif a == "penetration" and b == "penetration":
                needs2 = ["violent_ap" if ap_carry else "violent_ad"]
            elif "penetration" in ss:
                needs2 = ["violent_ap" if ap_carry else "violent_ad", "startup"]
            else:
                needs2 = ["violent_or_shred"]
        else:
            needs2 = ["violent", "startup"]

        cand = _collect_candidates(
            pool=pool,
            worn=worn,
            equip_map=equip_map,
            needs=needs2,
            ap_carry=ap_carry,
            tank=False,
        )
        cand = _fill_from_fallback(worn, equip_map, needs2, ap_carry, False, need, cand)
        return cand[:need]

    # ---------- 其他定位：沿用羁绊期望 ----------
    if slot_role == "挂件":
        return []
    exp2: Optional[str]
    if ap_f or hybrid_f:
        exp2 = "AP"
    elif ad_carry:
        exp2 = "AD"
    else:
        exp2 = exp or ("TANK" if "坦克" in role_tags else "AD")

    def _pref_ok(item_name: str) -> bool:
        row = equip_map.get(item_name)
        if not row:
            return True
        bd = str(row.get("basic_desc") or "")
        if equip_audit_excluded(item_name, row):
            return True
        pk = equip_primary_stat_kind(bd)
        if exp2 == "AP":
            return pk in ("AP", "MANA", "MIX", "AS", "CRIT", "UNK")
        if exp2 == "AD":
            return pk in ("AD", "AS", "CRIT", "MANA", "MIX", "UNK")
        if exp2 == "TANK":
            return pk == "TANK" or "生命" in bd or "护甲" in bd or "魔抗" in bd
        return True

    cand = [x for x in pool if x not in worn and _pref_ok(x)]
    if not cand:
        fb: Dict[str, List[str]] = {
            "AP": ["朔极之矛", "蓝霸符", "珠光护手", "大天使之杖"],
            "AD": ["无尽之刃", "巨人捕手", "最后的轻语", "正义之手"],
            "TANK": ["狂徒铠甲", "石像鬼石板甲", "巨龙之爪"],
        }
        key = exp2 or ("TANK" if "坦克" in role_tags else "AD")
        cand = [x for x in fb.get(key, fb["AD"]) if x not in worn]

    doms: List[str] = []
    for en in eq_names:
        r = equip_map.get(en)
        if equip_audit_excluded(en, r):
            continue
        doms.append(equip_primary_stat_kind(str((r or {}).get("basic_desc") or "")))
    doms = [d for d in doms if d != "UNK"]

    out: List[str] = []
    if len(doms) >= 2 and doms[0] == doms[1]:
        pref = doms[0]
        same_first = [
            x
            for x in cand
            if equip_primary_stat_kind(str((equip_map.get(x) or {}).get("basic_desc") or ""))
            in (pref, "MANA", "MIX")
        ]
        cand = same_first or cand

    for x in cand:
        if x not in out:
            out.append(x)
        if len(out) >= need:
            break
    return out[:need]
