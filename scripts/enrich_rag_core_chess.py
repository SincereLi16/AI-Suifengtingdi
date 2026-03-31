# -*- coding: utf-8 -*-
"""
一次性/可重复：
- 将 rag_legend_chess.jsonl 中尚未出现在 rag_core_chess.jsonl 的棋子补为 core_chess 条（定位：挂件）；
- 从 rag_lineup_lineup.jsonl 全量语料挖掘「装备：」成装（rag_legend_equip 白名单），
  写回每条 core 的 meta.recommended_equips / meta.top_equips，并同步 text 中推荐出装段落。

用法（仓库根目录）:
  python scripts/enrich_rag_core_chess.py

战报侧 gemini_v1 仅从 rag_core_chess.jsonl 的 meta 读推荐装，不再单独扫阵容库。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.extra.equip_audit import load_legend_equip_full_map  # noqa: E402

CORE_PATH = REPO / "data" / "rag_core_chess.jsonl"
LEGEND_PATH = REPO / "data" / "rag_legend_chess.jsonl"
LINEUP_PATH = REPO / "data" / "rag_lineup_lineup.jsonl"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _lineup_corpus(lineups: List[Dict[str, Any]]) -> str:
    return "\n\n".join(str(d.get("text") or "") for d in lineups)


def _mine_counter_for_hero(hero: str, corpus: str, known_items: Set[str]) -> Counter[str]:
    """从攻略全文提取「棋子名 + 位置 + 装备：」块中的成装（known_items 白名单）。

    必须使用「位置」紧邻「装备：」，避免同一行内
    「沃利贝尔 位置1,3；塔姆 位置1,4 装备：…」把塔姆的出装误记到沃利贝尔上。
    """
    if not hero or len(hero) < 2:
        return Counter()
    h = re.escape(hero)
    # 可选（主C）等括号；位置 行 支持半角/全角逗号
    strict = re.compile(
        rf"{h}(?:（[^）]{{0,24}}）)?\s*位置\s*[\d,，]+\s*装备[：:]([^。\n；]+)"
    )
    cnt: Counter[str] = Counter()
    for m in strict.finditer(corpus):
        chunk = m.group(1)
        for part in re.split(r"[、，,；;]", chunk):
            part = part.strip()
            part = re.split(r"\s+", part)[0]
            if len(part) < 2 or part in {"1", "2", "3"}:
                continue
            if part in known_items:
                cnt[part] += 1
    return cnt


def _mine_items_for_hero(hero: str, corpus: str, known_items: Set[str]) -> List[str]:
    cnt = _mine_counter_for_hero(hero, corpus, known_items)
    return [k for k, _ in cnt.most_common(6)]


def _sync_text_recommended_equips(text: str, cnt: Counter[str], eqs: List[str]) -> str:
    """同步 text 中「推荐出装」与「攻略原始聚合」两行。"""
    eq_s = "、".join(eqs)
    text = re.sub(
        r"【推荐出装（至多6件，人工表优先\+攻略聚合补全）】[^\n]*",
        f"【推荐出装（至多6件，人工表优先+攻略聚合补全）】{eq_s}",
        text,
        count=1,
    )
    agg_parts = [f"{name}（×{int(w)}）" for name, w in cnt.most_common(6)]
    agg_line = "、".join(agg_parts) if agg_parts else eq_s
    text = re.sub(
        r"与主C 绑定的常见装备（攻略原始聚合）：[^\n]*",
        f"与主C 绑定的常见装备（攻略原始聚合）：{agg_line}",
        text,
        count=1,
    )
    return text


def _default_equips_by_traits(traits: List[str]) -> List[str]:
    tset = set(traits or [])
    if tset & {"护卫", "斗士", "主宰", "神盾使"}:
        return ["狂徒铠甲", "石像鬼石板甲", "棘刺背心", "巨龙之爪", "坚定之心"]
    if tset & {"法师", "耀光使"}:
        return ["珠光护手", "大天使之杖", "蓝霸符"]
    if tset & {"枪手", "狙神"}:
        return ["鬼索的狂暴之刃", "无尽之刃", "巨人捕手"]
    if tset & {"迅击战士", "裁决战士", "征服者"}:
        return ["泰坦的坚决", "汲取剑", "水银"]
    if tset & {"神谕者"}:
        return ["朔极之矛", "大天使之杖", "救赎"]
    return ["狂徒铠甲", "石像鬼石板甲", "离子火花"]


def _role_tags_from_traits(traits: List[str]) -> List[str]:
    tset = set(traits or [])
    tags: List[str] = []
    if tset & {"护卫", "斗士", "主宰", "神盾使"}:
        tags.append("坦克")
    if tset & {"法师", "耀光使"}:
        tags.append("法师")
    if tset & {"枪手", "狙神"}:
        tags.append("射手")
    if tset & {"神谕者"}:
        tags.append("辅助")
    if not tags:
        tags.append("混合")
    return tags[:3]


def _build_legend_row(
    name: str,
    cost: int,
    traits: List[str],
    legend_text: str,
    known_items: Set[str],
    corpus: str,
) -> Dict[str, Any]:
    slot_role = "挂件"
    cnt = _mine_counter_for_hero(name, corpus, known_items)
    equips = [k for k, _ in cnt.most_common(6)] if cnt else _default_equips_by_traits(traits)
    top_equips = [{"name": k, "w": int(v)} for k, v in cnt.most_common(6)] if cnt else []
    role_tags = _role_tags_from_traits(traits)
    traits_s = "、".join(traits) if traits else "（无）"
    loc_s = "、".join(role_tags)
    eq_s = "、".join(equips)
    agg_parts = [f"{n}（×{int(w)}）" for n, w in cnt.most_common(6)]
    agg_line = "、".join(agg_parts) if agg_parts else eq_s
    text = (
        f"【金铲铲 · 棋子池统计｜{name}】（{cost} 费，赛季 S17）\n"
        f"【官方羁绊】{traits_s}\n"
        f"【定位归纳（基于职业羁绊粗分）】{loc_s}向；本条目为图鉴补全，标识为挂件。\n"
        f"【攻略向标识】挂件（由 legend 图鉴补全，非统计聚合）\n"
        f"【推荐出装（至多6件，人工表优先+攻略聚合补全）】{eq_s}\n"
        f"与主C 绑定的常见装备（攻略原始聚合）：{agg_line}\n"
        f"说明：本行为 rag_legend_chess 补录；装备优先来自阵容攻略文本聚合，缺省按羁绊给通用参考。\n"
        f"传奇技能摘要：{legend_text[:400]}{'…' if len(legend_text) > 400 else ''}"
    )
    return {
        "id": f"core_chess:{name}",
        "type": "core_chess",
        "season": "S17",
        "chess_name": name,
        "cost": int(cost),
        "text": text,
        "meta": {
            "traits": traits,
            "role_tags": role_tags,
            "slot_role": slot_role,
            "as_carry_weighted": 0,
            "in_lineup_count": 0,
            "level6_weighted": 0,
            "early_mention_weighted": 0,
            "recommended_equips": equips,
            "top_equips": top_equips,
        },
        "source": "legend_supplement_v1",
    }


def _apply_lineup_mining_to_core_rows(
    core_rows: List[Dict[str, Any]],
    corpus: str,
    known_items: Set[str],
) -> int:
    """阵容语料有成装时覆盖 meta 与 text；无成装则保留原状。"""
    n = 0
    for d in core_rows:
        cn = str(d.get("chess_name") or "").strip()
        if not cn:
            continue
        cnt = _mine_counter_for_hero(cn, corpus, known_items)
        if not cnt:
            continue
        eqs = [k for k, _ in cnt.most_common(6)]
        m = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        m["recommended_equips"] = eqs
        m["top_equips"] = [{"name": k, "w": int(v)} for k, v in cnt.most_common(6)]
        d["meta"] = m
        d["text"] = _sync_text_recommended_equips(str(d.get("text") or ""), cnt, eqs)
        n += 1
    return n


def main() -> None:
    core_rows = _load_jsonl(CORE_PATH)
    legend_rows = _load_jsonl(LEGEND_PATH)
    lineups = _load_jsonl(LINEUP_PATH)
    corpus = _lineup_corpus(lineups)

    equip_map = load_legend_equip_full_map(REPO / "data" / "rag_legend_equip.jsonl")
    known_items: Set[str] = set(equip_map.keys())

    existing = {str(d.get("chess_name") or "") for d in core_rows}

    legend_by_name: Dict[str, Dict[str, Any]] = {}
    for d in legend_rows:
        nm = str(d.get("name") or "").strip()
        if not nm or nm in legend_by_name:
            continue
        legend_by_name[nm] = d

    added: List[str] = []
    for nm, d in sorted(legend_by_name.items(), key=lambda x: x[0]):
        if nm in existing:
            continue
        if nm.startswith("塞拉斯 -"):
            continue
        traits = list(d.get("traits") or [])
        if not isinstance(traits, list):
            traits = []
        cost = int(d.get("cost") or 0)
        lt = str(d.get("text") or "")
        core_rows.append(_build_legend_row(nm, cost, traits, lt, known_items, corpus))
        existing.add(nm)
        added.append(nm)

    lineup_written = _apply_lineup_mining_to_core_rows(core_rows, corpus, known_items)

    fill_roles = {"主C", "主坦", "打工仔", "混合"}
    filled = 0
    for d in core_rows:
        m = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        role = str(m.get("slot_role") or "")
        eqs = m.get("recommended_equips") if isinstance(m.get("recommended_equips"), list) else []
        if eqs:
            continue
        if role not in fill_roles:
            continue
        cn = str(d.get("chess_name") or "")
        traits = m.get("traits") if isinstance(m.get("traits"), list) else []
        mined = _mine_items_for_hero(cn, corpus, known_items)
        if mined:
            m["recommended_equips"] = mined
            cnt = _mine_counter_for_hero(cn, corpus, known_items)
            m["top_equips"] = [{"name": k, "w": int(v)} for k, v in cnt.most_common(6)]
        else:
            m["recommended_equips"] = _default_equips_by_traits(traits)
            m["top_equips"] = []
        filled += 1
        text = str(d.get("text") or "")
        if "（暂无：可在 data/core_chess_equip_supplement.json 补充）" in text:
            eq_s = "、".join(m["recommended_equips"])
            text = text.replace(
                "【推荐出装（至多6件，人工表优先+攻略聚合补全）】（暂无：可在 data/core_chess_equip_supplement.json 补充）",
                f"【推荐出装（至多6件，人工表优先+攻略聚合补全）】{eq_s}",
            )
            text = text.replace(
                "与主C 绑定的常见装备（攻略原始聚合）：（攻略聚合无成装）",
                f"与主C 绑定的常见装备（攻略原始聚合）：{eq_s}",
            )
            d["text"] = text
        d["meta"] = m

    _save_jsonl(CORE_PATH, core_rows)
    print(f"写入 {CORE_PATH}")
    print(f"自 legend 新增棋子: {len(added)}")
    if added:
        print("新增名单:", "、".join(added[:40]) + ("…" if len(added) > 40 else ""))
    print(f"阵容语料写入 recommended_equips 条目数: {lineup_written}")
    print(f"回填装备（仍为空的主C/主坦/打工仔/混合）条目数: {filled}")


if __name__ == "__main__":
    main()
