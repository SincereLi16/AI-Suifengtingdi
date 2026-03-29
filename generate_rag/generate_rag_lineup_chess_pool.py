# -*- coding: utf-8 -*-
"""
从阵容 lineup 数据聚合「棋子池」RAG：主 C 频次、常见出装、6 级过渡出现率等，
并结合 rag_legend_chess.jsonl 的 traits 标注坦克/法师/射手等定位，
输出 data/rag_core_chess.jsonl，供 LLM 检索（与 rag_lineup_lineup 互补）。

数据来源（二选一或同时加权）：
  1) 默认：data/rag_lineup_lineup.jsonl（由 generate_rag_lineup.py 生成，解析 text 字段）
  2) 可选：lineup_detail_total.json（掌盟静态包，结构更准，含 is_carry_hero、levelMap）

用法：
  python generate_rag/generate_rag_lineup_chess_pool.py
  python generate_rag/generate_rag_lineup_chess_pool.py --lineup-json path/to/lineup_detail_total.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_RAG_LINEUP = DATA_DIR / "rag_lineup_lineup.jsonl"
DEFAULT_LINEUP_TOTAL = REPO_ROOT / "lineup_detail_total.json"
DEFAULT_OUT = DATA_DIR / "rag_core_chess.jsonl"
DEFAULT_CHESS = DATA_DIR / "rag_legend_chess.jsonl"
# 可选：棋子名 -> 推荐成装列表（4～6 件为宜）；与攻略聚合合并去重后写入 meta.recommended_equips，供 RAG 检索
DEFAULT_EQUIP_SUPPLEMENT = DATA_DIR / "core_chess_equip_supplement.json"
RECOMMENDED_EQUIP_CAP = 6

# 不写入 rag_core_chess.jsonl 的棋子（人工排除）
CORE_CHESS_EXCLUDE_NAMES: Set[str] = {
    "亚索",
    "俄洛伊",
    "普朗克",
    "加里奥",
    "可酷伯与悠米",
    "德莱厄斯",
    "锤石",
}

# 覆盖启发式 slot_role（写入 meta.slot_role 与正文【攻略向标识】）
SLOT_ROLE_OVERRIDES: Dict[str, str] = {
    "斯维因": "主坦",
    "塔里克": "主坦",
    "斯卡纳": "主坦",
    "安蓓萨": "主C",
    "希瓦娜": "主C",
    "菲兹": "主C",
}


def _load_equip_supplement(path: Path) -> Dict[str, List[str]]:
    """{ 棋子中文名: [ "无尽之刃", ... ] }；文件不存在或非法则返回空。"""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        name = str(k).strip()
        if not name or name.startswith("_"):
            continue
        if isinstance(v, list):
            out[name] = [str(x).strip() for x in v if str(x).strip()]
        elif isinstance(v, str) and v.strip():
            out[name] = [v.strip()]
    return out


def _merge_recommended_equips(
    name: str,
    top_eq: List[Tuple[str, int]],
    supplement: Dict[str, List[str]],
    cap: int = RECOMMENDED_EQUIP_CAP,
) -> List[str]:
    """人工表优先，再用攻略聚合补到至多 cap 件（去重）。"""
    seen: Set[str] = set()
    merged: List[str] = []
    for e in supplement.get(name, []):
        e = str(e).strip()
        if e and e not in seen:
            seen.add(e)
            merged.append(e)
        if len(merged) >= cap:
            return merged
    for e, _ in top_eq:
        e = str(e).strip()
        if e and e not in seen:
            seen.add(e)
            merged.append(e)
        if len(merged) >= cap:
            break
    return merged


# 金铲铲「职业/特殊」羁绊 → 粗粒度定位（地区羁绊如德玛西亚、巨神峰等不映射，仅在文中列出）
_TRAIT_TO_ROLE_TAG: Dict[str, str] = {
    "斗士": "坦克",
    "护卫": "坦克",
    "神盾使": "坦克",
    "法师": "法师",
    "神谕者": "辅助",
    "耀光使": "法师",
    "狙神": "射手",
    "枪手": "射手",
    "迅击战士": "战士",
    "裁决战士": "战士",
    "征服者": "战士",
    "主宰": "战士",
    "星界游神": "辅助",
    "狂野女猎手": "战士",
    "符文法师": "法师",
    "远古巫灵": "法师",
    "时光守护者": "辅助",
    "正义巨像": "坦克",
    "远古恐惧": "法师",
    "龙血武姬": "战士",
    "腕豪": "战士",
    "沙漠皇帝": "法师",
    "铸星龙王": "法师",
    "黑暗之女": "法师",
    "暗裔剑魔": "战士",
    "暗裔": "战士",
    "不落魔锋": "战士",
    "系魂圣枪": "射手",
    "解脱者": "战士",
    "永猎双子": "射手",
    "纳什男爵": "战士",
    "海克斯机甲": "射手",
    "河流之王": "坦克",
    "虚空之女": "射手",
}

_ROLE_ORDER = ["坦克", "战士", "刺客", "射手", "法师", "辅助"]


def _role_tags_from_traits(traits: List[str]) -> Tuple[List[str], str]:
    """从棋子 traits 推导定位标签（去重有序）+ 一句话说明。"""
    seen: Set[str] = set()
    tags: List[str] = []
    for t in traits:
        tag = _TRAIT_TO_ROLE_TAG.get(str(t).strip())
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    tags.sort(key=lambda x: _ROLE_ORDER.index(x) if x in _ROLE_ORDER else 99)
    if not tags:
        tip = "（当前棋子羁绊以地区/特殊故事为主，无常见「职业」映射时不在此列粗定位；详见下方羁绊列表。）"
    elif len(tags) == 1:
        tip = f"偏{tags[0]}向。"
    elif "坦克" in tags and "法师" in tags:
        tip = "法坦/混搭前排与法系，具体看装备。"
    elif "坦克" in tags and "射手" in tags:
        tip = "可前可后，视阵容多为前排坦或后排输出挂件。"
    else:
        tip = "、".join(tags) + "混合倾向，以当局主C 与装备为准。"
    return tags, tip


def _slot_role_label(cw: int, l6: int, role_tags: List[str]) -> str:
    """
    根据攻略聚合的「主C 加权 / 6 级过渡加权」与职业定位，打主C、主坦、打工仔、混合 四选一。
    优先级：主C > 打工仔 > 主坦；其余为混合。
    """
    tags = {str(x) for x in role_tags}
    is_tank = "坦克" in tags

    # 主C：攻略中高频标主C，或明确输出核（法师/射手）且有一定主C 标注
    if cw >= 3:
        return "主C"
    if cw >= 2 and ("法师" in tags or "射手" in tags):
        return "主C"

    # 打工仔：6 级过渡权重远高于「主C」标注，典型凑数/保血
    if (l6 >= 8 and cw <= 1) or (cw > 0 and l6 >= cw * 2 and l6 >= 5):
        return "打工仔"
    if l6 >= 8 and cw == 0:
        return "打工仔"

    # 主坦：前排职业向、非主C、且不像纯打工（过渡权重不过度碾压）
    if is_tank and cw < 3:
        if not ((l6 >= 12 and cw <= 1) or (l6 >= 10 and cw == 0)):
            return "主坦"
    if is_tank and cw == 0 and 3 <= l6 < 8:
        return "主坦"

    # 兜底：仍有主C 倾向或输出位
    if "战士" in tags and cw >= 2:
        return "主C"
    if cw >= 2:
        return "主C"
    if l6 >= 5 and cw <= 1:
        return "打工仔"
    return "混合"


def _load_chess_names(path: Path) -> Tuple[Set[str], Dict[str, int]]:
    """棋子中文名集合 + name -> cost（用于输出里写费用档）。"""
    names: Set[str] = set()
    cost_by_name: Dict[str, int] = {}
    if not path.is_file():
        return names, cost_by_name
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = str(o.get("name") or "").strip()
            if n:
                names.add(n)
                try:
                    cost_by_name[n] = int(o.get("cost") or 0)
                except (TypeError, ValueError):
                    cost_by_name[n] = 0
    return names, cost_by_name


def _load_chess_by_name(path: Path) -> Dict[str, Dict[str, Any]]:
    """name -> legend_chess 整条记录（含 traits）。"""
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = str(o.get("name") or "").strip()
            if n:
                out[n] = o
    return out


def _quality_weight(q: Any) -> int:
    s = str(q or "").strip().upper()
    if not s:
        return 1
    return {"S": 3, "A": 2, "B": 1}.get(s[0], 1)


def _split_ids(raw: Any) -> List[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [x.strip() for x in s.replace("|", ",").split(",") if x.strip()]


def _parse_carries_from_lineup_text(text: str) -> List[Tuple[str, List[str]]]:
    """
    从攻略长文里抓「主C」与同行装备列表。
    例：薇恩（主C） 位置4,1 装备：鬼索的狂暴之刃、锐利之刃
    """
    out: List[Tuple[str, List[str]]] = []
    for m in re.finditer(
        r"([\u4e00-\u9fffA-Za-z·与0-9]+)（主C）[^。\n]*?装备[：:]([^；\n]+)",
        text,
    ):
        name = m.group(1).strip()
        raw_eq = m.group(2).strip()
        parts = re.split(r"[、，,]", raw_eq)
        eqs = [p.strip() for p in parts if p.strip()]
        out.append((name, eqs))
    # 部分攻略「主C」与装备分行，补一轮仅主C 名
    if not out:
        for m in re.finditer(r"([\u4e00-\u9fffA-Za-z·与0-9]+)（主C）", text):
            out.append((m.group(1).strip(), []))
    return out


def _parse_level6_hero_names(text: str) -> List[str]:
    """从「按等级构筑参考」里 6 级段抽「棋子名 + 位置」。"""
    if "6级：" not in text:
        return []
    chunk = text.split("6级：", 1)[1]
    for sep in ("8级：", "9级：", "【成型站位"):
        if sep in chunk:
            chunk = chunk.split(sep, 1)[0]
            break
    names = re.findall(r"([\u4e00-\u9fffA-Za-z·与0-9]+)\s+位置", chunk)
    return [n.strip() for n in names if n.strip()]


def _parse_early_hero_mentions(text: str, valid_names: Set[str]) -> Counter:
    """前期与过渡段落里出现了哪些已知棋子名（按长名优先匹配）。"""
    sec = ""
    if "【前期与过渡】" in text:
        sec = text.split("【前期与过渡】", 1)[1]
        for stop in ("【节奏与升星】", "【站位】", "【"):
            if stop in sec:
                sec = sec.split(stop, 1)[0]
                break
    if not sec:
        return Counter()
    names_sorted = sorted(valid_names, key=len, reverse=True)
    hit: Counter = Counter()
    for n in names_sorted:
        if len(n) >= 2 and n in sec:
            hit[n] += sec.count(n)
    return hit


def _filter_name(name: str, valid: Set[str]) -> Optional[str]:
    n = name.strip()
    if not n:
        return None
    if re.match(r"^英雄\d+$", n):
        return None
    if n in valid:
        return n
    return None


def _parse_final_board_hero_names(text: str) -> List[str]:
    """【成型站位与出装】段内「棋子 + 位置」。"""
    if "【成型站位与出装】" not in text:
        return []
    block = text.split("【成型站位与出装】", 1)[1]
    for stop in ("【主C装备替换】", "【棋子替换】", "【可追三星】", "\n【"):
        if stop in block:
            block = block.split(stop, 1)[0]
    names = re.findall(r"([\u4e00-\u9fffA-Za-z·与0-9]+)\s+位置", block)
    return [n.strip() for n in names if n.strip()]


def aggregate_from_rag_jsonl(
    path: Path,
    valid_names: Set[str],
) -> Tuple[
    Counter,
    DefaultDict[str, Counter],
    Counter,
    Counter,
    Counter,
]:
    """
    返回：
      carry_w, equip_w[hero][equip], level6_w, early_w, lineup_count（该棋子出现在多少条不同阵容里）
    """
    carry_w: Counter = Counter()
    equip_w: DefaultDict[str, Counter] = defaultdict(Counter)
    level6_w: Counter = Counter()
    early_w: Counter = Counter()
    in_lineup: DefaultDict[str, set] = defaultdict(set)

    if not path.is_file():
        return carry_w, equip_w, level6_w, early_w, Counter()

    lid = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            lid += 1
            w = _quality_weight(doc.get("quality"))
            text = str(doc.get("text") or "")

            for h, eqs in _parse_carries_from_lineup_text(text):
                hn = _filter_name(h, valid_names)
                if hn:
                    carry_w[hn] += w
                    for e in eqs:
                        if e and len(e) < 40:
                            equip_w[hn][e] += w
                    in_lineup[hn].add(lid)

            for h in _parse_level6_hero_names(text):
                hn = _filter_name(h, valid_names)
                if hn:
                    level6_w[hn] += w
                    in_lineup[hn].add(lid)

            for h in _parse_final_board_hero_names(text):
                hn = _filter_name(h, valid_names)
                if hn:
                    in_lineup[hn].add(lid)

            for h, c in _parse_early_hero_mentions(text, valid_names).items():
                early_w[h] += c * w

    lineup_count: Counter = Counter({k: len(v) for k, v in in_lineup.items()})
    return carry_w, equip_w, level6_w, early_w, lineup_count


def aggregate_from_lineup_total(
    path: Path,
    chess: Dict[str, str],
    equip: Dict[str, str],
    valid_names: Set[str],
) -> Tuple[Counter, DefaultDict[str, Counter], Counter, Counter, Counter]:
    """从 lineup_detail_total.json 结构化字段聚合（更准确的主C/装备）。"""
    carry_w: Counter = Counter()
    equip_w: DefaultDict[str, Counter] = defaultdict(Counter)
    level6_w: Counter = Counter()
    early_w: Counter = Counter()
    in_lineup: DefaultDict[str, set] = defaultdict(set)

    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("lineup_list")
    if not isinstance(items, list):
        return carry_w, equip_w, level6_w, early_w, Counter()

    lid = 0
    for outer in items:
        if not isinstance(outer, dict):
            continue
        detail_raw = outer.get("detail")
        if not detail_raw:
            continue
        try:
            inner = json.loads(detail_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(inner, dict):
            continue

        lid += 1
        w = _quality_weight(outer.get("quality"))

        hero_loc = inner.get("hero_location")
        if isinstance(hero_loc, list):
            for h in hero_loc:
                if not isinstance(h, dict):
                    continue
                hid = str(h.get("hero_id") or "")
                name = chess.get(hid, "").strip()
                if not name or name not in valid_names:
                    continue
                in_lineup[name].add(lid)
                if h.get("is_carry_hero"):
                    carry_w[name] += w
                    for eid in _split_ids(h.get("equipment_id")):
                        en = equip.get(eid, "").strip()
                        if en:
                            equip_w[name][en] += w

        lm = inner.get("levelMap")
        if isinstance(lm, dict):
            for lv_key, rows in lm.items():
                if str(lv_key) != "6":
                    continue
                if not isinstance(rows, list):
                    continue
                for h in rows:
                    if not isinstance(h, dict):
                        continue
                    hid = str(h.get("hero_id") or "")
                    name = chess.get(hid, "").strip()
                    if name in valid_names:
                        level6_w[name] += w

        early = str(inner.get("early_info") or "")
        for n in sorted(valid_names, key=len, reverse=True):
            if len(n) >= 2 and n in early:
                early_w[n] += early.count(n) * w

    lineup_count = Counter({k: len(v) for k, v in in_lineup.items()})
    return carry_w, equip_w, level6_w, early_w, lineup_count


def _load_jsonl_id_map(path: Path, prefix: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(o.get("id") or "")
            if not rid.startswith(prefix + ":"):
                continue
            num = rid.split(":", 1)[1]
            name = str(o.get("name") or "").strip()
            if num and name:
                out[num] = name
    return out


def _build_chess_doc(
    name: str,
    carry_w: Counter,
    equip_w: DefaultDict[str, Counter],
    level6_w: Counter,
    early_w: Counter,
    lineup_n: int,
    cost_by_name: Dict[str, int],
    season: str,
    chess_row: Optional[Dict[str, Any]] = None,
    *,
    equip_supplement: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    top_eq = equip_w[name].most_common(12)
    sup_map = equip_supplement or {}
    merged_equips = _merge_recommended_equips(name, top_eq, sup_map, cap=RECOMMENDED_EQUIP_CAP)
    merged_s = "、".join(merged_equips) if merged_equips else "（暂无：可在 data/core_chess_equip_supplement.json 补充）"
    top_eq_s = "、".join(f"{e}（×{c}）" for e, c in top_eq[:8]) if top_eq else "（攻略聚合无成装）"
    fee = cost_by_name.get(name, 0)
    fee_s = f"{fee} 费" if fee else "费用未知"

    traits: List[str] = []
    if chess_row:
        raw = chess_row.get("traits")
        if isinstance(raw, list):
            traits = [str(x).strip() for x in raw if str(x).strip()]
        elif isinstance(raw, str) and raw.strip():
            traits = [raw.strip()]
    role_tags, role_tip = _role_tags_from_traits(traits)
    traits_s = "、".join(traits) if traits else "（未在 legend_chess 中读到羁绊）"

    cw = int(carry_w.get(name, 0))
    l6 = int(level6_w.get(name, 0))
    slot_role = _slot_role_label(cw, l6, role_tags)
    slot_role = SLOT_ROLE_OVERRIDES.get(name, slot_role)
    if cw >= 3:
        role_hint = "多数阵容模板将其标为「主C」或核心输出位。"
    elif l6 >= 8 and cw <= 1:
        role_hint = "更常出现在 6 级过渡与低费构筑，偏打工、凑羁绊与前期保血。"
    elif l6 >= cw * 2 and l6 >= 5:
        role_hint = "前期/中期登场频率高于「主C」标注，多为过渡或挂件。"
    else:
        role_hint = "定位需结合当场来牌与装备，统计上主C 与过渡均有出现。"

    lines = [
        f"【金铲铲 · 棋子池统计｜{name}】（{fee_s}，赛季 {season}）",
        f"【官方羁绊】{traits_s}",
        f"【定位归纳（基于职业羁绊粗分）】{role_tip}",
        f"【攻略向标识】{slot_role}（主C/主坦/打工仔/混合，由主C 与 6 级过渡加权启发式划分）",
        f"在阵容攻略库中出现的阵容条数：约 {lineup_n} 套（按 lineup 条去重）。",
        f"标注或解析为「主C」的加权次数：{cw}（S/A/B 阵容权重不同）。",
        f"6 级构筑段常见登场（过渡向）加权：{l6}。",
        f"「前期与过渡」文案中提及次数（加权）：{early_w.get(name, 0)}。",
        f"【推荐出装（至多{RECOMMENDED_EQUIP_CAP}件，人工表优先+攻略聚合补全）】{merged_s}",
        f"与主C 绑定的常见装备（攻略原始聚合）：{top_eq_s}",
        f"角色倾向（启发式）：{role_hint}",
        "说明：数据来自掌盟阵容模板聚合，仅供参考；具体对局以实时棋子与装备为准。",
    ]
    text = "\n".join(lines)

    return {
        "id": f"core_chess:{name}",
        "type": "core_chess",
        "season": season,
        "chess_name": name,
        "cost": fee,
        "text": text,
        "meta": {
            "traits": traits,
            "role_tags": role_tags,
            "slot_role": slot_role,
            "as_carry_weighted": int(carry_w.get(name, 0)),
            "in_lineup_count": int(lineup_n),
            "level6_weighted": int(level6_w.get(name, 0)),
            "early_mention_weighted": int(early_w.get(name, 0)),
            "recommended_equips": merged_equips,
            "top_equips": [{"name": e, "w": int(c)} for e, c in top_eq[:15]],
        },
        "source": "lineup_aggregate_v1",
    }


def generate(
    *,
    rag_lineup_path: Path,
    lineup_total_path: Optional[Path],
    chess_path: Path,
    equip_path: Path,
    out_path: Path,
    prefer_lineup_json: bool,
    equip_supplement_path: Optional[Path] = None,
) -> int:
    valid_names, cost_by_name = _load_chess_names(chess_path)
    if not valid_names:
        raise SystemExit(f"未加载到棋子名: {chess_path}")

    chess_by_name = _load_chess_by_name(chess_path)
    chess_id_map = _load_jsonl_id_map(chess_path, "legend_chess")
    equip_id_map = _load_jsonl_id_map(equip_path, "legend_equip")

    # 若存在 lineup_detail_total.json，优先用结构化字段（主C/装备/6 级更准），否则仅解析 jsonl 文本
    if prefer_lineup_json and lineup_total_path and lineup_total_path.is_file():
        c1, e1, l1, er1, n1 = aggregate_from_lineup_total(
            lineup_total_path, chess_id_map, equip_id_map, valid_names
        )
    else:
        c1, e1, l1, er1, n1 = aggregate_from_rag_jsonl(rag_lineup_path, valid_names)

    season = "S17"
    esp = equip_supplement_path or DEFAULT_EQUIP_SUPPLEMENT
    equip_supplement = _load_equip_supplement(esp)

    docs: List[Dict[str, Any]] = []
    # 只输出在阵容库中至少出现过的棋子
    for name in sorted(n1.keys(), key=lambda x: (-n1[x], x)):
        if not _filter_name(name, valid_names):
            continue
        if name in CORE_CHESS_EXCLUDE_NAMES:
            continue
        docs.append(
            _build_chess_doc(
                name,
                c1,
                e1,
                l1,
                er1,
                int(n1.get(name, 0)),
                cost_by_name,
                season,
                chess_row=chess_by_name.get(name),
                equip_supplement=equip_supplement,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return len(docs)


def main() -> None:
    ap = argparse.ArgumentParser(description="从阵容 jsonl 聚合棋子池 RAG（主C/过渡/装备）")
    ap.add_argument("--rag-lineup", type=Path, default=DEFAULT_RAG_LINEUP, help="rag_lineup_lineup.jsonl")
    ap.add_argument(
        "--lineup-json",
        type=Path,
        default=None,
        help="可选：lineup_detail_total.json，与 jsonl 结果合并（结构化主C 更准）",
    )
    ap.add_argument("--chess", type=Path, default=DEFAULT_CHESS)
    ap.add_argument("--equip", type=Path, default=DATA_DIR / "rag_legend_equip.jsonl")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--equip-supplement",
        type=Path,
        default=None,
        help=f"可选 JSON：棋子->推荐成装列表，默认 {DEFAULT_EQUIP_SUPPLEMENT.name}",
    )
    ap.add_argument(
        "--prefer-jsonl-only",
        action="store_true",
        help="强制只用 rag_lineup_lineup.jsonl 文本解析（忽略 lineup_detail_total.json）",
    )
    args = ap.parse_args()

    lj = args.lineup_json
    if lj is None and DEFAULT_LINEUP_TOTAL.is_file():
        lj = DEFAULT_LINEUP_TOTAL

    use_struct = bool(not args.prefer_jsonl_only and lj and lj.is_file())

    esp = args.equip_supplement.resolve() if args.equip_supplement else None
    n = generate(
        rag_lineup_path=args.rag_lineup.resolve(),
        lineup_total_path=lj.resolve() if lj else None,
        chess_path=args.chess.resolve(),
        equip_path=args.equip.resolve(),
        out_path=args.output.resolve(),
        prefer_lineup_json=use_struct,
        equip_supplement_path=esp,
    )
    print(f"已写入 {args.output}，共 {n} 条棋子池 RAG。")
    print("可将该文件用于关键词检索或并入 gemini_v1 的 RAG 流程。")


if __name__ == "__main__":
    main()
