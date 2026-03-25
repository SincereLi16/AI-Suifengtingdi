import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests


VERSION_CONFIG_URL = (
    "https://game.gtimg.cn/images/lol/act/jkzlk/js/config/versiondataconfig.js"
)
BASE_JS_URL = "https://game.gtimg.cn/images/lol/act/jkzlk/js/"

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data"

LEVEL_LABEL = {"1": "银色", "2": "金色", "3": "棱彩"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _fetch_json(url: str) -> Any:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _write_jsonl(path: Path, docs: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 自动从 versiondataconfig.js 获取最新 S17 天选福星 URL
# ---------------------------------------------------------------------------

def _get_latest_s17_urls() -> Dict[str, str]:
    """
    从官方版本配置中自动查找最新 S17 天选福星（mode=4）条目，
    返回 trait / chess / hex / equip 的完整 URL 字典。
    未找到时退回硬编码的备用 URL。
    """
    FALLBACK_PREFIX = "https://game.gtimg.cn/images/lol/act/jkzlk/js//4/16.16.6-S17/"
    fallback = {
        "trait": FALLBACK_PREFIX + "trait.js",
        "chess": FALLBACK_PREFIX + "chess.js",
        "hex":   FALLBACK_PREFIX + "hex.js",
        "equip": FALLBACK_PREFIX + "equip.js",
        "job":   FALLBACK_PREFIX + "job.js",
        "race":  FALLBACK_PREFIX + "race.js",
    }

    try:
        all_versions = _fetch_json(VERSION_CONFIG_URL)
    except Exception as e:
        print(f"[警告] 无法获取版本配置，使用备用 URL：{e}")
        return fallback

    # 找 season=S17, mode=4（天选福星），is_newest_version=1
    candidates = [
        v for v in all_versions
        if v.get("season") == "S17"
        and str(v.get("mode")) == "4"
        and int(v.get("is_newest_version", 0)) == 1
    ]

    # 如果没有 is_newest_version=1 的，则取同条件中 version 字符串最大的一条
    if not candidates:
        candidates = [
            v for v in all_versions
            if v.get("season") == "S17" and str(v.get("mode")) == "4"
        ]
        candidates.sort(key=lambda v: v.get("version", ""), reverse=True)

    if not candidates:
        print("[警告] versiondataconfig 中未找到 S17 天选福星版本，使用备用 URL")
        return fallback

    entry = candidates[0]
    version_str = entry.get("version", "")
    season_str  = entry.get("season", "S17")
    print(f"[版本自动检测] 当前 S17 天选福星最新版本：{version_str}（{season_str}）")

    def _resolve(relative_url: str) -> str:
        return BASE_JS_URL + relative_url.lstrip("/")

    return {
        "trait": _resolve(entry["traiturl"]),
        "chess": _resolve(entry["herourl"]),
        "hex":   _resolve(entry["hexurl"]),
        "equip": _resolve(entry["equipurl"]),
        "job":   _resolve(entry["joburl"]),
        "race":  _resolve(entry["raceurl"]),
    }


# ---------------------------------------------------------------------------
# 羁绊 RAG 文档构建
# ---------------------------------------------------------------------------

def _build_trait_docs(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data   = raw.get("data", {}) or {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for _tid, t in data.items():
        name = t.get("name")
        if not name:
            continue
        grouped.setdefault(name, []).append(t)

    docs: List[Dict[str, Any]] = []
    for name, levels in grouped.items():
        levels.sort(key=lambda x: x.get("level", 0))

        pieces: List[str] = [f"【S17 羁绊：{name}】"]
        for lv in levels:
            real = (lv.get("realDesc") or "").strip()
            if real:
                pieces.append(real)

        if len(pieces) == 1:
            for lv in levels:
                desc2 = (lv.get("desc2") or "").strip()
                if desc2:
                    pieces.append(desc2)
                    break

        text = " ".join(pieces)

        thresholds_set: set = set()
        for lv in levels:
            num_list_raw = lv.get("numList") or lv.get("values") or ""
            for part in str(num_list_raw).split("|"):
                part = part.strip()
                if not part:
                    continue
                try:
                    thresholds_set.add(int(part))
                except (TypeError, ValueError):
                    continue
        thresholds = sorted(thresholds_set)

        docs.append({
            "id":         f"trait:{name}",
            "type":       "trait",
            "season":     season,
            "setId":      set_id,
            "name":       name,
            "thresholds": thresholds,
            "text":       text,
            "source":     "trait.js",
        })

    return docs


# ---------------------------------------------------------------------------
# 棋子 RAG 文档构建（依赖 job.js / race.js 的 ID→名称映射）
# ---------------------------------------------------------------------------

def _build_id_to_trait_name(raw_data: Dict[str, Any]) -> Dict[str, str]:
    """
    从 job.js 或 race.js 的 data 部分构建 {ID: 羁绊名} 映射。
    job.js 与 race.js 的 ID 编号体系不同，需分别构建，勿合并。
    """
    return {tid: entry["name"] for tid, entry in raw_data.items() if entry.get("name")}


def _build_chess_docs(
    raw: Dict[str, Any],
    job_id_to_name: Dict[str, str],
    race_id_to_name: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    将 chess.js 转成 RAG 文档，正确把 class（职业）和 species（种族）
    数字 ID 映射为可读的羁绊名称字符串。

    关键字段：
      heroType = "0"  → 正式英雄（非假人、非技能展示单位）
      class           → 职业 ID，pipe 分隔，对应 job.js
      species         → 种族 ID，pipe 分隔，对应 race.js
      price           → 金币费用
    每个英雄名称只保留第一次出现的条目（去重）。
    """
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data   = raw.get("data", {}) or {}

    seen_names: set = set()
    docs: List[Dict[str, Any]] = []

    for cid, c in data.items():
        name = (c.get("name") or "").strip()
        if not name or name == "木桩假人":
            continue
        # 只保留正式英雄单位
        if c.get("heroType", "1") != "0":
            continue
        # 同名只取第一条（chess.js 里同一英雄有多个变体条目）
        if name in seen_names:
            continue
        seen_names.add(name)

        # 费用
        try:
            cost = int(c.get("price") or c.get("buyPrice") or 0)
        except (TypeError, ValueError):
            cost = 0
        if cost <= 0:
            continue

        # 职业 ID → 羁绊名（job.js）
        class_raw = str(c.get("class") or "")
        job_names = [
            job_id_to_name[jid]
            for jid in (x.strip() for x in class_raw.split("|"))
            if jid and jid not in ("-1", "0") and jid in job_id_to_name
        ]

        # 种族 ID → 羁绊名（race.js）
        species_raw = str(c.get("species") or "")
        race_names = [
            race_id_to_name[rid]
            for rid in (x.strip() for x in species_raw.split("|"))
            if rid and rid not in ("-1", "0") and rid in race_id_to_name
        ]

        # 合并，保持顺序、去重
        trait_names: List[str] = list(dict.fromkeys(job_names + race_names))

        # 技能描述（拼入 text，方便全文检索）
        desc_parts: List[str] = []
        for key in ("skillName", "skillDesc", "skillBriefValue"):
            value = (c.get(key) or "").strip()
            if value and value not in desc_parts:
                desc_parts.append(value)

        trait_str = "、".join(trait_names) if trait_names else "无"
        text = (
            f"【S17 英雄：{name}】"
            f"（费用：{cost} 金币，羁绊：{trait_str}）"
            + (" " + " ".join(desc_parts) if desc_parts else "")
        ).strip()

        docs.append({
            "id":      f"chess:{cid}",
            "type":    "chess",
            "season":  season,
            "setId":   set_id,
            "name":    name,
            "cost":    cost,
            "traits":  trait_names,
            "picture": c.get("picture") or "",
            "text":    text,
            "source":  "chess.js",
        })

    return docs


# ---------------------------------------------------------------------------
# 海克斯 RAG 文档构建
# ---------------------------------------------------------------------------

def _build_hex_docs(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将 hex.js 转成 RAG 文档。
    level: 1=银色，2=金色，3=棱彩
    """
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data   = raw.get("data", {}) or {}

    docs: List[Dict[str, Any]] = []

    for hid, h in data.items():
        name = h.get("name") or f"ID{hid}"
        level_raw = str(h.get("level", ""))
        level_label = LEVEL_LABEL.get(level_raw, f"未知({level_raw})")
        desc = (h.get("desc") or "").strip()
        fetter_id = h.get("fetterId") or ""
        icon = h.get("icon") or ""

        text_parts = [f"【S17 海克斯：{name}】（{level_label}）"]
        if desc:
            text_parts.append(desc)
        if fetter_id:
            text_parts.append(f"关联羁绊 ID：{fetter_id}")
        text = " ".join(text_parts)

        docs.append({
            "id":         f"hex:{hid}",
            "type":       "hex",
            "season":     season,
            "setId":      set_id,
            "name":       name,
            "level":      int(level_raw) if level_raw.isdigit() else None,
            "level_label": level_label,
            "fetter_id":  fetter_id,
            "icon":       icon,
            "text":       text,
            "source":     "hex.js",
        })

    return docs


# ---------------------------------------------------------------------------
# 装备 RAG 文档构建
# ---------------------------------------------------------------------------

def _build_equip_docs(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将 equip.js 转成 RAG 文档。
    synthesis1/synthesis2 为基础装备 ID，这里解析为名称，方便查询合成公式。
    """
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data   = raw.get("data", {}) or {}

    # 先建 ID→name 索引，用于合成公式解析
    id_to_name: Dict[str, str] = {
        eid: e.get("name", "") for eid, e in data.items()
    }

    docs: List[Dict[str, Any]] = []

    for eid, e in data.items():
        name      = e.get("name") or f"ID{eid}"
        equip_type = e.get("type") or ""
        basic_desc = (e.get("basicDesc") or "").strip()
        desc       = (e.get("desc") or "").strip()
        picture    = e.get("picture") or ""

        # 合成公式：基础装备 ID → 名称
        syn1_id = str(e.get("synthesis1") or "0")
        syn2_id = str(e.get("synthesis2") or "0")
        syn1_name = id_to_name.get(syn1_id, "") if syn1_id != "0" else ""
        syn2_name = id_to_name.get(syn2_id, "") if syn2_id != "0" else ""

        is_basic     = equip_type == "基础装备"
        is_composed  = bool(syn1_name and syn2_name)

        text_parts = [f"【S17 装备：{name}】（{equip_type}）"]
        if basic_desc:
            text_parts.append(basic_desc)
        if desc:
            text_parts.append(desc)
        if is_composed:
            text_parts.append(f"合成公式：{syn1_name} + {syn2_name}")
        text = " ".join(text_parts)

        docs.append({
            "id":          f"equip:{eid}",
            "type":        "equip",
            "season":      season,
            "setId":       set_id,
            "name":        name,
            "equip_type":  equip_type,
            "basic_desc":  basic_desc,
            "desc":        desc,
            "is_basic":    is_basic,
            "synthesis":   [syn1_name, syn2_name] if is_composed else [],
            "picture":     picture,
            "text":        text,
            "source":      "equip.js",
        })

    return docs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def generate_rag_s17() -> None:
    """
    从官方接口拉取 S17 天选福星全量数据，生成 4 个 JSONL RAG 文件到 data/ 目录：
      - rag_s17_traits.jsonl  羁绊
      - rag_s17_chess.jsonl   棋子（英雄）
      - rag_s17_hex.jsonl     海克斯强化
      - rag_s17_equip.jsonl   装备（含合成公式）
    """
    OUT_DIR.mkdir(exist_ok=True)

    print("正在自动检测 S17 最新版本 URL...")
    urls = _get_latest_s17_urls()

    # 羁绊
    print(f"\n拉取羁绊数据：{urls['trait']}")
    trait_docs = _build_trait_docs(_fetch_json(urls["trait"]))
    _write_jsonl(OUT_DIR / "rag_s17_traits.jsonl", trait_docs)
    print(f"  [OK] rag_s17_traits.jsonl  - {len(trait_docs)} 条")

    # 职业 & 种族（用于 chess traits 映射）
    print(f"\n拉取职业数据：{urls['job']}")
    job_id_to_name  = _build_id_to_trait_name(_fetch_json(urls["job"]).get("data", {}))
    print(f"  [OK] 职业词典 - {len(job_id_to_name)} 条（如：{list(job_id_to_name.values())[:5]}）")

    print(f"\n拉取种族数据：{urls['race']}")
    race_id_to_name = _build_id_to_trait_name(_fetch_json(urls["race"]).get("data", {}))
    print(f"  [OK] 种族词典 - {len(race_id_to_name)} 条（如：{list(race_id_to_name.values())[:5]}）")

    # 棋子
    print(f"\n拉取棋子数据：{urls['chess']}")
    chess_docs = _build_chess_docs(_fetch_json(urls["chess"]), job_id_to_name, race_id_to_name)
    _write_jsonl(OUT_DIR / "rag_s17_chess.jsonl", chess_docs)
    print(f"  [OK] rag_s17_chess.jsonl   - {len(chess_docs)} 条")

    # 海克斯
    print(f"\n拉取海克斯数据：{urls['hex']}")
    hex_docs = _build_hex_docs(_fetch_json(urls["hex"]))
    _write_jsonl(OUT_DIR / "rag_s17_hex.jsonl", hex_docs)
    print(f"  [OK] rag_s17_hex.jsonl     - {len(hex_docs)} 条")

    # 装备
    print(f"\n拉取装备数据：{urls['equip']}")
    equip_docs = _build_equip_docs(_fetch_json(urls["equip"]))
    _write_jsonl(OUT_DIR / "rag_s17_equip.jsonl", equip_docs)
    print(f"  [OK] rag_s17_equip.jsonl   - {len(equip_docs)} 条")

    print("\n全部 RAG 文件生成完毕！")


if __name__ == "__main__":
    generate_rag_s17()
