import json
from pathlib import Path
from typing import Any, Dict, List

import requests


URLS = {
    "trait": "https://game.gtimg.cn/images/lol/act/jkzlk/js//16/16.16.6-S17/trait.js",
    "hex": "https://game.gtimg.cn/images/lol/act/jkzlk/js//16/16.16.6-S17/hex.js",
    "chess": "https://game.gtimg.cn/images/lol/act/jkzlk/js//16/16.16.6-S17/chess.js",
    "mission": "https://game.gtimg.cn/images/lol/act/jkzlk/js//16/16.16.6-S17/mission.js",
    "equip": "https://game.gtimg.cn/images/lol/act/jkzlk/js//16/16.16.6-S17/equip.js",
}

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data"

LEVEL_LABEL = {"1": "银色", "2": "金色", "3": "棱彩"}


def _fetch_json(url: str) -> Dict[str, Any]:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, docs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def _build_trait_docs(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data = raw.get("data", {}) or {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in data.values():
        name = (t.get("name") or "").strip()
        if not name:
            continue
        grouped.setdefault(name, []).append(t)

    docs: List[Dict[str, Any]] = []
    for name, levels in grouped.items():
        levels.sort(key=lambda x: int(x.get("level") or 0))
        check_ids = sorted({str(x.get("checkId")) for x in levels if x.get("checkId") not in (None, "")})
        type_ids = sorted({str(x.get("type")) for x in levels if x.get("type") not in (None, "")})

        thresholds_set: set[int] = set()
        for lv in levels:
            for key in ("numList", "values"):
                for part in str(lv.get(key) or "").split("|"):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        thresholds_set.add(int(part))
                    except (TypeError, ValueError):
                        continue
        thresholds = sorted(thresholds_set)

        pieces: List[str] = [f"【英雄联盟传奇 羁绊：{name}】"]
        for lv in levels:
            real = (lv.get("realDesc") or "").strip()
            if real:
                pieces.append(real)
        if len(pieces) == 1:
            fallback_desc = (levels[0].get("desc2") or "").strip() if levels else ""
            if fallback_desc:
                pieces.append(fallback_desc)

        docs.append(
            {
                "id": f"legend_trait:{name}",
                "type": "trait",
                "season": season,
                "setId": set_id,
                "name": name,
                "check_ids": check_ids,
                "trait_type_ids": type_ids,
                "thresholds": thresholds,
                "text": " ".join(pieces),
                "source": "trait.js",
            }
        )
    return docs


def _build_checkid_to_trait_name(trait_raw: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    data = trait_raw.get("data", {}) or {}
    for t in data.values():
        check_id = str(t.get("checkId") or "").strip()
        name = (t.get("name") or "").strip()
        if check_id and name and check_id not in out:
            out[check_id] = name
    return out


def _build_heroid_to_name(chess_raw: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    data = chess_raw.get("data", {}) or {}
    for cid, c in data.items():
        name = (c.get("name") or "").strip()
        if name and str(cid) not in out:
            out[str(cid)] = name
    return out


def _split_ids(raw_value: Any) -> List[str]:
    return [x.strip() for x in str(raw_value or "").split("|") if x.strip() and x.strip() not in {"0", "-1"}]


def _build_chess_docs(
    chess_raw: Dict[str, Any],
    checkid_to_trait: Dict[str, str],
    heroid_to_missions: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    season = chess_raw.get("season")
    set_id = chess_raw.get("setId") or chess_raw.get("setid")
    data = chess_raw.get("data", {}) or {}

    docs: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for cid, c in data.items():
        name = (c.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        if str(c.get("heroType", "1")) != "0":
            continue

        try:
            cost = int(c.get("price") or c.get("buyPrice") or 0)
        except (TypeError, ValueError):
            cost = 0
        if cost <= 0:
            continue

        seen_names.add(name)

        class_ids = _split_ids(c.get("class"))
        species_ids = _split_ids(c.get("species"))
        trait_names = [checkid_to_trait[x] for x in class_ids + species_ids if x in checkid_to_trait]
        trait_names = list(dict.fromkeys(trait_names))

        desc_parts: List[str] = []
        for key in ("skillName", "skillDesc", "skillBriefValue"):
            txt = (c.get(key) or "").strip()
            if txt and txt not in desc_parts:
                desc_parts.append(txt)

        mission_texts = heroid_to_missions.get(str(cid), [])
        text_parts: List[str] = [
            f"【英雄联盟传奇 英雄：{name}】（费用：{cost} 金币，羁绊：{'、'.join(trait_names) if trait_names else '无'}）"
        ]
        text_parts.extend(desc_parts)
        if mission_texts:
            text_parts.append("解锁条件：" + "；".join(mission_texts))

        docs.append(
            {
                "id": f"legend_chess:{cid}",
                "type": "chess",
                "season": season,
                "setId": set_id,
                "name": name,
                "cost": cost,
                "class_ids": class_ids,
                "species_ids": species_ids,
                "traits": trait_names,
                "picture": c.get("picture") or "",
                "missions": mission_texts,
                "text": " ".join(text_parts),
                "source": "chess.js",
            }
        )
    return docs


def _build_hex_docs(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data = raw.get("data", {}) or {}

    docs: List[Dict[str, Any]] = []
    for hid, h in data.items():
        name = (h.get("name") or "").strip() or f"ID{hid}"
        level_raw = str(h.get("level") or "")
        level = int(level_raw) if level_raw.isdigit() else None
        level_label = LEVEL_LABEL.get(level_raw, f"未知({level_raw})")
        desc = (h.get("desc") or "").strip()
        fetter_id = str(h.get("fetterId") or "").strip()
        icon = h.get("icon") or ""
        is_legend = h.get("is_legend")
        hero_enhancement_type = h.get("hero_enhancement_type")

        text = f"【英雄联盟传奇 海克斯：{name}】（{level_label}）"
        if desc:
            text += f" {desc}"
        if fetter_id:
            text += f" 关联羁绊ID：{fetter_id}"

        docs.append(
            {
                "id": f"legend_hex:{hid}",
                "type": "hex",
                "season": season,
                "setId": set_id,
                "name": name,
                "level": level,
                "level_label": level_label,
                "fetter_id": fetter_id,
                "icon": icon,
                "is_legend": is_legend,
                "hero_enhancement_type": hero_enhancement_type,
                "text": text,
                "source": "hex.js",
            }
        )
    return docs


def _build_mission_docs(raw: Dict[str, Any], heroid_to_name: Dict[str, str]) -> List[Dict[str, Any]]:
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data = raw.get("data", {}) or {}

    docs: List[Dict[str, Any]] = []
    for hero_id, entry in data.items():
        missions = entry.get("mission") or []
        if not isinstance(missions, list):
            continue
        hero_name = heroid_to_name.get(str(hero_id), "")
        for m in missions:
            mid = m.get("id")
            desc = (m.get("desc") or "").strip()
            tasktips = (m.get("tasktips") or "").strip()
            text_parts = [f"【英雄联盟传奇 阵容解锁条件】英雄ID={hero_id}"]
            if hero_name:
                text_parts.append(f"英雄名：{hero_name}")
            if tasktips:
                text_parts.append(f"任务提示：{tasktips}")
            if desc:
                text_parts.append(f"条件：{desc}")

            docs.append(
                {
                    "id": f"legend_mission:{mid}",
                    "type": "mission",
                    "season": season,
                    "setId": set_id,
                    "hero_id": str(hero_id),
                    "hero_name": hero_name,
                    "mission_id": mid,
                    "tasktips": tasktips,
                    "desc": desc,
                    "difficulty": m.get("difficulty"),
                    "show": m.get("show"),
                    "sort": m.get("sort"),
                    "text": " ".join(text_parts),
                    "source": "mission.js",
                }
            )
    return docs


def _build_heroid_to_missions(raw: Dict[str, Any]) -> Dict[str, List[str]]:
    data = raw.get("data", {}) or {}
    out: Dict[str, List[str]] = {}
    for hero_id, entry in data.items():
        missions = entry.get("mission") or []
        if not isinstance(missions, list):
            continue
        descs = []
        for m in missions:
            desc = (m.get("desc") or "").strip()
            if desc:
                descs.append(desc)
        if descs:
            out[str(hero_id)] = descs
    return out


def _build_equip_docs(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    season = raw.get("season")
    set_id = raw.get("setId") or raw.get("setid")
    data = raw.get("data", {}) or {}

    id_to_name = {str(eid): (e.get("name") or "") for eid, e in data.items()}
    docs: List[Dict[str, Any]] = []
    for eid, e in data.items():
        name = (e.get("name") or "").strip() or f"ID{eid}"
        equip_type = e.get("type") or ""
        basic_desc = (e.get("basicDesc") or "").strip()
        desc = (e.get("desc") or "").strip()
        picture = e.get("picture") or ""

        syn1_id = str(e.get("synthesis1") or "0")
        syn2_id = str(e.get("synthesis2") or "0")
        syn1_name = id_to_name.get(syn1_id, "") if syn1_id != "0" else ""
        syn2_name = id_to_name.get(syn2_id, "") if syn2_id != "0" else ""
        synthesis = [x for x in [syn1_name, syn2_name] if x]

        text_parts = [f"【英雄联盟传奇 装备：{name}】"]
        if equip_type:
            text_parts.append(f"类型：{equip_type}")
        if basic_desc:
            text_parts.append(basic_desc)
        if desc:
            text_parts.append(desc)
        if len(synthesis) == 2:
            text_parts.append(f"合成公式：{synthesis[0]} + {synthesis[1]}")

        docs.append(
            {
                "id": f"legend_equip:{eid}",
                "type": "equip",
                "season": season,
                "setId": set_id,
                "name": name,
                "equip_type": equip_type,
                "basic_desc": basic_desc,
                "desc": desc,
                "synthesis": synthesis if len(synthesis) == 2 else [],
                "picture": picture,
                "text": " ".join(text_parts),
                "source": "equip.js",
            }
        )
    return docs


def generate_rag_lol_legend() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("正在拉取英雄联盟传奇赛季数据...")
    raw_trait = _fetch_json(URLS["trait"])
    raw_hex = _fetch_json(URLS["hex"])
    raw_chess = _fetch_json(URLS["chess"])
    raw_mission = _fetch_json(URLS["mission"])
    raw_equip = _fetch_json(URLS["equip"])

    # 保存原始 response（便于排查）
    _write_json(OUT_DIR / "raw_legend_trait.json", raw_trait)
    _write_json(OUT_DIR / "raw_legend_hex.json", raw_hex)
    _write_json(OUT_DIR / "raw_legend_chess.json", raw_chess)
    _write_json(OUT_DIR / "raw_legend_mission.json", raw_mission)
    _write_json(OUT_DIR / "raw_legend_equip.json", raw_equip)

    checkid_to_trait = _build_checkid_to_trait_name(raw_trait)
    heroid_to_name = _build_heroid_to_name(raw_chess)
    heroid_to_missions = _build_heroid_to_missions(raw_mission)

    trait_docs = _build_trait_docs(raw_trait)
    hex_docs = _build_hex_docs(raw_hex)
    mission_docs = _build_mission_docs(raw_mission, heroid_to_name)
    chess_docs = _build_chess_docs(raw_chess, checkid_to_trait, heroid_to_missions)
    equip_docs = _build_equip_docs(raw_equip)

    _write_jsonl(OUT_DIR / "rag_legend_traits.jsonl", trait_docs)
    _write_jsonl(OUT_DIR / "rag_legend_hex.jsonl", hex_docs)
    _write_jsonl(OUT_DIR / "rag_legend_chess.jsonl", chess_docs)
    _write_jsonl(OUT_DIR / "rag_legend_mission.jsonl", mission_docs)
    _write_jsonl(OUT_DIR / "rag_legend_equip.jsonl", equip_docs)

    print("\n=== 生成完成 ===")
    print(f"rag_legend_traits.jsonl  : {len(trait_docs)}")
    print(f"rag_legend_hex.jsonl     : {len(hex_docs)}")
    print(f"rag_legend_chess.jsonl   : {len(chess_docs)}")
    print(f"rag_legend_mission.jsonl : {len(mission_docs)}")
    print(f"rag_legend_equip.jsonl   : {len(equip_docs)}")


if __name__ == "__main__":
    generate_rag_lol_legend()
