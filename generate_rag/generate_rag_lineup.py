# -*- coding: utf-8 -*-
"""
从 lineup_detail_total.json（掌盟阵容 CDN 静态包）生成 RAG 用 jsonl，
与 data/rag_legend_*.jsonl 风格一致，供 LLM 检索「阵容运营/装备/站位」等知识。

依赖：仓库根目录 data/rag_legend_chess.jsonl、rag_legend_equip.jsonl、rag_legend_hex.jsonl（用于 ID→中文名）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LINEUP_JSON = REPO_ROOT / "lineup_detail_total.json"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT = DATA_DIR / "rag_lineup_lineup.jsonl"


def _load_jsonl_id_name(path: Path, prefix: str, key: str = "name") -> Dict[str, str]:
    """id 形如 legend_chess:12402 → 映射 12402 -> name"""
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
            name = (o.get(key) or "").strip()
            if num and name:
                out[num] = name
    return out


def _split_ids(raw: Any) -> List[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [x.strip() for x in s.replace("|", ",").split(",") if x.strip()]


def _names_csv(ids: List[str], id_to_name: Dict[str, str]) -> str:
    return "、".join(id_to_name.get(i, f"ID{i}") for i in ids)


def _hero_line(
    h: Dict[str, Any],
    chess: Dict[str, str],
    equip: Dict[str, str],
) -> str:
    hid = str(h.get("hero_id") or "")
    hname = chess.get(hid, f"英雄{hid}")
    loc = str(h.get("location") or "")
    carry = bool(h.get("is_carry_hero"))
    eq = _split_ids(h.get("equipment_id"))
    eq_part = ""
    if eq:
        eq_part = " 装备：" + "、".join(equip.get(e, f"装备{e}") for e in eq)
    tag = "（主C）" if carry else ""
    return f"{hname}{tag} 位置{loc}{eq_part}"


def _summarize_level_map(
    level_map: Any,
    chess: Dict[str, str],
    equip: Dict[str, str],
) -> List[str]:
    if not isinstance(level_map, dict):
        return []
    lines: List[str] = []
    for lv in sorted(level_map.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        rows = level_map.get(lv)
        if not rows:
            continue
        if not isinstance(rows, list) or not rows:
            continue
        pieces = [_hero_line(h, chess, equip) for h in rows if isinstance(h, dict)]
        if pieces:
            lines.append(f"{lv}级：{'；'.join(pieces)}")
    return lines


def _build_lineup_text(
    outer: Dict[str, Any],
    inner: Dict[str, Any],
    chess: Dict[str, str],
    equip: Dict[str, str],
    hexes: Dict[str, str],
) -> str:
    line_name = (inner.get("line_name") or "").strip() or "未命名阵容"
    parts: List[str] = [f"【金铲铲之战 阵容攻略：{line_name}】"]

    author = outer.get("lineupauthor_data") or {}
    author_name = (author.get("name") or "").strip() if isinstance(author, dict) else ""
    quality = str(outer.get("quality") or "")
    mode = str(outer.get("mode") or "")
    lid = str(outer.get("id") or "")
    season = str(outer.get("simulator_season") or outer.get("season") or "")

    meta_bits = [f"阵容ID {lid}"]
    if season:
        meta_bits.append(f"赛季 {season}")
    if quality:
        meta_bits.append(f"评级 {quality}")
    if mode:
        meta_bits.append(f"模式 {mode}")
    if author_name:
        meta_bits.append(f"来源 {author_name}")
    parts.append("（" + "，".join(meta_bits) + "）")

    def add_section(title: str, key: str) -> None:
        v = (inner.get(key) or "").strip()
        if v:
            parts.append(f"{title}{v}")

    add_section("【前期与过渡】", "early_info")
    add_section("【节奏与升星】", "d_time")
    add_section("【站位】", "location_info")
    add_section("【对阵与环境】", "enemy_info")
    add_section("【海克斯思路】", "hex_info")

    equip_info = (inner.get("equipment_info") or "").strip()
    if equip_info:
        parts.append(f"【装备分配思路】{equip_info}")

    eo = inner.get("equipment_order")
    if eo:
        eids = _split_ids(eo)
        if eids:
            names = [equip.get(e, f"装备{e}") for e in eids]
            parts.append("【装备优先级（散件顺序）】" + " → ".join(names))

    hb = inner.get("hexbuff")
    if isinstance(hb, dict):
        recomm = _split_ids(hb.get("recomm"))
        replace = _split_ids(hb.get("replace"))
        if recomm:
            parts.append(
                "【推荐海克斯】"
                + "、".join(hexes.get(h, f"海克斯{h}") for h in recomm)
            )
        if replace:
            parts.append(
                "【备选海克斯】"
                + "、".join(hexes.get(h, f"海克斯{h}") for h in replace)
            )

    lm_lines = _summarize_level_map(inner.get("levelMap"), chess, equip)
    if lm_lines:
        parts.append("【按等级构筑参考】" + " ".join(lm_lines))

    hl = inner.get("hero_location")
    if isinstance(hl, list) and hl:
        final_lines = [_hero_line(h, chess, equip) for h in hl if isinstance(h, dict)]
        if final_lines:
            parts.append("【成型站位与出装】" + "；".join(final_lines))

    cre = inner.get("carry_hero_equip_replace")
    if isinstance(cre, dict):
        main = _split_ids(cre.get("main"))
        backup = _split_ids(cre.get("backup"))
        if main or backup:
            mp = "主选：" + _names_csv(main, equip) if main else ""
            bp = "备选：" + _names_csv(backup, equip) if backup else ""
            parts.append("【主C装备替换】" + " ".join(x for x in (mp, bp) if x))

    hr = inner.get("hero_replace")
    if isinstance(hr, list) and hr:
        rp: List[str] = []
        for it in hr:
            if not isinstance(it, dict):
                continue
            a = str(it.get("hero_id") or "")
            b = _split_ids(it.get("replace_heros"))
            if a and b:
                rp.append(
                    f"{chess.get(a, a)} 可换：{_names_csv(b, chess)}"
                )
        if rp:
            parts.append("【棋子替换】" + "；".join(rp))

    l3 = inner.get("level_3_heros")
    if l3:
        ids = _split_ids(l3)
        if ids:
            parts.append("【可追三星】" + _names_csv(ids, chess))

    return "\n".join(parts)


def _build_doc(
    outer: Dict[str, Any],
    inner: Dict[str, Any],
    chess: Dict[str, str],
    equip: Dict[str, str],
    hexes: Dict[str, str],
) -> Dict[str, Any]:
    lid = str(outer.get("id") or "")
    line_name = (inner.get("line_name") or "").strip() or "未命名阵容"
    author = outer.get("lineupauthor_data") or {}
    author_name = (author.get("name") or "").strip() if isinstance(author, dict) else ""

    return {
        "id": f"lineup_lineup:{lid}",
        "type": "lineup",
        "season": str(outer.get("simulator_season") or ""),
        # 与 rag_legend_* 一致：游戏「赛季集合」编号，缺省按 S17 → 16
        "setId": str(outer.get("setId") or "16"),
        "edition": str(outer.get("edition") or ""),
        "lineup_id": lid,
        "name": line_name,
        "quality": str(outer.get("quality") or ""),
        "mode": str(outer.get("mode") or ""),
        "author": author_name,
        "text": _build_lineup_text(outer, inner, chess, equip, hexes),
        "source": "lineup_detail_total.json",
    }


def generate(
    lineup_path: Path,
    out_path: Path,
    chess_path: Path,
    equip_path: Path,
    hex_path: Path,
) -> int:
    raw = json.loads(lineup_path.read_text(encoding="utf-8"))
    items = raw.get("lineup_list")
    if not isinstance(items, list):
        raise SystemExit("lineup_detail_total.json 缺少 lineup_list 数组")

    chess = _load_jsonl_id_name(chess_path, "legend_chess")
    equip = _load_jsonl_id_name(equip_path, "legend_equip")
    hexes = _load_jsonl_id_name(hex_path, "legend_hex")

    docs: List[Dict[str, Any]] = []
    for outer in items:
        if not isinstance(outer, dict):
            continue
        detail_raw = outer.get("detail")
        if not detail_raw:
            continue
        try:
            inner = json.loads(detail_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(inner, dict):
            continue
        docs.append(_build_doc(outer, inner, chess, equip, hexes))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return len(docs)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 lineup_detail_total.json 生成 rag_lineup_lineup.jsonl")
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_LINEUP_JSON,
        help="掌盟 lineup_detail_total.json 路径",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="输出 jsonl（默认 data/rag_lineup_lineup.jsonl）",
    )
    ap.add_argument("--chess", type=Path, default=DATA_DIR / "rag_legend_chess.jsonl")
    ap.add_argument("--equip", type=Path, default=DATA_DIR / "rag_legend_equip.jsonl")
    ap.add_argument("--hex", type=Path, default=DATA_DIR / "rag_legend_hex.jsonl")
    args = ap.parse_args()

    n = generate(args.input, args.output, args.chess, args.equip, args.hex)
    print(f"已写入 {args.output}，共 {n} 条。")


if __name__ == "__main__":
    main()
