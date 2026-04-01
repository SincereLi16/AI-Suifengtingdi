# -*- coding: utf-8 -*-
"""
从 data/rag_lineup_lineup.jsonl 生成结构化简化版 data/rag_lineup_lineup_v1.jsonl。

维度：阵容ID / 阵容质量 / 阵容名称 / 羁绊 / 海克斯 / 阵容构筑 / 前期过渡 / 运营节奏 / 装备思路 / 站位策略

默认不写「text_v1」字段，避免与上述结构化字段重复；需要整段喂模型时用 format_lineup_v1_text(record)。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generate_rag.lineup_text_normalize import normalize_v1_record

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
DEFAULT_IN = DATA_DIR / "rag_lineup_lineup.jsonl"
DEFAULT_OUT = DATA_DIR / "rag_lineup_lineup_v1.jsonl"


def _extract_section(text: str, title: str) -> str:
    """title 如 '【前期与过渡】'，返回到下一个 【 之前的内容。"""
    if title not in text:
        return ""
    start = text.index(title) + len(title)
    rest = text[start:]
    nxt = rest.find("【")
    if nxt == -1:
        return rest.strip()
    return rest[:nxt].strip()


def _parse_name_traits(name_field: str) -> Tuple[str, str]:
    """
    name: 【斗枪九五】3比尔吉沃特2斗士2枪手
    -> ('斗枪九五', '3比尔吉沃特2斗士2枪手')
    """
    s = (name_field or "").strip()
    m = re.match(r"^【([^】]+)】\s*(.*)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return s, ""


def _split_hex_line(line: str) -> List[str]:
    if not line:
        return []
    parts = re.split(r"[、，,]", line)
    return [p.strip() for p in parts if p.strip()]


def _strip_piece_equipment(segment: str) -> str:
    """单段「棋子 … 位置r,c」去掉末尾「 装备：…」。"""
    seg = segment.strip()
    if not seg:
        return ""
    seg = re.sub(r"\s*装备[：:].*$", "", seg, flags=re.DOTALL)
    return seg.strip()


def _level_body_to_pieces_line(body: str) -> str:
    """一段「6级：」后的正文：去装备、位置→(R,C)、用；连接。"""
    body = body.strip()
    if not body:
        return ""
    raw_parts = body.split("；")
    pieces: List[str] = []
    for p in raw_parts:
        cl = _strip_piece_equipment(p)
        if not cl:
            continue
        cl = re.sub(r"\s*位置(\d+),(\d+)", r"(R\1,C\2)", cl)
        cl = re.sub(r"\s+", " ", cl.strip())
        pieces.append(cl)
    return "；".join(pieces)


def _parse_build_levels(ref_block: str) -> List[Dict[str, Any]]:
    """
    ref_block: 【按等级构筑参考】之后、到下一个【之前的整段；或仅「6级：... 8级：...」
    实际从全文中截取不含标题的内容。
    """
    if not ref_block.strip():
        return []
    # 找所有「数字级：」起点
    it = list(re.finditer(r"(\d+)级：", ref_block))
    if not it:
        return []
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(it):
        lv = int(m.group(1))
        start = m.end()
        end = it[i + 1].start() if i + 1 < len(it) else len(ref_block)
        body = ref_block[start:end].strip()
        line = _level_body_to_pieces_line(body)
        out.append({"level": lv, "pieces": line})
    return out


def _builds_from_levels(levels: List[Dict[str, Any]]) -> Dict[str, str]:
    """[{level, pieces}, ...] -> {"6": "...", "8": "..."}"""
    out: Dict[str, str] = {}
    for row in levels:
        if not isinstance(row, dict):
            continue
        lv = row.get("level")
        pcs = str(row.get("pieces") or "").strip()
        if lv is None or not pcs:
            continue
        try:
            k = str(int(lv))
        except (TypeError, ValueError):
            continue
        out[k] = pcs
    return out


def _infer_arch_type(*, traits: str, equip_strategy: str) -> str:
    """
    粗分阵容主输出倾向：AD / AP（二选一，供本地筛选）。
    规则为启发式，尽量保守：证据更强的一侧胜出，平分默认 AD。
    """
    t = f"{traits} {equip_strategy}".lower()
    ad_kw = (
        "枪手",
        "狙神",
        "裁决",
        "征服者",
        "无尽",
        "鬼索",
        "海妖",
        "破甲",
        "巨杀",
        "物理",
    )
    ap_kw = (
        "法师",
        "神谕",
        "法术",
        "法装",
        "法强",
        "蓝",
        "青龙刀",
        "法爆",
        "虚空杖",
        "鬼书",
    )
    ad_score = sum(1 for kw in ad_kw if kw in t)
    ap_score = sum(1 for kw in ap_kw if kw in t)
    return "AP" if ap_score > ad_score else "AD"


def _build_strategy_fields(
    early_game: str,
    tempo: str,
) -> Dict[str, str]:
    """按阶段抽象：early(1-3) / mid(4)。"""
    return {
        "early": (early_game or "").strip(),
        "mid": (tempo or "").strip(),
    }


def _infer_cost_tier(
    *,
    name_short: str,
    early_game: str,
    tempo: str,
    build_levels: List[Dict[str, Any]],
) -> str:
    """
    阵容费率标签（低费阵容 / 高费阵容）：
      - 低费：通常以 1/2/3 费追 3 星为核心
      - 高费：通常以 4/5 费 2 星与上 9/95 为核心
    """
    s = f"{name_short} {early_game} {tempo}"
    low_score = 0
    high_score = 0

    # 低费核心：追3星 + 慢D/卡利息
    low_score += s.count("3星") * 3
    for kw in ("追3星", "慢D", "卡50", "D3星", "1费", "2费", "3费", "赌"):
        if kw in s:
            low_score += 2

    # 高费核心：上9/95/5费双2
    for kw in ("上9", "95", "九五", "5费", "4费", "转95", "9级"):
        if kw in s:
            high_score += 2

    max_lv = 0
    for row in build_levels:
        if not isinstance(row, dict):
            continue
        try:
            lv = int(row.get("level") or 0)
        except (TypeError, ValueError):
            lv = 0
        max_lv = max(max_lv, lv)
    if max_lv >= 9:
        high_score += 1
    elif 0 < max_lv <= 8:
        low_score += 1

    return "低费阵容" if low_score > high_score else "高费阵容"


def _format_build_levels_for_text(levels: List[Dict[str, Any]]) -> str:
    circled = "①②③④⑤⑥⑦⑧⑨⑩"
    lines: List[str] = []
    for i, row in enumerate(levels):
        mark = circled[i] if i < len(circled) else f"{i + 1}."
        lv = row.get("level")
        pcs = row.get("pieces") or ""
        lines.append(f"  {mark} {lv}级：{pcs}")
    return "\n".join(lines)


def _format_hex_block(priority: List[str], alt: List[str]) -> str:
    lines: List[str] = []
    if priority:
        lines.append(f"  ① 优先：{'、'.join(priority)}；")
    if alt:
        lines.append(f"  ② 备选：{'、'.join(alt)}；")
    return "\n".join(lines) if lines else ""


def format_lineup_v1_text(rec: Dict[str, Any]) -> str:
    """
    由一条 v1 记录（含 lineup_id、quality、name_short、traits、hex_*、build_levels、early_game 等）
    拼出与旧版 text_v1 相同的可读块，供 RAG/LLM 使用（避免在 jsonl 里存两份相同信息）。
    """
    lineup_id = str(rec.get("lineup_id") or "").strip()
    quality = str(rec.get("quality") or "").strip()
    name_short = str(rec.get("name_short") or "").strip()
    traits = str(rec.get("traits") or "").strip()
    hex_priority = rec.get("hex_priority") or []
    hex_alternative = rec.get("hex_alternative") or []
    if not isinstance(hex_priority, list):
        hex_priority = []
    if not isinstance(hex_alternative, list):
        hex_alternative = []
    build_levels = rec.get("build_levels") or []
    early = str(rec.get("early_game") or "").strip()
    tempo = str(rec.get("tempo") or "").strip()
    equip_strategy = str(rec.get("equip_strategy") or "").strip()
    positioning = str(rec.get("positioning") or "").strip()

    parts: List[str] = [
        f"阵容ID：{lineup_id}",
        f"阵容质量：{quality}",
        f"阵容名称：{name_short}",
        f"羁绊：{traits}",
        "海克斯：",
    ]
    hb = _format_hex_block(hex_priority, hex_alternative)
    if hb.strip():
        parts.append(hb.rstrip())
    else:
        parts[-1] = "海克斯：（无）"

    parts.append("阵容构筑：")
    bl = _format_build_levels_for_text(build_levels if isinstance(build_levels, list) else [])
    if bl.strip():
        parts.append(bl)
    else:
        parts.append("  （无按等级构筑参考）")

    parts.extend(
        [
            f"前期过渡：{early}" if early else "前期过渡：（无）",
            f"运营节奏：{tempo}" if tempo else "运营节奏：（无）",
            f"装备思路：{equip_strategy}" if equip_strategy else "装备思路：（无）",
            f"站位策略：{positioning}" if positioning else "站位策略：（无）",
        ]
    )
    return "\n".join(parts)


def build_v1_record(src: Dict[str, Any], *, include_text_v1: bool = False) -> Dict[str, Any]:
    text = str(src.get("text") or "")
    name_field = str(src.get("name") or "")
    lineup_id = str(src.get("lineup_id") or "").strip()
    quality = str(src.get("quality") or "").strip()

    name_short, traits = _parse_name_traits(name_field)

    early = _extract_section(text, "【前期与过渡】")
    tempo = _extract_section(text, "【节奏与升星】")
    positioning = _extract_section(text, "【站位】")
    equip_strategy = _extract_section(text, "【装备分配思路】")

    hex_pri_raw = _extract_section(text, "【推荐海克斯】")
    hex_alt_raw = _extract_section(text, "【备选海克斯】")
    hex_priority = _split_hex_line(hex_pri_raw)
    hex_alternative = _split_hex_line(hex_alt_raw)

    ref_full = _extract_section(text, "【按等级构筑参考】")
    build_levels = _parse_build_levels(ref_full)
    builds = _builds_from_levels(build_levels)
    strategy = _build_strategy_fields(early, tempo)
    arch_type = _infer_arch_type(traits=traits, equip_strategy=equip_strategy)
    cost_tier = _infer_cost_tier(
        name_short=name_short,
        early_game=early,
        tempo=tempo,
        build_levels=build_levels,
    )

    out: Dict[str, Any] = {
        "id": f"lineup_lineup_v1:{lineup_id}",
        "type": "lineup_v1",
        "lineup_id": lineup_id,
        "name": name_short,
        "quality": quality,
        "name_short": name_short,
        "arch_type": arch_type,
        "cost_tier": cost_tier,
        "traits": traits,
        "hex_priority": hex_priority,
        "hex_alternative": hex_alternative,
        "strategy": strategy,
        "builds": builds,
        "build_levels": build_levels,
        "early_game": early,
        "tempo": tempo,
        "equip_strategy": equip_strategy,
        "positioning": positioning,
    }
    out = normalize_v1_record(out)
    if include_text_v1:
        out["text_v1"] = format_lineup_v1_text(out)
    return out


def write_rag_lineup_v1_jsonl(
    inp: Path,
    out: Path,
    *,
    include_text_v1: bool = False,
) -> int:
    """从 rag_lineup_lineup.jsonl 写出 rag_lineup_lineup_v1.jsonl，返回条数。"""
    inp = inp.resolve()
    out = out.resolve()
    if not inp.is_file():
        raise FileNotFoundError(str(inp))
    lines_out: List[str] = []
    for line in inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        v1 = build_v1_record(o, include_text_v1=include_text_v1)
        lines_out.append(json.dumps(v1, ensure_ascii=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return len(lines_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 rag_lineup_lineup_v1.jsonl")
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--with-text-v1",
        action="store_true",
        help="额外写入 text_v1 字段（与结构化字段内容重复，仅兼容旧用法）",
    )
    args = ap.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    n = write_rag_lineup_v1_jsonl(inp, out, include_text_v1=args.with_text_v1)
    print(f"已写入 {out}，共 {n} 条。")


if __name__ == "__main__":
    main()
