# -*- coding: utf-8 -*-
"""
在不依赖 lineup_detail_total.json 的前提下，就地修复 data/rag_lineup_lineup.jsonl
中因 hero_id 未映射产生的「英雄11415」「ID14360」等占位（与 generate_rag_lineup 合并补全表后的效果一致）。

用法（仓库根目录）：
  python generate_rag/repair_lineup_hero_placeholders.py
  python generate_rag/repair_lineup_hero_placeholders.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_rag.chess_id_utils import load_chess_id_supplement, merge_chess_id_maps
from generate_rag.generate_rag_lineup import (  # noqa: E402
    DEFAULT_CHESS_ID_SUPPLEMENT,
    _load_jsonl_id_name,
)

DATA_DIR = REPO_ROOT / "data"
DEFAULT_LINEUP = DATA_DIR / "rag_lineup_lineup.jsonl"
DEFAULT_LEGEND = DATA_DIR / "rag_legend_chess.jsonl"

# 「可追三星」等段落里 _names_csv 兜底产生的 ID12345（非阵容ID，阵容ID 为「阵容ID 空格 数字」）
_RE_ID_FALLBACK = re.compile(r"(?<![\w\u4e00-\u9fff])ID(\d{4,})")


def _build_id_map() -> Dict[str, str]:
    chess = _load_jsonl_id_name(DEFAULT_LEGEND, "legend_chess")
    return merge_chess_id_maps(chess, load_chess_id_supplement(DEFAULT_CHESS_ID_SUPPLEMENT))


def _repair_text(text: str, id_map: Dict[str, str]) -> Tuple[str, int, int]:
    n_hero = 0
    n_id = 0

    def sub_hero(m: re.Match[str]) -> str:
        nonlocal n_hero
        hid = m.group(1)
        name = id_map.get(hid)
        if name:
            n_hero += 1
            return name
        return m.group(0)

    def sub_idfb(m: re.Match[str]) -> str:
        nonlocal n_id
        hid = m.group(1)
        name = id_map.get(hid)
        if name:
            n_id += 1
            return name
        return m.group(0)

    t = re.sub(r"英雄(\d+)", sub_hero, text)
    t = _RE_ID_FALLBACK.sub(sub_idfb, t)
    return t, n_hero, n_id


def repair_rag_lineup_jsonl(
    inp: Path,
    out: Optional[Path] = None,
    *,
    dry_run: bool = False,
    id_map: Optional[Dict[str, str]] = None,
) -> Tuple[int, int, int, int]:
    """
    修复 jsonl 中 text 字段的占位；默认覆盖 inp（out 同 inp）。
    返回 (英雄占位替换数, ID占位替换数, 发生改动的文档数, 总条数)。
    """
    inp = inp.resolve()
    target = (out or inp).resolve()
    if not inp.is_file():
        raise FileNotFoundError(str(inp))

    m = id_map if id_map is not None else _build_id_map()
    lines_out: list[str] = []
    total_h = total_i = 0
    n_docs = 0
    n_lines = 0

    for line in inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_lines += 1
        o = json.loads(line)
        text = str(o.get("text") or "")
        new_t, nh, ni = _repair_text(text, m)
        total_h += nh
        total_i += ni
        if new_t != text:
            o["text"] = new_t
            n_docs += 1
        lines_out.append(json.dumps(o, ensure_ascii=False))

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        tmp.replace(target)

    return total_h, total_i, n_docs, n_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="修复 rag_lineup_lineup.jsonl 中的英雄 ID 占位")
    ap.add_argument("--input", type=Path, default=DEFAULT_LINEUP)
    ap.add_argument("--output", type=Path, default=None, help="默认覆盖 --input")
    ap.add_argument("--dry-run", action="store_true", help="只统计替换次数，不写文件")
    args = ap.parse_args()

    inp = args.input.resolve()
    out = (args.output or args.input).resolve()
    total_h, total_i, n_docs, _n_lines = repair_rag_lineup_jsonl(
        inp, out, dry_run=args.dry_run
    )

    print(
        f"扫描 {inp.name}：替换 英雄NNNN → {total_h} 处，"
        f"IDNNNN → {total_i} 处，涉及 {n_docs} 条文档。"
    )
    if args.dry_run:
        return
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
