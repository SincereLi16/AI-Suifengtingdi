# -*- coding: utf-8 -*-
"""
从腾讯 CDN 拉取 chess.js，生成全量 hero_id → 中文名 JSON，
供 generate_rag_lineup / generate_rag_lineup_chess_pool 补全「别名 ID」导致的「英雄11415」占位。

与 generate_rag_lol_legend.py 中 URL 版本保持一致；赛季更新后请重跑本脚本。

用法（仓库根目录）：
  python generate_rag/fetch_chess_id_name_map.py
  python generate_rag/fetch_chess_id_name_map.py --output data/chess_id_name_supplement.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_rag.chess_id_utils import build_id_name_map_from_chess_js

# 与 generate_rag/generate_rag_lol_legend.py URLS["chess"] 一致
DEFAULT_CHESS_JS_URL = (
    "https://game.gtimg.cn/images/lol/act/jkzlk/js//16/16.16.6-S17/chess.js"
)
DEFAULT_OUT = REPO_ROOT / "data" / "chess_id_name_supplement.json"


def _fetch_json(url: str) -> Dict[str, Any]:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"拉取失败: {url} ({e})") from e
    return json.loads(raw)


def write_chess_id_name_supplement(
    out: Path,
    url: str = DEFAULT_CHESS_JS_URL,
    *,
    indent: int = 0,
) -> int:
    """拉取 chess.js 并写入补全表，返回条目数。"""
    raw = _fetch_json(url.strip())
    m = build_id_name_map_from_chess_js(raw)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(m, ensure_ascii=False, indent=indent or None)
    out.write_text(text + "\n", encoding="utf-8")
    return len(m)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 chess.js 生成 hero_id→中文名 JSON 补全表")
    ap.add_argument("--url", type=str, default=DEFAULT_CHESS_JS_URL, help="chess.js URL")
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUT,
        help="输出 JSON（默认 data/chess_id_name_supplement.json）",
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=0,
        help="JSON 缩进（0 表示一行压缩，便于 git diff）",
    )
    args = ap.parse_args()

    out = args.output.resolve()
    n = write_chess_id_name_supplement(out, url=args.url.strip(), indent=args.indent)
    print(f"已写入 {out}，共 {n} 条 id→name。")


if __name__ == "__main__":
    main()
