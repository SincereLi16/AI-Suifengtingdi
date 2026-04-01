# -*- coding: utf-8 -*-
"""
阵容 RAG 一键流水线：补全 hero_id → 生成/修复 rag_lineup_lineup.jsonl → 生成 rag_lineup_lineup_v1.jsonl

默认步骤（可跳过部分）：
  1. 拉取 chess.js → data/chess_id_name_supplement.json
  2. （可选）从 lineup_detail_total.json 生成 data/rag_lineup_lineup.jsonl
  3. 修复阵容 jsonl 中的「英雄11415」等占位
  4. 写出 data/rag_lineup_lineup_v1.jsonl

用法（仓库根目录）：
  python generate_rag/pipeline_lineup_rag.py
  python generate_rag/pipeline_lineup_rag.py --lineup-json lineup_detail_total.json
  python generate_rag/pipeline_lineup_rag.py --skip-fetch --skip-generate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_rag.build_rag_lineup_v1 import write_rag_lineup_v1_jsonl
from generate_rag.fetch_chess_id_name_map import DEFAULT_CHESS_JS_URL, write_chess_id_name_supplement
from generate_rag.generate_rag_lineup import DATA_DIR, DEFAULT_LINEUP_JSON, generate
from generate_rag.generate_rag_lineup import DEFAULT_CHESS_ID_SUPPLEMENT
from generate_rag.repair_lineup_hero_placeholders import repair_rag_lineup_jsonl

DEFAULT_RAG_LINEUP = DATA_DIR / "rag_lineup_lineup.jsonl"
DEFAULT_V1_OUT = DATA_DIR / "rag_lineup_lineup_v1.jsonl"
DEFAULT_CHESS = DATA_DIR / "rag_legend_chess.jsonl"
DEFAULT_EQUIP = DATA_DIR / "rag_legend_equip.jsonl"
DEFAULT_HEX = DATA_DIR / "rag_legend_hex.jsonl"


def run_pipeline(
    *,
    lineup_json: Optional[Path],
    skip_fetch: bool,
    skip_generate: bool,
    skip_repair: bool,
    chess_url: str,
    supplement_out: Path,
    rag_lineup_path: Path,
    chess_path: Path,
    equip_path: Path,
    hex_path: Path,
    v1_out: Path,
    with_text_v1: bool,
) -> None:
    if not skip_fetch:
        n = write_chess_id_name_supplement(supplement_out, url=chess_url)
        print(f"[1/4] chess 补全表 → {supplement_out}（{n} 条）")
    else:
        print("[1/4] 跳过拉取 chess.js（--skip-fetch）")

    if not skip_generate:
        lj = lineup_json
        if lj is None:
            lj = DEFAULT_LINEUP_JSON if DEFAULT_LINEUP_JSON.is_file() else None
        if lj is not None and lj.is_file():
            n = generate(
                lj.resolve(),
                rag_lineup_path.resolve(),
                chess_path.resolve(),
                equip_path.resolve(),
                hex_path.resolve(),
                no_chess_id_supplement=False,
            )
            print(f"[2/4] 从掌盟 JSON 生成阵容 RAG → {rag_lineup_path}（{n} 条）")
        else:
            print(
                f"[2/4] 跳过生成：未找到 lineup_detail_total.json（可传 --lineup-json 路径）"
            )
    else:
        print("[2/4] 跳过 generate（--skip-generate）")

    if not rag_lineup_path.is_file():
        raise SystemExit(
            f"缺少 {rag_lineup_path}，请先放置掌盟 lineup_detail_total.json 并去掉 --skip-generate，"
            "或从别处拷贝 rag_lineup_lineup.jsonl。"
        )

    if not skip_repair:
        th, ti, nd, _nl = repair_rag_lineup_jsonl(rag_lineup_path.resolve(), dry_run=False)
        print(
            f"[3/4] 修复占位：英雄→{th} 处，ID→{ti} 处，改动文档 {nd} 条 → {rag_lineup_path}"
        )
    else:
        print("[3/4] 跳过 repair（--skip-repair）")

    n = write_rag_lineup_v1_jsonl(
        rag_lineup_path.resolve(),
        v1_out.resolve(),
        include_text_v1=with_text_v1,
    )
    print(f"[4/4] v1 结构化 → {v1_out}（{n} 条）")


def main() -> None:
    ap = argparse.ArgumentParser(description="阵容 RAG 一键：补全表 → lineup jsonl → 修复 → v1")
    ap.add_argument(
        "--lineup-json",
        type=Path,
        default=None,
        help="掌盟 lineup_detail_total.json；省略时若仓库根存在同名文件则自动使用",
    )
    ap.add_argument("--skip-fetch", action="store_true", help="不拉取 chess.js")
    ap.add_argument(
        "--skip-generate",
        action="store_true",
        help="不从 lineup_detail_total 生成 rag_lineup_lineup.jsonl（仅用已有文件）",
    )
    ap.add_argument("--skip-repair", action="store_true", help="不做英雄 ID 占位修复")
    ap.add_argument("--chess-js-url", type=str, default=DEFAULT_CHESS_JS_URL, help="chess.js URL")
    ap.add_argument(
        "--supplement-out",
        type=Path,
        default=DEFAULT_CHESS_ID_SUPPLEMENT,
        help="hero_id 补全表输出路径",
    )
    ap.add_argument("--rag-lineup", type=Path, default=DEFAULT_RAG_LINEUP, help="阵容攻略 jsonl")
    ap.add_argument("--chess", type=Path, default=DEFAULT_CHESS)
    ap.add_argument("--equip", type=Path, default=DEFAULT_EQUIP)
    ap.add_argument("--hex", type=Path, default=DEFAULT_HEX)
    ap.add_argument("--v1-out", type=Path, default=DEFAULT_V1_OUT)
    ap.add_argument(
        "--with-text-v1",
        action="store_true",
        help="v1 每条额外写入 text_v1（与结构化字段重复）",
    )
    args = ap.parse_args()

    run_pipeline(
        lineup_json=args.lineup_json,
        skip_fetch=args.skip_fetch,
        skip_generate=args.skip_generate,
        skip_repair=args.skip_repair,
        chess_url=args.chess_js_url.strip(),
        supplement_out=args.supplement_out.resolve(),
        rag_lineup_path=args.rag_lineup.resolve(),
        chess_path=args.chess.resolve(),
        equip_path=args.equip.resolve(),
        hex_path=args.hex.resolve(),
        v1_out=args.v1_out.resolve(),
        with_text_v1=args.with_text_v1,
    )
    print("完成。")


if __name__ == "__main__":
    main()
