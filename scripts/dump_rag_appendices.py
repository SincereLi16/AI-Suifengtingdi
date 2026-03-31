# -*- coding: utf-8 -*-
"""
打印 build_coach_bundle 中的阵容/棋子 RAG 原文块，便于对照终端 brief。
注：当前 coach 首轮仅将「阵容攻略原文」附录喂给 LLM；棋子块仍可由本脚本导出供人工查看。

用法（在项目根目录）:
  python scripts/dump_rag_appendices.py --summary-json runs/.../01-a_summary.json
  python scripts/dump_rag_appendices.py --summary-json path/to.json -o appendices.txt

与 brief 的差异（概要）:
  - 阵容附录: 命中攻略的「全文」长文本（Top-K 条，每条可截断），不是战报里 🏆 的短 Top 行。
  - 棋子附录: 完整【棋子智库】结构化块（基准阵容、一目标、二挂件、三装备继承等），
    战报里仅把其中一部分拆成 🎯/♻️/🔗 等小节展示。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _brief_vs_appendix_note() -> str:
    return """【对照说明】终端 brief 与两份「附录」的差别（概要）

1) 阵容攻略原文（附录）
   - 来源: rag_lineup_lineup.jsonl，按对局关键词检索后的 Top-K 条攻略「正文」全文（单条过长会截断）。
   - brief 里对应: 「🏆 推荐阵容」下的 Top1/2/3 短行（名称 + 羁绊摘要），不含攻略长文。
   - 多出来的信息: 运营节奏、站位细节、出装段落、海克斯建议等攻略原文里的一切（若命中）。

2) 棋子智库原文（附录）
   - 来源: retrieve_core_chess_rag 拼出的完整【棋子智库】块。
   - brief 里对应: 同一 chess_meta 派生的 🎯 目标棋子、♻️ 弃子清单、🔗 装备继承（版式不同，数据同源）。
   - 多出来的信息通常包括:
     · 「基准阵容（阵容智库 Top1）」的 lineup_id / 名称说明；
     · 「一、目标棋子」与战报列表一致时，附录里多为行首两空格缩进；
     · 「二、可替挂件」逐条站位信息（战报打工区也有类似行，字段顺序可能略异）；
     · 「三、装备继承」完整段落（含无可继承时的说明），与 🔗 小节同源或等价。

以下两段即为送入模型的附录全文（与 coach 首轮 _coach_first_user_message 中附加块一致）。
"""


def main() -> None:
    from gemini_v1 import (
        DEFAULT_RAG_CORE_CHESS,
        DEFAULT_RAG_LINEUP,
        build_coach_bundle,
    )

    ap = argparse.ArgumentParser(
        description="导出阵容/棋子 RAG 附录全文，对照终端 brief。"
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="*_summary.json 路径",
    )
    ap.add_argument("--rag-lineup", type=Path, default=DEFAULT_RAG_LINEUP)
    ap.add_argument("--rag-top-k", type=int, default=3)
    ap.add_argument("--rag-core-chess", type=Path, default=DEFAULT_RAG_CORE_CHESS)
    ap.add_argument("--rag-chess-top-k", type=int, default=8)
    ap.add_argument(
        "--rag-min-quality",
        type=str,
        default="A",
        help="阵容评级过滤，与 gemini_v1 一致；传 - 或 all 表示不过滤",
    )
    ap.add_argument("--no-rag", action="store_true", help="不检索附录（仅占位句）")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="写入该文件；默认打印到 stdout",
    )
    ap.add_argument(
        "--include-brief",
        action="store_true",
        help="在附录前额外输出与终端一致的 brief（较长）",
    )
    args = ap.parse_args()

    sp = args.summary_json.resolve()
    if not sp.is_file():
        raise SystemExit(f"找不到文件: {sp}")

    ns = SimpleNamespace(
        no_rag=args.no_rag,
        rag_lineup=args.rag_lineup,
        rag_top_k=args.rag_top_k,
        rag_core_chess=args.rag_core_chess,
        rag_chess_top_k=args.rag_chess_top_k,
        rag_min_quality=args.rag_min_quality,
    )

    bundle = build_coach_bundle(ns, sp)
    brief = str(bundle.get("brief") or "")
    rag_block = str(bundle.get("rag_block") or "")
    chess_block = str(bundle.get("chess_block") or "")

    sep = "=" * 72
    chunks: list[str] = [
        _brief_vs_appendix_note(),
        "",
        sep,
        "【阵容攻略原文（附录）】（送入模型时的 rag_block）",
        sep,
        rag_block,
        "",
        sep,
        "【棋子智库原文（附录）】（送入模型时的 chess_block）",
        sep,
        chess_block,
        "",
    ]
    if args.include_brief:
        chunks = [
            sep,
            "【对局情报】brief（与终端预览一致，可选）",
            sep,
            brief,
            "",
        ] + chunks

    out = "\n".join(chunks).rstrip() + "\n"

    if args.output:
        args.output.write_text(out, encoding="utf-8")
        print(f"已写入: {args.output.resolve()}", file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
