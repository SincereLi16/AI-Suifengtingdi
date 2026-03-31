# -*- coding: utf-8 -*-
"""
多模型教练对比：战术快报 + 本地 RAG 与 gemini_v1 一致（system prompt 即 gv._coach_system_prompt()），
经 OpenRouter 调用各模型，对固定问题分别计时并打印回答。

用法（仓库根目录）：
  python benchmark_coach_llms.py
  python benchmark_coach_llms.py --summary-json runs/battle_pipeline_v3_out/01-a_summary.json
  python benchmark_coach_llms.py --models "google/gemini-2.5-flash,anthropic/claude-haiku-4.5"

依赖：.env 中 OPENROUTER_API_KEY；可选 data/rag_lineup_lineup.jsonl、data/rag_core_chess.jsonl。

说明：OpenRouter 上的 model id 会变更；若某模型 404，请改 --models 或到 https://openrouter.ai/models 核对。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 与 gemini_v1 一致：加载 .env
import gemini_v1 as gv  # noqa: E402

DEFAULT_SUMMARY_DIR = REPO_ROOT / "runs" / "battle_pipeline_v3_out"

# (展示名, OpenRouter model id) —— id 以 OpenRouter 控制台为准，可随时改 DEFAULT_MODEL_ROWS
DEFAULT_MODEL_ROWS: List[Tuple[str, str]] = [
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash"),
    ("Claude 4 Haiku", "anthropic/claude-haiku-4.5"),
    ("Llama 3.3 70B Instruct", "meta-llama/llama-3.3-70b-instruct"),
    ("Llama 4 Scout (Lightweight)", "meta-llama/llama-4-scout"),
    ("Qwen2.5 Coder 32B Instruct", "qwen/qwen-2.5-coder-32b-instruct"),
]

DEFAULT_QUESTIONS = [
    "这把玩什么?",
    "装备怎么给？",
    "站位如何调整",
]


def _build_user_block(
    brief: str, rag_lineup: str, rag_chess: str, question: str
) -> str:
    """与 gemini_v1._coach_first_user_message 一致：对局情报 + 阵容攻略附录，不重复棋子智库全文。"""
    _ = rag_chess
    parts = [
        "【哈基星问题】\n" + question.strip(),
        "【对局情报】\n" + brief.strip(),
    ]
    rl = (rag_lineup or "").strip()
    if rl and not rl.startswith("（本回合未注入"):
        parts.append("【阵容攻略原文（附录）】\n" + rl)
    return "\n\n".join(parts)


def _build_rag_blocks(
    summary: Dict[str, Any],
    *,
    rag_lineup: Path,
    rag_core: Path,
    rag_top_k: int,
    rag_chess_top_k: int,
    rag_min_quality: Optional[str],
) -> Tuple[str, str]:
    min_q: Optional[str] = None
    if (rag_min_quality or "").strip():
        t = str(rag_min_quality).strip().upper()
        c = t[0] if t else ""
        min_q = c if c in gv._LINEUP_QUALITY_ORDER else None
    lineup_block, _, _, lineup_docs = gv.retrieve_lineup_rag(
        summary,
        rag_lineup,
        top_k=max(1, rag_top_k),
        min_quality=min_q,
    )
    lineup_top1 = lineup_docs[0] if lineup_docs else None
    core_block, _, _ = gv.retrieve_core_chess_rag(
        summary,
        rag_core.resolve(),
        top_k=max(1, rag_chess_top_k),
        lineup_top_doc=lineup_top1,
        legend_chess_path=gv.DEFAULT_RAG_LEGEND_CHESS,
        summary_json_path=None,
    )
    return lineup_block, core_block


def _call_openrouter(model_id: str, user_text: str, *, temperature: float, timeout_s: float) -> str:
    return gv._openrouter_chat_completion(
        messages=[
            {"role": "system", "content": gv._coach_system_prompt()},
            {"role": "user", "content": user_text},
        ],
        model=model_id,
        temperature=temperature,
        timeout_s=timeout_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenRouter 多模型教练对比（与 gemini_v1 同 prompt/RAG）")
    ap.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="含 *_summary.json 的目录（未指定 --summary-json 时使用）",
    )
    ap.add_argument("--summary-json", type=Path, default=None, help="直接指定 *_summary.json")
    ap.add_argument("--rag-lineup", type=Path, default=gv.DEFAULT_RAG_LINEUP)
    ap.add_argument("--rag-core-chess", type=Path, default=gv.DEFAULT_RAG_CORE_CHESS)
    ap.add_argument("--rag-top-k", type=int, default=3)
    ap.add_argument("--rag-chess-top-k", type=int, default=8)
    ap.add_argument(
        "--rag-min-quality",
        type=str,
        default=(os.getenv("RAG_MIN_QUALITY") or "A").strip(),
        help="阵容 RAG 最低评级，S|A|B|…；空字符串表示不过滤",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="",
        help="逗号分隔的 OpenRouter model id，覆盖默认列表；无展示名时 id 即展示名",
    )
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json-out", type=Path, default=None, help="将原始结果写入 JSON（utf-8）")
    args = ap.parse_args()

    if args.summary_json:
        sp = args.summary_json.resolve()
    else:
        sp = gv._find_first_summary_json(args.summary_dir.resolve())

    if not sp.is_file():
        raise SystemExit(f"找不到 summary: {sp}")

    summary = json.loads(sp.read_text(encoding="utf-8"))
    brief = gv.build_tactical_brief(summary, summary_json_path=sp)
    mq = args.rag_min_quality if str(args.rag_min_quality).strip() else None
    lineup_rag, chess_rag = _build_rag_blocks(
        summary,
        rag_lineup=args.rag_lineup.resolve(),
        rag_core=args.rag_core_chess.resolve(),
        rag_top_k=args.rag_top_k,
        rag_chess_top_k=args.rag_chess_top_k,
        rag_min_quality=mq,
    )

    if args.models.strip():
        ids = [x.strip() for x in args.models.split(",") if x.strip()]
        model_rows: List[Tuple[str, str]] = [(mid, mid) for mid in ids]
    else:
        model_rows = list(DEFAULT_MODEL_ROWS)

    questions = list(DEFAULT_QUESTIONS)

    print("=" * 72)
    print("benchmark_coach_llms")
    print(f"summary: {sp}")
    print(f"模型数: {len(model_rows)}  问题数: {len(questions)}")
    print("=" * 72)

    results: List[Dict[str, Any]] = []
    coach_temp = float(args.temperature)
    timeout_s = float(args.timeout)

    for label, model_id in model_rows:
        print(f"\n{'#' * 72}\n# 模型: {label}\n# id: {model_id}\n{'#' * 72}")
        model_total = 0.0
        for qi, q in enumerate(questions, start=1):
            user_text = _build_user_block(brief, lineup_rag, chess_rag, q)
            err: Optional[str] = None
            answer = ""
            t0 = time.perf_counter()
            try:
                answer = _call_openrouter(
                    model_id, user_text, temperature=coach_temp, timeout_s=timeout_s
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            model_total += elapsed

            results.append(
                {
                    "model_label": label,
                    "model_id": model_id,
                    "question_index": qi,
                    "question": q,
                    "seconds": round(elapsed, 3),
                    "answer": answer if not err else "",
                    "error": err,
                }
            )

            print(f"\n--- 问题 {qi}/{len(questions)}: {q} ---")
            print(f"耗时: {elapsed:.2f}s")
            if err:
                print(f"[错误] {err}")
            else:
                print(answer)

        print(f"\n>> {label} 本批问题合计: {model_total:.2f}s")

    if args.json_out:
        out_path = args.json_out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary_path": str(sp),
            "models": [{"label": a, "id": b} for a, b in model_rows],
            "questions": questions,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 JSON: {out_path}")


if __name__ == "__main__":
    main()
