# -*- coding: utf-8 -*-
"""
多模型教练对比：战术快报 + 本地 RAG 与 gemini_v1 一致（system prompt 即 gv._coach_system_prompt()），
经 OpenRouter / 火山 ARK 调用各模型，对固定问题分别计时并打印回答。

用法（仓库根目录）：
  python benchmark_coach_llms.py
  python benchmark_coach_llms.py --summary-json runs/battle_pipeline_v3_out/01-a_summary.json
  python benchmark_coach_llms.py --models "google/gemini-2.5-flash,anthropic/claude-haiku-4.5"

依赖：.env 中 OPENROUTER_API_KEY；可选 ARK_API_KEY；可选 data/rag_lineup_lineup_v1.jsonl、data/rag_core_chess.jsonl。

说明：OpenRouter 上的 model id 会变更；若某模型 404，请改 --models 或到 https://openrouter.ai/models 核对。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 与 gemini_v1 一致：加载 .env
import gemini_v1 as gv  # noqa: E402

DEFAULT_SUMMARY_DIR = REPO_ROOT / "runs" / "battle_pipeline_v3_out"

# (展示名, model id, provider)
DEFAULT_MODEL_ROWS: List[Tuple[str, str, str]] = [
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash", "openrouter"),
    ("Gemini 2.5 Flash Lite", "google/gemini-2.5-flash-lite", "openrouter"),
    ("Gemini 2.5 Flash (Google直连)", "gemini-2.5-flash", "google"),
    ("Gemini 2.0 Flash (Google直连)", "gemini-2.0-flash", "google"),
    ("Doubao Seed 2.0 Mini", "doubao-seed-2-0-mini-260215", "ark"),
    ("Doubao Seed 1.6 Flash", "doubao-seed-1-6-flash-250828", "ark"),
]

DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

DEFAULT_QUESTIONS = [
    "这把怎么玩？",
    "装备怎么给？",
    "站位怎么调？",
]


def _build_user_block(
    brief: str, rag_lineup: str, rag_chess: str, question: str
) -> str:
    """与 gemini_v1._coach_first_user_message 一致：仅【对局情报】（阵容与目标棋子已写入战报）。"""
    _ = rag_lineup
    _ = rag_chess
    parts = [
        "【哈基星问题】\n" + question.strip(),
        "【对局情报】\n" + brief.strip(),
    ]
    return "\n\n".join(parts)


def _estimate_network_jitter_s(base: str, key: str, tries: int = 3) -> float:
    """粗略估计首包前网络抖动（RTT），用于拆分 TTFT。"""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/").strip(),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "gemini_v1").strip(),
    }
    samples: List[float] = []
    url = base.rstrip("/") + "/models"
    for _ in range(max(1, tries)):
        t0 = time.perf_counter()
        try:
            requests.get(url, headers=headers, timeout=8.0)
            samples.append(time.perf_counter() - t0)
        except Exception:
            continue
    if not samples:
        return 0.0
    samples.sort()
    return samples[len(samples) // 2]


def _extract_ark_text(resp_json: Dict[str, Any]) -> str:
    """尽量提取最终回答；若无最终回答，再回退到摘要。"""
    final_chunks: List[str] = []
    fallback_chunks: List[str] = []
    output = resp_json.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        if item_type == "reasoning":
                            fallback_chunks.append(t.strip())
                        else:
                            final_chunks.append(t.strip())
                    t2 = c.get("output_text")
                    if isinstance(t2, str) and t2.strip():
                        if item_type == "reasoning":
                            fallback_chunks.append(t2.strip())
                        else:
                            final_chunks.append(t2.strip())
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                if item_type == "reasoning":
                    fallback_chunks.append(text.strip())
                else:
                    final_chunks.append(text.strip())
            summary = item.get("summary")
            if isinstance(summary, list):
                for s in summary:
                    if not isinstance(s, dict):
                        continue
                    st = s.get("text")
                    if isinstance(st, str) and st.strip():
                        fallback_chunks.append(st.strip())
                    st2 = s.get("summary_text")
                    if isinstance(st2, str) and st2.strip():
                        fallback_chunks.append(st2.strip())
    direct_text = resp_json.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        final_chunks.append(direct_text.strip())

    if final_chunks:
        text = "\n".join(final_chunks).strip()
        text = text.replace("\r", " ").replace("\n", " ").strip()
        return text[:180]

    if fallback_chunks:
        text = "\n".join(fallback_chunks).strip()
        text = text.replace("\r", " ").replace("\n", " ").strip()
        return text[:180]

    return json.dumps(resp_json, ensure_ascii=False)


def _compact_answer_text(text: str, max_chars: int = 80) -> str:
    """将回答压成短句，便于测速场景展示。"""
    s = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if not s:
        return s
    # 优先截到第一个句末，避免冗长段落。
    cut_pos = -1
    for p in ("。", "！", "？", ".", "!", "?"):
        i = s.find(p)
        if i != -1:
            cut_pos = i + 1 if cut_pos == -1 else min(cut_pos, i + 1)
    if cut_pos != -1:
        s = s[:cut_pos].strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "..."
    return s


def _call_google_stream(
    model_id: str,
    user_text: str,
    *,
    temperature: float,
    timeout_s: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Google AI Studio 直连（流式），使用 SSE 接口。"""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not key:
        raise RuntimeError("缺少 GEMINI_API_KEY / GOOGLE_API_KEY 环境变量")

    first_chunk_ts: List[Optional[float]] = [None]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent"
    sys_prompt = gv._coach_system_prompt()
    contents = [
        {"role": "user", "parts": [{"text": sys_prompt}]},
        {"role": "model", "parts": [{"text": "好的，明白了。"}]},
        {"role": "user", "parts": [{"text": user_text}]},
    ]
    body = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }

    sess = requests.Session()
    t0 = time.perf_counter()
    r = sess.post(url, params={"key": key, "alt": "sse"}, json=body, timeout=timeout_s, stream=True)
    if not r.ok:
        try:
            err = r.json().get("error") or {}
            msg = err.get("message", r.text[:600])
        except Exception:
            msg = r.text[:600]
        raise RuntimeError(f"Google 流式 HTTP {r.status_code}: {msg}")

    acc: List[str] = []
    for raw in r.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, (bytes, bytearray)) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        for cand in evt.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content") or {}
            for p in content.get("parts") or []:
                if not isinstance(p, dict):
                    continue
                t = p.get("text")
                if isinstance(t, str) and t:
                    acc.append(t)
                    if first_chunk_ts[0] is None:
                        first_chunk_ts[0] = time.perf_counter()

    t1 = time.perf_counter()
    answer = "".join(acc).strip()
    ttft = (first_chunk_ts[0] - t0) if first_chunk_ts[0] is not None else (t1 - t0)
    gen_sec = (t1 - first_chunk_ts[0]) if first_chunk_ts[0] is not None else 0.0
    return {
        "answer": answer,
        "total_sec": (t1 - t0),
        "ttft_sec": ttft,
        "gen_sec": gen_sec,
    }


def _call_openrouter_stream(
    model_id: str,
    user_text: str,
    *,
    temperature: float,
    timeout_s: float,
    max_tokens: int,
) -> Dict[str, Any]:
    first_chunk_ts: List[Optional[float]] = [None]

    def _on_chunk(_: str) -> None:
        if first_chunk_ts[0] is None:
            first_chunk_ts[0] = time.perf_counter()

    t0 = time.perf_counter()
    answer = gv._openrouter_chat_completion(
        messages=[
            {"role": "system", "content": gv._coach_system_prompt()},
            {"role": "user", "content": user_text},
        ],
        model=model_id,
        temperature=temperature,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        stream_output=True,
        on_stream_chunk=_on_chunk,
    )
    t1 = time.perf_counter()
    ttft = (first_chunk_ts[0] - t0) if first_chunk_ts[0] is not None else (t1 - t0)
    gen_sec = (t1 - first_chunk_ts[0]) if first_chunk_ts[0] is not None else 0.0
    return {
        "answer": answer,
        "total_sec": (t1 - t0),
        "ttft_sec": ttft,
        "gen_sec": gen_sec,
    }


def _call_ark_responses(
    model_id: str,
    user_text: str,
    *,
    temperature: float,
    timeout_s: float,
    max_tokens: int,
    ark_base: str,
    ark_api_key: str,
    ark_thinking_type: str,
) -> Dict[str, Any]:
    if not ark_api_key.strip():
        raise RuntimeError("缺少 ARK API key，请设置环境变量 ARK_API_KEY 或 --ark-api-key")

    url = ark_base.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {ark_api_key}",
        "Content-Type": "application/json",
    }
    concise_user_text = (
        user_text.strip()
        + "\n\n【输出要求】只输出最终简短回答（1-2句），禁止展示分析过程、思考过程、推理过程。"
    )
    payload = {
        "model": model_id,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": gv._coach_system_prompt()}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": concise_user_text}],
            },
        ],
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": ark_thinking_type},
    }
    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    t1 = time.perf_counter()
    if r.status_code >= 400:
        body = (r.text or "").strip().replace("\n", " ")
        if len(body) > 300:
            body = body[:300] + "..."
        raise RuntimeError(f"ARK HTTP {r.status_code}: {body}")
    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"ARK 返回非 JSON: {e}") from e

    answer = _extract_ark_text(data)
    total = (t1 - t0)
    return {
        "answer": answer,
        "total_sec": total,
        "ttft_sec": total,  # 非流式，TTFT 近似总耗时
        "gen_sec": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenRouter/ARK 多模型教练对比（与 gemini_v1 同 prompt/RAG）")
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
        help="逗号分隔模型列表。支持 model_id（默认 openrouter）或 provider:model_id（如 ark:doubao-seed-2-0-mini-260215 / google:gemini-2.5-flash）",
    )
    ap.add_argument(
        "--ark-api-key",
        type=str,
        default=(os.getenv("ARK_API_KEY") or "").strip(),
        help="火山 ARK API Key；默认读取 ARK_API_KEY",
    )
    ap.add_argument(
        "--ark-base-url",
        type=str,
        default=(os.getenv("ARK_BASE_URL") or DEFAULT_ARK_BASE_URL).strip(),
        help="火山 ARK Base URL（默认 https://ark.cn-beijing.volces.com/api/v3）",
    )
    ap.add_argument(
        "--ark-thinking-type",
        type=str,
        choices=["disabled", "enabled"],
        default=(os.getenv("ARK_THINKING_TYPE") or "disabled").strip().lower(),
        help="ARK thinking 开关：disabled 仅输出最终答案；enabled 允许思考内容",
    )
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--max-tokens", type=int, default=100, help="限制输出 token 上限（默认100）")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json-out", type=Path, default=None, help="将原始结果写入 JSON（utf-8）")
    args = ap.parse_args()

    if args.summary_json:
        sp = args.summary_json.resolve()
    else:
        sp = gv._find_first_summary_json(args.summary_dir.resolve())

    if not sp.is_file():
        raise SystemExit(f"找不到 summary: {sp}")

    coach_args = SimpleNamespace(
        rag_lineup=args.rag_lineup.resolve(),
        rag_top_k=args.rag_top_k,
        rag_min_quality=args.rag_min_quality,
        rag_core_chess=args.rag_core_chess.resolve(),
        rag_chess_top_k=args.rag_chess_top_k,
        no_rag=False,
    )
    bundle = gv.build_coach_bundle(coach_args, sp.resolve())
    brief = str(bundle.get("brief") or "")
    lineup_rag = str(bundle.get("rag_block") or "")
    chess_rag = str(bundle.get("chess_block") or "")

    if args.models.strip():
        ids = [x.strip() for x in args.models.split(",") if x.strip()]
        model_rows: List[Tuple[str, str, str]] = []
        for raw in ids:
            provider = "openrouter"
            model_id = raw
            if ":" in raw:
                maybe_provider, maybe_id = raw.split(":", 1)
                mp = maybe_provider.strip().lower()
                if mp in {"openrouter", "ark", "google"} and maybe_id.strip():
                    provider = mp
                    model_id = maybe_id.strip()
            model_rows.append((model_id, model_id, provider))
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
    max_tokens = max(1, int(args.max_tokens))
    base, key, _ = gv._openrouter_env()
    net_jitter_openrouter = _estimate_network_jitter_s(base, key, tries=3)
    net_jitter_ark = _estimate_network_jitter_s(args.ark_base_url, args.ark_api_key, tries=3) if args.ark_api_key else 0.0

    print(f"流式: 开启 | max_tokens={max_tokens}")
    print(f"网络抖动估计(OpenRouter 中位RTT): {net_jitter_openrouter:.3f}s")
    print(f"网络抖动估计(ARK 中位RTT): {net_jitter_ark:.3f}s")

    for label, model_id, provider in model_rows:
        print(f"\n{'#' * 72}\n# 模型: {label}\n# id: {model_id}\n# provider: {provider}\n{'#' * 72}")
        model_total = 0.0
        for qi, q in enumerate(questions, start=1):
            user_text = _build_user_block(brief, lineup_rag, chess_rag, q)
            err: Optional[str] = None
            answer = ""
            total_sec = 0.0
            ttft_sec = 0.0
            gen_sec = 0.0
            prefirst_wait_sec = 0.0
            queue_prefill_est_sec = 0.0
            net_jitter_est = net_jitter_ark if provider == "ark" else (
                _estimate_network_jitter_s(
                    "https://generativelanguage.googleapis.com/v1beta/models/",
                    os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "",
                    tries=3,
                ) if provider == "google" else net_jitter_openrouter
            )
            try:
                if provider == "ark":
                    ret = _call_ark_responses(
                        model_id,
                        user_text,
                        temperature=coach_temp,
                        timeout_s=timeout_s,
                        max_tokens=max_tokens,
                        ark_base=args.ark_base_url,
                        ark_api_key=args.ark_api_key,
                        ark_thinking_type=args.ark_thinking_type,
                    )
                elif provider == "google":
                    ret = _call_google_stream(
                        model_id,
                        user_text,
                        temperature=coach_temp,
                        timeout_s=timeout_s,
                        max_tokens=max_tokens,
                    )
                else:
                    ret = _call_openrouter_stream(
                        model_id,
                        user_text,
                        temperature=coach_temp,
                        timeout_s=timeout_s,
                        max_tokens=max_tokens,
                    )
                answer = str(ret.get("answer") or "")
                answer = _compact_answer_text(answer, max_chars=80)
                total_sec = float(ret.get("total_sec") or 0.0)
                ttft_sec = float(ret.get("ttft_sec") or 0.0)
                gen_sec = float(ret.get("gen_sec") or 0.0)
                prefirst_wait_sec = ttft_sec
                queue_prefill_est_sec = max(0.0, prefirst_wait_sec - net_jitter_est)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            model_total += total_sec

            results.append(
                {
                    "model_label": label,
                    "model_id": model_id,
                    "provider": provider,
                    "question_index": qi,
                    "question": q,
                    "seconds": round(total_sec, 3),
                    "ttft_sec": round(ttft_sec, 3),
                    "prefirst_wait_sec": round(prefirst_wait_sec, 3),
                    "network_jitter_est_sec": round(net_jitter_est, 3),
                    "queue_prefill_est_sec": round(queue_prefill_est_sec, 3),
                    "generation_sec": round(gen_sec, 3),
                    "answer": answer if not err else "",
                    "error": err,
                }
            )

            print(f"\n--- 问题 {qi}/{len(questions)}: {q} ---")
            print(
                f"总耗时: {total_sec:.2f}s | TTFT: {ttft_sec:.2f}s | "
                f"首包前等待: {prefirst_wait_sec:.2f}s | "
                f"网关排队/预填充(估): {queue_prefill_est_sec:.2f}s | "
                f"网络抖动(估): {net_jitter_est:.2f}s | 生成: {gen_sec:.2f}s"
            )
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
            "models": [{"label": a, "id": b, "provider": c} for a, b, c in model_rows],
            "questions": questions,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 JSON: {out_path}")


if __name__ == "__main__":
    main()
