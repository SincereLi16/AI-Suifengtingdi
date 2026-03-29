# -*- coding: utf-8 -*-
"""
Pipeline 对局 JSON → 战术快报（含棋盘棋子星级）→ 阵容智库 RAG + 棋子智库 RAG → 随风听笛回答。

用法示例：
  python gemini_v1.py
  # 默认：若 runs/battle_pipeline_v3_out 已有 *_summary.json 则读缓存；否则静默跑 pipeline。
  python gemini_v1.py --summary-json runs/.../01-a_summary.json  # 跳过 pipeline，仅调试
  python gemini_v1.py --img-dir "对局截图" -q "我这把该先合什么？"
  python gemini_v1.py --no-rag   # 关闭本地 RAG（阵容+棋子），只喂战术快报 + 问题

  交互终端下：随风听笛答完后可继续输入追问（首轮已含快报与双智库，追问不重跑 RAG）；空行或 q / quit / exit 结束。
  管道或非 TTY  stdin：只跑一轮（与原先一致）。

说明：
  - 当前 LLM：默认 OpenRouter，模型 id 见 .env 的 OPENROUTER_TEXT_MODEL（如 google/gemini-2.5-flash，即 Gemini Flash 系）。
  - GEMINI_BACKEND=google 可走 Google 直连；模型见 GEMINI_MODEL。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent


def _load_repo_dotenv() -> None:
    """
    从仓库根目录 .env 注入环境变量（不依赖 python-dotenv，作后备）。
    若某键未设置或值为空，则用文件中的值覆盖（便于补全 OPENROUTER_API_KEY）。
    """
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, _, rest = s.partition("=")
        key = key.strip().strip("\ufeff")  # 防 BOM 粘在键名上
        val = rest.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if not key:
            continue
        cur = str(os.environ.get(key, "")).strip()
        if key not in os.environ or not cur:
            os.environ[key] = val


try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass
except Exception:
    pass

_load_repo_dotenv()

# requests 导入时会检查 urllib3/charset 版本；提示含换行，需 (?s) 才能整段匹配
warnings.filterwarnings("ignore", message=r"(?s).*doesn't match a supported version.*")

import requests  # noqa: E402
# 与 pipeline 默认输出一致；若目录内已有 *_summary.json，gemini_v1 优先读缓存、不重复跑识别
PIPELINE_CACHE_DIR = REPO_ROOT / "runs" / "battle_pipeline_v3_out"
DEFAULT_PIPELINE_OUT = PIPELINE_CACHE_DIR
DEFAULT_IMG_DIR = REPO_ROOT / "对局截图"
DEFAULT_RAG_LINEUP = REPO_ROOT / "data" / "rag_lineup_lineup.jsonl"
DEFAULT_RAG_CORE_CHESS = REPO_ROOT / "data" / "rag_core_chess.jsonl"
# 阵容攻略 quality：字母越靠前越强（与掌盟评级一致）
_LINEUP_QUALITY_ORDER = "SABCDEFGH"


def _lineup_quality_rank(q: str) -> int:
    c = (q or "").strip().upper()[:1]
    i = _LINEUP_QUALITY_ORDER.find(c)
    return i if i >= 0 else len(_LINEUP_QUALITY_ORDER) + 5


def _normalize_openrouter_api_key(raw: str) -> str:
    """去掉首尾空白与成对引号，避免 .env 里多打引号导致 401。"""
    k = (raw or "").strip()
    if len(k) >= 2 and k[0] == k[-1] and k[0] in "\"'":
        k = k[1:-1].strip()
    return k


def _openrouter_env() -> tuple[str, str, str]:
    """与 scripts/extra/router.OpenRouterConfig 一致：OpenRouter 兼容 Chat Completions（非 OpenAI 公司专属）。"""
    base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
    key = _normalize_openrouter_api_key(os.getenv("OPENROUTER_API_KEY", ""))
    vision = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash").strip()
    text = os.getenv("OPENROUTER_TEXT_MODEL", vision).strip()
    return base, key, text


def _openrouter_chat_completion(
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.35,
    timeout_s: float = 120.0,
) -> str:
    base, key, _ = _openrouter_env()
    if not key:
        raise RuntimeError("请配置环境变量 OPENROUTER_API_KEY（.env 或系统环境）")
    try:
        key.encode("ascii")
    except UnicodeEncodeError as e:
        raise RuntimeError("OPENROUTER_API_KEY 须为纯 ASCII，请从 OpenRouter 控制台重新复制") from e

    url = f"{base}/chat/completions"
    # OpenRouter 官方建议附带 Referer / Title，部分环境缺省时仍可避免异常路由
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/").strip()
    app_title = os.getenv("OPENROUTER_APP_TITLE", "gemini_v1").strip()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": app_title,
    }
    r = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=timeout_s,
    )
    if not r.ok:
        body = (r.text or "")[:800]
        if r.status_code == 401:
            hint = (
                "OpenRouter 返回 401（User not found）：API Key 无效、已撤销，或不是 OpenRouter 的 Key。\n"
                "请登录 https://openrouter.ai/keys 重新生成，写入 .env 的 OPENROUTER_API_KEY=\n"
                "（Key 一般以 sk-or-v1- 开头；勿加引号、勿复制进全角空格）。\n"
                f"原始响应: {body}"
            )
            raise RuntimeError(hint)
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {body}")
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter 响应无 choices: " + repr(data)[:500])
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    # 少数实现返回多段
    return "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in content
    ).strip()


def _google_gemini_key() -> str:
    """Google AI Studio / Gemini Developer API Key（与 OpenRouter 无关）。"""
    return _normalize_openrouter_api_key(
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    )


def _resolve_chat_backend() -> str:
    """
    GEMINI_BACKEND（默认 openrouter）：
      - openrouter / auto：优先 OPENROUTER；无 Key 或未配置时回退 GEMINI_API_KEY 直连。
      - google：优先直连 Google；无则回退 OpenRouter。
      - google_only：仅直连。
    """
    pref = os.getenv("GEMINI_BACKEND", "openrouter").strip().lower()
    gk = _google_gemini_key()
    ok = _normalize_openrouter_api_key(os.getenv("OPENROUTER_API_KEY", ""))

    if pref in ("auto", ""):
        pref = "openrouter"

    if pref == "google_only":
        if not gk:
            raise RuntimeError(
                "GEMINI_BACKEND=google_only，但未找到 GEMINI_API_KEY 或 GOOGLE_API_KEY。\n"
                "请到 https://aistudio.google.com/apikey 创建密钥并写入 .env"
            )
        return "google"

    if pref == "google":
        if gk:
            return "google"
        if ok:
            print("[gemini_v1] 未配置 GEMINI_API_KEY/GOOGLE_API_KEY，改用 OpenRouter")
            return "openrouter"
        raise RuntimeError(
            "请配置 GEMINI_API_KEY（见 https://aistudio.google.com/apikey ）或 OPENROUTER_API_KEY。"
        )

    # 默认 openrouter：优先 OpenRouter
    if pref == "openrouter":
        if ok:
            return "openrouter"
        if gk:
            env_path = REPO_ROOT / ".env"
            print(
                f"[gemini_v1] 未配置有效的 OPENROUTER_API_KEY（已读 {env_path}），改用 Google 直连（GEMINI_API_KEY）。"
            )
            return "google"
        raise RuntimeError(
            "请配置 OPENROUTER_API_KEY，或设置 GEMINI_API_KEY 以直连 Google。"
        )

    raise RuntimeError(f"未知 GEMINI_BACKEND={pref!r}，请使用 openrouter / google / google_only / auto")


def _google_hist_to_contents(chat_hist: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """OpenAI 风格 user/assistant 轮次 → Gemini generateContent 的 contents（assistant→model）。"""
    out: List[Dict[str, Any]] = []
    for m in chat_hist:
        role = str(m.get("role") or "")
        text = str(m.get("content") or "")
        if role == "user":
            out.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            out.append({"role": "model", "parts": [{"text": text}]})
    return out


def _google_gemini_generate(
    *,
    system_prompt: str,
    user_text: str,
    model: str,
    temperature: float = 0.35,
    timeout_s: float = 120.0,
) -> str:
    """Gemini 单轮（兼容旧调用）。"""
    return _google_gemini_chat(
        system_prompt=system_prompt,
        chat_hist=[{"role": "user", "content": user_text}],
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
    )


def _google_gemini_chat(
    *,
    system_prompt: str,
    chat_hist: List[Dict[str, str]],
    model: str,
    temperature: float = 0.35,
    timeout_s: float = 120.0,
) -> str:
    """Gemini Developer API 多轮：chat_hist 为 user/assistant 交替，须以 user 开头。"""
    key = _google_gemini_key()
    if not key:
        raise RuntimeError("缺少 GEMINI_API_KEY / GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    contents = _google_hist_to_contents(chat_hist)
    if not contents:
        raise RuntimeError("Gemini 多轮 contents 为空")
    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    r = requests.post(url, params={"key": key}, json=body, timeout=timeout_s)
    data = r.json()
    if not r.ok:
        err = data.get("error") or {}
        raise RuntimeError(
            f"Gemini HTTP {r.status_code}: {err.get('message', r.text[:600])}"
        )
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError("Gemini 无 candidates: " + repr(data)[:500])
    parts = (cands[0].get("content") or {}).get("parts") or []
    texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
    return "".join(texts).strip()


def _field_parsed(player: Dict[str, Any], key: str) -> str:
    fields = (player.get("fields") or {}) if isinstance(player, dict) else {}
    d = fields.get(key)
    if isinstance(d, dict):
        v = str(d.get("parsed") or "").strip()
        return v
    return ""


def _star_part_from_row(r: Dict[str, Any]) -> str:
    """棋子星级（fightboard / pipeline confirmed 行内的 star.pred）。"""
    st = r.get("star")
    if not isinstance(st, dict):
        return ""
    pr = st.get("pred")
    try:
        n = int(pr)
    except (TypeError, ValueError):
        return ""
    if 1 <= n <= 3:
        return f" {n}星"
    return ""


def _board_line_hero_display_name(r: Dict[str, Any]) -> str:
    """单格棋子展示名：与战报「棋盘与装备」一致（数据源为 TCV 写入的 confirmed_fightboard_results 每行 best；低置信度加 ?）。"""
    if not isinstance(r, dict):
        return "?"
    name = str(r.get("best") or "?")
    conf = str(r.get("confidence") or "")
    if conf == "low":
        name = f"{name}?"
    return name


def _tcv_board_hero_names_for_rag(summary: Dict[str, Any]) -> List[str]:
    """
    仅用于棋子智库：与战术快报棋盘行同源的棋子名列表（去重保序）。
    匹配 jsonl 时去掉展示用尾随 ?，不包含羁绊栏关键词。
    """
    rows = summary.get("confirmed_fightboard_results")
    if not isinstance(rows, list):
        return []
    out: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        disp = _board_line_hero_display_name(r)
        key = disp.rstrip("?").rstrip("？").strip()
        if key and key not in ("?", "？"):
            out.append(key)
    return list(dict.fromkeys(out))


def _format_board_lines(summary: Dict[str, Any]) -> List[str]:
    rows = summary.get("confirmed_fightboard_results")
    if not isinstance(rows, list):
        return []
    fight = (summary.get("modules") or {}).get("fightboard") or {}
    equip_by_bar = fight.get("equip_by_bar") or {}
    lines: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = _board_line_hero_display_name(r)
        pos = r.get("position") or {}
        loc = ""
        if isinstance(pos, dict):
            lab = str(pos.get("label") or "")
            cr, cc = pos.get("cell_row"), pos.get("cell_col")
            if cr is not None and cc is not None:
                loc = f"({int(cr)},{int(cc)})"  # 行,列
            elif lab:
                loc = lab
        bi = int(r.get("bar_index") or 0)
        eqs: List[str] = []
        raw_eq = equip_by_bar.get(str(bi))
        if raw_eq is None and isinstance(equip_by_bar, dict):
            raw_eq = equip_by_bar.get(bi)
        if isinstance(raw_eq, list):
            for e in raw_eq:
                if isinstance(e, dict):
                    n = str(e.get("name") or "").strip()
                    if n:
                        eqs.append(n)
        eq_str = "、".join(eqs) if eqs else "无"
        loc_s = loc if loc else "?"
        sx = _star_part_from_row(r)
        lines.append(f"{name} 站位{loc_s}{sx} | 装备:{eq_str}")
    return lines


def _self_hp_from_player(player: Dict[str, Any]) -> str:
    """从 player_onnx 血量列解析「我」对应行的血量；无则返回空串。"""
    fields = (player.get("fields") or {}) if isinstance(player, dict) else {}
    hp_nick = fields.get("hp_nick")
    if not isinstance(hp_nick, dict):
        return ""
    cells = hp_nick.get("player_cells")
    if not isinstance(cells, list):
        return ""
    for c in cells:
        if not isinstance(c, dict):
            continue
        idt = str(c.get("id_text") or "").strip()
        if idt in ("我", "自己"):
            hpv = str(c.get("hp") or "").strip()
            if hpv and hpv != "?":
                return hpv
            return ""
    # 部分对局 OCR 未认出「我」但列表第一项即本人
    if len(cells) == 1 and isinstance(cells[0], dict):
        hpv = str(cells[0].get("hp") or "").strip()
        if hpv and hpv != "?":
            return hpv
    return ""


def _equip_column_labels(summary: Dict[str, Any]) -> str:
    ec = (summary.get("modules") or {}).get("equip_column") or {}
    m = ec.get("matches") if isinstance(ec, dict) else None
    if not isinstance(m, list) or not m:
        return "无"
    names: List[str] = []
    for x in m:
        if isinstance(x, dict):
            n = str(x.get("name_stem") or "").strip()
            if n:
                names.append(n)
    return "、".join(names) if names else "无"


def build_tactical_brief(summary: Dict[str, Any]) -> str:
    """将 pipeline 的 *_summary.json 压成自然语言「战术快报」，不含 TCV 调试长文。"""
    player = (summary.get("modules") or {}).get("player") or {}
    analysis = summary.get("analysis") or {}
    gt = (analysis.get("group_traits_merged") or {}) if isinstance(analysis, dict) else {}

    phase = _field_parsed(player, "phase").replace("总", "")
    level = _field_parsed(player, "level")
    exp = _field_parsed(player, "exp")
    gold = _field_parsed(player, "gold")
    streak = _field_parsed(player, "streak")
    my_hp = _self_hp_from_player(player)
    hp_self = my_hp if my_hp else "?"

    bonds_line = str(gt.get("merged_bonds_one_line") or "").strip()
    tm = gt.get("trait_count_max") if isinstance(gt, dict) else None
    if isinstance(tm, dict) and tm:
        trait_summary = "，".join(f"{int(n)}{t}" for t, n in sorted(tm.items(), key=lambda x: int(x[1]), reverse=True)[:12])
    else:
        trait_summary = bonds_line or "(无)"

    board_lines = _format_board_lines(summary)
    board_block = "\n".join(f"  - {x}" for x in board_lines) if board_lines else "  (无)"

    return f"""[当前局势]
阶段 {phase or '?'} | 等级 {level or '?'} | 经验 {exp or '?'} | 金币 {gold or '?'} | 胜负 {streak or '?'} | 我的血量 {hp_self}

[羁绊]
{trait_summary}

[棋盘与装备]
（「站位」后为棋盘行,列；紧邻为棋子识别星级，来自 fightboard）
{board_block}

[左侧装备栏散件]
{_equip_column_labels(summary)}
"""


def _load_lineup_docs(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _extract_keywords_for_rag(
    summary: Dict[str, Any],
    *,
    include_fightboard_agg_top: bool = True,
) -> List[str]:
    """从 summary 抽可用于匹配阵容攻略的关键词（羁绊名、英雄名）。

    include_fightboard_agg_top：是否把 fightboard 每格 Top-K 备选名也当关键词。
    阵容检索开启有助于在识别摇摆时仍能命中攻略；棋子智库应关闭，否则会召回到「未采纳的备选」
    （战报只展示 best，用户会看到「棋盘上没人却命中该棋子」）。
    """
    keys: List[str] = []
    analysis = summary.get("analysis") or {}
    gt = (analysis.get("group_traits_merged") or {}) if isinstance(analysis, dict) else {}
    tm = gt.get("trait_count_max") if isinstance(gt, dict) else None
    if isinstance(tm, dict):
        keys.extend(str(k) for k in tm.keys())

    rows = summary.get("confirmed_fightboard_results")
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict):
                b = str(r.get("best") or "").strip()
                if b and b not in ("?",):
                    keys.append(b)
                if include_fightboard_agg_top:
                    for at in (r.get("agg_top") or [])[:2]:
                        if isinstance(at, dict):
                            n = str(at.get("name") or "").strip()
                            if n:
                                keys.append(n)
    # 去重保序
    seen: set[str] = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# generate_rag_lineup 里与「实时战术快报」棋盘重复度最高的两段（坐标流水账），运行时删掉以省 token
_RAG_COACH_DROP_LINE_PREFIXES: Tuple[str, ...] = (
    "【按等级构筑参考】",
    "【成型站位与出装】",
)


def _trim_lineup_rag_text_for_coach(raw: str) -> str:
    """方案 A：不动 jsonl，仅去掉攻略里超长坐标段再喂模型。"""
    if not (raw or "").strip():
        return raw
    kept: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if any(s.startswith(p) for p in _RAG_COACH_DROP_LINE_PREFIXES):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def retrieve_lineup_rag(
    summary: Dict[str, Any],
    rag_path: Path,
    top_k: int = 3,
    *,
    min_quality: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    简易检索：关键词在阵容 doc['text'] 中的命中数（无需向量库）。
    min_quality：如 A 表示仅保留评级不劣于 A 的攻略（S 优于 A）。
    返回 (拼好的 RAG 文本块, lineup_id 列表, 终端展示用元数据列表)。
    """
    docs = _load_lineup_docs(rag_path)
    mq = (min_quality or "").strip().upper()[:1]
    if mq and mq in _LINEUP_QUALITY_ORDER:
        max_rank = _lineup_quality_rank(mq)
        docs = [d for d in docs if _lineup_quality_rank(str(d.get("quality") or "")) <= max_rank]
    if not docs:
        return ("（阵容智库 RAG 在给定评级过滤下无可用条目或文件为空。）", [], [])

    kws = _extract_keywords_for_rag(summary)
    if not kws:
        kws = ["阵容"]

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for d in docs:
        text = str(d.get("text") or "")
        score = sum(1 for kw in kws if kw and kw in text)
        # 名称子串也计分
        name = str(d.get("name") or "")
        for kw in kws:
            if len(kw) >= 2 and kw in name:
                score += 2
        scored.append((score, d))

    scored.sort(key=lambda x: (-x[0], x[1].get("lineup_id", "")))
    picked = [x for x in scored if x[0] > 0][:top_k]
    if not picked:
        picked = scored[: min(2, len(scored))]

    blocks: List[str] = []
    ids: List[str] = []
    meta: List[Dict[str, Any]] = []
    for sc, d in picked:
        lid = str(d.get("lineup_id") or "")
        ids.append(lid)
        title = str(d.get("name") or lid)
        body = _trim_lineup_rag_text_for_coach(str(d.get("text") or ""))
        # 控制长度，避免单次请求过大
        if len(body) > 3500:
            body = body[:3500] + "\n…(截断)"
        blocks.append(f"--- 阵容攻略 [{lid}] {title} (匹配分≈{sc}) ---\n{body}")
        meta.append(
            {
                "lineup_id": lid,
                "name": title,
                "quality": str(d.get("quality") or ""),
                "season": str(d.get("season") or ""),
                "match_score": sc,
            }
        )

    return ("\n\n".join(blocks), ids, meta)


def retrieve_core_chess_rag(
    summary: Dict[str, Any],
    rag_path: Path,
    top_k: int = 8,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    棋子智库：rag_core_chess.jsonl，**仅**按战术快报「棋盘与装备」上的棋子名检索
   （summary.confirmed_fightboard_results，与 TCV confirmed 一致；不用羁绊、不用 agg_top、不扫文档全文）。
    每条输出棋子名、定位（主C/主坦/打工仔等）、推荐装备。
    """
    docs = _load_lineup_docs(rag_path)
    core = [
        d
        for d in docs
        if str(d.get("type") or "") == "core_chess"
        or str(d.get("id") or "").startswith("core_chess:")
    ]
    if not core:
        return ("（棋子智库 RAG 库未加载或为空。）", [], [])

    priority = _tcv_board_hero_names_for_rag(summary)
    if not priority:
        return (
            "（棋子智库：无 confirmed_fightboard_results 或无有效棋子名，跳过场上棋子检索。）",
            [],
            [],
        )

    def _score_one(d: Dict[str, Any]) -> int:
        cn = str(d.get("chess_name") or "")
        sc = 0
        for kw in priority:
            if not kw or len(kw) < 2:
                continue
            if cn == kw:
                sc += 10
            elif kw in cn or cn in kw:
                sc += 5
        return sc

    scored: List[Tuple[int, Dict[str, Any]]] = [(_score_one(d), d) for d in core]
    scored.sort(key=lambda x: (-x[0], str((x[1] or {}).get("chess_name") or "")))
    picked = [x for x in scored if x[0] > 0][: max(1, int(top_k))]
    if not picked:
        return (
            "（棋子智库：场上棋子名未在库中命中；可检查 legend/棋子库拼写是否一致。）",
            [],
            [],
        )

    blocks: List[str] = []
    ids: List[str] = []
    meta: List[Dict[str, Any]] = []
    for sc, d in picked:
        cid = str(d.get("id") or d.get("chess_name") or "")
        ids.append(cid)
        cn = str(d.get("chess_name") or "?")
        m = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        role = str(m.get("slot_role") or "未知")
        equips = m.get("recommended_equips") if isinstance(m.get("recommended_equips"), list) else []
        eq_lines = [str(x).strip() for x in equips if str(x).strip()]
        eq_s = "、".join(eq_lines) if eq_lines else "（库中暂无推荐成装）"
        blocks.append(
            f"--- [{cn}] 定位：{role}（匹配分≈{sc}）---\n推荐装备：{eq_s}"
        )
        meta.append(
            {
                "id": cid,
                "chess_name": cn,
                "slot_role": role,
                "match_score": int(sc),
                "recommended_equips": eq_lines,
            }
        )

    return ("\n\n".join(blocks), ids, meta)


def _coach_system_prompt() -> str:
    return """你是《金铲铲之战》S17 顶尖高手“随风听笛”。你正叼着烟在网吧带菜鸡兄弟“哈基星”上分。你不仅有全服顶级的操作直觉，还有一张能把哈基星喷退游的嘴以及奇妙的幽默感。

### 第一步：【你的输入情报汇总】
你会收到包含以下信息的结构化文本，这是你判定的唯一依据：
1. **哈基星提问**：他的语音原话（多轮时首轮为完整块，后续轮次可能只有一句追问，仍视为同一局、同一套快报与 RAG，勿让哈基星复述战报）。
2. **战术快报**：包含 [阶段/等级/经验/金币/胜负/我的血量/羁绊/棋盘装备/装备栏散件]。
3. **阵容智库 (RAG)**：当前的终点站目标与运营节奏。
4. **棋子智库 (RAG)**：谁是核心（主C/主坦/打工仔），谁该拿什么装备。

### 第二步：【随风听笛的脑内回路】（思考逻辑，不准输出）
1. **抓大腿**：对比快报与棋子智库，看场上谁是真神（2星高费 > 1星高费 > 2星低费）。
2. **找缺口**：主C满三件套没？主坦顶得住吗？站位是不是在送？
   - **搜牌保底逻辑**：如果 RAG 阵容已经成型或不知道搜谁，可下令搜【通用5费主C/主坦】：包括奥巴马/稻草人/龙女/千珏/炸弹人/塞拉斯）。
3. **断节奏**：5阶段以后血低于40还不D牌？9级50块等死？看[阶段/金币]断定该搜还是该攒。
4. **定方向**：对比阵容智库，看哈基星是在玩版本答案还是在玩勾八。

### 第三步：【你的输出协议】
1. **意图第一**：
    - 问具体的（合什么、D不D、找谁）：**直接给死命令**，不要扯其他方面的建议。
    - 问模糊的（救命、玩什么）：**抓当前最致命的短板（如：该D不D、主C没装、站位逆天）暴力纠错**。
2. **反教条主义**：场上已成型的 4费 / 5费 2星主C或主坦是绝对资产，**严禁** 为了凑 RAG 阵容让他卖掉大腿或扒装备。
3. **装备铁律：**装备只能给【战术快报】中存在的棋子，且优先给主C / 主坦，其次是打工仔，最后才是挂件；如果棋子已经满装备了，严禁再建议给装备。
4. **信息脱敏**：严禁复述 JSON 参数，严禁展示思考过程，严禁 1.2.3. 或 Markdown。

### 第四步：【性格与语言风格】
- **拒绝废话**：不准说建议、分析、根据、可能。只有【1句话】，必须是纯口语。
- **毒舌幽默**：挖苦哈基星是日常，但他操作偶尔对了一次，也要夸奖一下他。
- **黑话简称**：尽量不说棋子或装备的官方名称，改为非正式简称，例如奥巴马、反甲等。
- **真人气息：**若哈基星问你游戏之外的问题，按照你的兄弟人设正常回答即可，不一定非要跟金铲铲有关。"""


def _coach_first_user_message(
    tactical_brief: str,
    rag_lineup_block: str,
    rag_chess_block: str,
    user_question: str,
) -> str:
    return (
        "【哈基星问题】\n"
        f"{user_question.strip()}\n\n"
        "【战术快报】\n"
        f"{tactical_brief.strip()}\n\n"
        "【阵容智库 (RAG)】\n"
        f"{rag_lineup_block.strip()}\n\n"
        "【棋子智库 (RAG)】\n"
        f"{rag_chess_block.strip()}"
    )


def _coach_followup_user_message(user_question: str) -> str:
    return "【哈基星追问】\n" f"{user_question.strip()}"


def coach_chat_complete_turn(
    chat_hist: List[Dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.35,
) -> str:
    """
    多轮中的一步：chat_hist 须为 OpenAI 式 messages（不含 system），且以 user 开头、user/assistant 严格交替。
    本函数发送「system + 整段 chat_hist」，返回本步 assistant 文本。
    """
    if not chat_hist or chat_hist[0].get("role") != "user":
        raise ValueError("chat_hist 必须以 user 开头")
    backend = _resolve_chat_backend()
    _, _, default_or_model = _openrouter_env()
    sys_p = _coach_system_prompt()
    if backend == "google":
        gm = (model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")).strip()
        print(
            f"[gemini_v1] LLM 后端: google  模型: {gm}  上下文: {len(chat_hist)} 条 user/model 消息"
        )
        return _google_gemini_chat(
            system_prompt=sys_p,
            chat_hist=list(chat_hist),
            model=gm,
            temperature=temperature,
        )
    om = model or default_or_model
    print(
        f"[gemini_v1] LLM 后端: openrouter  模型: {om}  上下文: {len(chat_hist)} 条 user/assistant 消息"
    )
    return _openrouter_chat_completion(
        messages=[{"role": "system", "content": sys_p}, *chat_hist],
        model=om,
        temperature=temperature,
    )


def call_gemini_coach(
    tactical_brief: str,
    rag_lineup_block: str,
    rag_chess_block: str,
    user_question: str,
    *,
    model: str | None = None,
) -> str:
    """
    单轮便捷封装（兼容脚本与 benchmark）：等同只发一条首轮 user。
    """
    u = _coach_first_user_message(
        tactical_brief, rag_lineup_block, rag_chess_block, user_question
    )
    return coach_chat_complete_turn([{"role": "user", "content": u}], model=model)


def _dir_has_screenshot(d: Path) -> bool:
    if not d.is_dir():
        return False
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return any(p.suffix.lower() in exts for p in d.iterdir() if p.is_file())


def _find_first_summary_json(out_dir: Path) -> Path:
    """pipeline 输出目录下 *_summary.json，取主图（通常含 -a）优先。"""
    if not out_dir.is_dir():
        raise FileNotFoundError(f"输出目录不存在: {out_dir}")
    candidates = sorted(out_dir.glob("*_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"目录内无 *_summary.json: {out_dir}")
    for p in candidates:
        if "-a" in p.stem:
            return p
    return candidates[0]


def run_pipeline(
    img_dir: Path, out_dir: Path, *, quiet: bool = False
) -> Optional[Tuple[str, str]]:
    """
    调用根目录 pipeline.py。quiet=True 时不向终端转发子进程日志，并返回 (stdout, stderr) 供解析耗时；
    直接运行 `python pipeline.py` 时不受影响。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "pipeline.py"),
        "--img-dir",
        str(img_dir.resolve()),
        "--out",
        str(out_dir.resolve()),
    ]
    if quiet:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if r.returncode != 0:
            sys.stderr.write("\n[pipeline 失败] 以下为子进程 stderr 尾部（完整日志未在终端打印）\n")
            sys.stderr.write((r.stderr or "")[-6000:])
            sys.stderr.write("\n")
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
        return (r.stdout or "", r.stderr or "")
    print("[gemini_v1] 运行 pipeline:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return None


def _pipeline_timing_report_lines(merged_log: str, wall_s: float) -> List[str]:
    """从 pipeline 捕获日志中摘录耗时，便于判断瓶颈（非启动问题时多为 player_onnx / 并行墙钟）。"""
    lines: List[str] = [
        f"子进程墙钟（python pipeline.py 全程，含首次加载 torch / ONNX）: {wall_s:.2f}s",
    ]
    if not merged_log.strip():
        lines.append("  （无捕获日志）")
        return lines

    for m in re.finditer(
        r"\[Pipeline\]\s*本图合计耗时:\s*([\d.]+)s\s*\(([^)]+)\)",
        merged_log,
    ):
        lines.append(f"  · {m.group(2)} 本图合计（pipeline 报告）: {m.group(1)}s")

    m = re.search(
        r"并行段\(fightboard\+equip_column\+player wall\):\s*([\d.]+)s",
        merged_log,
    )
    if m:
        lines.append(f"  · 主图并行段墙钟 ≈ max(三路): {m.group(1)}s")

    m = re.search(r"fightboard_total:\s*([\d.]+)s", merged_log)
    if m:
        lines.append(f"  · fightboard（线程内）: {m.group(1)}s")
    m = re.search(r"equip_column_total:\s*([\d.]+)s", merged_log)
    if m:
        lines.append(f"  · equip_column（线程内）: {m.group(1)}s")
    m = re.search(r"player_onnx 全 ROI:\s*([\d.]+)s", merged_log)
    if m:
        lines.append(f"  · player_onnx 全 ROI（线程内，常占并行瓶颈）: {m.group(1)}s")

    m = re.search(r"player_onnx 羁绊栏:\s*([\d.]+)s", merged_log)
    if m:
        lines.append(f"  · 辅图 player_onnx 羁绊栏: {m.group(1)}s")

    m = re.search(r"chess_recog\(MobileNet\):\s*([\d.]+)s", merged_log)
    if m:
        lines.append(f"  · fightboard 棋子 MobileNet 小计: {m.group(1)}s")

    lines.append(
        "  说明: 并行墙钟通常接近 max(fightboard, equip_column, player)；"
        "总墙钟还含初始化、辅图、写 JSON/PNG。首次冷启动加载模型会多几秒。"
    )
    return lines


def _pipeline_quiet_progress_until(
    done: threading.Event,
    label: str,
    io_lock: threading.Lock,
) -> None:
    """
    stderr 单行 \\r 转圈（不刷屏）。input() 期间主线程持 io_lock，本线程抢不到锁则跳过本拍，避免打断输入。
    """
    spin = itertools.cycle("|/-\\")
    t0 = time.perf_counter()
    clear_w = 88

    def _try_spin() -> None:
        if not io_lock.acquire(blocking=False):
            return
        try:
            elapsed = time.perf_counter() - t0
            ch = next(spin)
            sys.stderr.write(
                f"\r  [gemini_v1] {label} {elapsed:5.1f}s {ch} "
            )
            sys.stderr.flush()
        finally:
            io_lock.release()

    while not done.wait(0.09):
        _try_spin()

    if io_lock.acquire(blocking=False):
        try:
            sys.stderr.write("\r" + " " * clear_w + "\r")
            sys.stderr.write(f"[gemini_v1] {label} 完成。\n")
            sys.stderr.flush()
        finally:
            io_lock.release()


def build_coach_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Pipeline JSON → 战术快报 + 双智库 RAG → 随风听笛",
        epilog="默认优先使用 runs/battle_pipeline_v3_out 下已有 *_summary.json（跳过 pipeline）；\n"
        "无缓存时再跑 pipeline（stderr 转圈；输入问题时暂停刷新）。\n"
        "快报预览后直接请求模型。--force-pipeline 可强制重新识别。--no-rag 可关闭阵容+棋子 RAG。--summary-json 指定单文件调试。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--img-dir",
        type=Path,
        help=f"含主图 -a 与辅图 -b 的目录；省略时使用「{DEFAULT_IMG_DIR.name}」（须含截图）",
    )
    ap.add_argument(
        "--pipeline-out",
        type=Path,
        default=DEFAULT_PIPELINE_OUT,
        help=f"pipeline 输出目录（默认 {PIPELINE_CACHE_DIR.relative_to(REPO_ROOT)}；无缓存时写入此处）",
    )
    ap.add_argument(
        "--force-pipeline",
        action="store_true",
        help="忽略 battle_pipeline_v3_out 中已有 *_summary.json，强制重新跑 pipeline",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        help="已生成的 *_summary.json，若指定则跳过 pipeline",
    )
    ap.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="用户问题；省略时在终端交互输入（仍可用管道传入 stdin）",
    )
    ap.add_argument(
        "--rag-lineup",
        type=Path,
        default=DEFAULT_RAG_LINEUP,
        help="rag_lineup_lineup.jsonl 路径",
    )
    ap.add_argument("--rag-top-k", type=int, default=3, help="阵容智库检索条数")
    ap.add_argument(
        "--rag-core-chess",
        type=Path,
        default=DEFAULT_RAG_CORE_CHESS,
        help="棋子智库 rag_core_chess.jsonl 路径",
    )
    ap.add_argument("--rag-chess-top-k", type=int, default=8, help="棋子智库检索条数上限")
    ap.add_argument(
        "--rag-min-quality",
        type=str,
        default="A",
        help="阵容智库：仅保留该评级及以上（S 最优；默认 A 即只检索 S 与 A）。传 - 或 all 表示不按评级过滤",
    )
    ap.add_argument(
        "--no-rag",
        action="store_true",
        help="不检索阵容/棋子 jsonl，仅用战术快报 + 问题",
    )
    ap.add_argument(
        "--pipeline-verbose",
        action="store_true",
        help="将 pipeline 子进程的 stdout/stderr 原样打到终端（关闭静默与进度条；调试用）",
    )
    return ap


def build_coach_bundle(
    args: argparse.Namespace,
    summary_path: Path,
) -> Dict[str, Any]:
    """读 summary、构建战术快报与双智库 RAG；供 gemini_v2 与录音并行。"""
    t_prep0 = time.perf_counter()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    brief = build_tactical_brief(summary)
    _rq = (args.rag_min_quality or "A").strip()
    if _rq.lower() in ("-", "all", "none", "off", "*"):
        min_q: Optional[str] = None
    else:
        mq = _rq.upper()[:1]
        min_q = mq if mq in _LINEUP_QUALITY_ORDER else "A"
    if args.no_rag:
        rag_block = "（本回合未注入阵容智库。）"
        rag_meta: List[Dict[str, Any]] = []
        chess_block = "（本回合未注入棋子智库。）"
        chess_meta: List[Dict[str, Any]] = []
    else:
        rag_block, _rag_ids, rag_meta = retrieve_lineup_rag(
            summary,
            args.rag_lineup,
            top_k=max(1, int(args.rag_top_k)),
            min_quality=min_q,
        )
        chess_block, _chess_ids, chess_meta = retrieve_core_chess_rag(
            summary,
            args.rag_core_chess.resolve(),
            top_k=max(1, int(args.rag_chess_top_k)),
        )
    t_prep1 = time.perf_counter()
    return {
        "summary": summary,
        "brief": brief,
        "rag_block": rag_block,
        "chess_block": chess_block,
        "rag_meta": rag_meta,
        "chess_meta": chess_meta,
        "t_prep0": t_prep0,
        "t_prep1": t_prep1,
    }


def print_coach_bundle_preview(args: argparse.Namespace, bundle: Dict[str, Any]) -> None:
    """终端打印三块预览（与送入模型一致）。"""
    brief = str(bundle.get("brief") or "")
    rag_meta = bundle.get("rag_meta") or []
    chess_meta = bundle.get("chess_meta") or []
    print("\n" + "=" * 60)
    print("【1. 战术快报】（预览）")
    print("=" * 60)
    print(brief[:2000] + ("…" if len(brief) > 2000 else ""))
    print("\n" + "=" * 60)
    print("【2. 阵容智库 RAG】命中（与送入模型一致）")
    print("=" * 60)
    if args.no_rag:
        print("（已跳过：--no-rag）")
    elif not rag_meta:
        print("（无命中或库为空）")
    else:
        for m in rag_meta:
            if not isinstance(m, dict):
                continue
            print(
                f"lineup_id={m.get('lineup_id', '')} | "
                f"评级={m.get('quality', '')} | "
                f"赛季={m.get('season', '')} | "
                f"匹配分≈{m.get('match_score', 0)} | "
                f"阵容名：{m.get('name', '')}"
            )
    print("\n" + "=" * 60)
    print("【3. 棋子智库 RAG】命中（与送入模型一致）")
    print("=" * 60)
    if args.no_rag:
        print("（已跳过：--no-rag）")
    elif not chess_meta:
        print("（无命中或库为空）")
    else:
        for m in chess_meta:
            if not isinstance(m, dict):
                continue
            eqs = m.get("recommended_equips") or []
            eq_show = (
                "、".join(str(x) for x in eqs[:12])
                if isinstance(eqs, list)
                else ""
            )
            print(
                f"{m.get('chess_name', '')} | 定位={m.get('slot_role', '')} | "
                f"匹配分≈{m.get('match_score', 0)} | 推荐装：{eq_show or '（无）'}"
            )
    print("=" * 60)


def run_coach_after_summary(
    args: argparse.Namespace,
    summary_path: Path,
    question: str,
    io_lock: threading.Lock,
    *,
    follow_up_reader: Optional[Callable[[], str]] = None,
    log_prefix: str = "gemini_v1",
    coach_bundle: Optional[Dict[str, Any]] = None,
    skip_bundle_preview_print: bool = False,
) -> None:
    """从已定位的 *_summary.json 起：RAG、预览、多轮教练对话（可注入追问输入函数，供 gemini_v2 语音等）。"""
    if coach_bundle is None:
        coach_bundle = build_coach_bundle(args, summary_path)
        if not skip_bundle_preview_print:
            print_coach_bundle_preview(args, coach_bundle)
    elif not skip_bundle_preview_print:
        print_coach_bundle_preview(args, coach_bundle)

    brief = str(coach_bundle.get("brief") or "")
    rag_block = str(coach_bundle.get("rag_block") or "")
    chess_block = str(coach_bundle.get("chess_block") or "")
    t_prep0 = float(coach_bundle.get("t_prep0") or 0.0)
    t_prep1 = float(coach_bundle.get("t_prep1") or 0.0)

    chat_hist: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": _coach_first_user_message(
                brief, rag_block, chess_block, question
            ),
        }
    ]
    ll_total = 0.0
    turn = 0
    answer = ""

    def _default_follow_up() -> str:
        io_lock.acquire()
        try:
            return input("哈基星：").strip()
        finally:
            io_lock.release()

    reader = follow_up_reader if follow_up_reader is not None else _default_follow_up

    while True:
        turn += 1
        print(f"\n正在请求 LLM（第 {turn} 轮）…\n")
        t_ll0 = time.perf_counter()
        answer = coach_chat_complete_turn(chat_hist)
        t_ll1 = time.perf_counter()
        round_sec = t_ll1 - t_ll0
        ll_total += round_sec
        print(
            f"[{log_prefix}] 第 {turn} 轮模型响应耗时: {round_sec:.2f}s",
            flush=True,
        )

        print("\n【随风听笛说】\n")
        print(answer)
        print()
        if not sys.stdin.isatty():
            break
        print(
            "—— 首轮已含战术快报与双智库 RAG，追问不必重跑检索；"
            "模型依赖对话历史继续推理（上下文过长被截断时需新开一局或重喂快报）。"
        )
        print("—— 继续提问直接输入；空行或 q / quit / exit 结束。\n")
        try:
            nxt = reader()
        except EOFError:
            break
        if not nxt or nxt.lower() in ("q", "quit", "exit", "bye", "再见"):
            break
        chat_hist.append({"role": "assistant", "content": answer})
        chat_hist.append(
            {"role": "user", "content": _coach_followup_user_message(nxt)}
        )

    print(f"[{log_prefix}] 本回合步骤耗时")
    if args.no_rag:
        print(f"  读 JSON + 构建快报（无 RAG）: {t_prep1 - t_prep0:.2f}s")
    else:
        print(f"  读 JSON + 构建快报 + 双智库 RAG: {t_prep1 - t_prep0:.2f}s")
    print(f"  LLM 合计（{turn} 轮）: {ll_total:.2f}s")
    print()


def main() -> None:
    ap = build_coach_argparser()
    args = ap.parse_args()
    io_lock = threading.Lock()

    question = (args.question or "").strip()
    summary_path: Optional[Path] = None

    if args.summary_json:
        summary_path = args.summary_json.resolve()
        if not summary_path.is_file():
            raise SystemExit(f"找不到文件: {summary_path}")
        if not question:
            if sys.stdin.isatty():
                question = input("你有什么要问的：").strip()
            else:
                question = sys.stdin.read().strip()
        if not question:
            raise SystemExit("未提供问题：请使用 --question / -q，或在「你有什么要问的」处输入。")
    else:
        img_dir = args.img_dir.resolve() if args.img_dir else DEFAULT_IMG_DIR
        if not img_dir.is_dir():
            raise SystemExit(f"截图目录不存在: {img_dir}")
        if not _dir_has_screenshot(img_dir):
            raise SystemExit(
                f"目录内无截图: {img_dir}\n"
                f"请放入主图 -a / 辅图 -b，或使用 --summary-json 指定已有 *_summary.json"
            )

        cache_dir = PIPELINE_CACHE_DIR.resolve()
        use_cache = (
            not bool(args.force_pipeline)
            and cache_dir.is_dir()
            and any(cache_dir.glob("*_summary.json"))
        )

        if use_cache:
            summary_path = _find_first_summary_json(cache_dir)
            print(f"[gemini_v1] 使用缓存: {cache_dir}（跳过 pipeline）", flush=True)
            if not question:
                if sys.stdin.isatty():
                    print(
                        "-" * 60 + "\n【提示】已使用本地缓存的识别结果，跳过 pipeline。\n" + "-" * 60,
                        flush=True,
                    )
                    question = input("你有什么要问的：").strip()
                else:
                    question = sys.stdin.read().strip()
            if not question:
                raise SystemExit("未提供问题：请使用 --question / -q，或在「你有什么要问的」处输入。")
        else:
            out_dir = args.pipeline_out.resolve()
            pipe_err: List[Optional[Exception]] = [None]
            pipe_data: Dict[str, Any] = {"cap": None, "wall": 0.0}
            done = threading.Event()
            quiet = not args.pipeline_verbose

            def _pipeline_worker() -> None:
                t0 = time.perf_counter()
                try:
                    pipe_data["cap"] = run_pipeline(img_dir, out_dir, quiet=quiet)
                except Exception as e:
                    pipe_err[0] = e
                finally:
                    pipe_data["wall"] = time.perf_counter() - t0
                    done.set()

            th = threading.Thread(target=_pipeline_worker, name="pipeline", daemon=True)
            th.start()

            st: Optional[threading.Thread] = None
            if quiet:
                st = threading.Thread(
                    target=_pipeline_quiet_progress_until,
                    args=(done, "对局 JSON 分析中", io_lock),
                    name="pipeline_progress",
                    daemon=True,
                )
                st.start()

            if sys.stdout.isatty():
                print(
                    "-" * 60
                    + "\n【提示】对局识别已在后台运行；下方可输入问题。\n"
                    "        输入过程中不会刷新进度行，避免打断打字；回车提交后若未识别完会继续显示进度。\n"
                    + "-" * 60,
                    flush=True,
                )

            if not question:
                if sys.stdin.isatty():
                    io_lock.acquire()
                    try:
                        question = input("你有什么要问的：").strip()
                    finally:
                        io_lock.release()
                else:
                    question = sys.stdin.read().strip()
            if not question:
                raise SystemExit("未提供问题：请使用 --question / -q，或在「你有什么要问的」处输入。")

            th.join()
            if st is not None:
                st.join(timeout=5.0)

            if pipe_err[0] is not None:
                raise pipe_err[0]
            if quiet and pipe_data.get("cap") is not None:
                cap = pipe_data["cap"]
                merged = (cap[0] or "") + "\n" + (cap[1] or "")
                print("\n[gemini_v1] 对局分析耗时（摘自子进程捕获日志）")
                for ln in _pipeline_timing_report_lines(merged, float(pipe_data["wall"])):
                    print(ln)
            summary_path = _find_first_summary_json(out_dir)

    assert summary_path is not None
    run_coach_after_summary(args, summary_path, question, io_lock)


if __name__ == "__main__":
    main()
