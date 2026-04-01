# -*- coding: utf-8 -*-
"""
Pipeline 对局 JSON → 战术快报（含检索命中的阵容要点与目标棋子详情）→ 随风听笛回答。

用法示例：
  python gemini_v1.py
  # 默认：若 runs/battle_pipeline_v3_out 已有 *_summary.json 则读缓存；否则静默跑 pipeline。
  python gemini_v1.py --summary-json runs/.../01-a_summary.json  # 跳过 pipeline，仅调试
  python gemini_v1.py --img-dir "对局截图" -q "我这把该先合什么？"
  python gemini_v1.py --no-rag   # 关闭本地 RAG（阵容检索+棋子智库），只喂战术快报 + 问题

  交互终端下：随风听笛答完后可继续输入追问（首轮已含完整快报，追问不重跑检索）；空行或 q / quit / exit 结束。
  管道或非 TTY  stdin：只跑一轮（与原先一致）。

说明：
  - 当前 LLM：默认 OpenRouter，模型 id 见 .env 的 OPENROUTER_TEXT_MODEL（如 google/gemini-2.5-flash，即 Gemini Flash 系）。
  - GEMINI_BACKEND=google 可走 Google 直连；模型见 GEMINI_MODEL。
  - 棋盘装备：审计见 scripts/extra/equip_audit.py，推荐见 scripts/extra/equip_recom.py。
  - 棋子智库结构化块：scripts/chess_recom.py（Top1 缺口、挂件替换等）。
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from scripts.chess_recom import (
    board_line_hero_display_name,
    chess_cost_from_core_or_legend,
    hero_name_matches_board,
    load_core_chess_list,
    load_legend_chess_name_map,
    lookup_core_chess_row,
    slot_role_and_cost_for_name,
    retrieve_core_chess_rag,
)

REPO_ROOT = Path(__file__).resolve().parent

# rag_legend_equip.jsonl：基础装集合 + 合成配方（散件计数 → 可合成成装）
_LEGEND_EQUIP_TABLES: Optional[Tuple[Set[str], List[Tuple[str, str, str]]]] = None


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

from scripts.extra.equip_audit import (  # noqa: E402
    equip_audit_excluded,
    equip_primary_stat_kind,
    format_equipment_audit_terminal_lines,
    load_legend_equip_full_map,
    main_c_ap_ad_flags,
    role_tags_expect_damage,
    traits_to_role_tags,
)
from scripts.extra.equip_recom import pick_next_finished_recommendations  # noqa: E402

# 与 pipeline 默认输出一致；若目录内已有 *_summary.json，gemini_v1 优先读缓存、不重复跑识别
PIPELINE_CACHE_DIR = REPO_ROOT / "runs" / "battle_pipeline_v3_out"
DEFAULT_PIPELINE_OUT = PIPELINE_CACHE_DIR
DEFAULT_IMG_DIR = REPO_ROOT / "对局截图"
DEFAULT_RAG_LINEUP = REPO_ROOT / "data" / "rag_lineup_lineup_v1.jsonl"
DEFAULT_RAG_CORE_CHESS = REPO_ROOT / "data" / "rag_core_chess.jsonl"
DEFAULT_RAG_LEGEND_CHESS = REPO_ROOT / "data" / "rag_legend_chess.jsonl"

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
    max_tokens: Optional[int] = None,
    stream_output: bool = False,
    on_stream_chunk: Optional[Callable[[str], None]] = None,
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
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None and int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)
    if stream_output:
        payload["stream"] = True

    r = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout_s,
        stream=bool(stream_output),
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
    if stream_output:
        # 流式场景固定按 UTF-8 解码，避免 Windows 终端出现乱码。
        r.encoding = "utf-8"
        parts: List[str] = []
        for raw in r.iter_lines(decode_unicode=False):
            if not raw:
                continue
            if isinstance(raw, (bytes, bytearray)):
                line = raw.decode("utf-8", errors="replace").strip()
            else:
                line = str(raw).strip()
            if not line.startswith("data:"):
                continue
            data_s = line[5:].strip()
            if not data_s or data_s == "[DONE]":
                continue
            try:
                evt = json.loads(data_s)
            except Exception:
                continue
            choices = evt.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            content = delta.get("content")
            chunk = ""
            if isinstance(content, str):
                chunk = content
            elif isinstance(content, list):
                chunk = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            if not chunk:
                continue
            parts.append(chunk)
            if callable(on_stream_chunk):
                on_stream_chunk(chunk)
        return "".join(parts).strip()

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


def _star_level_from_star_obj(st: Any) -> Optional[int]:
    """从 fightboard 的 star 字典解析 1～3 星；pred / pred_raw、int/float/str 均兼容。"""
    if not isinstance(st, dict):
        return None
    for key in ("pred", "pred_raw"):
        pr = st.get(key)
        if pr is None:
            continue
        try:
            n = int(float(pr))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= 3:
            return n
    return None


def _fightboard_star_index_from_results(rs: Any) -> Dict[int, Dict[str, Any]]:
    """fightboard results 列表 → bar_index → star 字典（仅含 isinstance(star, dict) 的项）。"""
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(rs, list):
        return out
    for r in rs:
        if not isinstance(r, dict):
            continue
        try:
            bi = int(r.get("bar_index") or -1)
        except (TypeError, ValueError):
            continue
        st = r.get("star")
        if bi >= 0 and isinstance(st, dict):
            out[bi] = st
    return out


def _load_fightboard_sidecar_star_index(
    summary: Dict[str, Any],
    summary_json_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    读取同主图 stem 的 *_fightboard_summary.json（单独跑 fightboard_mobilenet 时的输出），
    用于 pipeline 缓存的 *_summary.json 未写入 star 时的星级回退。

    查找顺序：与 summary_json 同目录的 {stem}_fightboard_summary.json → runs/fightboard_info_v2/。
    """
    fn = str(summary.get("file") or "").strip()
    if not fn:
        return {}
    stem = Path(fn).stem
    name = f"{stem}_fightboard_summary.json"
    candidates: List[Path] = []
    if summary_json_path is not None:
        candidates.append(summary_json_path.resolve().parent / name)
    candidates.append(REPO_ROOT / "runs" / "fightboard_info_v2" / name)
    for p in candidates:
        if not p.is_file():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fb = (raw.get("modules") or {}).get("fightboard") or raw
        return _fightboard_star_index_from_results(fb.get("results"))
    return {}


def _fightboard_star_by_bar(
    summary: Dict[str, Any],
    *,
    summary_json_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    """modules.fightboard.results 按 bar_index 建 star 索引；无有效 pred 时尝试同 stem 的 fightboard 侧写 JSON。"""
    fb = (summary.get("modules") or {}).get("fightboard") or {}
    out = _fightboard_star_index_from_results(fb.get("results"))
    side = _load_fightboard_sidecar_star_index(summary, summary_json_path=summary_json_path)
    for bi, st in side.items():
        cur = out.get(bi)
        if _star_level_from_star_obj(cur) is not None:
            continue
        if _star_level_from_star_obj(st) is not None:
            out[bi] = st
    return out


def _star_prefix_board(r: Dict[str, Any], star_by_bar: Dict[int, Dict[str, Any]]) -> str:
    """棋子星级前缀：优先本行 star，否则同 bar_index 的 fightboard.results[].star。"""
    st: Any = r.get("star")
    if not isinstance(st, dict):
        try:
            bi = int(r.get("bar_index") or -1)
        except (TypeError, ValueError):
            bi = -1
        if bi >= 0:
            st = star_by_bar.get(bi)
    n = _star_level_from_star_obj(st)
    if n is None:
        return ""
    return f"{n}星 "


def _star_display_segment(r: Dict[str, Any], star_by_bar: Dict[int, Dict[str, Any]]) -> str:
    """单行战报用星级片段：如 2星；无识别为 ?星。"""
    st: Any = r.get("star")
    if not isinstance(st, dict):
        try:
            bi = int(r.get("bar_index") or -1)
        except (TypeError, ValueError):
            bi = -1
        if bi >= 0:
            st = star_by_bar.get(bi)
    n = _star_level_from_star_obj(st)
    if n is not None:
        return f"{n}星"
    return "?星"


def _all_board_equipped_names(rows: List[Dict[str, Any]], equip_by_bar: Any) -> List[str]:
    """棋盘上所有格子的已携带成装名（顺序拼接，可重复）。"""
    out: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            bi = int(r.get("bar_index") or 0)
        except (TypeError, ValueError):
            bi = 0
        raw_eq = equip_by_bar.get(str(bi)) if isinstance(equip_by_bar, dict) else None
        if raw_eq is None and isinstance(equip_by_bar, dict):
            raw_eq = equip_by_bar.get(bi)
        if isinstance(raw_eq, list):
            for e in raw_eq:
                if isinstance(e, dict):
                    en = str(e.get("name") or "").strip()
                    if en:
                        out.append(en)
    return out


def _board_slot_role_sort_key(slot_tag: str) -> int:
    """战报行顺序：主坦 → 主C → 打工仔/混合 → 挂件；其余靠后。"""
    return {
        "主坦": 0,
        "主C": 1,
        "打工仔": 2,
        "混合": 2,
        "挂件": 3,
    }.get((slot_tag or "").strip(), 99)


REPORT_SEP = "————————————————————"
REPORT_WRAP_WIDTH = 64
# 战报：节标题 emoji 与文字 2 空格（缓解终端里符号与汉字挤在一起）；竖线分隔两侧各 1 空格
REPORT_EMOJI_GAP = "  "
REPORT_PIPE_SEP = " | "
# 棋盘槽位行：不用 ✅/对勾类，改用几何符号；审计行见 equip_audit（※ 正常等）
BOARD_EMOJI_TEXT_GAP = "  "
BOARD_SLOT_ICON_FULL = "◆"
BOARD_SLOT_ICON_EMPTY = "◇"


def _wrap_segments_line(parts: List[str], sep: str = REPORT_PIPE_SEP, width: int = REPORT_WRAP_WIDTH) -> str:
    """将若干片段用 sep 拼接，超长则换行续写。"""
    if not parts:
        return ""
    lines: List[str] = []
    cur = parts[0]
    for p in parts[1:]:
        trial = f"{cur}{sep}{p}"
        if len(trial) > width:
            lines.append(cur)
            cur = p
        else:
            cur = trial
    lines.append(cur)
    return "\n".join(lines)


def _sort_traits_for_report(trait_count_max: Dict[str, int]) -> List[Tuple[str, int]]:
    """1～2 人口羁绊优先（先 1 后 2），其余按激活数降序。"""
    if not trait_count_max:
        return []
    items = [(str(t), int(c)) for t, c in trait_count_max.items()]
    low = [(t, c) for t, c in items if c <= 2]
    high = [(t, c) for t, c in items if c > 2]
    low.sort(key=lambda x: (x[1], x[0]))
    high.sort(key=lambda x: (-x[1], x[0]))
    return low + high


def _format_traits_block(tm: Optional[Dict[str, int]]) -> str:
    if not tm:
        return "(无)"
    ordered = _sort_traits_for_report(tm)
    parts = [f"{c}{t}" for t, c in ordered]
    return _wrap_segments_line(parts)


def _star_int_for_sort(r: Dict[str, Any], star_by_bar: Dict[int, Dict[str, Any]]) -> int:
    st: Any = r.get("star")
    if not isinstance(st, dict):
        try:
            bi = int(r.get("bar_index") or -1)
        except (TypeError, ValueError):
            bi = -1
        if bi >= 0:
            st = star_by_bar.get(bi)
    n = _star_level_from_star_obj(st) if isinstance(st, dict) else None
    return int(n) if n is not None else -1


def _format_board_detail_lines(
    summary: Dict[str, Any],
    core_list: List[Dict[str, Any]],
    *,
    summary_json_path: Optional[Path] = None,
    legend_chess_path: Optional[Path] = None,
) -> List[str]:
    """战术快报棋盘：竖线分列；挂件不输出推荐；第三行为装备审计（正常|严重|警告）；推荐件数=空位。

    输出顺序按定位：主坦、主C、打工仔、混合、挂件。"""
    c, w = _build_board_emoji_sections(
        summary,
        core_list,
        summary_json_path=summary_json_path,
        legend_chess_path=legend_chess_path,
    )
    out: List[str] = []
    for b in c + w:
        out.extend(b.splitlines())
    return out


def _role_line_label(slot_tag: str) -> str:
    if slot_tag in ("打工仔", "混合"):
        return "打工"
    return slot_tag or "?"


def _build_board_emoji_sections(
    summary: Dict[str, Any],
    core_list: List[Dict[str, Any]],
    *,
    summary_json_path: Optional[Path] = None,
    legend_chess_path: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """主力核心、打工挂件；星级降序、费用降序。"""
    rows = summary.get("confirmed_fightboard_results")
    if not isinstance(rows, list):
        return [], []
    fight = (summary.get("modules") or {}).get("fightboard") or {}
    equip_by_bar = fight.get("equip_by_bar") or {}
    player_on = (summary.get("modules") or {}).get("player") or {}
    phase_raw = _field_parsed(player_on, "phase").replace("总", "")
    team_all_equips = _all_board_equipped_names(rows, equip_by_bar)
    star_by_bar = _fightboard_star_by_bar(summary, summary_json_path=summary_json_path)
    leg_path = legend_chess_path or DEFAULT_RAG_LEGEND_CHESS
    legend_map = load_legend_chess_name_map(leg_path)
    equip_map = load_legend_equip_full_map()
    raw: List[Dict[str, Any]] = []
    for row_idx, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        disp = board_line_hero_display_name(r)
        lookup_key = disp.rstrip("?").rstrip("？").strip()
        row_doc = lookup_core_chess_row(lookup_key, core_list)
        leg_row = legend_map.get(lookup_key) if row_doc is None else None
        leg_for_audit = legend_map.get(lookup_key) if lookup_key else None
        legend_text_audit = str((leg_for_audit or {}).get("text") or "") if isinstance(leg_for_audit, dict) else ""

        cost_v: Any = None
        meta: Dict[str, Any] = {}
        slot_tag = ""
        role_tags_list: List[str] = []
        if isinstance(row_doc, dict):
            cost_v = row_doc.get("cost")
            meta = row_doc.get("meta") if isinstance(row_doc.get("meta"), dict) else {}
            slot_tag = str(meta.get("slot_role") or "").strip()
            rt = meta.get("role_tags")
            if isinstance(rt, list):
                role_tags_list = [str(x).strip() for x in rt if str(x).strip()]
        elif isinstance(leg_row, dict):
            cost_v = leg_row.get("cost")
            traits = leg_row.get("traits")
            role_tags_list = traits_to_role_tags(traits if isinstance(traits, list) else [])
            meta = {"slot_role": "挂件", "recommended_equips": []}
            slot_tag = "挂件"

        try:
            cost_n = int(cost_v) if cost_v is not None else -1
        except (TypeError, ValueError):
            cost_n = -1
        cost_s = f"{cost_n}费" if cost_n >= 0 else "?费"

        rec_list = meta.get("recommended_equips") if isinstance(meta.get("recommended_equips"), list) else []

        pos = r.get("position") or {}
        cell_row: Optional[int] = None
        loc = "?"
        if isinstance(pos, dict):
            cr, cc = pos.get("cell_row"), pos.get("cell_col")
            if cr is not None and cc is not None:
                cell_row = int(cr)
                loc = f"(R{cell_row},C{int(cc)})"
            else:
                lab = str(pos.get("label") or "").strip()
                loc = lab if lab else "?"

        bi = int(r.get("bar_index") or 0)
        eqs: List[str] = []
        raw_eq = equip_by_bar.get(str(bi))
        if raw_eq is None and isinstance(equip_by_bar, dict):
            raw_eq = equip_by_bar.get(bi)
        if isinstance(raw_eq, list):
            for e in raw_eq:
                if isinstance(e, dict):
                    en = str(e.get("name") or "").strip()
                    if en:
                        eqs.append(en)
        disp_show = _hero_name_with_optional_short(
            disp, row_doc if isinstance(row_doc, dict) else None
        )
        carry_label = (
            "、".join(_equip_name_with_optional_short(e, equip_map) for e in eqs) if eqs else "空"
        )
        n_eq = len(eqs)
        star_seg = _star_display_segment(r, star_by_bar)
        star_n = _star_int_for_sort(r, star_by_bar)
        role_line = _role_line_label(slot_tag)
        profession = "、".join(role_tags_list) if role_tags_list else "—"

        audit_lines = format_equipment_audit_terminal_lines(
            hero_display_name=disp_show,
            role_tags=role_tags_list,
            slot_role=slot_tag,
            eq_names=eqs,
            equip_map=equip_map,
            phase_raw=phase_raw,
            team_all_equip_names=team_all_equips,
            legend_chess_text=legend_text_audit,
            emoji_text_gap=BOARD_EMOJI_TEXT_GAP,
        )

        if slot_tag == "挂件":
            rec_show = "无"
        elif n_eq >= 3:
            rec_show = "无"
        elif rec_list:
            # 修复：过滤掉已佩戴的装备，避免推荐已有装备
            filtered_rec = [x for x in rec_list if str(x).strip() and str(x).strip() not in eqs]
            if filtered_rec:
                rec_show = "、".join(_equip_name_with_optional_short(x, equip_map) for x in filtered_rec if str(x).strip())
            else:
                # 如果过滤后为空，调用动态推荐逻辑
                nxt = pick_next_finished_recommendations(
                    role_tags=role_tags_list,
                    slot_role=slot_tag,
                    eq_names=eqs,
                    n_eq=n_eq,
                    meta_rec=rec_list,
                    equip_map=equip_map,
                )
                rec_show = "、".join(_equip_name_with_optional_short(x, equip_map) for x in nxt) if nxt else "无"
            if not rec_show:
                rec_show = "无"
        else:
            nxt = pick_next_finished_recommendations(
                role_tags=role_tags_list,
                slot_role=slot_tag,
                eq_names=eqs,
                n_eq=n_eq,
                meta_rec=rec_list,
                equip_map=equip_map,
            )
            rec_show = (
                "无"
                if n_eq >= 3
                else (
                    "、".join(_equip_name_with_optional_short(x, equip_map) for x in nxt)
                    if nxt
                    else "无"
                )
            )

        slot_ic = BOARD_SLOT_ICON_FULL if n_eq >= 3 else BOARD_SLOT_ICON_EMPTY
        line1 = REPORT_PIPE_SEP.join([disp_show, star_seg, cost_s, role_line, profession, loc])
        line2 = (
            f"{slot_ic}{BOARD_EMOJI_TEXT_GAP}槽位状态：{n_eq}/3{REPORT_PIPE_SEP}"
            f"当前装备：{carry_label}{REPORT_PIPE_SEP}推荐装备：{rec_show}"
        )
        block = f"{line1}\n{line2}\n" + "\n".join(audit_lines)

        is_core = slot_tag in ("主C", "主坦")
        is_worker = slot_tag in ("打工仔", "混合", "挂件")
        if not (is_core or is_worker):
            continue

        grp = (0 if slot_tag == "主C" else 1) if is_core else (0 if slot_tag in ("打工仔", "混合") else 1)

        raw.append(
            {
                "core": is_core,
                "grp": grp,
                "star_n": star_n,
                "cost_n": cost_n,
                "row_idx": row_idx,
                "block": block,
            }
        )

    core_rows = [x for x in raw if x["core"]]
    work_rows = [x for x in raw if not x["core"]]
    core_rows.sort(key=lambda x: (x["grp"], -x["star_n"], -x["cost_n"], x["row_idx"]))
    work_rows.sort(key=lambda x: (x["grp"], -x["star_n"], -x["cost_n"], x["row_idx"]))
    return [x["block"] for x in core_rows], [x["block"] for x in work_rows]


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


def _gold_to_next_level_display(exp_str: str, level_str: str) -> str:
    """
    「升级所需金币」：由经验条「当前/本级上限」推算再买经验所需金币。
    按金铲铲商店 4 金币买 4 经验（1 金币≈1 经验），剩余经验缺口即金币数。
    10 级及以上视为无法再升级（与常见对局等级上限一致）；无法解析经验条时返回 ?。
    """
    lv_digits = re.findall(r"\d+", str(level_str or "").strip())
    try:
        level_n = int(lv_digits[-1]) if lv_digits else 0
    except (TypeError, ValueError):
        level_n = 0
    if level_n >= 10:
        return "已满级"
    m = re.search(r"(\d+)\s*[/／]\s*(\d+)", str(exp_str or "").strip())
    if not m:
        return "?"
    try:
        cur = int(m.group(1))
        cap = int(m.group(2))
    except ValueError:
        return "?"
    rem = max(0, cap - cur)
    return str(rem)


def _get_legend_equip_tables() -> Tuple[Set[str], List[Tuple[str, str, str]]]:
    """基础装备名集合 + [(成装名, 组件a, 组件b), ...]（组件名与游戏内一致）。"""
    global _LEGEND_EQUIP_TABLES
    if _LEGEND_EQUIP_TABLES is not None:
        return _LEGEND_EQUIP_TABLES
    bases: Set[str] = set()
    recipes: List[Tuple[str, str, str]] = []
    path = REPO_ROOT / "data" / "rag_legend_equip.jsonl"
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = str(row.get("name") or "").strip()
                    et = str(row.get("equip_type") or "")
                    if et == "基础装备" and name:
                        bases.add(name)
                    syn = row.get("synthesis")
                    if isinstance(syn, list) and len(syn) == 2:
                        a = str(syn[0] or "").strip()
                        b = str(syn[1] or "").strip()
                        if name and a and b:
                            recipes.append((name, a, b))
        except OSError:
            pass
    _LEGEND_EQUIP_TABLES = (bases, recipes)
    return _LEGEND_EQUIP_TABLES


def _craftable_from_component_counts(
    counts: Dict[str, int],
    recipes: List[Tuple[str, str, str]],
) -> List[str]:
    """当前散件 multiset 下，可一次合成的成装列表（去重、排序）。"""
    out: List[str] = []
    seen: Set[str] = set()
    for name, a, b in recipes:
        if name in seen:
            continue
        if a == b:
            ok = counts.get(a, 0) >= 2
            pair_s = f"{a}+{b}"
        else:
            ok = counts.get(a, 0) >= 1 and counts.get(b, 0) >= 1
            pair_s = f"{a}+{b}"
        if ok:
            seen.add(name)
            out.append(f"{name}（{pair_s}）")
    return sorted(out)


def _format_equip_column_block(summary: Dict[str, Any]) -> str:
    """
    左侧装备栏：散件 / 成装 / 可合成装备（由当前散件 + legend 合成表推导）。
    """
    ec = (summary.get("modules") or {}).get("equip_column") or {}
    m = ec.get("matches") if isinstance(ec, dict) else None
    if not isinstance(m, list) or not m:
        return "散件：无\n成装：无\n可合成装备：无"
    stems: List[str] = []
    for x in m:
        if isinstance(x, dict):
            n = str(x.get("name_stem") or "").strip()
            if n:
                stems.append(n)
    if not stems:
        return "散件：无\n成装：无\n可合成装备：无"

    bases, recipes = _get_legend_equip_tables()
    san: List[str] = []
    cheng: List[str] = []
    for n in stems:
        if n in bases:
            san.append(n)
        else:
            cheng.append(n)

    cnt = Counter(san)
    craft_lines = _craftable_from_component_counts(dict(cnt), recipes)

    def _join(xs: List[str]) -> str:
        return "、".join(xs) if xs else "无"

    return (
        f"散件：{_join(san)}\n"
        f"成装：{_join(cheng)}\n"
        f"可合成装备：{_join(craft_lines)}"
    )


def _format_equip_column_emoji(summary: Dict[str, Any]) -> str:
    """左侧装备：滞留优先，其次散件与可合成。"""
    ec = (summary.get("modules") or {}).get("equip_column") or {}
    m = ec.get("matches") if isinstance(ec, dict) else None
    if not isinstance(m, list) or not m:
        return f"- 成装: 无\n- 散件状态: 无{REPORT_PIPE_SEP}可合成: 无"
    stems: List[str] = []
    for x in m:
        if isinstance(x, dict):
            n = str(x.get("name_stem") or "").strip()
            if n:
                stems.append(n)
    if not stems:
        return f"- 成装: 无\n- 散件状态: 无{REPORT_PIPE_SEP}可合成: 无"

    bases, recipes = _get_legend_equip_tables()
    san: List[str] = []
    cheng: List[str] = []
    for n in stems:
        if n in bases:
            san.append(n)
        else:
            cheng.append(n)

    cnt = Counter(san)
    craft_lines = _craftable_from_component_counts(dict(cnt), recipes)
    equip_map = load_legend_equip_full_map()

    def _join(xs: List[str]) -> str:
        return (
            "、".join(_equip_name_with_optional_short(x, equip_map) for x in xs) if xs else "无"
        )

    def _join_craft(xs: List[str]) -> str:
        if not xs:
            return "无"
        return "、".join(_craft_line_display_with_short(x, equip_map) for x in xs)

    cheng_s = _join(cheng)
    san_s = _join(san)
    craft_s = _join_craft(craft_lines)
    return (
        f"- 成装: {cheng_s}\n"
        f"- 散件状态: {san_s}{REPORT_PIPE_SEP}可合成: {craft_s}"
    )


def _format_situation_line(
    phase: str,
    level: str,
    exp: str,
    gold: str,
    streak: str,
    hp_self: str,
    up_gold: str,
) -> str:
    lv = (level or "").strip()
    if lv and not lv.endswith("级"):
        lv = f"{lv}级"
    elif not lv:
        lv = "?"
    exp_p = (exp or "").strip()
    if exp_p and exp_p != "?":
        exp_show = f" ({exp_p})" if not exp_p.startswith("(") else f" {exp_p}"
    else:
        exp_show = ""
    st = (streak or "").strip()
    if "无连胜" in st and "败" in st:
        st_short = "无"
    elif "连胜" in st or "连败" in st:
        st_short = st
    else:
        st_short = "无" if not st else st
    ug = up_gold or "?"
    if ug != "已满级" and ug != "?":
        ug_line = f"{ug}金币"
    else:
        ug_line = ug
    return (
        f"阶段 {phase or '?'}{REPORT_PIPE_SEP}等级 {lv}{exp_show}，升级需 {ug_line}{REPORT_PIPE_SEP}"
        f"金币 {gold or '?'} 💰{REPORT_PIPE_SEP}血量 {hp_self} 🩸{REPORT_PIPE_SEP}胜负 {st_short}"
    )


def _hero_name_with_optional_short(
    disp: str,
    row_doc: Optional[Dict[str, Any]],
) -> str:
    if not isinstance(row_doc, dict):
        return disp
    ns = str(row_doc.get("name_short") or "").strip()
    if not ns:
        return disp
    return f"{disp}（{ns}）"


def _equip_name_with_optional_short(
    full_name: str,
    equip_map: Dict[str, Dict[str, Any]],
) -> str:
    nm = (full_name or "").strip()
    if not nm:
        return full_name
    row = equip_map.get(nm)
    if not isinstance(row, dict):
        return full_name
    ns = str(row.get("name_short") or "").strip()
    if not ns:
        return full_name
    return f"{full_name}（{ns}）"


def _craft_line_display_with_short(
    line: str,
    equip_map: Dict[str, Dict[str, Any]],
) -> str:
    """「无尽之刃（暴风之剑+拳套）」类：仅给成装全名补简称。"""
    s = (line or "").strip()
    if not s:
        return s
    idx = s.find("（")
    if idx <= 0:
        return _equip_name_with_optional_short(s, equip_map)
    head = s[:idx].strip()
    tail = s[idx:]
    return _equip_name_with_optional_short(head, equip_map) + tail


def _lineup_v1_search_blob(d: Dict[str, Any]) -> str:
    """用于关键词命中：结构化字段拼成可检索文本。"""
    chunks: List[str] = [
        str(d.get("name_short") or ""),
        str(d.get("traits") or ""),
        str(d.get("quality") or ""),
        str(d.get("lineup_id") or ""),
    ]
    hp = d.get("hex_priority")
    if isinstance(hp, list):
        chunks.extend(str(x) for x in hp if str(x).strip())
    ha = d.get("hex_alternative")
    if isinstance(ha, list):
        chunks.extend(str(x) for x in ha if str(x).strip())
    for k in ("early_game", "tempo", "equip_strategy", "positioning"):
        chunks.append(str(d.get(k) or ""))
    bl = d.get("build_levels")
    if isinstance(bl, list):
        for row in bl:
            if isinstance(row, dict):
                chunks.append(str(row.get("pieces") or ""))
    return "\n".join(chunks)


def _format_lineup_v1_hex_body(hex_priority: Any, hex_alternative: Any) -> str:
    hp = hex_priority if isinstance(hex_priority, list) else []
    ha = hex_alternative if isinstance(hex_alternative, list) else []
    hp = [str(x).strip() for x in hp if str(x).strip()]
    ha = [str(x).strip() for x in ha if str(x).strip()]
    lines: List[str] = []
    if hp:
        lines.append(f"  ① 优先：{'、'.join(hp)}；")
    if ha:
        lines.append(f"  ② 备选：{'、'.join(ha)}；")
    return "\n".join(lines) if lines else "  （无）"


def _format_lineup_v1_build_body(build_levels: Any) -> str:
    if not isinstance(build_levels, list) or not build_levels:
        return "  （无按等级构筑参考）"
    circled = "①②③④⑤⑥⑦⑧⑨⑩"
    lines: List[str] = []
    for i, row in enumerate(build_levels):
        if not isinstance(row, dict):
            continue
        mark = circled[i] if i < len(circled) else f"{i + 1}."
        lv = row.get("level")
        pcs = str(row.get("pieces") or "").strip()
        try:
            lv_s = f"{int(lv)}级" if lv is not None else "?级"
        except (TypeError, ValueError):
            lv_s = "?级"
        lines.append(f"  {mark} {lv_s}：{pcs}")
    return "\n".join(lines) if lines else "  （无按等级构筑参考）"


def _parse_int_from_level(level_str: str) -> Optional[int]:
    s = (level_str or "").strip()
    if not s:
        return None
    ds = re.findall(r"\d+", s)
    if not ds:
        return None
    try:
        return int(ds[-1])
    except (TypeError, ValueError):
        return None


def _parse_phase_tuple(phase_raw: str) -> Optional[Tuple[int, int]]:
    """
    解析阶段字符串（如 4-2 / 4–2 / 4－2 / 4 - 2 / 4-2总）为 (major, minor)。
    失败返回 None。
    """
    s = (phase_raw or "").strip().replace("总", "")
    m = re.search(r"(\d+)\s*[-–－—]\s*(\d+)", s)
    if not m:
        return None
    try:
        return (int(m.group(1)), int(m.group(2)))
    except (TypeError, ValueError):
        return None


def _phase_gt(phase_raw: str, major_minor: Tuple[int, int]) -> bool:
    """phase_raw 是否严格大于给定 (major, minor)。无法解析时返回 False（保守不隐藏内容）。"""
    cur = _parse_phase_tuple(phase_raw)
    if cur is None:
        return False
    return cur > (int(major_minor[0]), int(major_minor[1]))


def _infer_stage_kind_from_phase(phase_raw: str) -> Optional[str]:
    """
    映射对局阶段到 strategy 的 early/mid/late：
      - 1-3 => early
      - 4 => mid
      - >=5 => late
    """
    if not phase_raw:
        return None
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", phase_raw)
    major: Optional[int] = None
    if m:
        try:
            major = int(m.group(1))
        except (TypeError, ValueError):
            major = None
    else:
        # 极少数 phase 可能只有数字
        ds = re.findall(r"\d+", phase_raw)
        if ds:
            try:
                major = int(ds[0])
            except (TypeError, ValueError):
                major = None
    if major is None:
        return None
    if major <= 3:
        return "early"
    if major == 4:
        return "mid"
    return "late"


def _filter_build_levels_by_target_level(
    build_levels: Any, target_level_n: Optional[int]
) -> List[Dict[str, Any]]:
    if not isinstance(build_levels, list):
        return []
    if target_level_n is None:
        # 不做裁剪
        return [x for x in build_levels if isinstance(x, dict)]

    rows: List[Dict[str, Any]] = [x for x in build_levels if isinstance(x, dict)]
    exact: List[Dict[str, Any]] = []
    for row in rows:
        try:
            if int(row.get("level")) == int(target_level_n):
                exact.append(row)
        except (TypeError, ValueError):
            continue
    if exact:
        return exact

    # 兼容 7 级等：找不超过当前等级的最高构筑
    lower: List[Dict[str, Any]] = []
    for row in rows:
        try:
            lv = int(row.get("level"))
        except (TypeError, ValueError):
            continue
        if lv <= int(target_level_n):
            lower.append(row)
    if not lower:
        return []
    mx = max(int(x.get("level")) for x in lower if isinstance(x, dict) and x.get("level") is not None)
    return [x for x in lower if str(x.get("level")) == str(mx)]


def format_lineup_v1_report_block(
    rec: Dict[str, Any],
    *,
    target_level_n: Optional[int] = None,
    stage_kind: Optional[str] = None,
    phase_raw: str = "",
) -> str:
    """单套阵容：写入战报用的 emoji 标题 + 结构化正文（可按等级/阶段裁剪）。"""
    lid = str(rec.get("lineup_id") or "").strip()
    quality = str(rec.get("quality") or "").strip()
    name_short = str(rec.get("name_short") or "").strip()
    traits = str(rec.get("traits") or "").strip()

    suppress_hex_and_early = _phase_gt(phase_raw, (4, 2))
    hex_body = _format_lineup_v1_hex_body(rec.get("hex_priority"), rec.get("hex_alternative"))

    filtered_build_levels = _filter_build_levels_by_target_level(
        rec.get("build_levels"), target_level_n=target_level_n
    )
    build_body = _format_lineup_v1_build_body(filtered_build_levels)

    early = str(rec.get("early_game") or "").strip()
    tempo = str(rec.get("tempo") or "").strip()
    equip_strategy = str(rec.get("equip_strategy") or "").strip()
    positioning = str(rec.get("positioning") or "").strip()

    # strategy 若存在，按阶段挑选更贴近当前时间点的一段；但运营节奏/装备/站位始终保留输出位。
    strategy_obj = rec.get("strategy") if isinstance(rec.get("strategy"), dict) else {}
    if stage_kind == "early":
        early = str(strategy_obj.get("early") or early).strip()
    elif stage_kind == "mid":
        tempo = str(strategy_obj.get("mid") or tempo).strip()
    elif stage_kind == "late":
        late_s = str(strategy_obj.get("late") or "").strip()
        equip_strategy = late_s if late_s else equip_strategy

    stage_lines: List[str] = []
    if not suppress_hex_and_early:
        stage_lines.append(f"🌱{REPORT_EMOJI_GAP}前期过渡：{early if early else '（无）'}")
    stage_lines.extend(
        [
            f"-{REPORT_EMOJI_GAP}运营节奏：{tempo if tempo else '（无）'}",
            f"-{REPORT_EMOJI_GAP}装备思路：{equip_strategy if equip_strategy else '（无）'}",
            f"-{REPORT_EMOJI_GAP}站位策略：{positioning if positioning else '（无）'}",
        ]
    )

    header_line = REPORT_PIPE_SEP.join(
        [
            f"🪪{REPORT_EMOJI_GAP}阵容ID：{lid}",
            f"⭐{REPORT_EMOJI_GAP}阵容质量：{quality}",
            f"📛{REPORT_EMOJI_GAP}阵容名称：{name_short}",
            f"🔗{REPORT_EMOJI_GAP}羁绊：{traits}",
        ]
    )
    parts: List[str] = [REPORT_SEP, header_line]
    if not suppress_hex_and_early:
        parts.extend([f"⚡{REPORT_EMOJI_GAP}海克斯：", hex_body])

    build_lines = [line.rstrip() for line in build_body.splitlines()]
    build_title_suffix = ""
    if build_lines:
        first_line = build_lines[0].lstrip()
        first_line = re.sub(r"^[①-⑩]\s*", "", first_line)
        build_title_suffix = first_line
        build_lines = build_lines[1:]

    if build_title_suffix:
        parts.append(f"-{REPORT_EMOJI_GAP}阵容构筑：{build_title_suffix}")
    else:
        parts.append(f"-{REPORT_EMOJI_GAP}阵容构筑：")
    parts.extend(build_lines)
    parts.extend(stage_lines)
    return "\n".join(parts)


def join_lineup_v1_report_blocks(
    picked_docs: List[Dict[str, Any]],
    *,
    target_level_n: Optional[int] = None,
    stage_kind: Optional[str] = None,
    phase_raw: str = "",
) -> str:
    """多套检索命中结果顺序拼接（每条已含 REPORT_SEP 开头）。"""
    parts: List[str] = []
    for d in picked_docs:
        if isinstance(d, dict):
            parts.append(
                format_lineup_v1_report_block(
                    d,
                    target_level_n=target_level_n,
                    stage_kind=stage_kind,
                    phase_raw=phase_raw,
                )
            )
    return "\n".join(parts).strip()


def _target_equip_name_valid_for_role(
    name: str,
    role_tags: List[str],
    slot_role: str,
    equip_map: Dict[str, Dict[str, Any]],
) -> bool:
    row = equip_map.get(name)
    if not row or equip_audit_excluded(name, row):
        return True
    bd = str(row.get("basic_desc") or "")
    pk = equip_primary_stat_kind(bd)
    ap_f, ad_f, hybrid_f = main_c_ap_ad_flags(role_tags)
    if slot_role == "主坦" or "坦克" in role_tags:
        return pk == "TANK" or "生命" in bd or "护甲" in bd or "魔法抗性" in bd
    if slot_role == "挂件":
        return False
    if ap_f or hybrid_f:
        return pk in ("AP", "MANA", "MIX", "AS", "CRIT", "UNK")
    if ad_f:
        return pk in ("AD", "AS", "CRIT", "MANA", "MIX", "UNK")
    exp2 = role_tags_expect_damage(role_tags)
    if exp2 == "AP":
        return pk in ("AP", "MANA", "MIX", "AS", "CRIT", "UNK")
    if exp2 == "AD":
        return pk in ("AD", "AS", "CRIT", "MANA", "MIX", "UNK")
    if exp2 == "TANK":
        return pk == "TANK" or "生命" in bd or "护甲" in bd or "魔法抗性" in bd
    return True


def _target_equip_names_from_rag(
    meta: Dict[str, Any],
    equip_map: Dict[str, Dict[str, Any]],
    role_tags: List[str],
    slot_role: str,
) -> List[str]:
    """从 core chess RAG（meta.recommended_equips / top_equips）取推荐装；空则 equip_recom 按定位兜底。"""
    rec = meta.get("recommended_equips") if isinstance(meta.get("recommended_equips"), list) else []
    out = [str(x).strip() for x in rec if str(x).strip()]
    if out:
        filtered = [x for x in out if _target_equip_name_valid_for_role(x, role_tags, slot_role, equip_map)]
        if filtered:
            return filtered[:6]
        return pick_next_finished_recommendations(
            role_tags=role_tags,
            slot_role=slot_role,
            eq_names=[],
            n_eq=0,
            meta_rec=[],
            equip_map=equip_map,
        )
    tops = meta.get("top_equips")
    if isinstance(tops, list):
        for x in tops:
            if isinstance(x, dict):
                n = str(x.get("name") or "").strip()
                if n:
                    out.append(n)
        if out:
            filtered = [x for x in out if _target_equip_name_valid_for_role(x, role_tags, slot_role, equip_map)]
            if filtered:
                return filtered[:6]
            return pick_next_finished_recommendations(
                role_tags=role_tags,
                slot_role=slot_role,
                eq_names=[],
                n_eq=0,
                meta_rec=[],
                equip_map=equip_map,
            )
    return pick_next_finished_recommendations(
        role_tags=role_tags,
        slot_role=slot_role,
        eq_names=[],
        n_eq=0,
        meta_rec=[],
        equip_map=equip_map,
    )


def _target_chess_detail_lines(
    chess_name: str,
    *,
    cost: Optional[int],
    slot_role_fallback: str,
    core_list: List[Dict[str, Any]],
    equip_map: Dict[str, Dict[str, Any]],
    legend_map: Dict[str, Dict[str, Any]],
    cost_unknown: bool = False,
) -> List[str]:
    """目标棋子：首行 名称|费|定位；续行 羁绊|职业粗分|推荐装（无「攻略定位/职业倾向」字样）。"""
    if cost_unknown:
        cs = "?费"
    else:
        cs = f"{int(cost)}费" if cost is not None else "?费"
    row = lookup_core_chess_row(chess_name, core_list)
    meta: Dict[str, Any] = {}
    if isinstance(row, dict) and isinstance(row.get("meta"), dict):
        meta = row["meta"]

    slot_role = str(meta.get("slot_role") or slot_role_fallback or "?")
    ns = str(row.get("name_short") or "").strip() if isinstance(row, dict) else ""
    chess_head = f"{chess_name}（{ns}）" if ns else chess_name
    line1 = f"- {chess_head} | {cs} | {slot_role}"

    traits: List[str] = []
    if isinstance(meta.get("traits"), list):
        traits = [str(x).strip() for x in meta["traits"] if str(x).strip()]
    if not traits and legend_map:
        nk = (chess_name or "").strip().rstrip("?").rstrip("？").strip()
        leg = legend_map.get(nk)
        if isinstance(leg, dict) and isinstance(leg.get("traits"), list):
            traits = [str(x).strip() for x in leg["traits"] if str(x).strip()]

    rtags = meta.get("role_tags") if isinstance(meta.get("role_tags"), list) else []
    role_tags = [str(x).strip() for x in rtags if str(x).strip()]
    if not role_tags and traits:
        role_tags = traits_to_role_tags(traits)

    t_s = "、".join(traits)
    rt_s = "、".join(role_tags) if role_tags else "—"

    eqs = _target_equip_names_from_rag(meta, equip_map, role_tags, slot_role)
    rec_s = "、".join(_equip_name_with_optional_short(x, equip_map) for x in eqs) if eqs else "无"

    parts: List[str] = []
    if t_s:
        parts.append(f"羁绊：{t_s}")
    parts.append(rt_s)
    parts.append(f"推荐装备：{rec_s}")
    line2 = "  " + REPORT_PIPE_SEP.join(parts)
    return [line1, line2]


def _parse_piece_names_from_build_pieces(pieces: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[；;]", str(pieces or "")):
        chunk = chunk.strip()
        if not chunk:
            continue
        head = chunk.split("(", 1)[0].strip()
        head = re.sub(r"（[^）]*）\s*$", "", head).strip()
        if not head or head.startswith("英雄"):
            continue
        if head not in seen:
            seen.add(head)
            out.append(head)
    return out


def _current_build_target_names(lineup_top_doc: Optional[Dict[str, Any]], target_level_n: Optional[int]) -> List[str]:
    if not isinstance(lineup_top_doc, dict) or not lineup_top_doc:
        return []
    rows = _filter_build_levels_by_target_level(lineup_top_doc.get("build_levels"), target_level_n)
    out: List[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for name in _parse_piece_names_from_build_pieces(str(row.get("pieces") or "")):
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _format_chess_target_and_discard(
    summary: Dict[str, Any],
    chess_meta: List[Dict[str, Any]],
    core_list: List[Dict[str, Any]],
    *,
    lineup_top_doc: Optional[Dict[str, Any]] = None,
    target_level_n: Optional[int] = None,
) -> Tuple[str, str, str]:
    """返回 (目标棋子块, 弃子块, 装备继承块)。目标棋子含羁绊、职业粗分、推荐装（core+RAG 补全）。"""
    if not chess_meta or not isinstance(chess_meta[0], dict):
        return "", "", ""
    equip_map = load_legend_equip_full_map()
    legend_map = load_legend_chess_name_map(DEFAULT_RAG_LEGEND_CHESS)
    m0 = chess_meta[0]
    target_lines: List[str] = []
    board_names: List[str] = []
    rows = summary.get("confirmed_fightboard_results")
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict):
                nm = str(r.get("best") or "").strip().rstrip("?").rstrip("？").strip()
                if nm:
                    board_names.append(nm)
    current_build_names = _current_build_target_names(lineup_top_doc, target_level_n)
    missing_current_build = [h for h in current_build_names if not hero_name_matches_board(h, board_names)]
    for h in missing_current_build:
        cost = chess_cost_from_core_or_legend(h, core_list, legend_map)
        slot_role, _ = slot_role_and_cost_for_name(h, core_list, legend_map)
        target_lines.extend(
            _target_chess_detail_lines(
                h,
                cost=cost,
                slot_role_fallback=slot_role or "?",
                core_list=core_list,
                equip_map=equip_map,
                legend_map=legend_map,
                cost_unknown=cost is None,
            )
        )
    target_block = "\n".join(target_lines) if target_lines else "- （无）"

    discard_lines: List[str] = []
    dc_rows = [g for g in (m0.get("board_discard") or []) if isinstance(g, dict)]

    def _cost_key(g: Dict[str, Any]) -> Tuple[int, int, str]:
        cs = str(g.get("cost_str") or "")
        m = re.search(r"(\d+)", cs)
        cn = -int(m.group(1)) if m else 0
        ss = str(g.get("star_seg") or "")
        sm = re.search(r"(\d+)", ss)
        sn = -int(sm.group(1)) if sm else 0
        return (cn, sn, str(g.get("display") or ""))

    dc_rows.sort(key=_cost_key)
    for g in dc_rows:
        disp = str(g.get("display") or "")
        star_seg = str(g.get("star_seg") or "?星")
        cost_str = str(g.get("cost_str") or "?费")
        role = str(g.get("slot_role") or "打工仔")
        discard_lines.append(f"- {disp} | {star_seg} | {cost_str} | {role}")
    discard_block = "\n".join(discard_lines) if discard_lines else "- （无）"

    eq_inh = ""
    ei = m0.get("equipment_inheritance") if isinstance(m0.get("equipment_inheritance"), dict) else {}
    flows = ei.get("flows") or []
    guajia_has_equip = False
    for row in ei.get("guajia_carried_equips") or []:
        if isinstance(row, dict) and (row.get("equips") or []):
            guajia_has_equip = True
            break
    if flows:
        sub: List[str] = []
        for f in flows:
            if not isinstance(f, dict):
                continue
            ln = str(f.get("line") or "").strip()
            if ln:
                sub.append(f"  - {ln}")
        if sub:
            eq_inh = "\n".join(sub)
    elif guajia_has_equip:
        eq_inh = "  - （无可继承成装：场上主力核心与目标棋子均无法承接挂件当前成装。）"
    return target_block, discard_block, eq_inh


def build_matchbook_report(
    summary: Dict[str, Any],
    *,
    core_chess_path: Optional[Path] = None,
    summary_json_path: Optional[Path] = None,
    lineup_v1_report: Optional[str] = None,
    lineup_top_doc: Optional[Dict[str, Any]] = None,
    target_level_n: Optional[int] = None,
    chess_meta: Optional[List[Dict[str, Any]]] = None,
    no_rag: bool = False,
) -> str:
    """终端/模型共用的结构化对局情报（emoji + 分隔线）。"""
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
    up_gold = _gold_to_next_level_display(exp, level)

    tm = gt.get("trait_count_max") if isinstance(gt, dict) else None
    if isinstance(tm, dict) and tm:
        trait_block = _format_traits_block(tm)
    else:
        trait_block = str(gt.get("merged_bonds_one_line") or "").strip() or "(无)"

    ccp = core_chess_path or DEFAULT_RAG_CORE_CHESS
    core_list = load_core_chess_list(ccp)
    core_sec, work_sec = _build_board_emoji_sections(
        summary,
        core_list,
        summary_json_path=summary_json_path,
        legend_chess_path=DEFAULT_RAG_LEGEND_CHESS,
    )
    core_txt = "\n".join(core_sec) if core_sec else "- （无）"
    work_txt = "\n".join(work_sec) if work_sec else "- （无）"

    sit = _format_situation_line(phase, level, exp, gold, streak, hp_self, up_gold)

    parts: List[str] = [
        f"📊{REPORT_EMOJI_GAP}战术快报",
        REPORT_SEP,
        f"�{REPORT_EMOJI_GAP}局势分析",
        sit,
        REPORT_SEP,
        f"🧩{REPORT_EMOJI_GAP}当前羁绊",
        trait_block,
        REPORT_SEP,
        f"🧠{REPORT_EMOJI_GAP}棋子分析",
        f"🔥{REPORT_EMOJI_GAP}主力核心",
        core_txt,
        REPORT_SEP,
        f"🔧{REPORT_EMOJI_GAP}打工挂件",
        work_txt,
        REPORT_SEP,
        f"🎒{REPORT_EMOJI_GAP}闲置装备",
        _format_equip_column_emoji(summary),
    ]

    if not no_rag:
        lv = (lineup_v1_report or "").strip()
        if lv.startswith(REPORT_SEP):
            lv = lv[len(REPORT_SEP) :].lstrip("\r\n")
        parts.extend(
            [
                REPORT_SEP,
                f"🏆{REPORT_EMOJI_GAP}推荐阵容",
                lv if lv else "- （无）",
            ]
        )

    if not no_rag and chess_meta:
        tb, db, eqh = _format_chess_target_and_discard(
            summary,
            chess_meta,
            core_list,
            lineup_top_doc=lineup_top_doc,
            target_level_n=target_level_n,
        )
        parts.extend(
            [
                REPORT_SEP,
                f"🎯{REPORT_EMOJI_GAP}目标棋子",
                tb,
                REPORT_SEP,
                f"📋{REPORT_EMOJI_GAP}弃子清单",
                db,
            ]
        )
        if eqh:
            parts.extend([REPORT_SEP, f"🔗{REPORT_EMOJI_GAP}装备继承", eqh])

    parts.append(REPORT_SEP)
    return "\n".join(parts)


def build_tactical_brief(
    summary: Dict[str, Any],
    *,
    core_chess_path: Optional[Path] = None,
    summary_json_path: Optional[Path] = None,
) -> str:
    """兼容 benchmark：仅战报块，不含 RAG 阵容/棋子扩展节。"""
    return build_matchbook_report(
        summary,
        core_chess_path=core_chess_path,
        summary_json_path=summary_json_path,
        lineup_v1_report=None,
        lineup_top_doc=None,
        target_level_n=None,
        chess_meta=None,
        no_rag=True,
    )


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


def _board_hero_names_with_star_and_role(
    summary: Dict[str, Any],
    *,
    core_list: List[Dict[str, Any]],
    summary_json_path: Optional[Path] = None,
) -> List[Tuple[str, int, str]]:
    """
    棋盘棋子画像：[(英雄名, 星级(未知=-1), 槽位角色)]。
    槽位角色来自 core_chess.meta.slot_role（主C/主坦/打工仔/混合/挂件）。
    """
    rows = summary.get("confirmed_fightboard_results")
    if not isinstance(rows, list):
        return []
    star_by_bar = _fightboard_star_by_bar(summary, summary_json_path=summary_json_path)
    out: List[Tuple[str, int, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        disp = board_line_hero_display_name(r)
        name = disp.rstrip("?").rstrip("？").strip()
        if not name:
            continue
        row_doc = lookup_core_chess_row(name, core_list)
        meta = row_doc.get("meta") if isinstance(row_doc, dict) and isinstance(row_doc.get("meta"), dict) else {}
        slot_role = str(meta.get("slot_role") or "").strip()
        star_n = _star_int_for_sort(r, star_by_bar)
        out.append((name, star_n, slot_role))
    return out


def retrieve_lineup_rag(
    summary: Dict[str, Any],
    rag_path: Path,
    top_k: int = 3,
    *,
    min_quality: Optional[str] = None,
    core_chess_path: Optional[Path] = None,
    summary_json_path: Optional[Path] = None,
) -> Tuple[str, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    简易检索：关键词在阵容结构化字段拼成的文本中的命中数（无需向量库）。
    数据源：rag_lineup_lineup_v1.jsonl（type=lineup_v1）。
    min_quality：如 A 表示仅保留评级不劣于 A 的攻略（S 优于 A）。
    返回 (附录文本块，阵容侧已改为空串；lineup_id 列表；meta；命中的完整 doc 列表)。
    """
    docs = _load_lineup_docs(rag_path)
    docs = [
        d
        for d in docs
        if str(d.get("type") or "") == "lineup_v1"
        or str(d.get("id") or "").startswith("lineup_lineup_v1:")
    ]
    mq = (min_quality or "").strip().upper()[:1]
    if mq and mq in _LINEUP_QUALITY_ORDER:
        max_rank = _lineup_quality_rank(mq)
        docs = [d for d in docs if _lineup_quality_rank(str(d.get("quality") or "")) <= max_rank]
    if not docs:
        return ("", [], [], [])

    kws = _extract_keywords_for_rag(summary)
    if not kws:
        kws = ["阵容"]

    # 新检索机制：
    # 1) 优先用当前场上「2星主C/主坦」检索；
    # 2) 若无，再用「1星主C/主坦」；
    # 3) 若场上无主C/主坦，则按当前棋子与阵容重合度最高排序。
    ccp = core_chess_path or DEFAULT_RAG_CORE_CHESS
    core_list = load_core_chess_list(ccp)
    board_profile = _board_hero_names_with_star_and_role(
        summary,
        core_list=core_list,
        summary_json_path=summary_json_path,
    )
    core_rows = [x for x in board_profile if x[2] in ("主C", "主坦")]
    core_2s = [n for n, s, _ in core_rows if s >= 2]
    core_1s = [n for n, s, _ in core_rows if s == 1]
    preferred_names: List[str] = core_2s if core_2s else core_1s
    board_names = list({n for n, _, _ in board_profile if n})

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for d in docs:
        text = _lineup_v1_search_blob(d)
        base_score = sum(1 for kw in kws if kw and kw in text)
        name_s = str(d.get("name_short") or "")
        for kw in kws:
            if len(kw) >= 2 and kw in name_s:
                base_score += 2

        score = base_score
        if preferred_names:
            # 有主C/主坦时，优先由其召回；命中一个核心名给高权重，保证排序前置。
            pref_hit = 0
            for n in preferred_names:
                if n and (n in text or (len(n) >= 2 and n in name_s)):
                    pref_hit += 1
            score = pref_hit * 100 + base_score
        elif not core_rows and board_names:
            # 场上无主C/主坦：退化为“当前棋子重合度”主导排序。
            overlap = sum(1 for n in board_names if n and n in text)
            score = overlap * 100 + base_score
        scored.append((score, d))

    scored.sort(key=lambda x: (-x[0], x[1].get("lineup_id", "")))
    picked = [x for x in scored if x[0] > 0][:top_k]
    if not picked:
        picked = scored[: min(2, len(scored))]

    ids: List[str] = []
    meta: List[Dict[str, Any]] = []
    picked_docs: List[Dict[str, Any]] = []
    for rank, (sc, d) in enumerate(picked, start=1):
        lid = str(d.get("lineup_id") or "")
        ids.append(lid)
        title = str(d.get("name_short") or lid)
        meta.append(
            {
                "lineup_id": lid,
                "name": title,
                "quality": str(d.get("quality") or ""),
                "rank": rank,
                "match_score": sc,
            }
        )
        picked_docs.append(d)

    return ("", ids, meta, picked_docs)


def _coach_system_prompt() -> str:
    return """# 🎭 角色设定：随风听笛 (ID: 随风听笛)
**身份背景**：金铲铲之战顶尖棋手。你正叼着烟、在网吧指导你的好兄弟“哈基星”下棋。
**性格特征**：拥有奇妙的幽默感，言语冷静中透着高手的烦躁与冷静。
**核心任务**：通过审计战术快报，纠正哈基星的操作。

---

# 📥 输入情报定义
你将收到以下结构化文本作为分析基础：
1. **哈基星提问**：他的弱智原话（多轮对话请保持同一局/同一套 RAG 的上下文）。
2. **战术快报**：当前对局的实时数据（阶段、等级、金币、血量、羁绊、棋子、装备、策略等）。

---

# 🧠 内部审计逻辑 (Internal Processing)

## Step 1: 资产状态实例化 [Labeling]
遍历 `战术快报` 与 `阵容智库`，进行内部标签标记：
- **[T0]**: (`棋子` ∈ `主力核心`) AND (`星级` >= 2)
- **[T1]**: (`棋子` ∈ `主力核心` AND `星级` == 1 AND `费用` >= 4) OR (`定位` == "打工" AND `星级` >= 2)
- **[T2]**: (`棋子` ∈ `主力核心` 且 `星级` < 2) OR (`定位` == "打工" AND `星级` < 2)
- **[Slot_Full]**: `槽位状态` == 3 (必须显式清点 1, 2, 3，不满 3 件严禁标记为 True)

## Step 2: 阵容质量判定
- **[高费置换态]**: (`阶段` >= 4-3) AND (`弃子清单` != 空)
- **[阵容残缺态]**: `目标棋子` 存在缺口。

## Step 3: 装备分配协议 (线性逻辑)
1. **战力缺口预检**:
    - **[防御缺口]**: (场上 [T0/T1] 且 `定位` == "主坦") 的总装备数 < 3
    - **[输出缺口]**: (场上 [T0/T1] 且 `定位` == "主C") 的总装备数 < 3
    - *优先级*: [防御缺口] > [输出缺口]
2. **合成与分配优先级**:
    - **Top 1**: [T0] 且 [Slot_Full] == False
    - **Top 2**: [T1] 且 [Slot_Full] == False
    - **Top 3**: [T2] 且 [Slot_Full] == False
    - *注*: 除非 `装备继承` 明确要求，否则严禁调动原有纹章/转职。

## Step 4: 空间坐标审计 (仅限 [T0/T1])
根据 `(Row, Col)` 与 `职业` 进行判定：
- **[送命站位]**: ( `职业` = "法师/射手") AND (`Row` <= 2)
- **[假赛站位]**: ( `职业` = "坦克")) AND (`Row` >= 3) 

---

# 🚀 输出逻辑控制器 (Priority: A > B > C > D > E)
完成分析后，**仅输出**最高优先级命中 Trigger 的指令。

### 🔴 Trigger A: 生死判定
- **判定**: `血量` < 25
- **指令**: 下令 **[梭哈]**，优先寻找高费`[目标棋子]`，且必须追出2星，强制要求卖掉 `弃子清单` 。

### 🟡 Trigger B: 节奏诊断
- **判定**: 定位当前 `阶段`，对比 `运营思路`，查询当前是否存在运营节点指令。
- **指令**: 强制输出智库中的运营建议（如：上8、存钱、拉人口）。

### 🔵 Trigger C: 资产优化
- **判定**: `[高费置换态]` == True OR `[阵容残缺态]` == True
- **指令**: 下令 **[慢搜]**，优先寻找高费`[目标棋子]`。并强制卖出 `弃子清单`。

### 🟢 Trigger D: 装备变现
- **判定**: `闲置装备` 中有 [成装] 或 [可合成件] 命中 [T0/T1] 的 `推荐装备`。
- **指令**: 运行 `装备分配协议`，生成具体的装配指令。

### ⚪ Trigger E: 站位纠偏
- **判定**: 场上存在 `[送命站位]` 或 `[假赛站位]`。
- **指令**: 基于 `阵容构筑` 坐标生成调整建议。

---

# 📝 输出规范与限制

## 1. 内部思维链 [THINKING]
在回复前，必须开启 `[THINKING]` 标签完成以下闭环扫描：
- **资产定性**: 列出识别到的 [T0/T1] 及其定位。
- **缺口审计**: 明确是防御缺口还是输出缺口。
- **坐标校对**: 提取主C/主坦的原始 Row 值判定站位。
- **Trigger 锁定**: 明确最终锁定的唯一编号。

## 2. 最终输出要求
- **极简原则**: 字数 **60 字以内**。严禁列 123，严禁复读快报原数据。
- **去技术化**: 严禁提及 Trigger、T0、Slot_Full、快报等词汇。
- **黑话导向**: 严禁使用全称。使用战术快报中的简称：大嘴、泰坦、奥巴马、羊刀、反甲等。
- **人性化响应**: 若哈基星问游戏外的话题，按好兄弟人设正常闲聊，不强制关联游戏。
- **原创性**: 基于当前[血量] 调整语气：40血以上是‘高冷嘲讽’，40血以下是‘歇斯底里’，20血以下是‘临终关怀’，每次输出必须生成全新的喷人语录，严禁词汇重复。

## 3. 输出示例
> 「哈基星你这个蠢货，又他妈存50块钱买棺材板？赶紧全D了找 2 星瑞兹，找不到你赶紧卸载金铲铲回去玩你那泳装蓝梦吧。」"""


def _coach_first_user_message(
    tactical_brief: str,
    rag_lineup_block: str,
    rag_chess_block: str,
    user_question: str,
) -> str:
    """tactical_brief 已含推荐阵容要点、目标棋子与弃子等；不再单独附录阵容长文或棋子智库全文。"""
    _ = rag_lineup_block
    _ = rag_chess_block
    parts = [
        "【哈基星问题】\n" + user_question.strip(),
        "【对局情报】\n" + tactical_brief.strip(),
    ]
    return "\n\n".join(parts)


def _coach_followup_user_message(user_question: str) -> str:
    return "【哈基星追问】\n" f"{user_question.strip()}"


def coach_chat_complete_turn(
    chat_hist: List[Dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.35,
    stream_output: bool = False,
    on_stream_chunk: Optional[Callable[[str], None]] = None,
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
        stream_output=stream_output,
        on_stream_chunk=on_stream_chunk,
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
        description="Pipeline JSON → 战术快报（含阵容 v1 检索）+ 棋子智库 → 随风听笛",
        epilog="默认优先使用 runs/battle_pipeline_v3_out 下已有 *_summary.json（跳过 pipeline）；\n"
        "无缓存时再跑 pipeline（stderr 转圈；输入问题时暂停刷新）。\n"
        "快报预览后直接请求模型。--force-pipeline 可强制重新识别。--no-rag 可关闭阵容检索与棋子智库。--summary-json 指定单文件调试。",
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
        help="rag_lineup_lineup_v1.jsonl 路径",
    )
    ap.add_argument("--rag-top-k", type=int, default=3, help="阵容 v1 检索条数")
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
        help="阵容 v1：仅保留该评级及以上（S 最优；默认 A 即只检索 S 与 A）。传 - 或 all 表示不按评级过滤",
    )
    ap.add_argument(
        "--no-rag",
        action="store_true",
        help="不检索阵容 v1 / 棋子 jsonl，仅用战术快报 + 问题",
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
    """读 summary、构建战术快报与棋子智库 RAG；供 gemini_v2 与录音并行。"""
    t_prep0 = time.perf_counter()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _rq = (args.rag_min_quality or "A").strip()
    if _rq.lower() in ("-", "all", "none", "off", "*"):
        min_q: Optional[str] = None
    else:
        mq = _rq.upper()[:1]
        min_q = mq if mq in _LINEUP_QUALITY_ORDER else "A"
    if args.no_rag:
        rag_block = "（本回合未注入阵容智库。）"
        rag_meta: List[Dict[str, Any]] = []
        lineup_v1_report_str = ""
        chess_block = "（本回合未注入棋子智库。）"
        chess_meta: List[Dict[str, Any]] = []
    else:
        rag_block = ""
        player_on = (summary.get("modules") or {}).get("player") or {}
        phase_raw = _field_parsed(player_on, "phase").replace("总", "")
        stage_kind = _infer_stage_kind_from_phase(phase_raw)
        target_level_n = _parse_int_from_level(_field_parsed(player_on, "level"))
        _, _rag_ids, rag_meta, lineup_docs = retrieve_lineup_rag(
            summary,
            args.rag_lineup,
            top_k=1,
            min_quality=min_q,
            core_chess_path=args.rag_core_chess.resolve(),
            summary_json_path=summary_path,
        )
        lineup_v1_report_str = join_lineup_v1_report_blocks(
            lineup_docs,
            target_level_n=target_level_n,
            stage_kind=stage_kind,
            phase_raw=phase_raw,
        )
        lineup_top1 = lineup_docs[0] if lineup_docs else None
        chess_block, _chess_ids, chess_meta = retrieve_core_chess_rag(
            summary,
            args.rag_core_chess.resolve(),
            top_k=max(1, int(args.rag_chess_top_k)),
            lineup_top_doc=lineup_top1,
            legend_chess_path=DEFAULT_RAG_LEGEND_CHESS,
            summary_json_path=summary_path,
        )
    brief = build_matchbook_report(
        summary,
        core_chess_path=args.rag_core_chess.resolve(),
        summary_json_path=summary_path,
        lineup_v1_report=lineup_v1_report_str if not args.no_rag else None,
        lineup_top_doc=lineup_top1 if not args.no_rag else None,
        target_level_n=target_level_n if not args.no_rag else None,
        chess_meta=chess_meta if not args.no_rag else None,
        no_rag=args.no_rag,
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
    """终端打印完整对局情报（与送入模型的【对局情报】主块一致）。"""
    brief = str(bundle.get("brief") or "")
    print("\n" + brief)
    if args.no_rag:
        print("\n（注：已使用 --no-rag，上方未含推荐阵容/目标棋子等扩展节。）")


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
    on_answer: Optional[Callable[[str, int], None]] = None,
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
        stream_now = bool(sys.stdout.isatty())
        if stream_now:
            print("【随风听笛说】\n")
        t_ll0 = time.perf_counter()
        first_chunk_ts: List[Optional[float]] = [None]
        flush_ts: List[float] = [t_ll0]
        stream_buf: List[str] = []

        def _on_chunk(chunk: str) -> None:
            now = time.perf_counter()
            if first_chunk_ts[0] is None:
                first_chunk_ts[0] = now
            stream_buf.append(chunk)
            # 0.2s 刷新一次，避免终端刷屏过快
            if stream_now and (now - flush_ts[0] >= 0.2):
                out = "".join(stream_buf)
                stream_buf.clear()
                print(out, end="", flush=True)
                flush_ts[0] = now

        answer = coach_chat_complete_turn(
            chat_hist,
            stream_output=stream_now,
            on_stream_chunk=_on_chunk if stream_now else None,
        )
        t_ll1 = time.perf_counter()

        if stream_now and stream_buf:
            print("".join(stream_buf), end="", flush=True)
            stream_buf.clear()
            print()

        if first_chunk_ts[0] is not None:
            round_sec = t_ll1 - first_chunk_ts[0]
        else:
            round_sec = t_ll1 - t_ll0
        ll_total += round_sec
        print(
            f"[{log_prefix}] 第 {turn} 轮模型响应耗时: {round_sec:.2f}s",
            flush=True,
        )

        if not stream_now:
            print("\n【随风听笛说】\n")
            print(answer)
            print()
        if on_answer is not None:
            try:
                on_answer(answer, turn)
            except Exception as e:
                print(f"[{log_prefix}] TTS 回调失败（已忽略）: {e}", flush=True)
        if not sys.stdin.isatty():
            break
        print(
            "—— 首轮已含完整战术快报，追问不必重跑检索；"
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
        print(f"  读 JSON + 构建快报 + 阵容检索与棋子智库: {t_prep1 - t_prep0:.2f}s")
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
