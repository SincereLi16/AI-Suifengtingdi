# -*- coding: utf-8 -*-
"""
Pipeline 对局 JSON → 战术快报（语义压缩）→ 阵容 RAG 检索 → Gemini Flash 回答。

用法示例：
  python gemini_v1.py
  # 默认：若 runs/battle_pipeline_v3_out 已有 *_summary.json 则读缓存；否则静默跑 pipeline。
  python gemini_v1.py --summary-json runs/.../01-a_summary.json  # 跳过 pipeline，仅调试
  python gemini_v1.py --img-dir "对局截图" -q "我这把该先合什么？"
  python gemini_v1.py --no-rag   # 关闭本地阵容 RAG，只喂战术快报 + 问题

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
from typing import Any, Dict, List, Optional, Tuple

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


def _google_gemini_generate(
    *,
    system_prompt: str,
    user_text: str,
    model: str,
    temperature: float = 0.35,
    timeout_s: float = 120.0,
) -> str:
    """Gemini Developer API generateContent（REST），不经 OpenRouter。"""
    key = _google_gemini_key()
    if not key:
        raise RuntimeError("缺少 GEMINI_API_KEY / GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
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
        name = str(r.get("best") or "?")
        conf = str(r.get("confidence") or "")
        if conf == "low":
            name = f"{name}?"
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
        lines.append(f"{name} 站位{loc_s} | 装备:{eq_str}")
    return lines


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

    bonds_line = str(gt.get("merged_bonds_one_line") or "").strip()
    tm = gt.get("trait_count_max") if isinstance(gt, dict) else None
    if isinstance(tm, dict) and tm:
        trait_summary = "，".join(f"{int(n)}{t}" for t, n in sorted(tm.items(), key=lambda x: int(x[1]), reverse=True)[:12])
    else:
        trait_summary = bonds_line or "(无)"

    board_lines = _format_board_lines(summary)
    board_block = "\n".join(f"  - {x}" for x in board_lines) if board_lines else "  (无)"

    return f"""[当前局势]
阶段 {phase or '?'} | 等级 {level or '?'} | 经验 {exp or '?'} | 金币 {gold or '?'} | 胜负 {streak or '?'}

[羁绊]
{trait_summary}

[棋盘与装备]
（「站位」后括号为棋盘行,列）
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


def _extract_keywords_for_rag(summary: Dict[str, Any]) -> List[str]:
    """从 summary 抽可用于匹配阵容攻略的关键词（羁绊名、英雄名）。"""
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
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    简易检索：关键词在阵容 doc['text'] 中的命中数（无需向量库）。
    返回 (拼好的 RAG 文本块, lineup_id 列表, 终端展示用元数据列表)。
    """
    docs = _load_lineup_docs(rag_path)
    if not docs:
        return ("（阵容 RAG 库未加载或为空。）", [], [])

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


def _coach_system_prompt() -> str:
    return """
你是《金铲铲之战》顶尖玩家随风听笛，正在网吧叼着烟、开着语音带你的菜鸡好兄弟哈基星上分。
你不仅能看到他的实时 JSON 战报，还能听到他的提问。
你脑子里装着全版本的最强阵容 RAG 库，说话一针见血，带着高手那种带不动你的不耐烦和极致的冷静。

【你的输入结构说明】：
你会收到一段结构化文本：
1. [哈基星问题]：来自哈基星的弱智提问，请你根据他的问题＋下列对局信息回答。
2. [当前局势]：关注阶段（如5-5）和金币，这是判断该「梭哈」还是「存钱」的唯一依据。
3. [羁绊]：这是灵魂。你要一眼看出哪些是核心（如3比尔吉沃特/3诺克萨斯），哪些是路边捡的垃圾挂件。
4. [棋盘与装备]：关注核心输出（如4行后排的瑞兹/女枪）的装备是否对劲，前排（1行）肉装是否顶得住。
5. [左侧装备栏散件]：这是你唯一的变数，如果散件能合成关键装，立刻开喷让他合。
6. [阵容RAG库]：通过关键词命中的本地阵容库，用来对「阵容名 / 运营节奏」做参照。

【你的处理逻辑】：
- 意图识别：如果哈基星的问题是具体的（合什么装备、D不D牌），必须优先回答问题；如果问题是含糊的（怎么办、玩什么），则基于阵容RAG库与战术快报进行分析；
- 运营定性：只看羁绊和关键棋子的星级 / 装备，匹配相近的阵容RAG，并基于RAG库指导哈基星下一步的行动；
- 资源优化：基于阵容RAG，分析当前金币与装备散件是否最优利用，若无则激情开喷。

【你的分析逻辑】：
1. 抓大腿：扫一眼 JSON，确定场上谁是真正的爹。优先看 2 星及以上的核心羁绊卡，不要被 1 星的高费卡晃了眼。
2. 找缺口：对比【阵容 RAG 库】，看场上主 C 是不是对的人，装备是不是对的件。
3. 装备铁律：主 C 三件套没满前，散件全是主 C 的！除非主 C 只有 1 星且烂泥扶不上墙，否则绝不分给外人。
4. 战力即正义：5 阶段以后，所有不转成战力的金币和散件都是在等死。

【你的性格：随风听笛】
- 拒绝废话：严禁说建议、根据数据、分析显示，务必使用超级口语化的风格
- 毒舌专业：你可以骂哈基星操作烂，但必须告诉他下一步是卖谁、升几级、还是合什么。
- 战术直觉：看到 9 级 50 块还带一堆 1 星卡，你就要原地爆炸。

【输出要求】：
- 严禁任何 Markdown 标题、1.2.3. 结构、或列表。
- 严禁复述 JSON 里的参数。
- 只有【1~2 句话】，必须是纯口语，像在网吧叼着烟戴着耳机开黑。
- 语气示例：「哈基星你这个蠢货，又他妈要存50块钱买棺材板啊？赶紧全D了找 2 星瑞兹，找不到你赶紧卸载游戏回去玩你那泳装蓝梦吧。」
""".strip()


def call_gemini_coach(
    tactical_brief: str,
    rag_block: str,
    user_question: str,
    *,
    model: str | None = None,
) -> str:
    """
    单轮对话：system = 随风听笛人设；user = 一段字符串，按顺序拼接
    【哈基星问题】+【实时战术快报】+【阵容 RAG 库·命中摘录】（RAG 为本地 jsonl 检索出的攻略长文，非向量嵌入）。
    OpenRouter / Google 均为 messages[0]=system、messages[1]=user。
    """
    user_content = (
        "【哈基星问题】\n"
        f"{user_question.strip()}\n\n"
        "【实时战术快报（JSON 识别结果）】\n"
        f"{tactical_brief.strip()}\n\n"
        "【阵容 RAG 库·命中摘录】\n"
        f"{rag_block.strip()}"
    )
    backend = _resolve_chat_backend()
    _, _, default_or_model = _openrouter_env()
    coach_temp = 0.35
    if backend == "google":
        gm = (model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")).strip()
        print(f"[gemini_v1] LLM 后端: google  模型: {gm}")
        return _google_gemini_generate(
            system_prompt=_coach_system_prompt(),
            user_text=user_content,
            model=gm,
            temperature=coach_temp,
        )
    om = model or default_or_model
    print(f"[gemini_v1] LLM 后端: openrouter  模型: {om}")
    return _openrouter_chat_completion(
        messages=[
            {"role": "system", "content": _coach_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        model=om,
        temperature=coach_temp,
    )


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pipeline JSON → 战术快报 + RAG → Gemini 教练",
        epilog="默认优先使用 runs/battle_pipeline_v3_out 下已有 *_summary.json（跳过 pipeline）；\n"
        "无缓存时再跑 pipeline（stderr 转圈；输入问题时暂停刷新）。\n"
        "快报预览后直接请求教练。--force-pipeline 可强制重新识别。--no-rag 可关闭阵容 RAG。--summary-json 指定单文件调试。",
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
    ap.add_argument("--rag-top-k", type=int, default=3, help="检索阵容条数")
    ap.add_argument(
        "--no-rag",
        action="store_true",
        help="不检索、不注入 rag_lineup_lineup.jsonl，仅用战术快报 + 问题调教练（对比带 RAG 效果）",
    )
    ap.add_argument(
        "--pipeline-verbose",
        action="store_true",
        help="将 pipeline 子进程的 stdout/stderr 原样打到终端（关闭静默与进度条；调试用）",
    )
    args = ap.parse_args()

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
            io_lock = threading.Lock()
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
    t_prep0 = time.perf_counter()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    brief = build_tactical_brief(summary)
    if args.no_rag:
        rag_block = "（本回合未注入本地阵容 RAG，仅依据战术快报与问题回答。）"
        rag_ids = []
        rag_meta = []
    else:
        rag_block, rag_ids, rag_meta = retrieve_lineup_rag(
            summary, args.rag_lineup, top_k=max(1, int(args.rag_top_k))
        )
    t_prep1 = time.perf_counter()

    print("\n" + "=" * 60)
    print("战术快报（预览）")
    print("=" * 60)
    print(brief[:2000] + ("…" if len(brief) > 2000 else ""))
    print("\n" + "=" * 60)
    print("本地阵容 RAG 命中（与下方送入模型的条目一致）")
    print("=" * 60)
    if args.no_rag:
        print("（已跳过：使用了 --no-rag，未检索 jsonl）")
    elif not rag_meta:
        print("（无命中或 RAG 库为空）")
    else:
        for m in rag_meta:
            print(
                f"lineup_id={m.get('lineup_id', '')} | "
                f"评级={m.get('quality', '')} | "
                f"赛季={m.get('season', '')} | "
                f"匹配分≈{m.get('match_score', 0)} | "
                f"阵容名：{m.get('name', '')}"
            )
    print("=" * 60)

    print("\n正在请求 LLM…\n")
    t_ll0 = time.perf_counter()
    answer = call_gemini_coach(brief, rag_block, question)
    t_ll1 = time.perf_counter()

    print("\n【教练回答】\n")
    print(answer)
    print()
    print("[gemini_v1] 本回合步骤耗时")
    if args.no_rag:
        print(f"  读 JSON + 构建快报（无 RAG）: {t_prep1 - t_prep0:.2f}s")
    else:
        print(f"  读 JSON + 构建快报 + RAG: {t_prep1 - t_prep0:.2f}s")
    print(f"  LLM: {t_ll1 - t_ll0:.2f}s")
    print()


if __name__ == "__main__":
    main()
