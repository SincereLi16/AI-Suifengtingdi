import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from recombo_solver import RecomboSolver

_RECOMBO_SOLVER: Optional[RecomboSolver] = None


def _get_recombo_solver() -> RecomboSolver:
    """懒加载 RecomboSolver（只在首次调用时初始化，之后复用）。"""
    global _RECOMBO_SOLVER
    if _RECOMBO_SOLVER is None:
        _RECOMBO_SOLVER = RecomboSolver()
    return _RECOMBO_SOLVER


# 在项目启动时加载 .env / 环境变量配置
load_dotenv()

_TRAIT_RAG_INDEX_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_trait_rag_index() -> Dict[str, Dict[str, Any]]:
    """
    从 data/rag_legend_traits.jsonl 读取羁绊索引（Vision 白名单）：
    {
      "斗士": { "thresholds": [2,4,6,8], "text": "..." },
      ...
    }
    """
    global _TRAIT_RAG_INDEX_CACHE
    if _TRAIT_RAG_INDEX_CACHE is not None:
        return _TRAIT_RAG_INDEX_CACHE

    base_dir = Path(__file__).resolve().parent
    path = base_dir / "data" / "rag_legend_traits.jsonl"
    if not path.exists():
        _TRAIT_RAG_INDEX_CACHE = {}
        return _TRAIT_RAG_INDEX_CACHE

    index: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            thresholds = obj.get("thresholds") or []
            if not isinstance(thresholds, list):
                thresholds = []
            index[name] = {
                "thresholds": thresholds,
                "text": obj.get("text") or "",
            }

    _TRAIT_RAG_INDEX_CACHE = index
    return _TRAIT_RAG_INDEX_CACHE


def _build_s17_vision_system_prompt(trait_names: List[str]) -> str:
    """
    基于 S17 RAG（羁绊白名单）构造 Vision system prompt，减少羁绊幻觉。
    """
    whitelist_block = ""
    if trait_names:
        whitelist_block = (
            "\n【S17 合法羁绊白名单】\n"
            "你输出的 traits[].name 必须严格从下列名称中选择（必须一字不差）。\n"
            "如果看不清/不确定，宁可不输出该羁绊，也不要猜。\n"
            + "、".join(trait_names)
            + "\n"
        )

    return f"""
你是一个高精度的《金铲铲之战》局面分析助手。
当前赛季：【天选福星·新春返场】（S17 / setId=4）。

你的任务是从截图中提取【宏观运营相关的信息】，包括：
- 玩家血量、等级、经验条
- 当前回合（例如 "2-3"）
- 羁绊（种类 + 已激活数量；不输出 level）
- 场上已上阵的棋子列表（用于估算人口并与羁绊交叉校验）
- 胜负连胜/连败情况（如果能看出来）

禁止输出商店/备战席的所有细节，只需关注当前对局决策相关的信息。
{whitelist_block}
【字段规范】
- traits: 只输出你能从 UI 羁绊栏明确看到的羁绊；每个羁绊只输出 name 与 active_units，不要输出 level。
- traits[].name: 必须完全匹配白名单中的名称。
- traits[].active_units: 当前为该羁绊贡献人数（例如 4 斗士时为 4）。
- chosen_trait: 天选之人所属羁绊的名称（从白名单中取），看不清填 null。
- player.level: 当前人口等级（整数）。必须按以下优先级推断，严禁在能推断出时直接填 null：
  1）若能看到屏幕左侧人口 UI 数字，以其为准；
  2）若人口 UI 看不清，则用「棋盘上已识别到的棋子数量 chess 的长度」作为等级（金铲铲中上阵数=人口）；
  3）若棋子数量也不确定，再结合经验条进度推断（例如经验条在 6 级常见位置则填 6）；
  只有在以上三者都无法参考时，才填 null。
- exp/exp_to_level: 经验条数值，看不清填 null。
- streak: 只允许 "win" / "lose" / null（看不出就 null）。
- chess: 只包含当前已经「上阵」在棋盘上的棋子信息，不包含备战席。
- chess[].name: 英雄名称；必须与当前 traits 一致（见下方「交叉校验」）。
- chess[].star / is_chosen / position: 同上，看不清可填 null。

【Chess 与 Traits 交叉校验】（输出前必须执行）：
你拥有 S17 赛季英雄与羁绊的对应关系知识。输出前请自检：
- 场上每个棋子的英雄，其羁绊应能覆盖当前 traits 中的至少一个（即每个 chess.name 对应的英雄，在游戏设定中应包含当前列出的羁绊之一）。
- 若某棋子名称与当前激活羁绊明显不符（例如羁绊是斗士+福星，但该位写成了其他赛季英雄），请修正为符合当前羁绊的 S17 英雄；若仅凭外观无法确定是谁，可根据当前阶段、人口、费用和已有羁绊，推断「最可能符合该羁绊的在场棋子」并填写该英雄名。
- 若某格实在无法对应到任一 S17 英雄，再将 name 设为 null。

【chosen_trait（天选之人所属羁绊）识别规则】：
- 天选之人机制：每局可有一名天选英雄，该英雄的某个羁绊贡献 +2 而非 +1。
- 判断方法：在屏幕左侧羁绊栏中，天选英雄所属的羁绊通常有「特殊发光边框 / 天选图标标记」。
- 如果能从 UI 明确看出哪个羁绊被天选之人加成（该羁绊旁有特殊标识），则在 chosen_trait 中填写该羁绊名称（必须与白名单严格匹配）。
- 如果看不出来，填 null。

【输出 JSON 模板】（必须严格输出合法 JSON；不要 Markdown；不要注释；不要多余字段；traits 中不要包含 level）：
{{
  "traits": [
    {{
      "name": "羁绊名称",
      "active_units": 2
    }}
  ],
  "chosen_trait": null,
  "player": {{
    "hp": null,
    "level": null,
    "exp": null,
    "exp_to_level": null,
    "streak": null
  }},
  "chess": [
    {{
      "name": "英雄名称",
      "star": 1,
      "is_chosen": false,
      "position": [0, 0]
    }}
  ],
  "round": null
}}

【防御性策略】
- 如果某项信息看不清或界面上没有展示，请设为 null 或空列表，不要猜。
- player.level 在人口 UI 不清时，必须先用棋子数量或经验条推断，再决定是否 null。
""".strip()


def _sanitize_s17_state(state: Dict[str, Any], trait_names: List[str]) -> Dict[str, Any]:
    """
    对 Vision 返回的 JSON 做一层清洗/校验，让 traits 能与 S17 RAG 命中：
    - 只保留白名单羁绊名，不输出 level（仅保留 name、active_units）
    """
    allowed = set(trait_names) if trait_names else None

    traits_in = state.get("traits", [])
    traits_out: List[Dict[str, Any]] = []
    if isinstance(traits_in, list):
        for t in traits_in:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not isinstance(name, str):
                continue
            name = name.strip()
            if not name:
                continue
            if allowed is not None and name not in allowed:
                continue

            active_raw = t.get("active_units")
            try:
                active_units = int(active_raw) if active_raw is not None else None
            except (TypeError, ValueError):
                active_units = None

            traits_out.append(
                {
                    "name": name,
                    "active_units": active_units,
                }
            )

    player_in = state.get("player", {})
    if not isinstance(player_in, dict):
        player_in = {}

    def _to_int_or_none(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except (TypeError, ValueError):
            return None

    streak = player_in.get("streak")
    if streak not in ("win", "lose"):
        streak = None

    # 清洗棋子信息（用于后续推理/展示）
    chess_in = state.get("chess", [])
    chess_out: List[Dict[str, Any]] = []
    if isinstance(chess_in, list):
        for c in chess_in:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            if isinstance(name, str):
                name = name.strip() or None
            else:
                name = None
            star_raw = c.get("star")
            try:
                star = int(star_raw) if star_raw is not None else None
            except (TypeError, ValueError):
                star = None
            is_chosen = c.get("is_chosen")
            if not isinstance(is_chosen, bool):
                is_chosen = None
            pos = c.get("position")
            if (
                isinstance(pos, list)
                and len(pos) == 2
                and all(isinstance(x, (int, float)) for x in pos)
            ):
                position = [int(pos[0]), int(pos[1])]
            else:
                position = None
            chess_out.append(
                {
                    "name": name,
                    "star": star,
                    "is_chosen": is_chosen,
                    "position": position,
                }
            )

    # 人口等级：若模型未给出且棋盘有棋子，用棋子数量回填（金铲铲上阵数=人口）
    level = _to_int_or_none(player_in.get("level"))
    if level is None and len(chess_out) > 0:
        level = len(chess_out)

    # 天选之人所属羁绊：必须在白名单内
    chosen_trait_raw = state.get("chosen_trait")
    if isinstance(chosen_trait_raw, str):
        chosen_trait_raw = chosen_trait_raw.strip()
        chosen_trait = chosen_trait_raw if (allowed is None or chosen_trait_raw in allowed) and chosen_trait_raw else None
    else:
        chosen_trait = None

    cleaned = {
        "traits": traits_out,
        "chosen_trait": chosen_trait,
        "player": {
            "hp": _to_int_or_none(player_in.get("hp")),
            "level": level,
            "exp": _to_int_or_none(player_in.get("exp")),
            "exp_to_level": _to_int_or_none(player_in.get("exp_to_level")),
            "streak": streak,
        },
        "chess": chess_out,
        "round": state.get("round") if isinstance(state.get("round"), str) else None,
    }
    return cleaned


@dataclass
class OpenRouterConfig:
    """
    OpenRouter 多模态模型配置（默认使用 Google Gemini Flash）
    """

    # OpenRouter 兼容 OpenAI SDK
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # 默认都用同一个 Flash 模型；如需单独指定，可在 .env 中设置 OPENROUTER_VISION_MODEL / OPENROUTER_TEXT_MODEL
    vision_model: str = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash")
    text_model: str = os.getenv("OPENROUTER_TEXT_MODEL", vision_model)

    def create_client(self) -> OpenAI:
        if not self.api_key:
            raise RuntimeError("请先在环境变量 OPENROUTER_API_KEY 中配置 OpenRouter API Key")

        # 去掉首尾空白，并检查是否只包含 ASCII 字符（HTTP 头部只能是 ASCII）
        key = str(self.api_key).strip()
        try:
            key.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                f"API Key 中包含非 ASCII 字符，请重新从 OpenRouter 控制台复制一遍纯英文/数字的 Key。当前值为: {repr(key)}"
            ) from e

        return OpenAI(
            base_url=self.base_url,
            api_key=key,
        )


def _extract_content_text(raw_content: Any) -> str:
    """
    从 SDK 返回的 message.content 中提取纯文本：
    - 有的模型返回字符串
    - 有的模型返回由 Text/Image 分片组成的列表
    """
    # 文本字符串直接返回
    if isinstance(raw_content, str):
        return raw_content

    # 多模态场景下，可能是 ContentPart 列表
    text_parts = []
    try:
        for part in raw_content:
            # 新版 SDK 里通常有 .type 和 .text
            if getattr(part, "type", None) == "text":
                text_obj = getattr(part, "text", None)
                # text_obj 可能有 .value，也可能本身就是字符串
                if hasattr(text_obj, "value"):
                    text_parts.append(text_obj.value)
                elif isinstance(text_obj, str):
                    text_parts.append(text_obj)
    except TypeError:
        # raw_content 不是可迭代或者结构不符时，退回 str()
        return str(raw_content)

    return "".join(text_parts).strip()


def call_gemini_vision_extract_state(
    image_path: str, client: OpenAI, config: OpenRouterConfig
) -> Dict[str, Any]:
    """
    调用 OpenRouter 上的 Gemini Flash 多模态模型，从金铲铲截图中抽取「结构化局面信息」。

    返回一个 Python dict（内部是 JSON 结构），后续作为策略阶段的输入。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 读取图片，适当缩放并转为 JPEG，再编码为 base64，由多模态接口解析
    t_upload_start = time.perf_counter()
    with Image.open(image_path) as img:
        max_side = 1080
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buf = BytesIO()
        # 压缩为 JPEG，质量 85，显著减小体积，一般不影响金铲铲 UI 识别
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        image_bytes = buf.getvalue()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    t_upload_end = time.perf_counter()

    trait_index = _load_trait_rag_index()
    trait_names = sorted(trait_index.keys())
    system_prompt = _build_s17_vision_system_prompt(trait_names)

    # 使用 OpenAI SDK 兼容调用 OpenRouter 多模态模型
    # temperature=0：贪婪解码，让模型每次对同一图片给出稳定的最高概率输出，
    # 避免在白名单内相近羁绊名（如"神射手"/"枪术"、"福星"/"忍者"）之间随机选错
    t_api_start = time.perf_counter()
    response = client.chat.completions.create(
        model=config.vision_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # 使用 data URL 的方式传输本地图片
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "请严格按系统提示，从图片中识别信息并仅输出 JSON。",
                    },
                ],
            },
        ],
    )
    t_api_end = time.perf_counter()

    raw_content = response.choices[0].message.content
    content_text = _extract_content_text(raw_content)

    # 有些模型会自动包一层 ```json ... ``` 的 Markdown 代码块，这里做一次剥壳
    t_parse_start = time.perf_counter()
    cleaned = content_text.strip()
    if cleaned.startswith("```"):
        # 去掉开头的 ``` 或 ```json
        cleaned = cleaned.lstrip("`")
        # lstrip 之后可能变成 'json\n{...}' 或 '\n{...}'
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json") :]
        # 去掉第一行剩余到换行
        cleaned = cleaned.split("\n", 1)[-1]
        # 去掉结尾的 ```
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    # cleaned 应该是一个纯 JSON 字符串
    try:
        state = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # 出错时把原始内容也打印出来方便排查
        raise RuntimeError(
            f"模型返回内容不是合法 JSON：{content_text}"
        ) from e
    t_parse_end = time.perf_counter()

    print(f"[耗时统计] 图片读取+base64 编码耗时: {t_upload_end - t_upload_start:.3f} 秒")
    print(f"[耗时统计] 视觉模型 API 调用耗时: {t_api_end - t_api_start:.3f} 秒")
    print(f"[耗时统计] 视觉结果解析(JSON)耗时: {t_parse_end - t_parse_start:.3f} 秒")
    if isinstance(state, dict):
        state = _sanitize_s17_state(state, trait_names)
    return state


def call_gemini_text_strategy(
    state: Dict[str, Any], client: OpenAI, config: OpenRouterConfig
) -> str:
    """
    调用 OpenRouter 上的 Gemini Flash 文本能力，根据「局面 JSON」输出对局策略建议。
    """
    system_prompt = """
   你是《金铲铲之战》顶尖玩家“随风听笛”，正在语音带你的好兄弟哈基星上分。
    现在是【天选福星】新春返场赛季，节奏极快，你说话必须一针见血，带点高手的不耐烦和冷静。

    【你拿到的 JSON 数据说明】：
    现在的 JSON 只包含宏观运营相关信息：
    - traits：当前激活的羁绊及已激活数量（name、active_units）。
    - chosen_trait：天选之人所属羁绊（Vision 识别，可信）。
    - player.hp：玩家血量。
    - player.level：当前人口等级。
    - player.exp / player.exp_to_level：当前经验以及升下一级还差多少。
    - player.streak：连胜("win") / 连败("lose") / null。
    - round：当前阶段-回合，例如 "2-3"。
    - inferred_chess：由逻辑引擎（回溯+剪枝）从羁绊数据推断出的在场棋子，可信度高：
      - must_have：必然在场的英雄（每种合法组合都包含）
      - top_candidates：按出现频率排序的候选英雄列表

    【你的决策原则】（只做宏观运营）：
    - 优先依据 traits 和 inferred_chess.must_have 来判断阵容走向，不要依赖其他不确定信息。
    - 请你显式判断：
      - 当前有哪些羁绊已经明显成型（等级比较高、数量占比大），应该作为主体系去围绕。
      - 当前有哪些羁绊只是挂件（一层或零散），可以视情况后续卖掉或者合并到主体系里。
    - 重点回答：
      - 这局目前应该打连胜还是接受连败赌福星？
      - 这一两回合是拉人口还是攒经济？（即使看不到具体金币，也要基于阶段和血量给出偏向）
    - 如果某些字段是 null（比如看不到血量/经验），你就直接说“这条看不到”，但依然要基于能看到的信息给出决策。

    你的性格：
    - 拒绝废话：别说“根据数据分析”，直接告诉我买谁、卖谁、合什么。
    - 毒舌专业：如果兄弟玩得烂，先狠狠拿他出气，但一定要给出翻盘的神来之笔。
    - 绝活哥：对【天选】极其敏感，一眼就能看出这局是该走连胜还是走连败福星。

    你的任务：
    我会给你截图信息抽取后的 JSON。你只需要用【1 句话】说出这一局“接下来最关键的一步”该怎么做。

    输出要求：
    - 只说最关键的一条决策，不要复述 JSON 细节，不要描述当前局面参数。
    - 严禁列清单、逐条分析，只给一个一锤定音的操作建议即可。

    约束条件：
    - 严禁结构化输出（不要 1.2.3.，不要加粗标题）。
    - 语言要极其口语化，像在网吧坐我旁边说话一样。
    - 可以顺嘴提到“天选适配度”，但别上来就背论文。
    """.strip()

    user_content = (
        "下面是当前对局的结构化信息（JSON）：\n"
        f"{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
        "请根据上面的信息给出你的对局策略建议。"
    )

    t_api_start = time.perf_counter()
    response = client.chat.completions.create(
        model=config.text_model,
        temperature=0.3,  # 策略建议允许少量创意，但保持分析稳定
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    t_api_end = time.perf_counter()

    raw_content = response.choices[0].message.content
    text = _extract_content_text(raw_content)

    print(f"[耗时统计] 文本策略模型 API 调用耗时: {t_api_end - t_api_start:.3f} 秒")

    return text


def infer_chess_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    基于 Vision 识别到的 traits + player.level + chosen_trait，
    调用 RecomboSolver 回溯+剪枝引擎推断在场棋子组合。

    返回 RecomboSolver.solve() 的完整结果 dict，或 None（当缺少必要信息时）。
    """
    trait_counts: Dict[str, int] = {}
    for t in state.get("traits", []):
        name = t.get("name")
        units = t.get("active_units")
        if name and isinstance(units, int) and units > 0:
            trait_counts[name] = units

    if not trait_counts:
        print("[RecomboSolver] 跳过推断：未识别到任何有效羁绊。")
        return None

    player_level = state.get("player", {}).get("level")
    if not isinstance(player_level, int) or player_level <= 0:
        print("[RecomboSolver] 跳过推断：player.level 未知，无法确定牌池费用范围。")
        return None

    chosen_trait: Optional[str] = state.get("chosen_trait")

    solver = _get_recombo_solver()
    result = solver.solve(
        trait_counts=trait_counts,
        player_level=player_level,
        chosen_trait=chosen_trait,
        max_results=300,
        timeout_ms=2000.0,
    )
    return result


def _print_inference_result(result: Dict[str, Any]) -> None:
    """格式化输出 RecomboSolver 推断结果。"""
    timeout_str = " [超时提前终止，结果仅供参考]" if result.get("timeout") else ""
    print(f"\n  有效组合数: {result['total_valid']}{timeout_str}")
    print(f"  推断耗时:   {result['elapsed_ms']:.1f} ms")

    if result.get("note"):
        print(f"\n  [提示] {result['note']}")
        return

    must_have = result.get("must_have", [])
    priority = result.get("priority", [])
    top_combos = result.get("top_combos", [])

    if must_have:
        print(f"\n  ★ 必然在场（每种组合均含）: {must_have}")
    else:
        print("\n  ★ 必然在场: 无唯一必选英雄")

    print(f"\n  优先候选英雄（TOP {min(15, len(priority))}，按出现频率降序）:")
    for p in priority[:15]:
        mark = "  [必须]" if p["name"] in must_have else "        "
        print(
            f"  {mark} ★{p['cost']} {p['name']:<8}"
            f"  出现率={p['freq']*100:5.1f}%"
            f"  羁绊={p['traits']}"
        )

    if top_combos:
        print(f"\n  Top 3 完整活跃英雄组合（不含挂件）:")
        for i, c in enumerate(top_combos[:3], 1):
            print(f"    {i}. 英雄={c['names']}")
            print(f"       羁绊满足={c['trait_summary']}  得分={c['score']}")


def analyze_battle(image_path: str = "1.png") -> None:
    """
    完整对局分析链路：
      Step 1 — Vision 识别截图 → 结构化局面 JSON（traits / chosen_trait / level）
      Step 2 — RecomboSolver 回溯推断 → 在场棋子组合（must_have / candidates / priority）
      Step 3 — 文本大模型 → 宏观策略建议（基于 traits + 推断棋子）
    """
    config = OpenRouterConfig()
    client = config.create_client()

    t_start = time.perf_counter()

    # ── Step 1: Vision 识别 ──────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Vision 模型识别截图局面...")
    print("=" * 60)
    t1s = time.perf_counter()
    state = call_gemini_vision_extract_state(image_path, client, config)
    t1e = time.perf_counter()

    print("\n[Vision 识别结果]")
    print(json.dumps(state, ensure_ascii=False, indent=2))

    # ── Step 2: RecomboSolver 推断在场棋子 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: RecomboSolver 回溯+剪枝推断在场棋子...")
    print("=" * 60)
    trait_summary = ", ".join(
        f"{t['name']}×{t['active_units']}"
        for t in state.get("traits", [])
        if t.get("active_units")
    )
    level = state.get("player", {}).get("level")
    chosen = state.get("chosen_trait")
    print(f"  输入羁绊: {trait_summary or '(无)'}")
    print(f"  人口等级: {level}")
    print(f"  天选羁绊: {chosen or '(未识别)'}")

    t2s = time.perf_counter()
    inference = infer_chess_from_state(state)
    t2e = time.perf_counter()

    if inference:
        _print_inference_result(inference)
        # 将推断结果写入 state 供策略模型参考
        state["inferred_chess"] = {
            "must_have": inference.get("must_have", []),
            "top_candidates": [p["name"] for p in inference.get("priority", [])[:10]],
        }
    else:
        print("  [跳过] 无法进行棋子推断。")

    # ── Step 3: 文本策略建议 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: 文本大模型生成策略建议...")
    print("=" * 60)
    # 构造传给策略模型的 state 副本：
    # - 去掉 Vision 猜测的 chess[]（3D 模型识别不可靠，会误导策略）
    # - 保留 inferred_chess（RecomboSolver 推断的可靠结果）
    strategy_state = {k: v for k, v in state.items() if k != "chess"}
    t3s = time.perf_counter()
    strategy = call_gemini_text_strategy(strategy_state, client, config)
    t3e = time.perf_counter()

    print("\n[策略建议]")
    print(strategy)

    t_end = time.perf_counter()
    print("\n" + "=" * 60)
    print("耗时统计")
    print("=" * 60)
    print(f"  Step1 Vision 识别:   {t1e - t1s:.3f} 秒")
    print(f"  Step2 RecomboSolver:   {t2e - t2s:.3f} 秒")
    print(f"  Step3 策略建议:      {t3e - t3s:.3f} 秒")
    print(f"  整体总耗时:          {t_end - t_start:.3f} 秒")


if __name__ == "__main__":
    # 默认读取当前目录下的 1.png
    analyze_battle("1.png")
