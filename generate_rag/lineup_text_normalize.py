# -*- coding: utf-8 -*-
"""
金铲铲攻略文案统一：阶段记法、星级、D 牌、羁绊简写、常见错别字等。
供 build_rag_lineup_v1 与批量修复 jsonl 使用。
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

_CN_DIGIT = {
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}


def normalize_lineup_strategy_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return s
    t = s

    # --- 阶段：数字杠数字 → 数字-数字（多轮，防嵌套）---
    for _ in range(8):
        n = re.sub(r"(\d+)杠(\d+)", r"\1-\2", t)
        if n == t:
            break
        t = n

    # 中文单数字 杠 数字 / 数字 杠 中文单数字 / 中文 杠 中文（如 三杠五、5杠一、3杠五）
    def sub_cn_cn(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return f"{_CN_DIGIT.get(a, a)}-{_CN_DIGIT.get(b, b)}"

    t = re.sub(r"([一二三四五六七八九])杠([一二三四五六七八九十])", sub_cn_cn, t)
    t = re.sub(
        r"(\d+)杠([一二三四五六七八九十])",
        lambda m: f"{m.group(1)}-{_CN_DIGIT.get(m.group(2), m.group(2))}",
        t,
    )
    t = re.sub(
        r"([一二三四五六七八九])杠(\d+)",
        lambda m: f"{_CN_DIGIT.get(m.group(1), m.group(1))}-{m.group(2)}",
        t,
    )

    # 口语「正常杠2」= 2-2 阶段
    for a, b in [("正常杠2", "正常2-2"), ("中期杠2", "中期2-2")]:
        t = t.replace(a, b)

    # --- 星级：先「双三」再「三星」，避免「双三星」类歧义；最后「追三」仅作动词（不碰「追三个」）---
    t = t.replace("双三", "双3星")
    t = t.replace("三星", "3星")
    t = re.sub(r"追三(?![星3个])", "追3星", t)
    t = t.replace("搜三", "搜3星")
    t = t.replace("偷三", "偷3星")

    t = t.replace("全二星", "全2星")
    t = t.replace("全二", "全2星")
    t = t.replace("二星", "2星")
    t = t.replace("一星", "1星")

    # --- D 牌系列（长词优先）---
    _d_pairs = [
        ("d3星", "D3星"),
        ("花费金币d", "花费金币D"),
        ("不d牌", "不D牌"),
        ("不d到", "不D到"),
        ("不d", "不D"),
        ("d牌", "D牌"),
        ("d到", "D到"),
        ("d空", "D空"),
        ("d满", "D满"),
        ("d干", "D干"),
        ("大d", "大D"),
        ("小d", "小D"),
        ("慢d", "慢D"),
        ("全d", "全D"),
        ("猛d", "猛D"),
        ("硬d", "硬D"),
        ("就d", "就D"),
        ("先d", "先D"),
        ("只d", "只D"),
        ("多d", "多D"),
        ("再d", "再D"),
        ("去d", "去D"),
        ("可以d", "可以D"),
        ("尽量d", "尽量D"),
        ("花金币d", "花金币D"),
        ("不花金币d", "不花金币D"),
    ]
    for a, b in _d_pairs:
        t = t.replace(a, b)

    # --- 羁绊 / 费用简写（长词优先）---
    _trait_pairs = [
        ("三德玛西亚", "3德玛西亚"),
        ("三诺克萨斯", "3诺克萨斯"),
        ("三德玛", "3德玛"),
        ("三诺克", "3诺克"),
        ("三比尔", "3比尔"),
        ("三艾欧", "3艾欧"),
        ("三比", "3比"),
        ("三皮", "3皮"),
        ("三约", "3约"),
        ("三费", "3费"),
        ("五艾欧", "5艾欧"),
        ("五比", "5比"),
        ("五费", "5费"),
        ("四费", "4费"),
        ("二费", "2费"),
        ("开三艾欧", "开3艾欧"),
        ("七德玛", "7德玛"),
        ("五德玛", "5德玛"),
        ("六约", "6约"),
        ("八约", "8约"),
        ("六斗", "6斗"),
        ("上八", "上8"),
        ("上九", "上9"),
        ("上七", "上7"),
        ("上六", "上6"),
        ("上五", "上5"),
        ("拉八", "拉8"),
        ("拉九", "拉9"),
        ("拉七", "拉7"),
        ("拉六", "拉6"),
    ]
    for a, b in _trait_pairs:
        t = t.replace(a, b)

    # --- C 位 / 主C ---
    t = re.sub(r"(?<![A-Za-z])c位", "C位", t, flags=re.IGNORECASE)
    t = t.replace("双c", "双C").replace("双C位", "双C位")
    t = t.replace("主c", "主C")
    t = t.replace("副c", "副C")
    t = t.replace("对c", "对C")
    t = re.sub(r"([对防])着c", lambda m: m.group(1) + "着C", t)
    t = t.replace("敌方c", "敌方C").replace("对面c", "对面C")

    # --- 错别字 ---
    t = t.replace("安培撒", "安蓓萨")
    t = t.replace("安倍萨", "安蓓萨")
    t = t.replace("尼蔻", "妮蔻")
    t = t.replace("合破甲", "和破甲")
    t = t.replace("巨人补手", "巨人捕手")
    t = t.replace("薄雾。", "薄暮。").replace("或者薄雾", "或者薄暮")
    t = t.replace("妮寇", "妮蔻")
    t = t.replace("塞纳", "赛娜")
    # 「2刚1」多为「2-1」笔误
    t = re.sub(r"(\d)刚(\d)", r"\1-\2", t)

    # 统一空格：英文与中文之间可保留，不强制

    return t


def _walk_strings(obj: Any, fn) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_walk_strings(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk_strings(v, fn) for k, v in obj.items()}
    return obj


_V1_TEXT_KEYS = frozenset(
    {
        "early_game",
        "tempo",
        "equip_strategy",
        "positioning",
        "name",
        "name_short",
        "traits",
        "quality",
        "arch_type",
        "cost_tier",
    }
)


def normalize_v1_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """对 lineup_v1 记录中需要统一的字符串字段做规范化。"""
    out = copy.deepcopy(rec)

    for k in list(out.keys()):
        if k in _V1_TEXT_KEYS and isinstance(out[k], str):
            out[k] = normalize_lineup_strategy_text(out[k])

    hp = out.get("hex_priority")
    if isinstance(hp, list):
        out["hex_priority"] = [normalize_lineup_strategy_text(x) if isinstance(x, str) else x for x in hp]
    ha = out.get("hex_alternative")
    if isinstance(ha, list):
        out["hex_alternative"] = [normalize_lineup_strategy_text(x) if isinstance(x, str) else x for x in ha]

    bl = out.get("build_levels")
    if isinstance(bl, list):
        new_bl: List[Dict[str, Any]] = []
        for row in bl:
            if not isinstance(row, dict):
                new_bl.append(row)
                continue
            r = dict(row)
            if isinstance(r.get("pieces"), str):
                r["pieces"] = normalize_lineup_strategy_text(r["pieces"])
            new_bl.append(r)
        out["build_levels"] = new_bl

    # 新结构：strategy / builds
    st = out.get("strategy")
    if isinstance(st, dict):
        ns = dict(st)
        for k in ("early", "mid"):
            if isinstance(ns.get(k), str):
                ns[k] = normalize_lineup_strategy_text(ns[k])
        out["strategy"] = ns

    bd = out.get("builds")
    if isinstance(bd, dict):
        nb: Dict[str, Any] = {}
        for k, v in bd.items():
            if isinstance(v, str):
                nb[str(k)] = normalize_lineup_strategy_text(v)
            else:
                nb[str(k)] = v
        out["builds"] = nb

    return out


def normalize_lineup_v1_jsonl_file(path: Path) -> int:
    """就地读入 jsonl，规范化后写回。返回条数。"""
    path = path.resolve()
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        o = json.loads(line)
        o = normalize_v1_record(o)
        out_lines.append(json.dumps(o, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines)
