# -*- coding: utf-8 -*-
"""
从 data/chess_equip_name_short.json 将 name_short 写入：
  - data/rag_core_chess.jsonl（按 chess_name 匹配）
  - data/rag_legend_equip.jsonl（按 name 匹配）

映射由 rag_chess_equip_name_list.txt（用户维护的「全称 - 简称」）同步生成；
也可直接编辑 chess_equip_name_short.json。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data"
DEFAULT_MAP = DATA / "chess_equip_name_short.json"


def parse_name_list_txt(path: Path) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """解析清单：有「 - 」则为 全称→简称；无则仅占位。玛尔扎哈- - 蚂蚱 会规范为 玛尔扎哈→蚂蚱。"""
    chess: Dict[str, Optional[str]] = {}
    equip: Dict[str, Optional[str]] = {}
    mode: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("金铲铲"):
            continue
        if "棋子名称" in line and "共" in line:
            mode = "chess"
            continue
        if "装备名称" in line and "共" in line:
            mode = "equip"
            continue
        if mode == "chess":
            if " - " in line:
                left, right = line.split(" - ", 1)
                left = left.rstrip("-").strip()
                right = right.strip()
                if left:
                    chess[left] = right
            else:
                chess[line] = None
        elif mode == "equip":
            if " - " in line:
                left, right = line.split(" - ", 1)
                left = left.rstrip("-").strip()
                right = right.strip()
                if left:
                    equip[left] = right
            else:
                equip[line] = None
    return chess, equip


def build_short_json_from_list(list_path: Path) -> Dict[str, Any]:
    chess, equip = parse_name_list_txt(list_path)
    return {
        "chess": {k: v for k, v in chess.items() if v},
        "equip": {k: v for k, v in equip.items() if v},
    }


def _insert_after(d: Dict[str, Any], after_key: str, new_key: str, new_val: str) -> None:
    keys = list(d.keys())
    if after_key not in keys:
        d[new_key] = new_val
        return
    i = keys.index(after_key) + 1
    rest = {k: d.pop(k) for k in keys[i:]}
    d[new_key] = new_val
    for k, v in rest.items():
        d[k] = v


def apply_core_chess(path: Path, chess_map: Dict[str, str]) -> Tuple[int, int]:
    n_alias = 0
    n_lines = 0
    out_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        n_lines += 1
        o.pop("name_short", None)
        cn = (o.get("chess_name") or "").strip()
        if cn and cn in chess_map:
            _insert_after(o, "chess_name", "name_short", chess_map[cn])
            n_alias += 1
        out_lines.append(json.dumps(o, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return n_lines, n_alias


def apply_legend_equip(path: Path, equip_map: Dict[str, str]) -> Tuple[int, int]:
    n_alias = 0
    n_lines = 0
    out_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        n_lines += 1
        o.pop("name_short", None)
        name = (o.get("name") or "").strip()
        if name and name in equip_map:
            _insert_after(o, "name", "name_short", equip_map[name])
            n_alias += 1
        out_lines.append(json.dumps(o, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return n_lines, n_alias


def main() -> None:
    ap = argparse.ArgumentParser(description="将简称写入 rag_core_chess / rag_legend_equip")
    ap.add_argument(
        "--map-json",
        type=Path,
        default=DEFAULT_MAP,
        help="chess / equip 简称映射 JSON",
    )
    ap.add_argument(
        "--from-list",
        type=Path,
        default=None,
        help="若指定，先从该清单解析并覆盖写入 --map-json，再应用到 jsonl",
    )
    ap.add_argument("--core", type=Path, default=DATA / "rag_core_chess.jsonl")
    ap.add_argument("--equip", type=Path, default=DATA / "rag_legend_equip.jsonl")
    args = ap.parse_args()

    if args.from_list is not None:
        data = build_short_json_from_list(args.from_list.resolve())
        args.map_json.parent.mkdir(parents=True, exist_ok=True)
        args.map_json.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"已写入映射 {args.map_json}（棋子 {len(data['chess'])} 条，装备 {len(data['equip'])} 条）")

    mp = json.loads(args.map_json.read_text(encoding="utf-8"))
    chess_map = mp.get("chess") or {}
    equip_map = mp.get("equip") or {}

    t1, a1 = apply_core_chess(args.core.resolve(), chess_map)
    t2, a2 = apply_legend_equip(args.equip.resolve(), equip_map)
    print(f"rag_core_chess.jsonl：{t1} 条，写入 name_short {a1} 条")
    print(f"rag_legend_equip.jsonl：{t2} 条，写入 name_short {a2} 条")


if __name__ == "__main__":
    main()
