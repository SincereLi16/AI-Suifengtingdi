"""
recombo_solver.py — 金铲铲棋子组合逆推引擎（回溯 + 双重剪枝）

算法：回溯 + 双重剪枝（后缀不足剪枝 + 超标剪枝）

设计说明
--------
  - player_level 代表棋盘上的总棋子数（上限），不是约束的最少棋子数。
  - 搜索目标：找出一组"活跃英雄"组合，使每个活跃羁绊恰好满足 trait_counts。
  - 剩余空格（player_level - 活跃英雄数）由与活跃羁绊无关的"挂件"英雄填充。
  - 输出只报告活跃英雄，挂件不计入 must_have / candidates。

支持：天选之人机制（Vision 识别出天选属于哪个羁绊，算法自动扣除 +1 计数）

默认棋子池：``data/rag_legend_chess.jsonl``（英雄联盟传奇 RAG）。

用法示例
--------
    from recombo_solver import RecomboSolver
    solver = RecomboSolver()
    result = solver.solve(
        trait_counts={"斗士": 4, "永恒之森": 3},
        player_level=6,
        chosen_trait="永恒之森",
    )
    print(result["must_have"])   # 必然在场的英雄
    print(result["priority"])    # 按出现频率排序的候选英雄
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 等级 → 可出现的费用段（天选福星赛季）
# ──────────────────────────────────────────────────────────────────────────────
LEVEL_COST_TIERS: Dict[int, List[int]] = {
    1: [1],
    2: [1],
    3: [1, 2],
    4: [1, 2, 3],
    5: [1, 2, 3],
    6: [1, 2, 3, 4],
    7: [1, 2, 3, 4],
    8: [1, 2, 3, 4, 5],
    9: [1, 2, 3, 4, 5],
    10: [1, 2, 3, 4, 5],
}


class RecomboSolver:
    """
    基于回溯+剪枝的棋子组合逆推引擎。

    输入：Vision 识别的羁绊字典 + 玩家等级 + 天选羁绊（可选）
    输出：
      must_have   每个有效组合都包含的英雄（交集）
      candidates  至少一个有效组合包含的英雄（并集）
      priority    候选英雄按出现频率排序
      top_combos  得分最高的若干完整组合
    """

    def __init__(self, chess_jsonl_path: Optional[str] = None) -> None:
        base = Path(__file__).resolve().parent
        path = (
            Path(chess_jsonl_path)
            if chess_jsonl_path
            else base / "data" / "rag_legend_chess.jsonl"
        )
        self._heroes: List[Dict[str, Any]] = []
        self._load(path)

    def _load(self, path: Path) -> None:
        seen: set = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                name = (obj.get("name") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                try:
                    cost = int(obj.get("cost", 0))
                except (TypeError, ValueError):
                    cost = 0
                if cost <= 0:
                    continue
                traits = obj.get("traits") or []
                if not isinstance(traits, list):
                    traits = []
                self._heroes.append(
                    {
                        "name": name,
                        "cost": cost,
                        "traits": [str(t) for t in traits if t],
                    }
                )
        print(f"[RecomboSolver] 加载 {len(self._heroes)} 个英雄")

    # ──────────────────────────────────────────────────────────────────────────
    # 核心搜索
    # ──────────────────────────────────────────────────────────────────────────

    def solve(
        self,
        trait_counts: Dict[str, int],
        player_level: int,
        chosen_trait: Optional[str] = None,
        max_results: int = 300,
        timeout_ms: float = 2000.0,
    ) -> Dict[str, Any]:
        """
        执行组合搜索。

        Parameters
        ----------
        trait_counts  : Vision 识别到的 {羁绊名: 激活人数} 字典
                        例 {"斗士": 4, "永恒之森": 3}
        player_level  : 当前人口等级（棋盘总棋子数上限）
        chosen_trait  : 天选之人所属羁绊（Vision 已知时传入）
                        天选英雄额外贡献 +1 计数，算法内部自动扣除
        max_results   : 最多收集多少个有效组合后停止
        timeout_ms    : 搜索超时毫秒数

        Returns
        -------
        dict
          total_valid  int
          timeout      bool
          must_have    List[str]   所有有效组合的交集英雄
          candidates   List[str]   所有有效组合的并集英雄
          priority     List[dict]  按出现频率降序，含 name/cost/traits/freq
          top_combos   List[dict]  得分最高的 ≤10 组，含 names/trait_summary/score
          elapsed_ms   float
          note         str         （无结果时附加说明）
        """
        t0 = time.perf_counter()

        # ── 1. 目标预处理 ──────────────────────────────────────────────────────
        targets: Dict[str, int] = {t: c for t, c in trait_counts.items() if c > 0}

        # 天选调整：chosen_trait 展示人数 - 1 = 实际物理英雄数贡献
        # 例：福星×4 且天选=福星 → 物理英雄只需带 3 个福星计数
        if chosen_trait and chosen_trait in targets:
            adjusted = targets[chosen_trait] - 1
            if adjusted > 0:
                targets[chosen_trait] = adjusted
            else:
                # 调整为 0 → 只有天选自己带该羁绊，从约束字典删除
                # 搜索结束后仍会强制要求 combo 里有 chosen_trait 英雄
                del targets[chosen_trait]

        # ── 2. 构建候选池 ──────────────────────────────────────────────────────
        avail_costs = LEVEL_COST_TIERS.get(player_level, list(range(1, 6)))

        # 池中英雄：费用符合 + 至少携带一个活跃目标羁绊或 chosen_trait
        required_traits = set(targets.keys())
        if chosen_trait:
            required_traits.add(chosen_trait)

        pool = [
            h
            for h in self._heroes
            if h["cost"] in avail_costs
            and any(t in h["traits"] for t in required_traits)
        ]

        # ── 3. 排序：覆盖活跃羁绊最多的英雄优先（加速收敛）─────────────────────
        def _active_score(h: Dict) -> int:
            return sum(1 for t in targets if t in h["traits"])

        pool.sort(key=_active_score, reverse=True)

        n = len(pool)
        active_traits = list(targets.keys())

        # ── 4. 预计算后缀计数（剪枝 A 用）────────────────────────────────────
        # suffix[i][t] = pool[i:] 中携带活跃羁绊 t 的英雄数
        suffix: List[Dict[str, int]] = [
            {t: 0 for t in active_traits} for _ in range(n + 1)
        ]
        for i in range(n - 1, -1, -1):
            for t in active_traits:
                suffix[i][t] = suffix[i + 1][t] + (1 if t in pool[i]["traits"] else 0)

        # ── 5. 回溯搜索 ────────────────────────────────────────────────────────
        # 核心设计：
        #   - combo 不要求恰好填满 player_level 个槽，
        #     只需 remaining 全部归零（活跃约束满足）即为有效结果。
        #   - player_level 是 combo 的上限（不能超过棋盘格数）。
        #   - 剩余（player_level - combo 大小）的槽由挂件英雄填充，不计入输出。
        results: List[List[str]] = []
        combo: List[Dict] = []
        timed_out = [False]
        deadline = t0 + timeout_ms / 1000.0

        def backtrack(start: int, slots_left: int, remaining: Dict[str, int]) -> None:
            if timed_out[0] or len(results) >= max_results:
                return
            if time.perf_counter() > deadline:
                timed_out[0] = True
                return

            # ── 终止：约束全部满足 ────────────────────────────────────────────
            if all(v == 0 for v in remaining.values()):
                # 天选约束：combo 里必须有至少 1 个 chosen_trait 英雄
                if chosen_trait is None or any(
                    chosen_trait in h["traits"] for h in combo
                ):
                    results.append([h["name"] for h in combo])
                return  # 满足后不再添加（否则超标）

            # ── 剪枝：已无可用槽位但约束未满足 ────────────────────────────────
            if slots_left == 0:
                return

            # ── 剪枝 A：pool[start:] 中某羁绊的英雄数不够用 ─────────────────
            for t, need in remaining.items():
                if need > 0 and suffix[start][t] < need:
                    return

            # ── 剪枝 B：最紧迫的需求超出剩余槽位数 ────────────────────────────
            max_need = max((v for v in remaining.values() if v > 0), default=0)
            if max_need > slots_left:
                return

            # ── 递归枚举 ──────────────────────────────────────────────────────
            for i in range(start, n):
                h = pool[i]

                # 超标检查：加入该英雄后，任意目标羁绊不能超过目标值
                new_rem: Dict[str, int] = {}
                valid = True
                for t, need in remaining.items():
                    delta = 1 if t in h["traits"] else 0
                    new_val = need - delta
                    if new_val < 0:
                        valid = False
                        break
                    new_rem[t] = new_val
                if not valid:
                    continue

                combo.append(h)
                backtrack(i + 1, slots_left - 1, new_rem)
                combo.pop()

        backtrack(0, player_level, dict(targets))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── 6. 汇总结果 ────────────────────────────────────────────────────────
        if not results:
            return {
                "total_valid": 0,
                "timeout": timed_out[0],
                "must_have": [],
                "candidates": [],
                "priority": [],
                "top_combos": [],
                "elapsed_ms": round(elapsed_ms, 2),
                "note": (
                    "未找到满足条件的组合。可能原因：\n"
                    "  ① 该等级无法出现足够多的对应羁绊英雄（cost 超范围）\n"
                    "  ② 某羁绊目标值超过了当前赛季该羁绊的英雄总数\n"
                    "  ③ chosen_trait 调整后约束无解\n"
                    "建议：检查 trait_counts 数值或 player_level 是否合理。"
                ),
            }

        name_to_hero = {h["name"]: h for h in pool}
        total = len(results)

        # 出现频率统计
        freq_counter: Dict[str, int] = defaultdict(int)
        for combo_names in results:
            for name in combo_names:
                freq_counter[name] += 1

        # 交集（必然在场）
        result_sets = [set(c) for c in results]
        must_have_set: set = result_sets[0].copy()
        for s in result_sets[1:]:
            must_have_set &= s

        # 优先级排序（并集）
        priority = sorted(
            [
                {
                    "name": name,
                    "cost": name_to_hero[name]["cost"] if name in name_to_hero else 0,
                    "traits": (
                        name_to_hero[name]["traits"] if name in name_to_hero else []
                    ),
                    "freq": round(freq_counter[name] / total, 3),
                }
                for name in freq_counter
            ],
            key=lambda x: (-x["freq"], x["cost"]),
        )

        # Top 10 完整组合（按目标羁绊满足总分排序）
        def _score(names: List[str]) -> Tuple[int, Dict[str, int]]:
            cnt: Dict[str, int] = defaultdict(int)
            for name in names:
                if name in name_to_hero:
                    for t in name_to_hero[name]["traits"]:
                        cnt[t] += 1
            score = sum(cnt.get(t, 0) for t in targets)
            return score, dict(cnt)

        scored = sorted(
            [(_score(c), c) for c in results],
            key=lambda x: -x[0][0],
        )[:10]

        top_combos = [
            {
                "names": names,
                "trait_summary": {t: sc[1].get(t, 0) for t in targets},
                "score": sc[0],
            }
            for sc, names in scored
        ]

        return {
            "total_valid": total,
            "timeout": timed_out[0],
            "must_have": sorted(must_have_set),
            "candidates": [p["name"] for p in priority],
            "priority": priority,
            "top_combos": top_combos,
            "elapsed_ms": round(elapsed_ms, 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 端到端测试
# ──────────────────────────────────────────────────────────────────────────────

def _print_result(r: Dict[str, Any], scenario: str = "") -> None:
    if scenario:
        print(f"\n{'='*65}")
        print(f"  {scenario}")
        print(f"{'='*65}")
    timeout_str = " [超时提前终止]" if r["timeout"] else ""
    print(
        f"  耗时: {r['elapsed_ms']:.1f}ms  "
        f"有效组合(活跃英雄部分): {r['total_valid']}{timeout_str}"
    )
    if r.get("note"):
        print(f"  提示:\n{r['note']}")
        return

    print(f"  必然在场 ({len(r['must_have'])}): {r['must_have']}")
    print(f"  优先候选 (TOP 12):")
    for p in r["priority"][:12]:
        mark = "[必须]" if p["name"] in r["must_have"] else "      "
        print(
            f"    {mark} ★{p['cost']} {p['name']:<8}"
            f"  出现率={p['freq']*100:5.1f}%"
            f"  羁绊={p['traits']}"
        )
    if r["top_combos"]:
        print(f"  TOP 3 完整组合(活跃英雄):")
        for i, c in enumerate(r["top_combos"][:3], 1):
            print(f"    {i}. 得分={c['score']}  羁绊满足={c['trait_summary']}")
            print(f"       英雄={c['names']}")


def main() -> None:
    solver = RecomboSolver()

    # 以下示例使用 ``data/rag_legend_chess.jsonl`` 中的羁绊名（英雄联盟传奇池）。

    # ── 场景1：等级4，斗士×2 + 艾欧尼亚×2，无天选 ─────────────────────────────
    _print_result(
        solver.solve(
            trait_counts={"斗士": 2, "艾欧尼亚": 2},
            player_level=4,
            chosen_trait=None,
        ),
        "场景1：等级4，斗士×2 + 艾欧尼亚×2，无天选",
    )

    # ── 场景2：天选=斗士（展示计数含 +1，算法内会扣减）──────────────────────────
    _print_result(
        solver.solve(
            trait_counts={"斗士": 2, "艾欧尼亚": 2},
            player_level=4,
            chosen_trait="斗士",
        ),
        "场景2：等级4，斗士×2 + 艾欧尼亚×2，天选=斗士",
    )

    # ── 场景3：等级6，斗士×4 + 比尔吉沃特×2 ───────────────────────────────────
    _print_result(
        solver.solve(
            trait_counts={"斗士": 4, "比尔吉沃特": 2},
            player_level=6,
            chosen_trait=None,
            timeout_ms=3000.0,
        ),
        "场景3：等级6，斗士×4 + 比尔吉沃特×2，无天选",
    )


if __name__ == "__main__":
    main()
