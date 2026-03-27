# -*- coding: utf-8 -*-
"""对比 run_recognition_chess：legacy（整图 mark.copy + 写 PNG）vs 优化（ROI 回滚 + 跳过 PNG）。"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from element_recog import bars_recog as br

import fightboard_mobilenet as fm

PROJECT_DIR = Path(__file__).resolve().parent


def _pick_images(max_n: int = 6) -> List[Path]:
    cands: List[Path] = []
    for sub in ("对局截图", "备用截图"):
        d = PROJECT_DIR / sub
        if not d.is_dir():
            continue
        for pat in ("*-a.png", "*_a.png"):
            for p in sorted(d.glob(pat)):
                if p.is_file():
                    cands.append(p)
    out: List[Path] = []
    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= max_n:
            break
    return out


def _slim_rows(js: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    rows = []
    for r in js.get("results") or []:
        if not isinstance(r, dict):
            continue
        pos = r.get("position") if isinstance(r.get("position"), dict) else {}
        rows.append(
            (
                int(r.get("bar_index") or 0),
                str(r.get("best") or ""),
                str(r.get("confidence") or ""),
                pos.get("cell_row"),
                pos.get("cell_col"),
                str(pos.get("label") or ""),
            )
        )
    rows.sort(key=lambda t: t[0])
    return rows


def main() -> None:
    imgs = _pick_images(6)
    if not imgs:
        raise SystemExit("未找到 *-a.png / *_a.png（对局截图 或 备用截图）")
    bar_tpl = br.find_healthbar_template(PROJECT_DIR)
    piece_dir = PROJECT_DIR / "chess_gallery"
    bundle = fm._get_worker_mobilenet_bundle(
        piece_dir=piece_dir,
        torch_device="auto",
        prefer_rocm=False,
        torch_auto_priority="dml,cuda,cpu",
        torch_fallback_to_cpu=True,
    )

    common_kw = dict(
        template_path=bar_tpl,
        piece_dir=piece_dir,
        circle_diameter=84,
        alpha_tight=True,
        save_debug_crops=False,
        batch_embed=True,
        mobilenet_bundle=bundle,
        torch_device="auto",
        prefer_rocm=False,
        torch_auto_priority="dml,cuda,cpu",
        torch_fallback_to_cpu=True,
        print_timings=False,
    )

    legacy_ms: List[float] = []
    fast_ms: List[float] = []
    all_match = True

    for p in imgs:
        with tempfile.TemporaryDirectory(prefix="bench_ch_w_") as td:
            fm.run_recognition_chess(
                image_path=p,
                output_dir=Path(td),
                fast_bar_mark_rollback=True,
                save_annotated_png=False,
                **common_kw,
            )

        with tempfile.TemporaryDirectory(prefix="bench_ch_l_") as td:
            t0 = time.perf_counter()
            js_l = fm.run_recognition_chess(
                image_path=p,
                output_dir=Path(td),
                fast_bar_mark_rollback=False,
                save_annotated_png=True,
                **common_kw,
            )
            legacy_ms.append((time.perf_counter() - t0) * 1000.0)

        with tempfile.TemporaryDirectory(prefix="bench_ch_f_") as td:
            t0 = time.perf_counter()
            js_f = fm.run_recognition_chess(
                image_path=p,
                output_dir=Path(td),
                fast_bar_mark_rollback=True,
                save_annotated_png=False,
                **common_kw,
            )
            fast_ms.append((time.perf_counter() - t0) * 1000.0)

        slim_l = _slim_rows(js_l)
        slim_f = _slim_rows(js_f)
        same = slim_l == slim_f
        all_match = all_match and same
        print(
            f"{p.name}: legacy={legacy_ms[-1]:.1f}ms fast={fast_ms[-1]:.1f}ms "
            f"Δ={legacy_ms[-1] - fast_ms[-1]:.1f}ms 一致={same}"
        )

    print("---")
    print(
        f"平均 legacy={float(np.mean(legacy_ms)):.1f}ms  fast={float(np.mean(fast_ms)):.1f}ms "
        f"加速×{float(np.mean(legacy_ms) / max(1e-6, np.mean(fast_ms))):.2f}"
    )
    print(f"识别 slim 行完全一致: {all_match}")

    rep = {
        "images": [x.name for x in imgs],
        "avg_ms_legacy": float(np.mean(legacy_ms)),
        "avg_ms_fast": float(np.mean(fast_ms)),
        "all_slim_equal": bool(all_match),
    }
    out_p = PROJECT_DIR / "bench_chess_optim_ab_report.json"
    out_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写: {out_p.name}")


if __name__ == "__main__":
    main()
