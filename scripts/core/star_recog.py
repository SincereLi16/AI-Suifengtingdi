# -*- coding: utf-8 -*-
"""
血条旁 **星级** 取样与多尺度模板识别（原独立脚本已并入本模块）。

**子命令**

- ``export``：从对局截图按与 fightboard 一致的血条框裁星级 ROI 小图。
- ``preview``：对已裁 ROI 目录跑模板匹配，写出带预测文件名的 PNG 与汇总 JSON。

**库用法**（供 ``fightboard_mobilenet`` 等单链路内联）：``recognize_star_for_bar_box``、
``crop_star_roi_square``、``load_templates_bgr``、``build_scale_ratios`` 等。

::

    python scripts/core/star_recog.py export --img-dir 对局截图 --out-dir runs/star_roi_crops
    python scripts/core/star_recog.py preview --roi-dir runs/star_roi_crops --out-dir runs/star_tm_preview
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

for _d in Path(__file__).resolve().parents:
    if (_d / "repo_sys_path.py").exists():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
import repo_sys_path  # noqa: F401

from element_recog import bars_recog as br
from element_recog.chess_recog import ROI as CR_ROI
from element_recog.chess_recog import _detect_healthbars_in_roi, load_healthbar_templates

from project_paths import PROJECT_ROOT

METHOD = cv2.TM_CCOEFF_NORMED

DEFAULT_SCALE_MIN = 0.32
DEFAULT_SCALE_MAX = 1.42
DEFAULT_SCALE_STEPS = 6

DEFAULT_STAR_ROI_SIDE = 35
DEFAULT_STAR_SHIFT_RIGHT = 6
DEFAULT_STAR_SNAP_WIDTH = 106
DEFAULT_BAR_THRESHOLD_EXPORT = 0.58


def build_scale_ratios(
    scale_min: float,
    scale_max: float,
    steps: int,
) -> Tuple[float, ...]:
    if steps < 2:
        raise ValueError("scale steps 至少为 2")
    if scale_min <= 0 or scale_max <= 0 or scale_min >= scale_max:
        raise ValueError("需要 0 < scale_min < scale_max")
    return tuple(round(x, 4) for x in np.linspace(scale_min, scale_max, steps))


def snap_bar_xywh_fixed_width_centered(
    x: int,
    y: int,
    w: int,
    h: int,
    scene_width: int,
    target_w: int,
) -> Tuple[int, int, int, int]:
    """与 fightboard / export 一致：钉档宽度并相对检测框水平居中。"""
    tw = int(target_w)
    if tw < 1 or w < 1 or h < 1:
        return (int(x), int(y), int(w), int(h))
    cx = int(x) + int(w) // 2
    nx = int(cx - tw // 2)
    sw = max(1, int(scene_width))
    if nx < 0:
        nx = 0
    elif nx + tw > sw:
        nx = max(0, sw - tw)
    return (nx, int(y), tw, int(h))


def crop_star_roi_square(
    scene_bgr: np.ndarray,
    bar_x: int,
    bar_y: int,
    bar_w: int,
    bar_h: int,
    *,
    side: int,
    shift_right: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    H, W = scene_bgr.shape[:2]
    side = int(max(8, min(128, side)))
    ih = int(bar_h)
    sr = int(max(-32, min(32, shift_right)))
    cx = int(bar_x) + sr
    cy = int(bar_y) + ih // 2
    half = side // 2
    sx1 = cx - half
    sy1 = cy - half
    sx2 = sx1 + side
    sy2 = sy1 + side
    ix1 = max(0, sx1)
    iy1 = max(0, sy1)
    ix2 = min(W, sx2)
    iy2 = min(H, sy2)
    rw = ix2 - ix1
    rh = iy2 - iy1
    if rw < 4 or rh < 4:
        return None, None
    patch = scene_bgr[iy1:iy2, ix1:ix2].copy()
    return patch, (ix1, iy1, rw, rh)


def _load_bgr(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None:
        im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(str(path))
    return im


def resolve_template_dir(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        p = explicit.resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"模板路径不是目录: {p}")
        return p
    for name in ("星级模板", "star_templates", "star_template"):
        p = (PROJECT_ROOT / name).resolve()
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "未找到星级模板目录。请在项目根创建「星级模板」或 star_templates，"
        "或传入 template_dir 指向含 一星.png / 两星.png / 三星.png 的文件夹。"
    )


def load_templates_bgr(template_dir: Path) -> Dict[int, np.ndarray]:
    td = template_dir.resolve()
    mapping = {
        1: ("一星.png", "star_1.png", "1.png", "template_1.png"),
        2: ("两星.png", "star_2.png", "2.png", "template_2.png"),
        3: ("三星.png", "star_3.png", "3.png", "template_3.png"),
    }
    out: Dict[int, np.ndarray] = {}
    for k, names in mapping.items():
        got: Optional[np.ndarray] = None
        for n in names:
            p = td / n
            if p.is_file():
                got = _load_bgr(p)
                break
        if got is None:
            raise FileNotFoundError(
                f"模板目录 {td} 中未找到 {k} 星模板（试过: {names[:3]}…）"
            )
        if got.size < 9:
            raise ValueError(f"模板为空: {k}")
        out[k] = got
    return out


def _gray_for_match(bgr: np.ndarray, *, preprocess: str) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p = (preprocess or "none").strip().lower()
    if p in ("none", ""):
        return g
    if p == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(g)
    if p in ("eq", "equalize", "histeq"):
        return cv2.equalizeHist(g)
    raise ValueError(f"未知 preprocess: {preprocess!r}，可选 none / clahe / eq")


def max_match_score_multiscale(
    roi_bgr: np.ndarray,
    tmpl_bgr: np.ndarray,
    *,
    scale_ratios: Sequence[float],
    preprocess: str = "none",
) -> Tuple[float, Dict[str, Any]]:
    dbg: Dict[str, Any] = {}
    rh, rw = roi_bgr.shape[:2]
    th0, tw0 = tmpl_bgr.shape[:2]
    if th0 < 2 or tw0 < 2 or rh < 4 or rw < 4:
        return 0.0, dbg
    roi_g = _gray_for_match(roi_bgr, preprocess=preprocess)
    best_s = -1.0
    best_ratio = 0.0
    best_wh = (0, 0)
    for r in scale_ratios:
        tw = max(4, int(round(tw0 * float(r))))
        th = max(4, int(round(th0 * float(r))))
        if tw >= rw or th >= rh:
            continue
        t = cv2.resize(tmpl_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        tg = _gray_for_match(t, preprocess=preprocess)
        res = cv2.matchTemplate(roi_g, tg, METHOD)
        if res.size == 0:
            continue
        _mn, maxv, _mnloc, _ml = cv2.minMaxLoc(res)
        fv = float(maxv)
        if fv > best_s:
            best_s = fv
            best_ratio = float(r)
            best_wh = (tw, th)
    dbg["best_scale_ratio"] = round(best_ratio, 4)
    dbg["best_tpl_wh"] = [int(best_wh[0]), int(best_wh[1])]
    return max(0.0, best_s), dbg


def sanitize_filename_part(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s[:180] if len(s) > 180 else s


def predict_star_with_ambiguity_bias(
    scores: Dict[str, float],
    *,
    margin: float,
    min_score_for_upgrade: float,
) -> Tuple[int, int, Dict[str, Any]]:
    s1 = float(scores["1"])
    s2 = float(scores["2"])
    s3 = float(scores["3"])
    ranked = sorted(((1, s1), (2, s2), (3, s3)), key=lambda x: -x[1])
    pred_raw = int(ranked[0][0])
    meta: Dict[str, Any] = {"pred_raw": pred_raw, "upgrades": []}

    if margin <= 0:
        return pred_raw, pred_raw, meta

    pred = pred_raw
    ok2 = min_score_for_upgrade <= 0 or s2 >= min_score_for_upgrade
    ok3 = min_score_for_upgrade <= 0 or s3 >= min_score_for_upgrade
    if pred == 1 and s2 >= s1 - margin and ok2:
        pred = 2
        meta["upgrades"].append("1_to_2")
    if pred == 2 and s3 >= s2 - margin and ok3:
        pred = 3
        meta["upgrades"].append("2_to_3")

    return pred, pred_raw, meta


def recognize_star_from_roi_bgr(
    roi_bgr: np.ndarray,
    templates: Dict[int, np.ndarray],
    *,
    scale_ratios: Sequence[float],
    preprocess: str = "clahe",
    margin: float = 0.12,
    min_score_for_upgrade: float = 0.0,
) -> Optional[Dict[str, Any]]:
    if roi_bgr is None or roi_bgr.size < 9:
        return None
    scores: Dict[str, float] = {}
    dbg_by: Dict[str, Any] = {}
    for k in (1, 2, 3):
        s, d = max_match_score_multiscale(
            roi_bgr,
            templates[k],
            scale_ratios=scale_ratios,
            preprocess=preprocess,
        )
        scores[str(k)] = round(float(s), 4)
        dbg_by[str(k)] = d
    pred, pred_raw, amb_meta = predict_star_with_ambiguity_bias(
        scores,
        margin=float(margin),
        min_score_for_upgrade=float(min_score_for_upgrade),
    )
    pred_score = float(scores[str(pred)])
    others = [float(scores[str(k)]) for k in (1, 2, 3) if k != pred]
    second = max(others) if others else 0.0
    return {
        "pred": int(pred),
        "pred_raw": int(pred_raw),
        "pred_score": float(pred_score),
        "second_score": float(second),
        "margin_vs_second": float(pred_score - second),
        "scores": scores,
        "ambiguity": amb_meta,
        "dbg": dbg_by,
    }


def recognize_star_for_bar_box(
    scene_bgr: np.ndarray,
    bar_xywh: Sequence[int],
    templates: Dict[int, np.ndarray],
    *,
    scale_ratios: Sequence[float],
    side: int = DEFAULT_STAR_ROI_SIDE,
    shift_right: int = DEFAULT_STAR_SHIFT_RIGHT,
    preprocess: str = "clahe",
    margin: float = 0.12,
    min_score_for_upgrade: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    与 fightboard 单链路一致：使用 **最终** ``bar_box``（钉档/列投影后的全局 xywh）
    裁星级正方形 ROI，再做多尺度模板匹配 + D 让步。
    """
    x, y, w, h = map(int, bar_xywh)
    patch, rect = crop_star_roi_square(
        scene_bgr, x, y, w, h, side=int(side), shift_right=int(shift_right)
    )
    if patch is None or rect is None:
        return None
    inner = recognize_star_from_roi_bgr(
        patch,
        templates,
        scale_ratios=scale_ratios,
        preprocess=preprocess,
        margin=margin,
        min_score_for_upgrade=min_score_for_upgrade,
    )
    if inner is None:
        return None
    ix1, iy1, rw, rh = rect
    out = dict(inner)
    out["star_roi_rect_xywh"] = [int(ix1), int(iy1), int(rw), int(rh)]
    return out


def clear_preview_output_dir(out_dir: Path) -> int:
    n = 0
    if not out_dir.is_dir():
        return 0
    for p in out_dir.iterdir():
        if p.is_file():
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n


def _stem_matches_primary_suffix(stem: str, suffix: str) -> bool:
    if not suffix:
        return True
    s = stem.lower()
    su = suffix.lower()
    if s.endswith(f"-{su}") or s.endswith(f"_{su}"):
        return True
    if re.fullmatch(r"\d+", s or ""):
        return True
    return False


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main_export(argv: Optional[Sequence[str]] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="从对局图导出星级 ROI 小图（star_recog export）")
    ap.add_argument("--img-dir", type=Path, required=True, help="截图目录")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "star_roi_crops",
        help="裁剪输出目录",
    )
    ap.add_argument("--side", type=int, default=DEFAULT_STAR_ROI_SIDE, help="正方形边长")
    ap.add_argument(
        "--grid-sides",
        type=str,
        default="",
        help="逗号分隔多个边长；与 --grid-shifts 组合为笛卡尔积",
    )
    ap.add_argument(
        "--grid-shifts",
        type=str,
        default="",
        help="逗号分隔 shift-right，与 --grid-sides 组合",
    )
    ap.add_argument(
        "--shift-right",
        type=int,
        default=DEFAULT_STAR_SHIFT_RIGHT,
        help="ROI 中心相对血条左缘向右平移像素",
    )
    ap.add_argument("--snap-width", type=int, default=DEFAULT_STAR_SNAP_WIDTH, help="血条钉档宽度")
    ap.add_argument("--bar-threshold", type=float, default=DEFAULT_BAR_THRESHOLD_EXPORT, help="血条 TM 阈值")
    ap.add_argument("--primary-suffix", type=str, default="a", help="仅导出主图 stem")
    ap.add_argument("--all-images", action="store_true", help="处理目录内全部图片")
    args = ap.parse_args(list(argv) if argv is not None else None)

    img_dir = args.img_dir.resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"目录不存在: {img_dir}")
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sides_grid = _parse_int_list(args.grid_sides)
    shifts_grid = _parse_int_list(args.grid_shifts)
    if not sides_grid:
        combos: List[Tuple[int, int]] = [(int(args.side), int(args.shift_right))]
    elif not shifts_grid:
        combos = [(s, int(args.shift_right)) for s in sides_grid]
    else:
        combos = [(s, sh) for s in sides_grid for sh in shifts_grid]

    tpl_path = br.find_healthbar_template(PROJECT_ROOT)
    if not tpl_path or not tpl_path.exists():
        raise SystemExit(f"未找到血条模板: {tpl_path}")
    hb_templates = load_healthbar_templates(tpl_path)
    suf = (args.primary_suffix or "").strip()

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths: List[Path] = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not args.all_images:
        paths = [p for p in paths if _stem_matches_primary_suffix(p.stem, suf)]
    if not paths:
        raise SystemExit("无可用图片（检查 --img-dir / --all-images / --primary-suffix）")

    n_saved = 0
    for side_i, shift_i in combos:
        if len(combos) == 1:
            out_dir = out_root
        else:
            out_dir = out_root / f"s{int(side_i)}_dr{int(shift_i)}"
            out_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            scene = _load_bgr(p)
            H, W = scene.shape[:2]
            boxes = _detect_healthbars_in_roi(
                scene,
                hb_templates,
                CR_ROI,
                strategy="simple_tiled",
                simple_threshold=float(args.bar_threshold),
            )
            snap_w = max(1, int(args.snap_width))
            boxes = [
                snap_bar_xywh_fixed_width_centered(int(x), int(y), int(w), int(h), W, snap_w)
                for (x, y, w, h) in boxes
            ]
            for bi, (x, y, w, h) in enumerate(boxes):
                patch, rect = crop_star_roi_square(
                    scene,
                    x,
                    y,
                    w,
                    h,
                    side=int(side_i),
                    shift_right=int(shift_i),
                )
                if patch is None or rect is None:
                    continue
                stem = p.stem
                fn = f"{stem}_bar{bi + 1:02d}_s{int(side_i)}_dr{int(shift_i)}.png"
                out_path = out_dir / fn
                cv2.imencode(".png", patch)[1].tofile(str(out_path))
                n_saved += 1
                try:
                    rel = out_path.relative_to(out_root)
                except ValueError:
                    rel = out_path.name
                print(f"[OK] {rel}  <- {p.name} bar{bi + 1}")

    print(f"\n共写出 {n_saved} 张 ROI 到 {out_root}")
    print("标注：建子文件夹 1/2/3 或 一星/两星/三星，移入小图后可做人工真值或离线评测。")


def main_preview(argv: Optional[Sequence[str]] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="星级 ROI 多尺度模板匹配预览（star_recog preview）")
    ap.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="含 一星.png / 两星.png / 三星.png 的目录",
    )
    ap.add_argument(
        "--roi-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "star_roi_crops",
        help="ROI 小图目录",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "star_tm_preview",
        help="输出目录",
    )
    ap.add_argument(
        "--no-clear-out",
        action="store_true",
        help="不清空输出目录顶层旧文件",
    )
    ap.add_argument("--scale-min", type=float, default=DEFAULT_SCALE_MIN)
    ap.add_argument("--scale-max", type=float, default=DEFAULT_SCALE_MAX)
    ap.add_argument("--scale-steps", type=int, default=DEFAULT_SCALE_STEPS)
    ap.add_argument(
        "--preprocess",
        type=str,
        default="clahe",
        choices=("none", "clahe", "eq"),
    )
    ap.add_argument("--star-ambiguity-margin", type=float, default=0.12, metavar="D")
    ap.add_argument("--star-upgrade-min-score", type=float, default=0.0, metavar="T")
    args = ap.parse_args(list(argv) if argv is not None else None)

    scale_ratios = build_scale_ratios(args.scale_min, args.scale_max, args.scale_steps)

    tdir = resolve_template_dir(args.template_dir)
    roi_dir = args.roi_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not roi_dir.is_dir():
        raise SystemExit(f"ROI 目录不存在: {roi_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clear_out:
        n_rm = clear_preview_output_dir(out_dir)
        if n_rm:
            print(f"[INFO] 已清空输出目录 {n_rm} 个旧文件: {out_dir}")

    templates = load_templates_bgr(tdir)
    print(f"[INFO] 模板目录: {tdir}")
    for k in (1, 2, 3):
        h, w = templates[k].shape[:2]
        print(f"       {k}星模板尺寸: {w}x{h}")
    print(
        f"[INFO] 尺度 {args.scale_min}～{args.scale_max} 共 {len(scale_ratios)} 档；"
        f" preprocess={args.preprocess}；D={args.star_ambiguity_margin}"
    )

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    roi_paths = sorted(
        p for p in roi_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not roi_paths:
        raise SystemExit(f"{roi_dir} 下无图片")

    results: List[Dict[str, Any]] = []
    for rp in roi_paths:
        roi = _load_bgr(rp)
        inner = recognize_star_from_roi_bgr(
            roi,
            templates,
            scale_ratios=scale_ratios,
            preprocess=str(args.preprocess),
            margin=float(args.star_ambiguity_margin),
            min_score_for_upgrade=float(args.star_upgrade_min_score),
        )
        assert inner is not None
        pred = int(inner["pred"])
        pred_raw = int(inner["pred_raw"])
        pred_score = float(inner["pred_score"])
        margin = float(inner["margin_vs_second"])
        scores = inner["scores"]
        dbg_by = inner["dbg"]
        amb_meta = inner["ambiguity"]

        stem = rp.stem
        safe = sanitize_filename_part(stem)
        score_tag = f"{pred_score:.3f}".replace(".", "p")
        margin_tag = f"{margin:.3f}".replace(".", "p")
        out_name = f"pred{pred}星_tm{score_tag}_mg{margin_tag}_r{pred_raw}_bf_{safe}.png"
        out_path = out_dir / out_name
        cv2.imencode(".png", roi)[1].tofile(str(out_path))

        results.append(
            {
                "file": rp.name,
                "pred": pred,
                "pred_raw": pred_raw,
                "pred_score": pred_score,
                "margin": margin,
                "scores": scores,
                "ambiguity": amb_meta,
                "dbg": dbg_by,
                "saved_as": out_name,
            }
        )
        print(f"[OK] {rp.name} -> {out_name}")

    summary_path = out_dir / "star_tm_preview_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "template_dir": str(tdir),
                "roi_dir": str(roi_dir),
                "scale_min": args.scale_min,
                "scale_max": args.scale_max,
                "scale_steps": args.scale_steps,
                "scale_ratios": list(scale_ratios),
                "preprocess": args.preprocess,
                "star_ambiguity_margin": args.star_ambiguity_margin,
                "star_upgrade_min_score": args.star_upgrade_min_score,
                "no_clear_out": bool(args.no_clear_out),
                "count": len(results),
                "items": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n共 {len(results)} 张 -> {out_dir}")
    print(f"汇总 JSON: {summary_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("用法: python scripts/core/star_recog.py {export|preview} [选项…]", file=sys.stderr)
        raise SystemExit(2)
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "export":
        main_export(rest)
    elif cmd in ("preview", "match"):
        main_preview(rest)
    else:
        print("未知子命令；请使用 export 或 preview", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
