# -*- coding: utf-8 -*-
"""
fightboard_recog_v2：棋子识别（模板 或 MobileNetV3-Small 特征检索）+ 装备模板匹配。

- ``--chess-backend template``：灰度 TM_CCOEFF_NORMED（快但易错）。
- ``--chess-backend mobilenet_v3_small``：torchvision MobileNetV3-Small（ImageNet 预训练）提取向量，
  与图库做余弦 top-k，流程对齐 ``chess_recog`` 的 DINOv2 检索式，但更轻、更快。

多区域 Stage1/Stage2 投票 + softmax 聚合默认**保留**：当前为**检索式**而非在英雄类上训练的分类头，
偏移与光照仍会扰动单点特征，投票可压噪。若以后改为「全监督分类 + 数据对齐」，可再评估减采样点。

输出：每张图 ``{stem}_fightboard_综合标注.png``；可选 ``--json``。默认 ``fightboard_info_v2``。

血条：``load_healthbar_templates`` → ``_detect_healthbars_in_roi`` → 可选宽度钉档（``--bar-snap-width``；``--bar-width-snap`` / ``--no-bar-width-snap``）→ 可选将钉档框收进检测框内「非纯白」列范围（``--no-bar-clip-to-content`` 可关）；不在此重复绘制与检测不一致的分块网格（已删），以减开销、避免误导。
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from element_recog import chess_recog as cr
from element_recog import equip_recog as er
from element_recog import bars_recog as br
from element_recog.chess_recog import (
    BELOW_BAR_PX,
    CENTER_OFFSET_X,
    DEFAULT_SAMPLES,
    EXTRA_SAMPLES,
    TOP_ZONE_ALLOWED_LABELS,
    TOP_ZONE_MAX_Y_ABS,
    _FOOT_DY,
    _bgra_to_bgr_white,
    _board_cells,
    _collapse_topk_to_hero,
    _compute_confidence,
    _crop_circle_bgra,
    _detect_healthbars_in_roi,
    _draw_chinese_text,
    _infer_row_for_mapping,
    _nearest_cell_row_first,
    _row_ys,
    _save_image,
    _tall_bar_extra_foot_dy,
    _tight_crop_by_alpha,
    load_healthbar_templates,
)
from element_recog.chess_recog import ROI as CR_ROI
from element_recog.chess_recog import _prepare_piece_db_matrix, _topk_cosine

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "对局截图"
DEFAULT_PIECE_DIR = PROJECT_DIR / "chess_gallery"
DEFAULT_EQUIP_GALLERY = PROJECT_DIR / "equip_gallery"
DEFAULT_OUT = PROJECT_DIR / "fightboard_info_v2"

# 钉档默认宽度（像素）：相对「检测框」水平居中，使定宽框相对检测框左右对称伸缩。
FIGHTBOARD_BAR_SNAP_WIDTH_DEFAULT: int = 106


def _snap_bar_xywh_fixed_width_centered(
    x: int,
    y: int,
    w: int,
    h: int,
    scene_width: int,
    target_w: int,
) -> Tuple[int, int, int, int]:
    """
    将宽度设为 target_w；左缘使定宽框与检测框**共用同一水平中心**（相对检测框左右对称）。
    越界时夹紧到 [0, scene_width-target_w]，此时不再严格居中。
    """
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


def _bar_column_extent_from_detection_strip(
    scene_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> Optional[Tuple[int, int]]:
    """
    在**模板检测框**内按列估计血条「条体」的左右范围，返回全局坐标 ``(left_x, right_ex)``（右为开区间）。

    用中间行带的列灰度均值：以 ``(min+median)/2`` 为阈值取偏暗列，可去掉检测框左侧偏亮描边/留白；
    条体中间可能有间断（星级等），取**命中列的首尾外包络**（非最长连续段）。对比度过低时退回「列最小值」阈值。
    """
    H, W = scene_bgr.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x) + int(w))
    y1 = min(H, int(y) + int(h))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = scene_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    rh = int(gray.shape[0])
    rw = int(gray.shape[1])
    if rh < 2 or rw < 2:
        return None
    r0 = int(rh * 0.22)
    r1 = int(rh * 0.78)
    if r1 <= r0:
        r0, r1 = 0, rh
    band = gray[r0:r1, :].astype(np.float32)
    col_mean = band.mean(axis=0)
    mn = float(np.min(col_mean))
    mx = float(np.max(col_mean))

    def _envelope_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
        idx = np.flatnonzero(mask)
        if idx.size < 3:
            return None
        return (int(idx[0]), int(idx[-1]))

    i0: Optional[int] = None
    i1: Optional[int] = None
    if mx - mn >= 10.0:
        thr = (mn + float(np.median(col_mean))) / 2.0
        env = _envelope_from_mask(col_mean < thr)
        if env is not None:
            i0, i1 = env
        if env is None or i1 - i0 + 1 < 8:
            thr2 = float(np.percentile(col_mean, 38))
            env2 = _envelope_from_mask(col_mean < thr2)
            if env2 is not None:
                i0, i1 = env2
    if i0 is None or i1 is None or i1 - i0 + 1 < 8:
        col_min = gray.min(axis=0).astype(np.float32)
        bg = float(np.percentile(col_min, 90))
        thr3 = max(233.0, min(252.0, bg - 8.0))
        env3 = _envelope_from_mask(col_min < thr3)
        if env3 is not None:
            i0, i1 = env3
    if i0 is None or i1 is None:
        return None
    if i1 - i0 + 1 < 8:
        return None
    left = x0 + i0
    right_ex = x0 + i1 + 1
    return (left, right_ex)


def _clip_snapped_box_to_bar_column_extent(
    scene_bgr: np.ndarray,
    snapped: Tuple[int, int, int, int],
    det: Tuple[int, int, int, int],
    scene_w: int,
    *,
    min_span_px: int = 8,
) -> Tuple[int, int, int, int]:
    """
    将钉档后的 ``(nx,ny,nw,nh)`` 收束到 ``_bar_column_extent_from_detection_strip`` 给出的区间内：
    可视宽度 >= nw：在 [L,R) 内居中放置定宽 nw；
    不足 nw：左对齐 L，宽度改为可视跨度（仍可能小于 nw）。
    """
    nx, ny, nw, nh = map(int, snapped)
    dx, dy, dw, dh = map(int, det)
    ext = _bar_column_extent_from_detection_strip(scene_bgr, dx, dy, dw, dh)
    if ext is None:
        return (nx, ny, nw, nh)
    left, right_ex = ext
    span = int(right_ex) - int(left)
    sw = max(1, int(scene_w))
    if span < int(min_span_px):
        return (nx, ny, nw, nh)
    if span >= nw:
        nnx = int(left) + (span - nw) // 2
        nnx = max(0, min(nnx, sw - nw))
        # 若因图像边界夹紧导致仍略越出 [L,R)，再向内收一档
        if nnx < left:
            nnx = left
        if nnx + nw > right_ex:
            nnx = max(left, right_ex - nw)
        nnx = max(0, min(nnx, sw - nw))
        return (nnx, ny, nw, nh)
    # 可视比定宽还窄：用可视宽度，左缘对齐条带左端
    nnw = span
    nnx = max(0, min(int(left), sw - nnw))
    return (nnx, ny, nnw, nh)


def _iter_piece_image_paths(piece_dir: Path) -> List[Path]:
    files: List[Path] = []
    report = piece_dir / "report.json"
    if report.is_file():
        try:
            obj = json.loads(report.read_text(encoding="utf-8"))
            for it in obj.get("items") or []:
                dst = it.get("dst") or it.get("src")
                if not dst:
                    continue
                p = piece_dir / dst
                if p.is_file():
                    files.append(p)
        except Exception:
            pass
    if not files:
        files = sorted(
            [p for p in piece_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
    return files


def _build_gray_template_gallery(piece_dir: Path) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    for p in _iter_piece_image_paths(piece_dir):
        try:
            bgr = cr._load_image(p, allow_alpha=True)
        except Exception:
            continue
        if bgr.ndim == 2:
            g = bgr
        else:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if g.shape[0] < 2 or g.shape[1] < 2:
            continue
        out.append((p.stem, g))
    return out


def _topk_template_match(
    crop_bgr: np.ndarray,
    gallery: List[Tuple[str, np.ndarray]],
    *,
    size: int = 64,
    topk: int = 5,
) -> List[Tuple[str, float]]:
    if crop_bgr.size == 0 or not gallery:
        return []
    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        return []
    crop_r = cv2.resize(crop_bgr, (size, size))
    if crop_r.ndim == 2:
        cg = crop_r
    else:
        cg = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)
    scores: List[Tuple[str, float]] = []
    for stem, tmpl_gray in gallery:
        if tmpl_gray.shape[0] < 2 or tmpl_gray.shape[1] < 2:
            continue
        t = cv2.resize(tmpl_gray, (size, size))
        res = cv2.matchTemplate(cg, t, cv2.TM_CCOEFF_NORMED)
        score = float(np.reshape(res, (-1,))[0])
        scores.append((stem, score))
    scores.sort(key=lambda x: -x[1])
    k = max(1, min(int(topk), len(scores)))
    return scores[:k]


def _get_mobilenet_encoder(device: "Any") -> "Any":
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    enc = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten())
    enc.eval()
    enc.to(device)
    return enc


def _mobilenet_transform() -> "Any":
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _embed_mobilenet_bgr(
    bgr: np.ndarray,
    model: "Any",
    device: "Any",
    transform: "Any",
) -> np.ndarray:
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    n = float(np.linalg.norm(feat) + 1e-8)
    return feat / n


def _build_mobilenet_piece_db(
    piece_dir: Path,
    model: "Any",
    device: "Any",
    transform: "Any",
) -> List[Tuple[str, np.ndarray]]:
    db: List[Tuple[str, np.ndarray]] = []
    for p in _iter_piece_image_paths(piece_dir):
        try:
            bgr = cr._load_image(p, allow_alpha=True)
        except Exception:
            continue
        if bgr.size == 0:
            continue
        feat = _embed_mobilenet_bgr(bgr, model, device, transform)
        db.append((p.stem, feat))
    return db


def _print_timing_block(title: str, timings: Dict[str, float]) -> None:
    """将各阶段耗时（秒）打印到终端，便于调参。"""
    print(f"  --- {title} ---")
    for key in sorted(timings.keys()):
        v = float(timings[key])
        disp = f"{v * 1000.0:.2f} ms" if v < 1.0 else f"{v:.4f} s"
        print(f"  {key}: {disp}")


def run_recognition_chess(
    *,
    backend: str = "mobilenet_v3_small",
    image_path: Path,
    template_path: Path,
    piece_dir: Path,
    output_dir: Path,
    circle_diameter: int = 84,
    sample_topm: int = 3,
    temp: float = 0.08,
    min_sim: float = 0.35,
    min_gap: float = 0.60,
    min_gap_2votes: float = 0.30,
    min_gap_3votes: float = 0.20,
    sample_min_sim: float = 0.76,
    sample_min_margin: float = 0.010,
    min_votes: int = 2,
    two_stage: bool = True,
    stage2_min_sim: float = 0.74,
    stage2_min_margin: float = 0.005,
    stage2_top1sum_gap: float = 0.12,
    stage2_single_vote_min_top1max: float = 0.75,
    stage2_single_vote_min_gap_score: float = 0.30,
    alpha_tight: bool = True,
    alpha_thresh: int = 18,
    template_match_size: int = 64,
    topk_raw: int = 5,
    save_debug_crops: bool = True,
    bar_detect_strategy: str = "simple_tiled",
    bar_detect_simple_threshold: float = 0.58,
    mobilenet_bundle: Optional[Tuple[Any, Any, Any, List[str], np.ndarray]] = None,
    print_timings: bool = True,
    bar_width_snap: bool = True,
    bar_snap_width: int = FIGHTBOARD_BAR_SNAP_WIDTH_DEFAULT,
    bar_clip_to_content: bool = True,
) -> Dict[str, Any]:
    if backend not in ("template", "mobilenet_v3_small"):
        raise ValueError(f"未知 chess backend: {backend!r}")
    if bar_detect_strategy not in ("simple_tiled", "simple_tiled_edges"):
        raise ValueError(f"未知 bar_detect_strategy: {bar_detect_strategy!r}")

    img_path = Path(image_path)
    tpl_path = Path(template_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"输入图不存在: {img_path}")
    if not tpl_path.exists():
        raise FileNotFoundError(f"血条模板不存在: {tpl_path}")
    if not piece_dir.is_dir():
        raise FileNotFoundError(f"特征库目录不存在: {piece_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    circle_diameter = int(circle_diameter)
    r = max(8, circle_diameter // 2)
    t_prof0 = time.perf_counter()
    t_seg = t_prof0
    prof: Dict[str, float] = {}

    t0 = time.perf_counter()
    scene = cr._load_image(img_path)
    prof["load_scene"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    hb_templates = load_healthbar_templates(tpl_path)
    prof["load_healthbar_templates"] = time.perf_counter() - t0
    t_seg = time.perf_counter()

    boxes = _detect_healthbars_in_roi(
        scene,
        hb_templates,
        CR_ROI,
        strategy=str(bar_detect_strategy),
        simple_threshold=float(bar_detect_simple_threshold),
    )
    prof["detect_healthbars"] = time.perf_counter() - t_seg
    t_seg = time.perf_counter()

    boxes_raw = [tuple(map(int, t)) for t in boxes]
    snap_w = max(1, int(bar_snap_width))
    scene_w = int(scene.shape[1])
    if bool(bar_width_snap):
        t_snap0 = time.perf_counter()
        boxes = [
            _snap_bar_xywh_fixed_width_centered(int(x), int(y), int(w), int(h), int(scene_w), snap_w)
            for (x, y, w, h) in boxes_raw
        ]
        prof["snap_bar_fixed_width_centered"] = time.perf_counter() - t_snap0
    else:
        prof["snap_bar_fixed_width_centered"] = 0.0
    if bool(bar_width_snap) and bool(bar_clip_to_content):
        t_clip0 = time.perf_counter()
        boxes = [
            _clip_snapped_box_to_bar_column_extent(scene, snap, raw, scene_w)
            for raw, snap in zip(boxes_raw, boxes)
        ]
        prof["clip_bar_to_column_extent"] = time.perf_counter() - t_clip0
    else:
        prof["clip_bar_to_column_extent"] = 0.0
    t_seg = time.perf_counter()

    gallery: Optional[List[Tuple[str, np.ndarray]]] = None
    db_names: Optional[List[str]] = None
    db_mat: Optional[np.ndarray] = None
    mobilenet_model: Any = None
    mobilenet_device: Any = None
    mobilenet_transform: Any = None

    if backend == "template":
        gallery = _build_gray_template_gallery(piece_dir)
        if not gallery:
            raise RuntimeError(f"模板库为空: {piece_dir}")
        prof["prepare_chess_gallery"] = time.perf_counter() - t_seg
        t_seg = time.perf_counter()
    else:
        import torch

        if mobilenet_bundle is not None:
            mobilenet_model, mobilenet_device, mobilenet_transform, db_names, db_mat = mobilenet_bundle
            # 本图不重复建库；耗时见 main() 中「MobileNet 一次性准备」
        else:
            t0 = time.perf_counter()
            mobilenet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mobilenet_model = _get_mobilenet_encoder(mobilenet_device)
            mobilenet_transform = _mobilenet_transform()
            prof["mobilenet_load_model"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            piece_db = _build_mobilenet_piece_db(piece_dir, mobilenet_model, mobilenet_device, mobilenet_transform)
            if not piece_db:
                raise RuntimeError("MobileNet 特征库为空（检查 chess_gallery / report.json）")
            prof["mobilenet_build_piece_db"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            db_names, db_mat = _prepare_piece_db_matrix(piece_db)
            prof["prepare_piece_matrix"] = time.perf_counter() - t0
        prof["prepare_chess_gallery"] = time.perf_counter() - t_seg
        t_seg = time.perf_counter()

    mark = scene.copy()
    x1, y1, x2, y2 = CR_ROI
    cv2.rectangle(mark, (x1, y1), (x2, y2), (128, 128, 255), 2)
    prof["mark_roi_annotation"] = time.perf_counter() - t_seg
    t_seg = time.perf_counter()

    samples = DEFAULT_SAMPLES
    results: List[Dict[str, Any]] = []
    filtered_wild_top_zone: List[Dict[str, Any]] = []

    sz = int(max(16, template_match_size))
    t_chess_crop_acc = 0.0
    t_chess_rec_acc = 0.0

    def _one_top(bgr: np.ndarray) -> List[Tuple[str, float]]:
        if backend == "template":
            assert gallery is not None
            return _topk_template_match(bgr, gallery, size=sz, topk=int(topk_raw))
        assert db_names is not None and db_mat is not None
        assert mobilenet_model is not None and mobilenet_device is not None and mobilenet_transform is not None
        q = _embed_mobilenet_bgr(bgr, mobilenet_model, mobilenet_device, mobilenet_transform)
        return _topk_cosine(q, db_names, db_mat, topk=int(topk_raw))

    for bi, (x, y, w, h) in enumerate(boxes):
        mark_before = mark.copy()
        cv2.rectangle(mark, (x, y), (x + w, y + h), (0, 255, 0), 2)
        base_cx = x + w // 2 + CENTER_OFFSET_X
        base_cy = y + h

        agg: Dict[str, float] = {}
        votes: Dict[str, int] = {}
        per_samples: List[Dict[str, Any]] = []
        used_samples = 0

        for si, (dx, dy) in enumerate(samples):
            cx = int(base_cx + dx)
            cy = int(base_cy + dy)
            cv2.circle(mark, (cx, cy), 2, (0, 0, 255), -1)
            cv2.circle(mark, (cx, cy), r, (0, 255, 255), 2)

            _t0 = time.perf_counter()
            bgra = _crop_circle_bgra(scene, cx, cy, r)
            if alpha_tight:
                bgra = _tight_crop_by_alpha(bgra, alpha_thresh=int(alpha_thresh), pad=2)
            bgr = _bgra_to_bgr_white(bgra)
            t_chess_crop_acc += time.perf_counter() - _t0
            crop_path = crops_dir / f"bar{bi+1:02d}_stage1_s{si+1:02d}_{dx}_{dy}.png"
            if save_debug_crops:
                ok, buf = cv2.imencode(".png", bgra)
                if ok:
                    Path(crop_path).write_bytes(buf.tobytes())

            _t0 = time.perf_counter()
            top = _one_top(bgr)
            t_chess_rec_acc += time.perf_counter() - _t0
            per_samples.append({"dx": dx, "dy": dy, "top": top})
            if not top:
                continue

            top_for_filter = _collapse_topk_to_hero(top)
            if not top_for_filter:
                continue
            top1_name, top1_sim = top_for_filter[0]
            top2_sim = float(top_for_filter[1][1]) if len(top_for_filter) >= 2 else -1.0
            margin = float(top1_sim) - float(top2_sim)
            if float(top1_sim) < float(sample_min_sim) or margin < float(sample_min_margin):
                continue

            used_samples += 1
            votes[top1_name] = votes.get(top1_name, 0) + 1

            topm = max(1, int(sample_topm))
            top_hero = _collapse_topk_to_hero(top)
            use = top_hero[:topm]
            sims = np.array([s for _, s in use], dtype=np.float32)
            t = float(temp)
            z = (sims - float(np.max(sims))) / max(1e-6, t)
            ez = np.exp(z)
            probs = ez / float(np.sum(ez))
            for (name, _sim), pprob in zip(use, probs):
                agg[name] = agg.get(name, 0.0) + float(pprob)

        agg_sorted = sorted(agg.items(), key=lambda x: -x[1])[:5]
        best = agg_sorted[0][0] if agg_sorted else None
        best_score = float(agg_sorted[0][1]) if agg_sorted else 0.0
        second_score = float(agg_sorted[1][1]) if len(agg_sorted) > 1 else 0.0
        gap = best_score - second_score
        best_votes = int(votes.get(best, 0)) if best else 0

        eff_min_gap = float(min_gap)
        if best_votes >= 3:
            eff_min_gap = min(eff_min_gap, float(min_gap_3votes))
        elif best_votes >= 2:
            eff_min_gap = min(eff_min_gap, float(min_gap_2votes))

        if used_samples < 1 or best_score < float(min_sim) or gap < eff_min_gap or best_votes < int(min_votes):
            best = None

        stage = "stage1"
        stage2_detail: Optional[Dict[str, Any]] = None

        if best is None and bool(two_stage):
            stage = "stage2"
            samples2 = list(samples) + list(EXTRA_SAMPLES)
            agg2: Dict[str, float] = {}
            votes2: Dict[str, int] = {}
            top1_sum2: Dict[str, float] = {}
            top1_max2: Dict[str, float] = {}
            per_samples2: List[Dict[str, Any]] = []
            used_samples2 = 0

            for si, (dx, dy) in enumerate(samples2):
                cx = int(base_cx + dx)
                cy = int(base_cy + dy)
                _t0 = time.perf_counter()
                bgra = _crop_circle_bgra(scene, cx, cy, r)
                if alpha_tight:
                    bgra = _tight_crop_by_alpha(bgra, alpha_thresh=int(alpha_thresh), pad=2)
                bgr = _bgra_to_bgr_white(bgra)
                t_chess_crop_acc += time.perf_counter() - _t0
                crop_path = crops_dir / f"bar{bi+1:02d}_stage2_s{si+1:02d}_{dx}_{dy}.png"
                if save_debug_crops:
                    ok, buf = cv2.imencode(".png", bgra)
                    if ok:
                        Path(crop_path).write_bytes(buf.tobytes())

                _t0 = time.perf_counter()
                top = _one_top(bgr)
                t_chess_rec_acc += time.perf_counter() - _t0
                per_samples2.append({"dx": dx, "dy": dy, "top": top})
                if not top:
                    continue

                top_for_filter = _collapse_topk_to_hero(top)
                if not top_for_filter:
                    continue
                top1_name, top1_sim = top_for_filter[0]
                top2_sim = float(top_for_filter[1][1]) if len(top_for_filter) >= 2 else -1.0
                margin = float(top1_sim) - float(top2_sim)
                if float(top1_sim) < float(stage2_min_sim) or margin < float(stage2_min_margin):
                    continue

                used_samples2 += 1
                votes2[top1_name] = votes2.get(top1_name, 0) + 1
                top1_sum2[top1_name] = top1_sum2.get(top1_name, 0.0) + float(top1_sim)
                top1_max2[top1_name] = max(top1_max2.get(top1_name, -1.0), float(top1_sim))

                topm = max(1, int(sample_topm))
                top_hero = _collapse_topk_to_hero(top)
                use = top_hero[:topm]
                sims = np.array([s for _, s in use], dtype=np.float32)
                t = float(temp)
                z = (sims - float(np.max(sims))) / max(1e-6, t)
                ez = np.exp(z)
                probs = ez / float(np.sum(ez))
                for (name, _sim), pprob in zip(use, probs):
                    agg2[name] = agg2.get(name, 0.0) + float(pprob)

            agg2_sorted = sorted(agg2.items(), key=lambda x: -x[1])[:5]

            def _agg_score(name: str) -> float:
                return float(dict(agg2_sorted).get(name, 0.0))

            def _decide_key(name: str) -> Tuple[int, float, float, float]:
                return (
                    int(votes2.get(name, 0)),
                    float(top1_sum2.get(name, 0.0)),
                    float(top1_max2.get(name, -1.0)),
                    _agg_score(name),
                )

            cand = sorted(votes2.keys(), key=_decide_key, reverse=True) if votes2 else []
            best2 = cand[0] if cand else (agg2_sorted[0][0] if agg2_sorted else None)
            second2 = cand[1] if len(cand) >= 2 else None

            best2_votes = int(votes2.get(best2, 0)) if best2 else 0
            second2_votes = int(votes2.get(second2, 0)) if second2 else 0
            best2_top1sum = float(top1_sum2.get(best2, 0.0)) if best2 else 0.0
            second2_top1sum = float(top1_sum2.get(second2, 0.0)) if second2 else 0.0

            best2_score = _agg_score(best2) if best2 else 0.0
            second2_score = _agg_score(second2) if second2 else 0.0
            gap2 = best2_score - second2_score

            ok2 = True
            if used_samples2 < 1:
                ok2 = False
            elif best2_votes < int(min_votes):
                best2_top1max = float(top1_max2.get(best2, -1.0)) if best2 else -1.0
                ok2 = (
                    int(used_samples2) == 1
                    and best2_votes == 1
                    and second2_votes == 0
                    and best2_top1max >= float(stage2_single_vote_min_top1max)
                    and gap2 >= float(stage2_single_vote_min_gap_score)
                )
            else:
                if best2_votes > second2_votes:
                    ok2 = (best2_votes - second2_votes) >= 1
                else:
                    ok2 = (best2_top1sum - second2_top1sum) >= float(stage2_top1sum_gap)

            if not ok2:
                best2 = None

            stage2_detail = {
                "best": best2,
                "best_score": float(best2_score),
                "second_score": float(second2_score),
                "gap": float(gap2),
                "used_samples": int(used_samples2),
                "vote_top": sorted(
                    [{"name": k, "votes": int(v)} for k, v in votes2.items()], key=lambda x: -x["votes"]
                )[:10],
                "agg_top": [{"name": n, "score": float(s)} for (n, s) in agg2_sorted],
                "top1_sum": {k: float(v) for k, v in sorted(top1_sum2.items(), key=lambda x: -x[1])[:10]},
                "top1_max": {k: float(v) for k, v in sorted(top1_max2.items(), key=lambda x: -x[1])[:10]},
                "decision_best": best2,
                "decision_second": second2,
                "decision_votes": {"best": int(best2_votes), "second": int(second2_votes)},
                "decision_top1sum": {"best": float(best2_top1sum), "second": float(second2_top1sum)},
                "stage2_min_sim": float(stage2_min_sim),
                "stage2_min_margin": float(stage2_min_margin),
                "stage2_top1sum_gap": float(stage2_top1sum_gap),
                "stage2_extra_samples": EXTRA_SAMPLES,
                "samples": per_samples2,
                "filled_by_consensus": False,
                "filled_reason": None,
            }

            if best2 is not None:
                best = best2
                best_score = float(best2_score)
                second_score = float(second2_score)
                gap = float(gap2)
                used_samples = int(used_samples2)
                votes = votes2
                agg_sorted = agg2_sorted
                per_samples = per_samples2

        if int(used_samples) <= 0:
            mark = mark_before
            continue

        if int(y) <= int(TOP_ZONE_MAX_Y_ABS) and (best not in TOP_ZONE_ALLOWED_LABELS):
            filtered_wild_top_zone.append(
                {
                    "bar_index": bi + 1,
                    "bar_box": [int(x), int(y), int(w), int(h)],
                    "best": best,
                    "best_score": float(best_score),
                    "reason": "top_zone_non_whitelist",
                }
            )
            mark = mark_before
            continue

        if stage == "stage2" and stage2_detail:
            sd = stage2_detail
            if sd.get("filled_by_consensus"):
                fr = sd.get("filled_reason") or {}
                if fr.get("method") == "stage2_retry_min_sim":
                    wms = float(fr.get("winner_max_sim", -1.0))
                    if wms < 0.67:
                        mark = mark_before
                        continue

        results.append(
            {
                "bar_index": bi + 1,
                "bar_box": [int(x), int(y), int(w), int(h)],
                "stage": stage,
                "best": best,
                "best_score": best_score,
                "second_score": second_score,
                "gap": gap,
                "used_samples": used_samples,
                "vote_top": sorted(
                    [{"name": k, "votes": int(v)} for k, v in votes.items()], key=lambda x: -x["votes"]
                )[:10],
                "agg_top": [{"name": n, "score": float(s)} for (n, s) in agg_sorted],
                "samples": per_samples,
                "stage2": stage2_detail,
                "confidence": _compute_confidence(stage, used_samples, float(best_score), stage2_detail),
            }
        )

        _confidence = _compute_confidence(stage, used_samples, float(best_score), stage2_detail)
        if best:
            _label = best if _confidence != "low" else best + "?"
            _color = (255, 255, 255) if _confidence != "low" else (0, 200, 255)
            _draw_chinese_text(mark, _label, (x, max(0, y - 22)), font_size=18, color=_color)
        else:
            cv2.putText(mark, "?", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    prof["chess_crop_all_samples"] = t_chess_crop_acc
    prof["chess_recognize_all_samples"] = t_chess_rec_acc
    prof["chess_bar_loop"] = time.perf_counter() - t_seg
    t_seg = time.perf_counter()

    if results:
        cells = _board_cells()
        row_ys = _row_ys()
        bottoms_all = [int(r["bar_box"][1] + r["bar_box"][3]) for r in results]
        for r in results:
            bx, by, bw, bh = map(int, r["bar_box"])
            tall_ex = _tall_bar_extra_foot_dy(by, bh, bottoms_all)
            ax = float(bx + bw / 2.0)
            ay = float(by + bh + _FOOT_DY + tall_ex)
            cell, dist = _nearest_cell_row_first(
                ax,
                ay,
                cells,
                row_ys=row_ys,
                bar_top_y=float(by),
                bar_bottom_y=float(by + bh),
            )
            inferred_row = _infer_row_for_mapping(ay, float(by), float(by + bh), row_ys)
            if cell is None:
                r["position"] = {
                    "anchor_mode": "bar",
                    "anchor_xy": [round(ax, 2), round(ay, 2)],
                    "inferred_row_from_y": int(inferred_row),
                    "cell_row": None,
                    "cell_col": None,
                    "label": "?",
                    "dist_to_cell_px": round(float(dist), 2),
                }
                continue
            crow, ccol, _cx, _cy = cell
            pos_label = f"第{crow}排第{ccol}格"
            r["position"] = {
                "anchor_mode": "bar",
                "anchor_xy": [round(ax, 2), round(ay, 2)],
                "inferred_row_from_y": int(inferred_row),
                "cell_row": int(crow),
                "cell_col": int(ccol),
                "label": pos_label,
                "dist_to_cell_px": round(float(dist), 2),
            }
            _draw_chinese_text(mark, pos_label, (bx, max(0, by - 40)), font_size=14, color=(120, 255, 120))
            cv2.circle(mark, (int(round(ax)), int(round(ay))), 3, (120, 255, 120), -1)

    prof["position_mapping"] = time.perf_counter() - t_seg
    t_seg = time.perf_counter()

    mark_suffix = "MobileNetV3_标记.png" if backend == "mobilenet_v3_small" else "多点Template_标记.png"
    mark_path = output_dir / f"{img_path.stem}_{mark_suffix}"
    _save_image(mark, mark_path)

    prof["save_outputs"] = time.perf_counter() - t_seg
    prof["total_recognition"] = time.perf_counter() - t_prof0

    if print_timings:
        _print_timing_block(f"fightboard 棋子管线 | {img_path.name}", prof)

    out_json: Dict[str, Any] = {
        "image": str(img_path),
        "piece_dir": str(piece_dir),
        "backend": backend,
        "bar_detect_strategy": str(bar_detect_strategy),
        "bar_detect_simple_threshold": float(bar_detect_simple_threshold),
        "bar_width_snap": bool(bar_width_snap),
        "bar_snap_width": int(snap_w),
        "bar_clip_to_content": bool(bar_clip_to_content),
        "timings_s": prof,
        "template_match_size": sz,
        "roi": list(CR_ROI),
        "circle_diameter": circle_diameter,
        "below_bar_px": BELOW_BAR_PX,
        "center_offset_x": CENTER_OFFSET_X,
        "samples": samples,
        "results": results,
        "top_zone_rule": {
            "y_max_abs": int(TOP_ZONE_MAX_Y_ABS),
            "allowed_labels": sorted(list(TOP_ZONE_ALLOWED_LABELS)),
            "filtered_count": int(len(filtered_wild_top_zone)),
        },
        "filtered_wild_top_zone": filtered_wild_top_zone,
    }
    (output_dir / "result.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def run_recognition_template(**kwargs: Any) -> Dict[str, Any]:
    """兼容旧名：等价于 ``run_recognition_chess(..., backend=\"template\")``。"""
    kwargs.pop("backend", None)
    return run_recognition_chess(backend="template", **kwargs)


def _detect_one_bar_equip(
    scene_bgr: "cv2.Mat",
    *,
    bar_box_xywh: Sequence[int],
    templates: List[Tuple[str, "Any"]],
    scales: Sequence[int],
    method: int,
    threshold: float,
    max_peaks_per_scale: int,
    top_k: int,
    nms_iou: float,
    below_px: int,
    crop_w: int,
    crop_h: int,
    min_roi: int,
    blue_buff_gap_min: float,
    label_topn: int,
) -> List[Dict[str, Any]]:
    x, y, w, h = map(int, bar_box_xywh)
    crop_bgr, (crop_left, crop_top, _crop_aw, _crop_ah) = er.crop_below_bar_center(
        scene_bgr,
        (x, y, w, h),
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        below_px=int(below_px),
    )
    roi_g = er._to_gray(crop_bgr)
    rh, rw = roi_g.shape[:2]
    if rw < int(min_roi) or rh < int(min_roi):
        return []

    candidates: List[Tuple[float, int, int, str, int]] = []
    for name, tmpl_gray in templates:
        peaks = er._collect_peaks_for_template(
            roi_g,
            tmpl_gray,
            scales,
            method,
            float(threshold),
            int(max_peaks_per_scale),
        )
        for score, px, py, side_win in peaks:
            candidates.append((score, int(px), int(py), str(name), int(side_win)))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)
    boxes_xy = [(c[1], c[2], c[4], c[4]) for c in candidates]
    scores = [c[0] for c in candidates]
    keep_idx = er._nms_xywh(boxes_xy, scores, float(nms_iou)) if candidates else []
    picked = [candidates[i] for i in keep_idx[: int(top_k)]]

    picked = er._apply_blue_buff_special_case(picked, gap_min=float(blue_buff_gap_min))
    if not picked:
        return []

    picked = er._apply_spatula_special_case(picked)
    if not picked:
        return []

    out: List[Dict[str, Any]] = []
    for rank, (score, px, py, tname, side_win) in enumerate(picked[: int(label_topn)], start=1):
        gx1 = int(crop_left + int(px))
        gy1 = int(crop_top + int(py))
        out.append(
            {
                "name": er._equip_label_stem(tname),
                "score": float(score),
                "gx1": gx1,
                "gy1": gy1,
                "side": int(side_win),
                "rank": int(rank),
            }
        )
    return out


def _overlay_fightboard(
    *,
    scene_bgr: "cv2.Mat",
    results: List[Dict[str, Any]],
    equip_by_bar: Dict[int, List[Dict[str, Any]]],
    font_size: int,
) -> "cv2.Mat":
    vis = scene_bgr.copy()
    for r in results:
        bar_index = int(r.get("bar_index") or 0)
        bar_box = r.get("bar_box") or [0, 0, 0, 0]
        x, y, w, h = map(int, bar_box)
        piece = r.get("best") or "?"
        conf = r.get("confidence") or ""
        piece_label = f"{piece}?" if str(conf) == "low" else str(piece)
        text_color = (0, 200, 255) if str(conf) == "low" else (0, 255, 0)
        pos = "?"
        pos_obj = r.get("position") or {}
        if isinstance(pos_obj, dict):
            pos = pos_obj.get("label") or "?"

        line1 = f"{piece_label} {pos}".strip()
        tx, ty = x, max(0, y - (int(font_size) + 8))
        cr._draw_chinese_text(
            vis,
            line1,
            (int(tx), int(ty)),
            font_size=int(font_size),
            color=text_color,
        )

        equip_list = equip_by_bar.get(bar_index) or []
        if equip_list:
            for eq in equip_list:
                gx1 = int(eq.get("gx1") or x)
                gy1 = int(eq.get("gy1") or (y + h + 2))
                side = int(eq.get("side") or 0)
                rank = int(eq.get("rank") or 1)
                eq_name = str(eq.get("name") or "?")
                score = float(eq.get("score") or 0.0)

                cv2.rectangle(
                    vis,
                    (gx1, gy1),
                    (min(vis.shape[1] - 1, gx1 + side), min(vis.shape[0] - 1, gy1 + side)),
                    (0, 200, 255),
                    1,
                )

                text_x = max(0, min(gx1, vis.shape[1] - 1))
                text_y = max(0, min(gy1 + (rank - 1) * (int(font_size) + 3), vis.shape[0] - 1))
                cr._draw_chinese_text(
                    vis,
                    f"{eq_name} {score:.2f}",
                    (int(text_x), int(text_y)),
                    font_size=max(10, int(font_size) - 2),
                    color=(255, 255, 0),
                )
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(
        description="fightboard_recog_v2：棋子（模板 或 MobileNetV3-Small 特征检索）+ 装备模板匹配"
    )
    ap.add_argument("--img-dir", type=Path, default=DEFAULT_INPUT, help="对局截图文件夹/单张图")
    ap.add_argument("--piece-dir", type=Path, default=DEFAULT_PIECE_DIR, help="chess_gallery")
    ap.add_argument("--equip-gallery", type=Path, default=DEFAULT_EQUIP_GALLERY, help="装备图鉴目录")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出根目录（会清空）")
    ap.add_argument(
        "--chess-backend",
        choices=["mobilenet_v3_small", "template"],
        default="mobilenet_v3_small",
        help="棋子识别：MobileNetV3-Small 特征检索（默认）或灰度模板",
    )
    ap.add_argument(
        "--bar-detect-strategy",
        choices=["simple_tiled", "simple_tiled_edges"],
        default="simple_tiled",
        help="血条检测：_detect_healthbars_in_roi（细切块+TM）；simple_tiled_edges 在 Sobel 边缘平面上匹配",
    )
    ap.add_argument(
        "--bar-detect-simple-threshold",
        type=float,
        default=0.58,
        help="simple_tiled / simple_tiled_edges 下的模板匹配阈值（0~1，越高越严；默认 0.58 以召回略弱匹配条如部分顶行）",
    )
    bar_snap = ap.add_mutually_exclusive_group()
    bar_snap.add_argument(
        "--bar-width-snap",
        dest="bar_width_snap",
        action="store_true",
        help="检测后将血条框宽钉为固定宽度，并相对检测框水平居中（默认开启）",
    )
    bar_snap.add_argument(
        "--no-bar-width-snap",
        dest="bar_width_snap",
        action="store_false",
        help="关闭宽度钉档，使用模板匹配原始框宽",
    )
    ap.set_defaults(bar_width_snap=True, bar_clip_to_content=True)
    ap.add_argument(
        "--bar-snap-width",
        type=int,
        default=FIGHTBOARD_BAR_SNAP_WIDTH_DEFAULT,
        help=f"钉档时的固定框宽（像素），相对检测框居中；默认 {FIGHTBOARD_BAR_SNAP_WIDTH_DEFAULT}",
    )
    ap.add_argument(
        "--no-bar-clip-to-content",
        dest="bar_clip_to_content",
        action="store_false",
        help="关闭「钉档后按检测框内列投影收束」；默认开启，使定宽框不超出估计的血条左右端",
    )
    ap.add_argument("--circle-diameter", type=int, default=84, help="棋子采样圆直径")
    ap.add_argument("--alpha-tight", action="store_true", default=True, help="alpha 紧裁（默认开）")
    ap.add_argument("--template-match-size", type=int, default=64, help="模板与 crop 对齐边长（像素）")
    ap.add_argument("--save-debug-crops", action="store_true", help="写出 chess crops PNG")
    ap.add_argument("--equip-threshold", type=float, default=0.78, help="装备模板匹配阈值")
    ap.add_argument("--equip-scales", type=str, default="24,25,26,27,28", help="装备多尺度")
    ap.add_argument("--equip-max-peaks-per-scale", type=int, default=4)
    ap.add_argument("--equip-top-k", type=int, default=15)
    ap.add_argument("--equip-nms-iou", type=float, default=0.35)
    ap.add_argument("--equip-width", type=int, default=120)
    ap.add_argument("--equip-height", type=int, default=50)
    ap.add_argument("--equip-below-px", type=int, default=2)
    ap.add_argument("--blue-buff-gap-min", type=float, default=0.05)
    ap.add_argument("--equip-label-topn", type=int, default=3)
    ap.add_argument("--label-font-size", type=int, default=16)
    ap.add_argument("--json", action="store_true", help="每张图额外输出 fightboard summary json")
    args = ap.parse_args()

    img_dir = args.img_dir.resolve()
    if not img_dir.exists():
        raise SystemExit(f"img-dir 不存在: {img_dir}")
    piece_dir = args.piece_dir.resolve()
    if not piece_dir.is_dir():
        raise SystemExit(f"piece-dir 不存在: {piece_dir}")
    equip_gallery_dir = args.equip_gallery.resolve()
    if not equip_gallery_dir.is_dir():
        raise SystemExit(f"equip-gallery 不存在: {equip_gallery_dir}")

    out_root = args.out.resolve()
    if out_root.exists():
        import shutil

        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    templates = er._build_templates(equip_gallery_dir)
    if not templates:
        raise SystemExit(f"equip-gallery 无可用图片: {equip_gallery_dir}")
    method = cv2.TM_CCOEFF_NORMED
    scales = tuple(int(x.strip()) for x in args.equip_scales.split(",") if x.strip())
    if not scales or any(s < 2 for s in scales):
        raise SystemExit("--equip-scales 无效")
    min_roi = max(scales)

    bar_tpl = br.find_healthbar_template(PROJECT_DIR)

    if bool(args.bar_width_snap) and int(args.bar_snap_width) < 1:
        raise SystemExit("--bar-snap-width 在开启钉档时须为正整数")

    mobilenet_bundle: Optional[Tuple[Any, Any, Any, List[str], np.ndarray]] = None
    if str(args.chess_backend) == "mobilenet_v3_small":
        import torch

        t_prep0 = time.perf_counter()
        mobilenet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        mobilenet_model = _get_mobilenet_encoder(mobilenet_device)
        t_load_model = time.perf_counter() - t0
        mobilenet_transform = _mobilenet_transform()
        t0 = time.perf_counter()
        piece_db = _build_mobilenet_piece_db(piece_dir, mobilenet_model, mobilenet_device, mobilenet_transform)
        t_build_db = time.perf_counter() - t0
        if not piece_db:
            raise SystemExit(f"MobileNet 特征库为空: {piece_dir}")
        t0 = time.perf_counter()
        db_names, db_mat = _prepare_piece_db_matrix(piece_db)
        t_prep_matrix = time.perf_counter() - t0
        mobilenet_bundle = (mobilenet_model, mobilenet_device, mobilenet_transform, db_names, db_mat)
        _print_timing_block(
            "MobileNet 一次性准备（本批次共用，无磁盘缓存）",
            {
                "mobilenet_load_model": t_load_model,
                "mobilenet_build_piece_db": t_build_db,
                "prepare_piece_matrix": t_prep_matrix,
                "mobilenet_prep_total": time.perf_counter() - t_prep0,
            },
        )

    images = br.iter_input_images(img_dir)
    for image_path in images:
        stem = image_path.stem
        scene_bgr = cr._load_image(image_path)
        with tempfile.TemporaryDirectory(prefix="fightboard_recog_v2_chess_") as td:
            chess_out_dir = Path(td) / "chess"
            chess_out_dir.mkdir(parents=True, exist_ok=True)
            out_json = run_recognition_chess(
                backend=str(args.chess_backend),
                image_path=image_path,
                template_path=bar_tpl,
                piece_dir=piece_dir,
                output_dir=chess_out_dir,
                circle_diameter=int(args.circle_diameter),
                alpha_tight=bool(args.alpha_tight),
                template_match_size=int(args.template_match_size),
                save_debug_crops=bool(args.save_debug_crops),
                bar_detect_strategy=str(args.bar_detect_strategy),
                bar_detect_simple_threshold=float(args.bar_detect_simple_threshold),
                mobilenet_bundle=mobilenet_bundle,
                bar_width_snap=bool(args.bar_width_snap),
                bar_snap_width=int(args.bar_snap_width),
                bar_clip_to_content=bool(args.bar_clip_to_content),
            )
            results: List[Dict[str, Any]] = out_json.get("results") or []

        t_equip0 = time.perf_counter()
        equip_by_bar: Dict[int, List[Dict[str, Any]]] = {}
        for r in results:
            bar_index = int(r.get("bar_index") or 0)
            bar_box = r.get("bar_box") or [0, 0, 0, 0]
            equip_by_bar[bar_index] = _detect_one_bar_equip(
                scene_bgr,
                bar_box_xywh=bar_box,
                templates=templates,
                scales=scales,
                method=method,
                threshold=float(args.equip_threshold),
                max_peaks_per_scale=int(args.equip_max_peaks_per_scale),
                top_k=int(args.equip_top_k),
                nms_iou=float(args.equip_nms_iou),
                below_px=int(args.equip_below_px),
                crop_w=int(args.equip_width),
                crop_h=int(args.equip_height),
                min_roi=int(min_roi),
                blue_buff_gap_min=float(args.blue_buff_gap_min),
                label_topn=int(args.equip_label_topn),
            )

        t_equip_done = time.perf_counter()
        t_vis0 = time.perf_counter()
        vis = _overlay_fightboard(
            scene_bgr=scene_bgr,
            results=results,
            equip_by_bar=equip_by_bar,
            font_size=int(args.label_font_size),
        )
        t_overlay = time.perf_counter() - t_vis0
        out_path = out_root / f"{stem}_fightboard_综合标注.png"
        t_save0 = time.perf_counter()
        cr._save_image(vis, out_path)
        t_save_img = time.perf_counter() - t_save0
        _print_timing_block(
            f"本图后续阶段 | {image_path.name}",
            {
                "equip_detect_all_bars": t_equip_done - t_equip0,
                "overlay_composite": t_overlay,
                "save_annotated_png": t_save_img,
            },
        )
        if args.json:
            chess_timings = out_json.get("timings_s") or {}
            summary = {
                "file": image_path.name,
                "pipeline": "fightboard_recog_v2",
                "chess_backend": str(args.chess_backend),
                "bar_detect_strategy": str(args.bar_detect_strategy),
                "bar_detect_simple_threshold": float(args.bar_detect_simple_threshold),
                "bar_width_snap": bool(out_json.get("bar_width_snap")),
                "bar_snap_width": int(out_json.get("bar_snap_width") or 0),
                "bar_clip_to_content": bool(out_json.get("bar_clip_to_content")),
                "annotated_image": out_path.name,
                "timings_s": {
                    **dict(chess_timings),
                    "equip_detect_all_bars": float(t_equip_done - t_equip0),
                    "overlay_composite": float(t_overlay),
                    "save_annotated_png": float(t_save_img),
                },
                "results": results,
                "equip_by_bar": equip_by_bar,
            }
            out_json_path = out_root / f"{stem}_fightboard_summary.json"
            out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {image_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
