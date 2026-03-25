# -*- coding: utf-8 -*-
"""
bars_recog：血条管线共用——模板路径、ROI 字符串解析、输入图枚举、result.json、血条框来源（bars-from）解析。

与 `chess_recog` 中 `default_batch_result_json` / `load_bar_boxes_from_result_json` 及
`equip_recog` 约定一致。

**后处理（与 `chess_recog.run_recognition` 对齐，默认开启）**
  - **空框 / 无有效 embedding**：读 result.json 时跳过 `used_samples<=0` 的条目。
  - **野怪 / 顶小条弱证据**：读 result.json 时跳过弱证据顶小条。
  - **顶带野怪小龙**：仅「极窄」条再滤（与 chess_recog.is_wild_dragon_strip_geometry 一致），避免误伤纳什顶条。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    from .chess_recog import (
        ROI as DEFAULT_VOTE_ROI,
        _detect_healthbars_in_roi,
        default_batch_result_json,
        is_wild_dragon_strip_geometry,
        load_bar_boxes_from_result_json,
    )
except ImportError:
    from chess_recog import (
        ROI as DEFAULT_VOTE_ROI,
        _detect_healthbars_in_roi,
        default_batch_result_json,
        is_wild_dragon_strip_geometry,
        load_bar_boxes_from_result_json,
    )

# 与 chess_recog 顶带小龙几何默认一致
DEFAULT_DRAGON_STRIP_BAND_Y = 90
DEFAULT_DRAGON_STRIP_MAX_W = 70
DEFAULT_DRAGON_STRIP_MAX_H = 12
# 与 `_should_drop_non_chess_bar` 中「顶带」相对 ROI 顶向下的偏移一致
DEFAULT_INPUT_IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def filter_degenerate_bar_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """去掉 w/h≤0 的退化框。"""
    out: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        _x, _y, w, h = map(int, b)
        if w <= 0 or h <= 0:
            continue
        out.append((int(b[0]), int(b[1]), int(w), int(h)))
    return out


def is_dragon_strip_bar(
    y_abs: int,
    w: int,
    h: int,
    roi_y1: int,
    *,
    band_y: int = DEFAULT_DRAGON_STRIP_BAND_Y,
    max_w: int = DEFAULT_DRAGON_STRIP_MAX_W,
    max_h: int = DEFAULT_DRAGON_STRIP_MAX_H,
) -> bool:
    """
    野怪小龙「极窄条」判定，与 chess_recog.is_wild_dragon_strip_geometry 一致。
    max_w/max_h 仅保留兼容旧 CLI，实际规则以 chess_recog 为准（纳什顶条略宽则保留）。
    """
    return bool(
        is_wild_dragon_strip_geometry(
            y_abs=int(y_abs),
            w=int(w),
            h=int(h),
            roi_y1=int(roi_y1),
            band_y=int(band_y),
        )
    )


def filter_dragon_strip_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    roi_y1: int,
    *,
    band_y: int = DEFAULT_DRAGON_STRIP_BAND_Y,
    max_w: int = DEFAULT_DRAGON_STRIP_MAX_W,
    max_h: int = DEFAULT_DRAGON_STRIP_MAX_H,
) -> List[Tuple[int, int, int, int]]:
    """剔除 ROI 顶带内「极窄野怪小龙」条（与 chess_recog 检测阶段一致）。"""
    if band_y <= 0 or max_w <= 0 or max_h <= 0:
        return list(boxes)
    out: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        _bx, by, bw, bh = map(int, b)
        if is_dragon_strip_bar(by, bw, bh, roi_y1, band_y=band_y, max_w=max_w, max_h=max_h):
            continue
        out.append((int(b[0]), int(b[1]), int(bw), int(bh)))
    return out


def should_drop_empty_embedding_bar(used_samples: int) -> bool:
    """与 `run_recognition` 末尾一致：无有效采样则丢弃该条。"""
    return int(used_samples) <= 0


def load_bar_boxes_from_result_json_filtered(path: Path) -> List[Tuple[int, int, int, int]]:
    """
    读取 result.json 中 bar_box，并应用与 `run_recognition` 写入前一致的筛除：
    - `used_samples<=0`（空框 / 无 embedding 证据）
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    roi = obj.get("roi")
    roi_y1 = int(roi[1]) if roi and len(roi) >= 2 else int(DEFAULT_VOTE_ROI[1])
    out: List[Tuple[int, int, int, int]] = []
    for it in obj.get("results") or []:
        bb = it.get("bar_box")
        if not bb or len(bb) != 4:
            continue
        x, y, w, h = map(int, bb)
        # 与 batch 新版 JSON 一致；缺字段的旧文件视为已通过识别（避免整表被当空框删光）
        if "used_samples" in it:
            used_samples = int(it.get("used_samples") or 0)
        else:
            used_samples = 1
        if "best_score" in it:
            best_score = float(it.get("best_score") or 0.0)
        else:
            best_score = 1.0
        if should_drop_empty_embedding_bar(used_samples):
            continue
        out.append((x, y, w, h))
    return out


def apply_standard_bar_post_filters(
    boxes: Sequence[Tuple[int, int, int, int]],
    roi_y1: int,
    *,
    apply_dragon_strip: bool = False,
    dragon_band_y: int = DEFAULT_DRAGON_STRIP_BAND_Y,
    dragon_max_w: int = DEFAULT_DRAGON_STRIP_MAX_W,
    dragon_max_h: int = DEFAULT_DRAGON_STRIP_MAX_H,
    post_stats: Optional[Dict[str, int]] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    对已有条列表做退化剔除 + 可选顶带小龙几何滤除（模板检测路径二次防御；与 `_detect_healthbars_in_roi` 默认参数对齐）。
    """
    b = filter_degenerate_bar_boxes(boxes)
    if post_stats is not None:
        post_stats["degenerate_dropped"] = int(len(boxes) - len(b))
    before = b
    if apply_dragon_strip:
        b = filter_dragon_strip_boxes(
            b,
            roi_y1,
            band_y=dragon_band_y,
            max_w=dragon_max_w,
            max_h=dragon_max_h,
        )
    if post_stats is not None:
        post_stats["dragon_strip_dropped"] = int(len(before) - len(b))
    return b


def find_healthbar_template(project_root: Path) -> Path:
    """
    优先目录「血条模板/」，否则「血条参考.png」，否则根目录下文件名含「参考」的 png。
    """
    root = project_root
    d = root / "血条模板"
    if d.is_dir():
        return d
    cand = root / "血条参考.png"
    if cand.is_file():
        return cand
    for p in root.glob("*.png"):
        if "参考" in p.stem and p.is_file():
            return p
    raise FileNotFoundError("未找到血条模板（血条模板/ 或 血条参考.png）")


def resolve_bloodbar_template_path(template_arg: Optional[Union[str, Path]], project_dir: Path) -> Path:
    """--template 省略时自动查找；否则相对项目根解析为绝对路径。"""
    if template_arg is None:
        return find_healthbar_template(project_dir)
    s = str(template_arg).strip()
    if not s:
        return find_healthbar_template(project_dir)
    p = Path(template_arg)
    return p.resolve() if p.is_absolute() else (project_dir / p).resolve()


def parse_roi_string(s: str) -> Tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 需为 x1,y1,x2,y2 四个整数")
    return (parts[0], parts[1], parts[2], parts[3])


def iter_input_images(path: Path, image_exts: Optional[Set[str]] = None) -> List[Path]:
    exts = image_exts if image_exts is not None else DEFAULT_INPUT_IMAGE_EXTS
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() not in exts:
            raise SystemExit(f"不支持的图片格式（{exts}）: {path}")
        return [path]
    if path.is_dir():
        files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: p.name.lower())
        if not files:
            raise SystemExit(f"目录内无图片: {path}")
        return files
    raise SystemExit(f"路径不存在: {path}")


def roi_from_result_json(path: Path) -> Optional[Tuple[int, int, int, int]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        r = obj.get("roi")
        if r and len(r) == 4:
            return (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
    except Exception:
        pass
    return None


def result_json_candidate_path(
    project_dir: Path,
    image_path: Path,
    result_json_opt: str,
    *,
    embed_dirname: str,
) -> Path:
    """与 equip_recog 一致：显式路径相对 project_dir；否则默认 batch 输出路径。"""
    rjs = (result_json_opt or "").strip()
    if rjs:
        rjp = Path(rjs)
        return (rjp.resolve() if rjp.is_absolute() else (project_dir / rjp).resolve())
    return default_batch_result_json(project_dir, image_path, embed_dirname=embed_dirname)


def resolve_bars_boxes(
    scene: np.ndarray,
    templates: List[np.ndarray],
    *,
    project_dir: Path,
    image_path: Path,
    roi_start: Tuple[int, int, int, int],
    roi_user: bool,
    bars_from: str,
    result_json_opt: str,
    batch_embed_dirname: str,
    post_filter: bool = True,
    apply_json_recognition_filters: bool = False,
    apply_dragon_strip: bool = False,
    dragon_band_y: int = DEFAULT_DRAGON_STRIP_BAND_Y,
    dragon_max_w: int = DEFAULT_DRAGON_STRIP_MAX_W,
    dragon_max_h: int = DEFAULT_DRAGON_STRIP_MAX_H,
    post_stats: Optional[Dict[str, int]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], str, Optional[Path], Tuple[int, int, int, int]]:
    """
    返回 (boxes, source, result_json_path, roi)。
    source: result_json | template_detect

    post_filter：总开关；为 True 时默认做退化框剔除 + 顶带小龙几何滤除。
    apply_json_recognition_filters：从 result.json 读条时使用「空框 + 野怪弱证据」过滤（与 run_recognition 输出一致）。
    apply_dragon_strip：顶带小龙几何滤除（模板检测路径下 `_detect_healthbars_in_roi` 已有一层，此处为二次对齐）。
    post_stats：可选写入 degenerate_dropped、dragon_strip_dropped。
    """
    roi = roi_start
    result_json_path: Optional[Path] = None
    cand = result_json_candidate_path(
        project_dir, image_path, result_json_opt, embed_dirname=batch_embed_dirname
    )
    bf = (bars_from or "").strip().lower()

    def _finish(boxes: List[Tuple[int, int, int, int]], src: str) -> Tuple[List[Tuple[int, int, int, int]], str, Optional[Path], Tuple[int, int, int, int]]:
        if not post_filter:
            return boxes, src, result_json_path, roi
        y1 = int(roi[1])
        out = apply_standard_bar_post_filters(
            boxes,
            y1,
            apply_dragon_strip=apply_dragon_strip,
            dragon_band_y=dragon_band_y,
            dragon_max_w=dragon_max_w,
            dragon_max_h=dragon_max_h,
            post_stats=post_stats,
        )
        return out, src, result_json_path, roi

    if bf == "result-json":
        if not cand.is_file():
            raise SystemExit(f"--bars-from result-json 需要有效文件: {cand}")
        if post_filter and apply_json_recognition_filters:
            boxes = load_bar_boxes_from_result_json_filtered(cand)
        else:
            boxes = load_bar_boxes_from_result_json(cand)
        result_json_path = cand
        if not roi_user:
            rr = roi_from_result_json(cand)
            if rr is not None:
                roi = rr
        return _finish(list(boxes), "result_json")

    if bf == "auto":
        if cand.is_file():
            if post_filter and apply_json_recognition_filters:
                boxes = load_bar_boxes_from_result_json_filtered(cand)
            else:
                boxes = load_bar_boxes_from_result_json(cand)
            result_json_path = cand
            if not roi_user:
                rr = roi_from_result_json(cand)
                if rr is not None:
                    roi = rr
            return _finish(list(boxes), "result_json")
        boxes = _detect_healthbars_in_roi(scene, templates, roi)
        return _finish(list(boxes), "template_detect")

    if bf == "detect":
        boxes = _detect_healthbars_in_roi(scene, templates, roi)
        return _finish(list(boxes), "template_detect")

    raise SystemExit(f"未知 --bars-from: {bars_from!r}（需 auto/detect/result-json）")


def resolve_bars_for_one_image(
    image_path: Path,
    *,
    project_dir: Path,
    bars_from: str,
    result_json_opt: Optional[str],
    template_opt: Optional[str],
    roi_user: bool,
    roi_start: Tuple[int, int, int, int],
    batch_embed_dirname: str,
    post_filter: bool = True,
    apply_json_recognition_filters: bool = False,
    apply_dragon_strip: bool = False,
    dragon_band_y: int = DEFAULT_DRAGON_STRIP_BAND_Y,
    dragon_max_w: int = DEFAULT_DRAGON_STRIP_MAX_W,
    dragon_max_h: int = DEFAULT_DRAGON_STRIP_MAX_H,
    post_stats: Optional[Dict[str, int]] = None,
) -> Tuple[
    List[Tuple[int, int, int, int]],
    Optional[Path],
    Tuple[int, int, int, int],
    str,
    Optional[Path],
]:
    """
    equip_recog 等同路径用：优先读 result.json 时不预读整图；需模板检测时再读图。
    返回 (boxes, template_path, roi, resolved_label, result_json_path)；
    resolved_label 为 result-json | detect（与旧脚本 JSON/meta 一致）。
    后处理参数与 `resolve_bars_boxes` 一致（默认启用野怪几何滤除 + JSON 空框/弱证据滤除）。
    """
    from chess_recog import _load_image, load_healthbar_templates

    cand = result_json_candidate_path(
        project_dir, image_path, result_json_opt or "", embed_dirname=batch_embed_dirname
    )
    bf = (bars_from or "").strip().lower()
    tpl_path: Optional[Path] = None
    result_json_path: Optional[Path] = None

    def _roi_after_json(rjp: Path, roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if roi_user:
            return roi
        rr = roi_from_result_json(rjp)
        return rr if rr is not None else roi

    def _post_json_boxes(boxes: List[Tuple[int, int, int, int]], roi: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        if not post_filter:
            return boxes
        return apply_standard_bar_post_filters(
            boxes,
            int(roi[1]),
            apply_dragon_strip=apply_dragon_strip,
            dragon_band_y=dragon_band_y,
            dragon_max_w=dragon_max_w,
            dragon_max_h=dragon_max_h,
            post_stats=post_stats,
        )

    if bf == "result-json":
        if not cand.is_file():
            raise SystemExit(f"--bars-from result-json 需要有效文件: {cand}")
        if post_filter and apply_json_recognition_filters:
            boxes = load_bar_boxes_from_result_json_filtered(cand)
        else:
            boxes = load_bar_boxes_from_result_json(cand)
        result_json_path = cand
        roi = _roi_after_json(cand, roi_start)
        boxes = _post_json_boxes(list(boxes), roi)
        return boxes, tpl_path, roi, "result-json", result_json_path

    if bf == "auto" and cand.is_file():
        if post_filter and apply_json_recognition_filters:
            boxes = load_bar_boxes_from_result_json_filtered(cand)
        else:
            boxes = load_bar_boxes_from_result_json(cand)
        result_json_path = cand
        roi = _roi_after_json(cand, roi_start)
        boxes = _post_json_boxes(list(boxes), roi)
        return boxes, tpl_path, roi, "result-json", result_json_path

    tpl_path = resolve_bloodbar_template_path(template_opt, project_dir)
    if not tpl_path.exists():
        raise SystemExit(f"血条模板不存在: {tpl_path}")

    scene = _load_image(image_path)
    templates = load_healthbar_templates(tpl_path)
    sub_bf = "auto" if bf == "auto" else "detect"
    boxes, _src, rj, roi = resolve_bars_boxes(
        scene,
        templates,
        project_dir=project_dir,
        image_path=image_path,
        roi_start=roi_start,
        roi_user=roi_user,
        bars_from=sub_bf,
        result_json_opt=result_json_opt or "",
        batch_embed_dirname=batch_embed_dirname,
        post_filter=post_filter,
        apply_json_recognition_filters=apply_json_recognition_filters,
        apply_dragon_strip=apply_dragon_strip,
        dragon_band_y=dragon_band_y,
        dragon_max_w=dragon_max_w,
        dragon_max_h=dragon_max_h,
        post_stats=post_stats,
    )
    if bf == "auto":
        print(f"[auto] {image_path.name}: 未找到 result.json，改用模板检测: {cand}")
    return boxes, tpl_path, roi, "detect", rj
