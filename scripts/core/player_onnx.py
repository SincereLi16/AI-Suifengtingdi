# -*- coding: utf-8 -*-
"""
player_onnx：对局截图固定 ROI 的玩家信息识别（ONNX Runtime + 可选 DirectML）。

特点：
- OCR 引擎使用 rapidocr_onnxruntime（PP-OCR ONNX）
- 设备策略支持 auto/dml/cuda/cpu（不可用时自动降级）
- 主流程为 one-pass ROI 拼图识别（本文件内联实现，不依赖独立测试脚本）
- 支持批量输出 stitched 图与 summary.json（默认写出，可用 --no-json 关闭）
- 主图（文件名形如 *-a）走完整 ROI；副图（*-b 等，非 a 尾缀）仅识别羁绊栏；同序号一对只写一份
  {序号}_player_onnx_summary.json，羁绊栏为 a 与副图识别结果合并
"""

from __future__ import annotations

import sys
from pathlib import Path

for _d in Path(__file__).resolve().parents:
    if (_d / "repo_sys_path.py").exists():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
import repo_sys_path  # noqa: F401

import argparse
import copy
import difflib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR

import player_recog as pr
from project_paths import DEFAULT_OUT_PLAYER_ONNX, PROJECT_ROOT

_thread_local = threading.local()


def create_ocr_engine(device_kwargs: Dict[str, bool] | None = None) -> RapidOCR:
    kw = {"print_verbose": False}
    if device_kwargs:
        kw.update(device_kwargs)
    return RapidOCR(**kw)


@dataclass
class PatchMeta:
    key: str
    name: str
    src_rect: Tuple[int, int, int, int]
    stitch_rect: Tuple[int, int, int, int]


def _clamp_rect(rect: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def _scale_hp_rect_ref_to_image(
    rect: Tuple[int, int, int, int],
    w_img: int,
    h_img: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    sx = w_img / float(pr._HP_REF_W)
    sy = h_img / float(pr._HP_REF_H)
    nx1 = max(0, int(round(x1 * sx)))
    ny1 = max(0, int(round(y1 * sy)))
    nx2 = min(w_img, int(round(x2 * sx)))
    ny2 = min(h_img, int(round(y2 * sy)))
    if nx2 <= nx1:
        nx2 = min(w_img, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h_img, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _build_patches(bgr: np.ndarray, *, image_stem: str) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
    h, w = bgr.shape[:2]
    centers, rects = pr._default_layout()
    out: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
    bond_dy = pr.BOND_ROI_AUXILIARY_DY_PX if pr._stem_uses_auxiliary_bond_roi_shift(image_stem) else 0

    for c in centers:
        rect = _clamp_rect(c.to_rect(w, h), w, h)
        if c.key == "gold":
            x1, y1, x2, y2 = rect
            nx2 = x1 + int(round((x2 - x1) * 0.40))
            rect = _clamp_rect((x1, y1, nx2, y2), w, h)
        if c.key == "streak":
            x1, y1, x2, y2 = rect
            rect = _clamp_rect((x1, y1 + 10, x2 - 20, y2), w, h)
        out.append((c.key, c.name, rect))
    bonds_rect: Tuple[int, int, int, int] | None = None
    for r in rects:
        rect = (int(r.x1), int(r.y1), int(r.x2), int(r.y2))
        if r.key == "bonds" and bond_dy != 0:
            x1, y1, x2, y2 = rect
            rect = (x1, y1 + int(bond_dy), x2, y2 + int(bond_dy))
        rect = _clamp_rect(rect, w, h)
        if r.key == "bonds":
            bonds_rect = rect
            continue
        if r.key == "hp_nick":
            continue
        out.append((r.key, r.name, rect))

    if bonds_rect is None:
        raise RuntimeError("未找到 bonds ROI")
    bx1, by1, bx2, by2 = bonds_rect
    bond_cells = pr._split_roi_grid(bx1, by1, bx2, by2, cols=1, rows=7)
    for i, rc in enumerate(bond_cells, 1):
        out.append((f"bonds_row_{i}", f"羁绊第{i}行", rc))

    hp_col = _scale_hp_rect_ref_to_image((1970, 110, 2140, 880), w, h)
    fx1, fy1, fx2, fy2 = hp_col
    out.append(("hp_col_scaled", "血量列缩放框", hp_col))

    fixed_refs = pr._hp_fixed_digit_rects_ref()
    id_w, id_h = 300, 40
    fixed_ix1 = max(0, min(fx1 - 120, w - id_w))
    for idx, nref in enumerate(fixed_refs, 1):
        nx1, ny1, nx2, ny2 = _scale_hp_rect_ref_to_image(nref, w, h)
        out.append((f"hp_digit_{idx}", f"血量数字{idx}", (nx1, ny1, nx2, ny2)))
        ex1 = max(0, nx1 - pr._HP_SELF_EXPAND_LEFT_PX)
        ey1 = max(0, ny1 - pr._HP_SELF_EXPAND_UP_PX)
        out.append((f"hp_digit_ex_{idx}", f"血量数字扩展{idx}", (ex1, ey1, nx2, ny2)))
        iy2 = max(id_h, ny1 - 2)
        iy1 = max(0, iy2 - id_h)
        ix2 = min(w, fixed_ix1 + id_w)
        out.append((f"hp_id_{idx}", f"玩家ID{idx}", (fixed_ix1, iy1, ix2, iy2)))

    return out


def _stitch_patches(
    bgr: np.ndarray,
    patches: Sequence[Tuple[str, str, Tuple[int, int, int, int]]],
    *,
    cols: int = 6,
    gap: int = 20,
    pad: int = 8,
) -> Tuple[np.ndarray, List[PatchMeta]]:
    tiles: List[Tuple[str, str, np.ndarray, Tuple[int, int, int, int]]] = []
    for key, name, rect in patches:
        x1, y1, x2, y2 = rect
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if key in {"gold", "streak"}:
            crop = cv2.resize(
                crop,
                (max(1, crop.shape[1] * 2), max(1, crop.shape[0] * 2)),
                interpolation=cv2.INTER_CUBIC,
            )
        tile = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        tiles.append((key, name, tile, rect))

    if not tiles:
        raise RuntimeError("没有可拼接的 ROI")

    rows = (len(tiles) + cols - 1) // cols
    col_ws = [0] * cols
    row_hs = [0] * rows
    for i, (_, _, tile, _) in enumerate(tiles):
        r, c = divmod(i, cols)
        th, tw = tile.shape[:2]
        col_ws[c] = max(col_ws[c], tw)
        row_hs[r] = max(row_hs[r], th)

    W = sum(col_ws) + gap * (cols + 1)
    H = sum(row_hs) + gap * (rows + 1)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    metas: List[PatchMeta] = []
    y = gap
    idx = 0
    for r in range(rows):
        x = gap
        for c in range(cols):
            if idx >= len(tiles):
                break
            key, name, tile, src_rect = tiles[idx]
            th, tw = tile.shape[:2]
            canvas[y : y + th, x : x + tw] = tile
            metas.append(PatchMeta(key=key, name=name, src_rect=src_rect, stitch_rect=(x, y, x + tw, y + th)))
            x += col_ws[c] + gap
            idx += 1
        y += row_hs[r] + gap
    return canvas, metas


def _center_of_quad(quad: Any) -> Tuple[float, float]:
    xs: List[float] = []
    ys: List[float] = []
    if isinstance(quad, (list, tuple)):
        for p in quad:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    if not xs:
        return 0.0, 0.0
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _assign_texts_to_patches(result: Any, metas: Sequence[PatchMeta]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {m.key: [] for m in metas}
    if not result:
        return out
    for it in result:
        if not isinstance(it, (list, tuple)) or len(it) < 2:
            continue
        txt = str(it[1] or "").strip()
        if not txt:
            continue
        cx, cy = _center_of_quad(it[0])
        for m in metas:
            x1, y1, x2, y2 = m.stitch_rect
            if x1 <= cx < x2 and y1 <= cy < y2:
                out[m.key].append(txt)
                break
    return out


def _join(ss: Sequence[str]) -> str:
    return " ".join(s.strip() for s in ss if s and s.strip()).strip()


def _load_legend_traits_meta(path: Path) -> Tuple[List[str], Dict[str, List[int]]]:
    names: List[str] = []
    thresholds_map: Dict[str, List[int]] = {}
    if not path.is_file():
        return names, thresholds_map
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if str(o.get("type") or "") != "trait":
            continue
        n = str(o.get("name") or "").strip()
        if n:
            names.append(n)
            th = o.get("thresholds") or []
            if isinstance(th, list):
                vals: List[int] = []
                for x in th:
                    try:
                        vals.append(int(x))
                    except Exception:
                        continue
                if vals:
                    thresholds_map[n] = vals
    return sorted(set(names)), thresholds_map


def _canonicalize_trait_name(raw: str, legend_names: Sequence[str]) -> str:
    t = re.sub(r"\s+", "", raw or "")
    if not t:
        return ""
    t = re.sub(r"^[无丶`~\-\._]+", "", t)
    single_alias = {
        "壮": "斗士",
    }
    if t in single_alias:
        t = single_alias[t]
    if t in legend_names:
        return t
    subs = [n for n in legend_names if n and n in t]
    if subs:
        subs.sort(key=len, reverse=True)
        return subs[0]
    prefs = [n for n in legend_names if n.startswith(t)]
    if len(prefs) == 1:
        return prefs[0]
    if len(prefs) > 1:
        prefs.sort(key=len, reverse=True)
        return prefs[0]
    pick = difflib.get_close_matches(t, list(legend_names), n=1, cutoff=0.34)
    return pick[0] if pick else t


def _rapid_texts(ocr: RapidOCR, img: np.ndarray) -> List[str]:
    res, _ = ocr(img, use_det=True, use_cls=False, use_rec=True)
    out: List[str] = []
    for it in (res or []):
        if not isinstance(it, (list, tuple)) or len(it) < 2:
            continue
        t = str(it[1] or "").strip()
        if t:
            out.append(t)
    return out


def _fallback_gold_digits(ocr: RapidOCR, crop: np.ndarray) -> str:
    if crop.size == 0:
        return ""
    h, w = crop.shape[:2]
    best = ""
    for tr in (0.0, 0.26):
        x0 = int(round(w * tr))
        if x0 >= w - 6:
            continue
        sub = crop[:, x0:]
        if sub.size == 0:
            continue
        for img in (sub, pr._preprocess_ui_text_bgr(sub)):
            ts = _rapid_texts(ocr, img)
            joined = _join(ts)
            ds = re.findall(r"\d+", joined)
            if not ds:
                continue
            cand = max(ds, key=len)
            if len(cand) > len(best):
                best = cand
                if len(best) >= 2:
                    return best
    return best


def _fallback_bond_row_text(
    ocr: RapidOCR,
    bgr: np.ndarray,
    rect: Tuple[int, int, int, int],
) -> str:
    x1, y1, x2, y2 = rect
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    up = cv2.resize(
        crop,
        (max(1, crop.shape[1] * 2), max(1, crop.shape[0] * 2)),
        interpolation=cv2.INTER_CUBIC,
    )
    best = ""
    for img in (up, pr._preprocess_ui_text_bgr(up)):
        txt = _join(_rapid_texts(ocr, img))
        if len(re.sub(r"\s+", "", txt)) > len(re.sub(r"\s+", "", best)):
            best = txt
    return best


def _parse_bond_row(s: str, legend_names: Sequence[str]) -> str:
    t = re.sub(r"\s+", "", s or "")
    m = re.search(r"([\u4e00-\u9fffA-Za-z·]+).*?(\d)\s*/\s*\d", t)
    if m:
        return f"{m.group(2)}{_canonicalize_trait_name(m.group(1), legend_names)}"
    m2 = re.search(r"(\d)\s*/\s*\d.*?([\u4e00-\u9fffA-Za-z·]+)", t)
    if m2:
        return f"{m2.group(1)}{_canonicalize_trait_name(m2.group(2), legend_names)}"
    m3 = re.search(r"(\d)([\u4e00-\u9fffA-Za-z·]+)", t)
    if m3:
        return f"{m3.group(1)}{_canonicalize_trait_name(m3.group(2), legend_names)}"
    return ""


_PAIR_STEM_RE = re.compile(r"^(.+?)[-_]([a-z])$", re.I)


def _pair_base_from_stem(stem: str) -> Optional[str]:
    m = _PAIR_STEM_RE.match(stem)
    return m.group(1) if m else None


def _is_main_variant_a(stem: str) -> bool:
    m = _PAIR_STEM_RE.match(stem)
    return bool(m and m.group(2).lower() == "a")


def _empty_bonds_field() -> Dict[str, Any]:
    return {
        "name": "羁绊栏",
        "raw": "",
        "parsed": "识别字段：无",
        "bond_items": [],
        "bond_summary": "识别字段：无",
        "bond_cells": [],
    }


def _empty_fields_shell(bonds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    b = bonds if bonds is not None else _empty_bonds_field()
    return {
        "phase": {"name": "阶段", "raw": "", "parsed": ""},
        "level": {"name": "等级", "raw": "", "parsed": ""},
        "exp": {"name": "经验", "raw": "", "parsed": ""},
        "gold": {"name": "金币", "raw": "", "parsed": ""},
        "streak": {"name": "连胜/连败", "raw": "", "parsed": ""},
        "bonds": b,
        "hp_nick": {
            "name": "血量/昵称",
            "raw": "",
            "parsed": [],
            "player_lines": [],
            "player_cells": [],
        },
    }


def _merge_bonds_fields(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    items_a = list(a.get("bond_items") or [])
    items_b = list(b.get("bond_items") or [])
    seen: set[str] = set()
    merged_items: List[str] = []
    for it in items_a + items_b:
        if it and it not in seen:
            seen.add(it)
            merged_items.append(it)
    raw_a = (a.get("raw") or "").strip()
    raw_b = (b.get("raw") or "").strip()
    raw_join = " | ".join(x for x in [raw_a, raw_b] if x)
    bond_summary = "识别字段：" + (" / ".join(merged_items) if merged_items else "无")
    bond_cells = list(a.get("bond_cells") or []) + list(b.get("bond_cells") or [])
    return {
        "name": "羁绊栏",
        "raw": raw_join,
        "parsed": bond_summary,
        "bond_items": merged_items,
        "bond_summary": bond_summary,
        "bond_cells": bond_cells,
    }


def _fill_bonds_field(
    bucket: Dict[str, List[str]],
    patch_map: Dict[str, Tuple[int, int, int, int]],
    bgr: np.ndarray,
    ocr: RapidOCR,
    legend_names: Sequence[str],
    trait_thresholds: Dict[str, List[int]],
) -> Dict[str, Any]:
    bond_items: List[str] = []
    bond_cells: List[Dict[str, Any]] = []
    raws: List[str] = []
    for i in range(1, 8):
        key = f"bonds_row_{i}"
        raw = _join(bucket.get(key, []))
        piece = _parse_bond_row(raw, legend_names)
        bad = False
        if not piece:
            bad = True
        else:
            trait_name = re.sub(r"^\d+", "", piece)
            if (trait_name not in legend_names) or len(trait_name) <= 1:
                bad = True
        if bad:
            rc = patch_map.get(key)
            if rc is not None:
                raw2 = _fallback_bond_row_text(ocr, bgr, rc)
                piece2 = _parse_bond_row(raw2, legend_names)
                if piece2:
                    raw = raw2
                    piece = piece2
        if not piece:
            trait_only = _canonicalize_trait_name(raw, legend_names)
            th = trait_thresholds.get(trait_only) or []
            if th == [1]:
                piece = f"1{trait_only}"
        raws.append(raw)
        if piece:
            bond_items.append(piece)
        bond_cells.append({"index": i, "ocr_rows": raw, "piece": piece or None, "raw": raw})
    bond_summary = "识别字段：" + (" / ".join(bond_items) if bond_items else "无")
    return {
        "name": "羁绊栏",
        "raw": " | ".join(raws),
        "parsed": bond_summary,
        "bond_items": bond_items,
        "bond_summary": bond_summary,
        "bond_cells": bond_cells,
    }


def run_once(
    img_path: Path,
    out_json: Optional[Path],
    out_stitch_png: Path,
    *,
    ocr_engine: RapidOCR | None = None,
    bonds_only: bool = False,
) -> Dict[str, Any]:
    bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"读取失败: {img_path}")

    patches = _build_patches(bgr, image_stem=img_path.stem)
    if bonds_only:
        patches = [p for p in patches if p[0].startswith("bonds_row_")]
        if not patches:
            raise RuntimeError(f"羁绊 ROI 为空: {img_path}")
    patch_map = {k: rc for k, _n, rc in patches}
    stitched, metas = _stitch_patches(bgr, patches, cols=6, gap=20, pad=8)
    ok, buf = cv2.imencode(".png", stitched)
    if ok:
        out_stitch_png.write_bytes(buf.tobytes())

    ocr = ocr_engine if ocr_engine is not None else create_ocr_engine(
        {"det_use_dml": True, "cls_use_dml": False, "rec_use_dml": True}
    )
    result, _ = ocr(stitched, use_det=True, use_cls=False, use_rec=True)
    bucket = _assign_texts_to_patches(result, metas)
    legend_names, trait_thresholds = _load_legend_traits_meta(
        PROJECT_ROOT / "data" / "rag_legend_traits.jsonl"
    )

    bonds_field = _fill_bonds_field(
        bucket, patch_map, bgr, ocr, legend_names, trait_thresholds
    )

    if bonds_only:
        summary = {
            "file": img_path.name,
            "mode": "bonds_only_stitch_ocr",
            "stitched_image": out_stitch_png.name,
            "fields": {"bonds": bonds_field},
        }
        if out_json is not None:
            out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    fields: Dict[str, Any] = {}
    for k, name in [
        ("phase", "阶段"),
        ("level", "等级"),
        ("exp", "经验"),
        ("gold", "金币"),
        ("streak", "连胜/连败"),
    ]:
        raw = _join(bucket.get(k, []))
        if k == "gold":
            ds = re.findall(r"\d+", raw or "")
            raw = max(ds, key=len) if ds else ""
            if not raw:
                gx1, gy1, gx2, gy2 = patch_map.get("gold", (0, 0, 0, 0))
                raw = _fallback_gold_digits(ocr, bgr[gy1:gy2, gx1:gx2])
        if k == "streak":
            ds = re.findall(r"\d+", raw or "")
            raw = max(ds, key=len) if ds else "0"
        parsed = pr.parse_field(k, raw) if raw else ""
        fields[k] = {"name": name, "raw": raw, "parsed": parsed}

    fields["bonds"] = bonds_field

    player_cells: List[Dict[str, Any]] = []
    parsed_lines: List[str] = []
    raw_rows: List[str] = []
    for i in range(1, 9):
        id_raw = _join(bucket.get(f"hp_id_{i}", []))
        id_text = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_·]+", re.sub(r"\s+", "", id_raw)))
        if re.fullmatch(r"\d{1,3}", id_text or ""):
            id_text = ""
        if not id_text:
            id_text = "我"
        hp_raw = _join(bucket.get(f"hp_digit_{i}", []))
        hp_raw_ex = _join(bucket.get(f"hp_digit_ex_{i}", []))
        hp = pr._parse_hp_digits_from_ocr_text(re.sub(r"\s+", "", hp_raw))
        if (not hp) and id_text == "我":
            hp = pr._parse_hp_digits_from_ocr_text(re.sub(r"\s+", "", hp_raw_ex))
            if hp:
                hp_raw = hp_raw_ex
        if not hp:
            hp = "0"
        line = f"Top{i} {id_text} {hp}血"
        parsed_lines.append(line)
        raw_rows.append(hp_raw)
        player_cells.append({"display": line, "hp": hp, "id_text": id_text, "ocr_rows": hp_raw})

    fields["hp_nick"] = {
        "name": "血量/昵称",
        "raw": " | ".join(raw_rows),
        "parsed": parsed_lines,
        "player_lines": parsed_lines,
        "player_cells": player_cells,
    }

    summary = {
        "file": img_path.name,
        "mode": "onepass_stitch_ocr",
        "stitched_image": out_stitch_png.name,
        "fields": fields,
    }
    if out_json is not None:
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _resolve_device(requested: str) -> Tuple[str, Dict[str, bool]]:
    req = (requested or "auto").strip().lower()
    providers = set(ort.get_available_providers())
    has_dml = "DmlExecutionProvider" in providers
    has_cuda = "CUDAExecutionProvider" in providers

    flags = {"det_use_dml": False, "cls_use_dml": False, "rec_use_dml": False}
    cflags = {"det_use_cuda": False, "cls_use_cuda": False, "rec_use_cuda": False}

    if req == "dml":
        if has_dml:
            flags = {k: True for k in flags}
            return "dml", {**flags, **cflags}
        return "cpu", {**flags, **cflags}
    if req == "cuda":
        if has_cuda:
            cflags = {k: True for k in cflags}
            return "cuda", {**flags, **cflags}
        return "cpu", {**flags, **cflags}
    if req == "cpu":
        return "cpu", {**flags, **cflags}

    if has_dml:
        flags = {k: True for k in flags}
        return "dml", {**flags, **cflags}
    if has_cuda:
        cflags = {k: True for k in cflags}
        return "cuda", {**flags, **cflags}
    return "cpu", {**flags, **cflags}


def _get_thread_ocr(device_kwargs: Dict[str, bool]):
    eng = getattr(_thread_local, "ocr_engine", None)
    if eng is None:
        eng = create_ocr_engine(device_kwargs)
        _thread_local.ocr_engine = eng
    return eng


def _partition_images(
    images: List[Path],
) -> Tuple[Dict[str, List[Path]], List[Path]]:
    groups: Dict[str, List[Path]] = {}
    standalone: List[Path] = []
    for p in images:
        base = _pair_base_from_stem(p.stem)
        if base is None:
            standalone.append(p)
        else:
            groups.setdefault(base, []).append(p)
    return groups, standalone


def _process_standalone(
    img_path: Path,
    device_kwargs: Dict[str, bool],
    out_dir: Path,
    write_json: bool,
) -> Tuple[str, Optional[str]]:
    ocr_engine = _get_thread_ocr(device_kwargs)
    stem = img_path.stem
    out_png = out_dir / f"{stem}_player_onnx_stitched.png"
    out_json = out_dir / f"{stem}_player_onnx_summary.json" if write_json else None
    run_once(img_path, out_json, out_png, ocr_engine=ocr_engine, bonds_only=False)
    return out_png.name, out_json.name if write_json else None


def _process_pair_group(
    base: str,
    paths: List[Path],
    device_kwargs: Dict[str, bool],
    out_dir: Path,
    write_json: bool,
) -> Tuple[str, Optional[str]]:
    """同序号 a + 副图：各写 stitched PNG，合并写 {base}_player_onnx_summary.json。"""
    ocr_engine = _get_thread_ocr(device_kwargs)
    main_a = next((p for p in paths if _is_main_variant_a(p.stem)), None)
    secondary = [p for p in paths if not _is_main_variant_a(p.stem)]

    merged_json = out_dir / f"{base}_player_onnx_summary.json"
    bond_summaries: List[Dict[str, Any]] = []
    summ_a: Optional[Dict[str, Any]] = None

    if main_a is not None:
        out_png_a = out_dir / f"{main_a.stem}_player_onnx_stitched.png"
        summ_a = run_once(main_a, None, out_png_a, ocr_engine=ocr_engine, bonds_only=False)

    for p in secondary:
        out_png = out_dir / f"{p.stem}_player_onnx_stitched.png"
        s = run_once(p, None, out_png, ocr_engine=ocr_engine, bonds_only=True)
        bond_summaries.append(s)

    if summ_a is not None:
        final = copy.deepcopy(summ_a)
        mb = final["fields"]["bonds"]
        for s in bond_summaries:
            mb = _merge_bonds_fields(mb, s["fields"]["bonds"])
        final["fields"]["bonds"] = mb
        final["mode"] = "onepass_stitch_ocr_pair_merged"
        final["pair_base"] = base
        final["files"] = [main_a.name] + [p.name for p in secondary]
        stitched_map: Dict[str, str] = {main_a.name: f"{main_a.stem}_player_onnx_stitched.png"}
        for p in secondary:
            stitched_map[p.name] = f"{p.stem}_player_onnx_stitched.png"
        final["stitched_images"] = stitched_map
        final["stitched_image"] = f"{main_a.stem}_player_onnx_stitched.png"
    else:
        mb = _empty_bonds_field()
        for s in bond_summaries:
            mb = _merge_bonds_fields(mb, s["fields"]["bonds"])
        sm = {p.name: f"{p.stem}_player_onnx_stitched.png" for p in paths}
        first_png = next(iter(sm.values()), "")
        final = {
            "file": f"{base}-pair",
            "mode": "bonds_only_pair_missing_main",
            "pair_base": base,
            "files": [p.name for p in paths],
            "fields": _empty_fields_shell(mb),
            "stitched_images": sm,
            "stitched_image": first_png,
        }

    if write_json:
        merged_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        ",".join(sorted(p.name for p in paths)),
        merged_json.name if write_json else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="player_onnx：ONNX Runtime + DML/CUDA/CPU 的玩家信息识别"
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=str(pr.DEFAULT_INPUT),
        help=f"对局截图目录或单张 PNG（默认: {pr.DEFAULT_INPUT}）",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUT_PLAYER_ONNX),
        help="输出目录",
    )
    ap.add_argument(
        "--no-json",
        action="store_true",
        help="不写出每图 *_player_onnx_summary.json（默认会写出）",
    )
    ap.add_argument("--no-clear", action="store_true", help="不清空输出目录（追加写入）")
    ap.add_argument(
        "--device",
        choices=["auto", "dml", "cuda", "cpu"],
        default="auto",
        help="OCR 设备策略（默认 auto：dml>cuda>cpu）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4))),
        help="图片并发数（GPU/DML 下默认 1~4；纯 CPU 时上限自动压到 2，减轻卡顿）",
    )
    ap.add_argument(
        "--allow-dml-multi",
        action="store_true",
        help="允许 DML 模式下 workers>1（实验开关，可能触发驱动/运行时崩溃）",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    if not args.no_clear:
        pr._clear_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    images = pr._iter_pngs(in_path)
    device_name, device_kwargs = _resolve_device(args.device)
    workers = max(1, int(args.workers))
    if device_name == "cpu":
        capped = min(workers, 2)
        if capped < workers:
            print(
                f"[player_onnx] 纯 CPU 模式将 workers 从 {workers} 降为 {capped}（减轻卡顿；"
                "若需单线程可加 --workers 1）。"
            )
        workers = capped
    if device_name == "dml" and workers > 1 and not bool(args.allow_dml_multi):
        print(
            "[player_onnx] 检测到 DML + workers>1，已自动降级为 workers=1 以避免底层崩溃。"
            "若需强制并发，请显式加 --allow-dml-multi。"
        )
        workers = 1
    print(f"[player_onnx] device={device_name}, workers={workers}")

    device_kwargs["cls_use_dml"] = False
    device_kwargs["cls_use_cuda"] = False

    write_json = not bool(args.no_json)

    groups, standalone = _partition_images(list(images))

    jobs: List[Tuple[str, ...]] = [("standalone", p) for p in standalone]
    for base, gpaths in sorted(groups.items()):
        jobs.append(("pair", base, tuple(gpaths)))

    def _run_job(job: Tuple[str, ...]) -> Tuple[str, Optional[str]]:
        if job[0] == "standalone":
            return _process_standalone(job[1], device_kwargs, out_dir, write_json)
        return _process_pair_group(str(job[1]), list(job[2]), device_kwargs, out_dir, write_json)

    if workers == 1:
        for job in jobs:
            out_png_name, jname = _run_job(job)
            print(f"[OK] {out_png_name}")
            if jname:
                print(f"     json -> {jname}")
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_job, job): job for job in jobs}
        for fut in as_completed(futs):
            out_png_name, jname = fut.result()
            print(f"[OK] {out_png_name}")
            if jname:
                print(f"     json -> {jname}")


if __name__ == "__main__":
    main()
