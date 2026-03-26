# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

import player_recog as pr
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
            # 金币只保留左侧 40% 区域（根据当前 UI 标注，数字位于该区域）。
            x1, y1, x2, y2 = rect
            nx2 = x1 + int(round((x2 - x1) * 0.40))
            rect = _clamp_rect((x1, y1, nx2, y2), w, h)
        if c.key == "streak":
            # 连胜连败：顶部下切 10px，右侧左收 20px。
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
            # 默认不把大羁绊栏放进 OCR 拼图（仅保留逐行小图）。
            continue
        if r.key == "hp_nick":
            # 默认不把大血量列放进 OCR 拼图（仅保留数字/ID小图）。
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
    # 保留 hp 大列：对小 ROI 的检测/识别稳定性有辅助作用。
    out.append(("hp_col_scaled", "血量列缩放框", hp_col))

    fixed_refs = pr._hp_fixed_digit_rects_ref()
    id_w, id_h = 300, 40
    fixed_ix1 = max(0, min(fx1 - 120, w - id_w))
    for idx, nref in enumerate(fixed_refs, 1):
        nx1, ny1, nx2, ny2 = _scale_hp_rect_ref_to_image(nref, w, h)
        out.append((f"hp_digit_{idx}", f"血量数字{idx}", (nx1, ny1, nx2, ny2)))
        # 兜底框：向左/向上扩展，覆盖“我”这一行常见的左突数字。
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
        # 小 ROI（金币/连胜）放大后再拼图，提升小数字检出。
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
    # 去掉常见噪声前缀，如“无弗雷尔卓德”
    t = re.sub(r"^[无丶`~\-\._]+", "", t)
    # 常见单字误识别纠错（OCR 将“斗士”读成“壮”等）
    single_alias = {
        "壮": "斗士",
    }
    if t in single_alias:
        t = single_alias[t]
    if t in legend_names:
        return t
    # 含有合法名子串时，优先取最长合法名
    subs = [n for n in legend_names if n and n in t]
    if subs:
        subs.sort(key=len, reverse=True)
        return subs[0]
    # 前缀补全
    prefs = [n for n in legend_names if n.startswith(t)]
    if len(prefs) == 1:
        return prefs[0]
    if len(prefs) > 1:
        prefs.sort(key=len, reverse=True)
        return prefs[0]
    # 模糊匹配纠错（“壮”->“斗士”等）
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
    # 轻量化：减少 trim 分支与增强分支，优先快路径。
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
    # 行级兜底：放大 + 对比增强 + 二值增强
    up = cv2.resize(
        crop,
        (max(1, crop.shape[1] * 2), max(1, crop.shape[0] * 2)),
        interpolation=cv2.INTER_CUBIC,
    )
    best = ""
    # 轻量化：保留两路，去掉第三路二值分支。
    for img in (up, pr._preprocess_ui_text_bgr(up)):
        txt = _join(_rapid_texts(ocr, img))
        if len(re.sub(r"\s+", "", txt)) > len(re.sub(r"\s+", "", best)):
            best = txt
    return best


def _parse_bond_row(s: str, legend_names: Sequence[str]) -> str:
    t = re.sub(r"\s+", "", s or "")
    # 允许 1/2、2/3/4/5 等梯度串
    m = re.search(r"([\u4e00-\u9fffA-Za-z·]+).*?(\d)\s*/\s*\d", t)
    if m:
        return f"{m.group(2)}{_canonicalize_trait_name(m.group(1), legend_names)}"
    m2 = re.search(r"(\d)\s*/\s*\d.*?([\u4e00-\u9fffA-Za-z·]+)", t)
    if m2:
        return f"{m2.group(1)}{_canonicalize_trait_name(m2.group(2), legend_names)}"
    # 无 x/y 结构时：尝试“数字 + 名称”兜底
    m3 = re.search(r"(\d)([\u4e00-\u9fffA-Za-z·]+)", t)
    if m3:
        return f"{m3.group(1)}{_canonicalize_trait_name(m3.group(2), legend_names)}"
    return ""


def run_once(
    img_path: Path,
    out_json: Path,
    out_stitch_png: Path,
    *,
    ocr_engine: RapidOCR | None = None,
) -> Dict[str, Any]:
    bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"读取失败: {img_path}")

    patches = _build_patches(bgr, image_stem=img_path.stem)
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
        Path(__file__).resolve().parent / "data" / "rag_legend_traits.jsonl"
    )

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
            # 金币只保留最长数字串。
            ds = re.findall(r"\d+", raw or "")
            raw = max(ds, key=len) if ds else ""
            if not raw:
                gx1, gy1, gx2, gy2 = patch_map.get("gold", (0, 0, 0, 0))
                raw = _fallback_gold_digits(ocr, bgr[gy1:gy2, gx1:gx2])
        if k == "streak":
            # 无数字即视为 0 连胜/连败。
            ds = re.findall(r"\d+", raw or "")
            raw = (max(ds, key=len) if ds else "0")
        parsed = pr.parse_field(k, raw) if raw else ""
        fields[k] = {"name": name, "raw": raw, "parsed": parsed}

    bond_items: List[str] = []
    bond_cells: List[Dict[str, Any]] = []
    raws: List[str] = []
    for i in range(1, 8):
        key = f"bonds_row_{i}"
        raw = _join(bucket.get(key, []))
        piece = _parse_bond_row(raw, legend_names)
        # 行内只识别到异常短名称/噪声时，放大该行重跑 OCR 再解析
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
        # 特殊羁绊补全：若仅识别出羁绊名且该羁绊阈值为 [1]，补全为 1羁绊名。
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
    fields["bonds"] = {
        "name": "羁绊栏",
        "raw": " | ".join(raws),
        "parsed": bond_summary,
        "bond_items": bond_items,
        "bond_summary": bond_summary,
        "bond_cells": bond_cells,
    }

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
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    in_dir = root / "对局截图"
    out_dir = root / "player_onepass_stitch_out"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"], key=lambda p: p.name.lower())
    for img in imgs:
        stem = img.stem
        out_json = out_dir / f"{stem}_summary.json"
        out_stitch = out_dir / f"{stem}_stitched.png"
        summ = run_once(img, out_json, out_stitch)
        f = summ.get("fields", {})
        print(f"[OK] {img.name} -> {out_json.name}, {out_stitch.name}")
        print(
            "    "
            + " | ".join(
                [
                    f"阶段:{((f.get('phase') or {}).get('parsed') or '').replace('总', '')}",
                    f"等级:{(f.get('level') or {}).get('parsed') or ''}",
                    f"经验:{(f.get('exp') or {}).get('parsed') or ''}",
                    f"金币:{(f.get('gold') or {}).get('parsed') or ''}",
                    f"胜负:{(f.get('streak') or {}).get('parsed') or ''}",
                ]
            )
        )
