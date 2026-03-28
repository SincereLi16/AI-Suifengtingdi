# -*- coding: utf-8 -*-
"""
player_recog：对局截图固定 ROI 的玩家信息识别，并输出带框与中文标识图。

引擎：
- paddle：PaddleOCR（需 paddlepaddle + paddleocr）

输入：对局截图目录（或单张 PNG）下的 PNG
输出：每张图一张标注图（ROI 矩形 + 字段名 + 解析结果 / 原始 OCR 文本）

识别数据：阶段、羁绊栏、等级、经验、金币、连胜/连败、血量/昵称等展示内容
均来自本脚本对截图 ROI 的 OCR，不依赖本仓库内 chess_recog / equip_recog 等其它识别程序。

依赖：见 docs/依赖与环境说明.txt 中 OCR 小节。默认自动使用项目内 `.venv_battle_ocr`（首次运行会创建并
安装 paddle 等），与全局 protobuf/torch 隔离。若要在当前解释器直接运行（不建 venv），可设环境变量
`BATTLE_UI_OCR_NO_VENV=1`。

说明：脚本开头会设置 FLAGS_use_mkldnn=0，避免在部分 Windows 环境下因 oneDNN 推理链触发
NotImplementedError。

模型缓存：PaddleOCR 检测/识别权重会缓存在用户目录下（Windows 一般为
%USERPROFILE%\\.paddleocr），仅首次联网下载；之后每次运行仍会「加载模型到内存」，
这是进程内一次性开销，无法像浏览器那样跨次程序常驻内存。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

for _d in Path(__file__).resolve().parents:
    if (_d / "repo_sys_path.py").exists():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break
import repo_sys_path  # noqa: F401

from project_paths import DEFAULT_OUT_PLAYER_INFO, PROJECT_ROOT

_ROOT = PROJECT_ROOT
_VENV_DIR = _ROOT / ".venv_battle_ocr"
_VENV_PY = (
    _VENV_DIR / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else _VENV_DIR / "bin" / "python"
)
_OCR_PIP = [
    "paddlepaddle==2.6.2",
    "paddleocr>=2.6.1,<3",
    "opencv-python",
    "numpy",
    "pillow",
]


def _bootstrap_ocr_venv() -> None:
    print("首次运行：正在创建专用虚拟环境 .venv_battle_ocr …")
    subprocess.run([sys.executable, "-m", "venv", str(_VENV_DIR)], check=True)
    if not _VENV_PY.is_file():
        raise SystemExit(f"未找到解释器: {_VENV_PY}")
    subprocess.run([str(_VENV_PY), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    print("正在安装 PaddleOCR 依赖（仅写入该 venv）…")
    subprocess.run([str(_VENV_PY), "-m", "pip", "install", *_OCR_PIP], check=True)
    print("虚拟环境就绪。\n")


def _ensure_running_in_ocr_venv() -> None:
    import os

    if os.environ.get("BATTLE_UI_OCR_NO_VENV") == "1":
        return
    try:
        cur = Path(sys.executable).resolve()
    except Exception:
        return
    if _VENV_PY.is_file() and cur == _VENV_PY.resolve():
        return
    if not _VENV_PY.is_file():
        _bootstrap_ocr_venv()
    if not _VENV_PY.is_file():
        raise SystemExit("无法创建或使用 .venv_battle_ocr")
    script = Path(__file__).resolve()
    rc = subprocess.call([str(_VENV_PY), str(script), *sys.argv[1:]])
    raise SystemExit(rc)


import os

# 必须在首次 import paddlepaddle / paddleocr 之前设置。
# Windows 上若开启 MKLDNN/oneDNN，部分 PP-OCR 推理会出现：
# NotImplementedError: ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc
os.environ.setdefault("FLAGS_use_mkldnn", "0")
# 跳过 PaddleX 首次联网检查模型源（可选，减少启动等待）
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
# 降低 Paddle C++ 侧日志（若环境支持）
os.environ.setdefault("GLOG_minloglevel", "3")

import argparse
import json
import logging
import re
import shutil
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _fast_ocr_enabled() -> bool:
    """快速模式：优先减少重复 OCR 调用次数。"""
    return os.environ.get("BATTLE_UI_OCR_FAST", "1").strip() != "0"

# 仅用于「把 OCR 结果画到图上」的字体；与识别逻辑无关
_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]

DEFAULT_INPUT = _ROOT / "对局截图"
DEFAULT_OUT = DEFAULT_OUT_PLAYER_INFO

PNG_EXTS = {".png"}

# 血量数字 ROI：参考分辨率（与布局其它 ROI 一致）。
# 固定矩形 60×36，中心 x=2075；各行中心 y 见 _HP_DIGIT_CENTER_Y_REF（按图像缩放）。
_HP_REF_W = 2196
_HP_REF_H = 1253
_HP_DIGIT_CX_REF = 2075
_HP_DIGIT_W_REF = 60
_HP_DIGIT_H_REF = 36
_HP_DIGIT_CENTER_Y_REF = [156, 277, 371, 467, 560, 654, 748, 844]
# 判定为「我」（ID 区无有效昵称）时，血量数字 ROI 相对常规框向左、向上各扩展的像素（图像坐标）。
_HP_SELF_EXPAND_LEFT_PX = 40
_HP_SELF_EXPAND_UP_PX = 40


@dataclass
class RectRoi:
    """x1,y1,x2,y2 与图像坐标一致（左闭右开可视为 [x1,x2)×[y1,y2) 画框时 x2-1,y2-1）。"""

    x1: int
    y1: int
    x2: int
    y2: int
    name: str
    key: str
    grid_cols: int = 0
    grid_rows: int = 0


@dataclass
class CenterRoi:
    cx: int
    cy: int
    half_w: int
    half_h: int
    name: str
    key: str

    def to_rect(self, w_img: int, h_img: int) -> Tuple[int, int, int, int]:
        x1 = max(0, self.cx - self.half_w)
        y1 = max(0, self.cy - self.half_h)
        x2 = min(w_img, self.cx + self.half_w)
        y2 = min(h_img, self.cy + self.half_h)
        return (x1, y1, x2, y2)


# 副图（文件名 *-b / *_b，如 01-b.png）羁绊栏相对主图 UI 略下移，动态羁绊候选框整体下移以对准未激活人数等小 ROI。
BOND_ROI_AUXILIARY_DY_PX = 10


def _stem_uses_auxiliary_bond_roi_shift(stem: str) -> bool:
    """是否为对局副图命名（与主图 xx-a 成对的 xx-b）。"""
    s = stem.strip().lower()
    return s.endswith("-b") or s.endswith("_b")


def _default_layout() -> Tuple[List[CenterRoi], List[RectRoi]]:
    """默认 ROI：与需求中的中心/矩形大致一致。"""
    centers: List[CenterRoi] = [
        # 阶段 / 等级 / 经验：收窄左右宽度，减少框进无关 UI
        CenterRoi(808, 30, 52, 28, "阶段", "phase"),
        CenterRoi(423, 1000, 38, 36, "等级", "level"),
        CenterRoi(540, 1000, 54, 36, "经验", "exp"),
        # 金币：整框只用于画框；OCR 再裁掉左侧图标区，避免大 ROI 里硬币/高光干扰 PP-OCR。
        # 中心略偏右对准数字列，宽度适中（非整块底部条）。
        CenterRoi(1150, 1000, 52, 32, "金币", "gold"),
        CenterRoi(1256, 1000, 50, 32, "连胜/连败", "streak"),
    ]
    # 羁绊：1×7 分块 OCR（每行一名羁绊槽）；血量：整块列 OCR
    rects: List[RectRoi] = [
        RectRoi(110, 100, 470, 700, "羁绊栏", "bonds", grid_cols=1, grid_rows=7),
        RectRoi(1910, 100, 2190, 880, "血量/昵称", "hp_nick", grid_cols=0, grid_rows=0),
    ]
    return centers, rects


def _split_roi_grid(
    rx1: int,
    ry1: int,
    rx2: int,
    ry2: int,
    *,
    cols: int,
    rows: int,
) -> List[Tuple[int, int, int, int]]:
    """将矩形 ROI 等分为 cols×rows 块，返回 [(bx1,by1,bx2,by2), ...] 自上而下、自左而右。"""
    if cols <= 0 or rows <= 0 or rx2 <= rx1 or ry2 <= ry1:
        return []
    total_w = rx2 - rx1
    total_h = ry2 - ry1
    base_w = total_w // cols
    rem_w = total_w % cols
    base_h = total_h // rows
    rem_h = total_h % rows
    blocks: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        by1 = ry1 + r * base_h + min(r, rem_h)
        bh = base_h + (1 if r < rem_h else 0)
        by2 = by1 + bh
        for c in range(cols):
            bx1 = rx1 + c * base_w + min(c, rem_w)
            bw = base_w + (1 if c < rem_w else 0)
            bx2 = bx1 + bw
            blocks.append((bx1, by1, bx2, by2))
    return blocks


def _iter_pngs(path: Path) -> List[Path]:
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() not in PNG_EXTS:
            raise SystemExit(f"仅支持 PNG 文件: {path}")
        return [path]
    if path.is_dir():
        files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in PNG_EXTS]
        files.sort(key=lambda p: p.name.lower())
        if not files:
            raise SystemExit(f"目录内无 PNG: {path}")
        return files
    raise SystemExit(f"路径不存在: {path}")


def _clear_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def _maybe_upscale(crop_bgr: np.ndarray, min_side: int = 48) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return crop_bgr
    m = min(h, w)
    if m >= min_side:
        return crop_bgr
    scale = float(min_side) / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)


def _scale_crop_min_height(bgr: np.ndarray, min_h: int = 100) -> np.ndarray:
    """分块较矮时放大整幅，便于检测/识别小字（羁绊格单行等）。"""
    if bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    if h >= min_h:
        return bgr
    scale = float(min_h) / float(h)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)


def _preprocess_ui_text_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    针对「平面上的文字+数字」HUD：提升局部对比与边缘，利于 PP-OCR 检测与识别。
    使用 LAB 的 L 通道 CLAHE，避免灰度直转时丢色信息；再轻度反锐化。
    """
    if bgr.size == 0:
        return bgr
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l2 = clahe.apply(l_ch)
        lab2 = cv2.merge([l2, a_ch, b_ch])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        out = bgr.copy()
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=0.85)
    out = cv2.addWeighted(out, 1.38, blur, -0.38, 0)
    return out


def _ocr_score(entries: List[Tuple[str, Any]]) -> Tuple[int, int]:
    joined = "".join(t for t, _ in entries)
    return (len(joined), len(entries))


def _looks_like_paddle_det_line(item: Any) -> bool:
    """PaddleOCR 2.x 单行：[[[x,y],...×4], (识别文本, 置信度)]，有时第二元为 str。"""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return False
    second = item[1]
    if isinstance(second, str):
        return True
    if isinstance(second, (list, tuple)) and len(second) >= 1:
        return isinstance(second[0], str)
    return False


def _text_from_paddle_line(item: Any) -> str:
    second = item[1]
    if isinstance(second, str):
        return second.strip()
    if isinstance(second, (list, tuple)) and len(second) >= 1:
        return str(second[0]).strip()
    return ""


def _extract_ocr_lines(ocr_result: Any) -> List[str]:
    """将 PaddleOCR 返回结构统一为文本行列表（只取识别文本，不用检测框坐标）。"""
    if ocr_result is None:
        return []
    out: List[str] = []

    if isinstance(ocr_result, dict):
        for k in ("rec_texts", "texts", "ocr_text"):
            if k in ocr_result and isinstance(ocr_result[k], list):
                return [str(x).strip() for x in ocr_result[k] if str(x).strip()]
        return []

    if isinstance(ocr_result, list) and ocr_result and isinstance(ocr_result[0], dict):
        for d in ocr_result:
            if not isinstance(d, dict):
                continue
            for k in ("text", "rec_text", "transcription", "label"):
                if k in d and str(d[k]).strip():
                    out.append(str(d[k]).strip())
                    break
        return out

    if not isinstance(ocr_result, list) or not ocr_result:
        return []

    # 单元素即一整行（少数返回形态）
    if _looks_like_paddle_det_line(ocr_result):
        s = _text_from_paddle_line(ocr_result)
        return [s] if s else []

    # 多行：每行均为 [box, (text, conf)]
    if _looks_like_paddle_det_line(ocr_result[0]):
        for item in ocr_result:
            if not _looks_like_paddle_det_line(item):
                continue
            s = _text_from_paddle_line(item)
            if s:
                out.append(s)
        return out

    # 单页包装：[[ line1, line2, ... ]]
    if (
        len(ocr_result) == 1
        and isinstance(ocr_result[0], list)
        and ocr_result[0]
        and _looks_like_paddle_det_line(ocr_result[0][0])
    ):
        for item in ocr_result[0]:
            if not _looks_like_paddle_det_line(item):
                continue
            s = _text_from_paddle_line(item)
            if s:
                out.append(s)
        return out

    return out


def _iter_paddle_det_items(ocr_result: Any) -> List[Any]:
    """收集 [box, (text, conf)] 条目（与 _extract_ocr_lines 解包逻辑一致）。"""
    if ocr_result is None:
        return []
    if isinstance(ocr_result, dict):
        return []
    if not isinstance(ocr_result, list) or not ocr_result:
        return []
    if _looks_like_paddle_det_line(ocr_result):
        return [ocr_result]
    if _looks_like_paddle_det_line(ocr_result[0]):
        return [x for x in ocr_result if _looks_like_paddle_det_line(x)]
    if (
        len(ocr_result) == 1
        and isinstance(ocr_result[0], list)
        and ocr_result[0]
        and _looks_like_paddle_det_line(ocr_result[0][0])
    ):
        return [x for x in ocr_result[0] if _looks_like_paddle_det_line(x)]
    return []


def _boxes_and_texts_from_result(ocr_result: Any) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for it in _iter_paddle_det_items(ocr_result):
        t = _text_from_paddle_line(it)
        if not t:
            continue
        out.append((t, it[0]))
    return out


def _box_y_min(box: Any) -> float:
    if not isinstance(box, (list, tuple)):
        return 0.0
    ys = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
    return min(ys) if ys else 0.0


def _box_x_min(box: Any) -> float:
    if not isinstance(box, (list, tuple)):
        return 0.0
    xs = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
    return min(xs) if xs else 0.0


def _quad_to_xyxy(box: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(box, (list, tuple)) or len(box) < 3:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for p in box:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _union_digit_boxes_xyxy(
    entries: List[Tuple[str, Any]], *, pad: int = 4
) -> Optional[Tuple[int, int, int, int]]:
    rects: List[Tuple[float, float, float, float]] = []
    for t, box in entries:
        if not re.search(r"\d", t or ""):
            continue
        q = _quad_to_xyxy(box)
        if q is None:
            continue
        rects.append(q)
    if not rects:
        return None
    x1 = min(r[0] for r in rects) - pad
    y1 = min(r[1] for r in rects) - pad
    x2 = max(r[2] for r in rects) + pad
    y2 = max(r[3] for r in rects) + pad
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def _proc_xyxy_to_sub_xyxy(
    rx1: int,
    ry1: int,
    rx2: int,
    ry2: int,
    proc_shape: Tuple[int, int, int],
    sub_shape: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    ph, pw = int(proc_shape[0]), int(proc_shape[1])
    sh, sw = int(sub_shape[0]), int(sub_shape[1])
    if pw <= 0 or ph <= 0:
        return 0, 0, sw, sh
    sx = sw / float(pw)
    sy = sh / float(ph)
    return (
        int(round(rx1 * sx)),
        int(round(ry1 * sy)),
        int(round(rx2 * sx)),
        int(round(ry2 * sy)),
    )


def _ocr_raw_from_engine(ocr_engine: Any, proc: np.ndarray) -> Any:
    if hasattr(ocr_engine, "predict"):
        out = ocr_engine.predict(proc)
        return _normalize_predict_result(out) if out is not None else None
    for call in (
        lambda: ocr_engine.ocr(proc, cls=True),
        lambda: ocr_engine.ocr(proc),
    ):
        try:
            out = call()
            if out is None:
                continue
            return _normalize_predict_result(out)
        except Exception:
            continue
    return None


def _read_text_boxes(ocr_engine: Any, proc: np.ndarray) -> List[Tuple[str, Any]]:
    """统一输出 (文本, 框)；仅 PaddleOCR。"""
    if proc.size == 0:
        return []
    raw = _ocr_raw_from_engine(ocr_engine, proc)
    return _boxes_and_texts_from_result(raw)


def _ocr_boxes(ocr_engine: Any, crop_bgr: np.ndarray) -> List[Tuple[str, Any]]:
    """OCR 返回 (文本, 框) 列表，框坐标为裁剪图内。"""
    if crop_bgr.size == 0:
        return []
    proc = _maybe_upscale(crop_bgr)
    return _read_text_boxes(ocr_engine, proc)


def _ocr_boxes_game_text(ocr_engine: Any, crop_bgr: np.ndarray) -> List[Tuple[str, Any]]:
    """
    纯色/渐变底上的文字与数字：单路 OCR（不再做原图/增强图双路择优）。
    保留适度放大，减少重复 OCR 调用开销。
    """
    if crop_bgr.size == 0:
        return []
    proc = _maybe_upscale(crop_bgr, min_side=64)
    proc = _scale_crop_min_height(proc, min_h=100)
    return _read_text_boxes(ocr_engine, proc)


def _hp_fixed_digit_rects_ref() -> List[Tuple[int, int, int, int]]:
    """参考坐标系下 8 个血量数字 ROI（Top1..Top8）：60×36，中心 (_HP_DIGIT_CX_REF, cy)。"""
    half_w = _HP_DIGIT_W_REF // 2
    half_h = _HP_DIGIT_H_REF // 2
    cx = _HP_DIGIT_CX_REF
    rects: List[Tuple[int, int, int, int]] = []
    for cy in _HP_DIGIT_CENTER_Y_REF:
        rects.append((cx - half_w, cy - half_h, cx + half_w, cy + half_h))
    return rects


def _parse_hp_in_digit_rect(
    ocr_engine: Any,
    bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    self_mode: bool,
) -> Tuple[str, str, Optional[Tuple[int, int, int, int]]]:
    """在给定血量数字 ROI 内 OCR 并解析血量；返回 (hp, ocr_joined, digit_box_union)。"""
    if x2 <= x1 or y2 <= y1:
        return "", "", None
    crop = bgr[y1:y2, x1:x2]
    ent, txt = _ocr_hp_crop_entries(ocr_engine, crop, self_mode=self_mode)
    glob = _offset_entries_to_global(ent, x1, y1)
    hp_txt_n = _hp_parse_concat_from_entries(glob)
    hp_pick, digit_box = _hp_pick_value_and_digit_union(glob)
    hp = _hp_reconcile_pick_and_text(hp_pick, hp_txt_n)
    if (
        hp_txt_n
        and hp == hp_txt_n
        and len(hp_txt_n) > len(hp_pick or "")
    ):
        ub = _union_digit_boxes_xyxy(
            [(t, b) for t, b in glob if re.search(r"\d", t or "")],
            pad=2,
        )
        if ub is not None:
            digit_box = ub
    has_any_digit = _entries_contain_digit(glob)
    if not hp:
        ft = re.sub(r"\s+", "", _hp_raw_text_from_entries(glob))
        compact = _parse_hp_digits_from_ocr_text(ft)
        if not compact:
            compact = _parse_hp_digits_from_ocr_text(re.sub(r"\s+", "", txt or ""))
        if compact:
            hp = compact
    if not hp:
        hp = "0" if not has_any_digit else ""
    return hp, txt, digit_box


def _scale_hp_rect_ref_to_image(
    rect: Tuple[int, int, int, int],
    w_img: int,
    h_img: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    sx = w_img / float(_HP_REF_W)
    sy = h_img / float(_HP_REF_H)
    nx1 = max(0, int(round(x1 * sx)))
    ny1 = max(0, int(round(y1 * sy)))
    nx2 = min(w_img, int(round(x2 * sx)))
    ny2 = min(h_img, int(round(y2 * sy)))
    if nx2 <= nx1:
        nx2 = min(w_img, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h_img, ny1 + 1)
    return (nx1, ny1, nx2, ny2)


def _hp_dark_digit_variant(bgr: np.ndarray) -> np.ndarray:
    """偏暗数字（如 0、小字）：提亮 + 自适应二值，供 self_mode OCR 加跑一路。"""
    if bgr.size == 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.48, beta=18)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -2
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def _hp_raw_text_from_entries(entries: List[Tuple[str, Any]]) -> str:
    if not entries:
        return ""
    return " ".join(t for t, _ in _sort_entries_reading_order(entries))


def _hp_parse_concat_from_entries(entries: List[Tuple[str, Any]]) -> str:
    if not entries:
        return ""
    ts = " ".join(t for t, _ in _sort_entries_reading_order(entries))
    return _parse_hp_digits_from_ocr_text(re.sub(r"\s+", "", ts))


def _hp_reconcile_pick_and_text(hp_pick: str, hp_txt: str) -> str:
    """整段 OCR 拼接往往比子集框更完整（如 100）；优先更长且有效的血量串。"""
    if not hp_txt:
        return hp_pick
    if not hp_pick:
        return hp_txt
    if hp_txt == hp_pick:
        return hp_txt
    if len(hp_txt) > len(hp_pick):
        return hp_txt
    if len(hp_txt) < len(hp_pick):
        return hp_pick
    return hp_txt


def _ocr_hp_crop_entries(
    ocr_engine: Any, crop_bgr: np.ndarray, *, self_mode: bool = False
) -> Tuple[List[Tuple[str, Any]], str]:
    """
    小 ROI OCR：放大 + 原图/CLAHE 各跑一遍；返回的框坐标已映射回 crop_bgr 尺寸（非检测图尺寸）。
    self_mode：略放大输入并加跑暗字二值一路，便于「我」血条多位数与暗 0。
    """
    if crop_bgr.size == 0:
        return [], ""
    min_side = 64 if self_mode else 56
    min_h = 80 if self_mode else 72
    proc = _maybe_upscale(crop_bgr, min_side=min_side)
    proc = _scale_crop_min_height(proc, min_h=min_h)
    variants: List[np.ndarray] = [proc, _preprocess_ui_text_bgr(proc)]
    if self_mode:
        variants.append(_hp_dark_digit_variant(proc))
    best_ent: List[Tuple[str, Any]] = []
    best_sc = (-1, -1)
    best_img = proc
    for img in variants:
        ent = _read_text_boxes(ocr_engine, img)
        sc = _ocr_score(ent)
        if sc > best_sc:
            best_sc = sc
            best_ent = ent
            best_img = img
    mapped: List[Tuple[str, Any]] = []
    for t, box in best_ent:
        q = _quad_to_xyxy(box)
        if q is None:
            continue
        x1, y1, x2, y2 = q
        mx1, my1, mx2, my2 = _proc_xyxy_to_sub_xyxy(
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
            best_img.shape,
            crop_bgr.shape,
        )
        box_crop = [
            [mx1, my1],
            [mx2, my1],
            [mx2, my2],
            [mx1, my2],
        ]
        mapped.append((t, box_crop))
    txt = " ".join(
        t
        for t, _ in sorted(
            mapped, key=lambda z: (_box_y_min(z[1]), _box_x_min(z[1]))
        )
    ).strip()
    return mapped, txt


def _offset_entries_to_global(
    entries: List[Tuple[str, Any]], ox: int, oy: int
) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for t, box in entries:
        q = _quad_to_xyxy(box)
        if q is None:
            continue
        x1, y1, x2, y2 = q
        box_g = [
            [x1 + ox, y1 + oy],
            [x2 + ox, y1 + oy],
            [x2 + ox, y2 + oy],
            [x1 + ox, y2 + oy],
        ]
        out.append((t, box_g))
    return out


def _hp_pick_value_and_digit_union(
    entries: List[Tuple[str, Any]],
) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    """
    在固定血量 ROI 内 OCR 得到的框中，拼接读序文本解析血量；返回血量与数字框并集（全图坐标）。
    """
    digit_entries = [(t, b) for t, b in entries if re.search(r"\d", t or "")]
    if not digit_entries:
        return "", None
    digit_entries.sort(key=lambda z: (_box_y_min(z[1]), _box_x_min(z[1])))

    def _try_subset(sub: List[Tuple[str, Any]]) -> str:
        compact = "".join(re.sub(r"\s+", "", t or "") for t, _ in sub)
        return _parse_hp_digits_from_ocr_text(compact)

    hp0 = _try_subset(digit_entries)
    if hp0:
        for k in range(1, len(digit_entries) + 1):
            sub = digit_entries[-k:]
            if _try_subset(sub) == hp0:
                return hp0, _union_digit_boxes_xyxy(sub, pad=2)
        return hp0, _union_digit_boxes_xyxy(digit_entries, pad=2)
    # 优先从右侧取连续子串（血量多在右侧）
    for k in range(len(digit_entries), 0, -1):
        sub = digit_entries[-k:]
        hp = _try_subset(sub)
        if hp:
            return hp, _union_digit_boxes_xyxy(sub, pad=2)
    # 再从左侧尝试（偶有左侧数字）
    for k in range(len(digit_entries), 0, -1):
        sub = digit_entries[:k]
        hp = _try_subset(sub)
        if hp:
            return hp, _union_digit_boxes_xyxy(sub, pad=2)
    return "", None


def _entries_contain_digit(entries: List[Tuple[str, Any]]) -> bool:
    return any(re.search(r"\d", t or "") for t, _ in entries)


def _cluster_text_boxes_by_y(
    entries: List[Tuple[str, Any]],
    y_gap: float,
) -> List[List[Tuple[str, Any]]]:
    """按框顶 y 聚成多行，行内按 x 排序。"""
    if not entries:
        return []
    es = sorted(entries, key=lambda t: _box_y_min(t[1]))
    rows: List[List[Tuple[str, Any]]] = []
    cur: List[Tuple[str, Any]] = [es[0]]
    ref_y = _box_y_min(es[0][1])
    for t, box in es[1:]:
        yb = _box_y_min(box)
        if abs(yb - ref_y) <= y_gap:
            cur.append((t, box))
            ref_y = sum(_box_y_min(b[1]) for b in cur) / len(cur)
        else:
            rows.append(sorted(cur, key=lambda x: _box_x_min(x[1])))
            cur = [(t, box)]
            ref_y = yb
    rows.append(sorted(cur, key=lambda x: _box_x_min(x[1])))
    return rows


def _parse_bond_first_line(s: str) -> Tuple[Optional[str], str]:
    """
    首行布局：「3 护卫」「3护卫」「2 约德尔人」「2约德尔人」；也可能 OCR 成「护卫 3」。
    返回 (activation_digits, name_or_rest)
    """
    s = re.sub(r"\s+", " ", s.strip())
    if not s:
        return None, ""
    m = re.match(r"^(\d+)\s+(.+)$", s)
    if m:
        return m.group(1), m.group(2).strip()
    m2 = re.match(r"^(.+?)\s+(\d+)$", s)
    if m2:
        left, right = m2.group(1).strip(), m2.group(2)
        if re.search(r"[\u4e00-\u9fffA-Za-z·]", left):
            return right, left
    # 无空格紧贴：UI 常见「2约德尔人」，仅靠上一分支会漏判为未激活
    compact = re.sub(r"\s+", "", s)
    m3 = re.match(r"^([0-9]+)([\u4e00-\u9fffA-Za-z·]+)$", compact)
    if m3:
        return m3.group(1), m3.group(2)
    return None, s


def _strip_bond_tier_text(s: str) -> str:
    """去掉羁绊梯度（如 2/4/6、2 / 4 / 6），仅保留名称与激活数相关片段。"""
    s = re.sub(r"\s*\d+\s*/\s*\d+(?:\s*/\s*\d+)*\s*", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _try_parse_activated_bond_piece(
    slot_rows: List[List[Tuple[str, Any]]],
) -> Optional[str]:
    """
    从一名羁绊的 OCR 行组解析「数字+羁绊名」片段（如 3护卫、2狙神）。
    仅当首段可解析出激活数 >=1 时返回；仅有羁绊名无数量（如虚空、迅击战士）返回 None。
    """
    if not slot_rows:
        return None

    # 优先按 token 解析：先去梯度文本，再在剩余 token 里抽取激活数与名称，
    # 可减少「数字粘到后缀/梯度」导致的漏判。
    tokens: List[str] = []
    for row in slot_rows[:4]:
        for t, _ in row:
            st = re.sub(r"\s+", " ", str(t).strip())
            if st:
                tokens.append(st)
    if tokens:
        keep: List[str] = []
        for tk in tokens:
            # 梯度 token：2/4/6、1-6、3 / 5 / 7 等，直接过滤
            if re.search(r"\d+\s*[/-]\s*\d+", tk):
                continue
            if re.fullmatch(r"[?°`~!@#$%^&*_+=<>\\|]+", tk):
                continue
            keep.append(tk)
        # 激活数字通常是独立短数字，优先取 1~9
        cand_nums: List[str] = []
        for tk in keep:
            if re.fullmatch(r"\d{1,2}", tk):
                cand_nums.append(tk)
                continue
            mnum = re.match(r"^(\d{1,2})([\u4e00-\u9fffA-Za-z·]+)$", tk)
            if mnum:
                cand_nums.append(mnum.group(1))
        act = ""
        for n in cand_nums:
            try:
                iv = int(n)
                if 1 <= iv <= 9:
                    act = str(iv)
                    break
            except ValueError:
                continue
        name_parts: List[str] = []
        for tk in keep:
            # 去掉 token 内部的前置数字，仅保留名称片段
            s = re.sub(r"^\d{1,2}", "", tk).strip()
            if not s:
                continue
            if re.search(r"[\u4e00-\u9fffA-Za-z·]", s):
                name_parts.append(s)
        if act and name_parts:
            name = re.sub(r"\s+", " ", "".join(name_parts)).strip()
            if name:
                return f"{act}{name}"

    merged_lines: List[str] = []
    for row in slot_rows[:4]:
        merged_lines.append(" ".join(t for t, _ in row))
    top = merged_lines[0].strip() if merged_lines else ""
    rest = " ".join(merged_lines[1:]).strip() if len(merged_lines) > 1 else ""
    # 行顺序不稳定时（梯度行在上）：合并去梯度后再解析
    all_joined = _strip_bond_tier_text(" ".join(merged_lines).strip())
    if all_joined:
        act0, name0 = _parse_bond_first_line(all_joined)
        if act0 and name0:
            try:
                n0 = int(act0)
                if 1 <= n0 <= 9:
                    return f"{act0}{name0.strip()}"
            except ValueError:
                pass
    tc = _strip_bond_tier_text(top)
    act, name = _parse_bond_first_line(tc)
    if act and name:
        try:
            n = int(act)
            if not (1 <= n <= 9):
                return None
            return f"{act}{name.strip()}"
        except ValueError:
            pass
    full = _strip_bond_tier_text((top + " " + rest).strip())
    act2, name2 = _parse_bond_first_line(full)
    if act2 and name2:
        try:
            n2 = int(act2)
            if not (1 <= n2 <= 9):
                return None
            return f"{act2}{name2.strip()}"
        except ValueError:
            pass
    comp = re.sub(r"\s+", "", full)
    # 名称可为单字（如「狙」），勿用「首字+.+」误伤
    m = re.match(r"^(\d+)([\u4e00-\u9fffA-Za-z·]+)$", comp)
    if m:
        try:
            if 1 <= int(m.group(1)) <= 9:
                return f"{m.group(1)}{m.group(2)}"
        except ValueError:
            pass
    return None


def _bond_line_rows_from_cell(
    entries: List[Tuple[str, Any]],
    cell_h: int,
) -> List[List[Tuple[str, Any]]]:
    """单格内按 y 分成名称行 / 梯度行等，供与整块槽相同的解析逻辑使用。"""
    if not entries:
        return []
    gap = max(6.0, min(float(cell_h) * 0.14, float(cell_h) * 0.42))
    return _cluster_text_boxes_by_y(entries, gap)


def _build_bond_fields_line(fields: List[str]) -> str:
    if not fields:
        return "识别字段：无"
    return "识别字段：" + " / ".join(fields)


def _extract_digits_only(s: str) -> str:
    ds = re.findall(r"\d+", s or "")
    if not ds:
        return ""
    # 激活数优先取最短且靠前的数字块（通常为 1 位）
    ds = sorted(ds, key=lambda x: (len(x), s.find(x)))
    for d in ds:
        try:
            iv = int(d)
            if 1 <= iv <= 9:
                return str(iv)
        except ValueError:
            continue
    return ds[0]


def _parse_hp_digits_from_ocr_text(s: str) -> str:
    """从血量小 ROI 的 OCR 拼接串中解析 0~120。"""
    t = re.sub(r"\s+", "", s or "")
    if not t:
        return ""
    for seg in sorted(re.findall(r"\d+", t), key=len, reverse=True):
        if len(seg) <= 3:
            try:
                iv = int(seg)
                if 0 <= iv <= 120:
                    return str(iv)
            except ValueError:
                pass
            continue
        for w in (3, 2, 1):
            for i in range(len(seg) - w + 1):
                chunk = seg[i : i + w]
                try:
                    iv = int(chunk)
                    if 0 <= iv <= 120:
                        return str(iv)
                except ValueError:
                    continue
    if re.search(r"[Oo〇０]", t) and not re.search(r"\d", t):
        return "0"
    return ""


def _extract_chinese_only(s: str) -> str:
    parts = re.findall(r"[\u4e00-\u9fff]+", s or "")
    if not parts:
        return ""
    return "".join(parts).strip()


def _ocr_compact_text(ocr_engine: Any, crop_bgr: np.ndarray) -> str:
    if crop_bgr.size == 0:
        return ""
    entries = _ocr_boxes_game_text(ocr_engine, crop_bgr)
    return " ".join(t for t, _ in _sort_entries_reading_order(entries)).strip()


def _parse_bond_piece_precise(
    ocr_engine: Any,
    bgr: np.ndarray,
    bx1: int,
    by1: int,
    bx2: int,
    by2: int,
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    精细分区解析：
    1) 左 30px 读纯数字（激活数）
    2) 紧随其后 90~120px 读纯中文（羁绊名）
    3) 若数字缺失：在首个中文字符下方约 20x20 小窗再读数字
    返回 (「数字+中文名」或空串, 调试采样框列表)。
    """
    cell_w = max(1, bx2 - bx1)
    num_w = max(18, min(30, cell_w))
    name_w = min(120, max(90, cell_w - num_w - 8))
    name_w = max(40, min(name_w, max(1, cell_w - num_w)))
    nx1, nx2 = bx1, min(bx2, bx1 + num_w)
    # 未激活行名称会略偏左，中文窗向左回退一点，减少名称被数字窗吃掉
    tx1 = max(bx1, nx2 - 8)
    tx2 = min(bx2, tx1 + name_w)
    debug_rects: List[Tuple[int, int, int, int]] = [(nx1, by1, nx2, by2), (tx1, by1, tx2, by2)]

    num_raw = _ocr_compact_text(ocr_engine, bgr[by1:by2, nx1:nx2])
    name_crop = bgr[by1:by2, tx1:tx2]
    name_raw = _ocr_compact_text(ocr_engine, name_crop)
    n = _extract_digits_only(num_raw)
    nm = _extract_chinese_only(name_raw)

    # 兜底：左侧数字窗没有识别到时，在首个中文字符下方偏左 25x25 再读一遍数字
    if not n and nm:
        entries = _ocr_boxes_game_text(ocr_engine, name_crop)
        first_x = tx1
        for t, box in _sort_entries_reading_order(entries):
            if re.search(r"[\u4e00-\u9fff]", t or ""):
                first_x = tx1 + int(_box_x_min(box))
                break
        sq = 25
        sx1 = max(bx1, min(first_x - 10, bx2 - sq))
        sy1 = max(by1, min(by1 + int((by2 - by1) * 0.66) - 4, by2 - sq))
        sx2, sy2 = sx1 + sq, sy1 + sq
        debug_rects.append((sx1, sy1, sx2, sy2))
        n2_raw = _ocr_compact_text(ocr_engine, bgr[sy1:sy2, sx1:sx2])
        n = _extract_digits_only(n2_raw)

    # 最后兜底：若仍无数字，但整格有 x/y 结构（如 1/2），取斜杠前数字
    if not n and nm:
        cell_raw = _ocr_compact_text(ocr_engine, bgr[by1:by2, bx1:bx2])
        m = re.search(r"(\d{1,2})\s*/\s*\d{1,2}", cell_raw)
        if m:
            try:
                iv = int(m.group(1))
                if 1 <= iv <= 9:
                    n = str(iv)
            except ValueError:
                pass

    if n and nm:
        return f"{n}{nm}", debug_rects
    return "", debug_rects


def _draw_bond_summary_below_roi(
    vis: np.ndarray,
    x1: int,
    x2: int,
    y2: int,
    text: str,
    color: Tuple[int, int, int],
    font_size: int,
    *,
    h_img: int,
) -> None:
    """在羁绊 ROI 下沿下方输出整段汇总（自动换行）。"""
    t = text.strip()
    if not t:
        return
    fs = max(10, font_size - 2)
    line_h = fs + 6
    margin = 10
    max_w = max(26, min(44, max(1, (x2 - x1) // 11)))
    wrapped = textwrap.fill(t, width=max_w)
    y = y2 + margin
    for ln in wrapped.splitlines():
        s = ln.strip()
        if not s:
            continue
        if y + line_h > h_img - 2:
            break
        _draw_chinese_text_bgr(vis, s, (x1, y), font_size=fs, color=color)
        y += line_h


def _process_bonds_grid(
    vis: np.ndarray,
    bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    ocr_engine: Any,
    color: Tuple[int, int, int],
    font_size: int,
    *,
    cols: int,
    rows: int,
    h_img: int,
) -> Dict[str, Any]:
    """
    羁绊栏按网格分块（默认 1×7）：每格单独 OCR；
    每格在右侧标注该行识别到的完整内容；ROI 下沿输出整段「识别字段：…」。
    """
    blocks = _split_roi_grid(x1, y1, x2, y2, cols=cols, rows=rows)
    pieces: List[str] = []
    fields: List[str] = []
    bond_cells: List[Dict[str, Any]] = []
    raws: List[str] = []
    row_fs = max(10, font_size - 4)
    est_h = row_fs + 4

    fast_mode = _fast_ocr_enabled()
    for bi, (bx1, by1, bx2, by2) in enumerate(blocks):
        crop = bgr[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        sample_rects: List[Tuple[int, int, int, int]] = []
        entries = _ocr_boxes_game_text(ocr_engine, crop)
        merged = " ".join(t for t, _ in _sort_entries_reading_order(entries))
        raws.append(merged)
        cell_h = max(1, by2 - by1)
        line_rows = _bond_line_rows_from_cell(entries, cell_h)
        piece = _try_parse_activated_bond_piece(line_rows) if line_rows else None
        # 精细分区 OCR 成本较高：仅在常规解析失败且非 fast 模式时兜底。
        if not piece and (not fast_mode):
            piece, sample_rects = _parse_bond_piece_precise(
                ocr_engine, bgr, bx1, by1, bx2, by2
            )
        ocr_cell = (
            " | ".join(" ".join(t for t, _ in r) for r in line_rows)
            if line_rows
            else merged
        )
        bond_cells.append(
            {
                "index": bi + 1,
                "ocr_rows": ocr_cell,
                "piece": piece,
                "raw": merged,
            }
        )
        if piece:
            pieces.append(piece)
            fields.append(piece)
        elif merged.strip():
            fields.append(re.sub(r"\s+", " ", merged.strip()))
        _draw_roi_rect(vis, bx1, by1, bx2, by2, color, 1)
        # 调试用：标出采样框（数字窗、中文名窗、缺数字时的20x20兜底数字窗）
        # 采样框颜色区分：
        # idx=0 左侧数字窗（黄色），idx=1 中文窗（紫色），idx>=2 兜底数字窗（青色）
        for i, (sx1, sy1, sx2, sy2) in enumerate(sample_rects):
            if i == 0:
                sample_col = (0, 255, 255)
            elif i == 1:
                sample_col = (255, 80, 220)
            else:
                sample_col = (255, 255, 0)
            _draw_roi_rect(vis, sx1, sy1, sx2, sy2, sample_col, 1)
        # 每行分块右侧标注该分块识别到的完整内容
        if piece:
            row_txt = piece
        else:
            row_txt = re.sub(r"\s+", " ", merged.strip()) if merged.strip() else "（空）"
        ry = by1 + max(0, ((by2 - by1) - est_h) // 2)
        _draw_chinese_text_bgr(
            vis,
            row_txt,
            (min(bx2 + 6, max(0, vis.shape[1] - 4)), ry),
            font_size=row_fs,
            color=color,
        )

    raw = " | ".join(raws)
    summary = _build_bond_fields_line(fields)
    _draw_bond_summary_below_roi(
        vis, x1, x2, y2, summary, color, font_size, h_img=h_img
    )
    return {
        "name": "羁绊栏",
        "raw": raw,
        "parsed": summary,
        "bond_items": pieces,
        "bond_summary": summary,
        "bond_cells": bond_cells,
    }


def _bond_roi_density_score(
    ocr_engine: Any,
    bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[int, int]:
    """
    评估候选羁绊区的文本密度分数。
    返回 (score, char_count)；score 越高越像有效羁绊文本区域。
    """
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return (-1, 0)
    entries = _ocr_boxes_game_text(ocr_engine, crop)
    joined = "".join(t for t, _ in entries)
    if not joined:
        return (0, 0)
    compact = re.sub(r"\s+", "", joined)
    chars = len(compact)
    zh = len(re.findall(r"[\u4e00-\u9fff]", compact))
    nums = len(re.findall(r"\d", compact))
    slash_pairs = len(re.findall(r"\d+\s*/\s*\d+", joined))
    noise = len(re.findall(r"[?°`~!@#$%^&*_+=<>\\|]", joined))
    score = chars + zh * 2 + nums + slash_pairs * 4 - noise * 2
    return (score, chars)


def _pick_bonds_roi(
    ocr_engine: Any,
    bgr: np.ndarray,
    w_img: int,
    h_img: int,
    *,
    dy: int = 0,
) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
    """
    在两个固定候选区域中选择文本密度更高者作为羁绊区。
    dy：整块羁绊区在 y 方向平移（像素，正数为下移），用于副图 *-b 与主图 UI 差。
    """
    candidates = [
        (260, 110, 480, 690),
        (180, 110, 380, 690),
    ]
    scored: List[Dict[str, Any]] = []
    for (x1, y1, x2, y2) in candidates:
        y1, y2 = y1 + int(dy), y2 + int(dy)
        cx1 = max(0, min(x1, w_img - 1))
        cx2 = max(cx1 + 1, min(x2, w_img))
        cy1 = max(0, min(y1, h_img - 1))
        cy2 = max(cy1 + 1, min(y2, h_img))
        sc, ch = _bond_roi_density_score(ocr_engine, bgr, cx1, cy1, cx2, cy2)
        scored.append({"rect": (cx1, cy1, cx2, cy2), "score": sc, "chars": ch})
    scored.sort(key=lambda d: (int(d["score"]), int(d["chars"])), reverse=True)
    best = scored[0]
    return best["rect"], {"candidates": scored, "selected": best}


def _sort_entries_reading_order(
    entries: List[Tuple[str, Any]],
) -> List[Tuple[str, Any]]:
    return sorted(entries, key=lambda t: (_box_y_min(t[1]), _box_x_min(t[1])))


def _process_hp_column(
    vis: np.ndarray,
    bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    ocr_engine: Any,
    color: Tuple[int, int, int],
    font_size: int,
    *,
    h_img: int,
) -> Dict[str, Any]:
    """
    血量：在参考分辨率下固定的 60×36 数字 ROI（中心 x=2075，8 行中心 y 给定）上按图像缩放裁切，
    仅在裁切区内 OCR 并解析数字；ROI 内完全无数字时记为 0 血。
    若该行 ID 区未识别到有效昵称（显示为「我」），则将血量 ROI 向左、上各扩展 40 像素后重新 OCR。
    左侧输出 Top、玩家 ID、血量；标注图画最终使用的血量 ROI 矩形。
    """
    _ = (x1, y1, x2, y2, h_img)
    h_i, w_img = int(bgr.shape[0]), int(bgr.shape[1])
    fx1, fy1, fx2, fy2 = _scale_hp_rect_ref_to_image((1970, 110, 2140, 880), w_img, h_i)

    fs = max(11, font_size - 3)
    parsed_lines: List[str] = []
    player_cells: List[Dict[str, Any]] = []
    raw_rows: List[str] = []
    id_w, id_h = 300, 40
    fixed_ix1 = max(0, min(fx1 - 120, w_img - id_w))

    fixed_refs = _hp_fixed_digit_rects_ref()
    digit_draw_col = color

    for idx, nref in enumerate(fixed_refs):
        nx1, ny1, nx2, ny2 = _scale_hp_rect_ref_to_image(nref, w_img, h_i)
        if idx == 0:
            row_top = fy1
        else:
            prev = _scale_hp_rect_ref_to_image(fixed_refs[idx - 1], w_img, h_i)
            row_top = (prev[3] + ny1) // 2
        if idx == 7:
            row_bot = fy2
        else:
            nxt = _scale_hp_rect_ref_to_image(fixed_refs[idx + 1], w_img, h_i)
            row_bot = (ny2 + nxt[1]) // 2

        hp, txt_n, digit_box = _parse_hp_in_digit_rect(
            ocr_engine, bgr, nx1, ny1, nx2, ny2, self_mode=False
        )
        hp_source = "fixed_roi"
        ocr_used = txt_n

        if digit_box is not None:
            hpy_abs = digit_box[1]
        else:
            hpy_abs = ny1
        ix1 = fixed_ix1
        iy2 = max(id_h, hpy_abs - 2)
        iy1 = max(0, iy2 - id_h)
        ix2 = min(w_img, ix1 + id_w)
        id_crop = bgr[iy1:iy2, ix1:ix2]
        id_raw = _ocr_compact_text(ocr_engine, id_crop)
        id_text = re.sub(r"\s+", "", id_raw)
        id_text = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_·]+", id_text))
        if re.fullmatch(r"\d{1,3}", id_text or ""):
            id_text = ""
        if not id_text:
            id_text = "我"

        # 「我」：ID 区无有效昵称（空串经上处理后默认「我」，或 OCR 直接为「我」）。
        # 将血量 ROI 向左、上各扩展 40px 后重跑 OCR（self_mode 便于偏暗数字）。
        hp_roi_self_expanded = False
        if id_text == "我":
            ex1 = max(0, nx1 - _HP_SELF_EXPAND_LEFT_PX)
            ey1 = max(0, ny1 - _HP_SELF_EXPAND_UP_PX)
            hp, txt_n, digit_box = _parse_hp_in_digit_rect(
                ocr_engine, bgr, ex1, ey1, nx2, ny2, self_mode=True
            )
            ocr_used = txt_n
            hp_source = "fixed_roi_self_expanded"
            hp_roi_self_expanded = True
            tx1, ty1, tx2, ty2 = ex1, ey1, nx2, ny2
        else:
            tx1, ty1, tx2, ty2 = nx1, ny1, nx2, ny2

        rank = idx + 1
        hp_show = hp if hp else "?"
        line = f"Top{rank} {id_text} {hp_show}血"
        _draw_roi_rect(vis, tx1, ty1, tx2, ty2, digit_draw_col, 2)

        draw_x = max(2, fx1 - 360)
        draw_y = row_top + max(0, ((row_bot - row_top) - (fs + 4)) // 2)
        _draw_chinese_text_bgr(vis, line, (draw_x, draw_y), font_size=fs, color=color)

        parsed_lines.append(line)
        raw_rows.append(ocr_used)
        player_cells.append(
            {
                "display": line,
                "hp": hp_show,
                "id_text": id_text,
                "row_rect": [fx1, row_top, fx2, row_bot],
                "hp_rect": [tx1, ty1, tx2, ty2],
                "hp_digit_rect": [tx1, ty1, tx2, ty2],
                "hp_digit_rect_loose": (
                    list(digit_box) if digit_box is not None else None
                ),
                "hp_source": hp_source,
                "hp_roi_self_expanded": hp_roi_self_expanded,
                "hp_roi_nominal": [nx1, ny1, nx2, ny2],
                "id_rect": [ix1, iy1, ix2, iy2],
                "ocr_rows": ocr_used,
                "draw_xy": [draw_x, draw_y],
            }
        )

    return {
        "name": "血量/昵称",
        "raw": " | ".join(raw_rows),
        "parsed": parsed_lines,
        "player_lines": parsed_lines,
        "player_cells": player_cells,
        "ocr_row_dump": " | ".join(raw_rows),
        "roi_selected": [fx1, fy1, fx2, fy2],
    }


def _set_ppocr_log_level(*, verbose: bool) -> None:
    """压低 ppocr / paddle 的 WARNING 刷屏（含 angle classifier 提示）。"""
    if verbose:
        warnings.filterwarnings("default")
        for name in ("ppocr", "paddleocr", "paddle"):
            lg = logging.getLogger(name)
            lg.disabled = False
            lg.setLevel(logging.DEBUG)
        return
    warnings.filterwarnings("ignore")
    for name in ("ppocr", "paddleocr", "paddle", "paddle.fluid"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.disabled = True
    for name in list(logging.root.manager.loggerDict.keys()):
        if not isinstance(name, str):
            continue
        low = name.lower()
        if "ppocr" in low or "paddle" in low:
            lg = logging.getLogger(name)
            lg.setLevel(logging.CRITICAL)
            lg.disabled = True


def _create_paddle_ocr(*, verbose: bool = False) -> Any:
    _set_ppocr_log_level(verbose=verbose)
    try:
        # 必须在 import 前已调用 _set_ppocr_log_level
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise SystemExit(
            "未安装 paddleocr。请执行: pip install paddlepaddle paddleocr\n"
            f"原始错误: {e}"
        ) from e
    # 对局 HUD 多为正向文字：优先关闭角度分类器，避免每次推理都提示 cls 未使用。
    # 兼容 2.x / 3.x：可再尝试 use_textline_orientation / use_angle_cls=True。
    attempts: List[Dict[str, Any]] = [
        {"use_angle_cls": False, "lang": "ch", "show_log": False},
        {"use_textline_orientation": True, "lang": "ch", "show_log": False},
        {"use_angle_cls": True, "lang": "ch", "show_log": False},
        {"lang": "ch", "show_log": False},
        {"lang": "ch"},
    ]
    last_err: Optional[Exception] = None
    for kw in attempts:
        try:
            ocr = PaddleOCR(**kw)
            try:
                import paddle

                paddle.set_flags({"FLAGS_use_mkldnn": False})
            except Exception:
                pass
            # import 后可能新建子 logger，再压一次
            _set_ppocr_log_level(verbose=verbose)
            return ocr
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法初始化 PaddleOCR，最后一次错误: {last_err}")


def _create_ocr_engine(*, verbose: bool = False) -> Any:
    return _create_paddle_ocr(verbose=verbose)


def _normalize_predict_result(res: Any) -> Any:
    """predict/ocr 可能返回 generator 或单元素列表。"""
    if res is None:
        return None
    if hasattr(res, "__iter__") and not isinstance(res, (str, bytes, dict, np.ndarray)):
        try:
            lst = list(res)
            return lst[0] if len(lst) == 1 else lst
        except Exception:
            return res
    return res


def _ocr_one(ocr_engine: Any, crop_bgr: np.ndarray) -> List[str]:
    if crop_bgr.size == 0:
        return []
    proc = _maybe_upscale(crop_bgr)
    # 新版 PaddleOCR（paddleocr._pipelines.ocr）：ocr() 会 return self.predict(img, **kwargs)，
    # 默认带 cls 等参数，而 predict() 已不支持 cls → 只能直接 predict(proc)，切勿再调 ocr()。
    if hasattr(ocr_engine, "predict"):
        try:
            out = ocr_engine.predict(proc)
            if out is None:
                return []
            return _extract_ocr_lines(_normalize_predict_result(out))
        except Exception as e:
            raise RuntimeError(f"OCR predict 失败: {e}") from e

    # 旧版 2.x API：仅有 ocr(img, cls=...)
    last_err: Optional[Exception] = None
    for call in (
        lambda: ocr_engine.ocr(proc, cls=True),
        lambda: ocr_engine.ocr(proc),
    ):
        try:
            out = call()
            if out is None:
                continue
            return _extract_ocr_lines(_normalize_predict_result(out))
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise RuntimeError(f"OCR 调用失败: {last_err}") from last_err
    return []


def _gold_binarize_variant(bgr: np.ndarray) -> np.ndarray:
    """HUD 数字常为上亮下暗底；自适应二值化给 OCR 第三路输入。"""
    if bgr.size == 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -3
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def _score_gold_ocr_candidate(joined: str) -> Tuple[int, int, int]:
    """
    越高越好：最长数字串长度、数字总位数、非数字噪声惩罚。
    """
    if not joined.strip():
        return (0, 0, 0)
    parts = re.findall(r"\d+", joined)
    longest = max((len(x) for x in parts), default=0)
    digit_chars = sum(len(x) for x in parts)
    noise = len(re.sub(r"[\d\s]", "", joined))
    return (longest, digit_chars, -noise)


def _score_streak_ocr_candidate(joined: str) -> Tuple[int, int, int]:
    """连胜区：有数字、有胜负字、噪声少者优先。"""
    s = joined.strip()
    if not s:
        return (0, 0, 0)
    has_d = 1 if re.search(r"\d", s) else 0
    zh = 0
    if "胜" in s or "败" in s or "负" in s:
        zh = 2
    noise = len(re.sub(r"[\d\s连胜败负连·.]", "", s))
    return (has_d + zh, len(re.findall(r"\d+", s)), -noise)


def _ocr_gold_field(
    ocr_engine: Any, crop_bgr: np.ndarray
) -> Tuple[List[str], Optional[Tuple[int, int, int, int]]]:
    """
    金币专用：多路「去掉左侧图标」的裁剪 + 适中放大
    + 原图 / CLAHE / 自适应二值，按数字得分选最优。
    第二项为相对 crop_bgr 的紧框 (x1,y1,x2,y2)（仅含数字 OCR 框并集），用于把右侧画窄；失败为 None。
    """
    empty_t: Tuple[List[str], Optional[Tuple[int, int, int, int]]] = ([], None)
    if crop_bgr.size == 0:
        return empty_t
    if os.environ.get("BATTLE_UI_OCR_UI_ENHANCE", "1").strip() == "0":
        lines = _ocr_one(ocr_engine, crop_bgr)
        return lines, None

    h, w = crop_bgr.shape[:2]
    trim_left_ratios = (0.32, 0.24, 0.40, 0.12, 0.0)

    best_lines: List[str] = []
    best_score = (-1, -1, -999999)
    best_tight: Optional[Tuple[int, int, int, int]] = None

    fast_mode = _fast_ocr_enabled()
    should_stop = False
    for tr in trim_left_ratios:
        if tr <= 0:
            sub = crop_bgr
            x0_trim = 0
        else:
            x0_trim = int(round(w * tr))
            if x0_trim >= w - 8:
                continue
            sub = crop_bgr[:, x0_trim:]
        if sub.shape[1] < 12 or sub.shape[0] < 8:
            continue
        base = _maybe_upscale(sub, min_side=76)
        base = _scale_crop_min_height(base, min_h=88)
        variants = (
            base,
            _preprocess_ui_text_bgr(base),
            _gold_binarize_variant(base),
        )
        for proc in variants:
            entries = _read_text_boxes(ocr_engine, proc)
            lines = [t for t, _ in _sort_entries_reading_order(entries)]
            joined = _join_lines(lines)
            sc = _score_gold_ocr_candidate(joined)
            if sc > best_score:
                best_score = sc
                best_lines = lines
                uproc = _union_digit_boxes_xyxy(entries, pad=4)
                if uproc is not None:
                    px1, py1, px2, py2 = uproc
                    sx1, sy1, sx2, sy2 = _proc_xyxy_to_sub_xyxy(
                        px1, py1, px2, py2, proc.shape, sub.shape
                    )
                    tx1 = int(x0_trim + min(sx1, sx2))
                    ty1 = int(min(sy1, sy2))
                    tx2 = int(x0_trim + max(sx1, sx2))
                    ty2 = int(max(sy1, sy2))
                    tx1 = max(0, min(tx1, w - 1))
                    tx2 = max(tx1 + 1, min(tx2, w))
                    ty1 = max(0, min(ty1, h - 1))
                    ty2 = max(ty1 + 1, min(ty2, h))
                    best_tight = (tx1, ty1, tx2, ty2)
                else:
                    best_tight = None
                # fast 模式下识别到稳定数字后提前结束，避免重复尝试。
                if fast_mode and best_score[0] >= 2 and best_score[1] >= 2 and best_score[2] >= -1:
                    should_stop = True
                    break
        if should_stop:
            break

    return best_lines, best_tight


def _ocr_streak_field(ocr_engine: Any, crop_bgr: np.ndarray) -> List[str]:
    """
    连胜/连败：参照金币多路裁剪（避开左侧火焰/图标）+ 增强，提高数字与「胜/败」检出。
    """
    if crop_bgr.size == 0:
        return []
    if os.environ.get("BATTLE_UI_OCR_UI_ENHANCE", "1").strip() == "0":
        return _ocr_one(ocr_engine, crop_bgr)

    h, w = crop_bgr.shape[:2]
    trim_left_ratios = (0.34, 0.26, 0.42, 0.16, 0.0)
    best_lines: List[str] = []
    best_score = (-1, -1, -999999)

    fast_mode = _fast_ocr_enabled()
    should_stop = False
    for tr in trim_left_ratios:
        if tr <= 0:
            sub = crop_bgr
        else:
            x0 = int(round(w * tr))
            if x0 >= w - 8:
                continue
            sub = crop_bgr[:, x0:]
        if sub.shape[1] < 12 or sub.shape[0] < 8:
            continue
        base = _maybe_upscale(sub, min_side=72)
        base = _scale_crop_min_height(base, min_h=88)
        for proc in (base, _preprocess_ui_text_bgr(base), _gold_binarize_variant(base)):
            entries = _read_text_boxes(ocr_engine, proc)
            lines = [t for t, _ in _sort_entries_reading_order(entries)]
            joined = _join_lines(lines)
            sc = _score_streak_ocr_candidate(joined)
            if sc > best_score:
                best_score = sc
                best_lines = lines
                # fast 模式：同时出现「数字 + 胜/败字」后提前结束。
                if fast_mode and best_score[0] >= 3 and best_score[1] >= 1 and best_score[2] >= -2:
                    should_stop = True
                    break
        if should_stop:
            break

    return best_lines


def _join_lines(lines: Sequence[str]) -> str:
    return " ".join(x.strip() for x in lines if x.strip()).strip()


def _parse_phase(raw: str) -> str:
    s = raw.replace(" ", "")
    if not s:
        return ""
    return s


def _parse_slash_pair(raw: str) -> Optional[str]:
    m = re.search(r"(\d+)\s*/\s*(\d+)", raw)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def _parse_level(raw: str) -> str:
    s = raw.strip()
    if "级" in s:
        return s
    d = re.findall(r"\d+", raw)
    if not d:
        return s
    # 取最后一个数字块常为等级；展示为「10级」形式
    return f"{d[-1]}级"


def _parse_exp(raw: str) -> str:
    p = _parse_slash_pair(raw)
    return p if p else raw.strip()


def _parse_gold(raw: str) -> str:
    s = raw.strip()
    d = re.findall(r"\d+", s)
    if d:
        return max(d, key=len)
    t = re.sub(r"\s+", "", s)
    if re.fullmatch(r"[Oo〇０]", t):
        return "0"
    if re.fullmatch(r"[lI|｜!]", t):
        return "1"
    return s


def _parse_streak(raw: str, crop_bgr: Optional[np.ndarray] = None) -> str:
    s = raw.strip()
    m = re.search(r"(\d+)\s*连胜", s)
    if m:
        return f"连胜{m.group(1)}"
    m = re.search(r"连胜\s*(\d+)", s)
    if m:
        return f"连胜{m.group(1)}"
    m = re.search(r"(\d+)\s*连败", s)
    if m:
        return f"连败{m.group(1)}"
    m = re.search(r"连败\s*(\d+)", s)
    if m:
        return f"连败{m.group(1)}"
    d = re.findall(r"\d+", s)
    if not d:
        return "无连胜/连败"
    try:
        n = int(d[-1])
    except ValueError:
        return "无连胜/连败"
    if n <= 0:
        return "无连胜/连败"
    if "败" in s or "负" in s:
        return f"连败{n}"
    if "胜" in s:
        return f"连胜{n}"
    if crop_bgr is None or crop_bgr.size == 0:
        return f"连胜{n}"
    # 图标颜色：橙偏连胜，蓝偏连败；采样略放宽饱和度/亮度门槛
    h, w = crop_bgr.shape[:2]
    icon = crop_bgr[:, : max(12, int(w * 0.38))]
    hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
    hch = hsv[:, :, 0]
    sch = hsv[:, :, 1]
    vch = hsv[:, :, 2]
    sat_mask = (sch >= 55) & (vch >= 50)
    orange = int(np.count_nonzero(sat_mask & ((hch >= 0) & (hch <= 28))))
    blue = int(np.count_nonzero(sat_mask & ((hch >= 85) & (hch <= 135))))
    if blue > orange and blue >= 8:
        return f"连败{n}"
    if orange >= blue and orange >= 8:
        return f"连胜{n}"
    if blue > orange:
        return f"连败{n}"
    # 仅有数字时默认连胜（与金铲铲常见 HUD 一致；误判连败可再改 OCR）
    return f"连胜{n}"


def _parse_bonds(raw: str) -> str:
    p = _parse_slash_pair(raw)
    if p:
        return f"羁绊 {p}"
    return raw.strip()


def parse_field(key: str, raw: str) -> str:
    if key == "phase":
        return _parse_phase(raw)
    if key == "level":
        return _parse_level(raw)
    if key == "exp":
        return _parse_exp(raw)
    if key == "gold":
        return _parse_gold(raw)
    if key == "streak":
        return _parse_streak(raw)
    if key == "bonds":
        return _parse_bonds(raw)
    if key == "hp_nick":
        return _join_lines(raw.split()) if raw else ""
    return raw.strip()


def _draw_roi_rect(
    vis: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), color, thickness)


def _draw_chinese_text_bgr(
    bgr: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    *,
    font_size: int = 18,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """在 BGR 图上绘制 UTF-8 文本（仅渲染用，不参与识别）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        cv2.putText(bgr, text[:80], xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return
    font = None
    for fp in _FONT_CANDIDATES:
        if fp.exists():
            try:
                font = ImageFont.truetype(str(fp), font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    pil_color = (color[2], color[1], color[0])
    draw.text(xy, text, font=font, fill=pil_color)
    bgr[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _annotate_outer(
    vis: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    title: str,
    parsed: str,
    raw: str,
    color: Tuple[int, int, int],
    font_size: int,
    *,
    h_img: int,
    label_parsed_only: bool = False,
) -> None:
    """标注画在 ROI 框外：优先框上方，不够则框下方，避免盖住截图内文字。"""
    if label_parsed_only:
        line1 = parsed if parsed else "(空)"
    else:
        line1 = f"{title}: {parsed}" if parsed else f"{title}: (空)"
    line2 = f"OCR: {raw}" if raw else ""
    fs2 = max(12, font_size - 4)
    line_h1 = font_size + 6
    line_h2 = fs2 + 6
    total_h = line_h1 + (line_h2 if line2 else 0)
    margin = 8
    y_above = y1 - margin - total_h
    if y_above >= 4:
        y0 = y_above
    else:
        y0 = min(max(2, h_img - total_h - 2), y2 + margin)
    _draw_chinese_text_bgr(vis, line1, (x1, y0), font_size=font_size, color=color)
    if line2:
        _draw_chinese_text_bgr(
            vis,
            line2,
            (x1, y0 + line_h1),
            font_size=fs2,
            color=(180, 180, 180),
        )


def process_image(
    ocr_engine: Any,
    image_path: Path,
    centers: Sequence[CenterRoi],
    rects: Sequence[RectRoi],
    font_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    h_img, w_img = bgr.shape[:2]
    vis = bgr.copy()
    bond_roi_y_shift = BOND_ROI_AUXILIARY_DY_PX if _stem_uses_auxiliary_bond_roi_shift(image_path.stem) else 0
    summary: Dict[str, Any] = {
        "file": image_path.name,
        "bond_roi_y_shift_px": bond_roi_y_shift,
        "fields": {},
    }

    colors = [
        (0, 220, 255),
        (0, 255, 128),
        (255, 180, 0),
        (255, 100, 200),
        (180, 120, 255),
        (100, 200, 255),
        (255, 255, 100),
    ]
    ci = 0

    for c in centers:
        x1, y1, x2, y2 = c.to_rect(w_img, h_img)
        crop = bgr[y1:y2, x1:x2]
        tight: Optional[Tuple[int, int, int, int]] = None
        if c.key == "gold":
            lines, tight = _ocr_gold_field(ocr_engine, crop)
        elif c.key == "streak":
            lines = _ocr_streak_field(ocr_engine, crop)
        else:
            lines = _ocr_one(ocr_engine, crop)
        raw = _join_lines(lines)
        if c.key == "streak":
            parsed = _parse_streak(raw, crop)
        else:
            parsed = parse_field(c.key, raw)
        summary["fields"][c.key] = {"name": c.name, "raw": raw, "parsed": parsed}
        col = colors[ci % len(colors)]
        ci += 1
        vx1, vy1, vx2, vy2 = x1, y1, x2, y2
        if c.key == "gold" and tight is not None:
            tx1, ty1, tx2, ty2 = tight
            vx1, vy1 = x1 + tx1, y1 + ty1
            vx2, vy2 = x1 + tx2, y1 + ty2
        _draw_roi_rect(vis, vx1, vy1, vx2, vy2, col, 2)
        _annotate_outer(
            vis,
            vx1,
            vy1,
            vx2,
            vy2,
            c.name,
            parsed,
            raw,
            col,
            font_size,
            h_img=h_img,
            label_parsed_only=(c.key == "streak"),
        )

    for r in rects:
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(y1 + 1, min(y2, h_img))
        col = colors[ci % len(colors)]
        ci += 1
        if r.key == "bonds" and r.grid_cols > 0 and r.grid_rows > 0:
            use_dynamic_bond_roi = os.environ.get("BATTLE_UI_BONDS_DYNAMIC_ROI", "1").strip() != "0"
            if use_dynamic_bond_roi:
                (x1, y1, x2, y2), pick_meta = _pick_bonds_roi(
                    ocr_engine, bgr, w_img, h_img, dy=bond_roi_y_shift
                )
            else:
                y1_d, y2_d = y1 + int(bond_roi_y_shift), y2 + int(bond_roi_y_shift)
                x1 = max(0, min(x1, w_img - 1))
                x2 = max(x1 + 1, min(x2, w_img))
                y1 = max(0, min(y1_d, h_img - 1))
                y2 = max(y1 + 1, min(y2_d, h_img))
                pick_meta = {
                    "mode": "fixed",
                    "selected": {"rect": (x1, y1, x2, y2), "score": -1, "chars": -1},
                    "candidates": [],
                }
            _draw_roi_rect(vis, x1, y1, x2, y2, col, 3)
            summary["fields"][r.key] = _process_bonds_grid(
                vis,
                bgr,
                x1,
                y1,
                x2,
                y2,
                ocr_engine,
                col,
                font_size,
                cols=r.grid_cols,
                rows=r.grid_rows,
                h_img=h_img,
            )
            summary["fields"][r.key]["roi_selected"] = [x1, y1, x2, y2]
            summary["fields"][r.key]["roi_pick"] = pick_meta
            _annotate_outer(
                vis,
                x1,
                y1,
                x2,
                y2,
                "羁绊栏(动态)",
                f"score={int(pick_meta['selected']['score'])}",
                "",
                col,
                font_size,
                h_img=h_img,
            )
        elif r.key == "hp_nick":
            summary["fields"][r.key] = _process_hp_column(
                vis,
                bgr,
                x1,
                y1,
                x2,
                y2,
                ocr_engine,
                col,
                font_size,
                h_img=h_img,
            )
        else:
            _draw_roi_rect(vis, x1, y1, x2, y2, col, 2)
            crop = bgr[y1:y2, x1:x2]
            lines = _ocr_one(ocr_engine, crop)
            raw = _join_lines(lines)
            parsed = parse_field(r.key, raw)
            summary["fields"][r.key] = {"name": r.name, "raw": raw, "parsed": parsed}
            _annotate_outer(
                vis,
                x1,
                y1,
                x2,
                y2,
                r.name,
                parsed,
                raw,
                col,
                font_size,
                h_img=h_img,
            )

    return vis, summary


def _print_bonds_ocr_report(summary: Dict[str, Any]) -> None:
    """在终端打印羁绊 ROI 内各分块识别到的原始字段与解析结果。"""
    fields = summary.get("fields") or {}
    b = fields.get("bonds")
    if not isinstance(b, dict):
        print("  [羁绊区] 无数据")
        return
    print("  —— 羁绊区 OCR 字段（Paddle / 分块）——")
    raw_all = (b.get("raw") or "").strip()
    if raw_all:
        print(f"  全 ROI 合并字符串: {raw_all}")
    cells = b.get("bond_cells") or []
    if not cells:
        print("  （无分块条目）")
    else:
        for cell in cells:
            idx = cell.get("index", "?")
            ocr_rows = (cell.get("ocr_rows") or "").strip()
            raw_cell = (cell.get("raw") or "").strip()
            piece = cell.get("piece")
            piece_s = piece if piece else "（未解析为已激活）"
            print(f"  第{idx}行 | 单条合并: {raw_cell}")
            if ocr_rows and ocr_rows != raw_cell:
                print(f"    行内分框: {ocr_rows}")
            print(f"    解析片段: {piece_s}")
    summ = (b.get("bond_summary") or b.get("parsed") or "").strip()
    if summ:
        print(f"  汇总: {summ}")
    print("  —— 羁绊区 结束 ——")


def main() -> None:
    _ensure_running_in_ocr_venv()
    ap = argparse.ArgumentParser(description="player_recog：对局 UI 固定 ROI 识别标注")
    ap.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"对局截图目录或单张 PNG（默认: {DEFAULT_INPUT}）",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUT),
        help=f"输出目录（默认: {DEFAULT_OUT}）",
    )
    ap.add_argument("--font-size", type=int, default=18, help="标注主文字大小")
    ap.add_argument("--json", action="store_true", help="同目录写出每图对应的 summary.json")
    ap.add_argument("--no-clear", action="store_true", help="不清空输出目录（追加写入）")
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="打印 PaddleOCR（ppocr）详细日志；默认只显示 ERROR，减少刷屏",
    )
    ap.add_argument(
        "--print-bonds",
        dest="print_bonds",
        action="store_true",
        default=False,
        help="在终端打印羁绊区各分块 OCR 原始字段",
    )
    ap.add_argument(
        "--no-print-bonds",
        dest="print_bonds",
        action="store_false",
        help="不打印羁绊区字段",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    if not args.no_clear:
        _clear_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    images = _iter_pngs(in_path)
    centers, rects = _default_layout()

    os.environ["BATTLE_UI_OCR_ENGINE"] = "paddle"
    _set_ppocr_log_level(verbose=bool(args.verbose))
    print(
        "正在加载 PaddleOCR（权重缓存在用户目录 .paddleocr，仅首次需联网下载；"
        "每次运行仍会加载模型到内存）…"
    )
    ocr_engine = _create_ocr_engine(verbose=bool(args.verbose))

    for img_path in images:
        vis, summary = process_image(ocr_engine, img_path, centers, rects, args.font_size)
        stem = img_path.stem
        out_png = out_dir / f"{stem}_player_recog_annotated.png"
        ok, buf = cv2.imencode(".png", vis)
        if not ok:
            raise RuntimeError(f"imencode 失败: {out_png}")
        out_png.write_bytes(buf.tobytes())
        print(f"[OK] {img_path.name} -> {out_png.name}")
        if args.print_bonds:
            _print_bonds_ocr_report(summary)
        if args.json:
            jpath = out_dir / f"{stem}_player_recog_summary.json"
            jpath.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"完成。输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
