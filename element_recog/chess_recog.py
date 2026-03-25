# -*- coding: utf-8 -*-
"""
chess_recog：血条模板匹配 + 圆形采样 DINOv2/ResNet embedding 投票；供 bars_recog / equip_recog 使用。

**命令行入口**：文件末尾 ``main()`` —— 批量处理 ``对局截图/*.png``，输出至 ``chess_recog/{图名}/``
（带棋子标注图、result.json、根目录 summary.json），与原先 ``batch_recog_d84.py`` 一致。

事先构建 embedding 缓存：``python chess_recog.py --build-embed-cache-only``（写入 ``.piece_db_cache/``）；
``run_recognition`` 在未传入 ``piece_db`` 时也会自动从该缓存加载，无需每次全量建库。
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 中文标注字体（Windows 常见）
FONT_PATHS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


def load_healthbar_templates(path: Path) -> List[np.ndarray]:
    """
    读取血条模板：
    - 若 path 是文件：读一张
    - 若 path 是目录：读取目录下所有 png/jpg/jpeg/webp，并按文件名排序
    返回 [BGR, ...]，为空则抛错。
    """
    path = Path(path)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files: List[Path] = []
    if path.is_dir():
        # 递归读取：确保像 v2~v5 这种新模板即使放在子目录里也能被识别
        # 注意：只匹配图片后缀，且按相对路径排序保证稳定性
        files = sorted(
            [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: str(p.relative_to(path)),
        )
    else:
        files = [path]
    templates: List[np.ndarray] = []
    for p in files:
        if not p.is_file():
            continue
        try:
            templates.append(_load_image(p))
        except Exception:
            continue
    if not templates:
        raise FileNotFoundError(f"未找到可用血条模板: {path}")
    return templates


def _load_image(path: Path, allow_alpha: bool = False) -> np.ndarray:
    """支持中文路径的读图。allow_alpha=True 时若为 RGBA 则叠白底转 BGR。"""
    raw = np.fromfile(str(path), dtype=np.uint8)
    if allow_alpha:
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    # OpenCV: 彩色图为 (H,W,C)，BGRA 也是 ndim==3 且 C==4
    if allow_alpha and img.ndim == 3 and img.shape[-1] == 4:
        # BGRA -> 白底 BGR
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white = np.ones_like(bgr) * 255.0
        img = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
    return img


def _save_image(img: np.ndarray, path: Path) -> None:
    """支持中文路径的写图（Windows 上勿用 cv2.imwrite / np.tofile(str)，易乱码）"""
    ok, buf = cv2.imencode(".png", img)
    if not ok or buf is None:
        raise RuntimeError(f"无法编码图像: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_threshold: float = 0.3,
    center_dist_px: Optional[float] = None,
    area_ratio_max: float = 4.0,
) -> List[Tuple[int, int, int, int]]:
    """按得分排序后做 NMS，去掉重叠过大的框"""
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    # 按得分从高到低
    order = np.argsort(-scores)
    keep: List[int] = []

    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = np.maximum(a[0], b[0])
        y1 = np.maximum(a[1], b[1])
        x2 = np.minimum(a[0] + a[2], b[0] + b[2])
        y2 = np.minimum(a[1] + a[3], b[1] + b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]
        return inter / (area_a + area_b - inter + 1e-6)

    for i in order:
        ok = True
        for j in keep:
            iou = _box_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                ok = False
                break

            # 额外抑制“多尺度残留”的近似重复框：
            # 当中心点非常接近时（即使 IoU 因尺寸差异偏低），也可以合并。
            if center_dist_px is not None and center_dist_px > 0:
                wi, hi = boxes[i][2], boxes[i][3]
                wj, hj = boxes[j][2], boxes[j][3]
                # 面积比例过大（差太多尺寸）时不强行合并，避免误把相邻血条当成同一个
                area_i = max(1.0, wi * hi)
                area_j = max(1.0, wj * hj)
                ratio = max(area_i / area_j, area_j / area_i)
                if ratio <= area_ratio_max:
                    cxi = boxes[i][0] + wi / 2.0
                    cyi = boxes[i][1] + hi / 2.0
                    cxj = boxes[j][0] + wj / 2.0
                    cyj = boxes[j][1] + hj / 2.0
                    dx = float(cxi - cxj)
                    dy = float(cyi - cyj)
                    if dx * dx + dy * dy <= float(center_dist_px * center_dist_px):
                        ok = False
                        break
        if ok:
            keep.append(i)

    return [tuple(map(int, boxes[i])) for i in keep]


def detect_healthbars_by_templates(
    scene: np.ndarray,
    templates: List[np.ndarray],
    *,
    scales: List[float] = (0.6, 0.8, 1.0, 1.2, 1.4),
    scales_w: Optional[List[float]] = None,
    scales_h: Optional[List[float]] = None,
    threshold: float = 0.65,
    nms_iou: float = 0.35,
    nms_center_dist_px: Optional[float] = 28.0,
) -> List[Tuple[int, int, int, int]]:
    """
    多模板版本：对每个 template 做多尺度匹配后汇总，再统一 NMS。
    适用于：一星/二星/三星血条模板同时存在的场景。
    """
    if not templates:
        return []
    H, W = scene.shape[:2]
    all_boxes: List[Tuple[int, int, int, int]] = []
    all_scores: List[float] = []

    # 默认：等比例缩放宽高
    # 进阶：允许分别缩放宽高，提高“更长/更密血条”的召回
    if scales_w is None and scales_h is None:
        pairs = [(s, s) for s in scales]
    else:
        if scales_w is None:
            scales_w = list(scales)
        if scales_h is None:
            scales_h = list(scales)
        pairs = [(sw, sh) for sw in scales_w for sh in scales_h]
    for template in templates:
        th, tw = template.shape[:2]
        if th < 2 or tw < 2:
            continue
        for sw, sh in pairs:
            w = max(4, int(tw * sw))
            h = max(2, int(th * sh))
            if w > W or h > H:
                continue
            resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_LINEAR)
            result = cv2.matchTemplate(scene, resized, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= threshold)
            for x, y in zip(xs, ys):
                all_boxes.append((int(x), int(y), w, h))
                all_scores.append(float(result[y, x]))
    if not all_boxes:
        return []
    return _nms_boxes(
        all_boxes,
        all_scores,
        iou_threshold=nms_iou,
        center_dist_px=nms_center_dist_px,
    )


# ---------- 深度特征（AI 嵌入）匹配：需安装 torch + torchvision ----------
# DINOv2 首次会从 torch.hub 拉取 facebookresearch/dinov2，需联网
_EMBED_BACKBONE_ALIASES = {
    "dinov2": "dinov2_vits14",
    "dinov2_s": "dinov2_vits14",
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_b": "dinov2_vitb14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_l": "dinov2_vitl14",
    "dinov2_vitl14": "dinov2_vitl14",
    "dinov2_g": "dinov2_vitg14",
    "dinov2_vitg14": "dinov2_vitg14",
}


def _resolve_embed_backbone(name: str) -> str:
    k = (name or "dinov2_vits14").strip().lower().replace("-", "_")
    return _EMBED_BACKBONE_ALIASES.get(k, k)


def _get_embedding_model(backbone: str = "dinov2_vits14"):
    """
    返回 (model, device, transform)。
    model 输入 (N,3,224,224)，输出全局向量（维度因骨干而异），调用方再做 L2 归一化。

    backbone（默认 DINOv2-ViT-S/14，自监督、对小目标/纹理更稳）:
      - dinov2_vits14 / dinov2_s / dinov2
      - dinov2_vitb14 / dinov2_b
      - dinov2_vitl14 / dinov2_l
      - dinov2_vitg14 / dinov2_g
    """
    try:
        import torch
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    except ImportError as e:
        raise RuntimeError(
            "使用 embedding 需安装: pip install torch torchvision\n" + str(e)
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    resolved = _resolve_embed_backbone(backbone)
    # DINOv2：取 CLS token 作为图像嵌入（与检索/细粒度匹配常用做法一致）
    if not resolved.startswith("dinov2_"):
        raise RuntimeError(
            f"未知 embed backbone: {backbone!r}（可用 dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14）"
        )
    try:
        raw = torch.hub.load("facebookresearch/dinov2", resolved, trust_repo=True)
    except Exception as e:
        raise RuntimeError(
            f"加载 DINOv2 ({resolved}) 失败（首次需联网下载 hub）。"
            f"详情: {e}"
        ) from e
    raw.to(device)
    raw.eval()

    class _DinoV2Embed(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.net.forward_features(x)
            if isinstance(out, dict):
                return out["x_norm_clstoken"]
            return out[:, 0] if out.dim() == 3 else out

    model = _DinoV2Embed(raw).to(device)
    model.eval()
    return model, device, transform


def _draw_chinese_text(
    bgr: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    font_size: int = 18,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """在 BGR 图上用支持中文的字体绘制文字（原地修改）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        cv2.putText(bgr, "?", xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return
    font = None
    for fp in FONT_PATHS:
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
    # PIL 颜色 (R,G,B)
    pil_color = (color[2], color[1], color[0])
    draw.text(xy, text, font=font, fill=pil_color)
    out_rgb = np.array(pil)
    bgr_out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    bgr[:] = bgr_out


# ========== merged from bars_vote ==========

# ROI：只在此区域内找血条（你给的 4 点矩形）
# 你反馈：board_fight 范围太小 -> 往上 +400、往右 +100
# 由于 y 轴向下为正，所以“往上 +400”对应 y1 - 400，并在运行时 clamp 到 >=0。
# board_fight 的矩形顶部边缘“下降 100 像素”（y1 增加 100），而不是改底部。
ROI = (600, 100, 1720, 720)  # x1,y1,x2,y2

# 与 `batch_recog_d84.py` 默认 `--out` 根目录名一致。
# 棋盘/裁切类脚本应优先用 `default_batch_result_json` + `load_bar_boxes_from_result_json` 读取与 batch 完全一致的 bar_box；
# 无 JSON 时用 `_detect_healthbars_in_roi`（默认会去掉 ROI 顶带小龙式细条，与识别管线前置一致）。
DEFAULT_BATCH_EMBED_OUT_DIRNAME = "chess_recog"

# 顶部「碎片框 → 整段血条」时，水平位置用碎片几何中心对齐整条血条宽度。
# 纳什男爵等 Boss 顶栏：模板常命中略偏左的装饰/小条，相对真实血条中心会整体偏左约 5~10px，
# 此处对**仅经 snap 修正的顶栏框**做右移补偿（可调）。
TOP_BAR_SNAP_NUDGE_X_PX = 8

BELOW_BAR_PX = 60
CENTER_OFFSET_X = 10  # 与你当前圆心偏移一致

# 多点采样：以血条下边界为基准的 (dx, dy)
DEFAULT_SAMPLES: List[Tuple[int, int]] = [
    (0, 40),
    (0, 60),
    (0, 80),
    (-14, 60),
    (14, 60),
]

# 顶部高区（全图坐标）语义限制：仅允许这些标签作为棋子条输出。
TOP_ZONE_MAX_Y_ABS = 250
TOP_ZONE_ALLOWED_LABELS = {"纳什男爵", "海克斯霸龙"}

# 默认棋盘参数（用于在 chess_recog 内直接输出格子位置）
_BOARD_COORD_SHIFT_Y_PX = -10
_BOARD_ROW_X1_X7_Y: List[Tuple[int, int, int]] = [
    (750, 1450, 450 + _BOARD_COORD_SHIFT_Y_PX),
    (800, 1500, 550 + _BOARD_COORD_SHIFT_Y_PX),
    (730, 1470, 650 + _BOARD_COORD_SHIFT_Y_PX),
    (770, 1550, 750 + _BOARD_COORD_SHIFT_Y_PX),
]
_ROW12_BOUNDARY_BIAS_PX = 48.0
_ROW4_BAR_TOP_MIN = 592.0
_ROW4_PROMOTE_BAR_BOTTOM_MIN = 600.0
_MAX_CELL_DIST = 135.0
_FOOT_DY = 48
_CELL_SIZE = 150
_MAX_DX_COL = max(70.0, float(_CELL_SIZE) * 0.52)
_TALL_BAR_BOOST = 0.52
_TALL_BAR_BOOST_MAX = 80.0
_TALL_BAR_BOOST_MIN_BARS = 3


def _board_cells() -> List[Tuple[int, int, float, float]]:
    out: List[Tuple[int, int, float, float]] = []
    for ri, (x1, x7, y) in enumerate(_BOARD_ROW_X1_X7_Y, start=1):
        for ci in range(7):
            t = ci / 6.0
            cx = float(x1) + (float(x7) - float(x1)) * t
            out.append((ri, ci + 1, cx, float(y)))
    return out


def _row_ys() -> List[float]:
    return [float(y) for _x1, _x7, y in _BOARD_ROW_X1_X7_Y]


def _infer_row_from_anchor_y(py: float, row_ys: Sequence[float], y_bias: float = 0.0) -> int:
    ys = [float(y) + float(y_bias) for y in row_ys]
    for i in range(len(ys) - 1):
        mid = (ys[i] + ys[i + 1]) / 2.0
        if i == 0:
            mid += float(_ROW12_BOUNDARY_BIAS_PX)
        if py < mid:
            return i + 1
    return len(ys)


def _infer_row_for_mapping(anchor_py: float, bar_top_y: float, bar_bottom_y: float, row_ys: Sequence[float]) -> int:
    if float(bar_top_y) >= float(_ROW4_BAR_TOP_MIN):
        return 4
    base = _infer_row_from_anchor_y(anchor_py, row_ys, y_bias=0.0)
    if base == 3 and float(bar_bottom_y) >= float(_ROW4_PROMOTE_BAR_BOTTOM_MIN):
        return 4
    return base


def _nearest_cell_euclidean(
    px: float,
    py: float,
    cells: Sequence[Tuple[int, int, float, float]],
    max_dist: float,
) -> Tuple[Optional[Tuple[int, int, float, float]], float]:
    best: Optional[Tuple[int, int, float, float]] = None
    best_d2 = float("inf")
    for c in cells:
        _r, _col, cx, cy = c
        dx = px - cx
        dy = py - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = c
    d = float(np.sqrt(best_d2)) if best is not None else float("inf")
    if best is None or d > float(max_dist):
        return None, d
    return best, d


def _nearest_cell_row_first(
    px: float,
    py: float,
    cells: Sequence[Tuple[int, int, float, float]],
    row_ys: Sequence[float],
    bar_top_y: float,
    bar_bottom_y: float,
) -> Tuple[Optional[Tuple[int, int, float, float]], float]:
    r = _infer_row_for_mapping(py, bar_top_y, bar_bottom_y, row_ys)
    same_row = [c for c in cells if c[0] == r]
    if not same_row:
        return _nearest_cell_euclidean(px, py, cells, _MAX_CELL_DIST)
    best = min(same_row, key=lambda c: abs(px - c[2]))
    dx_col = abs(px - best[2])
    if dx_col <= float(_MAX_DX_COL):
        dist = float(np.hypot(px - best[2], py - best[3]))
        return best, dist
    return _nearest_cell_euclidean(px, py, cells, _MAX_CELL_DIST)


def _tall_bar_extra_foot_dy(by: int, bh: int, bar_bottoms: Sequence[int]) -> float:
    if len(bar_bottoms) < int(_TALL_BAR_BOOST_MIN_BARS) or float(_TALL_BAR_BOOST) <= 0.0:
        return 0.0
    med = float(np.median(np.asarray(bar_bottoms, dtype=np.float64)))
    raw = med - float(by + bh)
    if raw <= 0.0:
        return 0.0
    return float(min(float(_TALL_BAR_BOOST_MAX), raw * float(_TALL_BAR_BOOST)))

# 二阶段“加密采样”：只在第一阶段不确定(best=null)时启用
EXTRA_SAMPLES: List[Tuple[int, int]] = [
    (0, 30),
    (0, 50),
    (0, 70),
    (0, 90),
    # 更靠近头像中心（对 慎/格雷福斯、妮蔻/梅尔/米利欧 这类相似脸更容易拉开）
    (0, 100),
    (0, 110),
    (-22, 50),
    (22, 50),
    (-22, 70),
    (22, 70),
    (-22, 90),
    (22, 90),
    (-22, 110),
    (22, 110),
    (-8, 60),
    (8, 60),
]


def _hero_name(label: Optional[str]) -> Optional[str]:
    """
    特征库图片名形如：英雄__s01_0_40
    识别/投票/聚合时按“英雄”合并，避免同一英雄的不同采样变体互相分票导致 best=None。
    """
    if not label:
        return None
    s = str(label)
    return s.split("__", 1)[0] if "__" in s else s


def _collapse_topk_to_hero(top: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """把 topK (raw_name, sim) 合并到 (hero, max_sim)。"""
    best: Dict[str, float] = {}
    for raw, sim in top:
        h = _hero_name(raw)
        if not h:
            continue
        prev = best.get(h)
        if prev is None or float(sim) > float(prev):
            best[h] = float(sim)
    out = sorted(best.items(), key=lambda x: -x[1])
    return out


def _dedup_boxes_by_center(
    boxes: List[Tuple[int, int, int, int]],
    dist_thresh: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    二次去重：对多模板/多尺度匹配残留的“近似重复框”做中心点距离聚类。
    """
    if not boxes:
        return []
    dist2 = float(dist_thresh * dist_thresh)
    kept: List[Tuple[int, int, int, int]] = []
    centers: List[Tuple[float, float]] = []
    # 面积大的优先（同一血条的重复框通常面积更接近）
    for (x, y, w, h) in sorted(boxes, key=lambda b: -(b[2] * b[3])):
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        ok = True
        for (kx, ky) in centers:
            dx = cx - kx
            dy = cy - ky
            if dx * dx + dy * dy <= dist2:
                ok = False
                break
        if ok:
            kept.append((x, y, w, h))
            centers.append((cx, cy))
    return kept


def _xywh_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    bx, by, bw, bh = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    ua = float(max(1, aw * ah) + max(1, bw * bh) - inter)
    return inter / ua


def _xywh_inter_over_min_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    bx, by, bw, bh = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    aa = float(max(1, aw * ah))
    ab = float(max(1, bw * bh))
    return inter / min(aa, ab)


def _near_duplicate_healthbars_row(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    *,
    center_dist2_max: float,
    hor_ov_min: float,
    vert_rel_max: float,
) -> bool:
    """同一行上、中心很近且水平投影重叠大：多为同一血条的模板重复命中。"""
    ax, ay, aw, ah = (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    bx, by, bw, bh = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    if min(aw, ah, bw, bh) < 6:
        return False
    cax, cay = ax + aw / 2.0, ay + ah / 2.0
    cbx, cby = bx + bw / 2.0, by + bh / 2.0
    if abs(cay - cby) > vert_rel_max * max(float(ah), float(bh)):
        return False
    dx, dy = cax - cbx, cay - cby
    if dx * dx + dy * dy > center_dist2_max:
        return False
    awf, bwf = float(aw), float(bw)
    ax2, bx2 = ax + awf, bx + bwf
    il, ir = max(ax, bx), min(ax2, bx2)
    inter = max(0.0, ir - il)
    hor_ov = inter / max(1e-6, min(awf, bwf))
    return hor_ov >= hor_ov_min


def _merge_overlapping_bar_boxes(
    boxes: List[Tuple[int, int, int, int]],
    *,
    iou_merge: float = 0.12,
    iomin_merge: float = 0.28,
    center_dist_px: float = 36.0,
    hor_ov_min: float = 0.52,
    vert_rel_max: float = 0.62,
) -> List[Tuple[int, int, int, int]]:
    """
    将明显重叠/套叠的重复血条框合并为一条（保留面积最大的框）。
    解决 fightboard 上同一英雄出现两个高度重叠检测框的问题。
    """
    if len(boxes) <= 1:
        return list(boxes)

    n = len(boxes)
    parent = list(range(n))

    def _root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _root(i), _root(j)
        if ri != rj:
            parent[rj] = ri

    dist2 = float(center_dist_px * center_dist_px)
    for i in range(n):
        for j in range(i + 1, n):
            io = _xywh_iou(boxes[i], boxes[j])
            im = _xywh_inter_over_min_area(boxes[i], boxes[j])
            near = _near_duplicate_healthbars_row(
                boxes[i],
                boxes[j],
                center_dist2_max=dist2,
                hor_ov_min=hor_ov_min,
                vert_rel_max=vert_rel_max,
            )
            if io >= iou_merge or im >= iomin_merge or near:
                _union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = _root(i)
        groups.setdefault(r, []).append(i)

    merged: List[Tuple[int, int, int, int]] = []
    for _rid, idxs in groups.items():
        best_i = max(idxs, key=lambda k: int(boxes[k][2]) * int(boxes[k][3]))
        merged.append(tuple(map(int, boxes[best_i])))
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


def _build_roi_tiles_adaptive(
    cw: int,
    ch: int,
    ref_bar_w: int,
    ref_bar_h: int,
) -> List[Tuple[int, int, int, int]]:
    """
    自适应分块：
    - 根据“参考血条尺寸”动态决定块宽/重叠，避免固定 3x2 在不同画面密度下不贴合。
    """
    bw = max(40, int(ref_bar_w))
    bh = max(8, int(ref_bar_h))
    # 每块大约覆盖 3~5 个血条宽度；在大 ROI 会自动增加列数
    tile_w = int(max(280, min(520, bw * 4)))
    tile_h = int(max(180, min(360, bh * 14)))
    overlap_x = int(max(48, min(140, bw)))
    overlap_y = int(max(28, min(90, bh * 3)))
    step_x = max(64, tile_w - overlap_x)
    step_y = max(64, tile_h - overlap_y)
    x_starts = list(range(0, max(1, cw - tile_w + 1), step_x))
    y_starts = list(range(0, max(1, ch - tile_h + 1), step_y))
    if not x_starts:
        x_starts = [0]
    if not y_starts:
        y_starts = [0]
    last_x = max(0, cw - tile_w)
    last_y = max(0, ch - tile_h)
    if x_starts[-1] != last_x:
        x_starts.append(last_x)
    if y_starts[-1] != last_y:
        y_starts.append(last_y)

    tiles: List[Tuple[int, int, int, int]] = []
    for ty in y_starts:
        for tx in x_starts:
            tx2 = min(cw, tx + tile_w)
            ty2 = min(ch, ty + tile_h)
            tiles.append((int(tx), int(ty), int(tx2 - tx), int(ty2 - ty)))
    return tiles


def _estimate_ref_bar_size(templates: List[np.ndarray]) -> Tuple[int, int]:
    """从模板估计参考血条尺寸（取中位数，抗异常模板）。"""
    if not templates:
        return (109, 18)
    ws = sorted([int(t.shape[1]) for t in templates if t is not None and t.ndim >= 2])
    hs = sorted([int(t.shape[0]) for t in templates if t is not None and t.ndim >= 2])
    if not ws or not hs:
        return (109, 18)
    mid = len(ws) // 2
    return (int(ws[mid]), int(hs[mid]))


def _snap_top_bar_fragment_to_full_bar(
    box: Tuple[int, int, int, int],
    ref_w: int,
    ref_h: int,
    cw: int,
    ch: int,
    *,
    top_band_y: int = 90,
    nudge_x: int = TOP_BAR_SNAP_NUDGE_X_PX,
) -> Tuple[int, int, int, int]:
    """
    顶部区域（如 11.png 纳什男爵）模板匹配常会命中「血条上方装饰/小条」而非整条血条，
    得到又扁又窄的框。若判定为碎片，则在 ROI 局部坐标下改为「装饰下方 + 参考宽高」的整段血条。

    水平方向：默认以碎片中心对齐整条血条；Boss 顶栏碎片相对真实血条常略偏左，故用 nudge_x 右移补偿。
    """
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if y > top_band_y:
        return box
    # 已是常见整段血条尺寸，不改动（与 filt_main 中 is_normal 一致）
    if w >= 80 and h >= 13:
        return box
    w_bar = int(max(90, min(ref_w, 120)))
    h_bar = int(max(14, min(ref_h, 22)))
    cx = x + w / 2.0
    x_new = int(round(cx - w_bar / 2.0)) + int(nudge_x)
    x_new = max(0, min(x_new, max(0, cw - w_bar)))
    # 碎片在整条血条上方：新框顶边 = 碎片底边（血条从这里开始）
    y_new = y + h
    if y_new + h_bar > ch:
        y_new = max(0, ch - h_bar)
    if y_new < 0:
        y_new = 0
    return (x_new, y_new, w_bar, h_bar)


def is_wild_dragon_strip_geometry(
    *,
    y_abs: int,
    w: int,
    h: int,
    roi_y1: int,
    band_y: int = 90,
) -> bool:
    """
    仅剔除「典型野怪小龙」顶条（约 52×8～54×9），与纳什 Boss 顶条碎片区分。

    注意：纳什碎片常见 ~59×11、面积 600+，若用「w≤60 且 h≤11 且面积≤720」会误伤并在 snap 前删掉，
    导致男爵条只能靠其它检测路径补回甚至丢失。故这里**收紧**为偏小的宽/高/面积上限；
    略大的顶条一律保留，走 `_snap_top_bar_fragment_to_full_bar`。
    """
    if int(y_abs) > int(roi_y1) + int(band_y):
        return False
    iw, ih = int(w), int(h)
    # 已是常见整段血条尺寸 → 不是「仅小龙细条」语义
    if iw >= 80 and ih >= 13:
        return False
    # 典型小龙：宽度不超过 54、高度不超过 9（纳什碎片往往更宽或更高）
    if iw <= 54 and ih <= 9:
        return True
    # 略扁但仍偏窄：限制宽≤54 且面积，避免 60×11 级纳什碎片命中
    if iw <= 54 and ih <= 10 and (iw * ih) <= 540:
        return True
    # 极少数略宽野怪条（仍明显窄于 60）：面积上限收紧，60×11=660 不会命中
    if iw <= 56 and ih <= 10 and (iw * ih) <= 560:
        return True
    return False


def load_bar_boxes_from_result_json(path: Path) -> List[Tuple[int, int, int, int]]:
    """读取 `run_recognition` 写入的 result.json 中每条 `bar_box`（与 batch 最终输出顺序一致）。"""
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[int, int, int, int]] = []
    for it in obj.get("results") or []:
        bb = it.get("bar_box")
        if not bb or len(bb) != 4:
            continue
        out.append((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))
    return out


def default_batch_result_json(
    project_dir: Path,
    image_path: Path,
    *,
    embed_dirname: str = DEFAULT_BATCH_EMBED_OUT_DIRNAME,
) -> Path:
    """默认 `{project_dir}/{embed_dirname}/{图 stem}/result.json`，与 chess_recog 批量输出 / equip_recog 约定一致。"""
    return (project_dir / embed_dirname / image_path.stem / "result.json").resolve()


def _compute_confidence(stage: str, used_samples: int, best_score: float,
                        stage2_detail) -> str:
    """
    根据识别路径与证据强度评定置信度：
      high   - stage1 通过且有效采样 >= 2；或 stage2 非填充且有效采样 >= 3
      medium - stage1 used=1；stage2 非填充 used>=1；stage2_retry 补全 support>=2
      low    - consensus_top1 兜底（无 sim 约束，最不可信）
    在标记图上 low → 黄色文字 + "?" 后缀，提示结果存疑。
    """
    s2 = stage2_detail or {}
    filled = bool(s2.get("filled_by_consensus", False))
    method = (s2.get("filled_reason") or {}).get("method", "")
    support = int((s2.get("filled_reason") or {}).get("support", 0))

    if not filled:
        # 正常识别路径（stage1 / stage2 均有真实采样通过）
        if stage == "stage1":
            return "high" if int(used_samples) >= 2 else "medium"
        else:
            # stage2 有真实采样通过（min_sim=0.74 门槛下）
            return "high" if int(used_samples) >= 3 else "medium"
    else:
        # consensus 补全路径
        if method == "stage2_retry_min_sim" and support >= 2:
            # 宽松 sim=0.58 门槛下 >=2 个默认采样点一致 → medium（存疑但有相似度约束）
            return "medium"
        else:
            # consensus_top1（纯投票，无 sim 约束）→ 最不可信，标 low
            return "low"



def _detect_healthbars_in_roi(
    scene: np.ndarray,
    templates: List[np.ndarray],
    roi: Tuple[int, int, int, int],
) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    H, W = scene.shape[:2]
    # clamp 到图像范围，避免 y1 变负导致切片语义错误
    x1 = max(0, min(int(x1), W))
    x2 = max(0, min(int(x2), W))
    y1 = max(0, min(int(y1), H))
    y2 = max(0, min(int(y2), H))
    crop = scene[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    ch, cw = crop.shape[:2]
    # 分块检测（带重叠）：
    # 在密集血条场景下，局部检测能减少相邻血条互相干扰/抑制。
    ref_w, ref_h = _estimate_ref_bar_size(templates)
    tiles = _build_roi_tiles_adaptive(cw, ch, ref_w, ref_h)

    all_boxes: List[Tuple[int, int, int, int]] = []
    for tx, ty, tw, th in tiles:
        sub = crop[ty : ty + th, tx : tx + tw]
        if sub.size == 0:
            continue
        # 分块内保持原始保守参数，避免误检上升
        sub_boxes = detect_healthbars_by_templates(sub, templates, threshold=0.60, nms_iou=0.50)
        for (bx, by, bw, bh) in sub_boxes:
            all_boxes.append((int(bx + tx), int(by + ty), int(bw), int(bh)))

    # 极端情况下回退到整图检测（防止分块参数不适配导致空检）
    if not all_boxes:
        all_boxes = detect_healthbars_by_templates(crop, templates, threshold=0.60, nms_iou=0.50)

    boxes_main = _dedup_boxes_by_center(all_boxes, dist_thresh=14)
    # 过滤中下区域的极小伪框（例如 65x10），减少重复识别；
    # 顶部保留小框，兼容纳什男爵等高位小血条。
    top_small_band_y = 90
    filt_main: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes_main:
        is_normal = (int(w) >= 80 and int(h) >= 13)
        is_top_small = (int(y) <= top_small_band_y and int(w) >= 50 and int(h) >= 8)
        if is_normal or is_top_small:
            filt_main.append((int(x), int(y), int(w), int(h)))
    boxes_main = _dedup_boxes_by_center(filt_main, dist_thresh=14)

    # 二次补检（低阈值）：
    # 仅吸收“形状像血条”的框，并且要求与主检测不重合，避免误检明显上升。
    lo_boxes = detect_healthbars_by_templates(crop, templates, threshold=0.56, nms_iou=0.50)
    added: List[Tuple[int, int, int, int]] = []
    # 经验形状约束：适配当前项目常见的 109x18 / 87x14 血条簇
    min_w, max_w = 80, 116
    min_h, max_h = 13, 20
    dist2_gate = float(24 * 24)
    for (x, y, w, h) in lo_boxes:
        if not (min_w <= int(w) <= max_w and min_h <= int(h) <= max_h):
            continue
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        too_close = False
        for (bx, by, bw, bh) in boxes_main:
            kx = float(bx + bw / 2.0)
            ky = float(by + bh / 2.0)
            dx = cx - kx
            dy = cy - ky
            if dx * dx + dy * dy <= dist2_gate:
                too_close = True
                break
        if not too_close:
            added.append((int(x), int(y), int(w), int(h)))

    # 低阈值高区补检（通用几何，不做语义特判）：
    # 全区 0.56 对顶区某些整段条（如 11.png 的纳什）召回不足，这里仅在高区再走一遍 0.50。
    lo_top_boxes = detect_healthbars_by_templates(crop, templates, threshold=0.50, nms_iou=0.50)
    for (x, y, w, h) in lo_top_boxes:
        y_abs = int(y + y1)
        if int(y_abs) > int(TOP_ZONE_MAX_Y_ABS):
            continue
        if not (min_w <= int(w) <= max_w and min_h <= int(h) <= max_h):
            continue
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        too_close = False
        for (bx, by, bw, bh) in boxes_main:
            kx = float(bx + bw / 2.0)
            ky = float(by + bh / 2.0)
            if (cx - kx) ** 2 + (cy - ky) ** 2 <= dist2_gate:
                too_close = True
                break
        if not too_close:
            for (ax, ay, aw, ah) in added:
                kx = float(ax + aw / 2.0)
                ky = float(ay + ah / 2.0)
                if (cx - kx) ** 2 + (cy - ky) ** 2 <= dist2_gate:
                    too_close = True
                    break
        if not too_close:
            added.append((int(x), int(y), int(w), int(h)))

    boxes = _dedup_boxes_by_center(boxes_main + added, dist_thresh=14)

    # 顶部小框再聚合一次：避免一个高位血条被拆成多个 54x9 小框
    top_big: List[Tuple[int, int, int, int]] = []
    top_small: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        x, y, w, h = b
        if int(y) <= top_small_band_y and int(w) < 70 and int(h) <= 12:
            top_small.append(b)
        else:
            top_big.append(b)
    if top_small:
        merged_top_small: List[Tuple[int, int, int, int]] = []
        centers: List[Tuple[float, float]] = []
        dist2 = float(46 * 46)
        # 面积大优先保留
        for (x, y, w, h) in sorted(top_small, key=lambda b: -(b[2] * b[3])):
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
            ok = True
            for (kx, ky) in centers:
                dx = cx - kx
                dy = cy - ky
                if dx * dx + dy * dy <= dist2:
                    ok = False
                    break
            if ok:
                merged_top_small.append((int(x), int(y), int(w), int(h)))
                centers.append((cx, cy))
        boxes = top_big + merged_top_small

    # 顶部碎片框 → 整段血条（纳什男爵等）：在转回全图坐标前处理
    ref_w, ref_h = _estimate_ref_bar_size(templates)
    fixed: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        fx, fy, fw, fh = _snap_top_bar_fragment_to_full_bar(
            b, ref_w, ref_h, cw, ch, top_band_y=top_small_band_y
        )
        fixed.append((fx, fy, fw, fh))
    boxes = _dedup_boxes_by_center(fixed, dist_thresh=18)
    boxes = _merge_overlapping_bar_boxes(boxes)

    out: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes:
        ya = int(y + y1)
        out.append((int(x + x1), ya, int(w), int(h)))
    return out


def _crop_circle_bgra(scene_bgr: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    """裁直径 2r 的圆形区域，圆外 alpha=0，返回 BGRA。"""
    d = 2 * r
    H, W = scene_bgr.shape[:2]
    x0, y0 = cx - r, cy - r
    x1, y1 = x0 + d, y0 + d
    need_pad = x0 < 0 or y0 < 0 or x1 > W or y1 > H
    img = scene_bgr
    if need_pad:
        pad_l = max(0, -x0)
        pad_t = max(0, -y0)
        pad_r = max(0, x1 - W)
        pad_b = max(0, y1 - H)
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0 += pad_l
        y0 += pad_t
    rect = img[y0 : y0 + d, x0 : x0 + d].copy()
    if rect.shape[0] != d or rect.shape[1] != d:
        out = np.zeros((d, d, 3), dtype=np.uint8)
        rh, rw = rect.shape[:2]
        out[:rh, :rw] = rect
        rect = out
    mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    return np.dstack([rect, mask])


def _bgra_to_bgr_white(bgra: np.ndarray) -> np.ndarray:
    """BGRA 叠白底转 BGR。"""
    bgr = bgra[:, :, :3].astype(np.float32)
    a = (bgra[:, :, 3:4].astype(np.float32) / 255.0)
    white = np.ones_like(bgr) * 255.0
    out = (bgr * a + white * (1 - a)).astype(np.uint8)
    return out


def _tight_crop_by_alpha(bgra: np.ndarray, alpha_thresh: int = 18, pad: int = 2) -> np.ndarray:
    """
    按 alpha 紧裁切圆内有效区域，减少白底占比，提高 embedding 稳定性。
    返回裁切后的 BGRA（仍保留透明通道）。
    """
    a = bgra[:, :, 3]
    ys, xs = np.where(a >= alpha_thresh)
    if ys.size == 0 or xs.size == 0:
        return bgra
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(bgra.shape[1], x2 + pad)
    y2 = min(bgra.shape[0], y2 + pad)
    return bgra[y1:y2, x1:x2].copy()


def _build_piece_embedding_db(piece_dir: Path, model, device, transform) -> List[Tuple[str, np.ndarray]]:
    """特征库 -> embedding（L2 归一化）"""
    import torch

    # 优先读 report.json 列表
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
            files = []
    if not files:
        # 兼容“子目录存图”（例如 build_gallery_from_preprocessed_200 输出到 images/）
        files = sorted([p for p in piece_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    db: List[Tuple[str, np.ndarray]] = []
    with torch.no_grad():
        for p in files:
            try:
                bgr = _load_image(p, allow_alpha=True)
            except Exception:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image

            x = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            feat = model(x).detach().cpu().numpy().flatten().astype(np.float32)
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            db.append((p.stem, feat))
    return db


def project_dir() -> Path:
    """项目根目录（本文件所在目录）。"""
    return Path(__file__).resolve().parent


def _piece_dir_fingerprint(piece_dir: Path) -> str:
    """
    生成特征库指纹，用于稳定缓存键。
    - 若存在 report.json：按其内容哈希；
    - 否则：按图片文件清单（相对路径/大小/mtime）哈希。
    """
    report = piece_dir / "report.json"
    if report.is_file():
        try:
            return "report:" + hashlib.sha1(report.read_bytes()).hexdigest()[:20]
        except Exception:
            try:
                st = report.stat()
                return f"report_mtime:{int(st.st_mtime_ns)}"
            except Exception:
                return "report_unknown"

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([p for p in piece_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    h = hashlib.sha1()
    for p in files:
        try:
            st = p.stat()
        except Exception:
            continue
        rel = str(p.relative_to(piece_dir)).replace("\\", "/")
        h.update(rel.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(str(int(st.st_size)).encode("utf-8"))
        h.update(b"|")
        h.update(str(int(st.st_mtime_ns)).encode("utf-8"))
        h.update(b"\n")
    return f"scan:{len(files)}:{h.hexdigest()[:20]}"


def piece_embedding_db_cache_path(
    piece_dir: Path,
    embed_backbone: str,
    *,
    root: Optional[Path] = None,
) -> Path:
    """
    embedding 库缓存文件路径；与 `load_or_build_piece_embedding_db` / ``main()`` 使用同一键规则。
    键含：特征库路径、特征库内容指纹、骨干名。
    """
    root = root or project_dir()
    cache_dir = root / ".piece_db_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _piece_dir_fingerprint(piece_dir)
    resolved = _resolve_embed_backbone(embed_backbone)
    key = f"{str(piece_dir.resolve())}|{fp}|{resolved}|piece_db_v2"
    cache_key = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"piece_db_{cache_key}.pkl"


def load_or_build_piece_embedding_db(
    piece_dir: Path,
    model,
    device,
    transform,
    embed_backbone: str,
    *,
    root: Optional[Path] = None,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> Tuple[List[Tuple[str, np.ndarray]], Path]:
    """
    从 ``.piece_db_cache`` 加载缓存；不存在或 ``force_rebuild`` 时构建并写入。
    单独跑识别脚本且未传 ``piece_db`` 时也会走此逻辑，避免每次全量重算 embedding。
    """
    cache_path = piece_embedding_db_cache_path(piece_dir, embed_backbone, root=root)
    if not force_rebuild and cache_path.is_file():
        try:
            db = pickle.loads(cache_path.read_bytes())
            if isinstance(db, list) and len(db) > 0:
                if verbose:
                    print(f"[embedding] 加载缓存: {cache_path.name}（{len(db)} 条）")
                return db, cache_path
        except Exception:
            pass
    if verbose:
        print("[embedding] 构建特征库向量 …")
    db = _build_piece_embedding_db(piece_dir, model, device, transform)
    if not db:
        raise RuntimeError("特征库 embedding 为空（检查特征库图片/权限）")
    cache_path.write_bytes(pickle.dumps(db))
    if verbose:
        print(f"[embedding] 已写入: {cache_path}")
    return db, cache_path


def _prepare_piece_db_matrix(db: List[Tuple[str, np.ndarray]]) -> Tuple[List[str], np.ndarray]:
    names = [str(name) for name, _ in db]
    vecs = [np.asarray(v, dtype=np.float32) for _, v in db]
    if not vecs:
        return names, np.zeros((0, 0), dtype=np.float32)
    mat = np.stack(vecs, axis=0)
    return names, mat


def _topk_cosine(
    q: np.ndarray,
    db_names: List[str],
    db_mat: np.ndarray,
    topk: int = 5,
) -> List[Tuple[str, float]]:
    if db_mat.size == 0 or not db_names:
        return []
    sims = db_mat @ q
    k = int(max(1, min(int(topk), sims.shape[0])))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(db_names[int(i)], float(sims[int(i)])) for i in idx]


def run_recognition(
    *,
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
    # 仅在 stage2 过滤后只剩 1 个有效采样点时的保守兜底：
    # 若仍然能看到“顶1相似度足够高 + 聚合分差足够大”，则允许输出，避免把高置信样本误判为 null/?
    stage2_single_vote_min_top1max: float = 0.75,
    stage2_single_vote_min_gap_score: float = 0.30,
    # 三阶段兜底：仅对 best=None 的血条尝试“共识补全”（不改已有 best，最大化保持准确率）
    fill_unknown: bool = True,
    fill_min_support: int = 6,
    fill_min_vote_gap: int = 2,
    fill_min_avg_sim: float = 0.64,
    # 当 stage2 过滤后 used_samples==0 时：允许较弱投票但要求“总相似度/峰值”仍明显领先
    fill_min_support_no_votes: int = 2,
    fill_min_vote_gap_no_votes: int = 0,
    fill_min_max_sim_no_votes: float = 0.65,
    fill_min_sum_gap_no_votes: float = 0.08,
    # stage2 used_samples==0 时：额外做一次“更低阈值过滤重判”
    fill_stage2_retry_min_sim: float = 0.58,
    fill_stage2_retry_min_margin: float = 0.005,
    fill_stage2_retry_top1sum_gap: float = 0.15,
    alpha_tight: bool = True,
    alpha_thresh: int = 18,
    embed_backbone: str = "dinov2_vits14",
    # 允许外部预热/复用，避免每张图重复建库/加载模型（批处理会快很多）
    model=None,
    device=None,
    transform=None,
    piece_db: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    以“多点圆形采样 + embedding 聚合投票”运行一次识别。
    会写入 output_dir：标记图、crops、result.json；同时返回 out_json dict。
    """
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

    # 采样圆直径/半径
    circle_diameter = int(circle_diameter)
    r = max(8, circle_diameter // 2)

    scene = _load_image(img_path)
    templates = load_healthbar_templates(tpl_path)

    boxes = _detect_healthbars_in_roi(scene, templates, ROI)

    # embedding 模型（默认 DINOv2-ViT-S）
    if model is None or device is None or transform is None:
        model, device, transform = _get_embedding_model(embed_backbone)
    if piece_db is None:
        piece_db, _ = load_or_build_piece_embedding_db(
            piece_dir, model, device, transform, embed_backbone, root=project_dir()
        )
    db_names, db_mat = _prepare_piece_db_matrix(piece_db)

    import torch
    from PIL import Image

    mark = scene.copy()
    x1, y1, x2, y2 = ROI
    cv2.rectangle(mark, (x1, y1), (x2, y2), (128, 128, 255), 2)
    # 可视化分块网格，便于观察局部检测窗口
    roi_w = max(0, int(x2 - x1))
    roi_h = max(0, int(y2 - y1))
    ref_w, ref_h = _estimate_ref_bar_size(templates)
    for tx, ty, tw, th in _build_roi_tiles_adaptive(roi_w, roi_h, ref_w, ref_h):
        cv2.rectangle(
            mark,
            (int(x1 + tx), int(y1 + ty)),
            (int(x1 + tx + tw), int(y1 + ty + th)),
            (255, 128, 0),
            1,
        )

    # 这里沿用 main() 默认采样点（你当前工程的最佳组合）
    samples = DEFAULT_SAMPLES

    results: List[Dict[str, Any]] = []
    filtered_wild_top_zone: List[Dict[str, Any]] = []
    with torch.no_grad():
        for bi, (x, y, w, h) in enumerate(boxes):
            # 先保存画布快照；若后续判定为“应丢弃”，回滚掉这条的框/圈，避免残留可视化噪声。
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

                bgra = _crop_circle_bgra(scene, cx, cy, r)
                if alpha_tight:
                    bgra = _tight_crop_by_alpha(bgra, alpha_thresh=int(alpha_thresh), pad=2)
                bgr = _bgra_to_bgr_white(bgra)
                crop_path = crops_dir / f"bar{bi+1:02d}_stage1_s{si+1:02d}_{dx}_{dy}.png"
                ok, buf = cv2.imencode(".png", bgra)
                if ok:
                    Path(crop_path).write_bytes(buf.tobytes())

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                xt = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
                q = model(xt).detach().cpu().numpy().flatten().astype(np.float32)
                q = q / (np.linalg.norm(q) + 1e-8)

                # embedding-only
                top = _topk_cosine(q, db_names, db_mat, topk=5)
                # 记录 raw top（便于 debug），但用于聚合/投票的 top 会按英雄名合并
                per_samples.append({"dx": dx, "dy": dy, "top": top})
                if not top:
                    continue

                # 单点过滤：embedding-only
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
                # 聚合也按英雄合并，避免同英雄变体分票
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
                    bgra = _crop_circle_bgra(scene, cx, cy, r)
                    if alpha_tight:
                        bgra = _tight_crop_by_alpha(bgra, alpha_thresh=int(alpha_thresh), pad=2)
                    bgr = _bgra_to_bgr_white(bgra)
                    crop_path = crops_dir / f"bar{bi+1:02d}_stage2_s{si+1:02d}_{dx}_{dy}.png"
                    ok, buf = cv2.imencode(".png", bgra)
                    if ok:
                        Path(crop_path).write_bytes(buf.tobytes())

                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    xt = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
                    q = model(xt).detach().cpu().numpy().flatten().astype(np.float32)
                    q = q / (np.linalg.norm(q) + 1e-8)

                    # embedding-only
                    top = _topk_cosine(q, db_names, db_mat, topk=5)
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
                    # 只有极少数“过滤后只剩 1 个有效采样点”的情况：
                    # 使用更保守的判据（顶1 max 足够高 + 聚合 gap 足够大），
                    # 让这类高置信样本从 null 变为具体英雄，尽量不牺牲准确率。
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

                # 三阶段兜底：当 stage2 仍输出 None，但“同一英雄在多数采样点 top1”时补全。
                # 典型：克格莫这类在对局截图里相似度整体偏低，但跨采样点非常一致。
                filled_by_consensus = False
                filled_reason: Optional[Dict[str, Any]] = None
                if best2 is None and bool(fill_unknown):
                    # 先做：stage2 used_samples==0 的重判
                    # 原因：stage2_min_sim=0.74 偏高会让所有采样点都被过滤掉（used_samples==0）。
                    # 对这种“彻底失败”的情况，允许用更低的 min_sim 再算一轮 votes，避免把塞恩错补成相似英雄。
                    if int(used_samples2) == 0:
                        # 优先只用 stage1 的 DEFAULT_SAMPLES 重判，避免 extra 样本把结果带偏
                        samples_set = {(int(dx), int(dy)) for (dx, dy) in samples}

                        votes_default: Dict[str, int] = {}
                        top1_sum_default: Dict[str, float] = {}

                        votes_all: Dict[str, int] = {}
                        top1_sum_all: Dict[str, float] = {}

                        for s in per_samples2:
                            top_raw = s.get("top") or []
                            if not top_raw:
                                continue

                            dx_i = int(s.get("dx", 0))
                            dy_i = int(s.get("dy", 0))
                            is_default = (dx_i, dy_i) in samples_set

                            # 投票/过滤按英雄合并后的 top1/top2
                            top_for_filter = _collapse_topk_to_hero([(str(n), float(v)) for (n, v) in top_raw])
                            if not top_for_filter:
                                continue

                            top1_name, top1_sim = top_for_filter[0]
                            top2_sim = float(top_for_filter[1][1]) if len(top_for_filter) >= 2 else -1.0
                            margin = float(top1_sim) - float(top2_sim)

                            if float(top1_sim) >= float(fill_stage2_retry_min_sim) and margin >= float(fill_stage2_retry_min_margin):
                                # all
                                votes_all[top1_name] = votes_all.get(top1_name, 0) + 1
                                top1_sum_all[top1_name] = top1_sum_all.get(top1_name, 0.0) + float(top1_sim)

                                # default only
                                if is_default:
                                    votes_default[top1_name] = votes_default.get(top1_name, 0) + 1
                                    top1_sum_default[top1_name] = top1_sum_default.get(top1_name, 0.0) + float(top1_sim)

                        def _decide_from_votes(votes_map: Dict[str, int], sum_map: Dict[str, float]) -> Optional[str]:
                            if not votes_map:
                                return None
                            ordered = sorted(
                                votes_map.keys(),
                                key=lambda k: (int(votes_map.get(k, 0)), float(sum_map.get(k, 0.0))),
                                reverse=True,
                            )
                            c0r = ordered[0]
                            c1r = ordered[1] if len(ordered) >= 2 else None
                            best_votes_r = int(votes_map.get(c0r, 0))
                            second_votes_r = int(votes_map.get(c1r, 0)) if c1r else 0
                            top1sum_gap_r = float(sum_map.get(c0r, 0.0)) - float(sum_map.get(c1r, 0.0)) if c1r else float(sum_map.get(c0r, 0.0))

                            if best_votes_r < int(min_votes):
                                return None
                            if best_votes_r > second_votes_r:
                                if (best_votes_r - second_votes_r) >= 1:
                                    return c0r
                            else:
                                if top1sum_gap_r >= float(fill_stage2_retry_top1sum_gap):
                                    return c0r
                            return None

                        # 先 default，再全量
                        c0r = _decide_from_votes(votes_default, top1_sum_default)
                        decide_stage = "default"
                        if c0r is None:
                            c0r = _decide_from_votes(votes_all, top1_sum_all)
                            decide_stage = "all"

                        if c0r is not None:
                            winner_max_sim = -1.0
                            for s in per_samples2:
                                top_raw = s.get("top") or []
                                if not top_raw:
                                    continue
                                th = _collapse_topk_to_hero([(str(n), float(v)) for (n, v) in top_raw])
                                if th and str(th[0][0]) == str(c0r):
                                    winner_max_sim = max(winner_max_sim, float(th[0][1]))
                            best2 = c0r
                            filled_by_consensus = True
                            filled_reason = {
                                "method": "stage2_retry_min_sim",
                                "used_samples_stage2": int(used_samples2),
                                "retry_min_sim": float(fill_stage2_retry_min_sim),
                                "retry_min_margin": float(fill_stage2_retry_min_margin),
                                "support": int(votes_default.get(c0r, votes_all.get(c0r, 0))),
                                "decide_stage": decide_stage,
                                "retry_top1sum_gap_min": float(fill_stage2_retry_top1sum_gap),
                                "winner_max_sim": float(winner_max_sim),
                            }

                    # 若重判已填充则跳过后续统计补全
                    if best2 is not None:
                        pass

                    # 统计：所有采样点（不过滤）各英雄的 top1 票数，以及平均 top1_sim
                    counts: Dict[str, int] = {}
                    sim_sum: Dict[str, float] = {}
                    sim_n: Dict[str, int] = {}
                    sim_max: Dict[str, float] = {}
                    # 共识统计：各采样点（未过滤）的 top1 英雄
                    for s in per_samples2:
                        top_raw = s.get("top") or []
                        if not top_raw:
                            continue
                        # per_samples2 存的是融合/embedding 的 top；这里按英雄合并后取 top1
                        top_hero = _collapse_topk_to_hero([(str(n), float(v)) for (n, v) in top_raw])
                        if not top_hero:
                            continue
                        h1, sim1 = top_hero[0]
                        counts[h1] = counts.get(h1, 0) + 1
                        sim_sum[h1] = sim_sum.get(h1, 0.0) + float(sim1)
                        sim_n[h1] = sim_n.get(h1, 0) + 1
                        sim_max[h1] = max(sim_max.get(h1, -1.0), float(sim1))

                    # 选择共识英雄：
                    # 1) 票数最多（更稳，适合像克格莫那种跨采样点高度一致）
                    # 2) 若阶段二 used_samples==0：用“总相似度 sim_sum”领先作为弱证据补全（用于塞恩这类整体偏低但仍有一致倾向）
                    cand_by_vote = sorted(
                        counts.keys(),
                        key=lambda k: (
                            int(counts.get(k, 0)),
                            float(sim_sum.get(k, 0.0)) / max(1, int(sim_n.get(k, 0))),
                        ),
                        reverse=True,
                    )
                    cand_by_sum = sorted(sim_sum.keys(), key=lambda k: float(sim_sum.get(k, 0.0)), reverse=True)
                    if cand_by_vote:
                        c0 = cand_by_vote[0]
                        c1 = cand_by_vote[1] if len(cand_by_vote) >= 2 else None
                        c0_cnt = int(counts.get(c0, 0))
                        c1_cnt = int(counts.get(c1, 0)) if c1 else 0
                        vote_gap = c0_cnt - c1_cnt
                        avg_sim = float(sim_sum.get(c0, 0.0)) / max(1, int(sim_n.get(c0, 0)))
                        c0_max_sim = float(sim_max.get(c0, -1.0))
                        # 用“总相似度”来计算二者差距（总量领先比单次 peak 更稳定）
                        s0 = float(sim_sum.get(c0, 0.0))
                        s1 = float(sim_sum.get(cand_by_sum[1], 0.0)) if len(cand_by_sum) >= 2 else 0.0
                        sum_gap = s0 - s1
                        # A) 强一致补全（原逻辑）：要求票数领先 + 平均相似度足够
                        ok_vote_consensus = (
                            c0_cnt >= int(fill_min_support)
                            and vote_gap >= int(fill_min_vote_gap)
                            and avg_sim >= float(fill_min_avg_sim)
                        )

                        # B) 低投票补全：stage2 used_samples==0 时，用“总相似度/峰值”领先来补（用于塞恩这类情况）
                        ok_no_votes_consensus = (
                            int(used_samples2) == 0
                            and c0_cnt >= int(fill_min_support_no_votes)
                            and (vote_gap >= int(fill_min_vote_gap_no_votes))
                            and c0_max_sim >= float(fill_min_max_sim_no_votes)
                            and sum_gap >= float(fill_min_sum_gap_no_votes)
                        )

                        if best2 is None and (ok_vote_consensus or ok_no_votes_consensus):
                            best2 = c0
                            filled_by_consensus = True
                            filled_reason = {
                                "method": "consensus_top1",
                                "support": c0_cnt,
                                "second_support": c1_cnt,
                                "vote_gap": vote_gap,
                                "avg_sim": avg_sim,
                                "max_sim": c0_max_sim,
                                "sum_gap": sum_gap,
                                "fill_min_support": int(fill_min_support),
                                "fill_min_vote_gap": int(fill_min_vote_gap),
                                "fill_min_avg_sim": float(fill_min_avg_sim),
                                "fill_min_support_no_votes": int(fill_min_support_no_votes),
                                "fill_min_vote_gap_no_votes": int(fill_min_vote_gap_no_votes),
                                "fill_min_max_sim_no_votes": float(fill_min_max_sim_no_votes),
                                "fill_min_sum_gap_no_votes": float(fill_min_sum_gap_no_votes),
                            }

                stage2_detail = {
                    "best": best2,
                    "best_score": float(best2_score),
                    "second_score": float(second2_score),
                    "gap": float(gap2),
                    "used_samples": int(used_samples2),
                    "vote_top": sorted([{"name": k, "votes": int(v)} for k, v in votes2.items()], key=lambda x: -x["votes"])[:10],
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
                    "filled_by_consensus": bool(filled_by_consensus),
                    "filled_reason": filled_reason,
                }

                if best2 is not None:
                    best = best2
                    best_score = float(best2_score)
                    second_score = float(second2_score)
                    gap = float(gap2)
                    # 当 stage3 consensus 补全了结果但 stage2 used_samples==0 时，给合成计数避免被 <=0 过滤
                    used_samples = int(used_samples2) if int(used_samples2) > 0 else (1 if filled_by_consensus else 0)
                    votes = votes2
                    agg_sorted = agg2_sorted
                    per_samples = per_samples2

            # 防止“空气识别”：没有有效采样点时直接丢弃，不输出 ?。
            if int(used_samples) <= 0:
                mark = mark_before
                continue

            # 高区语义限制（全图坐标）：仅允许纳什男爵 / 海克斯霸龙作为棋子条输出，其余记为野怪并丢弃展示。
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

            # 丢弃仅靠「低阈值重判投票」补全、且顶相似度仍明显不足的血条框（多为模板误检伪框）。
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
                    "vote_top": sorted([{"name": k, "votes": int(v)} for k, v in votes.items()], key=lambda x: -x["votes"])[:10],
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

    # 输出阶段补充棋盘格定位（基于本文件内置参数）
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
            inferred_row = _infer_row_for_mapping(
                ay,
                float(by),
                float(by + bh),
                row_ys,
            )
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

    mark_path = output_dir / f"{img_path.stem}_多点Embedding_标记.png"
    _save_image(mark, mark_path)

    out_json: Dict[str, Any] = {
        "image": str(img_path),
        "piece_dir": str(piece_dir),
        "backend": "embedding",
        "roi": list(ROI),
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


def _clear_dir(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """批量：``对局截图`` 下全部 png → ``chess_recog/{图名}/`` 带棋子标注图与 result.json。运行开始时清空 ``chess_recog/``。"""
    import argparse
    import json
    import shutil

    import bars_recog as br

    ap = argparse.ArgumentParser(description="棋子 embedding 批量识别（输出至 chess_recog/）")
    ap.add_argument("--img-dir", type=Path, default=Path("对局截图"), help="对局截图目录")
    ap.add_argument("--png-count", type=int, default=10, help="自动探测目录时要求的 png 数量")
    ap.add_argument("--from", dest="start", type=int, default=None)
    ap.add_argument("--to", dest="end", type=int, default=None)
    ap.add_argument(
        "--piece-dir",
        type=Path,
        default=Path("chess_gallery"),
        help="特征库目录",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_BATCH_EMBED_OUT_DIRNAME),
        help="输出根目录（默认 chess_recog）",
    )
    ap.add_argument("--circle-diameter", type=int, default=84)
    ap.add_argument(
        "--build-embed-cache-only",
        action="store_true",
        help="仅构建/更新 .piece_db_cache 下的 embedding 缓存后退出（可事先执行，避免批量跑图时现场建库）",
    )
    ap.add_argument(
        "--force-rebuild-embed-cache",
        action="store_true",
        help="忽略已有 embedding 缓存并全量重建",
    )
    ap.add_argument(
        "--clean-other-out",
        action="store_true",
        default=True,
        help="清理项目下其它 _newgallery_emb_d84* 前缀旧输出目录（与原先 batch_recog_d84 默认一致）",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    img_dir = args.img_dir
    if img_dir is None:
        for d in root.iterdir():
            if not d.is_dir():
                continue
            try:
                n = len(list(d.glob("*.png")))
            except Exception:
                continue
            if n == int(args.png_count):
                img_dir = d
                break
        else:
            raise SystemExit(f"未找到包含 {args.png_count} 张 png 的文件夹，请用 --img-dir 指定")
    else:
        img_dir = root / img_dir if not img_dir.is_absolute() else img_dir
    if not img_dir.is_dir():
        raise SystemExit(f"img-dir 不存在: {img_dir}")

    piece_dir = args.piece_dir if args.piece_dir.is_absolute() else root / args.piece_dir
    if not piece_dir.is_dir():
        raise SystemExit(f"特征库不存在: {piece_dir}")

    backbone = "dinov2_vits14"
    if args.build_embed_cache_only:
        print(f"加载模型 {backbone} …")
        model, device, transform = _get_embedding_model(backbone)
        db, cache_path = load_or_build_piece_embedding_db(
            piece_dir,
            model,
            device,
            transform,
            backbone,
            root=root,
            force_rebuild=bool(args.force_rebuild_embed_cache),
        )
        print(f"完成。embedding 缓存: {cache_path}（{len(db)} 条）")
        raise SystemExit(0)

    try:
        template = br.find_healthbar_template(root)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            raise SystemExit("--from 和 --to 必须同时提供")
        imgs: List[Path] = []
        for n in range(int(args.start), int(args.end) + 1):
            p = img_dir / f"{n}.png"
            if p.is_file():
                imgs.append(p)
            else:
                print(f"[跳过] 不存在: {p}")
        imgs = sorted(imgs, key=lambda p: p.name)
    else:
        imgs = sorted(img_dir.glob("*.png"))
        if not imgs:
            raise SystemExit(f"目录内无 png: {img_dir}")
    if not imgs:
        raise SystemExit("没有可识别的 png")

    out_root = args.out if args.out.is_absolute() else root / args.out
    out_prefix = "_newgallery_emb_d84"
    if args.clean_other_out:
        try:
            for p in root.iterdir():
                if not p.is_dir() or p.name == out_root.name:
                    continue
                if p.name.startswith(out_prefix):
                    print(f"[清理旧输出] {p} …")
                    shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    print(f"[清空输出] {out_root} …")
    _clear_dir(out_root)

    print(f"加载模型 {backbone} …")
    model, device, transform = _get_embedding_model(backbone)

    piece_db, _ = load_or_build_piece_embedding_db(
        piece_dir,
        model,
        device,
        transform,
        backbone,
        root=root,
        force_rebuild=bool(args.force_rebuild_embed_cache),
    )
    print(f"库条目数: {len(piece_db)}")

    summary: list = []
    for img in imgs:
        odir = out_root / img.stem
        odir.mkdir(parents=True, exist_ok=True)
        print(f"→ {img.name}")
        out_json = run_recognition(
            image_path=img,
            template_path=template,
            piece_dir=piece_dir,
            output_dir=odir,
            circle_diameter=int(args.circle_diameter),
            embed_backbone=backbone,
            model=model,
            device=device,
            transform=transform,
            piece_db=piece_db,
        )
        bars = []
        for r in out_json.get("results") or []:
            bars.append({"bar_index": r.get("bar_index"), "best": r.get("best")})
        summary.append({"image": img.name, "bars": bars})

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成。输出: {out_root}")
    print(f"汇总: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
