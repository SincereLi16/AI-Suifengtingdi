# -*- coding: utf-8 -*-
"""工程根目录与约定路径（runs/、docs/、assets/）。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = PROJECT_ROOT / "assets"

# 流水线默认输出（均在 runs/ 下，便于根目录整洁）
DEFAULT_OUT_BATTLE_PIPELINE_V3 = RUNS_DIR / "battle_pipeline_v3_out"
DEFAULT_OUT_TCV = RUNS_DIR / "tcv_info"
DEFAULT_OUT_FIGHTBOARD_V2 = RUNS_DIR / "fightboard_info_v2"
DEFAULT_OUT_FIGHTBOARD = RUNS_DIR / "fightboard_info"
DEFAULT_OUT_PLAYER_INFO = RUNS_DIR / "player_info"
DEFAULT_OUT_PLAYER_ONNX = RUNS_DIR / "player_info_onnx"
DEFAULT_OUT_PREBOARD = RUNS_DIR / "preboard_info"
DEFAULT_OUT_PREBOARD_V3 = RUNS_DIR / "preboard_info_v3"
DEFAULT_OUT_EQUIP_COLUMN = RUNS_DIR / "equip_column_recog"
DEFAULT_OUT_EQUIP_RECOG = RUNS_DIR / "equip_recog"

# 基准 / 样例图（可选）
DEFAULT_ASSET_SPARE_SHOTS = ASSETS_DIR / "备用截图"
