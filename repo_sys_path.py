# -*- coding: utf-8 -*-
"""将仓库根、`scripts/core`、`scripts/extra` 加入 sys.path（供子目录中的入口脚本 import）。"""
from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def prepend() -> None:
    base = repo_root()
    for p in (base, base / "scripts", base / "scripts" / "core", base / "scripts" / "extra"):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


prepend()
