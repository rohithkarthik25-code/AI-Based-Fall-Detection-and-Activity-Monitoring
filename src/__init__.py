"""
AI Fall Detection and Activity Monitoring Package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

for _dir in [
    _ROOT_DIR,
    _SRC_DIR,
    _SRC_DIR / "core",
    _SRC_DIR / "web",
    _SRC_DIR / "alerts",
    _SRC_DIR / "visualization",
    _SRC_DIR / "utils",
]:
    _p = str(_dir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

