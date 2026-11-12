"""
Test suite bootstrap: ensures src and its subpackages are present in sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

for _p in [
    _ROOT,
    _SRC,
    _SRC / "core",
    _SRC / "web",
    _SRC / "alerts",
    _SRC / "visualization",
    _SRC / "utils",
]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
