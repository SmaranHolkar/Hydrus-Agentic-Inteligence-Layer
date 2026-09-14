"""Test package bootstrap for local monorepo imports.

Ensures `hail_core` (from HAIL/src) and sibling runtime packages under
`HAIL/` are importable when running tests via unittest.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAIL_ROOT = ROOT / "HAIL"
HAIL_SRC = HAIL_ROOT / "src"

for candidate in (HAIL_SRC, HAIL_ROOT):
    path_str = str(candidate)
    if candidate.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
