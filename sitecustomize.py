"""Workspace import bootstrap for tests and local scripts.

Python auto-imports sitecustomize during startup, so this ensures monorepo
packages are importable no matter which test runner/discovery mode is used.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HAIL_ROOT = ROOT / "HAIL"
HAIL_SRC = HAIL_ROOT / "src"

for candidate in (ROOT, HAIL_ROOT, HAIL_SRC):
    path_str = str(candidate)
    if candidate.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
