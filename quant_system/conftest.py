"""Pytest bootstrap.

Ensures the repository parent directory is on ``sys.path`` so that the
``quant_system`` package resolves identically under ``pytest``,
``python main.py`` and ``python -m quant_system.main``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
