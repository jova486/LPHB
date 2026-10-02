#!/usr/bin/env python
"""Run all Level 1 release analyses extracted from notebook Section C."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.release_analysis import run_all_release_analyses  # noqa: E402


if __name__ == "__main__":
    outputs = run_all_release_analyses()
    print("Created outputs:")
    for path in outputs:
        print(f"- {path.relative_to(REPO_ROOT)}")
