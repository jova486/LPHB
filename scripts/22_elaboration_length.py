#!/usr/bin/env python
"""Generate the answer-length reduction analysis for the elaboration intervention."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.release_analysis import run_elaboration_length  # noqa: E402


if __name__ == "__main__":
    outputs = run_elaboration_length()
    print("Created outputs:")
    for path in outputs:
        print(f"- {path.relative_to(REPO_ROOT)}")
