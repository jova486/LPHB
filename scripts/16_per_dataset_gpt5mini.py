#!/usr/bin/env python
"""Generate per-dataset GPT-5-mini p1/p2 kappa table."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.release_analysis import run_per_dataset_gpt5mini  # noqa: E402


if __name__ == "__main__":
    outputs = run_per_dataset_gpt5mini()
    print("Created outputs:")
    for path in outputs:
        print(f"- {path.relative_to(REPO_ROOT)}")
