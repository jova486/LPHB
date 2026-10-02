#!/usr/bin/env python
"""Generate per-dataset prompt-ablation audit CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.release_analysis import run_per_dataset_prompt_ablation  # noqa: E402


if __name__ == "__main__":
    outputs = run_per_dataset_prompt_ablation()
    print("Created outputs:")
    for path in outputs:
        print(f"- {path.relative_to(REPO_ROOT)}")
