"""Repository paths for the review artifact."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Find repository root using stable artifact markers.
    """
    current = start if start.is_dir() else start.parent

    for path in [current, *current.parents]:
        has_stable_markers = (
            (path / "README.md").exists()
            and (path / "data").is_dir()
            and (path / "scripts").is_dir()
            and (path / "src").is_dir()
        )
        if has_stable_markers:
            return path

        has_git_and_layout = (
            (path / ".git").exists()
            and (path / "data").is_dir()
            and (path / "scripts").is_dir()
            and (path / "src").is_dir()
        )
        if has_git_and_layout:
            return path

    raise FileNotFoundError(
        "Could not locate repository root. Expected README.md, data/, scripts/, and src/."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve())

DATA_DIR = REPO_ROOT / "data"
AUDITS_DIR = REPO_ROOT / "audits"
TABLES_DIR = REPO_ROOT / "tables"
FIGURES_DIR = REPO_ROOT / "figures"
PROMPTS_DIR = REPO_ROOT / "prompts"

MASTER_CSV = DATA_DIR / "human_judge_release_master.csv"
