#!/usr/bin/env python
"""Validate the Level 1 release master CSV."""

from __future__ import annotations

import sys
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    BINARY_LABEL_COLUMNS,
    EXPECTED_GENERATORS_PER_QID,
    EXPECTED_QID_COUNT,
    EXPECTED_ROW_COUNT,
    FORBIDDEN_PUBLIC_COLUMN_PATTERNS,
    REQUIRED_COLUMNS,
    TERNARY_LABEL_COLUMNS,
)
from src.paths import MASTER_CSV  # noqa: E402


def _record(checks: list[tuple[str, bool, str]], name: str, passed: bool, detail: str) -> None:
    checks.append((name, passed, detail))


def _valid_values(rows: list[dict[str, str]], column: str, allowed: set[int]) -> bool:
    values: set[int] = set()
    for row in rows:
        raw_value = row.get(column, "")
        try:
            value = int(raw_value)
        except ValueError:
            return False
        values.add(value)
    return values.issubset(allowed)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    repo_root_markers = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "data",
        REPO_ROOT / "scripts",
        REPO_ROOT / "src",
        REPO_ROOT / "data" / "human_judge_release_master.csv",
    ]

    _record(
        checks,
        "repo root",
        all(path.exists() for path in repo_root_markers),
        str(REPO_ROOT),
    )

    master_exists = MASTER_CSV.is_file()
    _record(checks, "master csv exists", master_exists, str(MASTER_CSV.relative_to(REPO_ROOT)))

    if not master_exists:
        print_summary(checks)
        return 1

    try:
        with MASTER_CSV.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames or []
    except Exception as exc:
        _record(checks, "master csv readable", False, repr(exc))
        print_summary(checks)
        return 1

    _record(checks, "master csv readable", True, f"{len(rows)} rows, {len(columns)} columns")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    _record(
        checks,
        "required columns",
        not missing_columns,
        "all present" if not missing_columns else ", ".join(missing_columns),
    )

    _record(
        checks,
        "row count",
        len(rows) == EXPECTED_ROW_COUNT,
        f"found {len(rows)}, expected {EXPECTED_ROW_COUNT}",
    )

    if "qid" in columns:
        qid_counts = Counter(row["qid"] for row in rows)
        qid_count = len(qid_counts)
        _record(
            checks,
            "qid count",
            qid_count == EXPECTED_QID_COUNT,
            f"found {qid_count}, expected {EXPECTED_QID_COUNT}",
        )

        bad_qids = [
            qid
            for qid, count in qid_counts.items()
            if count != EXPECTED_GENERATORS_PER_QID
        ]
        _record(
            checks,
            "generator rows per qid",
            not bad_qids,
            (
                f"all qids have {EXPECTED_GENERATORS_PER_QID} rows"
                if not bad_qids
                else f"{len(bad_qids)} qids do not have {EXPECTED_GENERATORS_PER_QID} rows"
            ),
        )
    else:
        _record(checks, "qid count", False, "missing qid column")
        _record(checks, "generator rows per qid", False, "missing qid column")

    for column in BINARY_LABEL_COLUMNS:
        if column in columns:
            _record(
                checks,
                f"{column} binary labels",
                _valid_values(rows, column, {0, 1}),
                "allowed values: 0, 1",
            )

    for column in TERNARY_LABEL_COLUMNS:
        if column in columns:
            _record(
                checks,
                f"{column} label values",
                _valid_values(rows, column, {-1, 0, 1}),
                "allowed values: -1, 0, 1",
            )

    forbidden_columns = [
        column
        for column in columns
        if any(pattern in column for pattern in FORBIDDEN_PUBLIC_COLUMN_PATTERNS)
    ]
    _record(
        checks,
        "no P2B/P3B public columns",
        not forbidden_columns,
        "none found" if not forbidden_columns else ", ".join(forbidden_columns),
    )

    print_summary(checks)
    return 0 if all(passed for _, passed, _ in checks) else 1


def print_summary(checks: list[tuple[str, bool, str]]) -> None:
    print("Input validation summary")
    print("=" * 80)
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
    print("=" * 80)
    if all(passed for _, passed, _ in checks):
        print("PASS: release inputs are valid.")
    else:
        print("FAIL: release inputs did not pass validation.")


if __name__ == "__main__":
    raise SystemExit(main())
