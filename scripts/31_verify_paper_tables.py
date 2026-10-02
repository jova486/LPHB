#!/usr/bin/env python
"""Verify generated release-analysis tables against paper-extracted values."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIR = REPO_ROOT / "results" / "expected"
AUDIT_DIR = REPO_ROOT / "audits" / "table_comparison"

TOL = 0.0005

TABLE_SPECS = {
    "iaa": {
        "expected": "iaa_expected.csv",
        "computed": "audits/iaa/inter_annotator_agreement_recomputed.csv",
        "key_cols": ["dataset"],
        "value_cols": ["n", "disagreements", "agreement", "kappa", "ci_low", "ci_high"],
        "int_cols": ["n", "disagreements"],
    },
    "per_dataset_gpt5mini": {
        "expected": "per_dataset_gpt5mini_expected.csv",
        "computed": "audits/per_dataset_gpt5mini/per_dataset_gpt5mini_kappa.csv",
        "key_cols": ["generator", "dataset", "prompt"],
        "value_cols": ["kappa", "n_valid", "n_total"],
        "int_cols": ["n_valid", "n_total"],
    },
    "prompt_ablation": {
        "expected": "prompt_ablation_expected.csv",
        "computed": "audits/prompt_ablation/prompt_ablation_long.csv",
        "key_cols": ["judge", "model", "prompt"],
        "value_cols": ["kappa", "bias", "fp", "fn", "err", "fp_pct", "n_valid", "n_total"],
        "int_cols": ["fp", "fn", "err", "n_valid", "n_total"],
    },
    # Table 1 (main text) is a compact judge x generator layout of the same
    # judge/model/prompt rows as Table 11; both tables are built from the
    # same audits/prompt_ablation/prompt_ablation_long.csv (see
    # run_prompt_ablation), so Table 1 is checked against the same
    # Table-11-derived expected values rather than a second transcription.
    "prompt_ablation_wide": {
        "expected": "prompt_ablation_expected.csv",
        "computed": "audits/prompt_ablation/prompt_ablation_long.csv",
        "key_cols": ["judge", "model", "prompt"],
        "value_cols": ["kappa", "bias", "fp", "fn", "err", "fp_pct", "n_valid", "n_total"],
        "int_cols": ["fp", "fn", "err", "n_valid", "n_total"],
    },
    "lexical_nli": {
        "expected": "lexical_nli_expected.csv",
        "computed": "audits/lexical_nli/table_results_lexical_nli.csv",
        "key_cols": ["metric", "model"],
        "value_cols": ["kappa", "bias", "fp", "fn", "err", "fp_pct", "n_valid", "n_total"],
        "int_cols": ["fp", "fn", "err", "n_valid", "n_total"],
    },
    # Table 10.
    "new_metrics": {
        "expected": "new_metrics_expected.csv",
        "computed": "audits/new_metrics/new_metric_audit.csv",
        "key_cols": ["metric", "model"],
        "value_cols": ["kappa", "bias", "fp", "fn", "err", "fp_pct", "n_valid", "n_total"],
        "int_cols": ["fp", "fn", "err", "n_valid", "n_total"],
    },
    "bootstrap_mcnemar": {
        "expected": "bootstrap_mcnemar_expected.csv",
        "computed": "audits/bootstrap_mcnemar/bootstrap_delta_mcnemar.csv",
        "key_cols": ["model", "prompt_a", "prompt_b"],
        "value_cols": ["kappa_a", "kappa_b", "delta_kappa", "delta_ci_low", "delta_ci_high", "mcnemar_p_exact"],
        "int_cols": [],
    },
    # Figure 2. audits/kappa_matrix/kappa_matrix_avg.csv is a square matrix
    # (methods x methods), not a long table, so it is handled by the "matrix"
    # kind below rather than key_cols/value_cols. The figure prints only two
    # decimals per cell, so this table uses a wider fixed tolerance than the
    # decimal-derived default.
    "kappa_matrix": {
        "expected": "kappa_matrix_expected.csv",
        "computed": "audits/kappa_matrix/kappa_matrix_avg.csv",
        "kind": "matrix",
        "tol": 0.006,
    },
    # Table 8.
    "factorial_agreement": {
        "expected": "factorial_agreement_expected.csv",
        "computed": "audits/factorial_prompt_run/gpt5mini_factorial_v3/agreement_summary.csv",
        "key_cols": ["prompt", "generator"],
        "value_cols": ["kappa", "bias", "fp", "fn", "fpr"],
        "int_cols": ["fp", "fn"],
    },
    # Table 9.
    "factorial_contrasts": {
        "expected": "factorial_contrasts_expected.csv",
        "computed": "audits/factorial_prompt_run/gpt5mini_factorial_v3/paired_contrasts.csv",
        "key_cols": ["prompt_a", "prompt_b"],
        "value_cols": ["difference_pct", "kappa_a", "kappa_b", "delta_kappa", "ci_low", "ci_high"],
        "int_cols": [],
    },
    # Appendix E / Section 5.1: extraction/exclusion counts and FP-rate
    # reduction. mcnemar_p_exact is checked separately below with a tight
    # tolerance -- the decimal-count tolerance derived from "8.23e-10" is
    # meaningless at that magnitude.
    "elaboration_summary": {
        "expected": "elaboration_summary_expected.csv",
        "computed": "audits/elaboration_core_only/gpt56_core_extraction_v1/elaboration_summary.csv",
        "key_cols": ["key"],
        "value_cols": [
            "n_extractions", "n_excluded", "n_paired", "paired_pct",
            "original_fp", "core_fp", "original_fpr_pct", "core_fpr_pct",
            "absolute_reduction_pct", "ci_low_pct", "ci_high_pct",
            "relative_reduction_pct", "n_fp_to_tn", "n_tn_to_fp",
            "n_fp_stays_fp", "n_tn_stays_tn",
        ],
        "int_cols": [
            "n_extractions", "n_excluded", "n_paired", "original_fp", "core_fp",
            "n_fp_to_tn", "n_tn_to_fp", "n_fp_stays_fp", "n_tn_stays_tn",
        ],
    },
    "elaboration_mcnemar": {
        "expected": "elaboration_mcnemar_expected.csv",
        "computed": "audits/elaboration_core_only/gpt56_core_extraction_v1/elaboration_summary.csv",
        "key_cols": ["key"],
        "value_cols": ["mcnemar_p_exact"],
        "int_cols": [],
        "tol": 5e-13,
    },
    # Section 5.1: answer length reduction, original vs core-only.
    "answer_length_summary": {
        "expected": "answer_length_expected.csv",
        "computed": "audits/elaboration_core_only/gpt56_core_extraction_v1/p1_original_vs_core_b4matched/answer_length_summary.csv",
        "key_cols": ["key"],
        "value_cols": [
            "n", "original_mean_words", "original_median_words",
            "core_mean_words", "core_median_words",
            "median_word_percent_removed", "overall_word_percent_removed",
        ],
        "int_cols": ["n", "original_median_words", "core_median_words"],
    },
}

def row_key(row: dict[str, object], key_cols: list[str]) -> str:
    return normalize_key("|".join(str(row[col]) for col in key_cols))


def normalize_key(value: str) -> str:
    text = str(value)
    text = text.replace("$", "")
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    text = " ".join(text.split())
    return text


def normalize_expected(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    rows: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"row_key", "column", "expected"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            value_type = row.get("value_type", "float") or "float"
            rows[(normalize_key(row["row_key"]), row["column"])] = (row["expected"], value_type)
    return rows


def normalize_computed(spec: dict[str, object]) -> dict[tuple[str, str], str]:
    path = REPO_ROOT / str(spec["computed"])
    if not path.exists():
        return {}
    df = pd.read_csv(path)

    if spec.get("kind") == "matrix":
        return normalize_computed_matrix(df)

    key_cols = list(spec["key_cols"])
    value_cols = list(spec["value_cols"])
    missing = [col for col in key_cols + value_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    rows: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        key = row_key(row, key_cols)
        for col in value_cols:
            rows[(key, col)] = str(row[col])
    return rows


def normalize_computed_matrix(df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Flatten a square methods x methods kappa matrix into row_key="a|b" pairs.

    The first (unnamed) column holds the row method label; every other
    column is a method label too. Both already match the paper's Figure 2
    labels directly (src/release_analysis.py's MATRIX_METHODS is paper-facing),
    so no label translation is needed here.
    """
    index_col = df.columns[0]
    methods = [c for c in df.columns if c != index_col]

    rows: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        method_a = str(row[index_col])
        for method_b in methods:
            key = row_key({"pair": f"{method_a}|{method_b}"}, ["pair"])
            rows[(key, "kappa")] = str(row[method_b])
    return rows


def parse_number(value: str) -> float:
    text = str(value).strip()
    text = text.replace("$", "").replace(",", "")
    text = text.replace("\\textbf{", "").replace("}", "")
    text = text.replace("+", "")
    if text.startswith("<"):
        text = text[1:]
    return float(text)


def tolerance_for_expected(expected: str, value_type: str) -> float:
    if value_type == "int":
        return 0.0
    text = str(expected).strip()
    if text.startswith("<"):
        return parse_number(text)
    match = None
    for match in __import__("re").finditer(r"\d+\.(\d+)", text):
        pass
    if match is None:
        return TOL
    decimals = len(match.group(1))
    return 0.5 * (10 ** (-decimals))


def compare_value(
    expected: str, computed: str, value_type: str, tol_override: float | None = None
) -> tuple[str, float]:
    exp = parse_number(expected)
    got = parse_number(computed)
    diff = abs(exp - got)
    if str(expected).strip().startswith("<"):
        return ("ROUNDING_ONLY" if got < exp else "DIFFER", diff)
    if value_type == "int":
        return ("MATCH" if int(round(exp)) == int(round(got)) else "DIFFER", diff)
    tol = tolerance_for_expected(expected, value_type) if tol_override is None else tol_override
    if diff == 0:
        return "MATCH", diff
    if diff <= tol:
        return "ROUNDING_ONLY", diff
    return "DIFFER", diff


def add_diff(
    diffs: list[dict[str, object]],
    table: str,
    row_key_value: str,
    column: str,
    expected: str,
    computed: str,
    abs_diff: str | float,
    status: str,
    severity: str,
    recommendation: str,
) -> None:
    diffs.append(
        {
            "table": table,
            "row_key": row_key_value,
            "column": column,
            "expected": expected,
            "computed": computed,
            "abs_diff": abs_diff,
            "status": status,
            "severity": severity,
            "recommendation": recommendation,
        }
    )


def write_outputs(diffs: list[dict[str, object]]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = AUDIT_DIR / "table_value_diffs.csv"
    md_path = AUDIT_DIR / "table_value_diffs.md"
    fieldnames = [
        "table",
        "row_key",
        "column",
        "expected",
        "computed",
        "abs_diff",
        "status",
        "severity",
        "recommendation",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diffs)

    failures = [d for d in diffs if d["status"] in {"DIFFER", "MISSING_EXPECTED", "MISSING_COMPUTED"}]
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Table Value Diffs\n\n")
        handle.write(f"Total comparisons/findings: {len(diffs)}\n\n")
        handle.write(f"Blocking findings: {len(failures)}\n\n")
        by_status: dict[str, int] = {}
        for diff in diffs:
            by_status[str(diff["status"])] = by_status.get(str(diff["status"]), 0) + 1
        for status, count in sorted(by_status.items()):
            handle.write(f"- {status}: {count}\n")
        handle.write("\n")
        for diff in diffs:
            if diff["status"] == "MATCH":
                continue
            handle.write(
                f"- `{diff['table']}` `{diff['row_key']}` `{diff['column']}`: "
                f"{diff['status']} expected={diff['expected']} computed={diff['computed']} "
                f"abs_diff={diff['abs_diff']}. {diff['recommendation']}\n"
            )


def main() -> int:
    diffs: list[dict[str, object]] = []

    for table, spec in TABLE_SPECS.items():
        expected_path = EXPECTED_DIR / str(spec["expected"])
        computed_path = REPO_ROOT / str(spec["computed"])

        if not expected_path.exists():
            add_diff(
                diffs,
                table,
                "*",
                "*",
                str(expected_path.relative_to(REPO_ROOT)),
                "",
                "",
                "MISSING_EXPECTED",
                "HIGH",
                "Extract expected values from the companion paper; do not infer them from generated outputs.",
            )
            continue

        if not computed_path.exists():
            add_diff(
                diffs,
                table,
                "*",
                "*",
                "",
                str(computed_path.relative_to(REPO_ROOT)),
                "",
                "MISSING_COMPUTED",
                "HIGH",
                "Run scripts/20_run_release_analyses.py to generate computed outputs.",
            )
            continue

        expected = normalize_expected(expected_path)
        computed = normalize_computed(spec)

        for key, (expected_value, value_type) in expected.items():
            if key not in computed:
                add_diff(
                    diffs,
                    table,
                    key[0],
                    key[1],
                    expected_value,
                    "",
                    "",
                    "MISSING_COMPUTED",
                    "HIGH",
                    "Computed output lacks a paper-reported row or column.",
                )
                continue
            try:
                status, abs_diff = compare_value(
                    expected_value, computed[key], value_type, spec.get("tol")
                )
            except Exception as exc:
                add_diff(
                    diffs,
                    table,
                    key[0],
                    key[1],
                    expected_value,
                    computed[key],
                    "",
                    "NEEDS_MANUAL_CHECK",
                    "MEDIUM",
                    f"Could not parse value: {exc}",
                )
                continue
            severity = "LOW" if status in {"MATCH", "ROUNDING_ONLY"} else "HIGH"
            recommendation = "No change." if status in {"MATCH", "ROUNDING_ONLY"} else "Investigate before changing reported values."
            add_diff(
                diffs,
                table,
                key[0],
                key[1],
                expected_value,
                computed[key],
                f"{abs_diff:.12g}",
                status,
                severity,
                recommendation,
            )

    write_outputs(diffs)

    blocking = [d for d in diffs if d["status"] in {"DIFFER", "MISSING_EXPECTED", "MISSING_COMPUTED"}]
    print(f"Wrote {AUDIT_DIR / 'table_value_diffs.csv'}")
    print(f"Wrote {AUDIT_DIR / 'table_value_diffs.md'}")
    print(f"Blocking findings: {len(blocking)}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
