"""Standalone Level 1 release analyses extracted from Section C."""

from __future__ import annotations

import difflib
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import AUDITS_DIR, FIGURES_DIR, MASTER_CSV, PROMPTS_DIR, TABLES_DIR

TARGET_MODELS = ["llama3-8b", "gemma-2-9b", "mistral-7b"]
TARGET_DATASETS = ["triviaqa", "hotpotqa", "truthfulqa"]
PROMPTS = ["p1", "p2", "p3"]
HUMAN_COL = "annotator1_label"

GEN_LABELS = {
    "llama3-8b": "L",
    "gemma-2-9b": "G",
    "mistral-7b": "M",
}

DATASET_DISPLAY = {
    "triviaqa": "TriviaQA",
    "truthfulqa": "TruthfulQA",
    "hotpotqa": "HotpotQA",
    "overall": "Overall",
}

# Display names are paper-facing (Tables 1 and 11): the "Local judges"
# group heading carries the local/API distinction, so it is not repeated in
# every judge name. The paper never prints "(local)" anywhere.
JUDGES = [
    ("GPT-5-mini", "gpt_5_mini", "api"),
    ("GPT-5-nano", "gpt_5_nano", "api"),
    ("GPT-5.4", "gpt_5_4", "api"),
    ("Claude Opus 4.7", "claude_opus_4_7", "api"),
    ("Gemma-2-9B", "gemma_2_9b_local", "local"),
    ("Qwen2.5-7B", "qwen2_5_7b_local", "local"),
    ("Llama-3-8B", "llama_3_8b_local", "local"),
]

# Table 1/11 "Gen" column overrides for local judges scoring their own
# generator's outputs, keyed by judge prefix then generator model name
# (e.g. Gemma-2-9B judging gemma-2-9b outputs prints "G (self)", not "G").
PROMPT_ABLATION_SELF_OVERRIDE = {
    "gemma_2_9b_local": {"gemma-2-9b": "G (self)"},
    "llama_3_8b_local": {"llama3-8b": "L (self)"},
}


def ensure_dirs() -> None:
    for path in [AUDITS_DIR, TABLES_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib_cache() -> None:
    cache_dir = Path("audits") / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def read_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_CSV)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def validate_master_basic(df: pd.DataFrame) -> None:
    require_columns(df, ["model", "dataset", HUMAN_COL])
    if len(df) != 900:
        raise ValueError(f"Expected 900 rows, got {len(df)}")
    if not df[HUMAN_COL].isin([0, 1]).all():
        raise ValueError(f"Invalid values in {HUMAN_COL}")
    for model_name in TARGET_MODELS:
        n = int((df["model"] == model_name).sum())
        if n != 300:
            raise ValueError(f"Expected 300 rows for {model_name}, got {n}")


def cohen_kappa_binary(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("Kappa inputs must have equal length")
    if len(y_true) == 0:
        return float("nan")

    labels = [0, 1]
    cm = np.zeros((2, 2), dtype=float)
    for a, b in zip(y_true, y_pred):
        if a in labels and b in labels:
            cm[int(a), int(b)] += 1

    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.dot(row, col) / (n * n))
    if abs(1.0 - pe) < 1e-15:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def metric_row(df: pd.DataFrame, pred_col: str, require_complete: bool = False) -> dict:
    require_columns(df, [HUMAN_COL, pred_col])
    yh_all = pd.to_numeric(df[HUMAN_COL], errors="coerce")
    yp_all = pd.to_numeric(df[pred_col], errors="coerce")
    mask = yh_all.isin([0, 1]) & yp_all.isin([0, 1])
    n_total = len(df)
    n_valid = int(mask.sum())
    if require_complete and n_valid != n_total:
        raise ValueError(f"{pred_col} has invalid labels: valid={n_valid}/{n_total}")
    if n_valid < 5:
        raise ValueError(f"Too few valid rows for {pred_col}: {n_valid}/{n_total}")
    yh = yh_all.loc[mask].astype(int).to_numpy()
    yp = yp_all.loc[mask].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_counts(yh, yp)
    err = fp + fn
    return {
        "kappa": cohen_kappa_binary(yh, yp),
        "bias": float(yp.mean() - yh.mean()),
        "fp": fp,
        "fn": fn,
        "err": err,
        "fp_pct": float(100.0 * fp / max(err, 1)),
        "n_valid": n_valid,
        "n_total": n_total,
        "has_refusal": bool(n_valid < n_total),
    }


def fmt_bias(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"$+${abs(value):.3f}" if value >= 0 else f"$-${abs(value):.3f}"


def fmt_kappa(value: float, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def assert_no_internal_prompt_names(text: str | list[str]) -> None:
    values = [text] if isinstance(text, str) else list(text)
    bad = [v for v in values if "p2b" in str(v).lower() or "p3b" in str(v).lower()]
    if bad:
        raise ValueError(f"Internal prompt names leaked into output: {bad}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    assert_no_internal_prompt_names(text)


def bootstrap_kappa_ci(y_true, y_pred, n_boot: int, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        kappa = cohen_kappa_binary(y_true[idx], y_pred[idx])
        if np.isfinite(kappa):
            vals.append(kappa)
    if not vals:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi), len(vals)

def bootstrap_kappa_ci_clustered(
    y_true,
    y_pred,
    groups,
    n_boot: int,
    seed: int,
) -> tuple[float, float, int]:
    """Question-level bootstrap for Cohen's kappa.

    The doubly annotated subset contains three generator responses per
    question, which are not independent observations. Resampling rows would
    understate uncertainty, so questions are resampled instead and each
    question's rows are kept together.

    Do not merge this with bootstrap_delta_kappa_clustered. That function
    resamples groups in pd.unique's first-occurrence order (to reproduce the
    exact RNG call sequence of the notebook cell it was ported from), while
    this one resamples np.unique's sorted order. The two orderings draw a
    different sequence of groups from the same seed, so unifying them onto
    one ordering would silently change the published confidence intervals
    (IAA, Table 3) even though both orderings are statistically valid.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}

    vals = []
    for _ in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[group] for group in picked])
        kappa = cohen_kappa_binary(y_true[idx], y_pred[idx])
        if np.isfinite(kappa):
            vals.append(kappa)

    if not vals:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi), len(vals)

def run_iaa() -> list[Path]:
    ensure_dirs()
    out_dir = AUDITS_DIR / "iaa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "inter_annotator_agreement_recomputed.csv"
    out_tex_audit = out_dir / "table_iaa_recomputed.tex"
    out_tex = TABLES_DIR / "table_iaa_recomputed.tex"

    df = read_master()
    require_columns(df, ["qid", "dataset", HUMAN_COL, "annotator2_label"])
    df2 = df[df["annotator2_label"].isin([0, 1]) & df[HUMAN_COL].isin([0, 1])].copy()

    rows = []
    for dataset in ["triviaqa", "truthfulqa", "hotpotqa", "overall"]:
        sub = df2 if dataset == "overall" else df2[df2["dataset"] == dataset]
        y1 = sub[HUMAN_COL].astype(int).to_numpy()
        y2 = sub["annotator2_label"].astype(int).to_numpy()
        agree = int((y1 == y2).sum())
        lo, hi, n_boot = bootstrap_kappa_ci_clustered(
            y1,
            y2,
            sub["qid"].to_numpy(),
            n_boot=1000,
            seed=42,
        )
        rows.append({
            "dataset": dataset,
            "n": len(sub),
            "disagreements": int(len(sub) - agree),
            "agreement": float(agree / len(sub)),
            "kappa": cohen_kappa_binary(y1, y2),
            "ci_low": lo,
            "ci_high": hi,
            "n_boot_valid": n_boot,
        })

    iaa = pd.DataFrame(rows)
    iaa.to_csv(out_csv, index=False)
    latex = """\\begin{table}[t]
\\centering
\\caption{Inter-annotator agreement on the 300-item doubly annotated subset. Confidence intervals are 95\\% question-level bootstrap intervals for Cohen's $\\kappa$ with 1000 resamples, keeping the three generator responses to each question together.}
\\label{tab:iaa}
\\small
\\begin{tabular}{lrrrrrr}
\\toprule
Dataset & $n$ & Disagr. & Agree. & $\\kappa$ & CI low & CI high \\\\
\\midrule
"""
    for _, row in iaa.iterrows():
        latex += (
            f"{DATASET_DISPLAY[row['dataset']]} & {int(row['n'])} & "
            f"{int(row['disagreements'])} & {row['agreement']:.3f} & "
            f"{row['kappa']:.3f} & {row['ci_low']:.3f} & {row['ci_high']:.3f} \\\\\n"
        )
        if row["dataset"] == "hotpotqa":
            latex += "\\midrule\n"
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    write_text(out_tex, latex)
    write_text(out_tex_audit, latex)
    return [out_csv, out_tex_audit, out_tex]


def prompt_ablation_rows(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for judge_name, prefix, family in JUDGES:
        gen_override = PROMPT_ABLATION_SELF_OVERRIDE.get(prefix, {})
        for model_name in TARGET_MODELS:
            sub = master[master["model"] == model_name].copy()
            gen_label = gen_override.get(model_name, GEN_LABELS[model_name])
            block = []
            for prompt in PROMPTS:
                pred_col = f"label_{prefix}_{prompt}"
                stats = metric_row(sub, pred_col)
                row = {
                    "judge": judge_name,
                    "group": "Local judges" if family == "local" else "API judges",
                    "clean_prefix": prefix,
                    "prompt": prompt,
                    "model": model_name,
                    "gen": gen_label,
                    **stats,
                }
                block.append(row)
            best_kappa = max(row["kappa"] for row in block)
            for row in block:
                row["best"] = abs(row["kappa"] - best_kappa) < 1e-12
                rows.append(row)
    return pd.DataFrame(rows)


def run_prompt_ablation() -> list[Path]:
    ensure_dirs()
    audit_dir = AUDITS_DIR / "prompt_ablation"
    audit_dir.mkdir(parents=True, exist_ok=True)
    wide_out = TABLES_DIR / "table_results_prompt_wide.tex"
    long_out = TABLES_DIR / "table_results_full_appendix.tex"
    csv_out = audit_dir / "prompt_ablation_long.csv"
    master = read_master()
    validate_master_basic(master)
    required = [f"label_{prefix}_{prompt}" for _, prefix, _ in JUDGES for prompt in PROMPTS]
    require_columns(master, required)
    rows = prompt_ablation_rows(master)
    rows.to_csv(csv_out, index=False)

    latex_wide = """\\begin{table*}[t!]
  \\centering
  \\caption{Prompt-ablation results against human labels for each judge--generator pair. Each row fixes the judge and generator; columns compare p1, p2, and p3. Bias is positive when the automatic labeler over-labels hallucinations relative to humans.}
  \\label{tab:results_prompt}
  \\scriptsize
  \\setlength{\\tabcolsep}{3pt}
  \\begin{tabular}{llrrrrrrrrrrrr}
    \\toprule
    & & \\multicolumn{4}{c}{p1} & \\multicolumn{4}{c}{p2} & \\multicolumn{4}{c}{p3} \\\\
    \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}
    Judge & Gen & $\\kappa$ & Bias & FP/FN & FP\\% & $\\kappa$ & Bias & FP/FN & FP\\% & $\\kappa$ & Bias & FP/FN & FP\\% \\\\
    \\midrule
"""
    for judge_name, _, _ in JUDGES:
        for model_name in TARGET_MODELS:
            sub = rows[(rows["judge"] == judge_name) & (rows["model"] == model_name)]
            gen_label = sub.iloc[0]["gen"]
            cells = []
            for prompt in PROMPTS:
                row = sub[sub["prompt"] == prompt].iloc[0]
                cells.extend([
                    fmt_kappa(row["kappa"], bool(row["best"])),
                    fmt_bias(row["bias"]),
                    f"{int(row['fp'])}/{int(row['fn'])}",
                    f"{row['fp_pct']:.1f}",
                ])
            latex_wide += (
                f"    {judge_name} & {gen_label} & "
                + " & ".join(cells)
                + " \\\\\n"
            )
    latex_wide += """    \\bottomrule
  \\end{tabular}
\\end{table*}
"""

    latex_long = """\\begin{table*}[p]
  \\centering
  \\caption{Full prompt-ablation results against human labels for each judge--generator--prompt combination.}
  \\label{tab:results_full}
  \\scriptsize
  \\begin{tabular}{lllrrrrrr}
    \\toprule
    Judge & Prompt & Gen & $\\kappa$ & Bias & FP & FN & Err & FP\\% \\\\
    \\midrule
"""
    for _, row in rows.iterrows():
        latex_long += (
            f"    {row['judge']} & {row['prompt']} & {row['gen']} & "
            f"{fmt_kappa(row['kappa'], bool(row['best']))} & {fmt_bias(row['bias'])} & "
            f"{int(row['fp'])} & {int(row['fn'])} & {int(row['err'])} & {row['fp_pct']:.1f} \\\\\n"
        )
    latex_long += """    \\bottomrule
  \\end{tabular}
\\end{table*}
"""
    write_text(wide_out, latex_wide)
    write_text(long_out, latex_long)
    return [csv_out, wide_out, long_out]


BASELINE_SPECS = [
    (r"ROUGE-L (oracle-$\tau$)", "ROUGE-L", "rougeL_label_oracle", ("rougeL_tau_oracle",), "single"),
    (r"ROUGE-L (CV-$\tau$)", "ROUGE-L", "rougeL_label_cv", ("rougeL_tau_cv_mean", "rougeL_tau_cv_min", "rougeL_tau_cv_max"), "cv"),
    (r"BERTScore (oracle-$\tau$)", "BERTScore", "bertscore_label_oracle", ("bertscore_tau_oracle",), "single"),
    (r"BERTScore (CV-$\tau$)", "BERTScore", "bertscore_label_cv", ("bertscore_tau_cv_mean", "bertscore_tau_cv_min", "bertscore_tau_cv_max"), "cv"),
    (r"NLI (T5-11B) strict", "NLI", "nli_label_strict", ("nli_tau_strict",), "single"),
    (r"NLI (T5-11B) best-$\tau$", "NLI", "nli_label_best_tau", ("nli_tau_best",), "single"),
]


def unique_tau(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df[column], errors="coerce").dropna().round(10).unique()
    if len(values) != 1:
        raise ValueError(f"Expected one unique tau in {column}, found {values}")
    return float(values[0])


def format_tau(df: pd.DataFrame, cols: tuple[str, ...], tau_type: str) -> str:
    if tau_type == "single":
        return f"{unique_tau(df, cols[0]):.2f}"
    mean_tau, min_tau, max_tau = [unique_tau(df, col) for col in cols]
    return f"{mean_tau:.2f} [{min_tau:.2f}, {max_tau:.2f}]"


def run_lexical_nli() -> list[Path]:
    ensure_dirs()
    audit_dir = AUDITS_DIR / "lexical_nli"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_csv = audit_dir / "table_results_lexical_nli.csv"
    out_tex = TABLES_DIR / "table_results_lexical_nli.tex"
    master = read_master()
    validate_master_basic(master)
    required = [spec[2] for spec in BASELINE_SPECS] + [c for spec in BASELINE_SPECS for c in spec[3]]
    require_columns(master, required)
    rows = []
    for metric, group, label_col, tau_cols, tau_type in BASELINE_SPECS:
        for model_name in TARGET_MODELS:
            sub = master[master["model"] == model_name].copy()
            rows.append({
                "group": group,
                "metric": metric,
                "label_col": label_col,
                "model": model_name,
                "gen": GEN_LABELS[model_name],
                "tau": format_tau(sub, tau_cols, tau_type),
                **metric_row(sub, label_col, require_complete=True),
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(out_csv, index=False)
    latex = """\\begin{table*}[t]
  \\centering
  \\caption{Lexical similarity metric and NLI baselines against human labels.}
  \\label{tab:results_lexical}
  \\scriptsize
  \\begin{tabular}{lllrrrrr}
    \\toprule
    Metric & Gen & $\\tau$ & $\\kappa$ & Bias & FP & FN & FP\\% \\\\
    \\midrule
"""
    for _, row in audit.iterrows():
        latex += (
            f"    {row['metric']} & {row['gen']} & {row['tau']} & {row['kappa']:.3f} & "
            f"{fmt_bias(row['bias'])} & {int(row['fp'])} & {int(row['fn'])} & {row['fp_pct']:.1f} \\\\\n"
        )
    latex += """    \\bottomrule
  \\end{tabular}
\\end{table*}
"""
    write_text(out_tex, latex)
    return [out_csv, out_tex]


# Table 10. Ported from the analysis half of notebook CELL B9 -- the scores
# (meteor, bartscore_r2h, bartscore_h2r) and their oracle/CV thresholds and
# labels are already columns in the release master, produced by CELL B9's
# threshold search over METEOR/BARTScore-CNN scores. This only reads those
# stored columns, exactly as run_lexical_nli does for ROUGE-L/BERTScore/NLI;
# it does not recompute METEOR or BARTScore, so it needs no NLTK, no BART
# model, and no GPU.
NEW_METRIC_SPECS = [
    (r"METEOR (oracle-$\tau$)", "METEOR", "meteor_label_oracle", ("meteor_tau_oracle",), "single"),
    (r"METEOR (CV-$\tau$)", "METEOR", "meteor_label_cv", ("meteor_tau_cv_mean", "meteor_tau_cv_min", "meteor_tau_cv_max"), "cv"),
    (r"BARTScore ref$\to$hyp (oracle-$\tau$)", "BARTScore-CNN", "bartscore_r2h_label_oracle", ("bartscore_r2h_tau_oracle",), "single"),
    (r"BARTScore ref$\to$hyp (CV-$\tau$)", "BARTScore-CNN", "bartscore_r2h_label_cv", ("bartscore_r2h_tau_cv_mean", "bartscore_r2h_tau_cv_min", "bartscore_r2h_tau_cv_max"), "cv"),
    (r"BARTScore hyp$\to$ref (oracle-$\tau$)", "BARTScore-CNN", "bartscore_h2r_label_oracle", ("bartscore_h2r_tau_oracle",), "single"),
    (r"BARTScore hyp$\to$ref (CV-$\tau$)", "BARTScore-CNN", "bartscore_h2r_label_cv", ("bartscore_h2r_tau_cv_mean", "bartscore_h2r_tau_cv_min", "bartscore_h2r_tau_cv_max"), "cv"),
]


def format_tau_signed(df: pd.DataFrame, cols: tuple[str, ...], tau_type: str) -> str:
    """Like format_tau, but renders negative values with a LaTeX math minus.

    BARTScore's threshold is an unbounded mean token log-likelihood and is
    usually negative, unlike the bounded [0, 1] tau values format_tau's other
    callers use.
    """
    def fmt(x: float) -> str:
        return f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}"

    if tau_type == "single":
        return fmt(unique_tau(df, cols[0]))
    mean_tau, min_tau, max_tau = [unique_tau(df, col) for col in cols]
    return f"{fmt(mean_tau)} [{fmt(min_tau)}, {fmt(max_tau)}]"


def run_new_metrics() -> list[Path]:
    ensure_dirs()
    audit_dir = AUDITS_DIR / "new_metrics"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_csv = audit_dir / "new_metric_audit.csv"
    out_tex = TABLES_DIR / "table_new_metrics.tex"
    master = read_master()
    validate_master_basic(master)
    required = [spec[2] for spec in NEW_METRIC_SPECS] + [c for spec in NEW_METRIC_SPECS for c in spec[3]]
    require_columns(master, required)
    rows = []
    for metric, group, label_col, tau_cols, tau_type in NEW_METRIC_SPECS:
        for model_name in TARGET_MODELS:
            sub = master[master["model"] == model_name].copy()
            rows.append({
                "group": group,
                "metric": metric,
                "label_col": label_col,
                "model": model_name,
                "gen": GEN_LABELS[model_name],
                "tau": format_tau_signed(sub, tau_cols, tau_type),
                **metric_row(sub, label_col, require_complete=True),
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(out_csv, index=False)
    latex = """\\begin{table*}[t]
  \\centering
  \\caption{METEOR and BARTScore-CNN baselines against human labels.}
  \\label{tab:results_new_metrics}
  \\scriptsize
  \\begin{tabular}{lllrrrrr}
    \\toprule
    Metric & Gen & $\\tau$ & $\\kappa$ & Bias & FP & FN & FP\\% \\\\
    \\midrule
"""
    for _, row in audit.iterrows():
        latex += (
            f"    {row['metric']} & {row['gen']} & {row['tau']} & {row['kappa']:.3f} & "
            f"{fmt_bias(row['bias'])} & {int(row['fp'])} & {int(row['fn'])} & {row['fp_pct']:.1f} \\\\\n"
        )
    latex += """    \\bottomrule
  \\end{tabular}
\\end{table*}
"""
    write_text(out_tex, latex)
    return [out_csv, out_tex]


def bootstrap_delta_ci(y_true, pred_a, pred_b, n_boot: int, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        delta = cohen_kappa_binary(y_true[idx], pred_b[idx]) - cohen_kappa_binary(y_true[idx], pred_a[idx])
        if np.isfinite(delta):
            vals.append(delta)
    if not vals:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def chi2_mcnemar_p_continuity(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    return float(math.erfc(math.sqrt(stat / 2.0)))


def run_bootstrap_mcnemar(n_boot: int = 2000) -> list[Path]:
    ensure_dirs()
    audit_dir = AUDITS_DIR / "bootstrap_mcnemar"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_kappa = audit_dir / "bootstrap_kappa_by_prompt.csv"
    out_delta = audit_dir / "bootstrap_delta_mcnemar.csv"
    out_tex = TABLES_DIR / "table_prompt_delta_bootstrap_mcnemar.tex"
    master = read_master()
    validate_master_basic(master)
    cols = [f"label_gpt_5_mini_{p}" for p in PROMPTS]
    require_columns(master, cols)
    kappa_rows = []
    delta_rows = []
    for model_name in TARGET_MODELS:
        sub = master[master["model"] == model_name].copy()
        y_true = sub[HUMAN_COL].astype(int).to_numpy()
        preds = {p: sub[f"label_gpt_5_mini_{p}"].astype(int).to_numpy() for p in PROMPTS}
        for prompt in PROMPTS:
            lo, hi, valid = bootstrap_kappa_ci(y_true, preds[prompt], n_boot=n_boot, seed=42)
            kappa_rows.append({
                "judge": "GPT-5-mini",
                "model": model_name,
                "gen": GEN_LABELS[model_name],
                "prompt": prompt,
                "n_valid": len(y_true),
                "kappa": cohen_kappa_binary(y_true, preds[prompt]),
                "ci_low": lo,
                "ci_high": hi,
                "n_boot_valid": valid,
            })
        for prompt_a, prompt_b in [("p1", "p2"), ("p1", "p3"), ("p2", "p3")]:
            pred_a = preds[prompt_a]
            pred_b = preds[prompt_b]
            lo, hi, valid = bootstrap_delta_ci(y_true, pred_a, pred_b, n_boot=n_boot, seed=42)
            a_correct = pred_a == y_true
            b_correct = pred_b == y_true
            a_wrong_b_right = int(((~a_correct) & b_correct).sum())
            a_right_b_wrong = int((a_correct & (~b_correct)).sum())
            delta_rows.append({
                "judge": "GPT-5-mini",
                "model": model_name,
                "gen": GEN_LABELS[model_name],
                "contrast": f"{prompt_a}$\\to${prompt_b}",
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "n_valid": len(y_true),
                "kappa_a": cohen_kappa_binary(y_true, pred_a),
                "kappa_b": cohen_kappa_binary(y_true, pred_b),
                "delta_kappa": cohen_kappa_binary(y_true, pred_b) - cohen_kappa_binary(y_true, pred_a),
                "delta_ci_low": lo,
                "delta_ci_high": hi,
                "delta_boot_valid": valid,
                "a_wrong_b_right": a_wrong_b_right,
                "a_right_b_wrong": a_right_b_wrong,
                "discordant": a_wrong_b_right + a_right_b_wrong,
                "mcnemar_p_exact": exact_mcnemar_p(a_wrong_b_right, a_right_b_wrong),
                "mcnemar_p_chi2_continuity": chi2_mcnemar_p_continuity(a_wrong_b_right, a_right_b_wrong),
                "delta_ci_excludes_zero": bool(lo > 0 or hi < 0),
            })
    pd.DataFrame(kappa_rows).to_csv(out_kappa, index=False)
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(out_delta, index=False)
    latex = """\\begin{table}[t]
  \\centering
  \\caption{Paired prompt comparisons for GPT-5-mini against the primary human labels.}
  \\label{tab:prompt_comparisons}
  \\small
  \\begin{tabular}{llrrrr}
    \\toprule
    Gen & Contrast & $\\kappa_a$ & $\\kappa_b$ & $\\Delta\\kappa$ [95\\% CI] & McNemar $p$ \\\\
    \\midrule
"""
    for _, row in delta.iterrows():
        p_text = "$<0.0001$" if row["mcnemar_p_exact"] < 0.0001 else f"{row['mcnemar_p_exact']:.4f}"
        latex += (
            f"    {row['gen']} & {row['contrast']} & {row['kappa_a']:.3f} & {row['kappa_b']:.3f} & "
            f"{row['delta_kappa']:+.3f} [{row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f}] & {p_text} \\\\\n"
        )
    latex += """    \\bottomrule
  \\end{tabular}
\\end{table}
"""
    write_text(out_tex, latex)
    return [out_kappa, out_delta, out_tex]


# Display labels are paper-facing (Figure 2/3 heatmap cell labels); they
# must match the figure exactly, not the abbreviated forms used elsewhere.
MATRIX_METHODS = [
    ("rougeL_label_oracle", "ROUGE-L", "lexical"),
    ("bertscore_label_oracle", "BERTScore", "lexical"),
    ("nli_label_strict", "NLI (T5-11B) strict", "faith"),
    ("nli_label_best_tau", "NLI (T5-11B) best-tau", "faith"),
    ("label_gpt_5_mini_p1", "GPT-5-mini p1", "faith"),
    ("label_gpt_5_4_p1", "GPT-5.4 p1", "faith"),
    ("label_claude_opus_4_7_p1", "Opus 4.7 p1", "faith"),
    ("label_gpt_5_mini_p2", "GPT-5-mini p2", "factual"),
    ("label_gpt_5_mini_p3", "GPT-5-mini p3", "factual"),
    ("label_gpt_5_nano_p2", "GPT-nano p2", "factual"),
    ("label_gpt_5_4_p2", "GPT-5.4 p2", "factual"),
    ("label_claude_opus_4_7_p2", "Opus 4.7 p2", "factual"),
    ("label_gemma_2_9b_local_p2", "Gemma p2 local", "local"),
    ("label_qwen2_5_7b_local_p2", "Qwen p2 local", "local"),
]


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def pairwise_matrix(df: pd.DataFrame, methods: list[tuple[str, str, str]]) -> pd.DataFrame:
    labels = [m[1] for m in methods]
    cols = [m[0] for m in methods]
    values = np.full((len(cols), len(cols)), np.nan)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            x = pd.to_numeric(df[c1], errors="coerce")
            y = pd.to_numeric(df[c2], errors="coerce")
            mask = x.isin([0, 1]) & y.isin([0, 1])
            if int(mask.sum()) >= 10:
                values[i, j] = 1.0 if i == j else cohen_kappa_binary(
                    x.loc[mask].astype(int).to_numpy(),
                    y.loc[mask].astype(int).to_numpy(),
                )
    return pd.DataFrame(values, index=labels, columns=labels)


def run_kappa_matrix() -> list[Path]:
    ensure_dirs()
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    audit_dir = AUDITS_DIR / "kappa_matrix"
    audit_dir.mkdir(parents=True, exist_ok=True)
    master = read_master()
    validate_master_basic(master)
    require_columns(master, [m[0] for m in MATRIX_METHODS])
    methods_csv = audit_dir / "kappa_matrix_methods.csv"
    pd.DataFrame([{"column": c, "display_label": l, "group": g} for c, l, g in MATRIX_METHODS]).to_csv(methods_csv, index=False)
    outputs = [methods_csv]
    matrices = {}
    for model_name in TARGET_MODELS:
        matrix = pairwise_matrix(master[master["model"] == model_name], MATRIX_METHODS)
        matrices[model_name] = matrix
        out_csv = audit_dir / f"kappa_matrix_{safe_filename(model_name)}.csv"
        matrix.to_csv(out_csv)
        outputs.append(out_csv)
    avg = pd.DataFrame(
        np.nanmean(np.stack([matrices[m].to_numpy() for m in TARGET_MODELS]), axis=0),
        index=matrices[TARGET_MODELS[0]].index,
        columns=matrices[TARGET_MODELS[0]].columns,
    )
    avg_csv = audit_dir / "kappa_matrix_avg.csv"
    avg.to_csv(avg_csv)
    outputs.append(avg_csv)

    group_rows = []
    for g1 in sorted({g for _, _, g in MATRIX_METHODS}):
        labels1 = [l for _, l, g in MATRIX_METHODS if g == g1]
        for g2 in sorted({g for _, _, g in MATRIX_METHODS}):
            labels2 = [l for _, l, g in MATRIX_METHODS if g == g2]
            vals = []
            for i, l1 in enumerate(labels1):
                for j, l2 in enumerate(labels2):
                    if l1 == l2 or (g1 == g2 and j <= i):
                        continue
                    value = avg.loc[l1, l2]
                    if np.isfinite(value):
                        vals.append(float(value))
            if vals:
                group_rows.append({"group_1": g1, "group_2": g2, "mean_kappa": np.mean(vals), "min_kappa": np.min(vals), "max_kappa": np.max(vals), "n_pairs": len(vals)})
    group_csv = audit_dir / "kappa_matrix_group_summary.csv"
    pd.DataFrame(group_rows).to_csv(group_csv, index=False)
    outputs.append(group_csv)

    def plot(matrix: pd.DataFrame, path: Path, title: str | None = None) -> None:
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(matrix.to_numpy(), vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
        if title:
            ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, label="Cohen's kappa")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

    plot(avg, FIGURES_DIR / "kappa_matrix_avg.pdf")
    outputs.append(FIGURES_DIR / "kappa_matrix_avg.pdf")

    for model_name, matrix in matrices.items():
        fig_path = FIGURES_DIR / f"kappa_matrix_{safe_filename(model_name)}.pdf"
        plot(matrix, fig_path, model_name)
        outputs.append(fig_path)

    # Combined three-panel figure, built once.
    panel_paths = [
        FIGURES_DIR / "kappa_matrix_all_generators.pdf",
        FIGURES_DIR / "kappa_matrix_per_generator.pdf",
    ]

    fig = plt.figure(figsize=(34, 10), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        len(TARGET_MODELS) + 1,
        width_ratios=[1.0, 1.0, 1.0, 0.035],
        wspace=0.12,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(len(TARGET_MODELS))]
    cax = fig.add_subplot(gs[0, len(TARGET_MODELS)])

    last_image = None

    for ax, model_name in zip(axes, TARGET_MODELS):
        matrix = matrices[model_name]
        last_image = ax.imshow(matrix.to_numpy(), vmin=0.0, vmax=1.0, cmap="RdYlGn")

        ax.set_title(model_name)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(last_image, cax=cax, label="Cohen's kappa")

    for panel_path in panel_paths:
        fig.savefig(panel_path, dpi=300)
        outputs.append(panel_path)

    plt.close(fig)

    return outputs


def run_error_direction_space() -> list[Path]:
    ensure_dirs()
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    audit_dir = AUDITS_DIR / "error_direction_space"
    audit_dir.mkdir(parents=True, exist_ok=True)
    points_csv = audit_dir / "error_direction_space_points.csv"
    traj_csv = audit_dir / "error_direction_space_trajectories.csv"
    fig_out = FIGURES_DIR / "error_direction_space.pdf"
    master = read_master()
    validate_master_basic(master)
    defs = [
        ("ROUGE-L", [("best-tau", "rougeL_label_oracle")], False),
        ("BERTScore", [("best-tau", "bertscore_label_oracle")], False),
        ("NLI (T5-11B)", [("strict", "nli_label_strict"), ("best-tau", "nli_label_best_tau")], False),
        ("G5-mini", [(p, f"label_gpt_5_mini_{p}") for p in PROMPTS], True),
        ("G5-nano", [(p, f"label_gpt_5_nano_{p}") for p in PROMPTS], True),
        ("GPT-5.4", [(p, f"label_gpt_5_4_{p}") for p in PROMPTS], True),
        ("Opus 4.7", [(p, f"label_claude_opus_4_7_{p}") for p in PROMPTS], True),
        ("Gemma (local)", [(p, f"label_gemma_2_9b_local_{p}") for p in PROMPTS], True),
        ("Qwen (local)", [(p, f"label_qwen2_5_7b_local_{p}") for p in PROMPTS], True),
        ("Llama (local)", [(p, f"label_llama_3_8b_local_{p}") for p in PROMPTS], True),
    ]
    require_columns(master, [col for _, cols, _ in defs for _, col in cols])
    points = []
    for label, cols, draw_arrow in defs:
        prev = None
        for prompt, col in cols:
            fp_rates, fn_rates, kappas = [], [], []
            for model_name in TARGET_MODELS:
                sub = master[master["model"] == model_name]
                yh = sub[HUMAN_COL].astype(int).to_numpy()
                yp = sub[col].astype(int).to_numpy()
                tn, fp, fn, tp = confusion_counts(yh, yp)
                fp_rates.append(fp / max(fp + tn, 1))
                fn_rates.append(fn / max(fn + tp, 1))
                kappas.append(cohen_kappa_binary(yh, yp))
            fp_rate = float(np.mean(fp_rates))
            fn_rate = float(np.mean(fn_rates))
            points.append({
                "label": label,
                "prompt": prompt,
                "column": col,
                "fp_rate": fp_rate,
                "fn_rate": fn_rate,
                "kappa": float(np.mean(kappas)),
                "n_valid": 900,
                "n_total": 900,
                "draw_arrow": draw_arrow,
                "prev_x": np.nan if prev is None else prev[0],
                "prev_y": np.nan if prev is None else prev[1],
                "is_frontier": label in {"GPT-5.4", "Opus 4.7"},
            })
            prev = (fp_rate, fn_rate)
    df_points = pd.DataFrame(points)
    df_points.to_csv(points_csv, index=False)
    traj_rows = []
    for label in df_points["label"].unique():
        sub = df_points[df_points["label"] == label].set_index("prompt")
        if "p1" in sub.index and "p2" in sub.index:
            p1 = sub.loc["p1"]
            p2 = sub.loc["p2"]
            traj_rows.append({"label": label, "dFP_p1_p2": p2["fp_rate"] - p1["fp_rate"], "dFN_p1_p2": p2["fn_rate"] - p1["fn_rate"], "dkappa_p1_p2": p2["kappa"] - p1["kappa"]})
    pd.DataFrame(traj_rows).to_csv(traj_csv, index=False)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })

    model_colors = {
        "G5-nano": "#4ade80",
        "G5-mini": "#10a37f",
        "GPT-5.4": "#065f46",
        "Opus 4.7": "#d4761f",
        "Gemma (local)": "#4285f4",
        "Llama (local)": "#0c2d5c",
        "Qwen (local)": "#a855f7",
        "NLI (T5-11B)": "#c0392b",
        "ROUGE-L": "#6b7280",
        "BERTScore": "#9ca3af",
    }

    prompt_markers = {
        "p1": "v",
        "p2": "o",
        "p3": "s",
        "strict": "X",
        "best-tau": "D",
    }

    labels_to_show = {
        ("GPT-5.4", "p1"),
        ("Opus 4.7", "p1"),
        ("G5-mini", "p1"),
        ("NLI (T5-11B)", "strict"),
        ("NLI (T5-11B)", "best-tau"),
        ("ROUGE-L", "best-tau"),
        ("BERTScore", "best-tau"),
        ("Llama (local)", "p1"),
        ("Gemma (local)", "p1"),
        ("Qwen (local)", "p1"),
    }

    label_offsets = {
        ("GPT-5.4", "p1"): (0.005, 0.018),
        ("Opus 4.7", "p1"): (0.015, -0.045),
        ("G5-mini", "p1"): (0.015, -0.035),
        ("NLI (T5-11B)", "strict"): (-0.15, 0.070),
        ("NLI (T5-11B)", "best-tau"): (0.018, 0.025),
        ("ROUGE-L", "best-tau"): (-0.10, 0.030),
        ("BERTScore", "best-tau"): (-0.07, 0.040),
        ("Llama (local)", "p1"): (0.015, -0.035),
        ("Gemma (local)", "p1"): (0.015, 0.020),
        ("Qwen (local)", "p1"): (0.015, 0.020),
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.35)

    for _, row in df_points.iterrows():
        if bool(row["draw_arrow"]) and np.isfinite(row["prev_x"]) and np.isfinite(row["prev_y"]):
            if bool(row["is_frontier"]):
                arrow_lw = 2.0
                arrow_alpha = 1.0
                arrow_style = "->,head_width=0.7,head_length=0.9"
            else:
                arrow_lw = 1.4
                arrow_alpha = 0.75
                arrow_style = "->,head_width=0.3,head_length=0.4"
            ax.annotate(
                "",
                xy=(row["fp_rate"], row["fn_rate"]),
                xytext=(row["prev_x"], row["prev_y"]),
                arrowprops={
                    "arrowstyle": arrow_style,
                    "color": model_colors[row["label"]],
                    "lw": arrow_lw,
                    "alpha": arrow_alpha,
                    "connectionstyle": "arc3,rad=0.06",
                },
            )

    point_size = 90
    frontier_size = 140

    for _, row in df_points.iterrows():
        if bool(row["is_frontier"]):
            size = frontier_size
            edge_color = "#222222"
            edge_lw = 1.5
        else:
            size = point_size
            edge_color = "white"
            edge_lw = 0.7
        ax.scatter(
            row["fp_rate"],
            row["fn_rate"],
            c=model_colors[row["label"]],
            marker=prompt_markers.get(row["prompt"], "o"),
            s=size,
            edgecolors=edge_color,
            linewidths=edge_lw,
            zorder=5,
            alpha=0.92,
        )

    for _, row in df_points.iterrows():
        key = (row["label"], row["prompt"])
        if key not in labels_to_show:
            continue
        dx, dy = label_offsets.get(key, (0.013, 0.013))
        tag = f" ({row['prompt']})" if row["prompt"] in ("best-tau", "strict") else f" {row['prompt']}"
        ax.annotate(
            f"{row['label']}{tag}",
            xy=(row["fp_rate"], row["fn_rate"]),
            xytext=(row["fp_rate"] + dx, row["fn_rate"] + dy),
            fontsize=7,
            color="#222222",
            ha="left",
            va="bottom",
            fontweight="bold" if bool(row["is_frontier"]) else "normal",
            arrowprops={
                "arrowstyle": "-",
                "color": "#bbbbbb",
                "lw": 0.5,
                "shrinkA": 0,
                "shrinkB": 3,
            },
        )

    model_patches = [
        mpatches.Patch(color=model_colors["G5-nano"], label="GPT-5-nano"),
        mpatches.Patch(color=model_colors["G5-mini"], label="GPT-5-mini"),
        mpatches.Patch(color=model_colors["GPT-5.4"], label="GPT-5.4 (frontier)"),
        mpatches.Patch(color=model_colors["Opus 4.7"], label="Claude Opus 4.7 (frontier)"),
        mpatches.Patch(color=model_colors["Gemma (local)"], label="Gemma-2-9B (local)"),
        mpatches.Patch(color=model_colors["Llama (local)"], label="Llama-3-8B (local)"),
        mpatches.Patch(color=model_colors["Qwen (local)"], label="Qwen2.5-7B (local)"),
        mpatches.Patch(color=model_colors["NLI (T5-11B)"], label="NLI (T5-11B)"),
        mpatches.Patch(color=model_colors["ROUGE-L"], label="ROUGE-L"),
        mpatches.Patch(color=model_colors["BERTScore"], label="BERTScore"),
    ]

    prompt_handles = [
        plt.scatter([], [], marker="v", c="gray", s=70, label="p1 (faithfulness)"),
        plt.scatter([], [], marker="o", c="gray", s=70, label="p2 (factual)"),
        plt.scatter([], [], marker="s", c="gray", s=70, label="p3 (factual ext.)"),
        plt.scatter([], [], marker="X", c="gray", s=70, label="strict (NLI)"),
        plt.scatter([], [], marker="D", c="gray", s=70, label="best-tau (oracle)"),
    ]

    leg1 = ax.legend(
        handles=model_patches,
        title="Judge / baseline (color)",
        loc="upper left",
        fontsize=7.2,
        title_fontsize=8,
        framealpha=0.92,
    )
    ax.legend(
        handles=prompt_handles,
        title="Criterion (shape)",
        loc="upper right",
        fontsize=7.2,
        title_fontsize=8,
        framealpha=0.92,
    )
    ax.add_artist(leg1)

    ax.set_xlabel("FP-rate = FP / (FP + TN)", fontsize=11)
    ax.set_ylabel("FN-rate = FN / (FN + TP)", fontsize=11)
    ax.set_title("Error-direction space of automatic label sources", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [points_csv, traj_csv, fig_out]


def run_per_dataset_gpt5mini() -> list[Path]:
    ensure_dirs()
    audit_dir = AUDITS_DIR / "per_dataset_gpt5mini"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_csv = audit_dir / "per_dataset_gpt5mini_kappa.csv"
    out_tex = TABLES_DIR / "table_per_dataset_gpt5mini_kappa.tex"
    master = read_master()
    validate_master_basic(master)
    rows = []
    for model_name in TARGET_MODELS:
        for dataset in ["hotpotqa", "triviaqa", "truthfulqa"]:
            sub = master[(master["model"] == model_name) & (master["dataset"] == dataset)]
            for prompt in ["p1", "p2"]:
                pred_col = f"label_gpt_5_mini_{prompt}"
                stats = metric_row(sub, pred_col, require_complete=True)
                rows.append({"generator": model_name, "dataset": dataset, "dataset_display": DATASET_DISPLAY[dataset], "prompt": prompt, "label_col": pred_col, "kappa": stats["kappa"], "n_valid": stats["n_valid"], "n_total": stats["n_total"]})
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    latex = """\\begin{table}[t]
  \\centering
  \\caption{Per-dataset breakdown of Cohen's $\\kappa$ agreement between the GPT-5-mini judge and human labels.}
  \\label{tab:per_dataset_gpt5mini}
  \\small
  \\begin{tabular}{lrrrrrr}
    \\toprule
    & \\multicolumn{2}{c}{HotpotQA} & \\multicolumn{2}{c}{TriviaQA} & \\multicolumn{2}{c}{TruthfulQA} \\\\
    Generator & p1 & p2 & p1 & p2 & p1 & p2 \\\\
    \\midrule
"""
    model_display = {"llama3-8b": "Llama-3-8B", "gemma-2-9b": "Gemma-2-9B", "mistral-7b": "Mistral-7B"}
    for model_name in TARGET_MODELS:
        cells = []
        for dataset in ["hotpotqa", "triviaqa", "truthfulqa"]:
            for prompt in ["p1", "p2"]:
                value = table[(table["generator"] == model_name) & (table["dataset"] == dataset) & (table["prompt"] == prompt)]["kappa"].iloc[0]
                cells.append(f"{value:.3f}")
        latex += f"    {model_display[model_name]} & " + " & ".join(cells) + " \\\\\n"
    latex += """    \\bottomrule
  \\end{tabular}
\\end{table}
"""
    write_text(out_tex, latex)
    return [out_csv, out_tex]


def run_per_dataset_prompt_ablation() -> list[Path]:
    ensure_dirs()
    out_dir = AUDITS_DIR / "per_dataset_prompt_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_long = out_dir / "per_dataset_prompt_kappa_long.csv"
    out_compact = out_dir / "per_dataset_prompt_kappa_p1_vs_p2.csv"
    master = read_master()
    validate_master_basic(master)
    rows = []
    for judge, prefix, family in JUDGES:
        for model_name in TARGET_MODELS:
            for dataset in TARGET_DATASETS:
                sub = master[(master["model"] == model_name) & (master["dataset"] == dataset)]
                for prompt in PROMPTS:
                    pred_col = f"label_{prefix}_{prompt}"
                    stats = metric_row(sub, pred_col)
                    rows.append({"judge": judge, "judge_prefix": prefix, "judge_family": family, "generator": model_name, "gen": GEN_LABELS[model_name], "dataset": dataset, "dataset_display": DATASET_DISPLAY[dataset], "prompt": prompt, "label_col": pred_col, **stats})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(out_long, index=False)
    compact_rows = []
    for judge, prefix, family in JUDGES:
        for dataset in TARGET_DATASETS:
            for model_name in TARGET_MODELS:
                sub = long_df[(long_df["judge"] == judge) & (long_df["dataset"] == dataset) & (long_df["generator"] == model_name)]
                p1 = float(sub[sub["prompt"] == "p1"]["kappa"].iloc[0])
                p2 = float(sub[sub["prompt"] == "p2"]["kappa"].iloc[0])
                compact_rows.append({"judge": judge, "judge_prefix": prefix, "judge_family": family, "dataset": dataset, "dataset_display": DATASET_DISPLAY[dataset], "generator": model_name, "gen": GEN_LABELS[model_name], "p1_kappa": p1, "p2_kappa": p2, "delta_p2_minus_p1": p2 - p1})
    pd.DataFrame(compact_rows).to_csv(out_compact, index=False)
    return [out_long, out_compact]


# =============================================================================
# Factorial ablation (Tables 8 and 9). Ported from notebook CELL C9.
# =============================================================================

FACTORIAL_HUMAN_COL = "annotator1_label"
FACTORIAL_QID_COL = "qid"

# Display name, criterion, structure.
FACTORIAL_PROMPT_DEFS = [
    ("p1", "p1", "Faithfulness", "Terse"),
    ("p1s", "p1s", "Faithfulness", "Structured"),
    ("p2t", "p2t", "Factual", "Terse"),
    ("p2", "p2", "Factual", "Structured"),
]

# Contrast name, prompt_a, prompt_b, held-constant string for the table.
# The held-constant strings are the manuscript's wording; do not shorten them.
FACTORIAL_CONTRAST_DEFS = [
    ("Criterion, terse pair", "p1", "p2t", "Structure: terse"),
    ("Criterion, structured pair", "p1s", "p2", "Structure: structured"),
    ("Structure, factual criterion", "p2t", "p2", "Criterion: factual correctness"),
    ("Structure, faithfulness criterion", "p1", "p1s", "Criterion: reference faithfulness"),
]

FACTORIAL_GEN_DISPLAY = {**GEN_LABELS, "ALL": "All"}

FACTORIAL_N_BOOT = 2000
FACTORIAL_BOOT_SEED = 42

# Some stored runs use longer prompt keys; accept both so an existing
# labels.csv does not need to be regenerated.
FACTORIAL_COLUMN_ALIASES = {
    "p1": ["label_p1", "label_gpt_5_mini_p1_original", "label_p1_original"],
    "p1s": ["label_p1s"],
    "p2t": ["label_p2t", "label_gpt_5_mini_p2t", "label_p2t_terse"],
    "p2": ["label_p2", "label_gpt_5_mini_p2_repaired", "label_p2_repaired"],
}


def resolve_factorial_label_column(df: pd.DataFrame, prompt_key: str) -> str:
    for candidate in FACTORIAL_COLUMN_ALIASES[prompt_key]:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"No label column found for prompt {prompt_key!r}. "
        f"Tried: {FACTORIAL_COLUMN_ALIASES[prompt_key]}. "
        f"Available: {[c for c in df.columns if c.startswith('label_')]}"
    )


def as_int_labels(series: pd.Series) -> np.ndarray:
    """Coerce a label column to int, mapping unparseable values to -1."""
    return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(dtype=int)


def factorial_condition_metrics(y_true, y_pred) -> dict:
    """Agreement metrics on rows where both labels are valid (0/1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isin(y_true, [0, 1]) & np.isin(y_pred, [0, 1])
    yh = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    tn, fp, fn, tp = confusion_counts(yh, yp)
    return {
        "n_valid": int(mask.sum()),
        "n_invalid": int((~mask).sum()),
        "kappa": cohen_kappa_binary(yh, yp),
        "bias": float(yp.mean() - yh.mean()),
        "fp": fp,
        "fn": fn,
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
    }


def bootstrap_delta_kappa_clustered(
    y_true, pred_a, pred_b, groups, n_boot: int, seed: int
) -> tuple[float, float, int]:
    """Question-level bootstrap interval for kappa(pred_b) - kappa(pred_a).

    Questions are resampled with replacement, keeping the three generator
    responses to each question together, and both kappas are recomputed on
    the same resampled questions. Groups are resampled by first-occurrence
    position (matching pd.unique(groups)) rather than by sorted group label,
    so that resamples are reproducible against the notebook cell this was
    ported from.

    Do not merge this with bootstrap_kappa_ci_clustered. That function
    resamples np.unique's sorted group order; this one resamples
    pd.unique's first-occurrence order, specifically to reproduce the exact
    RNG call sequence of notebook CELL C9. The two orderings draw a
    different sequence of groups from the same seed, so unifying them onto
    one ordering would silently change the published factorial-contrast
    confidence intervals (Table 9) even though both orderings are
    statistically valid.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    groups = np.asarray(groups)

    unique_groups = pd.unique(groups)
    index_by_group = [np.flatnonzero(groups == group) for group in unique_groups]

    vals = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(index_by_group), size=len(index_by_group))
        idx = np.concatenate([index_by_group[i] for i in sampled])
        ka = cohen_kappa_binary(y_true[idx], pred_a[idx])
        kb = cohen_kappa_binary(y_true[idx], pred_b[idx])
        delta = kb - ka
        if np.isfinite(delta):
            vals.append(delta)

    if not vals:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def fmt_delta(x: float, digits: int = 3) -> str:
    """Delta-kappa formatting: plain '+' for non-negative, math minus for negative."""
    return f"+{x:.{digits}f}" if x >= 0 else f"$-${abs(x):.{digits}f}"


def latex_listing_block(text: str) -> str:
    """Wrap prompt text in an lstlisting block, matching the appendix prompt listings."""
    return "\\begin{lstlisting}[style=promptstyle]\n" + text.rstrip() + "\n\\end{lstlisting}\n"


def run_factorial_ablation() -> list[Path]:
    """Criterion x structure factorial prompt experiment (Tables 8 and 9)."""
    ensure_dirs()

    matches = sorted((AUDITS_DIR / "factorial_prompt_run").glob("*/labels.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one factorial labels.csv under "
            f"{AUDITS_DIR / 'factorial_prompt_run'}, found {len(matches)}."
        )
    labels_path = matches[0]
    run_dir = labels_path.parent

    labels = pd.read_csv(labels_path)
    if len(labels) != 900:
        raise ValueError(f"Expected 900 rows in {labels_path.name}, found {len(labels)}")
    require_columns(labels, [FACTORIAL_HUMAN_COL, FACTORIAL_QID_COL, "model"])

    qid_sizes = labels.groupby(FACTORIAL_QID_COL).size()
    if len(qid_sizes) != 300 or not (qid_sizes == 3).all():
        raise ValueError("Expected 300 questions with three generator responses each.")

    label_cols = {
        key: resolve_factorial_label_column(labels, key)
        for key, _, _, _ in FACTORIAL_PROMPT_DEFS
    }

    for key, col in label_cols.items():
        n_valid = int(pd.to_numeric(labels[col], errors="coerce").isin([0, 1]).sum())
        if n_valid != len(labels):
            raise ValueError(
                f"{col} has invalid labels: valid={n_valid}/{len(labels)}."
            )

    # =========================================================================
    # Agreement summary (Table 8)
    # =========================================================================
    summary_rows = []
    for key, prompt_display, criterion, structure in FACTORIAL_PROMPT_DEFS:
        col = label_cols[key]
        for generator in TARGET_MODELS + ["ALL"]:
            sub = labels if generator == "ALL" else labels[labels["model"] == generator]
            y = sub[FACTORIAL_HUMAN_COL].to_numpy(dtype=int)
            p = as_int_labels(sub[col])
            m = factorial_condition_metrics(y, p)
            summary_rows.append({
                "prompt": prompt_display,
                "criterion": criterion,
                "structure": structure,
                "generator": generator,
                "gen_display": FACTORIAL_GEN_DISPLAY[generator],
                **m,
            })
    summary = pd.DataFrame(summary_rows)
    out_summary_csv = run_dir / "agreement_summary.csv"
    summary.to_csv(out_summary_csv, index=False)

    # =========================================================================
    # Paired contrasts (Table 9)
    # =========================================================================
    y_all = labels[FACTORIAL_HUMAN_COL].to_numpy(dtype=int)
    qid_all = labels[FACTORIAL_QID_COL].astype(str).to_numpy()

    pair_rows = []
    for contrast_name, key_a, key_b, held_constant in FACTORIAL_CONTRAST_DEFS:
        a = as_int_labels(labels[label_cols[key_a]])
        b = as_int_labels(labels[label_cols[key_b]])

        valid = np.isin(y_all, [0, 1]) & np.isin(a, [0, 1]) & np.isin(b, [0, 1])
        yv, av, bv, qv = y_all[valid], a[valid], b[valid], qid_all[valid]

        ka = cohen_kappa_binary(yv, av)
        kb = cohen_kappa_binary(yv, bv)
        lo, hi, n_boot_valid = bootstrap_delta_kappa_clustered(
            yv, av, bv, qv, n_boot=FACTORIAL_N_BOOT, seed=FACTORIAL_BOOT_SEED
        )

        pair_rows.append({
            "contrast": contrast_name,
            "held_constant": held_constant,
            "prompt_a": key_a,
            "prompt_b": key_b,
            "n": int(len(yv)),
            "n_correct_a": int(np.sum(av == yv)),
            "n_correct_b": int(np.sum(bv == yv)),
            "label_differences": int(np.sum(av != bv)),
            "difference_pct": float(100.0 * np.mean(av != bv)),
            "kappa_a": ka,
            "kappa_b": kb,
            "delta_kappa": kb - ka,
            "ci_low": lo,
            "ci_high": hi,
            "n_boot_valid": n_boot_valid,
        })
    pairs = pd.DataFrame(pair_rows)
    out_pairs_csv = run_dir / "paired_contrasts.csv"
    pairs.to_csv(out_pairs_csv, index=False)

    # =========================================================================
    # Contrast invariants. These are identities, not empirical expectations.
    # A failure means a non-binary label leaked through, rows are misaligned,
    # or the bootstrap silently discarded resamples.
    # =========================================================================
    for _, r in pairs.iterrows():
        tag = f"{r['prompt_a']}->{r['prompt_b']}"
        if r["n_boot_valid"] != FACTORIAL_N_BOOT:
            raise ValueError(
                f"{tag}: only {r['n_boot_valid']}/{FACTORIAL_N_BOOT} bootstrap "
                "resamples were usable; the caption's resample count would be "
                "inaccurate."
            )
        if abs(r["delta_kappa"] - (r["kappa_b"] - r["kappa_a"])) >= 1e-12:
            raise ValueError(f"{tag}: delta_kappa does not equal kappa_b - kappa_a.")

    # A prompt appearing in several contrasts must carry the same kappa.
    kappa_by_prompt: dict[str, float] = {}
    for _, r in pairs.iterrows():
        for side in ["a", "b"]:
            key = r[f"prompt_{side}"]
            val = float(r[f"kappa_{side}"])
            if key in kappa_by_prompt:
                if abs(kappa_by_prompt[key] - val) >= 1e-12:
                    raise ValueError(
                        f"{key}: kappa differs across contrasts "
                        f"({kappa_by_prompt[key]:.6f} vs {val:.6f})."
                    )
            else:
                kappa_by_prompt[key] = val

    # The 2x2 must close: the interaction is the same computed either way.
    if {"p1", "p1s", "p2t", "p2"} <= set(kappa_by_prompt):
        by_rows = (
            (kappa_by_prompt["p2"] - kappa_by_prompt["p2t"])
            - (kappa_by_prompt["p1s"] - kappa_by_prompt["p1"])
        )
        by_cols = (
            (kappa_by_prompt["p2"] - kappa_by_prompt["p1s"])
            - (kappa_by_prompt["p2t"] - kappa_by_prompt["p1"])
        )
        if abs(by_rows - by_cols) >= 1e-12:
            raise ValueError("2x2 interaction is not additive.")

    # =========================================================================
    # Table 8 -- agreement by prompt and generator
    # =========================================================================
    agreement_caption = (
        r"Criterion-by-structure factorial prompt experiment with GPT-5-mini. "
        r"All four prompts were evaluated on the same 900 items in a single "
        r"interleaved pass. p1 and p2 are the paper prompts; p1s applies the "
        r"reference-faithfulness criterion in p2's structured decision-tree "
        r"format, and p2t applies the factual-correctness criterion in p1's "
        r"terse format. Bias is positive when the automated labeler "
        r"over-labels hallucinations relative to humans. L=Llama-3-8B, "
        r"G=Gemma-2-9B, M=Mistral-7B; All pools the three generator sets."
    )

    latex_agreement = r"""\begin{table*}[t]
  \centering
  \caption{""" + agreement_caption + r"""}
  \label{tab:factorial_agreement}
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lllrrrrrr}
    \toprule
    Prompt & Criterion & Structure & Gen & $\kappa$ & Bias & FP & FN & FPR \\
    \midrule
"""
    prev_prompt = None
    for key, prompt_display, criterion, structure in FACTORIAL_PROMPT_DEFS:
        block = summary[summary["prompt"] == prompt_display]
        if prev_prompt is not None:
            latex_agreement += "    \\addlinespace[2pt]\n"
        prev_prompt = prompt_display
        for generator in TARGET_MODELS + ["ALL"]:
            r = block[block["generator"] == generator].iloc[0]
            latex_agreement += (
                f"    {prompt_display} & {criterion} & {structure} & "
                f"{FACTORIAL_GEN_DISPLAY[generator]} & "
                f"{r['kappa']:.3f} & {fmt_bias(r['bias'])} & "
                f"{int(r['fp'])} & {int(r['fn'])} & {r['fpr']:.3f} \\\\\n"
            )
    latex_agreement += r"""    \bottomrule
  \end{tabular}
\end{table*}
"""
    out_agreement_tex = TABLES_DIR / "table_factorial_agreement.tex"
    write_text(out_agreement_tex, latex_agreement)

    # =========================================================================
    # Table 9 -- paired contrasts
    # =========================================================================
    contrasts_caption = (
        r"Paired contrasts from the criterion-by-structure factorial prompt "
        r"experiment, pooled over the three generator sets. The first two rows "
        r"hold prompt structure constant and vary the labeling criterion; the "
        r"last two hold the criterion constant and vary prompt structure. "
        r"$\Delta\kappa=\kappa_b-\kappa_a$. Brackets give 95\% nonparametric "
        r"bootstrap intervals based on 2,000 question-level resamples, keeping "
        r"the three generator responses to each question together. Diff\% is the "
        r"proportion of items receiving different labels under the two prompts."
    )

    latex_contrasts = r"""\begin{table*}[t]
  \centering
  \caption{""" + contrasts_caption + r"""}
  \label{tab:factorial_contrasts}
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{llrrrl}
    \toprule
    Contrast & Held constant & Diff\% & $\kappa_a$ & $\kappa_b$
    & $\Delta\kappa$ [95\% CI] \\
    \midrule
"""
    for _, r in pairs.iterrows():
        latex_contrasts += (
            f"    {r['prompt_a']} $\\to$ {r['prompt_b']} & "
            f"{r['held_constant']}\n"
            f"      & {r['difference_pct']:.1f} & "
            f"{r['kappa_a']:.3f} & {r['kappa_b']:.3f}\n"
            f"      & {fmt_delta(r['delta_kappa'])} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] \\\\\n"
        )
    latex_contrasts += r"""    \bottomrule
  \end{tabular}
\end{table*}
"""
    out_contrasts_tex = TABLES_DIR / "table_factorial_contrasts.tex"
    write_text(out_contrasts_tex, latex_contrasts)

    # =========================================================================
    # Auxiliary prompt listings and the p1 -> p1s diff for the appendix.
    # =========================================================================
    outputs = [out_summary_csv, out_pairs_csv, out_agreement_tex, out_contrasts_tex]

    p1_path = PROMPTS_DIR / "p1.txt"
    p1s_path = PROMPTS_DIR / "p1s.txt"
    p2t_path = PROMPTS_DIR / "p2t.txt"

    if p1s_path.exists():
        out_p1s_tex = TABLES_DIR / "prompt_p1s_verbatim.tex"
        write_text(out_p1s_tex, latex_listing_block(p1s_path.read_text(encoding="utf-8")))
        outputs.append(out_p1s_tex)

    if p2t_path.exists():
        out_p2t_tex = TABLES_DIR / "prompt_p2t_verbatim.tex"
        write_text(out_p2t_tex, latex_listing_block(p2t_path.read_text(encoding="utf-8")))
        outputs.append(out_p2t_tex)

    if p1_path.exists() and p1s_path.exists():
        p1_text = p1_path.read_text(encoding="utf-8")
        p1s_text = p1s_path.read_text(encoding="utf-8")
        diff_lines = list(difflib.unified_diff(
            p1_text.splitlines(),
            p1s_text.splitlines(),
            fromfile="p1 (faithfulness, terse)",
            tofile="p1s (faithfulness, structured)",
            lineterm="",
            n=2,
        ))
        out_diff_tex = TABLES_DIR / "prompt_p1_vs_p1s_diff.tex"
        write_text(out_diff_tex, latex_listing_block("\n".join(diff_lines)))
        outputs.append(out_diff_tex)

    return outputs


# =============================================================================
# Elaboration intervention (Table 12, Appendix E, Section 5.1). Ported from
# notebook CELL C10 (qualitative examples) and CELL C11 (answer length), plus
# the false-positive-rate/McNemar/bootstrap-CI analysis, which is computed
# purely from the already-stored paired_results.csv and
# human_validation_template_validated.csv columns -- no API calls, no GPU --
# but currently lives mixed into generation in CELL B10b rather than in a
# Section C cell of its own.
# =============================================================================

ELABORATION_QUALITATIVE_N = 10
ELABORATION_QUALITATIVE_SEED = 42
ELABORATION_BOOTSTRAP_N = 10_000
ELABORATION_BOOTSTRAP_SEED = 42


def find_elaboration_paired_csv() -> Path:
    matches = sorted(
        (AUDITS_DIR / "elaboration_core_only").glob(
            "*/p1_original_vs_core_b4matched/paired_results.csv"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one paired_results.csv under "
            f"{AUDITS_DIR / 'elaboration_core_only'}, found {len(matches)}."
        )
    return matches[0]


def run_elaboration_examples() -> list[Path]:
    """Table 12: qualitative FP -> TN examples for the elaboration intervention.

    Table 12 itself prints five hand-picked, illustrative examples with no
    item_ids, so it cannot be checked against expected values the way the
    numeric tables are. Instead, this asserts that the reproducibly sampled
    10 item_ids match the committed fp_to_tn_random10.csv, so a changed seed
    or a changed paired_results.csv is caught even though the table's own
    content can't be.
    """
    ensure_dirs()
    paired_csv = find_elaboration_paired_csv()
    out_dir = paired_csv.parent
    examples_csv = out_dir / "fp_to_tn_random10.csv"

    paired = pd.read_csv(paired_csv)
    require_columns(paired, ["item_id", "transition"])
    flips = paired[paired["transition"] == "FP_TO_TN"].copy()

    sample_n = min(ELABORATION_QUALITATIVE_N, len(flips))
    examples = (
        flips.sample(n=sample_n, random_state=ELABORATION_QUALITATIVE_SEED)
        if sample_n
        else flips
    )

    if examples_csv.exists():
        committed_ids = pd.read_csv(examples_csv)["item_id"].tolist()
        sampled_ids = examples["item_id"].tolist()
        if sampled_ids != committed_ids:
            raise ValueError(
                "Sampled FP->TN item_ids do not match the committed "
                f"{examples_csv.name}. This means the seed, the sample size, "
                "or the input paired_results.csv changed.\n"
                f"  committed: {committed_ids}\n"
                f"  sampled:   {sampled_ids}"
            )

    reference_lookup = (
        read_master()[["item_id", "ref_str"]].drop_duplicates(subset="item_id")
    )
    examples = examples.merge(
        reference_lookup, on="item_id", how="left", validate="one_to_one"
    )
    if examples["ref_str"].isna().any():
        missing = examples.loc[examples["ref_str"].isna(), "item_id"].tolist()
        raise ValueError(f"Missing reference answers for: {missing}")

    examples = examples[[
        "item_id", "question", "ref_str", "original_answer", "core_only_answer",
        "p1_original", "p1_core", "p1_original_confidence", "p1_core_confidence",
    ]]
    examples.to_csv(examples_csv, index=False)
    return [examples_csv]


def elaboration_word_count(text) -> int:
    """Word-like sequences, including contractions/hyphenated forms."""
    return len(re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", str(text), flags=re.UNICODE))


def elaboration_regex_units(text) -> list[str]:
    """Same regex-based units used in the extraction validator. NOT model-tokenizer tokens."""
    return re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)


def elaboration_regex_unit_count(text) -> int:
    return len(elaboration_regex_units(text))


def run_elaboration_length() -> list[Path]:
    """Answer length reduction from original to core-only answers (Section 5.1)."""
    ensure_dirs()
    paired_csv = find_elaboration_paired_csv()
    out_dir = paired_csv.parent
    length_csv = out_dir / "answer_length_analysis.csv"
    summary_csv = out_dir / "answer_length_summary.csv"

    paired = pd.read_csv(paired_csv)
    require_columns(paired, ["item_id", "original_answer", "core_only_answer"])

    length_df = paired[["item_id", "original_answer", "core_only_answer"]].copy()
    length_df["original_words"] = length_df["original_answer"].fillna("").apply(elaboration_word_count)
    length_df["core_words"] = length_df["core_only_answer"].fillna("").apply(elaboration_word_count)
    length_df["words_removed"] = length_df["original_words"] - length_df["core_words"]
    length_df["word_fraction_removed"] = np.where(
        length_df["original_words"] > 0,
        length_df["words_removed"] / length_df["original_words"],
        np.nan,
    )
    length_df["word_percent_removed"] = 100 * length_df["word_fraction_removed"]

    length_df["original_regex_units"] = length_df["original_answer"].fillna("").apply(elaboration_regex_unit_count)
    length_df["core_regex_units"] = length_df["core_only_answer"].fillna("").apply(elaboration_regex_unit_count)
    length_df["regex_units_removed"] = length_df["original_regex_units"] - length_df["core_regex_units"]
    length_df["regex_fraction_removed"] = np.where(
        length_df["original_regex_units"] > 0,
        length_df["regex_units_removed"] / length_df["original_regex_units"],
        np.nan,
    )
    length_df["regex_percent_removed"] = 100 * length_df["regex_fraction_removed"]
    length_df.to_csv(length_csv, index=False)

    original_total = int(length_df["original_words"].sum())
    core_total = int(length_df["core_words"].sum())

    summary = pd.DataFrame([{
        "key": "answer_length_summary",
        "n": len(length_df),
        "original_mean_words": float(length_df["original_words"].mean()),
        "original_median_words": float(length_df["original_words"].median()),
        "core_mean_words": float(length_df["core_words"].mean()),
        "core_median_words": float(length_df["core_words"].median()),
        "median_word_percent_removed": float(length_df["word_percent_removed"].median()),
        "overall_word_percent_removed": 100.0 * (original_total - core_total) / original_total,
    }])
    summary.to_csv(summary_csv, index=False)
    return [length_csv, summary_csv]


def bootstrap_mean_diff_ci(a, b, n_boot: int, seed: int) -> tuple[float, float]:
    """Item-level bootstrap CI for mean(a) - mean(b), for paired 0/1 arrays."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    vals = [
        a[idx].mean() - b[idx].mean()
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))
    ]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def run_elaboration_summary() -> list[Path]:
    """Extraction/exclusion counts and the FP-rate reduction (Section 5.1, Appendix E)."""
    ensure_dirs()
    matches = sorted(
        (AUDITS_DIR / "elaboration_core_only").glob(
            "*/human_validation_template_validated.csv"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one human_validation_template_validated.csv under "
            f"{AUDITS_DIR / 'elaboration_core_only'}, found {len(matches)}."
        )
    validated_csv = matches[0]
    out_csv = validated_csv.parent / "elaboration_summary.csv"

    validated = pd.read_csv(validated_csv)
    require_columns(validated, ["item_id", "human_valid"])
    n_extractions = len(validated)
    n_excluded = int((validated["human_valid"] == 0).sum())
    n_paired = int((validated["human_valid"] == 1).sum())
    if n_extractions != n_excluded + n_paired:
        raise ValueError(
            f"human_valid has values other than 0/1: {n_extractions} rows, "
            f"{n_excluded} excluded + {n_paired} paired != {n_extractions}."
        )

    paired_csv = find_elaboration_paired_csv()
    paired = pd.read_csv(paired_csv)
    require_columns(paired, ["p1_original", "p1_core", "transition"])
    if len(paired) != n_paired:
        raise ValueError(
            f"{paired_csv.name} has {len(paired)} rows, expected {n_paired} "
            f"from human_valid==1 in {validated_csv.name}."
        )

    n = len(paired)
    orig = paired["p1_original"].astype(int).to_numpy()
    core = paired["p1_core"].astype(int).to_numpy()

    original_fp = int(orig.sum())
    core_fp = int(core.sum())
    original_fpr = original_fp / n
    core_fpr = core_fp / n
    absolute_reduction = original_fpr - core_fpr
    relative_reduction = absolute_reduction / original_fpr

    counts = paired["transition"].value_counts().reindex(
        ["FP_TO_TN", "TN_TO_FP", "FP_STAYS_FP", "TN_STAYS_TN"], fill_value=0
    )
    n_fp_to_tn = int(counts["FP_TO_TN"])
    n_tn_to_fp = int(counts["TN_TO_FP"])
    mcnemar_p = exact_mcnemar_p(n_fp_to_tn, n_tn_to_fp)

    ci_low, ci_high = bootstrap_mean_diff_ci(
        orig, core, n_boot=ELABORATION_BOOTSTRAP_N, seed=ELABORATION_BOOTSTRAP_SEED
    )

    summary = pd.DataFrame([{
        "key": "elaboration_summary",
        "n_extractions": n_extractions,
        "n_excluded": n_excluded,
        "n_paired": n_paired,
        "paired_pct": 100.0 * n_paired / n_extractions,
        "original_fp": original_fp,
        "core_fp": core_fp,
        "original_fpr_pct": 100.0 * original_fpr,
        "core_fpr_pct": 100.0 * core_fpr,
        "absolute_reduction_pct": 100.0 * absolute_reduction,
        "ci_low_pct": 100.0 * ci_low,
        "ci_high_pct": 100.0 * ci_high,
        "relative_reduction_pct": 100.0 * relative_reduction,
        "n_fp_to_tn": n_fp_to_tn,
        "n_tn_to_fp": n_tn_to_fp,
        "n_fp_stays_fp": int(counts["FP_STAYS_FP"]),
        "n_tn_stays_tn": int(counts["TN_STAYS_TN"]),
        "mcnemar_p_exact": mcnemar_p,
    }])
    summary.to_csv(out_csv, index=False)
    return [out_csv]


def run_all_release_analyses() -> list[Path]:
    outputs = []
    for fn in [
        run_iaa,
        run_prompt_ablation,
        run_lexical_nli,
        run_bootstrap_mcnemar,
        run_kappa_matrix,
        run_error_direction_space,
        run_per_dataset_gpt5mini,
        run_per_dataset_prompt_ablation,
        run_factorial_ablation,
        run_new_metrics,
        run_elaboration_examples,
        run_elaboration_length,
        run_elaboration_summary,
    ]:
        outputs.extend(fn())
    return outputs
