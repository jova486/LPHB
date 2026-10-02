"""Generate results/expected/*.csv from values transcribed from the paper.

Values are taken from the manuscript tables, not from generated outputs, so
that verification cannot become circular. Table references below point at the
camera-ready paper.
"""

import csv
from pathlib import Path

OUT = Path("results/expected")
OUT.mkdir(parents=True, exist_ok=True)

GEN = {"L": "llama3-8b", "G": "gemma-2-9b", "M": "mistral-7b"}
DS = {"HotpotQA": "hotpotqa", "TriviaQA": "triviaqa", "TruthfulQA": "truthfulqa"}

# Judge display names as used in the computed audit CSVs. These match the
# paper's own tables verbatim: the paper never prints "(local)" anywhere
# (the "Local judges" section heading carries that distinction instead).
JUDGE = {
    "GPT-5-mini": "GPT-5-mini",
    "GPT-5-nano": "GPT-5-nano",
    "GPT-5.4": "GPT-5.4",
    "Claude Opus 4.7": "Claude Opus 4.7",
    "Gemma-2-9B": "Gemma-2-9B",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Llama-3-8B": "Llama-3-8B",
}


def write(name, rows):
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_key", "column", "expected", "value_type"])
        w.writerows(rows)
    print(f"{path}  ({len(rows)} values)")


# ---------------------------------------------------------------- Table 3
# dataset, n, disagreements, agreement, kappa, ci_low, ci_high
IAA = [
    ("triviaqa", 99, 9, 0.909, 0.796, 0.618, 0.935),
    ("truthfulqa", 99, 14, 0.859, 0.690, 0.544, 0.816),
    ("hotpotqa", 102, 6, 0.941, 0.832, 0.642, 0.968),
    ("overall", 300, 29, 0.903, 0.808, 0.729, 0.879),
]

rows = []
for ds, n, dis, agr, kap, lo, hi in IAA:
    rows += [
        (ds, "n", n, "int"),
        (ds, "disagreements", dis, "int"),
        (ds, "agreement", agr, "float"),
        (ds, "kappa", kap, "float"),
        (ds, "ci_low", lo, "float"),
        (ds, "ci_high", hi, "float"),
    ]
write("iaa_expected.csv", rows)


# ---------------------------------------------------------------- Table 7
# generator: (HotpotQA p1, p2), (TriviaQA p1, p2), (TruthfulQA p1, p2)
PER_DATASET = {
    "L": {"HotpotQA": (0.714, 0.821), "TriviaQA": (0.448, 0.805), "TruthfulQA": (0.352, 0.558)},
    "G": {"HotpotQA": (0.583, 0.711), "TriviaQA": (0.543, 0.821), "TruthfulQA": (0.273, 0.662)},
    "M": {"HotpotQA": (0.151, 0.564), "TriviaQA": (0.041, 0.597), "TruthfulQA": (0.212, 0.484)},
}

# Section 3.3: TriviaQA 100 questions, HotpotQA 101, TruthfulQA 99.
N_BY_DATASET = {"HotpotQA": 101, "TriviaQA": 100, "TruthfulQA": 99}

rows = []
for gen, per_ds in PER_DATASET.items():
    for ds, (k1, k2) in per_ds.items():
        n = N_BY_DATASET[ds]
        for prompt, kappa in (("p1", k1), ("p2", k2)):
            key = f"{GEN[gen]}|{DS[ds]}|{prompt}"
            rows += [
                (key, "kappa", kappa, "float"),
                (key, "n_valid", n, "int"),
                (key, "n_total", n, "int"),
            ]
write("per_dataset_gpt5mini_expected.csv", rows)


# --------------------------------------------------------------- Table 11
# judge, prompt, gen, kappa, bias, fp, fn, err, fp_pct
ABLATION = [
    ("GPT-5-mini", "p1", "L", 0.530, 0.223, 70, 3, 73, 95.9),
    ("GPT-5-mini", "p2", "L", 0.756, -0.020, 15, 21, 36, 41.7),
    ("GPT-5-mini", "p3", "L", 0.744, 0.007, 20, 18, 38, 52.6),
    ("GPT-5-mini", "p1", "G", 0.498, 0.193, 68, 10, 78, 87.2),
    ("GPT-5-mini", "p2", "G", 0.770, -0.033, 11, 21, 32, 34.4),
    ("GPT-5-mini", "p3", "G", 0.759, -0.007, 16, 18, 34, 47.1),
    ("GPT-5-mini", "p1", "M", 0.194, 0.383, 117, 2, 119, 98.3),
    ("GPT-5-mini", "p2", "M", 0.626, 0.027, 32, 24, 56, 57.1),
    ("GPT-5-mini", "p3", "M", 0.646, 0.030, 31, 22, 53, 58.5),
    ("GPT-5-nano", "p1", "L", 0.536, 0.150, 58, 13, 71, 81.7),
    ("GPT-5-nano", "p2", "L", 0.655, -0.017, 23, 28, 51, 45.1),
    ("GPT-5-nano", "p3", "L", 0.676, 0.000, 24, 24, 48, 50.0),
    ("GPT-5-nano", "p1", "G", 0.592, 0.080, 42, 18, 60, 70.0),
    ("GPT-5-nano", "p2", "G", 0.686, -0.020, 19, 25, 44, 43.2),
    ("GPT-5-nano", "p3", "G", 0.641, -0.027, 21, 29, 50, 42.0),
    ("GPT-5-nano", "p1", "M", 0.389, 0.177, 72, 19, 91, 79.1),
    ("GPT-5-nano", "p2", "M", 0.601, -0.047, 23, 37, 60, 38.3),
    ("GPT-5-nano", "p3", "M", 0.581, -0.057, 23, 40, 63, 36.5),
    ("GPT-5.4", "p1", "L", 0.306, 0.363, 110, 1, 111, 99.1),
    ("GPT-5.4", "p2", "L", 0.779, 0.023, 20, 13, 33, 60.6),
    ("GPT-5.4", "p3", "L", 0.740, 0.050, 27, 12, 39, 69.2),
    ("GPT-5.4", "p1", "G", 0.240, 0.413, 127, 3, 130, 97.7),
    ("GPT-5.4", "p2", "G", 0.766, -0.003, 16, 17, 33, 48.5),
    ("GPT-5.4", "p3", "G", 0.774, 0.000, 16, 16, 32, 50.0),
    ("GPT-5.4", "p1", "M", 0.021, 0.480, 144, 0, 144, 100.0),
    ("GPT-5.4", "p2", "M", 0.619, 0.077, 40, 17, 57, 70.2),
    ("GPT-5.4", "p3", "M", 0.672, 0.070, 35, 14, 49, 71.4),
    ("Claude Opus 4.7", "p1", "L", 0.712, 0.037, 27, 16, 43, 62.8),
    ("Claude Opus 4.7", "p2", "L", 0.764, -0.003, 17, 18, 35, 48.6),
    ("Claude Opus 4.7", "p3", "L", 0.738, 0.010, 21, 18, 39, 53.8),
    ("Claude Opus 4.7", "p1", "G", 0.740, 0.067, 29, 9, 38, 76.3),
    ("Claude Opus 4.7", "p2", "G", 0.764, -0.023, 13, 20, 33, 39.4),
    ("Claude Opus 4.7", "p3", "G", 0.737, -0.010, 17, 20, 37, 45.9),
    ("Claude Opus 4.7", "p1", "M", 0.693, 0.033, 28, 18, 46, 60.9),
    ("Claude Opus 4.7", "p2", "M", 0.706, 0.013, 24, 20, 44, 54.5),
    ("Claude Opus 4.7", "p3", "M", 0.673, 0.023, 28, 21, 49, 57.1),
    ("Gemma-2-9B", "p1", "L", 0.567, 0.127, 52, 14, 66, 78.8),
    ("Gemma-2-9B", "p2", "L", 0.591, -0.047, 23, 37, 60, 38.3),
    ("Gemma-2-9B", "p3", "L", 0.557, 0.020, 36, 30, 66, 54.5),
    ("Gemma-2-9B", "p1", "G", 0.463, 0.183, 69, 14, 83, 83.1),
    ("Gemma-2-9B", "p2", "G", 0.623, -0.083, 13, 38, 51, 25.5),
    ("Gemma-2-9B", "p3", "G", 0.628, -0.020, 23, 29, 52, 44.2),
    ("Gemma-2-9B", "p1", "M", 0.378, 0.090, 60, 33, 93, 64.5),
    ("Gemma-2-9B", "p2", "M", 0.496, -0.153, 15, 61, 76, 19.7),
    ("Gemma-2-9B", "p3", "M", 0.455, -0.087, 28, 54, 82, 34.1),
    ("Qwen2.5-7B", "p1", "L", 0.313, 0.280, 96, 12, 108, 88.9),
    ("Qwen2.5-7B", "p2", "L", 0.483, 0.023, 42, 35, 77, 54.5),
    ("Qwen2.5-7B", "p3", "L", 0.472, 0.100, 55, 25, 80, 68.8),
    ("Qwen2.5-7B", "p1", "G", 0.376, 0.283, 93, 8, 101, 92.1),
    ("Qwen2.5-7B", "p2", "G", 0.607, -0.053, 19, 35, 54, 35.2),
    ("Qwen2.5-7B", "p3", "G", 0.559, 0.083, 45, 20, 65, 69.2),
    ("Qwen2.5-7B", "p1", "M", 0.213, 0.203, 89, 28, 117, 76.1),
    ("Qwen2.5-7B", "p2", "M", 0.428, -0.060, 34, 52, 86, 39.5),
    ("Qwen2.5-7B", "p3", "M", 0.452, 0.047, 48, 34, 82, 58.5),
    ("Llama-3-8B", "p1", "L", 0.370, 0.077, 59, 36, 95, 62.1),
    ("Llama-3-8B", "p2", "L", 0.303, -0.240, 13, 85, 98, 13.3),
    ("Llama-3-8B", "p3", "L", 0.255, -0.230, 18, 87, 105, 17.1),
    ("Llama-3-8B", "p1", "G", 0.227, 0.210, 92, 29, 121, 76.0),
    ("Llama-3-8B", "p2", "G", 0.470, -0.157, 11, 58, 69, 15.9),
    ("Llama-3-8B", "p3", "G", 0.384, -0.160, 16, 64, 80, 20.0),
    ("Llama-3-8B", "p1", "M", 0.278, 0.073, 65, 43, 108, 60.2),
    ("Llama-3-8B", "p2", "M", 0.218, -0.383, 2, 117, 119, 1.7),
    ("Llama-3-8B", "p3", "M", 0.185, -0.380, 5, 119, 124, 4.0),
]

rows = []
for judge, prompt, gen, kap, bias, fp, fn, err, fp_pct in ABLATION:
    assert fp + fn == err, f"{judge} {prompt} {gen}: fp+fn != err"
    key = f"{JUDGE[judge]}|{GEN[gen]}|{prompt}"
    rows += [
        (key, "kappa", kap, "float"),
        (key, "bias", bias, "float"),
        (key, "fp", fp, "int"),
        (key, "fn", fn, "int"),
        (key, "err", err, "int"),
        (key, "fp_pct", fp_pct, "float"),
        (key, "n_valid", 300, "int"),
        (key, "n_total", 300, "int"),
    ]
write("prompt_ablation_expected.csv", rows)


# ---------------------------------------------------------------- Table 2
# metric label as it appears in the computed CSV, gen, kappa, bias, fp, fn, fp_pct
LEXICAL = [
    (r"ROUGE-L (oracle-$\tau$)", "L", 0.403, -0.183, 15, 70, 17.6),
    (r"ROUGE-L (oracle-$\tau$)", "G", 0.483, -0.057, 27, 44, 38.0),
    (r"ROUGE-L (oracle-$\tau$)", "M", 0.412, -0.257, 6, 83, 6.7),
    (r"ROUGE-L (CV-$\tau$)", "L", 0.353, -0.143, 25, 68, 26.9),
    (r"ROUGE-L (CV-$\tau$)", "G", 0.473, -0.067, 26, 46, 36.1),
    (r"ROUGE-L (CV-$\tau$)", "M", 0.385, -0.230, 12, 81, 12.9),
    (r"BERTScore (oracle-$\tau$)", "L", 0.387, -0.047, 38, 52, 42.2),
    (r"BERTScore (oracle-$\tau$)", "G", 0.486, 0.117, 56, 21, 72.7),
    (r"BERTScore (oracle-$\tau$)", "M", 0.415, -0.067, 34, 54, 38.6),
    (r"BERTScore (CV-$\tau$)", "L", 0.318, -0.003, 50, 51, 49.5),
    (r"BERTScore (CV-$\tau$)", "G", 0.486, 0.117, 56, 21, 72.7),
    (r"BERTScore (CV-$\tau$)", "M", 0.394, -0.037, 40, 51, 44.0),
    (r"NLI (T5-11B) strict", "L", 0.053, 0.513, 155, 1, 99.4),
    (r"NLI (T5-11B) strict", "G", 0.025, 0.573, 175, 3, 98.3),
    (r"NLI (T5-11B) strict", "M", 0.049, 0.453, 138, 2, 98.6),
    (r"NLI (T5-11B) best-$\tau$", "L", 0.380, 0.100, 62, 32, 66.0),
    (r"NLI (T5-11B) best-$\tau$", "G", 0.217, 0.200, 91, 31, 74.6),
    (r"NLI (T5-11B) best-$\tau$", "M", 0.395, -0.077, 34, 57, 37.4),
]

rows = []
for metric, gen, kap, bias, fp, fn, fp_pct in LEXICAL:
    key = f"{metric}|{GEN[gen]}"
    rows += [
        (key, "kappa", kap, "float"),
        (key, "bias", bias, "float"),
        (key, "fp", fp, "int"),
        (key, "fn", fn, "int"),
        (key, "err", fp + fn, "int"),
        (key, "fp_pct", fp_pct, "float"),
        (key, "n_valid", 300, "int"),
        (key, "n_total", 300, "int"),
    ]
write("lexical_nli_expected.csv", rows)


# ---------------------------------------------------------------- Table 6
# gen, prompt_a, prompt_b, kappa_a, kappa_b, delta, ci_low, ci_high, mcnemar_p
MCNEMAR = [
    ("L", "p1", "p2", 0.530, 0.756, 0.226, 0.127, 0.328, "<0.0001"),
    ("L", "p1", "p3", 0.530, 0.744, 0.214, 0.121, 0.313, "<0.0001"),
    ("L", "p2", "p3", 0.756, 0.744, -0.012, -0.060, 0.039, "0.7905"),
    ("G", "p1", "p2", 0.498, 0.770, 0.272, 0.179, 0.360, "<0.0001"),
    ("G", "p1", "p3", 0.498, 0.759, 0.261, 0.171, 0.347, "<0.0001"),
    ("G", "p2", "p3", 0.770, 0.759, -0.011, -0.059, 0.037, "0.7744"),
    ("M", "p1", "p2", 0.194, 0.626, 0.432, 0.332, 0.533, "<0.0001"),
    ("M", "p1", "p3", 0.194, 0.646, 0.452, 0.349, 0.559, "<0.0001"),
    ("M", "p2", "p3", 0.626, 0.646, 0.020, -0.047, 0.087, "0.7011"),
]

rows = []
for gen, pa, pb, ka, kb, delta, lo, hi, pval in MCNEMAR:
    assert abs((kb - ka) - delta) < 0.0011, f"{gen} {pa}->{pb}: delta mismatch"
    key = f"{GEN[gen]}|{pa}|{pb}"
    rows += [
        (key, "kappa_a", ka, "float"),
        (key, "kappa_b", kb, "float"),
        (key, "delta_kappa", delta, "float"),
        (key, "delta_ci_low", lo, "float"),
        (key, "delta_ci_high", hi, "float"),
        (key, "mcnemar_p_exact", pval, "float"),
    ]
write("bootstrap_mcnemar_expected.csv", rows)

# ----------------------------------------------------------------- Figure 2
# Macro-averaged pairwise inter-method kappa matrix, upper triangle only
# (diagonal is always 1.0 by construction and is not a reported value).
# Transcribed by hand from the heatmap cell labels in Figure 2 of
# neurIPS_paper.pdf, which prints two decimals per cell. Row/column order
# matches the figure: ROUGE-L, BERTScore, NLI (T5-11B) strict,
# NLI (T5-11B) best-tau, GPT-5-mini p1, GPT-5.4 p1, Opus 4.7 p1,
# GPT-5-mini p2, GPT-5-mini p3, GPT-nano p2, GPT-5.4 p2, Opus 4.7 p2,
# Gemma p2 local, Qwen p2 local.
KAPPA_MATRIX = [
    ("ROUGE-L", "BERTScore", 0.60),
    ("ROUGE-L", "NLI (T5-11B) strict", 0.03),
    ("ROUGE-L", "NLI (T5-11B) best-tau", 0.25),
    ("ROUGE-L", "GPT-5-mini p1", 0.22),
    ("ROUGE-L", "GPT-5.4 p1", 0.11),
    ("ROUGE-L", "Opus 4.7 p1", 0.44),
    ("ROUGE-L", "GPT-5-mini p2", 0.41),
    ("ROUGE-L", "GPT-5-mini p3", 0.43),
    ("ROUGE-L", "GPT-nano p2", 0.38),
    ("ROUGE-L", "GPT-5.4 p2", 0.37),
    ("ROUGE-L", "Opus 4.7 p2", 0.40),
    ("ROUGE-L", "Gemma p2 local", 0.39),
    ("ROUGE-L", "Qwen p2 local", 0.33),
    ("BERTScore", "NLI (T5-11B) strict", 0.06),
    ("BERTScore", "NLI (T5-11B) best-tau", 0.30),
    ("BERTScore", "GPT-5-mini p1", 0.27),
    ("BERTScore", "GPT-5.4 p1", 0.15),
    ("BERTScore", "Opus 4.7 p1", 0.41),
    ("BERTScore", "GPT-5-mini p2", 0.40),
    ("BERTScore", "GPT-5-mini p3", 0.42),
    ("BERTScore", "GPT-nano p2", 0.35),
    ("BERTScore", "GPT-5.4 p2", 0.41),
    ("BERTScore", "Opus 4.7 p2", 0.37),
    ("BERTScore", "Gemma p2 local", 0.34),
    ("BERTScore", "Qwen p2 local", 0.34),
    ("NLI (T5-11B) strict", "NLI (T5-11B) best-tau", 0.09),
    ("NLI (T5-11B) strict", "GPT-5-mini p1", 0.09),
    ("NLI (T5-11B) strict", "GPT-5.4 p1", 0.07),
    ("NLI (T5-11B) strict", "Opus 4.7 p1", 0.06),
    ("NLI (T5-11B) strict", "GPT-5-mini p2", 0.06),
    ("NLI (T5-11B) strict", "GPT-5-mini p3", 0.06),
    ("NLI (T5-11B) strict", "GPT-nano p2", 0.04),
    ("NLI (T5-11B) strict", "GPT-5.4 p2", 0.06),
    ("NLI (T5-11B) strict", "Opus 4.7 p2", 0.03),
    ("NLI (T5-11B) strict", "Gemma p2 local", 0.04),
    ("NLI (T5-11B) strict", "Qwen p2 local", 0.04),
    ("NLI (T5-11B) best-tau", "GPT-5-mini p1", 0.17),
    ("NLI (T5-11B) best-tau", "GPT-5.4 p1", 0.03),
    ("NLI (T5-11B) best-tau", "Opus 4.7 p1", 0.33),
    ("NLI (T5-11B) best-tau", "GPT-5-mini p2", 0.35),
    ("NLI (T5-11B) best-tau", "GPT-5-mini p3", 0.36),
    ("NLI (T5-11B) best-tau", "GPT-nano p2", 0.35),
    ("NLI (T5-11B) best-tau", "GPT-5.4 p2", 0.32),
    ("NLI (T5-11B) best-tau", "Opus 4.7 p2", 0.32),
    ("NLI (T5-11B) best-tau", "Gemma p2 local", 0.34),
    ("NLI (T5-11B) best-tau", "Qwen p2 local", 0.34),
    ("GPT-5-mini p1", "GPT-5.4 p1", 0.42),
    ("GPT-5-mini p1", "Opus 4.7 p1", 0.46),
    ("GPT-5-mini p1", "GPT-5-mini p2", 0.43),
    ("GPT-5-mini p1", "GPT-5-mini p3", 0.44),
    ("GPT-5-mini p1", "GPT-nano p2", 0.40),
    ("GPT-5-mini p1", "GPT-5.4 p2", 0.47),
    ("GPT-5-mini p1", "Opus 4.7 p2", 0.42),
    ("GPT-5-mini p1", "Gemma p2 local", 0.32),
    ("GPT-5-mini p1", "Qwen p2 local", 0.31),
    ("GPT-5.4 p1", "Opus 4.7 p1", 0.23),
    ("GPT-5.4 p1", "GPT-5-mini p2", 0.18),
    ("GPT-5.4 p1", "GPT-5-mini p3", 0.20),
    ("GPT-5.4 p1", "GPT-nano p2", 0.19),
    ("GPT-5.4 p1", "GPT-5.4 p2", 0.21),
    ("GPT-5.4 p1", "Opus 4.7 p2", 0.19),
    ("GPT-5.4 p1", "Gemma p2 local", 0.15),
    ("GPT-5.4 p1", "Qwen p2 local", 0.16),
    ("Opus 4.7 p1", "GPT-5-mini p2", 0.70),
    ("Opus 4.7 p1", "GPT-5-mini p3", 0.73),
    ("Opus 4.7 p1", "GPT-nano p2", 0.70),
    ("Opus 4.7 p1", "GPT-5.4 p2", 0.71),
    ("Opus 4.7 p1", "Opus 4.7 p2", 0.78),
    ("Opus 4.7 p1", "Gemma p2 local", 0.59),
    ("Opus 4.7 p1", "Qwen p2 local", 0.54),
    ("GPT-5-mini p2", "GPT-5-mini p3", 0.88),
    ("GPT-5-mini p2", "GPT-nano p2", 0.76),
    ("GPT-5-mini p2", "GPT-5.4 p2", 0.84),
    ("GPT-5-mini p2", "Opus 4.7 p2", 0.76),
    ("GPT-5-mini p2", "Gemma p2 local", 0.64),
    ("GPT-5-mini p2", "Qwen p2 local", 0.56),
    ("GPT-5-mini p3", "GPT-nano p2", 0.74),
    ("GPT-5-mini p3", "GPT-5.4 p2", 0.82),
    ("GPT-5-mini p3", "Opus 4.7 p2", 0.76),
    ("GPT-5-mini p3", "Gemma p2 local", 0.64),
    ("GPT-5-mini p3", "Qwen p2 local", 0.56),
    ("GPT-nano p2", "GPT-5.4 p2", 0.72),
    ("GPT-nano p2", "Opus 4.7 p2", 0.72),
    ("GPT-nano p2", "Gemma p2 local", 0.66),
    ("GPT-nano p2", "Qwen p2 local", 0.62),
    ("GPT-5.4 p2", "Opus 4.7 p2", 0.81),
    ("GPT-5.4 p2", "Gemma p2 local", 0.61),
    ("GPT-5.4 p2", "Qwen p2 local", 0.53),
    ("Opus 4.7 p2", "Gemma p2 local", 0.65),
    ("Opus 4.7 p2", "Qwen p2 local", 0.54),
    ("Gemma p2 local", "Qwen p2 local", 0.68),
]

rows = []
for method_a, method_b, kappa in KAPPA_MATRIX:
    key = f"{method_a}|{method_b}"
    rows.append((key, "kappa", kappa, "float"))
write("kappa_matrix_expected.csv", rows)


# ----------------------------------------------------------------- Table 8
# prompt, gen, kappa, bias, fp, fn, fpr
FACTORIAL_GEN = {**GEN, "All": "ALL"}

FACTORIAL_AGREEMENT = [
    ("p1", "L", 0.532, 0.237, 72, 1, 0.434),
    ("p1", "G", 0.466, 0.220, 75, 9, 0.403),
    ("p1", "M", 0.174, 0.387, 119, 3, 0.810),
    ("p1", "All", 0.409, 0.281, 266, 13, 0.533),
    ("p1s", "L", 0.404, 0.293, 91, 3, 0.548),
    ("p1s", "G", 0.344, 0.280, 95, 11, 0.511),
    ("p1s", "M", 0.104, 0.433, 131, 1, 0.891),
    ("p1s", "All", 0.305, 0.336, 317, 15, 0.635),
    ("p2t", "L", 0.756, -0.020, 15, 21, 0.090),
    ("p2t", "G", 0.749, -0.023, 14, 21, 0.075),
    ("p2t", "M", 0.626, 0.020, 31, 25, 0.211),
    ("p2t", "All", 0.714, -0.008, 60, 67, 0.120),
    ("p2", "L", 0.811, -0.007, 13, 15, 0.078),
    ("p2", "G", 0.763, -0.030, 12, 21, 0.065),
    ("p2", "M", 0.633, 0.023, 31, 24, 0.211),
    ("p2", "All", 0.739, -0.004, 56, 60, 0.112),
]

rows = []
for prompt, gen, kappa, bias, fp, fn, fpr in FACTORIAL_AGREEMENT:
    key = f"{prompt}|{FACTORIAL_GEN[gen]}"
    rows += [
        (key, "kappa", kappa, "float"),
        (key, "bias", bias, "float"),
        (key, "fp", fp, "int"),
        (key, "fn", fn, "int"),
        (key, "fpr", fpr, "float"),
    ]
write("factorial_agreement_expected.csv", rows)


# ----------------------------------------------------------------- Table 9
# prompt_a, prompt_b, diff_pct, kappa_a, kappa_b, delta_kappa, ci_low, ci_high
FACTORIAL_CONTRASTS = [
    ("p1", "p2t", 29.6, 0.409, 0.714, 0.305, 0.240, 0.369),
    ("p1s", "p2", 34.2, 0.305, 0.739, 0.434, 0.371, 0.494),
    ("p2t", "p2", 4.3, 0.714, 0.739, 0.025, 0.000, 0.053),
    ("p1", "p1s", 9.9, 0.409, 0.305, -0.104, -0.142, -0.066),
]

rows = []
for pa, pb, diff_pct, ka, kb, delta, lo, hi in FACTORIAL_CONTRASTS:
    assert abs((kb - ka) - delta) < 0.0011, f"{pa}->{pb}: delta mismatch"
    key = f"{pa}|{pb}"
    rows += [
        (key, "difference_pct", diff_pct, "float"),
        (key, "kappa_a", ka, "float"),
        (key, "kappa_b", kb, "float"),
        (key, "delta_kappa", delta, "float"),
        (key, "ci_low", lo, "float"),
        (key, "ci_high", hi, "float"),
    ]
write("factorial_contrasts_expected.csv", rows)


# ---------------------------------------------------------------- Table 10
# metric label as it appears in the computed CSV, gen, kappa, bias, fp, fn, fp_pct
NEW_METRICS = [
    (r"METEOR (oracle-$\tau$)", "L", 0.423, -0.137, 21, 62, 25.3),
    (r"METEOR (oracle-$\tau$)", "G", 0.380, 0.077, 57, 34, 62.6),
    (r"METEOR (oracle-$\tau$)", "M", 0.410, -0.157, 21, 68, 23.6),
    (r"METEOR (CV-$\tau$)", "L", 0.354, -0.137, 26, 67, 28.0),
    (r"METEOR (CV-$\tau$)", "G", 0.346, 0.097, 63, 34, 64.9),
    (r"METEOR (CV-$\tau$)", "M", 0.384, -0.190, 18, 75, 19.4),
    (r"BARTScore ref$\to$hyp (oracle-$\tau$)", "L", 0.289, 0.113, 71, 37, 65.7),
    (r"BARTScore ref$\to$hyp (oracle-$\tau$)", "G", 0.402, -0.037, 36, 47, 43.4),
    (r"BARTScore ref$\to$hyp (oracle-$\tau$)", "M", 0.314, -0.023, 48, 55, 46.6),
    (r"BARTScore ref$\to$hyp (CV-$\tau$)", "L", 0.262, 0.107, 72, 40, 64.3),
    (r"BARTScore ref$\to$hyp (CV-$\tau$)", "G", 0.336, 0.023, 51, 44, 53.7),
    (r"BARTScore ref$\to$hyp (CV-$\tau$)", "M", 0.264, -0.150, 33, 78, 29.7),
    (r"BARTScore hyp$\to$ref (oracle-$\tau$)", "L", 0.379, -0.157, 21, 68, 23.6),
    (r"BARTScore hyp$\to$ref (oracle-$\tau$)", "G", 0.494, -0.070, 24, 45, 34.8),
    (r"BARTScore hyp$\to$ref (oracle-$\tau$)", "M", 0.360, -0.020, 45, 51, 46.9),
    (r"BARTScore hyp$\to$ref (CV-$\tau$)", "L", 0.371, -0.160, 21, 69, 23.3),
    (r"BARTScore hyp$\to$ref (CV-$\tau$)", "G", 0.449, -0.047, 31, 45, 40.8),
    (r"BARTScore hyp$\to$ref (CV-$\tau$)", "M", 0.322, -0.100, 36, 66, 35.3),
]

rows = []
for metric, gen, kap, bias, fp, fn, fp_pct in NEW_METRICS:
    key = f"{metric}|{GEN[gen]}"
    rows += [
        (key, "kappa", kap, "float"),
        (key, "bias", bias, "float"),
        (key, "fp", fp, "int"),
        (key, "fn", fn, "int"),
        (key, "err", fp + fn, "int"),
        (key, "fp_pct", fp_pct, "float"),
        (key, "n_valid", 300, "int"),
        (key, "n_total", 300, "int"),
    ]
write("new_metrics_expected.csv", rows)


# ------------------------------------------------- Appendix E / Section 5.1
# Elaboration intervention: extraction/exclusion counts, FP-rate reduction,
# transition counts, and exact McNemar p, from Section 5.1 and Appendix E
# (\label{app:elaboration_removal}). mcnemar_p_exact is transcribed to full
# precision in a separate file with its own tight tolerance: the paper
# rounds it to "8.23e-10" (3 significant figures), and the generic
# decimal-count tolerance derived from that string ("8.23" -> 2 decimals ->
# tol=0.005) is meaningless at this magnitude.
key = "elaboration_summary"
write("elaboration_summary_expected.csv", [
    (key, "n_extractions", 140, "int"),
    (key, "n_excluded", 2, "int"),
    (key, "n_paired", 138, "int"),
    (key, "paired_pct", 98.6, "float"),
    (key, "original_fp", 73, "int"),
    (key, "core_fp", 32, "int"),
    (key, "original_fpr_pct", 52.9, "float"),
    (key, "core_fpr_pct", 23.2, "float"),
    (key, "absolute_reduction_pct", 29.7, "float"),
    (key, "ci_low_pct", 21.0, "float"),
    (key, "ci_high_pct", 38.4, "float"),
    (key, "relative_reduction_pct", 56.2, "float"),
    (key, "n_fp_to_tn", 45, "int"),
    (key, "n_tn_to_fp", 4, "int"),
    (key, "n_fp_stays_fp", 28, "int"),
    (key, "n_tn_stays_tn", 61, "int"),
])

write("elaboration_mcnemar_expected.csv", [
    (key, "mcnemar_p_exact", 8.23e-10, "float"),
])


# ------------------------------------------------------------- Section 5.1
# Answer length reduction (original vs core-only), same Appendix E section.
key = "answer_length_summary"
write("answer_length_expected.csv", [
    (key, "n", 138, "int"),
    (key, "original_mean_words", 48.4, "float"),
    (key, "original_median_words", 51, "int"),
    (key, "core_mean_words", 15.6, "float"),
    (key, "core_median_words", 12, "int"),
    (key, "median_word_percent_removed", 66.0, "float"),
    (key, "overall_word_percent_removed", 67.7, "float"),
])


print("\nAll expected files written from paper tables.")
