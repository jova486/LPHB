# Provenance — gpt5mini_factorial_v3

## Judge model

From `metadata.json`:

- prefix: `gpt_5_mini`
- display: `GPT-5-mini`
- model_id: `gpt-5-mini`

## Prompts used in this run

Four prompt conditions, evaluated on the same 900 question/answer pairs in a single interleaved pass (shuffle seed 42):

| Key | Role | sha256 |
|---|---|---|
| `p1` | reference faithfulness, terse (paper prompt) | `53438c6c04f8e821e30c26759bbf9a2b45c687dc517eeabcb60c475e3af1d30c` |
| `p1s` | reference faithfulness, structured (auxiliary) | `474d605c3d9a42f59b935e060551a6114e380f5ca0e0e1bad0303e2c3a7737f7` |
| `p2t` | factual correctness, terse (auxiliary) | `25a90f5b55549b34e55b8316035832b8026346a6fadf91893cee5ed431b8f414` |
| `p2` | factual correctness, structured (paper prompt) | `cf88add533b4e52df18ac94b16a9d1211af9841e9231543d44362fc7783c44a5` |

## File integrity

Original sha256 of every file in this folder, recorded before any change (2026-10-01):

| File | sha256 (original, before rename) |
|---|---|
| `labels.csv` | `be6e66a0ac4de9e3cf7c76677dc361a8f3c914b23fbb41d0804e245510b31a16` |
| `agreement_summary.csv` | `9853180e260e234e3183db482d50abbe97d995d970f1e2232897b92c3de0ef2b` |
| `paired_contrasts.csv` | `2b12ba04474d49fe355365b6f756896fd7edc8098e35745b726b5be460d27146` |
| `metadata.json` | `02625ac128929c129aa82f3b38a621ac3130bda37b66286f7836b9216418479a` |
| `parse_failures.csv` | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` |

## What changed

On **2026-10-01**, only the **name** `p1x` was changed to `p1s`, in headers and metadata keys only:

- `labels.csv`: header columns `label_p1x, conf_p1x, raw_p1x, error_p1x` renamed to `label_p1s, conf_p1s, raw_p1s, error_p1s`. No data values were changed; verified by comparing the full dataframe against the original file with the old column names mapped onto the new ones — identical in all 900 rows × 24 columns.
- `metadata.json`: the `prompt_sha256` key `"p1x"` renamed to `"p1s"`; its value (the prompt's sha256 hash) is unchanged. Verified by comparing the full parsed JSON against the original with the key mapped — identical.
- `parse_failures.csv`: empty file (a single newline byte); nothing to rename; byte-identical to the original.
- `agreement_summary.csv` and `paired_contrasts.csv` in this folder are regenerated outputs of the analysis pipeline (not part of this file-level rename); they have since been rewritten with `p1s` row-key naming by `scripts/20_run_release_analyses.py` after `src/release_analysis.py` was updated to use the new prompt key.
- No cell *value* was changed to effect this rename — a check for a standalone prompt-key value of exactly `"p1x"` anywhere in `labels.csv`'s data rows found **0** occurrences, so no value-level substitution was needed or performed.

## Path scrub

On **2026-10-01**, `metadata.json`'s `master_path` value was changed from an absolute Google-Drive-mounted session path (recorded verbatim at the time the run was produced) to the repo-relative path `data/human_judge_release_master.csv`. No other value in the file was changed (confirmed by diff: exactly one line differs). `master_path` is write-only provenance — nothing in `scripts/`, `src/`, or any notebook cell reads it back — so this change has no effect on any computation.

`metadata.json` sha256: `84277b40486925beefc7de8b3d9e9e5ad365b2242a36c52570ad98e43f15c411` (old, after the p1x→p1s rename, before this path scrub) → `828992fad53d1ebfc04e5b4989cd1c32a70594e551f07c8e194c4a247c3f8db9` (new, after this path scrub).

## Relationship to the earlier run

The earlier `gpt5mini_factorial_v1` run (which used the older `p1x` prompt wording — the version requiring that any claim absent from the reference, even if factually correct, be labeled a hallucination) is **superseded** and is **not part of this repository**; it has been moved to `~/old_runs/gpt5mini_factorial_v1` outside the repo.
