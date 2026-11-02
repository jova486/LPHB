# Changelog

## 2.0.0 (2026-11-02)

Correction release (Dataverse version 2.0; a new major version because files were replaced). No existing value was changed.

### Added
- `data/human_judge_release_master.csv`: columns `truncated`, `answer_tokens` and `max_new_tokens` (now 900 rows x 90 columns). They are derived from the generated token sequences: an answer is truncated when generation stopped at `max_new_tokens` without an end-of-sequence token. 258 of the 900 answers are truncated (TruthfulQA 192/297, HotpotQA 39/303, TriviaQA 27/300).
- `data/human_judge_release_data_dictionary.csv`: rows for the three new columns.

### Corrected documentation
- The generation length limit is 96 tokens for Llama-3-8B and 128 for Gemma-2-9B and Mistral-7B. The first version of the paper's Answer Generation Setup appendix stated 128 for all three generators.
- The number of truncated answers is now stated (258 of 900); the dataset metadata previously described them as "a small number".
- `data/README.md`: SHA-256 of the master file, column count and the new column group.

### Unchanged
- All 87 pre-existing columns are identical to release 1.0.0 (verified column by column), including every label and `answer_text`.

### Known issue (documented, not changed)
In 11 answers, a leading or trailing double quotation mark present in the generated output was removed when the release file was prepared. All labels refer to the released text, so `answer_text` is kept as released. Affected rows:
`truthfulqa_s00028__llama3-8b`, `truthfulqa_s00499__llama3-8b`, `truthfulqa_s00698__llama3-8b`, `hotpotqa_s00214__mistral-7b`, `hotpotqa_s00253__mistral-7b`, `hotpotqa_s00401__mistral-7b`, `hotpotqa_s00859__mistral-7b`, `hotpotqa_s01756__mistral-7b`, `truthfulqa_s00277__mistral-7b`, `hotpotqa_s00707__gemma-2-9b`, `truthfulqa_s00414__gemma-2-9b`.

## 1.0.0 (2026-10-05)

Initial release.
