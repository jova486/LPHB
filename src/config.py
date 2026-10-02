"""Configuration constants for release-data validation."""

from __future__ import annotations

EXPECTED_ROW_COUNT = 900
EXPECTED_QID_COUNT = 300
EXPECTED_GENERATORS_PER_QID = 3

BASE_REQUIRED_COLUMNS = [
    "item_id",
    "qid",
    "qid_order",
    "dataset",
    "model",
    "question",
    "ref_str",
    "answer_text",
    "annotator1_label",
    "annotator2_label",
]

JUDGE_MODELS = [
    "gpt_5_mini",
    "gpt_5_nano",
    "gpt_5_4",
    "claude_opus_4_7",
    "gemma_2_9b_local",
    "qwen2_5_7b_local",
    "llama_3_8b_local",
]

PROMPTS = ["p1", "p2", "p3"]

AUTO_LABEL_COLUMNS = [
    f"label_{judge_model}_{prompt}"
    for judge_model in JUDGE_MODELS
    for prompt in PROMPTS
]

REQUIRED_COLUMNS = BASE_REQUIRED_COLUMNS + AUTO_LABEL_COLUMNS

BINARY_LABEL_COLUMNS = ["annotator1_label"]
TERNARY_LABEL_COLUMNS = ["annotator2_label", *AUTO_LABEL_COLUMNS]

FORBIDDEN_PUBLIC_COLUMN_PATTERNS = ["P2B", "P3B", "p2b", "p3b"]
