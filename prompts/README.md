# Judge and extraction prompts

Verbatim prompt texts extracted from the notebook (Section B cell definitions).
Each `.txt` file contains only the prompt text — no surrounding Python.

| File | Prompt | Criterion | Structure | Role |
|---|---|---|---|---|
| `p1.txt` | P1 | Reference faithfulness | Terse | Main judge prompt (Appendix B). Run on all seven judges over all 900 items. Defined in CELL B1. |
| `p2.txt` | P2 | Factual correctness | Structured | Main judge prompt (Appendix B). Run on all seven judges over all 900 items. Defined in CELL B1. |
| `p3.txt` | P3 | Factual correctness (extended) | Structured | Main judge prompt (Appendix B); extends P2's decision tree with a special-case rule for non-committal/refusal reference answers. Defined in CELL B1. |
| `p1s.txt` | P1s | Reference faithfulness | Structured | Factorial control only (CELL B8). Applies P1's criterion in P2's structured format, isolating the effect of format from the effect of criterion. |
| `p2t.txt` | P2t | Factual correctness | Terse | Factorial control only (CELL B8). Applies P2's criterion in P1's terse format, isolating the effect of format from the effect of criterion. |
| `core_extraction.txt` | — | — (not a judge prompt) | — | Extracts the "core answer" from an original answer for the elaboration intervention, removing elaboration while preserving all answer-bearing content. Defined in CELL B10. |

The 2×2 factorial design (CELL B8/C9) crosses criterion against structure using
only P1, P1s, P2t, and P2:

|             | terse | structured |
|---|---|---|
| **reference faithfulness** | p1  | p1s |
| **factual correctness**    | p2t | p2  |

P3 and the core-extraction prompt sit outside this grid: P3 is a third
main-run judge prompt (not a factorial control), and the core-extraction
prompt performs answer extraction rather than hallucination judgment.
