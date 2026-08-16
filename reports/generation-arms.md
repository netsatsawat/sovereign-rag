# Stage 3: generation, full grid

10-11 Aug 2026. All 1,381 scorable queries × three arms, qwen3:8b, temperature
0, checkpointed through one GPU-contention pause (the user's own model took
the box; the grid yielded and resumed) and one hung-call crash (now recorded
outcomes, not dead runs). 3 transient errors retried to completion. Metric:
normalised containment, labelled pilot-grade in §7's terms; exact McNemar,
Holm-Bonferroni across the 12-test family; Wilson 95% intervals on rates.

## Results

| Arm | Containment | Wilson 95% | Abstained |
|---|---|---|---|
| closed-book (A0) | 39.54% | [36.99, 42.14] | 44.6% |
| **BM25 RAG** | **63.07%** | [60.49, 65.58] | 30.9% |
| graph RAG | 46.56% | [43.94, 49.20] | 47.3% |

| Pairwise (Holm-corrected) | Split | p | Verdict |
|---|---|---|---|
| plain vs A0 | 399 : 74 | ~0 | RAG lift is real: **+23.5 pts** |
| plain vs graph | 273 : 45 | ~0 | BM25 crushes the graph at generation too |
| graph vs A0 | 248 : 151 | 1e−06 | **Resolved from the pilot's p=0.20:** the graph does beat closed-book, by 7 pts after 21.2 h of indexing |

The graph arm's final ledger: 21.2 hours of index time buys +7.0 points over
doing nothing, while the 0.2-second index buys +23.5. At generation as at
retrieval, there is no mix where the graph pays.

## The synthesis bottleneck, confirmed at full n

| Stratum | n | A0 | plain | graph | evidence available (strict@10, BM25) |
|---|---|---|---|---|---|
| inference | 671 | 57.68 | **95.23** | 72.43 | 24.14 |
| comparison | 391 | 22.51 | 36.06 | 22.25 | **55.75** |
| temporal | 319 | 22.26 | 28.53 | 21.94 | 43.26 |

Read the last two columns against each other. On inference, where strict
evidence retrieval looked *worst*, generation reaches 95%. On comparison and
temporal, where retrieval delivered evidence *best*, answers barely clear
the closed-book floor, and the graph arm on comparison (22.25) sits exactly
at it (22.51). Retrieval hands the model the documents; an 8B generator
fails to synthesise across them. **The residual errors in this system are a
generation problem wearing a retrieval costume.**

Two mechanical notes: strict@10 undercounts usable
evidence for inference (partial evidence often suffices for single-entity
answers), which is part of why 24% evidence supports 95% answers; and
containment under-credits abstentions phrased differently, so abstention rates
are reported per arm.

## Provenance

Per-cell records in `stage3_pilot.jsonl` (retries supersede errors);
aggregates and tests in `stage3_final.json`; every number recomputes from the
jsonl with the scoring block in this commit.
