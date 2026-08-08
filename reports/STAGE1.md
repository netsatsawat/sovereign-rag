# Stage 1 — the retrieval layer, measured

Run 9 Aug 2026, immediately after Stage 0, on the same machine and the same
committed chunk files. Three retrievers over identical chunk text (the
fairness contract), two chunk budgets, per question type, 1,381 queries whose
entire evidence set lies inside the corpus. Generation arms are pending; the
graph index is building in the background (`ops/graph_index.py`, checkpointed).

## The metric fix worked

Stage 0 found Hits@10 saturated: 0.59 points of spread across six
configurations. **strict@10** — every gold document for the query present in
the top-10 chunks — spreads **30.1% to 43.1%** over the same grid. The metric
now discriminates, on the same corpus, without paying 2.4× for the full-609
graph index. The full-corpus option stays open but is no longer forced.

## Results, strict@10 overall

| | BM25 | dense | hybrid (RRF) |
|---|---|---|---|
| 600 @ k=10 | **37.51** | 30.12 | 35.92 |
| 1200 @ k=10 | **43.08** | 32.44 | 40.04 |
| 1200 @ k=5 (fixed-token control) | **25.34** | 18.75 | 22.74 |

Per stratum, the spread is larger: comparison 61.4% (BM25@1200) against
inference 20.0% (dense@600). The stratum decides the difficulty; the
retriever decides the rank.

## Three findings

**1. BM25 wins everywhere, and fusing it with dense retrieval hurts.**
Hybrid−BM25 at 1200 is **−3.04 points, 95% CI [−5.36, −0.72]** — reportable,
and negative. At 600 it is −1.59 [−3.62, 0.51], below the noise floor and
greyed. This replicates the *Dissecting Agentic RAG* direction at the fusion
level on a different corpus: the cleverer component costs points. The
pre-registered prediction that BM25 wins single-fact-adjacent traffic
survives contact with measurement.

**2. The chunk-size effect flips sign depending on what you hold fixed.**
At fixed k=10, 1200-token chunks beat 600 by ~5.6 points — but they read 2×
the tokens. At a fixed ~6,000-token budget, 600@k10 beats 1200@k5 by **12.2
points** (37.51 vs 25.34). §4.5.7 predicted exactly this confound; it is now
measured, and it means any chunk-size claim must state its budget convention
or it is meaningless.

**3. The pre-registered headline is NOT supported at this range, and that is
reported rather than buried.** The research projected chunk size moving the
primary metric more than architecture. Measured at fixed k across 600→1200:
chunk-size effect ≤5.6 points, retriever effect (BM25−dense at 1200) **10.64
points [8.04, 13.18]**. Over the range tested, retriever choice moves
strict@10 roughly twice as much as chunk size. The projection came from a
150→1200 sweep; only 600→1200 is tested here, so the wider claim remains
open — but at the sizes a practitioner would actually pick between, the
architecture-shaped decision dominates.

## Cost, retrieval layer

BM25 index build: 0.2 s. Dense index: 119.5 s of embedding (one-time,
saved to disk). All-query retrieval: BM25 3.2 s, dense 0.02 s.

## Artifacts

`reports/stage1_retrieval.json` — every number, with paired-bootstrap 95% CIs
and precomputed `reportable` flags. `reports/crossover.html` — single-file,
no-network rendering with the traffic-mix slider; regenerate with
`python src/make_crossover.py`. Deltas whose CI includes zero are greyed by
flag, never recomputed in the page.
