# Stage 1: the retrieval layer, measured

Run 9 Aug 2026 on the committed chunk files. Three retrievers over identical
chunk text (the fairness contract), a 2×2 grid of chunk budget × k, per
question type, 1,381 queries whose entire evidence set lies inside the corpus.
Generation arms pending; the graph index builds in the background, checkpointed.

**This file was rewritten after adversarial verification of its first version
found a fusion bug and three overclaims. The corrections are stated inline
rather than removed.** v1's headline "fusion actively hurts (−3.04
[−5.36, −0.72])" was an artifact of depth-asymmetric RRF: BM25 fused to depth
50 against dense capped at 10. With both lists fused at depth 50, the delta
shrinks to −1.52 [−3.69, 0.58], not reportable. The finding is retracted; what
survives is "fusion does not measurably help."

## The metric

**strict@k**: every gold document for the query present among the top-k
chunks' documents. Chosen after Stage 0 showed Hits@10 saturated (96.5-97.0%
across E4's six dense configs, 0.59-pt spread); declared in machine-measurement.md before
Stage 1 ran, selected post-hoc from two candidates. It is not pre-registered
and is not described as such. Stats follow the registered plan: paired
bootstrap, B=10,000, Holm-Bonferroni across all 105 deltas; `reportable`
means Holm-adjusted p<0.05. Per-query outcome vectors are committed in
`stage1_retrieval.json`, so every CI recomputes offline.

## Results, strict@k overall

| | BM25 | dense | hybrid (RRF, depth 50) |
|---|---|---|---|
| 600 @ k=10 | **37.51** | 30.12 | 35.99 |
| 1200 @ k=10 | **43.08** | 32.44 | 41.56 |
| 600 @ k=5 | **22.95** | 16.87 | 21.14 |
| 1200 @ k=5 | **25.34** | 18.75 | 22.95 |

## Findings

**1. BM25 leads in every cell, and fusing dense into it does not help.**
BM25−dense at 1200/k10: **+10.64 [8.04, 13.25]**, reportable. Hybrid−BM25 is
negative in all four cells but never clears the Holm bar (−1.52 at both
budgets, k=10). No per-stratum cell survives multiplicity in dense's favour.
Note what this is *not*: the pre-registered "BM25 wins single-fact lookup"
concerned a stratum this benchmark does not contain. All 1,381 queries are
multi-hop with 2-4 gold documents. What is measured is stronger and
different: BM25 wins *multi-hop evidence assembly* on this corpus.

**2. The budget convention decides the chunk-size winner, and the flip is a
slot effect, not a chunk-size effect.** At matched k, 1200-token chunks win
at both k=5 (+2.39 [1.01, 3.77]) and k=10 (+5.58 [3.91, 7.24]). At a fixed
~6,000-token budget, 600@k10 beats 1200@k5 by **+12.17 [10.35, 14.12]**,
because ten retrieval slots can hold ten distinct documents and five can hold
five, and strict@k needs 2-4. Larger chunks are better per chunk; more slots
are better per token. Any chunk-size claim that does not state its budget
convention is meaningless.

**3. Retriever choice moves strict@k about twice what chunk size does over
the range tested.** Retriever effect +10.64 vs chunk-size effect +5.58, same
metric, same queries. This does not support the projected headline (chunk
size dominating architecture), which came from a 150→1200 sweep; only
600→1200 is tested here. It rhymes with *Dissecting Agentic RAG* without replicating it: that paper ablated an agent's adaptive router, and
its winning fixed arm was itself hybrid retrieval. The shared direction is
narrower: added retrieval machinery must earn its keep, and here it did not.

## Cost, retrieval layer (600-budget cells; 1200 comparable)

BM25 index build 0.2 s; dense chunk embedding 119.5 s (one-time, saved).
Query side: BM25 3.2 s for all 1,381 queries; dense **≈41 s**, of which
query embedding is ~40 s (one-time, now saved to `data/emb_queries.npy`) and
similarity search 0.02 s. v1 published the 0.02 s alone, which was
apples-to-oranges; the embedding cost is the dense query cost.

## Artifacts

`stage1_retrieval.json`: configs, 105 deltas with CIs, bootstrap p-values,
Holm flags, and per-query outcome vectors (offline-recomputable CIs).
`crossover.html`: regenerated from that JSON by `src/make_crossover.py`;
greying reads the committed Holm flags. Producers for every stage0.json
block: `ops/evidence_severance.py` (severance + power control),
`ops/chunker_integrity.py` (chunker integrity), `ops/verify_bm25.py`,
`ops/repeat_determinism.py` (GPU-gated), `ops/fetch_data.py` (data
provenance, content-hash based). Pins in `requirements.txt`.
