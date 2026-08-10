# Stage 4 — the learning layer: a pre-registered negative

11 Aug 2026. Retrieval-feedback re-ranking (L1): document-level credit
accumulated from a 60% learning stream with ε-greedy exploration and clipped
IPW, applied to BM25 re-ranking at λ=1.0, five seeds, evaluated on held-out
queries split into "similar" (shares ≥1 gold document with the stream) and
"novel" (zero overlap). Four gates, declared before the run.

## The verdict

**The learning claim may not be made.** On this corpus, with this mechanism:

| Contrast (strict@10, pooled) | Δ | 95% CI | p |
|---|---|---|---|
| learned vs base, similar queries | **−1.40** | [−2.43, −0.33] | 0.011 |
| learned vs shuffled-credit, similar | −0.40 | [−1.54, 0.74] | **0.49** |
| learned vs base, novel queries | 0.00 | — | 1.0 |
| popularity-only vs base, similar | −26.15 | [−28.25, −24.09] | 0.0001 |

Reading those in order: feedback re-ranking makes similar queries slightly
but significantly **worse**; its effect is **indistinguishable from randomly
assigned credit** (the shuffled control), so what moves is bias, not signal;
novel queries are untouched (no credit applies); and the query-blind
popularity counter — destructive control #1 — collapses exactly as a
destructive control should.

| Gate | Result |
|---|---|
| G1 better on similar queries | **FAIL** (−1.40, wrong direction) |
| G2 not worse on novel | pass (exactly 0) |
| G3 beats the popularity counter | pass (+24.75) |
| G4 effect distinguishable from shuffled credit | **FAIL** (p=0.49) |

## Why it fails — the mechanism, not a mystery

Document-level credit is a *prior toward previously useful documents*. On a
corpus where BM25 already ranks well, adding that prior displaces the current
query's correct documents with historically successful ones — the
rich-get-richer failure this study's own research predicted, now measured.
Credit carries no information about *this* query; the shuffled control proves
it: permuting which documents hold the credit changes nothing detectable.

## Honest bounds

The similar-queries conclusion is well-powered (~2,700 pooled pairs). The
novel split is **underpowered** (5–11 queries per seed): gold documents are so
concentrated that nearly every held-out query shares evidence with a 60%
stream — itself a finding about this benchmark's structure. And this
falsifies one mechanism at one operating point (document-level credit,
λ=1.0, on top of a strong lexical baseline at strict@10); finer-grained
credit, weaker baselines, or query-conditioned memory are different claims.

The original requirement — "the system should get better every time a
similar query is asked" — gets a measured answer for the mechanism most
commonly deployed to satisfy it: **no, and slightly the opposite.**
