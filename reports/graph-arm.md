# Stage 2: the graph arm, measured

Run 10 Aug 2026. Index: 21.2 h wall clock on the M5, 1,214 chunks through
qwen3:8b extraction (gleanings disabled), 35 chunks (2.9%) excluded as
truncated/unparseable, better than E0's 4.8% projection. Graph: **12,604
entities, 18,350 edges** after dropping 2,046 phantom edges to never-emitted
entities; mean degree 3.17 (inside the 2.88-4.42 band where Leiden is
reproducible); coverage 1,178/1,214 chunks.

Retrieval: entity-BM25 over name+type+descriptions, one damped hop over
weighted edges, chunks reachable only through entities extracted from them.
Same queries, same metrics, same harness as Stage 1. Zero unreachable queries.

## The result

**Graph-mediated retrieval loses to plain BM25 in every cell, by a lot.**

| strict@10 | BM25 | dense | graph | graph − BM25 (95% CI) |
|---|---|---|---|---|
| overall | 37.51 | 30.12 | **13.32** | **−24.19 [−26.72, −21.72]** |
| comparison | 55.75 | 40.41 | 19.44 | −36.32 [−41.43, −31.20] |
| temporal | 43.26 | 38.87 | 20.69 | −22.57 [−27.90, −17.55] |
| inference | 24.14 | 20.42 | 6.26 | −17.88 [−21.01, −14.61] |

The prediction was that the graph would earn its keep on comparison and
temporal questions. It loses those two *worse* than it loses inference.
The index that took **21.2 hours** is beaten everywhere by an index that took
**0.2 seconds**.

At the retrieval layer, on this corpus, **the crossover does not exist**:
there is no question mix at which this graph arm pays for itself. The mix
slider now has a graph row, and it never wins.

## What this does and does not say

It is a lower bound on one implementation: entity-match local mode, one
config, 8B extractor, no gleanings, no community summaries. Microsoft-style
*global* mode (Leiden communities + summarisation) answers a different
question class and is a generation-stage feature, untested here and labelled
as such. A stronger extractor or community retrieval could close some gap;
they cannot be assumed to close a 24-point one.

The mechanism is visible in the arm's own numbers: hits@10 74.8% says the
graph *reaches* a right document often, but strict@10 13.3% says it rarely
assembles *all* evidence: entity hubs pull retrieval toward one salient
document and starve the second and third. Multi-hop evidence assembly is
exactly what the entity graph was supposed to buy, and it is where it is
weakest.

`p_boot` values are reported raw; these deltas join the study family at final
analysis rather than being Holm-corrected in isolation.

## Cost, complete

| | BM25 | dense | graph |
|---|---|---|---|
| Index build | 0.2 s | 119.5 s | **76,320 s (21.2 h)** |
| Truncation loss | n/a | n/a | 2.9% of chunks |
| strict@10 bought | 37.51 | 30.12 | 13.32 |
