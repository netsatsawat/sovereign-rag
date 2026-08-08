# Stage 0 — the results

Run 8–9 Aug 2026 on one Apple M5, 24 GB unified, Ollama 0.32.5. Generator
`qwen3:8b` Q4_K_M, embedder `qwen3-embedding:0.6b` q8_0. No API key, no second
machine, no hosted call. Every number below is measured on this box; nothing is
projected from a token price.

Stage 0 exists to answer one question — *can this study run here, as specified* —
cheaply enough that the answer costs an evening rather than four nights of
indexing. **The answer is no, not as specified.** Three things must change
before Stage 1, and one of them is the primary metric.

---

## 1. Hardware: passes, with margin

| Gate | Measured | Verdict |
|---|---|---|
| Decode | 20.3–22.3 tok/s | ✅ projection was 20 |
| Prefill | 369–1,697 tok/s | ✅ projection was 350 |
| Resident, 4k/8k/16k ctx | 5.9 / 6.6 / 7.9 GB, **100% GPU at every size** | ✅ |
| Generator + embedder co-resident | 7.94 GB of 24 | ✅ |
| Cold load | 11.4 s | — |

The 8B choice is vindicated by one number: **it never touches CPU.** The 27B ran
11% on CPU at 4k context and 19% at 32k, and that split is the whole reason it
was 3× slower to decode and up to 17× slower to prefill. On 24 GB of unified
memory, "fits in RAM" and "fits on the GPU" are different questions.

## 2. Chunker: passes

Sentence-aligned greedy packing, Qwen3 BPE, article-scoped, zero overlap.

| Budget | Chunks | Mean tokens | Fill | Over budget | Sentences lost |
|---|---|---|---|---|---|
| 600 | 1,214 | 498.4 | 0.83 | 0 | **0** |
| 1200 | 694 | 859.2 | 0.72 | 0 | **0** |

Measured **4.612 chars/token** on this subset. The prior code assumed 4.0, so
every "512-token" chunk quoted earlier in this project was really ~446 tokens.
An independent measurement in the chunking research put it at 4.607.

**A bug worth recording.** BPE merges across sentence joins, so a joined chunk
can tokenize *longer* than the sum of its sentences — 4 of 1,213 chunks exceeded
a 600 budget. Packing on the sum is not sufficient; the packer re-counts the
real joined string and carries overflow sentences into the next chunk. The first
fix attempt dropped them instead, which would have deleted evidence.

## 3. Evidence severance: passes, better than predicted

**0.00% of 4,243 gold evidence spans severed, at both budgets.** Zero queries
affected.

That was too clean to accept, so the detector was run against schemes known to
sever:

| Scheme | Spans severed | Queries affected |
|---|---|---|
| naive 256 tok, no overlap | 15.15% | 36.88% |
| naive 512 tok, no overlap | 7.73% | 20.84% |
| naive 1024 tok, no overlap | 4.86% | 13.41% |
| naive 1024 tok, 128 overlap | 0.00% | 0.00% |
| **sentence-aligned, no overlap** | **0.00%** | **0.00%** |

The detector has power — severance rises monotonically as chunks shrink. So the
finding is real and it is the useful kind: **sentence alignment buys at zero
overlap what the naive scheme needs 512 characters of overlap to buy.** Overlap
costs ~12.5% more chunks and ~20% more LLM calls on the graph arm. Alignment
costs nothing.

## 4. Extraction cost: the claim I made was backwards

n = 22 comparable articles, paired, same text at both chunkings.

| | 600 relative to 1200 | 95% CI |
|---|---|---|
| Wall clock | **1.328×** | 1.164 – 1.498 |
| Emitted tokens | **1.353×** | 1.189 – 1.523 |

Both intervals exclude 1.0. **Smaller chunks cost more**, bounded.

This closes a claim made earlier in this project from a single chunk, which had
the sign inverted and was used to argue for restructuring the study. One
measurement was an outlier; nine refuted it; twenty-two bound it.

## 5. The edge finding, and it is the one worth publishing

Of **413 relationships present at 1200 and absent at 600**:

| Class | Count | Share |
|---|---|---|
| Boundary-severed — endpoints in different 600-chunks | 17 | **4.1%** |
| Within-chunk unemitted — both endpoints in one chunk it was shown | 279 | **67.6%** |
| Unclassified — endpoint not locatable by substring | 117 | 28.3% |

**Overlap and gleanings — the standard mitigations, shipped by every GraphRAG
implementation — address the 4%.**

The 68% is the model declining to state a relation whose two endpoints both sat
inside a single chunk it was given. Extraction here is deterministic — same
input, byte-identical output, verified three times — so this is not sampling
noise. It is context-dependence, and no overlap setting recovers it.

Counterweight, stated so the finding is not oversold: 600-chunking is not
strictly worse. It found **755 relationships the 1200 chunking missed** and 25%
more entities (1,024 vs 821). The trade is cost and coherence against recall.

## 6. E0: fails on truncation

189 extraction calls. **0 emitted thinking tokens**, so `think:false` works and
the cost model is not secretly measuring reasoning.

But **9 of 189 (4.8%) hit the 8,000-token cap** and returned unparseable JSON.
Payload size varies enough across news chunks that any fixed budget truncates
the dense ones. This is a standing failure mode of local graph extraction, not a
cap to raise past — the first pilot ran unbounded and blocked past a 600-second
client timeout.

## 7. E4: the primary metric is saturated

Six configurations — two chunk sizes × three header variants — scored on 1,521
queries.

| | Hits@10 | MRR@10 |
|---|---|---|
| Best | 97.04% | 0.7404 |
| Worst | 96.45% | 0.7094 |
| **Spread** | **0.59 pts** | 0.0310 |

**Hits@10 cannot discriminate on this subset.** 253 articles produce 694–1,214
chunks; retrieving 10 is ~1% of the corpus, and the queries were written from
those very articles.

This matters because it blocks the pre-registered headline. The claim that chunk
size moves the primary metric more than architecture does **cannot be tested
with this metric here**: the measured chunk-size effect on the dense arm is
0.20–0.59 points, against the 16.1 points the research projected for the hybrid
arm. Either the projection does not transfer to dense retrieval, or the ceiling
is hiding it.

**Also measured: prepending the title does not help dense retrieval.** −0.33 pts
at 600, +0.06 at 1200. The research measured +1.0 for title and +3.5 for
title+source on BM25. Neither transfers — BM25 wins by matching the literal
outlet string and an embedding gets no comparable windfall. The outlet-name
leakage is an **arm-asymmetric BM25 artifact**, which strengthens the case for
reporting the stripped configuration as primary.

Hardest stratum even at the ceiling: **comparison, 94.04%**, three points below
inference and temporal.

---

## What must change before Stage 1

1. **A harder primary metric.** Strict all-evidence-retrieved, or Hits@1. MRR@10
   already shows 4× the spread of Hits@10 and is the cheap interim.
2. **Probably the full 609-article corpus** rather than the tech+business subset
   — 2.4× the distractors, at proportional graph-index cost.
3. **A corpus-hour budget, not a per-chunk gate.** A 90-second per-chunk gate
   evaluated on the mean is not a gate when 4.8% of calls truncate outright.

## What Stage 0 cost

About five hours of machine time, most of it E2/E3's 223 minutes. It caught a
saturated primary metric, an inverted cost claim, a chunker bug that would have
silently dropped evidence, and a 4.8% truncation rate — each of which would have
been discovered after an overnight index instead.
