# Stage 7 — the cross-domain replication (HotpotQA)

Stages 1–5 crowned BM25 on one corpus. This stage asks the only question
that decides whether that crowning means anything: does it transfer? Same
retrievers, same strict metric, a corpus from a different domain —
HotpotQA distractor dev: 1,500 multi-hop questions over a pooled
14,526-paragraph Wikipedia haystack (10× the news corpus), questions
about the world rather than about articles.

## The winner flips

| retriever | strict@10 | hits@10 |
|---|---|---|
| BM25 | 68.40 | 98.93 |
| dense (qwen3-embedding:0.6b) | **74.80** | — |

Paired per question: **dense +6.40 [3.93, 8.87]**. On MultiHop-RAG news,
BM25 beat dense by +7.4 to +12.95 across three configurations and an
oracle control; here dense wins. Neither retriever is "the winner."

By question type (BM25): bridge 62.11 (n=1,164), comparison 90.18
(n=336). Answer shape: 6.1% yes/no — this corpus does not have the
constant-answer pathology the news comparison stratum had.

## The finding that replicated instead

Hits@10 is saturated on BOTH corpora — 98.93 here, 96.7–99.3 on the news
corpus — while strict@10 discriminates on both. The metric lesson
transfers even though the winner does not. That pairing is the study's
most durable result: the architecture verdict is a property of the
corpus; the evaluation discipline is not.

## Recomputing

`reports/stage7_hotpot.json` carries per-query strict vectors for both
retrievers (`per_query_strict`, `per_query_strict_dense`, index-aligned,
n=1,500), so the paired delta and CI recompute without any model:

```python
import json, random
d = json.load(open("reports/stage7_hotpot.json"))
b, e = d["per_query_strict"], d["per_query_strict_dense"]
diffs = [ei - bi for bi, ei in zip(b, e)]
print("delta", 100 * sum(diffs) / len(diffs))
rng = random.Random(0)
boots = sorted(100 * sum(rng.choices(diffs, k=len(diffs))) / len(diffs)
               for _ in range(10_000))
print("ci95", boots[249], boots[9749])
```

(The delta reproduces exactly; the CI bounds wobble by ~0.1 with the
resampling seed — the committed [3.93, 8.87] came from the study's
B=10,000 pipeline.)

## Provenance debt, stated

The run script for this stage was not committed — the corpus fetch
(`data/hotpot_dev.parquet`, hash in `data/sources.manifest.json`), the
embedding pass (`reports/hotpot_dense.log`), and the scoring were driven
in-session and only the artifacts landed. The per-query vectors above are
the ground truth of what was measured; a `ops/stage7_hotpot.py` in the
committed-producer pattern belongs on the same debt list as the
multilingual prep script (see STAGE9.md).
