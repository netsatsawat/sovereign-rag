"""Stage 1, retrieval layer — three retrievers, two chunkings, hard metrics.

This is the first run against the actual study grid, and it exists because E4
showed Hits@10 saturated (0.59-point spread across six configurations). The
fix is metrics that ask a harder question:

  hits@10      any gold document in the top-10 chunks       (the saturated one,
               kept for continuity with E4)
  hits@1       the single top chunk belongs to a gold doc
  strict@10    EVERY gold document for the query appears among the top-10
               chunks' documents. Multi-hop queries carry 2-4 gold docs, so
               this is the metric that actually matches the benchmark's claim
  mrr@10       rank of the first gold document

Retrievers, all over the identical committed chunk text (fairness contract):

  bm25         in-repo implementation, postings-verified against brute force
  dense        qwen3-embedding vectors, cosine, precomputed by embed_index.py
  hybrid       reciprocal-rank fusion of the two, k_rrf=60 (the standard
               constant from the RRF paper, not tuned here)

Reportability: for each stratum and metric, the bm25-vs-dense and hybrid-vs-
best-single deltas carry a paired-bootstrap 95% CI over queries. A delta whose
CI includes zero is flagged reportable=false, and the renderer greys it — the
noise floor lives here, never in JavaScript (FR-U3).

    python src/score_retrieval.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bm25 import BM25

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "stage1_retrieval.json"
K = 10
K_RRF = 60


def load_queries(docs: set[str]) -> list[dict]:
    rows = pq.read_table(ROOT / "data" / "multihoprag_queries.parquet").to_pylist()
    out = []
    for r in rows:
        gold = {e["url"] for e in (r.get("evidence_list") or []) if e.get("url") in docs}
        # Only queries whose ENTIRE evidence set lives in the corpus are
        # scorable under strict@10 -- a query with one gold doc outside the
        # subset can never satisfy "all evidence retrieved" and would count
        # as a miss against every retriever equally, diluting rather than
        # discriminating.
        full = {e["url"] for e in (r.get("evidence_list") or []) if e.get("url")}
        if gold and gold == full:
            out.append({"query": r["query"], "type": r.get("question_type") or "unknown",
                        "gold": gold})
    return out


def doc_ranks(chunk_order: list[int], doc_of: list[str]) -> list[str]:
    """Top-k chunk list -> ordered unique documents."""
    seen, docs = set(), []
    for i in chunk_order:
        d = doc_of[i]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def metrics_for(top_chunks: list[int], doc_of: list[str], gold: set[str]) -> dict:
    docs = doc_ranks(top_chunks, doc_of)
    hit1 = doc_of[top_chunks[0]] in gold if top_chunks else False
    hit10 = any(d in gold for d in docs)
    strict = gold.issubset(set(docs))
    rr = 0.0
    for rank, d in enumerate(docs, 1):
        if d in gold:
            rr = 1.0 / rank
            break
    return {"hits1": hit1, "hits10": hit10, "strict10": strict, "rr": rr}


def boot_delta(a: list[float], b: list[float], n: int = 3000, seed: int = 0):
    """Paired bootstrap CI on mean(a) - mean(b) over queries."""
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    means = []
    for _ in range(n):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(statistics.fmean(s))
    means.sort()
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n)]
    return round(statistics.fmean(diffs), 4), round(lo, 4), round(hi, 4)


def main() -> None:
    qtext_vecs: dict[str, np.ndarray] = {}
    results: dict = {"k": K, "k_rrf": K_RRF, "configs": {}, "deltas": {}}

    for budget in (600, 1200):
        chunks = [json.loads(l) for l in (ROOT / "data" / f"chunks_{budget}.jsonl").open()]
        doc_of = [c["doc_id"] for c in chunks]
        docs = set(doc_of)
        qs = load_queries(docs)

        # dense side
        C = np.load(ROOT / "data" / f"emb_{budget}.npy")
        meta = json.loads((ROOT / "data" / f"emb_{budget}.meta.json").read_text())
        assert meta["doc_ids"] == doc_of, "embedding matrix out of step with chunk file"

        key = "queries"
        if key not in qtext_vecs:
            import sys
            sys.path.insert(0, str(ROOT / "src"))
            from embed_index import embed_all
            qtext_vecs[key] = embed_all([q["query"] for q in qs], "queries")
            results["query_count_600"] = len(qs)
        Q = qtext_vecs[key]
        if Q.shape[0] != len(qs):        # 1200 subset differs in scorable queries
            from embed_index import embed_all
            Q = embed_all([q["query"] for q in qs], f"queries@{budget}")

        t0 = time.time()
        sims = Q @ C.T
        topk_d = np.argpartition(-sims, K, axis=1)[:, :K]
        order_d = np.take_along_axis(
            topk_d, np.argsort(-np.take_along_axis(sims, topk_d, axis=1), axis=1), axis=1)
        dense_s = time.time() - t0

        # lexical side
        t0 = time.time()
        idx = BM25([c["text"] for c in chunks])
        build_s = time.time() - t0
        t0 = time.time()
        bm_tops = [idx.top_k(q["query"], max(K, 50)) for q in qs]
        bm25_s = time.time() - t0

        per_query: dict[str, list[dict]] = {"bm25": [], "dense": [], "hybrid": []}
        for qi, q in enumerate(qs):
            bm = bm_tops[qi]
            dn = list(order_d[qi])
            per_query["bm25"].append(metrics_for(bm[:K], doc_of, q["gold"]))
            per_query["dense"].append(metrics_for(dn, doc_of, q["gold"]))
            # RRF over chunk ranks from both lists
            score: dict[int, float] = defaultdict(float)
            for rank, i in enumerate(bm[:50], 1):
                score[i] += 1.0 / (K_RRF + rank)
            for rank, i in enumerate(dn, 1):
                score[i] += 1.0 / (K_RRF + rank)
            fused = sorted(score, key=lambda i: (-score[i], i))[:K]
            per_query["hybrid"].append(metrics_for(fused, doc_of, q["gold"]))

        for arm, rows in per_query.items():
            by_type: dict[str, list[dict]] = defaultdict(list)
            for q, r in zip(qs, rows):
                by_type[q["type"]].append(r)
            cfg = {
                "budget": budget, "arm": arm, "queries": len(qs),
                "overall": {m: round(100 * statistics.fmean(r[m] for r in rows), 2)
                            for m in ("hits1", "hits10", "strict10")}
                | {"mrr10": round(statistics.fmean(r["rr"] for r in rows), 4)},
                "by_type": {t: {m: round(100 * statistics.fmean(r[m] for r in v), 2)
                                for m in ("hits1", "hits10", "strict10")}
                            | {"mrr10": round(statistics.fmean(r["rr"] for r in v), 4),
                               "n": len(v)}
                            for t, v in sorted(by_type.items())},
                "cost_s": {"index_build": round(build_s, 1) if arm == "bm25" else
                           (meta["wall_s"] if arm == "dense" else None),
                           "query_total": round(bm25_s if arm == "bm25" else
                                                dense_s if arm == "dense" else
                                                bm25_s + dense_s, 2)},
            }
            results["configs"][f"{budget}_{arm}"] = cfg
            o = cfg["overall"]
            print(f"  {budget:5} {arm:7} hits@1 {o['hits1']:6.2f}  hits@10 {o['hits10']:6.2f}  "
                  f"strict@10 {o['strict10']:6.2f}  mrr {o['mrr10']:.4f}", flush=True)

        # reportability: paired deltas per stratum, strict@10 and hits1
        for metric in ("strict10", "hits1"):
            for pair in (("hybrid", "bm25"), ("hybrid", "dense"), ("bm25", "dense")):
                a = [float(r[metric]) for r in per_query[pair[0]]]
                b = [float(r[metric]) for r in per_query[pair[1]]]
                d, lo, hi = boot_delta(a, b)
                results["deltas"][f"{budget}_{pair[0]}_vs_{pair[1]}_{metric}"] = {
                    "delta_pts": round(100 * d, 2),
                    "ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)],
                    "reportable": not (lo <= 0 <= hi),
                }
                by_t: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
                for q, ra, rb in zip(qs, per_query[pair[0]], per_query[pair[1]]):
                    by_t[q["type"]][0].append(float(ra[metric]))
                    by_t[q["type"]][1].append(float(rb[metric]))
                for t, (aa, bb) in by_t.items():
                    d2, lo2, hi2 = boot_delta(aa, bb)
                    results["deltas"][f"{budget}_{pair[0]}_vs_{pair[1]}_{metric}_{t}"] = {
                        "delta_pts": round(100 * d2, 2),
                        "ci95_pts": [round(100 * lo2, 2), round(100 * hi2, 2)],
                        "reportable": not (lo2 <= 0 <= hi2), "n": len(aa),
                    }

    # the pre-registered question, at the retrieval layer: chunk size vs arm
    for arm in ("bm25", "dense", "hybrid"):
        a = results["configs"].get(f"600_{arm}")
        b = results["configs"].get(f"1200_{arm}")
        if a and b:
            results[f"chunk_size_effect_{arm}"] = {
                m: round(a["overall"][m] - b["overall"][m], 2)
                for m in ("hits1", "hits10", "strict10")}

    OUT.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
