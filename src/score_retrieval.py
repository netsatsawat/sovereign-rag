"""Stage 1, retrieval layer: three retrievers, two chunkings, hard metrics.

v2, after adversarial verification of v1 found four defects in this file:

  RRF depth asymmetry   v1 fused BM25 to depth 50 against dense capped at 10,
                        so ranks 11-50 were BM25-only and every hybrid cell was
                        a BM25-tilted truncated fusion. Both lists now fuse at
                        FUSE_DEPTH=50.
  orphan control block  the fixed-token control was computed ad hoc with no
                        committed producer, no CI and no reportable flag. It is
                        now a first-class config (k=5 at BOTH budgets, so the
                        slot-count effect and the chunk-size effect separate).
  unrecomputable CIs    query embeddings were re-embedded live on every run and
                        never saved. They are now written to data/emb_queries.*
                        once and loaded thereafter; per-query outcome vectors
                        are committed in the results JSON, so every CI can be
                        recomputed offline.
  stats-plan deviation  the PRD registers B=10,000 and Holm-Bonferroni across a
                        pre-registered family; v1 shipped B=3,000 and no
                        multiplicity control. Both now follow the plan, and
                        `reportable` means Holm-adjusted p < 0.05; CI bounds
                        are displayed but do not decide the flag.

Metrics per (budget, k, arm): hits@k (any gold doc), hits@1, strict@k (EVERY
gold doc present, multi-hop queries carry 2-4), mrr@k over deduped docs.
Dense/hybrid query cost includes query embedding time, reported separately.

    python src/score_retrieval.py
"""

from __future__ import annotations

import hashlib
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
FUSE_DEPTH = 50
K_RRF = 60
B_RESAMPLES = 10_000
GRID = [(600, 10), (1200, 10), (1200, 5), (600, 5)]
ARMS = ("bm25", "dense", "hybrid")


def load_queries(docs: set[str]) -> list[dict]:
    rows = pq.read_table(ROOT / "data" / "multihoprag_queries.parquet").to_pylist()
    out = []
    for r in rows:
        gold = {e["url"] for e in (r.get("evidence_list") or []) if e.get("url") in docs}
        full = {e["url"] for e in (r.get("evidence_list") or []) if e.get("url")}
        # strict@k is only meaningful when the whole evidence set is in-corpus
        if gold and gold == full:
            out.append({"query": r["query"], "type": r.get("question_type") or "unknown",
                        "gold": gold})
    return out


def query_vectors(qs: list[dict]) -> tuple[np.ndarray, float]:
    """Load the saved query matrix, or embed once and save it. The saved
    matrix is what makes every dense CI recomputable offline."""
    qhash = hashlib.sha256("\n".join(q["query"] for q in qs).encode()).hexdigest()[:16]
    npy = ROOT / "data" / "emb_queries.npy"
    meta_p = ROOT / "data" / "emb_queries.meta.json"
    if npy.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("queries_sha256") == qhash:
            return np.load(npy), float(meta.get("embed_wall_s", 0.0))
    from embed_index import embed_all
    t0 = time.time()
    Q = embed_all([q["query"] for q in qs], "queries")
    wall = round(time.time() - t0, 1)
    np.save(npy, Q)
    meta_p.write_text(json.dumps({
        "embedder": "qwen3-embedding:0.6b", "queries_sha256": qhash,
        "rows": int(Q.shape[0]), "dim": int(Q.shape[1]), "embed_wall_s": wall}))
    return Q, wall


def doc_ranks(chunk_order: list[int], doc_of: list[str]) -> list[str]:
    seen, docs = set(), []
    for i in chunk_order:
        d = doc_of[i]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def metrics_for(top_chunks: list[int], doc_of: list[str], gold: set[str]) -> dict:
    docs = doc_ranks(top_chunks, doc_of)
    rr = 0.0
    for rank, d in enumerate(docs, 1):
        if d in gold:
            rr = 1.0 / rank
            break
    return {"hits1": bool(top_chunks) and doc_of[top_chunks[0]] in gold,
            "hitsk": any(d in gold for d in docs),
            "strict": gold.issubset(set(docs)),
            "rr": rr}


def boot(diffs: list[float], n: int = B_RESAMPLES, seed: int = 0):
    """Paired percentile bootstrap on per-query differences; also a two-sided
    bootstrap p-value for the Holm family."""
    rng = random.Random(seed)
    m = len(diffs)
    means = []
    for _ in range(n):
        means.append(statistics.fmean(diffs[rng.randrange(m)] for _ in range(m)))
    means.sort()
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n)]
    ge = sum(1 for x in means if x >= 0) / n
    le = sum(1 for x in means if x <= 0) / n
    p = max(min(2 * min(ge, le), 1.0), 1.0 / n)
    return statistics.fmean(diffs), lo, hi, p


def holm(family: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: reportable flags across the whole delta family."""
    items = sorted(family.items(), key=lambda kv: kv[1])
    out, m = {}, len(items)
    still = True
    for i, (k, p) in enumerate(items):
        if still and p <= alpha / (m - i):
            out[k] = True
        else:
            still = False
            out[k] = False
    return out


def main() -> None:
    chunks_by_budget, doc_by_budget = {}, {}
    for b in (600, 1200):
        chunks_by_budget[b] = [json.loads(l) for l in (ROOT / "data" / f"chunks_{b}.jsonl").open()]
        doc_by_budget[b] = [c["doc_id"] for c in chunks_by_budget[b]]
    docs = set(doc_by_budget[600])
    assert docs == set(doc_by_budget[1200]), "budgets must cover identical documents"
    qs = load_queries(docs)

    Q, q_embed_s = query_vectors(qs)
    assert Q.shape[0] == len(qs), "saved query matrix out of step with query filter"

    results = {"fuse_depth": FUSE_DEPTH, "k_rrf": K_RRF, "bootstrap_resamples": B_RESAMPLES,
               "multiplicity": "holm-bonferroni across all deltas, alpha=0.05",
               "queries": len(qs), "query_embed_wall_s": q_embed_s,
               "configs": {}, "deltas": {}, "per_query": {}}

    per_query_cache: dict[tuple, list[dict]] = {}
    for budget, k in GRID:
        chunks, doc_of = chunks_by_budget[budget], doc_by_budget[budget]
        C = np.load(ROOT / "data" / f"emb_{budget}.npy")
        meta = json.loads((ROOT / "data" / f"emb_{budget}.meta.json").read_text())
        assert meta["doc_ids"] == doc_of, "embedding matrix out of step with chunk file"

        t0 = time.time()
        sims = Q @ C.T
        depth = min(FUSE_DEPTH, sims.shape[1] - 1)
        part = np.argpartition(-sims, depth, axis=1)[:, :depth]
        order_all = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1), axis=1)
        dense_s = time.time() - t0

        t0 = time.time()
        idx = BM25([c["text"] for c in chunks])
        build_s = time.time() - t0
        t0 = time.time()
        bm_tops = [idx.top_k(q["query"], FUSE_DEPTH) for q in qs]
        bm25_s = time.time() - t0

        pq_rows: dict[str, list[dict]] = {a: [] for a in ARMS}
        for qi, q in enumerate(qs):
            bm, dn = bm_tops[qi], list(order_all[qi])
            pq_rows["bm25"].append(metrics_for(bm[:k], doc_of, q["gold"]))
            pq_rows["dense"].append(metrics_for(dn[:k], doc_of, q["gold"]))
            score: dict[int, float] = defaultdict(float)
            for rank, i in enumerate(bm, 1):          # both lists at FUSE_DEPTH
                score[i] += 1.0 / (K_RRF + rank)
            for rank, i in enumerate(dn, 1):
                score[i] += 1.0 / (K_RRF + rank)
            fused = sorted(score, key=lambda i: (-score[i], i))[:k]
            pq_rows["hybrid"].append(metrics_for(fused, doc_of, q["gold"]))

        for arm in ARMS:
            rows = pq_rows[arm]
            per_query_cache[(budget, k, arm)] = rows
            by_type: dict[str, list[dict]] = defaultdict(list)
            for q, r in zip(qs, rows):
                by_type[q["type"]].append(r)
            agg = lambda rs: {m: round(100 * statistics.fmean(r[m] for r in rs), 2)
                              for m in ("hits1", "hitsk", "strict")} | \
                             {"mrr": round(statistics.fmean(r["rr"] for r in rs), 4)}
            cfg = {"budget": budget, "k": k, "arm": arm,
                   "overall": agg(rows),
                   "by_type": {t: agg(v) | {"n": len(v)} for t, v in sorted(by_type.items())},
                   "cost_s": {
                       "index_build": round(build_s, 1) if arm == "bm25" else meta["wall_s"],
                       "query_side": round(bm25_s, 2) if arm == "bm25"
                       else round(q_embed_s + dense_s, 2) if arm == "dense"
                       else round(bm25_s + q_embed_s + dense_s, 2),
                       "note": "dense/hybrid query_side includes one-time query embedding "
                               f"({q_embed_s}s); similarity search itself is {dense_s:.2f}s"}}
            results["configs"][f"{budget}_k{k}_{arm}"] = cfg
            o = cfg["overall"]
            print(f"  {budget:5} k={k:2} {arm:7} hits@1 {o['hits1']:6.2f}  "
                  f"hits@k {o['hitsk']:6.2f}  strict@k {o['strict']:6.2f}  mrr {o['mrr']:.4f}",
                  flush=True)
        # committed per-query outcome vectors: every CI recomputable offline
        results["per_query"][f"{budget}_k{k}"] = {
            a: {"strict": [int(r["strict"]) for r in pq_rows[a]],
                "hits1": [int(r["hits1"]) for r in pq_rows[a]]} for a in ARMS}
    results["query_types"] = [q["type"] for q in qs]

    # ---- delta family: arm-vs-arm within config, and chunk-size/slot contrasts
    pvals: dict[str, float] = {}

    def add_delta(name: str, a_rows, b_rows, metric: str, qs_filter=None):
        ix = range(len(qs)) if qs_filter is None else [i for i, q in enumerate(qs)
                                                      if q["type"] == qs_filter]
        diffs = [float(a_rows[i][metric]) - float(b_rows[i][metric]) for i in ix]
        d, lo, hi, p = boot(diffs)
        results["deltas"][name] = {
            "delta_pts": round(100 * d, 2),
            "ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)],
            "p_boot": round(p, 5), "n": len(diffs)}
        pvals[name] = p

    for budget, k in GRID:
        for a, b in (("hybrid", "bm25"), ("hybrid", "dense"), ("bm25", "dense")):
            ra, rb = per_query_cache[(budget, k, a)], per_query_cache[(budget, k, b)]
            for metric in ("strict", "hits1"):
                add_delta(f"{budget}_k{k}_{a}_vs_{b}_{metric}", ra, rb, metric)
                for t in sorted({q["type"] for q in qs}):
                    add_delta(f"{budget}_k{k}_{a}_vs_{b}_{metric}_{t}", ra, rb, metric, t)
    # the two contrasts the write-up leans on, as first-class paired deltas
    for arm in ARMS:
        add_delta(f"fixed_tokens_600k10_vs_1200k5_{arm}_strict",
                  per_query_cache[(600, 10, arm)], per_query_cache[(1200, 5, arm)], "strict")
        add_delta(f"matched_k5_1200_vs_600_{arm}_strict",
                  per_query_cache[(1200, 5, arm)], per_query_cache[(600, 5, arm)], "strict")
        add_delta(f"matched_k10_1200_vs_600_{arm}_strict",
                  per_query_cache[(1200, 10, arm)], per_query_cache[(600, 10, arm)], "strict")

    flags = holm(pvals)
    for name, ok in flags.items():
        results["deltas"][name]["reportable"] = ok

    OUT.write_text(json.dumps(results, indent=1))
    n_rep = sum(flags.values())
    print(f"\n  {len(pvals)} deltas, {n_rep} reportable after Holm-Bonferroni")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
