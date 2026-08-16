"""The graph arm, retrieval level: scored on the same harness as Stage 1.

Local-mode graph retrieval, gleanings disabled, per the PRD's arm definition:

  1. entity matching  BM25 over per-entity documents (name + majority type +
                      up to five descriptions), query -> top-E entities
  2. expansion        one hop over weighted edges, neighbor contribution
                      damped by 0.5 and scaled by edge weight share
  3. chunk scoring    each chunk scores the sum of its matched entities'
                      scores; provenance comes from the extraction, so a
                      chunk can only be reached through entities actually
                      found in it
  4. metrics          identical metrics_for / strict@k as Stage 1, same
                      queries, same fairness contract

Everything is deterministic: BM25 ties break on entity index, chunk ties on
chunk index. The graph arm retrieves through the graph or not at all; no
fallback to text BM25, because a fallback would measure the fallback.

    python src/score_graph.py --budget 600 [--smoke]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from bm25 import BM25
from score_retrieval import load_queries, metrics_for

ROOT = Path(__file__).parent.parent
TOP_E = 10
DAMP = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="score against a partial graph; prints, writes nothing")
    args = ap.parse_args()

    g = json.loads((ROOT / "data" / f"graph_{args.budget}.build.json").read_text())
    if g["partial"] and not args.smoke:
        raise SystemExit("graph build is partial; rerun build_graph.py after the index completes")

    chunks = [json.loads(l) for l in (ROOT / "data" / f"chunks_{args.budget}.jsonl").open()]
    doc_of = [c["doc_id"] for c in chunks]
    chunk_ix = {f'{c["doc_id"]}#{c["chunk_index"]}': i for i, c in enumerate(chunks)}
    qs = load_queries(set(doc_of))

    names = sorted(g["entities"])
    ent_docs = [f'{n} {g["entities"][n]["type"]} ' + " ".join(g["entities"][n]["descriptions"])
                for n in names]
    ent_ix = {n: i for i, n in enumerate(names)}
    idx = BM25(ent_docs)

    nbrs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, e in g["edges"].items():
        a, b = key.split("||")
        nbrs[a].append((b, float(e["weight"])))
        nbrs[b].append((a, float(e["weight"])))

    rows = []
    unreachable = 0
    for q in qs:
        top_e = idx.top_k(q["query"], TOP_E)
        escore: dict[str, float] = {}
        for rank, ei in enumerate(top_e, 1):
            n = names[ei]
            s = 1.0 / rank                      # rank-based: robust across BM25 scales
            escore[n] = max(escore.get(n, 0.0), s)
            wsum = sum(w for _, w in nbrs[n]) or 1.0
            for m, w in nbrs[n]:
                escore[m] = max(escore.get(m, 0.0), DAMP * s * (w / wsum))
        cscore: dict[int, float] = defaultdict(float)
        for n, s in escore.items():
            for cid in g["entities"][n]["chunks"]:
                ci = chunk_ix.get(cid)
                if ci is not None:
                    cscore[ci] += s
        top = sorted(cscore, key=lambda i: (-cscore[i], i))[:args.k]
        if not top:
            unreachable += 1
        rows.append(metrics_for(top, doc_of, q["gold"]))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for q, r in zip(qs, rows):
        by_type[q["type"]].append(r)
    agg = lambda rs: {m: round(100 * statistics.fmean(r[m] for r in rs), 2)
                      for m in ("hits1", "hitsk", "strict")} | \
                     {"mrr": round(statistics.fmean(r["rr"] for r in rs), 4)}

    cfg = {"budget": args.budget, "k": args.k, "arm": "graph",
           "top_entities": TOP_E, "damping": DAMP,
           "graph_partial": g["partial"],
           "graph_stats": g["stats"],
           "chunks_excluded_from_graph": g["excluded_truncated_or_unparseable"],
           "queries": len(qs), "unreachable_queries": unreachable,
           "overall": agg(rows),
           "by_type": {t: agg(v) | {"n": len(v)} for t, v in sorted(by_type.items())},
           "per_query": {"strict": [int(r["strict"]) for r in rows],
                         "hits1": [int(r["hits1"]) for r in rows]}}

    o = cfg["overall"]
    tag = "SMOKE (partial graph, not a result)" if g["partial"] else "graph"
    print(f"  {tag} @{args.budget} k={args.k}: hits@1 {o['hits1']:.2f}  "
          f"hits@k {o['hitsk']:.2f}  strict@k {o['strict']:.2f}  mrr {o['mrr']:.4f}  "
          f"unreachable {unreachable}/{len(qs)}")
    for t, v in cfg["by_type"].items():
        print(f"    {t:22} n={v['n']:4}  strict {v['strict']:6.2f}")

    if not g["partial"]:
        # merge per-k, never overwrite the sibling k's result
        out = ROOT / "reports" / "stage2_graph.json"
        d = json.loads(out.read_text()) if out.exists() else {}
        d[f"k{args.k}"] = cfg
        out.write_text(json.dumps(d, indent=1))
        print(f"wrote {out} [k{args.k}]")


if __name__ == "__main__":
    main()
