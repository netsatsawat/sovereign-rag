"""Assemble the knowledge graph from the extraction checkpoint.

Reads data/graph_{budget}.jsonl (one extraction per chunk, written by
ops/graph_index.py), merges entities across chunks by normalised name, and
aggregates relationships into weighted undirected edges. Truncated and
unparseable chunks are excluded and counted — they are a published number,
not a silent gap.

Outputs data/graph_{budget}.build.json:
  entities  name -> {type (majority vote), chunk_ids, mention_count}
  edges     "a||b" -> {weight (sum of strengths), count}
  stats     nodes, edges, degree distribution, coverage, exclusions

    python src/build_graph.py --budget 600
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--allow-partial", action="store_true",
                    help="build from an incomplete checkpoint (smoke tests only)")
    args = ap.parse_args()

    src = ROOT / "data" / f"graph_{args.budget}.jsonl"
    chunks_file = ROOT / "data" / f"chunks_{args.budget}.jsonl"
    n_chunks = sum(1 for _ in chunks_file.open())

    rows = [json.loads(l) for l in src.open()]
    seen_ids = {r["chunk_id"] for r in rows}
    if len(seen_ids) < n_chunks and not args.allow_partial:
        raise SystemExit(f"checkpoint has {len(seen_ids)}/{n_chunks} chunks; "
                         f"index still running (use --allow-partial for smoke tests)")

    ok = [r for r in rows if r.get("status") == "ok"]
    excluded = len(rows) - len(ok)

    ent_chunks: dict[str, set[str]] = defaultdict(set)
    ent_types: dict[str, Counter] = defaultdict(Counter)
    ent_desc: dict[str, list[str]] = defaultdict(list)
    ent_mentions: Counter = Counter()
    edges: dict[str, dict] = {}

    for r in ok:
        cid = r["chunk_id"]
        for e in r.get("entities", []):
            n = norm(e.get("name"))
            if not n:
                continue
            ent_chunks[n].add(cid)
            ent_types[n][e.get("type", "OTHER")] += 1
            ent_mentions[n] += 1
            d = (e.get("description") or "").strip()
            if d and len(ent_desc[n]) < 5:          # cap: descriptions repeat heavily
                ent_desc[n].append(d)
        for rel in r.get("relationships", []):
            a, b = norm(rel.get("source")), norm(rel.get("target"))
            if not a or not b or a == b:
                continue
            key = f"{a}||{b}" if a <= b else f"{b}||{a}"
            w = rel.get("strength") or 1
            if key not in edges:
                edges[key] = {"weight": 0, "count": 0}
            edges[key]["weight"] += int(w)
            edges[key]["count"] += 1

    # drop edges whose endpoints were never emitted as entities: an edge to a
    # phantom node is unusable for retrieval and counting it inflates the graph
    known = set(ent_chunks)
    phantom = [k for k in edges if not set(k.split("||")) <= known]
    for k in phantom:
        del edges[k]

    degree: Counter = Counter()
    for k in edges:
        a, b = k.split("||")
        degree[a] += 1
        degree[b] += 1

    out = {
        "budget": args.budget,
        "source_rows": len(rows),
        "excluded_truncated_or_unparseable": excluded,
        "chunks_covered": len({c for s in ent_chunks.values() for c in s}),
        "chunks_total": n_chunks,
        "partial": len(seen_ids) < n_chunks,
        "entities": {n: {"type": ent_types[n].most_common(1)[0][0],
                         "chunks": sorted(ent_chunks[n]),
                         "mentions": ent_mentions[n],
                         "descriptions": ent_desc[n]}
                     for n in sorted(ent_chunks)},
        "edges": edges,
        "stats": {
            "nodes": len(ent_chunks),
            "edges": len(edges),
            "phantom_edges_dropped": len(phantom),
            "mean_degree": round(statistics.fmean(degree.values()), 2) if degree else 0,
            "median_degree": statistics.median(degree.values()) if degree else 0,
            "isolated_nodes": len(known) - len(degree),
        },
    }
    dest = ROOT / "data" / f"graph_{args.budget}.build.json"
    dest.write_text(json.dumps(out))
    s = out["stats"]
    print(f"{'PARTIAL ' if out['partial'] else ''}graph @{args.budget}: "
          f"{s['nodes']} nodes, {s['edges']} edges "
          f"(dropped {s['phantom_edges_dropped']} phantom), "
          f"mean degree {s['mean_degree']}, {s['isolated_nodes']} isolated, "
          f"{excluded} chunks excluded, "
          f"coverage {out['chunks_covered']}/{n_chunks}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
