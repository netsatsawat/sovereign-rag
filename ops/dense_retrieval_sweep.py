"""E4: price the chunk-size confound on the dense arm, and price the title.

Two questions this settles without running a single generation:

  1. How much does chunk size move dense retrieval? If it moves it more than
     the architecture margin does, that is the study's headline and it must be
     pre-registered rather than discovered.
  2. Is prepending the article title a real retrieval gain, or is it the
     benchmark leaking? MultiHop-RAG queries name their source outlet in
     ~92-100% of cases, so `source` is a known oracle. `title` might be too.

Three header variants x two chunkings, scored on the same queries:

  body          chunk text with no header at all
  title         title + "\\n\\n" + body   (the PRD's choice)
  title+source  adds the outlet name     (expected to be leakage)

Gold is the query's evidence URLs. A hit is a retrieved chunk belonging to a
gold document.

Landmine, documented in PRD 4.5.6 and respected here: passing `options` to
/api/embed kills the Ollama runner: the triggering call returns fine and
every later embed fails with EOF. So no options are sent, and
prompt_eval_count is asserted instead.

    python ops/dense_retrieval_sweep.py
"""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "stage0.json"
E4 = ROOT / "reports" / "e4_embed.json"
EMB = "qwen3-embedding:0.6b"
K = 10
BATCH = 32


def embed(texts: list[str]) -> list[list[float]]:
    """No `options`; see the module docstring."""
    body = json.dumps({"model": EMB, "input": texts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    if "embeddings" not in d:
        raise RuntimeError(f"embed failed: {str(d)[:200]}")
    return d["embeddings"]


def embed_all(texts: list[str], label: str) -> list[list[float]]:
    out: list[list[float]] = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        out.extend(embed(texts[i:i + BATCH]))
        if i and i % (BATCH * 20) == 0:
            print(f"    {label}: {i}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)
    return out


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def main() -> None:
    qrows = pq.read_table(ROOT / "data" / "multihoprag_queries.parquet").to_pylist()

    results: dict[str, dict] = {}
    for budget in (600, 1200):
        chunks = [json.loads(l) for l in (ROOT / "data" / f"chunks_{budget}.jsonl").open()]
        docs = {c["doc_id"] for c in chunks}

        # queries whose evidence lives inside this subset
        qs = []
        for r in qrows:
            gold = {e["url"] for e in (r.get("evidence_list") or []) if e.get("url") in docs}
            if gold:
                qs.append({"query": r["query"], "type": r.get("question_type") or "unknown",
                           "gold": gold})
        print(f"budget {budget}: {len(chunks)} chunks, {len(qs)} scorable queries")

        qraw = embed_all([q["query"] for q in qs], f"queries@{budget}")
        qvecs = np.asarray(qraw, dtype=np.float32)
        qvecs /= (np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-12)

        for variant in ("body", "title", "title+source"):
            texts = []
            for c in chunks:
                t = c["text"]
                title = c.get("title") or ""
                if title and t.startswith(title + "\n\n"):
                    t = t[len(title) + 2:]          # strip the header the chunker added
                if variant == "title":
                    t = f"{title}\n\n{t}" if title else t
                elif variant == "title+source":
                    src = c.get("source") or ""
                    # The separator is embedded text, not prose: the committed
                    # e4_embed_sweep numbers were measured with this exact
                    # string. Changing it changes the vectors and the result.
                    head = " — ".join(x for x in (title, src) if x)
                    t = f"{head}\n\n{t}" if head else t
                texts.append(t)

            cvecs = embed_all(texts, f"{variant}@{budget}")
            dim = len(cvecs[0])

            # Brute force in pure Python is ~1.9e9 multiply-adds per variant.
            # Vectors are L2-normalised, so cosine is one matrix product.
            C = np.asarray(cvecs, dtype=np.float32)
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            Q = np.asarray(qvecs, dtype=np.float32)
            sims = Q @ C.T                                   # (queries, chunks)
            topk = np.argpartition(-sims, K, axis=1)[:, :K]
            order = np.take_along_axis(
                topk, np.argsort(-np.take_along_axis(sims, topk, axis=1), axis=1), axis=1)
            doc_of = [c["doc_id"] for c in chunks]

            hits = 0
            rr_sum = 0.0
            per_type = defaultdict(lambda: {"n": 0, "hits": 0, "rr": 0.0})
            for qi, q in enumerate(qs):
                top = [doc_of[i] for i in order[qi]]
                hit = any(d in q["gold"] for d in top)
                rr = 0.0
                for rank, d in enumerate(top, 1):
                    if d in q["gold"]:
                        rr = 1.0 / rank
                        break
                hits += hit
                rr_sum += rr
                pt = per_type[q["type"]]
                pt["n"] += 1; pt["hits"] += hit; pt["rr"] += rr

            key = f"{budget}_{variant}"
            results[key] = {
                "budget": budget, "variant": variant, "dim": dim,
                "queries": len(qs),
                "hits_at_10": round(100 * hits / len(qs), 2),
                "mrr_at_10": round(rr_sum / len(qs), 4),
                "by_type": {t: {"n": v["n"],
                                "hits_at_10": round(100 * v["hits"] / v["n"], 2),
                                "mrr_at_10": round(v["rr"] / v["n"], 4)}
                            for t, v in sorted(per_type.items())},
            }
            r = results[key]
            print(f"  {budget:5} {variant:13} Hits@10 {r['hits_at_10']:6.2f}%  "
                  f"MRR@10 {r['mrr_at_10']:.4f}", flush=True)

    # the two comparisons this experiment exists to make
    summary = {"variants": results}
    for v in ("body", "title", "title+source"):
        a, b = results.get(f"600_{v}"), results.get(f"1200_{v}")
        if a and b:
            summary[f"chunk_size_effect_{v}"] = {
                "hits_600": a["hits_at_10"], "hits_1200": b["hits_at_10"],
                "delta_points": round(a["hits_at_10"] - b["hits_at_10"], 2)}
    for budget in (600, 1200):
        base = results.get(f"{budget}_body")
        if not base:
            continue
        for v in ("title", "title+source"):
            r = results.get(f"{budget}_{v}")
            if r:
                summary[f"header_effect_{v}_at_{budget}"] = round(
                    r["hits_at_10"] - base["hits_at_10"], 2)

    E4.write_text(json.dumps(summary, indent=1))
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d["e4_embed_sweep"] = summary
    OUT.write_text(json.dumps(d, indent=1))

    print("\n  chunk-size effect (600 minus 1200), Hits@10 points:")
    for v in ("body", "title", "title+source"):
        k = f"chunk_size_effect_{v}"
        if k in summary:
            print(f"    {v:13} {summary[k]['delta_points']:+.2f}")
    print("  header effect vs body-only, Hits@10 points:")
    for k in sorted(x for x in summary if x.startswith("header_effect")):
        print(f"    {k.replace('header_effect_',''):22} {summary[k]:+.2f}")
    print(f"\nwrote {E4}")


if __name__ == "__main__":
    main()
