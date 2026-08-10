"""Stage 3 pilot — generation, three arms, enough queries to size the grid.

Purpose: estimate π_d, the per-pair disagreement rate on binary correctness,
which is the parameter McNemar's power depends on (PRD §7: at π_d = 0.10,
~312 paired queries suffice; the pilot decides whether the full 1,381 grid is
necessary or wasteful).

Arms:
  a0      closed-book: the generator answers with no context at all. The
          leakage/parametric-knowledge floor every RAG delta must clear.
  plain   BM25@600, k=10 chunks as context (Stage 1's winning retriever).
  graph   the Stage 2 graph arm's top-10 chunks as context.

Correctness (pilot-grade, labelled as such): normalised containment — the
gold answer string, NFKC/casefold/punctuation-stripped, appears in the
generated answer. Median gold here is short and entity-like, which makes
containment the honest cheap metric; EM is logged alongside. The full study's
scoring plan (§7) still applies at final analysis.

Generation: temperature 0, seed 0, num_predict 64, think:false, one shared
answer-format prompt. Checkpointed per (arm, query); resumable.

    python ops/stage3_pilot.py --n 100
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq              # noqa: E402
from bm25 import BM25                     # noqa: E402


def load_queries_with_answers(docs: set[str]) -> list[dict]:
    """score_retrieval.load_queries plus the gold ANSWER string, same filter:
    only queries whose entire evidence set lies inside the corpus."""
    rows = pq.read_table(ROOT / "data" / "multihoprag_queries.parquet").to_pylist()
    out = []
    for r in rows:
        gold = {e["url"] for e in (r.get("evidence_list") or []) if e.get("url") in docs}
        full = {e["url"] for e in (r.get("evidence_list") or []) if e.get("url")}
        if gold and gold == full:
            out.append({"query": r["query"], "type": r.get("question_type") or "unknown",
                        "gold": gold, "answer": r.get("answer") or ""})
    return out

GEN = "qwen3:8b"
K = 10
OUT = ROOT / "reports" / "stage3_pilot.jsonl"
SUMMARY = ROOT / "reports" / "stage3_pilot.json"

PROMPT_RAG = """Answer the question using ONLY the context below. Be direct and brief:
give the answer itself, no preamble. If the context does not contain the answer,
reply exactly: insufficient information.

CONTEXT:
{context}

QUESTION: {q}

ANSWER:"""

PROMPT_CLOSED = """Answer the question directly and briefly: the answer itself, no
preamble. If you do not know, reply exactly: insufficient information.

QUESTION: {q}

ANSWER:"""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def generate(prompt: str) -> tuple[str, dict]:
    body = json.dumps({
        "model": GEN, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 64},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    meta = {"wall_s": round(time.time() - t0, 1),
            "prompt_tokens": d.get("prompt_eval_count", 0),
            "gen_tokens": d.get("eval_count", 0),
            "done_reason": d.get("done_reason")}
    return (d.get("response") or "").strip(), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in (ROOT / "data" / f"chunks_600.jsonl").open()]
    doc_of = [c["doc_id"] for c in chunks]
    qs_all = load_queries_with_answers(set(doc_of))

    # stratified proportional sample, deterministic: every ceil(len/type share)
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qs_all):
        by_type[q["type"]].append(i)
    sample_ix: list[int] = []
    for t, ixs in sorted(by_type.items()):
        want = max(1, round(args.n * len(ixs) / len(qs_all)))
        step = max(len(ixs) // want, 1)
        sample_ix.extend(ixs[::step][:want])
    sample_ix = sorted(sample_ix)[:args.n]

    # retrieval contexts
    idx = BM25([c["text"] for c in chunks])
    g = json.loads((ROOT / "data" / "graph_600.build.json").read_text())
    names = sorted(g["entities"])
    ent_docs = [f'{n} {g["entities"][n]["type"]} ' + " ".join(g["entities"][n]["descriptions"])
                for n in names]
    eidx = BM25(ent_docs)
    chunk_ix = {f'{c["doc_id"]}#{c["chunk_index"]}': i for i, c in enumerate(chunks)}
    nbrs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, e in g["edges"].items():
        a, b = key.split("||")
        nbrs[a].append((b, float(e["weight"])))
        nbrs[b].append((a, float(e["weight"])))

    def graph_topk(query: str, k: int) -> list[int]:
        escore: dict[str, float] = {}
        for rank, ei in enumerate(eidx.top_k(query, 10), 1):
            n = names[ei]
            s = 1.0 / rank
            escore[n] = max(escore.get(n, 0.0), s)
            wsum = sum(w for _, w in nbrs[n]) or 1.0
            for m, w in nbrs[n]:
                escore[m] = max(escore.get(m, 0.0), 0.5 * s * (w / wsum))
        cscore: dict[int, float] = defaultdict(float)
        for n, s in escore.items():
            for cid in g["entities"][n]["chunks"]:
                ci = chunk_ix.get(cid)
                if ci is not None:
                    cscore[ci] += s
        return sorted(cscore, key=lambda i: (-cscore[i], i))[:k]

    done: set[tuple[str, int]] = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                r = json.loads(line)
                if not str(r.get("done_reason", "")).startswith("error"):
                    done.add((r["arm"], r["qi"]))
            except Exception:
                pass

    t_start = time.time()
    with OUT.open("a") as fh:
        for n_i, qi in enumerate(sample_ix, 1):
            q = qs_all[qi]
            for arm in ("a0", "plain", "graph"):
                if (arm, qi) in done:
                    continue
                if arm == "a0":
                    prompt = PROMPT_CLOSED.format(q=q["query"])
                else:
                    top = idx.top_k(q["query"], K) if arm == "plain" \
                        else graph_topk(q["query"], K)
                    ctx = "\n\n".join(chunks[i]["text"] for i in top)
                    prompt = PROMPT_RAG.format(context=ctx, q=q["query"])
                try:
                    ans, meta = generate(prompt)
                except Exception as ex:
                    # A hung or failed call is a recorded outcome, not a dead
                    # run: the first full-grid attempt died at query ~493 when
                    # one call blew the client timeout.
                    ans, meta = "", {"wall_s": None, "prompt_tokens": 0,
                                     "gen_tokens": 0,
                                     "done_reason": f"error:{type(ex).__name__}"}
                fh.write(json.dumps({"qi": qi, "arm": arm, "type": q["type"],
                                     "gold": q["answer"], "answer": ans, **meta}) + "\n")
                fh.flush()
            if n_i % 10 == 0:
                el = time.time() - t_start
                print(f"  {n_i}/{len(sample_ix)} queries · {el/60:.0f}m elapsed", flush=True)
    print("pilot generation complete", flush=True)


if __name__ == "__main__":
    main()
