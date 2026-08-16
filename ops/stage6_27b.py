"""Stage 6: the causal test of the synthesis-bottleneck claim.

The claim: on comparison and temporal questions, retrieval delivers the
evidence and the 8B generator fails to synthesise across it. If that is the
mechanism, a larger generator on IDENTICAL contexts should close the gap; if
the 27B fails the same way, the bottleneck is elsewhere (prompting, metric,
task) and the claim must be softened.

Design: paired at the query level against the committed Stage-3 8B rows.
  sample   comparison and temporal queries only, the strata where the
           bottleneck lives (inference is already at 95% and can't move)
  arms     closed-book and plain-RAG, both at qwen3.6:27b, with byte-identical
           prompts to Stage 3 (same PROMPT_RAG/PROMPT_CLOSED, same BM25@600
           k=10 contexts, temperature 0)
  metric   the same normalised containment, scored by the same code
  test     exact McNemar on 27B-vs-8B per (arm, stratum), plus the deltas
           that matter: does RAG-over-closed-book grow with scale?

Checkpointed per (arm, qi); resumable. The 27B does not fully fit this GPU
(prior calibration: partial CPU offload), so calls are slow: the run is
sized accordingly and the watcher owns the wait.

    python ops/stage6_27b.py --n-comparison 120 --n-temporal 60
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bm25 import BM25                                  # noqa: E402
from stage3_pilot import (PROMPT_CLOSED, PROMPT_RAG,   # noqa: E402
                          load_queries_with_answers)
import urllib.request                                   # noqa: E402

GEN = "qwen3.6:27b"
K = 10
OUT = ROOT / "reports" / "stage6_27b.jsonl"


def generate(prompt: str) -> tuple[str, dict]:
    body = json.dumps({
        "model": GEN, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 64},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return (d.get("response") or "").strip(), {
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "gen_tokens": d.get("eval_count", 0),
        "done_reason": d.get("done_reason")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-comparison", type=int, default=120)
    ap.add_argument("--n-temporal", type=int, default=60)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks_600.jsonl").open()]
    doc_of = [c["doc_id"] for c in chunks]
    qs_all = load_queries_with_answers(set(doc_of))
    idx = BM25([c["text"] for c in chunks])

    # deterministic stratified sample: evenly spaced within each stratum
    want = {"comparison_query": args.n_comparison, "temporal_query": args.n_temporal}
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qs_all):
        by_type[q["type"]].append(i)
    sample: list[int] = []
    for t, n in want.items():
        ixs = by_type[t]
        step = max(len(ixs) // n, 1)
        sample.extend(ixs[::step][:n])
    sample = sorted(sample)

    done: set[tuple[str, int]] = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                r = json.loads(line)
                if not str(r.get("done_reason", "")).startswith("error"):
                    done.add((r["arm"], r["qi"]))
            except Exception:
                pass

    t0 = time.time()
    with OUT.open("a") as fh:
        for n_i, qi in enumerate(sample, 1):
            q = qs_all[qi]
            for arm in ("a0_27b", "plain_27b"):
                if (arm, qi) in done:
                    continue
                if arm == "a0_27b":
                    prompt = PROMPT_CLOSED.format(q=q["query"])
                else:
                    top = idx.top_k(q["query"], K)
                    ctx = "\n\n".join(chunks[i]["text"] for i in top)
                    prompt = PROMPT_RAG.format(context=ctx, q=q["query"])
                try:
                    ans, meta = generate(prompt)
                except Exception as ex:
                    ans, meta = "", {"wall_s": None, "prompt_tokens": 0, "gen_tokens": 0,
                                     "done_reason": f"error:{type(ex).__name__}"}
                fh.write(json.dumps({"qi": qi, "arm": arm, "type": q["type"],
                                     "gold": q["answer"], "answer": ans, **meta}) + "\n")
                fh.flush()
            if n_i % 5 == 0 or n_i == len(sample):
                print(f"  {n_i}/{len(sample)} queries · {(time.time()-t0)/60:.0f}m", flush=True)
    print("stage6 27B run complete", flush=True)


if __name__ == "__main__":
    main()
