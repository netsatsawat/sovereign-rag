"""Stage 10 — 27B on the entity stratum: recency or scale?

Glimmer's 96.7% closed-book on the 60 inference questions could come from
being newer (more late-2023 news in training) or bigger (better recall of
what both saw). The 27B — same generation as the 8B, three times the size —
splits the difference: closed-book high like Glimmer -> scale; closed-book
near the 8B's 51.7% -> recency/generation.

Arms: a0_27b_inf (closed) and plain_27b_inf (BM25@600 k=10 RAG), the 60
inference qi from the stage-8 sample, stage-3 prompts, qwen3.6:27b via
ollama with think:false and num_predict 64 — the stage-6 configuration.

    python ops/stage10_27b_inf.py             # run (checkpointed)
    python ops/stage10_27b_inf.py --analyze   # -> reports/stage10_27b_inf.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops"))
from bm25 import BM25                                          # noqa: E402
from stage3_pilot import (PROMPT_CLOSED, PROMPT_RAG,           # noqa: E402
                          load_queries_with_answers)
from stage8_glimmer import (contained, abstained, latest_rows,  # noqa: E402
                            mcnemar_exact, sample_indices, wilson)
import stage8_glimmer                                          # noqa: E402

GEN = "qwen3.6:27b"
OUT = ROOT / "reports" / "stage10_27b_inf.jsonl"
SUMMARY = ROOT / "reports" / "stage10_27b_inf.json"


def generate(prompt: str) -> tuple[str, dict]:
    body = json.dumps({
        "model": GEN, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192,
                    "num_predict": 64},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return (d.get("response") or "").strip(), {
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "gen_tokens": d.get("eval_count", 0),
        "done_reason": d.get("done_reason"),
        "thinking_len": 0}


def inference_sample() -> tuple[list[dict], list[int], BM25, list[dict]]:
    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks_600.jsonl").open()]
    qs_all = load_queries_with_answers({c["doc_id"] for c in chunks})
    sample = [qi for qi in sample_indices(qs_all)
              if qs_all[qi]["type"] == "inference_query"]
    return qs_all, sample, BM25([c["text"] for c in chunks]), chunks


def run() -> None:
    qs_all, sample, idx, chunks = inference_sample()
    print(f"27B inference stratum: {len(sample)} queries x 2 arms", flush=True)

    done = set()
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
            for arm in ("a0_27b_inf", "plain_27b_inf"):
                if (arm, qi) in done:
                    continue
                if arm == "a0_27b_inf":
                    prompt = PROMPT_CLOSED.format(q=q["query"])
                else:
                    top = idx.top_k(q["query"], stage8_glimmer.K)
                    ctx = "\n\n".join(chunks[i]["text"] for i in top)
                    prompt = PROMPT_RAG.format(context=ctx, q=q["query"])
                try:
                    ans, meta = generate(prompt)
                except Exception as ex:
                    ans, meta = "", {"wall_s": None, "prompt_tokens": 0,
                                     "gen_tokens": 0,
                                     "done_reason": f"error:{type(ex).__name__}",
                                     "thinking_len": 0}
                fh.write(json.dumps({"qi": qi, "arm": arm, "type": q["type"],
                                     "gold": q["answer"], "answer": ans,
                                     **meta}) + "\n")
                fh.flush()
            if n_i % 5 == 0 or n_i == len(sample):
                print(f"  {n_i}/{len(sample)} · {(time.time()-t0)/60:.0f}m",
                      flush=True)
    print("stage10 27B inference run complete", flush=True)


def analyze() -> None:
    mine = latest_rows(OUT, ("a0_27b_inf", "plain_27b_inf"))
    g = latest_rows(stage8_glimmer.OUT, ("a0_glimmer", "plain_glimmer"))
    q8 = latest_rows(stage8_glimmer.STAGE3_ROWS, ("a0", "plain"))
    qis = sorted({qi for (_a, qi) in mine})

    summary: dict = {"model": GEN, "stratum": "inference", "n": len(qis),
                     "arms": {}, "paired_closed_book": {}}
    for arm in ("a0_27b_inf", "plain_27b_inf"):
        rows = [mine[(arm, qi)] for qi in qis if (arm, qi) in mine]
        ok = sum(contained(r["gold"], r["answer"]) for r in rows)
        summary["arms"][arm] = {
            "n": len(rows),
            "containment": round(100 * ok / len(rows), 2),
            "wilson95": wilson(ok, len(rows)),
            "abstained_pct": round(100 * sum(abstained(r["answer"])
                                             for r in rows) / len(rows), 1)}

    def paired(name: str, ref: dict, ref_arm: str) -> None:
        common = [qi for qi in qis
                  if ("a0_27b_inf", qi) in mine and (ref_arm, qi) in ref]
        b = c = 0
        for qi in common:
            m = contained(mine[("a0_27b_inf", qi)]["gold"],
                          mine[("a0_27b_inf", qi)]["answer"])
            o = contained(ref[(ref_arm, qi)]["gold"],
                          ref[(ref_arm, qi)]["answer"])
            b += m and not o
            c += o and not m
        summary["paired_closed_book"][name] = {
            "n": len(common), "only_27b": b, "only_ref": c,
            "delta_pp": round(100 * (b - c) / len(common), 2) if common else None,
            "mcnemar_p": mcnemar_exact(b, c)}

    paired("vs_glimmer", g, "a0_glimmer")
    paired("vs_8b", q8, "a0")
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze() if args.analyze else run()
