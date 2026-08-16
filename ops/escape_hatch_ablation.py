"""Stage 10: the escape-hatch ablation.

Stage 8's headline mechanism: Glimmer delivers its English RAG loss through
the prescribed escape phrase, refusing 56% of evidence-complete queries. The
causal question the drafts name as the designed next arm: is the hatch
*causing* the loss, or merely labelling failures that would happen anyway?

Design: the same 240-query stage-8 sample, same BM25@600 k=10 contexts, same
generator configuration, with the escape-hatch sentence deleted from the
prompt. One arm, Glimmer RAG only (the mechanism under test lives there).

Readout, paired per query against the committed plain_glimmer rows:
  recovered   hatch-arm abstentions that become CORRECT answers -> the
              hatch was suppressing knowledge (causal).
  exposed     hatch-arm abstentions that become WRONG answers -> the hatch
              was honest labelling of failure (not causal).
Plus the paired containment delta and what replaced each abstention.

    python ops/escape_hatch_ablation.py             # run (checkpointed)
    python ops/escape_hatch_ablation.py --analyze   # -> reports/stage10_hatch.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops"))
from bm25 import BM25                                        # noqa: E402
from stage3_pilot import PROMPT_RAG, load_queries_with_answers  # noqa: E402
from stage8_glimmer import (ARMSETS, STAGE6_ROWS, contained,    # noqa: E402
                            abstained, generate, latest_rows,
                            mcnemar_exact, sample_indices, wilson)
import stage8_glimmer                                        # noqa: E402
import time                                                  # noqa: E402

ARM = "plain_glimmer_nohatch"
OUT = ROOT / "reports" / "stage10_hatch.jsonl"
SUMMARY = ROOT / "reports" / "stage10_hatch.json"

HATCH_SENTENCE = (" If the context does not contain the answer,\n"
                  "reply exactly: insufficient information.")
if HATCH_SENTENCE not in PROMPT_RAG:
    raise SystemExit("stage3 PROMPT_RAG no longer contains the expected "
                     "escape-hatch sentence; ablation would be vacuous")
PROMPT_RAG_NOHATCH = PROMPT_RAG.replace(HATCH_SENTENCE, "")


def run() -> None:
    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks_600.jsonl").open()]
    qs_all = load_queries_with_answers({c["doc_id"] for c in chunks})
    idx = BM25([c["text"] for c in chunks])
    sample = sample_indices(qs_all)
    s6 = {json.loads(l)["qi"] for l in STAGE6_ROWS.open()}
    mine = {qi for qi in sample
            if qs_all[qi]["type"] in ("comparison_query", "temporal_query")}
    if mine != s6:
        raise SystemExit(f"sample drifted from stage6: {len(mine ^ s6)} qi")
    print(f"hatch ablation: {len(sample)} queries, arm {ARM}", flush=True)

    done = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                r = json.loads(line)
                if not str(r.get("done_reason", "")).startswith("error"):
                    done.add(r["qi"])
            except Exception:
                pass

    options = ARMSETS["primary"][1]
    t0 = time.time()
    with OUT.open("a") as fh:
        for n_i, qi in enumerate(sample, 1):
            if qi in done:
                continue
            q = qs_all[qi]
            top = idx.top_k(q["query"], stage8_glimmer.K)
            ctx = "\n\n".join(chunks[i]["text"] for i in top)
            prompt = PROMPT_RAG_NOHATCH.format(context=ctx, q=q["query"])
            try:
                ans, meta = generate(prompt, options)
            except Exception as ex:
                ans, meta = "", {"wall_s": None, "prompt_tokens": 0,
                                 "gen_tokens": 0,
                                 "done_reason": f"error:{type(ex).__name__}",
                                 "thinking_len": 0}
            fh.write(json.dumps({"qi": qi, "arm": ARM, "type": q["type"],
                                 "gold": q["answer"], "answer": ans, **meta}) + "\n")
            fh.flush()
            if n_i % 10 == 0 or n_i == len(sample):
                print(f"  {n_i}/{len(sample)} · {(time.time()-t0)/60:.0f}m",
                      flush=True)
    print("stage10 hatch ablation complete", flush=True)


def analyze() -> None:
    nh = latest_rows(OUT, (ARM,))
    base = latest_rows(stage8_glimmer.OUT, ("plain_glimmer",))
    qis = sorted({qi for (_a, qi) in nh})

    rows = [nh[(ARM, qi)] for qi in qis]
    ok = sum(contained(r["gold"], r["answer"]) for r in rows)
    summary: dict = {"arm": ARM, "n": len(rows),
                     "containment": round(100 * ok / len(rows), 2),
                     "wilson95": wilson(ok, len(rows)),
                     "abstained_pct": round(100 * sum(
                         abstained(r["answer"]) for r in rows) / len(rows), 1)}

    common = [qi for qi in qis if ("plain_glimmer", qi) in base]
    b = c = 0
    fates = {"recovered": 0, "exposed": 0, "still_abstains": 0}
    for qi in common:
        rn, rb = nh[(ARM, qi)], base[("plain_glimmer", qi)]
        cn = contained(rn["gold"], rn["answer"])
        cb = contained(rb["gold"], rb["answer"])
        b += cn and not cb
        c += cb and not cn
        if abstained(rb["answer"]):
            if abstained(rn["answer"]):
                fates["still_abstains"] += 1
            elif cn:
                fates["recovered"] += 1
            else:
                fates["exposed"] += 1
    summary["paired_vs_hatch"] = {
        "n": len(common), "nohatch_only": b, "hatch_only": c,
        "delta_pp": round(100 * (b - c) / len(common), 2),
        "mcnemar_p": mcnemar_exact(b, c)}
    summary["hatch_abstention_fates"] = fates
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze() if args.analyze else run()
