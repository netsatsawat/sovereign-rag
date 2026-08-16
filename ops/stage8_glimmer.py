"""Stage 8: Muse Glimmer 30B, day-one, on the committed harness.

Meta released Muse Glimmer (2026-08-10) as an open agentic model built for
local deployment, quantised to ~4-bit with "minimal to no degradation on
agentic tasks", benchmarked against the same Qwen3.6-27B this study already
ran. That makes it a free experiment: the harness has committed paired rows
for qwen3:8b (all 1,381 queries, Stage 3) and qwen3.6:27b (180
comparison/temporal queries, Stage 6), with byte-identical prompts, contexts
and scoring. One new generator slots in; everything else is already paid for.

The questions this run can answer that a launch post cannot:
  abstention   the 27B refused 84% of strict-grounding queries and lost to
               the 8B. Does an agent-trained model treat the same
               "insufficient information" escape hatch differently?
  RAG delta    does retrieval-over-closed-book grow, shrink or flip on a
               model tuned for tool-and-context workloads?
  always-Yes   the constant-Yes baseline scores 59.8% on comparison. Does
               Glimmer clear the coin?

Design: paired at the query level.
  sample   the exact Stage-6 deterministic sample (120 comparison + 60
           temporal; asserted equal to the committed stage6 qi set) plus an
           evenly-spaced 60-query inference sample, the stratum where the
           8B's real RAG lift lives (57.7 -> 95.2).
  arms     a0_glimmer (closed-book) and plain_glimmer (BM25@600, k=10),
           byte-identical PROMPT_CLOSED/PROMPT_RAG, temperature 0, seed 0,
           max_tokens 1024 with system message "Reasoning strength: low"
           (the two documented deviations from the 8B/27B protocol
           at num_predict 64, think off), forced by the model having no
           non-thinking mode; see the comment above ARMSETS. The rec
           armset swaps in Meta's recommended sampling.
  metric   the same normalised containment, plus abstention as a
           first-class number.

Engine note, because it is itself a finding: the official Ollama MLX build
(30b-mlx, 21 GB) loads on a 24 GB Mac only after raising iogpu.wired_limit_mb,
and then generates at ~40 s/token while the OS thrashes in the ~3 GB it has
left: one 64-token call took 52m32s, and its template parser returned an
empty response for all 64 tokens. The GGUF tags in the Ollama library demand a
pre-release runtime. What actually runs on a 24 GB Mac today is Meta's
official muse-glimmer-30B-kquant-17gb.gguf under llama.cpp (day-0 support,
release b10353), which is what this stage tests. Calls go through
llama-server's /v1/chat/completions with the model's own chat template and
--reasoning off --reasoning-format deepseek (the launch line below, pinned
in models.manifest.json); the raw reasoning field is recorded when the
model thinks anyway.

    models/llama-b10353/llama-server -m models/muse-glimmer-30B-kquant-17gb.gguf \
        --port 8095 -c 8192 --jinja --reasoning off --reasoning-format deepseek
    python ops/stage8_glimmer.py                # run (checkpointed, resumable)
    python ops/stage8_glimmer.py --analyze      # score + paired stats -> reports/stage8_glimmer.json
    python ops/stage8_glimmer.py --calibration  # glimmer-day-one.md's calibration cross
                                                #   -> reports/stage8_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bm25 import BM25                                  # noqa: E402
from stage3_pilot import (PROMPT_CLOSED, PROMPT_RAG,   # noqa: E402
                          load_queries_with_answers, norm)

GEN = "muse-glimmer-30B-kquant-17gb.gguf @ llama.cpp b10353"
BASE = "http://localhost:8095"
K = 10
OUT = ROOT / "reports" / "stage8_glimmer.jsonl"
SUMMARY = ROOT / "reports" / "stage8_glimmer.json"
STAGE3_ROWS = ROOT / "reports" / "stage3_pilot.jsonl"
STAGE6_ROWS = ROOT / "reports" / "stage6_27b.jsonl"

WANT = {"comparison_query": 120, "temporal_query": 60, "inference_query": 60}

# Two documented deviations from the 8B/27B protocol, both forced by the
# model having no non-thinking mode (--reasoning off is a no-op for its
# template; probe: all 64 tokens went to reasoning, content empty):
#   system   "Reasoning strength: low", Meta's documented control; without
#            it the reasoning spend doubles (664 vs ~270 gen tokens, probe).
#   budget   max_tokens 1024 vs the others' 64, because reasoning and answer
#            share the budget. Only the answer channel is scored, same as
#            every other arm; the reasoning spend is reported as cost.
# primary pairs with the committed 8B/27B rows (temperature 0); rec is the
# sensitivity arm at Meta's recommended sampling (model card: T=1.0,
# top-p 0.95, top-k 64) so "you ran it off-spec" has an answer with numbers.
SYSMSG = "Reasoning strength: low"
ARMSETS = {
    "primary": (("a0_glimmer", "plain_glimmer"),
                {"temperature": 0, "seed": 0, "max_tokens": 1024}),
    "rec": (("a0_glimmer_rec", "plain_glimmer_rec"),
            {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed": 0,
             "max_tokens": 1024}),
}


def generate(prompt: str, options: dict) -> tuple[str, dict]:
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSMSG},
                     {"role": "user", "content": prompt}], **options,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = d.get("usage") or {}
    return (msg.get("content") or "").strip(), {
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "gen_tokens": usage.get("completion_tokens", 0),
        "done_reason": ch.get("finish_reason"),
        "thinking_len": len(msg.get("reasoning_content") or "")}


def sample_indices(qs_all: list[dict]) -> list[int]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qs_all):
        by_type[q["type"]].append(i)
    sample: list[int] = []
    for t, n in WANT.items():
        ixs = by_type[t]
        step = max(len(ixs) // n, 1)
        sample.extend(ixs[::step][:n])
    return sorted(sample)


def run(armset: str) -> None:
    arms, options = ARMSETS[armset]
    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks_600.jsonl").open()]
    qs_all = load_queries_with_answers({c["doc_id"] for c in chunks})
    idx = BM25([c["text"] for c in chunks])
    sample = sample_indices(qs_all)

    # The comparison/temporal half of the sample must be the committed
    # Stage-6 sample or the 27B pairing is fiction. Hard-fail, don't assume
    # (raise, not assert: asserts are stripped under python -O).
    s6 = {json.loads(l)["qi"] for l in STAGE6_ROWS.open()}
    mine = {qi for qi in sample
            if qs_all[qi]["type"] in ("comparison_query", "temporal_query")}
    if mine != s6:
        raise SystemExit(
            f"sample drifted from stage6: {len(mine ^ s6)} differing qi")
    print(f"sample: {len(sample)} queries "
          f"({', '.join(f'{t.split(chr(95))[0]} {n}' for t, n in WANT.items())}); "
          f"stage6 pairing verified", flush=True)

    done: set[tuple[str, int]] = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                r = json.loads(line)
                arm_qi = (r["arm"], r["qi"])
            except Exception:
                continue
            if str(r.get("done_reason", "")).startswith("error"):
                continue
            # qi is only an index; the row's recorded gold must still be
            # what the CURRENT data says that qi means, or resume would mix
            # answers to different questions under one qi, e.g. a
            # regenerated multihoprag_queries.parquet / chunks_600.jsonl
            # that preserves the sampled qi set but shifts which question a
            # qi denotes. Verified to pass on all committed rows; same
            # pattern as stage9's question_id guard. (SystemExit rather
            # than assert: not strippable under python -O.)
            cur = (qs_all[r["qi"]]["answer"]
                   if isinstance(r["qi"], int) and 0 <= r["qi"] < len(qs_all)
                   else None)
            if r.get("gold") != cur:
                raise SystemExit(
                    f"{OUT}: row (arm={r['arm']}, qi={r['qi']}) has gold "
                    f"{r.get('gold')!r} but the current queries say {cur!r}: "
                    f"the query/chunk data changed under the checkpoint; "
                    f"resolve before resuming")
            done.add(arm_qi)

    t0 = time.time()
    with OUT.open("a") as fh:
        for n_i, qi in enumerate(sample, 1):
            q = qs_all[qi]
            for arm in arms:
                if (arm, qi) in done:
                    continue
                if arm.startswith("a0"):
                    prompt = PROMPT_CLOSED.format(q=q["query"])
                else:
                    top = idx.top_k(q["query"], K)
                    ctx = "\n\n".join(chunks[i]["text"] for i in top)
                    prompt = PROMPT_RAG.format(context=ctx, q=q["query"])
                try:
                    ans, meta = generate(prompt, options)
                except Exception as ex:
                    ans, meta = "", {
                        "wall_s": None, "prompt_tokens": 0, "gen_tokens": 0,
                        "done_reason": f"error:{type(ex).__name__}", "thinking_len": 0}
                # "query" rides along so row identity can be audited
                # directly; gold text alone is weak identity on the
                # comparison stratum, where half the golds are 'Yes'/'No'.
                # Committed pre-guard rows lack the field; their identity
                # check is the gold-vs-current-queries guard in the resume
                # loop above.
                fh.write(json.dumps({"qi": qi, "arm": arm, "type": q["type"],
                                     "query": q["query"],
                                     "gold": q["answer"], "answer": ans, **meta}) + "\n")
                fh.flush()
            if n_i % 5 == 0 or n_i == len(sample):
                print(f"  {n_i}/{len(sample)} queries · {(time.time()-t0)/60:.0f}m", flush=True)
    print("stage8 glimmer run complete", flush=True)


# ----------------------------------------------------------------- analysis

def contained(gold: str, answer: str) -> bool:
    return bool(norm(gold)) and norm(gold) in norm(answer)


def abstained(answer: str) -> bool:
    return "insufficient information" in norm(answer)


def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    # KNOWN DISPLAY DEFECT: rounding to 4 dp reports 0.0 whenever p < 5e-5,
    # a value an exact binomial test cannot produce. Three committed stage8
    # pairings (plain_vs_8b, a0_vs_27b, rec_plain_vs_8b) carry mcnemar_p =
    # 0.0 this way; quote them as p < 0.0001, never "p = 0.0". Returning
    # the unrounded value is the right fix but rewrites committed summary
    # JSONs on the next --analyze; change only with that decision.
    return min(1.0, 2 * p)   # full precision; display rounding is prose's job


def latest_rows(path: Path, arms: tuple[str, ...]) -> dict[tuple[str, int], dict]:
    """Last non-error row wins per (arm, qi), same rule as every resume."""
    out: dict[tuple[str, int], dict] = {}
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("arm") in arms and not str(r.get("done_reason", "")).startswith("error"):
            out[(r["arm"], r["qi"])] = r
    return out


ALL_ARMS = ("a0_glimmer", "plain_glimmer", "a0_glimmer_rec", "plain_glimmer_rec")


def analyze() -> None:
    g = latest_rows(OUT, ALL_ARMS)
    q8 = latest_rows(STAGE3_ROWS, ("a0", "plain"))
    q27 = latest_rows(STAGE6_ROWS, ("a0_27b", "plain_27b"))

    # Completeness check: the qi universe below is whatever non-error rows
    # exist in the jsonl, so an interrupted run yields a summary computed
    # on the surviving subset, and rows for qi OUTSIDE the current sample
    # (a WANT change, a superseded armset experiment left in the file)
    # would be silently included. Warn loudly; stderr only, so the summary
    # JSON stays byte-identical on complete data.
    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks_600.jsonl").open()]
    expected = set(sample_indices(
        load_queries_with_answers({c["doc_id"] for c in chunks})))
    for arm in ALL_ARMS:
        have = {qi for (a, qi) in g if a == arm}
        if have and have != expected:
            print(f"WARNING: arm {arm} has {len(have & expected)}/"
                  f"{len(expected)} sampled qi and {len(have - expected)} "
                  f"rows outside the current sample: the summary is "
                  f"computed on this mismatched set; treat its numbers as "
                  f"partial", file=sys.stderr)

    qis = sorted({qi for (_a, qi) in g})
    summary: dict = {"model": GEN, "n": len(qis), "arms": {}, "paired": {}}

    present = [a for a in ALL_ARMS if any((a, qi) in g for qi in qis)]
    for arm in present:
        rows = [g[(arm, qi)] for qi in qis if (arm, qi) in g]
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_type[r["type"]].append(r)
        ok = sum(contained(r["gold"], r["answer"]) for r in rows)
        summary["arms"][arm] = {
            "n": len(rows),
            "containment": round(100 * ok / len(rows), 2) if rows else None,
            "wilson95": wilson(ok, len(rows)),
            "abstained_pct": round(100 * sum(abstained(r["answer"]) for r in rows)
                                   / len(rows), 1) if rows else None,
            "truncated_pct": round(100 * sum(r.get("done_reason") == "length"
                                             for r in rows) / len(rows), 1) if rows else None,
            "by_type": {t: {
                "n": len(rs),
                "containment": round(100 * sum(contained(r["gold"], r["answer"])
                                               for r in rs) / len(rs), 2),
                "abstained_pct": round(100 * sum(abstained(r["answer"])
                                                 for r in rs) / len(rs), 1),
            } for t, rs in sorted(by_type.items())},
        }

    def paired(name: str, mine_arm: str, ref: dict, ref_arm: str) -> None:
        common = [qi for qi in qis if (mine_arm, qi) in g and (ref_arm, qi) in ref]
        if not common:
            return
        b = c = 0
        for qi in common:
            # qi is trusted as identity across row files; verify it. Passes
            # on all committed rows; fires only if a regenerated file
            # reassigned qi, where pairing would otherwise silently lie.
            # Gold TEXT is weak identity on the comparison stratum (half
            # the golds are 'Yes'/'No'); rows written after the guard also
            # carry the query text for stronger audits. (raise, not
            # assert: must survive python -O.)
            if g[(mine_arm, qi)]["gold"] != ref[(ref_arm, qi)]["gold"]:
                raise SystemExit(
                    f"{name}: gold mismatch at qi={qi}, the qi pairing "
                    f"across row files is broken; refusing to compare")
            m = contained(g[(mine_arm, qi)]["gold"], g[(mine_arm, qi)]["answer"])
            o = contained(ref[(ref_arm, qi)]["gold"], ref[(ref_arm, qi)]["answer"])
            b += m and not o
            c += o and not m
        summary["paired"][name] = {
            "n": len(common), "glimmer_only": b, "ref_only": c,
            "delta_pp": round(100 * (b - c) / len(common), 2),
            "mcnemar_p": mcnemar_exact(b, c)}

    paired("plain_vs_8b", "plain_glimmer", q8, "plain")
    paired("a0_vs_8b", "a0_glimmer", q8, "a0")
    paired("plain_vs_27b", "plain_glimmer", q27, "plain_27b")
    paired("a0_vs_27b", "a0_glimmer", q27, "a0_27b")
    if "plain_glimmer_rec" in present:
        paired("rec_vs_primary_rag", "plain_glimmer_rec", g, "plain_glimmer")
        paired("rec_plain_vs_8b", "plain_glimmer_rec", q8, "plain")

    # RAG-over-closed-book within glimmer, the delta every arm is judged on
    for label, a0_arm, rag_arm in (("rag_over_closed_book", "a0_glimmer", "plain_glimmer"),
                                   ("rag_over_closed_book_rec", "a0_glimmer_rec",
                                    "plain_glimmer_rec")):
        both = [qi for qi in qis if (a0_arm, qi) in g and (rag_arm, qi) in g]
        if not both:
            continue
        b = c = 0
        for qi in both:
            # Same identity requirement as paired(), within one file: the
            # closed and RAG rows being paired must carry the same gold.
            # Verified to pass on all committed rows. (raise, not assert:
            # must survive python -O.)
            if g[(a0_arm, qi)]["gold"] != g[(rag_arm, qi)]["gold"]:
                raise SystemExit(
                    f"{label}: gold mismatch at qi={qi}, the within-file "
                    f"qi pairing is broken; refusing to compare")
            rag = contained(g[(rag_arm, qi)]["gold"], g[(rag_arm, qi)]["answer"])
            cb = contained(g[(a0_arm, qi)]["gold"], g[(a0_arm, qi)]["answer"])
            b += rag and not cb
            c += cb and not rag
        summary["paired"][label] = {
            "n": len(both), "rag_only": b, "closed_only": c,
            "delta_pp": round(100 * (b - c) / len(both), 2),
            "mcnemar_p": mcnemar_exact(b, c)}

    # the coin: always-Yes on the comparison stratum
    comp = [g[("plain_glimmer", qi)] for qi in qis
            if ("plain_glimmer", qi) in g and g[("plain_glimmer", qi)]["type"] == "comparison_query"]
    if comp:
        summary["always_yes_comparison"] = round(
            100 * sum(contained(r["gold"], "Yes") for r in comp) / len(comp), 2)

    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def calibration() -> None:
    """glimmer-day-one.md's calibration cross as a command, not a prose spec:
    finding 1 (abstention vs evidence-completeness, with the 8B/27B anchors
    and answered-when-complete accuracy) and finding 2's paired-60 8B
    closed-book figure, recomputed from committed artifacts only:
    stage8_glimmer.jsonl, stage1_retrieval.json
    per_query.600_k10.bm25.strict, stage3_pilot.jsonl, stage6_27b.jsonl.
    Purely derived; writes reports/stage8_calibration.json and touches
    nothing --analyze depends on."""
    strict = json.loads((ROOT / "reports" / "stage1_retrieval.json").read_text())[
        "per_query"]["600_k10"]["bm25"]["strict"]
    g = latest_rows(OUT, ALL_ARMS)
    q8 = latest_rows(STAGE3_ROWS, ("a0", "plain"))
    q27 = latest_rows(STAGE6_ROWS, ("a0_27b", "plain_27b"))

    qis = sorted({qi for (a, qi) in g if a == "plain_glimmer"})
    comp = [qi for qi in qis if strict[qi] == 1]   # every gold doc in context
    inc = [qi for qi in qis if strict[qi] == 0]    # evidence incomplete
    ab = {qi: abstained(g[("plain_glimmer", qi)]["answer"]) for qi in qis}
    tp = sum(ab[qi] for qi in inc)   # abstained, evidence truly incomplete
    fp = sum(ab[qi] for qi in comp)  # abstained, evidence complete

    def anchor(ref: dict, arm: str) -> dict:
        rows = [qi for qi in comp if (arm, qi) in ref]
        return {"n": len(rows),
                "abstain_pct": round(100 * sum(abstained(ref[(arm, qi)]["answer"])
                                               for qi in rows) / len(rows), 1)}

    answered = [qi for qi in comp if not ab[qi]]
    okc = sum(contained(g[("plain_glimmer", qi)]["gold"],
                        g[("plain_glimmer", qi)]["answer"]) for qi in answered)
    inf = [qi for qi in qis
           if g[("plain_glimmer", qi)]["type"] == "inference_query"
           and ("a0", qi) in q8]
    ok8 = sum(contained(q8[("a0", qi)]["gold"], q8[("a0", qi)]["answer"])
              for qi in inf)

    out = {
        "arm": "plain_glimmer",
        "evidence_vector": "stage1_retrieval.json per_query.600_k10.bm25.strict",
        "n": len(qis), "evidence_complete": len(comp), "evidence_incomplete": len(inc),
        "abstain_pct_given_complete": round(100 * fp / len(comp), 1),
        "abstain_pct_given_incomplete": round(100 * tp / len(inc), 1),
        "abstention_as_missing_evidence_detector": {
            "precision": round(tp / (tp + fp), 2),
            "recall": round(tp / len(inc), 2),
            "base_rate_incomplete_pct": round(100 * len(inc) / len(qis), 1)},
        "anchors_abstain_pct_on_complete": {
            "qwen8b_plain": anchor(q8, "plain"),
            "qwen27b_plain": anchor(q27, "plain_27b")},
        "answered_when_complete": {
            "contained": okc, "n": len(answered),
            "pct": round(100 * okc / len(answered), 1)},
        "qwen8b_closed_book_inference_same60": {
            "n": len(inf), "containment": round(100 * ok8 / len(inf), 2)},
    }
    (ROOT / "reports" / "stage8_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--armset", choices=sorted(ARMSETS), default="primary")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    elif args.calibration:
        calibration()
    else:
        run(args.armset)
