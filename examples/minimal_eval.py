"""The whole study's method, distilled to one script you point at your data.

Runs the paired evaluation the companion articles argue every RAG team
should own: a closed-book arm (catches answerability/contamination), a
BM25-RAG arm, containment scoring with abstention as a first-class number,
the calibration cross (does the model refuse where retrieval failed, or
where it succeeded?), the constant-answer baseline, and — because it moved
one model forty points — the escape-hatch ablation behind a single flag.

Data format (two jsonl files, the study's own schema):

    corpus.jsonl   {"article_id": str, "context": str}
    queries.jsonl  {"qi": int, "question": str, "gold_article": str,
                    "golds": [str, ...]}

Needs an ollama server on :11434 with your model pulled. Examples:

    python examples/minimal_eval.py --corpus examples/sample_data/corpus.jsonl \
        --queries examples/sample_data/queries.jsonl --model qwen3:8b

    # the ablation: same run, escape-hatch sentence deleted from the prompt
    python examples/minimal_eval.py ... --no-escape

    # Thai corpus? segmented BM25 is one flag (also: ja, zh)
    python examples/minimal_eval.py ... --tokenizer th

Results land in results.json next to a per-row log; rerunning resumes.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ops"))
from stage9_lang import LangBM25, contained, norm, wilson, mcnemar_exact  # noqa: E402

PROMPT_RAG = ("Answer the question using ONLY the context below. Be direct and "
              "brief: give the answer itself, no preamble.{escape_rag}\n\n"
              "CONTEXT:\n{context}\n\nQUESTION: {q}\n\nANSWER:")
PROMPT_CLOSED = ("Answer the question directly and briefly: the answer itself, "
                 "no preamble.{escape_closed}\n\nQUESTION: {q}\n\nANSWER:")
ESCAPE_RAG = (" If the context does not contain the answer, reply exactly: "
              "{phrase}")
ESCAPE_CLOSED = " If you do not know, reply exactly: {phrase}"


def generate(model: str, prompt: str, num_predict: int) -> tuple[str, str]:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192,
                    "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return (d.get("response") or "").strip(), str(d.get("done_reason"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--model", required=True, help="ollama model name")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tokenizer", default="regex", choices=["regex", "th", "ja", "zh"],
                    help="BM25 segmentation; th/ja/zh need pythainlp/janome/jieba")
    ap.add_argument("--escape", default="insufficient information",
                    help="the escape-hatch phrase")
    ap.add_argument("--no-escape", action="store_true",
                    help="the ablation: delete the escape-hatch sentence")
    ap.add_argument("--num-predict", type=int, default=128)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    corpus = [json.loads(l) for l in open(args.corpus)]
    queries = [json.loads(l) for l in open(args.queries)]
    if args.max_queries:
        queries = queries[:args.max_queries]
    lang = "en" if args.tokenizer == "regex" else args.tokenizer
    bm = LangBM25(lang, [c["context"] for c in corpus])
    aid_of = [c["article_id"] for c in corpus]

    esc_rag = "" if args.no_escape else ESCAPE_RAG.format(phrase=args.escape)
    esc_closed = "" if args.no_escape else ESCAPE_CLOSED.format(phrase=args.escape)

    rows_path = Path(args.out).with_suffix(".rows.jsonl")
    done = set()
    if rows_path.exists():
        for line in rows_path.open():
            try:
                r = json.loads(line)
                done.add((r["arm"], r["qi"]))
            except Exception:
                pass

    t0 = time.time()
    with rows_path.open("a") as fh:
        for n_i, q in enumerate(queries, 1):
            top = bm.top_k(q["question"], args.k)
            hit = q.get("gold_article") in {aid_of[i] for i in top}
            for arm in ("closed", "rag"):
                if (arm, q["qi"]) in done:
                    continue
                if arm == "closed":
                    prompt = PROMPT_CLOSED.format(escape_closed=esc_closed,
                                                  q=q["question"])
                else:
                    ctx = "\n\n".join(corpus[i]["context"] for i in top)
                    prompt = PROMPT_RAG.format(escape_rag=esc_rag, context=ctx,
                                               q=q["question"])
                ans, reason = generate(args.model, prompt, args.num_predict)
                fh.write(json.dumps({"qi": q["qi"], "arm": arm, "answer": ans,
                                     "golds": q["golds"], "gold_retrieved": hit,
                                     "done_reason": reason},
                                    ensure_ascii=False) + "\n")
                fh.flush()
            if n_i % 10 == 0 or n_i == len(queries):
                print(f"  {n_i}/{len(queries)} · {(time.time()-t0)/60:.0f}m",
                      flush=True)

    # ---- analysis
    rows: dict[tuple[str, int], dict] = {}
    for line in rows_path.open():
        r = json.loads(line)
        rows[(r["arm"], r["qi"])] = r
    qis = sorted({qi for (_a, qi) in rows})
    esc_norm = norm(args.escape)

    def abstained(ans: str) -> bool:
        return bool(esc_norm) and esc_norm in norm(ans)

    out: dict = {"model": args.model, "n": len(qis), "k": args.k,
                 "escape_hatch": None if args.no_escape else args.escape,
                 "arms": {}}
    for arm in ("closed", "rag"):
        rs = [rows[(arm, qi)] for qi in qis if (arm, qi) in rows]
        ok = sum(contained(r["golds"], r["answer"]) for r in rs)
        out["arms"][arm] = {
            "n": len(rs),
            "containment_pct": round(100 * ok / len(rs), 2) if rs else None,
            "wilson95": wilson(ok, len(rs)),
            "abstained_pct": round(100 * sum(abstained(r["answer"]) for r in rs)
                                   / len(rs), 1) if rs else None}

    both = [qi for qi in qis if ("closed", qi) in rows and ("rag", qi) in rows]
    b = c = 0
    for qi in both:
        rr = contained(rows[("rag", qi)]["golds"], rows[("rag", qi)]["answer"])
        cc = contained(rows[("closed", qi)]["golds"], rows[("closed", qi)]["answer"])
        b += rr and not cc
        c += cc and not rr
    out["rag_over_closed"] = {
        "n": len(both), "rag_only": b, "closed_only": c,
        "delta_pp": round(100 * (b - c) / len(both), 2) if both else None,
        "mcnemar_p": mcnemar_exact(b, c)}

    hit_rows = [rows[("rag", qi)] for qi in qis if ("rag", qi) in rows]
    found = [r for r in hit_rows if r["gold_retrieved"]]
    missed = [r for r in hit_rows if not r["gold_retrieved"]]
    out["calibration_cross"] = {
        "abstain_when_gold_retrieved": {
            "n": len(found),
            "pct": round(100 * sum(abstained(r["answer"]) for r in found)
                         / len(found), 1) if found else None},
        "abstain_when_gold_missed": {
            "n": len(missed),
            "pct": round(100 * sum(abstained(r["answer"]) for r in missed)
                         / len(missed), 1) if missed else None}}

    # the coin: does a constant answer beat your model anywhere?
    top_gold = Counter(g for q in queries for g in q["golds"][:1]).most_common(1)
    if top_gold:
        const = top_gold[0][0]
        out["constant_answer_baseline"] = {
            "answer": const,
            "containment_pct": round(100 * sum(
                contained(q["golds"], const) for q in queries) / len(queries), 2)}

    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
