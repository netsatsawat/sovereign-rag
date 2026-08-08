"""E1b — does the chunker sever the evidence the benchmark grades on?

The naive character scheme severed 6.83% of gold evidence spans and cost
17.18% of queries at least one span, asymmetrically by question type
(inference 24.8%, null 0%). Since the deliverable is a per-question-type
curve, a chunker that differentially destroys inference evidence puts a
question-shaped thumb on the scale before any architecture runs.

Sentence alignment should drive that to near zero. "Near" rather than zero,
for two reasons already known: 6 of 981 gold spans are multi-sentence and a
sentence packer will split them, and pySBD's ~2% boundary error rate
concentrates in the abbreviations news is made of. The honest expectation is
0.5-2%. This measures which.

A span is INTACT if it appears inside a single chunk. Comparison is on
normalised text, because chunk assembly rejoins sentences with a single space
and the source may have had other whitespace.

    python ops/e1b_severance.py
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
QUERIES = ROOT / "data" / "multihoprag_queries.parquet"
OUT = ROOT / "reports" / "stage0.json"
CATEGORIES = ("technology", "business")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", s).strip()


def load_chunks(budget: int) -> dict[str, list[str]]:
    by_doc: dict[str, list[str]] = defaultdict(list)
    with (ROOT / "data" / f"chunks_{budget}.jsonl").open() as fh:
        for line in fh:
            c = json.loads(line)
            by_doc[c["doc_id"]].append(norm(c["text"]))
    return by_doc


def main() -> None:
    rows = pq.read_table(QUERIES).to_pylist()
    results = {}

    for budget in (600, 1200):
        by_doc = load_chunks(budget)
        docs = set(by_doc)

        per_type = defaultdict(lambda: {"spans": 0, "severed": 0,
                                        "queries": 0, "queries_hit": 0})
        total = {"spans": 0, "severed": 0, "queries": 0, "queries_hit": 0,
                 "spans_out_of_subset": 0}

        for r in rows:
            qt = r.get("question_type") or "unknown"
            ev = r.get("evidence_list") or []
            in_subset = [e for e in ev if (e.get("url") in docs)]
            if not in_subset:
                continue                      # query's evidence is outside the subset

            per_type[qt]["queries"] += 1
            total["queries"] += 1
            hit = False
            for e in in_subset:
                fact = norm(e.get("fact") or "")
                if len(fact) < 12:
                    continue
                per_type[qt]["spans"] += 1
                total["spans"] += 1
                if not any(fact in ch for ch in by_doc[e["url"]]):
                    per_type[qt]["severed"] += 1
                    total["severed"] += 1
                    hit = True
            if hit:
                per_type[qt]["queries_hit"] += 1
                total["queries_hit"] += 1

        def pct(a, b):
            return round(100 * a / b, 2) if b else None

        results[f"at_{budget}"] = {
            "spans_checked": total["spans"],
            "spans_severed": total["severed"],
            "span_severance_pct": pct(total["severed"], total["spans"]),
            "queries_checked": total["queries"],
            "queries_losing_a_span": total["queries_hit"],
            "query_impact_pct": pct(total["queries_hit"], total["queries"]),
            "by_question_type": {
                qt: {"spans": v["spans"], "severed": v["severed"],
                     "span_severance_pct": pct(v["severed"], v["spans"]),
                     "queries": v["queries"], "queries_hit": v["queries_hit"],
                     "query_impact_pct": pct(v["queries_hit"], v["queries"])}
                for qt, v in sorted(per_type.items())
            },
        }

        r = results[f"at_{budget}"]
        print(f"budget {budget}: {r['spans_severed']}/{r['spans_checked']} spans severed "
              f"({r['span_severance_pct']}%)   "
              f"{r['queries_losing_a_span']}/{r['queries_checked']} queries affected "
              f"({r['query_impact_pct']}%)")
        for qt, v in r["by_question_type"].items():
            print(f"    {qt:22} {v['severed']:3}/{v['spans']:4} spans "
                  f"({v['span_severance_pct']}%)   {v['query_impact_pct']}% of queries")

    results["baseline_naive_char_scheme"] = {
        "span_severance_pct": 6.83, "query_impact_pct": 17.18,
        "by_type_query_impact": {"inference": 24.8, "comparison": 16.4,
                                 "temporal": 16.6, "null": 0.0},
        "source": "measured by the chunking research on the pre-existing scheme",
    }

    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d["e1b_severance"] = results
    OUT.write_text(json.dumps(d, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
