"""Does halving the chunk really cost you the relationships it saves?

The 1024-vs-512 measurement was one chunk, and from it I claimed the missing
relationships were "the severed ones": the pairs whose two entities land on
opposite sides of the new boundary. That was an inference, not a measurement.

This measures it. For each source chunk, extract once at 1024 tokens, then
extract its two 512-token halves, and compare the three outputs:

  entities      found at 1024 only / at 512 only / in both
  relationships same, plus the subset provably SEVERED: present at 1024 with
                one endpoint in each half, so no 512 extraction could see it

Severed relationships are the real cost of splitting. Relationships lost for
any other reason are extractor noise, and separating the two is the whole
point of running this.

    python ops/chunk_split_probe.py --n 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from gate_graph_extraction import chunks_from_corpus  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "chunk_split_probe.json"
GEN = "qwen3:8b"

SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["PERSON", "ORGANIZATION", "PRODUCT",
                                                "LOCATION", "EVENT", "OTHER"]},
            "description": {"type": "string"}},
            "required": ["name", "type", "description"]}},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "target": {"type": "string"},
            "strength": {"type": "integer"}},
            "required": ["source", "target", "strength"]}},
    },
    "required": ["entities", "relationships"],
}

PROMPT = """Extract a knowledge graph from this news text.
List every named entity, and every relationship between two of them the text states.
Only what the text says. No inference, no outside knowledge.
strength: 1-10, how central the relationship is.

TEXT:
{chunk}
"""


def norm(s: str) -> str:
    return " ".join(s.lower().split())


CAP = 8000


def extract(text: str) -> tuple[dict | None, float, int, str]:
    """Returns (payload_or_None, wall_s, gen_tokens, status).

    A cap is necessary. The first pilot ran past a 600 s client timeout with
    no bound. But hitting the cap truncates the JSON mid-object, and that is a
    real failure mode of local graph extraction rather than a bug to hide:
    payload size varies enormously across chunks, and a dense one blows any
    fixed budget. So truncation is recorded and counted, never retried into
    silence. Ollama reports done_reason="length", which is a cleaner signal
    than waiting for the JSON parse to fail.
    """
    body = json.dumps({
        "model": GEN, "prompt": PROMPT.format(chunk=text), "stream": False,
        "think": False, "format": SCHEMA,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": CAP},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    wall, tok = time.time() - t0, d.get("eval_count", 0)
    if d.get("done_reason") == "length":
        return None, wall, tok, "truncated"
    try:
        return json.loads(d["response"]), wall, tok, "ok"
    except json.JSONDecodeError:
        return None, wall, tok, "unparseable"


def rel_key(r: dict) -> tuple[str, str]:
    """Undirected: the extractor's source/target order is not reliable."""
    a, b = norm(r["source"]), norm(r["target"])
    return (a, b) if a <= b else (b, a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    chunks = chunks_from_corpus(0)
    step = max(len(chunks) // args.n, 1)
    sample = [chunks[i * step] for i in range(args.n)]

    rows = []
    for i, c in enumerate(sample, 1):
        text = c["text"]
        half = len(text) // 2
        a_txt, b_txt = text[:half], text[half:]

        whole, w_s, w_tok, w_st = extract(text)
        pa, a_s, a_tok, a_st = extract(a_txt)
        pb, b_s, b_tok, b_st = extract(b_txt)

        if whole is None or pa is None or pb is None:
            rows.append({
                "chunk_id": c["chunk_id"], "status": {"whole": w_st, "a": a_st, "b": b_st},
                "wall_s": {"whole": round(w_s, 1), "split": round(a_s + b_s, 1)},
                "gen_tokens": {"whole": w_tok, "split": a_tok + b_tok},
                "comparable": False,
            })
            print(f"  {i:2}/{args.n}  SKIPPED  whole:{w_st} a:{a_st} b:{b_st}"
                  f"  ({w_tok}/{a_tok}/{b_tok} tok)", flush=True)
            continue

        e_whole = {norm(e["name"]) for e in whole["entities"]}
        e_split = {norm(e["name"]) for e in pa["entities"]} | \
                  {norm(e["name"]) for e in pb["entities"]}
        r_whole = {rel_key(r) for r in whole["relationships"]}
        r_split = {rel_key(r) for r in pa["relationships"]} | \
                  {rel_key(r) for r in pb["relationships"]}

        # An entity is "in half A" if its name appears in A's text.
        in_a = {n for n in e_whole if n in norm(a_txt)}
        in_b = {n for n in e_whole if n in norm(b_txt)}
        # Provably severed: present at 1024, endpoints on opposite sides only.
        severed = {k for k in (r_whole - r_split)
                   if (k[0] in in_a and k[1] in in_b and k[1] not in in_a and k[0] not in in_b)
                   or (k[0] in in_b and k[1] in in_a and k[1] not in in_b and k[0] not in in_a)}

        row = {
            "chunk_id": c["chunk_id"],
            "wall_s": {"whole": round(w_s, 1), "split": round(a_s + b_s, 1)},
            "gen_tokens": {"whole": w_tok, "split": a_tok + b_tok},
            "entities": {"whole": len(e_whole), "split": len(e_split),
                         "only_whole": len(e_whole - e_split),
                         "only_split": len(e_split - e_whole)},
            "relationships": {"whole": len(r_whole), "split": len(r_split),
                              "lost_by_splitting": len(r_whole - r_split),
                              "gained_by_splitting": len(r_split - r_whole),
                              "provably_severed": len(severed)},
        }
        row["comparable"] = True
        row["status"] = {"whole": w_st, "a": a_st, "b": b_st}
        rows.append(row)
        r = row["relationships"]
        print(f"  {i:2}/{args.n}  1024:{w_s:6.1f}s/{w_tok:5}tok  512x2:{a_s+b_s:6.1f}s/{a_tok+b_tok:5}tok"
              f"   rels {r['whole']:3}->{r['split']:3}"
              f"  lost {r['lost_by_splitting']:3} (severed {r['provably_severed']:2})"
              f"  gained {r['gained_by_splitting']:3}", flush=True)

    good = [r for r in rows if r.get("comparable")]

    def agg(path):
        vals = []
        for r in good:
            cur = r
            for k in path:
                cur = cur[k]
            vals.append(cur)
        return {"mean": round(statistics.fmean(vals), 1), "min": min(vals), "max": max(vals),
                "total": sum(vals)}

    if not good:
        OUT.write_text(json.dumps({"n": args.n, "comparable": 0, "rows": rows}, indent=1))
        print("\nNo comparable pairs: every sample truncated. "
              "That is the finding: extraction payload exceeds any fixed cap on this corpus.")
        return

    summary = {
        "n": args.n, "comparable": len(good),
        "truncated_or_unparseable": len(rows) - len(good),
        "cap_tokens": CAP, "generator": GEN,
        "schema": "entity descriptions, relationships without descriptions",
        "wall_s_whole": agg(["wall_s", "whole"]),
        "wall_s_split": agg(["wall_s", "split"]),
        "gen_tokens_whole": agg(["gen_tokens", "whole"]),
        "gen_tokens_split": agg(["gen_tokens", "split"]),
        "entities_whole": agg(["entities", "whole"]),
        "entities_split": agg(["entities", "split"]),
        "entities_only_split": agg(["entities", "only_split"]),
        "rels_whole": agg(["relationships", "whole"]),
        "rels_split": agg(["relationships", "split"]),
        "rels_lost": agg(["relationships", "lost_by_splitting"]),
        "rels_severed": agg(["relationships", "provably_severed"]),
        "rels_gained": agg(["relationships", "gained_by_splitting"]),
        "rows": rows,
    }
    tw, ts = summary["wall_s_whole"]["total"], summary["wall_s_split"]["total"]
    summary["speedup_from_splitting"] = round(tw / ts, 2) if ts else None
    OUT.write_text(json.dumps(summary, indent=1))

    print(f"\n  n = {args.n} chunk pairs")
    print(f"  wall   1024 {tw:.0f}s total   vs 512x2 {ts:.0f}s   ratio {summary['speedup_from_splitting']}x")
    print(f"  tokens 1024 {summary['gen_tokens_whole']['total']}  vs 512x2 {summary['gen_tokens_split']['total']}")
    print(f"  rels   1024 {summary['rels_whole']['total']}  vs 512x2 {summary['rels_split']['total']}")
    print(f"    lost by splitting  {summary['rels_lost']['total']}"
          f"  of which provably severed {summary['rels_severed']['total']}")
    print(f"    gained by splitting {summary['rels_gained']['total']}")
    print(f"  entities 1024 {summary['entities_whole']['total']} vs 512x2 "
          f"{summary['entities_split']['total']}  (+{summary['entities_only_split']['total']} only found split)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
