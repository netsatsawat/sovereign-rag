"""E0 and E2/E3 — the cost model, from anecdote to bounded result.

E0 is a precondition, not a formality. Every cost number in the PRD assumes
`think:false` actually suppressed reasoning and that no call silently hit the
token cap. If either is false, the whole cost model is measuring the wrong
thing. It asserts done_reason=="stop" and logs eval_count on every call.

E2/E3 is the real experiment. For n articles, extract once at the 1200-token
chunking and once at the 600-token chunking of the SAME text, then:

  cost      paired log-ratio of wall clock and of emitted tokens, with a
            bootstrap CI, so the exponent k is bounded rather than asserted
            from one chunk (which is how I got the sign wrong the first time)
  edges     every relationship present at 1200 and absent at 600 classified as
            BOUNDARY-SEVERED (its two entities fall in different 600-chunks,
            so no 600 extraction could have seen the pair) or
            WITHIN-CHUNK-UNEMITTED (both entities sat inside one 600-chunk and
            the model simply did not state the relation)

That second split is the point. Extraction is deterministic here — same input
gives byte-identical output, verified — so an unemitted-but-visible relation
is not noise, it is the model behaving differently with less context. Severed
edges are recoverable with overlap; unemitted ones are not, and no amount of
overlap fixes them. Nobody appears to have separated the two.

    python ops/e0_e2e3.py --n 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "stage0.json"
E23 = ROOT / "reports" / "e2e3.json"
GEN = "qwen3:8b"
CAP = 8000

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
    return " ".join((s or "").lower().split())


def rel_key(r: dict) -> tuple[str, str]:
    a, b = norm(r["source"]), norm(r["target"])
    return (a, b) if a <= b else (b, a)


def call(text: str) -> dict:
    body = json.dumps({
        "model": GEN, "prompt": PROMPT.format(chunk=text), "stream": False,
        "think": False, "format": SCHEMA,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": CAP},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    out = {"wall_s": time.time() - t0, "gen_tokens": d.get("eval_count", 0),
           "prompt_tokens": d.get("prompt_eval_count", 0),
           "done_reason": d.get("done_reason"),
           "thinking_len": len(d.get("thinking") or "")}
    try:
        out["payload"] = json.loads(d["response"]) if out["done_reason"] == "stop" else None
    except json.JSONDecodeError:
        out["payload"] = None
    return out


def load(budget: int) -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with (ROOT / "data" / f"chunks_{budget}.jsonl").open() as fh:
        for line in fh:
            c = json.loads(line)
            by_doc[c["doc_id"]].append(c)
    for v in by_doc.values():
        v.sort(key=lambda c: c["chunk_index"])
    return by_doc


def boot_ci(vals: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(statistics.fmean(s))
    means.sort()
    return round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    c600, c1200 = load(600), load(1200)
    # Stratify by capitalised-token ratio: entity density drives extraction cost,
    # so a sample skewed to sparse articles would understate it.
    docs = sorted(set(c600) & set(c1200))

    def cap_ratio(d: str) -> float:
        t = " ".join(c["text"] for c in c1200[d]).split()
        return sum(1 for w in t if w[:1].isupper()) / max(len(t), 1)

    ranked = sorted(docs, key=cap_ratio)
    step = max(len(ranked) // args.n, 1)
    sample = [ranked[i * step] for i in range(args.n) if i * step < len(ranked)]

    e0 = {"calls": 0, "non_stop": 0, "thinking_seen": 0}
    rows = []
    t_start = time.time()

    for i, doc in enumerate(sample, 1):
        big, small = c1200[doc], c600[doc]

        r_big, tok_big, wall_big = set(), 0, 0.0
        ent_big = set()
        bad = False
        for c in big:
            o = call(c["text"])
            e0["calls"] += 1
            if o["done_reason"] != "stop":
                e0["non_stop"] += 1
                bad = True
            if o["thinking_len"]:
                e0["thinking_seen"] += 1
            tok_big += o["gen_tokens"]; wall_big += o["wall_s"]
            if o["payload"]:
                r_big |= {rel_key(r) for r in o["payload"]["relationships"]}
                ent_big |= {norm(e["name"]) for e in o["payload"]["entities"]}

        r_small, tok_small, wall_small = set(), 0, 0.0
        ent_small = set()
        # which 600-chunk each entity name appears in, for the severance test
        ent_home: dict[str, set[int]] = defaultdict(set)
        for j, c in enumerate(small):
            o = call(c["text"])
            e0["calls"] += 1
            if o["done_reason"] != "stop":
                e0["non_stop"] += 1
                bad = True
            if o["thinking_len"]:
                e0["thinking_seen"] += 1
            tok_small += o["gen_tokens"]; wall_small += o["wall_s"]
            if o["payload"]:
                r_small |= {rel_key(r) for r in o["payload"]["relationships"]}
                ent_small |= {norm(e["name"]) for e in o["payload"]["entities"]}
            body = norm(c["text"])
            for e in ent_big:
                if e and e in body:
                    ent_home[e].add(j)

        lost = r_big - r_small
        severed = {k for k in lost
                   if ent_home.get(k[0]) and ent_home.get(k[1])
                   and not (ent_home[k[0]] & ent_home[k[1]])}
        unemitted = {k for k in lost
                     if ent_home.get(k[0]) and ent_home.get(k[1])
                     and (ent_home[k[0]] & ent_home[k[1]])}

        row = {
            "doc_id": doc, "comparable": not bad,
            "chunks": {"at_1200": len(big), "at_600": len(small)},
            "wall_s": {"at_1200": round(wall_big, 1), "at_600": round(wall_small, 1)},
            "gen_tokens": {"at_1200": tok_big, "at_600": tok_small},
            "entities": {"at_1200": len(ent_big), "at_600": len(ent_small)},
            "relationships": {
                "at_1200": len(r_big), "at_600": len(r_small),
                "lost_at_600": len(lost),
                "boundary_severed": len(severed),
                "within_chunk_unemitted": len(unemitted),
                "unclassified": len(lost) - len(severed) - len(unemitted),
                "gained_at_600": len(r_small - r_big),
            },
        }
        rows.append(row)
        rr = row["relationships"]
        print(f"  {i:2}/{len(sample)} {'ok ' if not bad else 'BAD'} "
              f"1200:{len(big)}ch {wall_big:6.1f}s/{tok_big:5}tok  "
              f"600:{len(small)}ch {wall_small:6.1f}s/{tok_small:5}tok  "
              f"rels {rr['at_1200']:3}->{rr['at_600']:3} "
              f"lost {rr['lost_at_600']:3} (sev {rr['boundary_severed']:2} / "
              f"unemit {rr['within_chunk_unemitted']:2}) "
              f"[{(time.time()-t_start)/60:.0f}m]", flush=True)

    good = [r for r in rows if r["comparable"]]
    if not good:
        print("no comparable articles"); return

    lw = [math.log(r["wall_s"]["at_600"] / r["wall_s"]["at_1200"]) for r in good
          if r["wall_s"]["at_1200"] > 0]
    lt = [math.log(r["gen_tokens"]["at_600"] / r["gen_tokens"]["at_1200"]) for r in good
          if r["gen_tokens"]["at_1200"] > 0]
    tot = lambda k, s: sum(r[k][s] for r in good)
    rl = lambda k: sum(r["relationships"][k] for r in good)

    res = {
        "n_requested": args.n, "n_comparable": len(good),
        "e0_preconditions": {**e0,
                             "all_stop": e0["non_stop"] == 0,
                             "no_thinking": e0["thinking_seen"] == 0},
        "wall_ratio_600_over_1200": {
            "geometric_mean": round(math.exp(statistics.fmean(lw)), 3),
            "ci95": [round(math.exp(x), 3) for x in boot_ci(lw)]},
        "token_ratio_600_over_1200": {
            "geometric_mean": round(math.exp(statistics.fmean(lt)), 3),
            "ci95": [round(math.exp(x), 3) for x in boot_ci(lt)]},
        "totals": {
            "wall_s_1200": round(tot("wall_s", "at_1200"), 1),
            "wall_s_600": round(tot("wall_s", "at_600"), 1),
            "gen_tokens_1200": tot("gen_tokens", "at_1200"),
            "gen_tokens_600": tot("gen_tokens", "at_600"),
            "entities_1200": tot("entities", "at_1200"),
            "entities_600": tot("entities", "at_600"),
            "rels_1200": rl("at_1200"), "rels_600": rl("at_600"),
            "rels_lost_at_600": rl("lost_at_600"),
            "boundary_severed": rl("boundary_severed"),
            "within_chunk_unemitted": rl("within_chunk_unemitted"),
            "unclassified": rl("unclassified"),
            "rels_gained_at_600": rl("gained_at_600"),
        },
        "rows": rows,
    }
    t = res["totals"]
    if t["rels_lost_at_600"]:
        res["severed_share_of_loss"] = round(t["boundary_severed"] / t["rels_lost_at_600"], 3)
        res["unemitted_share_of_loss"] = round(t["within_chunk_unemitted"] / t["rels_lost_at_600"], 3)

    E23.write_text(json.dumps(res, indent=1))
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d["e0_e2e3"] = {k: v for k, v in res.items() if k != "rows"}
    OUT.write_text(json.dumps(d, indent=1))

    print(f"\n  E0: {e0['calls']} calls, non-stop {e0['non_stop']}, thinking seen {e0['thinking_seen']}"
          f"  -> {'PASS' if e0['non_stop']==0 and e0['thinking_seen']==0 else 'FAIL'}")
    print(f"  wall  600/1200 ratio {res['wall_ratio_600_over_1200']['geometric_mean']} "
          f"CI {res['wall_ratio_600_over_1200']['ci95']}")
    print(f"  token 600/1200 ratio {res['token_ratio_600_over_1200']['geometric_mean']} "
          f"CI {res['token_ratio_600_over_1200']['ci95']}")
    print(f"  rels  1200 {t['rels_1200']} -> 600 {t['rels_600']}  "
          f"(lost {t['rels_lost_at_600']}, gained {t['rels_gained_at_600']})")
    print(f"    boundary-severed      {t['boundary_severed']}"
          f"  ({100*res.get('severed_share_of_loss',0):.0f}% of loss)")
    print(f"    within-chunk unemitted {t['within_chunk_unemitted']}"
          f"  ({100*res.get('unemitted_share_of_loss',0):.0f}% of loss)")
    print(f"    unclassified           {t['unclassified']}")
    print(f"\nwrote {E23}")


if __name__ == "__main__":
    main()
