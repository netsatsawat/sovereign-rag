"""Stage 4 — retrieval-feedback re-ranking, and the gate that decides what it
may be called.

The claim under test is the user's original requirement: "the system should
get better every time a similar query is asked." The mechanism is L1
retrieval-feedback: during a learning stream the system observes which
retrieved documents were actually gold, accumulates credit, and re-ranks
future retrievals with that credit. No model is trained; the store is not
written to; the only state is a credit table.

The design has to beat two traps that make fake learning look real:

  the popularity counter   a retriever that IGNORES the query and ranks
                           documents by accumulated credit alone scores
                           Hit@10 ~65% on this corpus (gold documents are
                           concentrated). Any "learning" gain must exceed
                           what this query-blind counter gets for free.
  evaluation leakage       credit comes from knowing gold on STREAM queries
                           only; evaluation queries never contribute credit.
                           The signal is disclosed as an upper bound on what
                           deployed feedback (thumbs) could supply.

Split, per seed: 60% learning stream, 40% evaluation. Evaluation divides into
  N1 "similar"  — shares at least one gold document with some stream query
  N2 "novel"    — zero gold-document overlap with the stream
"Gets better on similar queries" is a claim about N1. "Does not get worse" is
a claim about N2. Both are measured; neither is assumed.

Arms, all evaluated identically at k=10 on strict@10 and hits@10:
  base        BM25 as shipped (Stage 1's winner)
  learned     score' = bm25_norm + LAMBDA * credit_norm  (LAMBDA fixed, declared)
  popularity  credit_norm alone, query-blind  (destructive control #1)
  shuffled    learned, but credit permuted across documents (control #2 —
              if this helps, the "signal" is structural, not learned)

Credit: for each stream query, retrieve top-k with epsilon-greedy exploration
(prob EPS: one uniformly random chunk injected at a uniformly random rank,
propensity logged); every retrieved-and-gold document earns 1/propensity(slot)
credit, aggregated at document level. Five seeds, permuted stream order;
paired per-query deltas with bootstrap CIs pooled across seeds.

    python ops/stage4_learning.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bm25 import BM25                      # noqa: E402
from score_retrieval import load_queries   # noqa: E402

K = 10
EPS = 0.1
LAMBDA = 1.0
SEEDS = 5
STREAM_FRAC = 0.6
OUT = ROOT / "reports" / "stage4_learning.json"


def boot_ci(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(diffs)
    if m == 0:
        return 0.0, [0.0, 0.0], 1.0
    means = sorted(statistics.fmean(diffs[rng.randrange(m)] for _ in range(m))
                   for _ in range(n))
    ge = sum(1 for x in means if x >= 0) / n
    le = sum(1 for x in means if x <= 0) / n
    p = max(min(2 * min(ge, le), 1.0), 1.0 / n)
    return (round(100 * statistics.fmean(diffs), 2),
            [round(100 * means[int(.025 * n)], 2), round(100 * means[int(.975 * n)], 2)],
            round(p, 5))


def main() -> None:
    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks_600.jsonl").open()]
    doc_of = [c["doc_id"] for c in chunks]
    n_chunks = len(chunks)
    qs = load_queries(set(doc_of))
    idx = BM25([c["text"] for c in chunks])

    # base rankings once — they never change across seeds
    base_top50 = [idx.top_k(q["query"], 50) for q in qs]

    def strict_at(top, gold):
        seen, docs = set(), []
        for i in top[:K]:
            d = doc_of[i]
            if d not in seen:
                seen.add(d)
                docs.append(d)
        return gold.issubset(set(docs)), any(d in gold for d in docs)

    per_seed = []
    pooled: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for seed in range(SEEDS):
        rng = random.Random(1000 + seed)
        order = list(range(len(qs)))
        rng.shuffle(order)
        cut = int(STREAM_FRAC * len(qs))
        stream, evaln = order[:cut], order[cut:]

        # ---- learning pass over the stream
        credit: dict[str, float] = defaultdict(float)
        stream_gold_docs: set[str] = set()
        for qi in stream:
            gold = qs[qi]["gold"]
            stream_gold_docs |= gold
            top = list(base_top50[qi][:K])
            # epsilon-greedy exploration with a logged propensity
            explored = None
            if rng.random() < EPS:
                slot = rng.randrange(K)
                cand = rng.randrange(n_chunks)
                explored = (slot, cand)
                top[slot] = cand
            for slot, ci in enumerate(top):
                d = doc_of[ci]
                if d in gold:
                    if explored and slot == explored[0]:
                        prop = EPS * (1 / n_chunks) * (1 / K)
                        prop = max(prop, 1e-6)
                        credit[d] += min(1.0 / prop, 50.0)   # clipped IPW, declared
                    else:
                        credit[d] += 1.0 / (1 - EPS)
        cmax = max(credit.values()) if credit else 1.0

        # ---- shuffled-credit control: same values, permuted documents
        docs_all = sorted({d for d in doc_of})
        perm = docs_all[:]
        rng.shuffle(perm)
        remap = dict(zip(docs_all, perm))
        credit_shuf = defaultdict(float)
        for d, v in credit.items():
            credit_shuf[remap[d]] = v

        def rerank(qi, table):
            base = base_top50[qi]
            n = len(base)
            scored = []
            for r, ci in enumerate(base):
                bm_norm = (n - r) / n
                c_norm = table.get(doc_of[ci], 0.0) / cmax
                scored.append((-(bm_norm + LAMBDA * c_norm), ci))
            scored.sort()
            return [ci for _, ci in scored]

        def pop_rank(qi, table):
            # query-blind: candidates are still the base pool (a fair pool),
            # ranked by credit alone; ties by chunk index
            base = base_top50[qi]
            return sorted(base, key=lambda ci: (-table.get(doc_of[ci], 0.0), ci))

        n1 = [qi for qi in evaln if qs[qi]["gold"] & stream_gold_docs]
        n2 = [qi for qi in evaln if not (qs[qi]["gold"] & stream_gold_docs)]

        row = {"seed": seed, "stream": len(stream), "n1": len(n1), "n2": len(n2),
               "credited_docs": len(credit)}
        for split_name, split in (("n1", n1), ("n2", n2)):
            for arm, fn in (("base", lambda qi: base_top50[qi]),
                            ("learned", lambda qi: rerank(qi, credit)),
                            ("popularity", lambda qi: pop_rank(qi, credit)),
                            ("shuffled", lambda qi: rerank(qi, credit_shuf))):
                s_hits = []
                for qi in split:
                    s, h = strict_at(fn(qi), qs[qi]["gold"])
                    s_hits.append((s, h))
                row[f"{split_name}_{arm}_strict"] = round(
                    100 * statistics.fmean(x[0] for x in s_hits), 2)
                row[f"{split_name}_{arm}_hits"] = round(
                    100 * statistics.fmean(x[1] for x in s_hits), 2)
                pooled[split_name][arm].extend(
                    (qi, seed, x[0]) for qi, x in zip(split, s_hits))
        per_seed.append(row)
        print(f"  seed {seed}: n1={len(n1)} n2={len(n2)} · "
              f"n1 strict base {row['n1_base_strict']} -> learned {row['n1_learned_strict']} "
              f"(pop {row['n1_popularity_strict']}, shuf {row['n1_shuffled_strict']}) · "
              f"n2 base {row['n2_base_strict']} -> learned {row['n2_learned_strict']}",
              flush=True)

    # ---- pooled paired deltas
    deltas = {}
    for split_name in ("n1", "n2"):
        by_key = {}
        for arm in ("base", "learned", "popularity", "shuffled"):
            for qi, seed, s in pooled[split_name][arm]:
                by_key.setdefault((qi, seed), {})[arm] = s
        complete = [v for v in by_key.values() if len(v) == 4]
        for a, b in (("learned", "base"), ("learned", "popularity"),
                     ("learned", "shuffled"), ("popularity", "base")):
            diffs = [float(v[a]) - float(v[b]) for v in complete]
            d, ci, p = boot_ci(diffs)
            deltas[f"{split_name}_{a}_vs_{b}_strict"] = {
                "delta_pts": d, "ci95": ci, "p_boot": p, "pairs": len(diffs)}

    gates = {
        "g1_better_on_similar": deltas["n1_learned_vs_base_strict"]["delta_pts"] > 0
            and deltas["n1_learned_vs_base_strict"]["p_boot"] < 0.05,
        "g2_not_worse_on_novel": deltas["n2_learned_vs_base_strict"]["ci95"][0] > -3.0,
        "g3_beats_popularity_counter": deltas["n1_learned_vs_popularity_strict"]["delta_pts"] > 0
            and deltas["n1_learned_vs_popularity_strict"]["p_boot"] < 0.05,
        "g4_not_structural": not (deltas["n1_learned_vs_shuffled_strict"]["p_boot"] > 0.05),
    }
    gates["verdict"] = ("LEARNING CLAIM PERMITTED" if all(
        v for k2, v in gates.items() if k2 != "verdict")
        else "PRE-REGISTERED NEGATIVE: the learning claim may not be made")

    out = {"k": K, "eps": EPS, "lambda": LAMBDA, "seeds": SEEDS,
           "stream_frac": STREAM_FRAC,
           "ipw": "clipped at 50, declared",
           "per_seed": per_seed, "deltas": deltas, "gates": gates}
    OUT.write_text(json.dumps(out, indent=1))

    print("\ndeltas (strict@10, pooled across seeds):")
    for k2, v in deltas.items():
        print(f"  {k2:38} {v['delta_pts']:+7.2f}  CI {v['ci95']}  p={v['p_boot']}")
    print("\ngates:")
    for k2, v in gates.items():
        print(f"  {k2}: {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
