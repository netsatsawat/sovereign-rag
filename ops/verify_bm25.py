"""Producer for the 'postings-verified against brute force' claim, in-repo.

The verification originally ran in the sovereign-learning-loop project this
implementation was ported from. A claim in this repo needs a verifier in this
repo. CPU-only; safe to run any time.

    python ops/verify_bm25.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bm25 import BM25  # noqa: E402


def main() -> None:
    rng = random.Random(0)
    vocab = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    docs = [" ".join(rng.choice(vocab) for _ in range(rng.randint(5, 40)))
            for _ in range(400)]
    idx = BM25(docs)

    def brute(q, k):
        scored = sorted(((-idx.score(q, i), i) for i in range(idx.n)))
        top = [i for s, i in scored[:k] if s < 0]
        return top

    mism = 0
    for _ in range(300):
        q = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 4)))
        for k in (1, 5, 10, 25):
            if idx.top_k(q, k) != brute(q, k):
                mism += 1
    print(f"1,200 comparisons, mismatches: {mism}")
    assert idx.top_k("zzzz qqqq", 5) == [], "no-match query must return empty"
    print("no-match query returns []: OK")
    assert mism == 0


if __name__ == "__main__":
    main()
