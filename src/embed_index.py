"""Embed a chunk file once, save the vectors, never embed twice.

E4 threw its embeddings away and had to re-embed for every metric question.
Saving them makes every scoring run instant and makes the scored vectors an
auditable artifact: the manifest records which chunk file (by sha) produced
which matrix.

Landmine, per PRD 4.5.6: never pass `options` to /api/embed — the triggering
call succeeds and every later embed fails with EOF.

    python src/embed_index.py --budget 600
    python src/embed_index.py --budget 600 --all-categories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
EMB = "qwen3-embedding:0.6b"
BATCH = 32


def embed(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": EMB, "input": texts}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    if "embeddings" not in d:
        raise RuntimeError(f"embed failed: {str(d)[:200]}")
    return d["embeddings"]


def embed_all(texts: list[str], label: str) -> np.ndarray:
    out: list[list[float]] = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        out.extend(embed(texts[i:i + BATCH]))
        if i and i % (BATCH * 20) == 0:
            print(f"    {label}: {i}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)
    M = np.asarray(out, dtype=np.float32)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--all-categories", action="store_true")
    args = ap.parse_args()

    suffix = "_all" if args.all_categories else ""
    src = ROOT / "data" / f"chunks_{args.budget}{suffix}.jsonl"
    chunks = [json.loads(l) for l in src.open()]

    t0 = time.time()
    M = embed_all([c["text"] for c in chunks], f"chunks@{args.budget}{suffix}")
    wall = time.time() - t0

    out = ROOT / "data" / f"emb_{args.budget}{suffix}"
    np.save(f"{out}.npy", M)
    meta = {
        "embedder": EMB,
        "source": src.name,
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()[:16],
        "rows": int(M.shape[0]), "dim": int(M.shape[1]),
        "wall_s": round(wall, 1),
        "doc_ids": [c["doc_id"] for c in chunks],
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta))
    print(f"{src.name}: {M.shape[0]} x {M.shape[1]} in {wall:.0f}s -> {out}.npy")


if __name__ == "__main__":
    main()
