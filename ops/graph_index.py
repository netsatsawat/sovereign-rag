"""The graph index, built to survive being killed.

One extraction call per chunk, appended to a jsonl checkpoint as it lands.
Restart skips everything already done, so a 20-hour index can be stopped and
resumed at will and a crash costs one chunk, not a night.

Truncation (4.8% of calls in E0) is recorded as a counted outcome with the
chunk id, never retried: extraction is deterministic here, so a retry at the
same cap reproduces the same truncation. Dense chunks are a standing failure
mode of local extraction and the honest report includes them.

    python ops/graph_index.py --budget 600
    tail -f reports/graph_600.log
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
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
    wall = round(time.time() - t0, 1)
    row = {"wall_s": wall, "gen_tokens": d.get("eval_count", 0),
           "prompt_tokens": d.get("prompt_eval_count", 0)}
    if d.get("done_reason") == "length":
        row["status"] = "truncated"
        return row
    try:
        p = json.loads(d["response"])
        row["status"] = "ok"
        row["entities"] = p.get("entities", [])
        row["relationships"] = p.get("relationships", [])
    except json.JSONDecodeError:
        row["status"] = "unparseable"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    args = ap.parse_args()

    src = ROOT / "data" / f"chunks_{args.budget}.jsonl"
    out = ROOT / "data" / f"graph_{args.budget}.jsonl"
    chunks = [json.loads(l) for l in src.open()]

    done: set[str] = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["chunk_id"])
            except Exception:
                pass
    todo = [c for c in chunks if f'{c["doc_id"]}#{c["chunk_index"]}' not in done]
    print(f"{len(chunks)} chunks, {len(done)} already indexed, {len(todo)} to go", flush=True)

    t0 = time.time()
    walls: list[float] = []
    with out.open("a") as fh:
        for n, c in enumerate(todo, 1):
            cid = f'{c["doc_id"]}#{c["chunk_index"]}'
            row = call(c["text"])
            row["chunk_id"] = cid
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            walls.append(row["wall_s"])
            if n % 10 == 0 or n == len(todo):
                rate = sum(walls[-50:]) / len(walls[-50:])
                eta_h = rate * (len(todo) - n) / 3600
                print(f"  {n}/{len(todo)}  {row['status']:11} {row['wall_s']:6.1f}s  "
                      f"recent {rate:5.1f}s/chunk  eta {eta_h:5.1f}h  "
                      f"elapsed {(time.time()-t0)/3600:.1f}h", flush=True)
    print("index complete", flush=True)


if __name__ == "__main__":
    main()
