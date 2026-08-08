"""Stage 0, gate G3 — the one that decides whether GraphRAG exists here.

Twenty real chunks from the real corpus through a real GraphRAG-style
entity/relation extraction prompt, with structured output enforced by Ollama's
`format` schema. Two thresholds, both from PRD §13:

    20/20 valid payloads, and under 90 s per chunk.

If either fails, the graph arm is not built and the study reports five arms
instead of seven. That is a legitimate outcome and it costs one evening to
discover rather than four nights of indexing to find out.

The per-chunk second is also the publishable number. Microsoft indexed ~1M
tokens in 281 minutes on GPT-4-turbo; nobody reports what the same operation
costs on a laptop with no API key.

    python ops/stage0_g3.py --n 20
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).parent
ROOT = HERE.parent
CORPUS = ROOT / "data" / "multihoprag_corpus.parquet"
OUT = ROOT / "reports" / "stage0.json"
SAMPLES = ROOT / "reports" / "stage0_extractions.json"

OLLAMA = "http://localhost:11434"
GEN = "qwen3:8b"

# PRD §4.1: tech + business subset, 1024-token chunks, 128 overlap.
# ~4 chars per token is the standard rough conversion and is stated rather
# than hidden, because it sets the chunk count and therefore the graph bill.
CHARS_PER_TOK = 4
CHUNK_TOK, OVERLAP_TOK = 1024, 128

SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string",
                             "enum": ["PERSON", "ORGANIZATION", "PRODUCT",
                                      "LOCATION", "EVENT", "OTHER"]},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "description": {"type": "string"},
                    "strength": {"type": "integer"},
                },
                "required": ["source", "target", "description", "strength"],
            },
        },
    },
    "required": ["entities", "relationships"],
}

PROMPT = """You are extracting a knowledge graph from a news article.

Identify every named entity in the text, and every direct relationship
between two of those entities that the text actually states.

Rules:
- Only entities and relationships present in the text. Do not infer, and do
  not use outside knowledge.
- description: one short clause, drawn from the text.
- strength: 1 to 10, how central the relationship is to the text.

TEXT:
{chunk}
"""


def chunks_from_corpus(limit_chars: int) -> list[dict]:
    rows = pq.read_table(CORPUS).to_pylist()
    rows = [r for r in rows if r.get("category") in ("technology", "business")]
    rows.sort(key=lambda r: (r.get("url") or "", r.get("title") or ""))
    size, step = CHUNK_TOK * CHARS_PER_TOK, (CHUNK_TOK - OVERLAP_TOK) * CHARS_PER_TOK
    out = []
    for r in rows:
        body = unicodedata.normalize("NFKC", r.get("body") or "")
        body = re.sub(r"\s+", " ", body).strip()
        for i in range(0, max(len(body) - OVERLAP_TOK * CHARS_PER_TOK, 1), step):
            piece = body[i:i + size]
            if len(piece) < 200:
                continue
            out.append({"chunk_id": f"{r['url']}#{i}", "title": r["title"],
                        "category": r["category"], "text": piece})
    return out


def extract(text: str) -> tuple[dict | None, float, dict]:
    body = json.dumps({
        "model": GEN,
        "prompt": PROMPT.format(chunk=text),
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    wall = time.time() - t0
    meta = {
        "prompt_tokens": resp.get("prompt_eval_count"),
        "gen_tokens": resp.get("eval_count"),
        "prefill_tok_s": round(resp["prompt_eval_count"] /
                               (resp["prompt_eval_duration"] / 1e9), 1)
        if resp.get("prompt_eval_duration") else None,
        "decode_tok_s": round(resp["eval_count"] / (resp["eval_duration"] / 1e9), 2)
        if resp.get("eval_duration") else None,
    }
    try:
        return json.loads(resp["response"]), wall, meta
    except Exception:
        return None, wall, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    all_chunks = chunks_from_corpus(0)
    print(f"tech+business chunks at {CHUNK_TOK}tok/{OVERLAP_TOK}overlap: {len(all_chunks)}")

    # Evenly spaced through the corpus rather than the first n, so the sample
    # is not twenty chunks of one long article.
    step = max(len(all_chunks) // args.n, 1)
    sample = [all_chunks[i * step] for i in range(args.n)]

    rows, kept = [], []
    for i, c in enumerate(sample, 1):
        payload, wall, meta = extract(c["text"])
        ok = bool(payload) and isinstance(payload.get("entities"), list) \
            and isinstance(payload.get("relationships"), list)
        rows.append({
            "chunk_id": c["chunk_id"], "valid": ok, "wall_s": round(wall, 1),
            "entities": len(payload.get("entities", [])) if payload else 0,
            "relationships": len(payload.get("relationships", [])) if payload else 0,
            **meta,
        })
        if payload:
            kept.append({"chunk_id": c["chunk_id"], "text": c["text"][:400],
                         "extraction": payload})
        print(f"  {i:2}/{args.n} {'ok ' if ok else 'BAD'} {wall:6.1f}s  "
              f"{rows[-1]['entities']:3}e {rows[-1]['relationships']:3}r  "
              f"{c['title'][:44]}", flush=True)

    walls = [r["wall_s"] for r in rows]
    valid = sum(r["valid"] for r in rows)
    total_chunks = len(all_chunks)
    mean_s = statistics.fmean(walls)

    result = {
        "generator": GEN,
        "chunking": {"chunk_tokens": CHUNK_TOK, "overlap_tokens": OVERLAP_TOK,
                     "chars_per_token": CHARS_PER_TOK,
                     "tech_business_chunks": total_chunks},
        "n": args.n,
        "valid": valid,
        "valid_rate": round(valid / args.n, 3),
        "wall_s": {"mean": round(mean_s, 1), "median": round(statistics.median(walls), 1),
                   "min": round(min(walls), 1), "max": round(max(walls), 1)},
        "entities_mean": round(statistics.fmean(r["entities"] for r in rows), 1),
        "relationships_mean": round(statistics.fmean(r["relationships"] for r in rows), 1),
        "projected_full_index_hours": round(total_chunks * mean_s / 3600, 1),
        "gate_20_of_20": valid == args.n,
        "gate_under_90s": max(walls) < 90,
        "rows": rows,
    }

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev["g3_extraction"] = result
    OUT.write_text(json.dumps(prev, indent=1))
    SAMPLES.write_text(json.dumps(kept, indent=1))

    print(f"\n  valid            {valid}/{args.n}   gate: "
          f"{'PASS' if result['gate_20_of_20'] else 'FAIL'}")
    print(f"  per chunk        mean {result['wall_s']['mean']}s  "
          f"max {result['wall_s']['max']}s   gate (<90s): "
          f"{'PASS' if result['gate_under_90s'] else 'FAIL'}")
    print(f"  corpus           {total_chunks} chunks")
    print(f"  full graph index {result['projected_full_index_hours']} h  "
          f"[measured, not projected from a token price]")


if __name__ == "__main__":
    main()
