"""E1 producer: the chunker-integrity block in stage0.json, from committed code.

Asserts, for both budgets: every pySBD sentence of every article is present in
that article's chunks (nothing lost), no sentence duplicated beyond its source
count, nothing over budget. Merges the result into stage0.json.

    python ops/chunker_integrity.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pysbd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from chunker import CATEGORIES, CORPUS, clean, units  # noqa: E402


def main() -> None:
    seg = pysbd.Segmenter(language="en", clean=False)
    rows = [r for r in pq.read_table(CORPUS).to_pylist() if r.get("category") in CATEGORIES]
    rows.sort(key=lambda r: r.get("url") or "")

    block = {}
    for budget in (600, 1200):
        chunks = [json.loads(l) for l in (ROOT / "data" / f"chunks_{budget}.jsonl").open()]
        by_doc: dict[str, list[dict]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)
        manifest = json.loads((ROOT / "data" / f"chunks_{budget}.manifest.json").read_text())

        lost = dupes = 0
        for a in rows:
            body = clean(a.get("body") or "")
            if not body:
                continue
            title = clean(a.get("title") or "")
            joined_parts = []
            for c in sorted(by_doc.get(a["url"], []), key=lambda x: x["chunk_index"]):
                t = c["text"]
                if title and t.startswith(title + "\n\n"):
                    t = t[len(title) + 2:]
                joined_parts.append(t)
            joined = " ".join(joined_parts)
            for s, _ in units(body, seg):
                if s not in joined:
                    lost += 1
                elif joined.count(s) > body.count(s):
                    dupes += 1

        over = sum(1 for c in chunks if c["n_tokens"] > budget)
        block[f"at_{budget}"] = {
            "chunks": len(chunks), "sentences_lost": lost,
            "sentences_duplicated": dupes, "over_budget": over,
            "manifest_sha256": manifest["chunks_sha256"],
            "dropped_articles_title_over_budget":
                manifest.get("dropped_articles_title_over_budget", 0),
            "pass": lost == 0 and dupes == 0 and over == 0,
        }
        b = block[f"at_{budget}"]
        print(f"budget {budget}: {b['chunks']} chunks, lost {lost}, dup {dupes}, "
              f"over {over} -> {'PASS' if b['pass'] else 'FAIL'}")

    out = ROOT / "reports" / "stage0.json"
    d = json.loads(out.read_text())
    d["e1_chunker_integrity"] = block          # merge, never overwrite
    out.write_text(json.dumps(d, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
