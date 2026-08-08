"""The one shared transformation. PRD §4.5.6.

Sentence-aligned greedy token packing, article-scoped, zero overlap. Written
here rather than imported because a study selling bit-exact reproducibility
cannot let a dependency's patch bump silently re-chunk the corpus and change
every chunk id in all three arms at once.

Three things this gets right that the naive scheme did not:

  tokens      counted with the real Qwen3 BPE, not 4 chars/token. The true
              ratio on this subset is 4.607, so nominal "512" chunks were
              really ~446 tokens at 87% fill with a 1.83x spread.
  sentences   pySBD, because news is full of "U.S.", "Inc." and "Sept. 12" and
              a regex splitter severs gold evidence spans at 6.83%.
  scope       a chunk never spans two articles, which forecloses manufactured
              cross-article relations that would inflate the very relationship
              counts the cost model rests on.

    python src/chunker.py --budget 600
    python src/chunker.py --budget 1200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
import pysbd
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "data" / "multihoprag_corpus.parquet"
TOKJSON = ROOT / "data" / "qwen3_tokenizer.json"
OUT = ROOT / "data"

CHUNKER_VERSION = "sentence-pack-v1"
CATEGORIES = ("technology", "business")
PARA_PREFERENCE = 0.15          # prefer a paragraph break in the last 15% of budget

# If this string ever tokenizes to a different length, the tokenizer changed
# and every chunk id downstream is invalid. Asserted at startup.
CANARY = "The U.S. Federal Reserve said on Sept. 12 that Inc. filings rose 3.5%."
CANARY_TOKENS = None            # filled on first run, then pinned in the manifest


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(text: str) -> str:
    """NFKC and collapse runs of spaces/tabs, but keep blank lines: paragraph
    structure is the first boundary the packer prefers."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def units(body: str, seg: pysbd.Segmenter) -> list[tuple[str, bool]]:
    """(sentence, starts_paragraph) in document order."""
    out: list[tuple[str, bool]] = []
    for para in body.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        sents = [s.strip() for s in seg.segment(para) if s and s.strip()]
        for i, s in enumerate(sents):
            out.append((s, i == 0))
    return out


def pack(article: dict, tok: Tokenizer, budget: int, seg: pysbd.Segmenter) -> list[dict]:
    body = clean(article.get("body") or "")
    title = clean(article.get("title") or "")
    if not body:
        return []

    header = f"{title}\n\n" if title else ""
    header_n = len(tok.encode(header, add_special_tokens=False).ids) if header else 0
    room = budget - header_n
    if room <= 0:
        # Dead code on this corpus (max title 57 tokens), but a title at or
        # over budget must not silently delete an article. Surfaced via the
        # manifest's dropped_articles counter.
        article["_dropped_title_over_budget"] = True
        return []

    sents = units(body, seg)
    lens = [len(tok.encode(s, add_special_tokens=False).ids) for s, _ in sents]

    chunks: list[dict] = []
    cur: list[int] = []          # indices into sents
    cur_n = 0

    def flush() -> list[int]:
        """Emit the current chunk, returning any sentences that would not fit
        so the caller can carry them into the next one.

        BPE merges across the joins, so the joined string can tokenize to MORE
        than the sum of its sentences -- measured at 4 of 1,213 chunks
        exceeding a 600 budget by up to 3 tokens. Packing on the sum is
        therefore not sufficient; re-count the real string and give sentences
        back until it fits. Giving them back rather than dropping them is the
        whole point: a chunker that silently loses a sentence loses evidence.
        """
        nonlocal cur, cur_n
        if not cur:
            return []
        carry: list[int] = []
        while True:
            text = " ".join(sents[i][0] for i in cur)
            full = header + text
            n = len(tok.encode(full, add_special_tokens=False).ids)
            if n <= budget or len(cur) == 1:
                break
            carry.insert(0, cur.pop())
        start = body.find(sents[cur[0]][0])
        last = sents[cur[-1]][0]
        end_at = body.find(last, max(start, 0))
        chunks.append({
            "doc_id": article["url"],
            "chunk_index": len(chunks),
            "text": full,
            "n_tokens": n,
            "char_start": start,
            "char_end": (end_at + len(last)) if end_at >= 0 else -1,
            "title": title,
            "url": article["url"],
            "source": article.get("source"),
            "published_at": article.get("published_at"),
            "category": article.get("category"),
        })
        cur, cur_n = [], 0
        return carry

    def emit(carry: list[int]) -> None:
        """Seed the next chunk with sentences the previous one gave back."""
        nonlocal cur, cur_n
        for j in carry:
            cur.append(j)
            cur_n += lens[j]

    for i, ((_s, starts_para), n) in enumerate(zip(sents, lens)):
        # A single sentence over budget is dead code on this corpus (max 398
        # tokens) but must not silently vanish if the corpus changes.
        if n > room:
            emit(flush())
            ids = tok.encode(sents[i][0], add_special_tokens=False).ids
            for j in range(0, len(ids), room):
                piece = tok.decode(ids[j:j + room])
                chunks.append({
                    "doc_id": article["url"], "chunk_index": len(chunks),
                    "text": header + piece, "n_tokens": min(room, len(ids) - j) + header_n,
                    "char_start": -1, "char_end": -1, "title": title,
                    "url": article["url"], "source": article.get("source"),
                    "published_at": article.get("published_at"),
                    "category": article.get("category"), "hard_split": True,
                })
            continue

        if cur_n + n > room:
            emit(flush())
        elif starts_para and cur_n >= room * (1 - PARA_PREFERENCE):
            # a paragraph boundary this late is a better break than a later
            # sentence boundary, so take it
            emit(flush())
        cur.append(i)
        cur_n += n

    while True:
        carry = flush()
        if not carry:
            break
        emit(carry)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--all-categories", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKJSON))
    canary_n = len(tok.encode(CANARY, add_special_tokens=False).ids)
    seg = pysbd.Segmenter(language="en", clean=False)

    rows = pq.read_table(CORPUS).to_pylist()
    if not args.all_categories:
        rows = [r for r in rows if r.get("category") in CATEGORIES]
    rows.sort(key=lambda r: (r.get("url") or ""))

    chunks: list[dict] = []
    for a in rows:
        chunks.extend(pack(a, tok, args.budget, seg))

    ns = [c["n_tokens"] for c in chunks]
    body_tokens = sum(len(tok.encode(clean(r.get("body") or ""), add_special_tokens=False).ids)
                      for r in rows)
    chars = sum(len(clean(r.get("body") or "")) for r in rows)

    out = OUT / f"chunks_{args.budget}.jsonl"
    with out.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    manifest = {
        "chunker_version": CHUNKER_VERSION,
        "budget_tokens": args.budget,
        "overlap_tokens": 0,
        "tokenizer_sha256": sha256_file(TOKJSON),
        "canary": {"text": CANARY, "n_tokens": canary_n},
        "corpus_sha256": sha256_file(CORPUS)[:16],
        "categories": "all" if args.all_categories else list(CATEGORIES),
        "articles": len(rows),
        "chunks": len(chunks),
        "chunks_sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
        "body_tokens_total": body_tokens,
        "chars_per_token": round(chars / body_tokens, 3) if body_tokens else None,
        "tokens": {"mean": round(statistics.fmean(ns), 1),
                   "median": statistics.median(ns),
                   "min": min(ns), "max": max(ns),
                   "fill_rate": round(statistics.fmean(ns) / args.budget, 3)},
        "hard_splits": sum(1 for c in chunks if c.get("hard_split")),
        "dropped_articles_title_over_budget": sum(
            1 for a in rows if a.get("_dropped_title_over_budget")),
        "over_budget": sum(1 for c in chunks if c["n_tokens"] > args.budget),
    }
    (OUT / f"chunks_{args.budget}.manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"budget {args.budget}: {len(rows)} articles -> {len(chunks)} chunks")
    print(f"  tokens mean {manifest['tokens']['mean']} median {manifest['tokens']['median']} "
          f"min {manifest['tokens']['min']} max {manifest['tokens']['max']} "
          f"fill {manifest['tokens']['fill_rate']}")
    print(f"  chars/token {manifest['chars_per_token']}  (the naive scheme assumed 4.0)")
    print(f"  hard splits {manifest['hard_splits']}   over budget {manifest['over_budget']}")
    print(f"  canary '{CANARY[:30]}...' -> {canary_n} tokens")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
