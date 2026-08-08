"""BM25, implemented here rather than imported, for two reasons.

Determinism: this experiment's whole claim is that the numbers reproduce
bit-exactly, and a pinned third-party ranker still leaves tie-breaking and
tokenisation to someone else's changelog.

Honesty about the growing corpus: BM25's IDF depends on N and df, and its
length normaliser on avgdl. So injecting documents moves the scores of
documents already in the index. That is not a wart to be engineered around
here -- it is part of what the experiment is measuring, and a retriever that
hid it would fake the finding.
"""

from __future__ import annotations

import math
import re
import unicodedata

K1 = 1.2
B = 0.75

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """NFKC, casefold, strip non-alphanumerics. Used for tokenising and for
    gold-answer containment, so both sides agree on what a match is."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


class BM25:
    """Rebuilt from scratch on every round. Rebuilding is the point: an index
    that cached IDF across rounds would not see the corpus it actually has."""

    def __init__(self, docs: list[str]):
        self.docs = docs
        self.toks = [tokenize(d) for d in docs]
        self.lens = [len(t) for t in self.toks]
        self.n = len(docs)
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0

        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        # Inverted index: term -> [(doc_ix, term_freq), ...]. Scoring every
        # document per query is O(N) and turns a 5-minute run into a 75-minute
        # one; postings give identical results because a document with no query
        # term scores exactly zero.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for ix, t in enumerate(self.toks):
            counts: dict[str, int] = {}
            for w in t:
                counts[w] = counts.get(w, 0) + 1
            self.tf.append(counts)
            for w, c in counts.items():
                df[w] = df.get(w, 0) + 1
                self.postings.setdefault(w, []).append((ix, c))
        self.df = df
        self.idf = {
            w: math.log(1.0 + (self.n - d + 0.5) / (d + 0.5)) for w, d in df.items()
        }

    def score(self, query: str, doc_ix: int) -> float:
        s = 0.0
        dl = self.lens[doc_ix]
        tf = self.tf[doc_ix]
        for w in tokenize(query):
            f = tf.get(w)
            if not f:
                continue
            denom = f + K1 * (1.0 - B + B * dl / self.avgdl) if self.avgdl else 1.0
            s += self.idf.get(w, 0.0) * (f * (K1 + 1.0)) / denom
        return s

    def top_k(self, query: str, k: int) -> list[int]:
        """Ties break on ascending doc index, which is assignment order, so a
        run is reproducible regardless of dict iteration order.

        Accumulates over postings rather than scanning the corpus. Documents
        sharing no term with the query score zero and cannot enter the top k
        while any positive score exists; the guard below covers the degenerate
        case where nothing matches at all."""
        acc: dict[int, float] = {}
        for w in tokenize(query):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for ix, f in self.postings[w]:
                dl = self.lens[ix]
                denom = f + K1 * (1.0 - B + B * dl / self.avgdl) if self.avgdl else 1.0
                acc[ix] = acc.get(ix, 0.0) + idf * (f * (K1 + 1.0)) / denom
        if not acc:
            return list(range(min(k, self.n)))
        ranked = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
        return [ix for ix, _ in ranked[:k]]
