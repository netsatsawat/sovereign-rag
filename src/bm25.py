"""BM25, implemented here rather than imported, for two reasons.

Determinism: this experiment's whole claim is that the numbers reproduce
bit-exactly, and a pinned third-party ranker still leaves tie-breaking and
tokenisation to someone else's changelog.

Honesty about the growing corpus: BM25's IDF depends on N and df, and its
length normaliser on avgdl. So injecting documents moves the scores of
documents already in the index. That is part of what the experiment is
measuring rather than a wart to engineer around, and a retriever that hid it
would fake the finding.

Latin script only: normalize() strips everything outside [a-z0-9], which
deletes Thai, Chinese and Japanese script outright and strips the diacritics
off Vietnamese and Spanish, breaking their words into fragments.

The failure this causes is not silence, which is what makes it dangerous. A
Thai sentence on its own normalises to nothing, but a real Thai document is
not pure Thai: it carries digits, names and Latin loanwords, and those
survive. Measured on this repo's own corpora, data/multiling/th indexes to
26,482 tokens across 1,863 of 1,912 documents, and 8 of its 100 queries come
back with a ranked list rather than []. The query for a film title tokenises
to ['2'] and returns five confidently ranked documents chosen by that digit.
zh and ja behave the same way (15 and 29 of 100). So the reader who checks
whether this tokeniser works on their language by looking for an empty result
gets the wrong answer: the residue is enough to look like retrieval.

That is this file's limit, not the study's: the five-language replication uses
LangBM25 in ops/stage9_lang.py, which tokenises per language (pythainlp /
janome / jieba, \\w+ for es and vi) at k1=1.5 rather than 1.2. It is not
papered over here either: normalize() warns the first time it discards a
non-ASCII word character, because that discard is the moment the scores stop
meaning what they appear to mean.
"""

from __future__ import annotations

import math
import re
import unicodedata
import warnings

K1 = 1.2
B = 0.75

_WORD = re.compile(r"[a-z0-9]+")

# A word character that is not ASCII, i.e. exactly what the [a-z0-9] filter
# below throws away. Only consulted when the text is not already ASCII.
_NON_ASCII_WORD = re.compile(r"[^\W\x00-\x7f]")

_LATIN_ONLY_WARNING = (
    "bm25.normalize() keeps [a-z0-9] and has just discarded word characters "
    "outside it: this tokeniser is Latin-only. Thai, Chinese and Japanese lose "
    "their script entirely, leaving only whatever digits and Latin fragments the "
    "text happened to contain, so scores are computed, and ranked, on the "
    "residue rather than on the language; Vietnamese and Spanish lose their "
    "diacritics and break into fragments; English drops accented characters whole "
    "rather than folding them, so a borrowed word is truncated ('cafe' loses its "
    "last letter, not just its accent) and will not match its unaccented spelling "
    "on either side. Non-Latin corpora want LangBM25 in ops/stage9_lang.py: "
    "per-language tokenisers, at k1=1.5 rather than this file's 1.2."
)


def normalize(text: str) -> str:
    """NFKC, casefold, strip non-alphanumerics. Used for tokenising and for
    gold-answer containment, so both sides agree on what a match is.

    Latin-only: non-ASCII word characters are dropped whole, not transliterated
    or folded, so 'cafe' with an accent loses the accented letter itself and
    Thai, Chinese and Japanese script vanishes. What is left of a real non-Latin
    document is its digits and Latin fragments, which still score. Warns the
    first time it discards anything; see the module docstring for what to reach
    for instead."""
    text = unicodedata.normalize("NFKC", text).lower()
    if not text.isascii() and _NON_ASCII_WORD.search(text):
        warnings.warn(_LATIN_ONLY_WARNING, stacklevel=2)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """The words of normalize()'s output, and its Latin-only limit with them:
    non-Latin script tokenises to nothing, so what comes back for such a string
    is whatever digits and Latin fragments it carried, often a short list that
    is worse than an empty one, because a BM25 built on it ranks confidently."""
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
            # Diverged from the sovereign-learning-loop original, which
            # returned chunks 0..k-1 here: reporting arbitrary documents as
            # "retrieved" for a zero-overlap query would fake a hit. An empty
            # list scores as a miss, which is what actually happened.
            return []
        ranked = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
        return [ix for ix, _ in ranked[:k]]
