"""Stage 10 prep — English single-hop control corpus (SQuAD v1.1 dev).

The five-language stage left one confound standing: language, corpus and
task shape changed together, so the abstention reversal cannot be pinned on
task shape alone. This corpus isolates it: same language as Stage 8
(English), same task shape as Stage 9 (single-hop extractive). If the
calibration reversal appears here too, task shape explains it; if English
single-hop still over-refuses, language/corpus does.

Contamination expectation, stated up front: SQuAD is the most-published QA
dataset in existence and its Wikipedia facts are old, so the closed-book
arm should land HIGH — that is not a bug in the control, it is a second
test of the parametric-answerability mechanism from the five-language
piece.

Follows the fetch_data.py pattern the multilingual prep lacked: source URL
pinned, content hash recorded, sampling expression stated, manifest updated
in place.

    python ops/prep_en_squad.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "multiling" / "en"
MANIFEST = ROOT / "data" / "multiling" / "sources.manifest.json"
URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
N_QUERIES = 100
CTX_TRIM = 2500


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=300).read()
    src_sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)["data"]

    corpus: list[dict] = []
    flat: list[dict] = []
    for art in data:
        for pi, para in enumerate(art["paragraphs"]):
            aid = f'{art["title"]}#p{pi}'
            corpus.append({"article_id": aid, "context": para["context"][:CTX_TRIM]})
            for qa in para["qas"]:
                golds = sorted({a["text"] for a in qa["answers"] if a["text"].strip()})
                if golds:
                    flat.append({"question_id": qa["id"], "question": qa["question"],
                                 "gold_article": aid, "golds": golds})

    step = max(len(flat) // N_QUERIES, 1)
    sample = flat[::step][:N_QUERIES]
    for i, q in enumerate(sample):
        q["qi"] = i

    with (OUT / "corpus.jsonl").open("w") as fh:
        for c in corpus:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (OUT / "queries.jsonl").open("w") as fh:
        for q in sample:
            fh.write(json.dumps(
                {k: q[k] for k in ("qi", "question_id", "question",
                                   "gold_article", "golds")},
                ensure_ascii=False) + "\n")

    aids = {c["article_id"] for c in corpus}
    missing = [q["qi"] for q in sample if q["gold_article"] not in aids]
    verbatim = sum(any(g in next(c["context"] for c in corpus
                                 if c["article_id"] == q["gold_article"])
                       for g in q["golds"]) for q in sample)

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["en"] = {
        "dataset": "SQuAD v1.1 dev",
        "source_url": URL,
        "source_sha256": src_sha,
        "native_or_translated": "native",
        "sampling": f"answerable flattened dev qas[::{step}][:{N_QUERIES}]",
        "article_id": "wikipedia title + paragraph index",
        "corpus_sha256": hashlib.sha256((OUT / "corpus.jsonl").read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256((OUT / "queries.jsonl").read_bytes()).hexdigest(),
        "n_corpus": len(corpus), "n_queries": len(sample),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    print(f"en: corpus {len(corpus)} paragraphs, queries {len(sample)}, "
          f"golds missing {len(missing)}, verbatim {verbatim}/{len(sample)}")
    if missing:
        raise SystemExit(f"gold articles missing from corpus: {missing}")


if __name__ == "__main__":
    main()
