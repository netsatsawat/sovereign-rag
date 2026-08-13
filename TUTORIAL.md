# Run this evaluation on your own corpus

The articles this repo accompanies end with three questions for your eval
team. This tutorial is the fifteen-minute version of answering them on
your own data, with one script and a local model.

What you get from one run:

- a **closed-book arm** — what the model scores with *no* retrieval. If
  this is high, your benchmark is answerable from the weights and your
  "RAG lift" number is measuring recall, not retrieval.
- a **RAG arm** with paired McNemar against closed-book — retrieval's
  actual, statistically-tested contribution.
- the **calibration cross** — when the model says "insufficient
  information," was the evidence actually missing? The difference between
  an honest model and an obedient one lives in this table.
- the **constant-answer baseline** — what the most common gold answer
  scores with no model at all. Any slice where the constant wins is a
  slice your evaluation cannot see.
- the **escape-hatch ablation** (one flag) — the same run with the
  "reply exactly: insufficient information" sentence deleted. In this
  study that single sentence moved one model forty points.

## 0. Prerequisites

- [ollama](https://ollama.com) running locally with a model pulled
  (`ollama pull qwen3:8b` is a fine start)
- this repo's Python env: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`

## 1. Smoke-run on the bundled sample

Fifteen documents and six questions (a subset of SQuAD dev, CC BY-SA):

```bash
python examples/minimal_eval.py \
  --corpus examples/sample_data/corpus.jsonl \
  --queries examples/sample_data/queries.jsonl \
  --model qwen3:8b
```

Twelve model calls, a couple of minutes on a laptop-class machine. You
get `results.json` plus a resumable per-row log. Read the output in this
order: `arms.closed.containment_pct` first (the answerability floor),
then `rag_over_closed` (the lift and its p-value), then
`calibration_cross` (are the refusals where the retrieval failures are?),
then `constant_answer_baseline` (is anything beating your model with no
model?).

## 2. Point it at your data

Two jsonl files:

```
corpus.jsonl   {"article_id": "...", "context": "..."}          one per document
queries.jsonl  {"qi": 0, "question": "...",
                "gold_article": "...", "golds": ["...", "..."]}  one per question
```

`golds` is every acceptable answer string; scoring is normalized
substring containment (NFKC, casefold, punctuation/space-stripped — see
`ops/stage9_lang.py` for the caveats, including digit boundaries and
combining marks). `gold_article` powers the calibration cross; if you
don't know which document holds each answer, you can omit the cross but
you lose the most interesting table.

A hundred questions labelled by your own team beats ten thousand
synthetic ones. That was this study's experience across six corpora.

## 3. The two runs that matter

```bash
# the real configuration
python examples/minimal_eval.py --corpus my/corpus.jsonl \
  --queries my/queries.jsonl --model qwen3:8b --out with_hatch.json

# the ablation: identical, escape hatch deleted
python examples/minimal_eval.py --corpus my/corpus.jsonl \
  --queries my/queries.jsonl --model qwen3:8b --no-escape --out no_hatch.json
```

Compare `arms.rag.containment_pct` across the two files. If they differ
by a lot, your escape-hatch sentence is a load-bearing part of your
system's accuracy — a dial you are turning blind until you measure it.
(If your production prompt uses a different phrase, pass it via
`--escape "your exact phrase"` so abstention detection matches.)

## 4. Non-English corpora

Thai, Japanese and Chinese write without spaces, and a whitespace BM25
silently destroys itself on them. One flag fixes it:

```bash
python examples/minimal_eval.py ... --tokenizer th   # or ja, zh
```

(needs `pythainlp`, `janome`, or `jieba` respectively — each is one pip
install). Prompts default to English; for a native-language run, borrow
the prompt/escape pairs in `ops/stage9_lang.py`, which were used for the
five-language study.

## 5. Where to go deeper

The full study's stages (README has the map) are the industrial version
of this loop: per-stratum analysis, paired cross-model comparisons,
bootstrap CIs, graph retrieval, the learning-layer negative. Every stage
script follows the same pattern as the minimal example — checkpointed
rows, deterministic sampling, committed artifacts — so promoting your
eval from this tutorial to that rigor is a matter of appetite, not
rewriting.
