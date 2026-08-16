"""Builds examples/minimal_eval.ipynb, the study's method as a runnable
notebook, then executes it against the local ollama so the committed copy
carries real outputs. Rerun after editing cells below.

    ./.venv/bin/python ops/build_example_notebook.py
"""

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "examples" / "minimal_eval.ipynb"

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# Run this study's evaluation on your own corpus

The [companion articles](https://github.com/netsatsawat/sovereign-rag#readme) end with three questions for your eval team.
This notebook is the fifteen-minute version of answering them on your own data, with a local model. One pass gives you:

1. a **closed-book arm**: what the model scores with *no* retrieval. High closed-book means your benchmark is answerable from the weights, and your "RAG lift" is measuring recall, not retrieval.
2. a **RAG arm**, paired per question against closed-book with an exact McNemar test.
3. the **calibration cross**: when the model says *insufficient information*, was the evidence actually missing?
4. the **constant-answer baseline**: what the most common gold answer scores with no model at all.
5. the **escape-hatch ablation**, the same RAG run with the "reply exactly: insufficient information" sentence deleted. In this study, that one sentence moved a 30B model **forty points**.

**Prerequisites:** [ollama](https://ollama.com) running locally with a model pulled (`ollama pull qwen3:8b`), and this repo's venv (`pip install -r requirements.txt`). The bundled sample is 15 documents and 6 questions (a SQuAD-dev subset, CC BY-SA) so the whole notebook runs in a few minutes on a laptop-class machine."""),

code("""import json, sys, urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, "../ops")          # the study's own machinery, reused
from stage9_lang import LangBM25, contained, norm, wilson, mcnemar_exact

MODEL = "qwen3:8b"                     # any ollama model
K = 5                                  # passages handed to the generator

corpus  = [json.loads(l) for l in open("sample_data/corpus.jsonl")]
queries = [json.loads(l) for l in open("sample_data/queries.jsonl")]
print(f"{len(corpus)} documents · {len(queries)} questions")
print("example:", queries[0]["question"][:80])
print("golds:  ", queries[0]["golds"])"""),

md("""## Data format

Two jsonl files are the whole contract: swap in your own and everything below runs unchanged:

```
corpus.jsonl   {"article_id": "...", "context": "..."}                          one per document
queries.jsonl  {"qi": 0, "question": "...", "gold_article": "...", "golds": [...]}  one per question
```

`golds` is every acceptable answer string (scoring is normalized substring containment: NFKC, casefold, punctuation-stripped; `ops/stage9_lang.py` documents the caveats). `gold_article` powers the calibration cross. A hundred questions labelled by your own team beats ten thousand synthetic ones. That was this study's experience across six corpora."""),

code("""# BM25 with language-aware segmentation. "en" = whitespace/regex; pass
# "th" / "ja" / "zh" instead and the same class segments properly. A
# whitespace BM25 silently deletes those scripts (the five-language study's
# retrieval trap).
bm = LangBM25("en", [c["context"] for c in corpus])
aid_of = [c["article_id"] for c in corpus]

q = queries[0]
top = bm.top_k(q["question"], K)
print("query:    ", q["question"][:70])
print("retrieved:", [aid_of[i] for i in top])
print("gold in top-k:", q["gold_article"] in {aid_of[i] for i in top})"""),

md("""## The two arms, and the sentence under test

Same prompts as the study's English arms. `ESCAPE` is the sentence the ablation later deletes: the standard escape hatch nearly every production RAG template ships in some form."""),

code("""ESCAPE_RAG = " If the context does not contain the answer, reply exactly: insufficient information"
ESCAPE_CLOSED = " If you do not know, reply exactly: insufficient information"

PROMPT_RAG = ("Answer the question using ONLY the context below. Be direct and "
              "brief: give the answer itself, no preamble.{esc}\\n\\n"
              "CONTEXT:\\n{context}\\n\\nQUESTION: {q}\\n\\nANSWER:")
PROMPT_CLOSED = ("Answer the question directly and briefly: the answer itself, "
                 "no preamble.{esc}\\n\\nQUESTION: {q}\\n\\nANSWER:")

def generate(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0, "seed": 0, "num_ctx": 8192,
                                   "num_predict": 128}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return (json.loads(r.read()).get("response") or "").strip()

def run_arms(with_escape=True):
    esc_r = ESCAPE_RAG if with_escape else ""
    esc_c = ESCAPE_CLOSED if with_escape else ""
    rows = []
    for q in queries:
        top = bm.top_k(q["question"], K)
        ctx = "\\n\\n".join(corpus[i]["context"] for i in top)
        gold_hit = q["gold_article"] in {aid_of[i] for i in top}
        rows.append({"qi": q["qi"], "golds": q["golds"], "gold_retrieved": gold_hit,
                     "closed": generate(PROMPT_CLOSED.format(esc=esc_c, q=q["question"])),
                     "rag": generate(PROMPT_RAG.format(esc=esc_r, context=ctx, q=q["question"]))})
    return rows

rows = run_arms(with_escape=True)
for r in rows[:3]:
    print(f"gold: {r['golds'][0][:30]:32s} rag: {r['rag'][:45]}")"""),

md("""## Scoring: containment, abstention, and the paired test

Read in this order: the closed-book floor first, then the paired lift, never the RAG number alone."""),

code("""def abstained(ans):
    return "insufficientinformation" in norm(ans)

def arm_stats(rows, arm):
    ok = sum(contained(r["golds"], r[arm]) for r in rows)
    ab = sum(abstained(r[arm]) for r in rows)
    return ok, ab

for arm in ("closed", "rag"):
    ok, ab = arm_stats(rows, arm)
    n = len(rows)
    print(f"{arm:7s} containment {100*ok/n:5.1f}%  (Wilson95 {wilson(ok, n)})  abstained {100*ab/n:.0f}%")

b = sum(contained(r["golds"], r["rag"]) and not contained(r["golds"], r["closed"]) for r in rows)
c = sum(contained(r["golds"], r["closed"]) and not contained(r["golds"], r["rag"]) for r in rows)
print(f"\\nRAG over closed-book: {b} won by RAG only, {c} by closed only "
      f"-> delta {100*(b-c)/len(rows):+.1f}pp, exact McNemar p = {mcnemar_exact(b, c):.3f}")
print("(6 questions is a smoke test; the p-value earns meaning at your real n)")"""),

md("""## The calibration cross

The single most diagnostic table this study produced. An honest model abstains where retrieval **missed**; an obedient one abstains wherever the instruction gives it an exit. In the study's English multi-hop arm the same model refused 56% of questions whose evidence was fully present, then was near-perfectly calibrated on single-hop arms in five languages. Task shape decides; only this cross shows you."""),

code("""for label, cond in (("gold retrieved", True), ("gold missed  ", False)):
    sub = [r for r in rows if r["gold_retrieved"] == cond]
    if sub:
        ab = sum(abstained(r["rag"]) for r in sub)
        print(f"abstains when {label}: {100*ab/len(sub):5.1f}%  (n={len(sub)})")

const = Counter(g for q in queries for g in q["golds"][:1]).most_common(1)[0][0]
hits = sum(contained(q["golds"], const) for q in queries)
print(f'\\nconstant answer "{const}": {100*hits/len(queries):.1f}% with no model at all')
print("any slice where the constant wins is a slice your evaluation cannot see")"""),

md("""## The ablation: delete one sentence, re-run

Identical queries, identical contexts, identical model, with the escape-hatch sentence removed. In the study this flipped a 15.8-point loss into a 24-point win for the same weights (94 of 155 refusals had been suppressing answers the model demonstrably had, 42 became honest-but-wrong, 19 just rephrased the refusal). Whatever it does on *your* stack is a number you are currently running blind."""),

code("""rows_nohatch = run_arms(with_escape=False)

for label, rs in (("with hatch   ", rows), ("without hatch", rows_nohatch)):
    ok, ab = arm_stats(rs, "rag")
    print(f"{label} RAG containment {100*ok/len(rs):5.1f}%  abstained {100*ab/len(rs):.0f}%")

flips = [(r1["golds"][0], r1["rag"][:35], r2["rag"][:35])
         for r1, r2 in zip(rows, rows_nohatch)
         if abstained(r1["rag"]) and not abstained(r2["rag"])]
for gold, before, after in flips:
    print(f"\\n  was refused, now answers -> gold: {gold[:28]} | answer: {after}")"""),

md("""## Point it at your data

1. Write your `corpus.jsonl` / `queries.jsonl` in the format above and change the two paths in cell 1.
2. Non-English corpus? `LangBM25("th" | "ja" | "zh", ...)` segments properly (one pip install each: `pythainlp`, `janome`, `jieba`); borrow the native prompt + escape-phrase pairs from `ops/stage9_lang.py`; they ran the five-language study.
3. For hundreds of queries, lift `run_arms` into a script with per-row checkpointing: every stage script in `ops/` follows that pattern (checkpointed jsonl, deterministic sampling, resumable), so promoting this notebook to that rigor is appetite, not rewriting.

The full study (GraphRAG arm, 27B and Muse Glimmer comparisons, bootstrap CIs, the learning-layer negative, five languages) is the industrial version of exactly this loop. [TUTORIAL.md](../TUTORIAL.md) is the prose map; the `reports/*.md` files are what it found."""),
]

nb.cells = cells
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}

client = NotebookClient(nb, timeout=900, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT / "examples")}})
client.execute()
nbf.write(nb, OUT)
print(f"wrote + executed {OUT}")
