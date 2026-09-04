# sovereign-rag

> Experiments that run on your own computer, with no internet service
> involved, and test whether the usual ways of handing documents to a
> language model (a program that reads text and writes text back) make its
> answers better. Every number here recomputes from a file committed in this
> repo.

Banks, hospitals, telcos (telecommunications companies) and governments,
the organizations I've spent 16 years with, often can't send their documents
to a cloud service. Their data
can't leave the building. So this study ran every model on one machine, with
no account or API key anywhere. It asked the three questions vendors usually
answer with slides. Does giving the model your documents help? Is a knowledge
graph, a map of the named things in the documents and how they link, better
still? Does the system get better from feedback? The measured answers were
yes, no, and no.

One more result fell out on the way. A single sentence in the instructions
given to one model changed how often it got the answer right, from 32.92% to
72.50% of the same 240 questions with the same documents in front of it
(`reports/stage10_hatch.json`). If you write instructions like that for a
model, that sentence matters more than you think. You will not know how much
until you measure it.

This repo is for teams in regulated or data-sensitive settings who want to
try RAG (retrieval-augmented generation: a search step feeds your own
documents to the model before it answers) without a single word of their
documents leaving their own machines.
It is also a copyable template for testing your own RAG setup. Every claim
below links to the report and the result file it came from.

## What you need to know first

- Language model. A program that reads text and writes text back. All the
  models here ran on one local machine. "8B" or "30B" after a model's name
  means 8 or 30 billion parameters, the numbers the model learned during
  training. More parameters usually means a bigger, slower model.
- Prompt and token. The prompt is the text you send to the model: the
  instructions, the question, and any documents. A token is the unit a model
  reads and writes in, roughly a short word or a piece of a longer one.
- RAG, short for retrieval-augmented generation. Before the model answers, a
  search step finds a few passages from your documents and pastes them into
  the prompt. The model then answers from your text instead of from memory
  alone. A corpus is the set of documents the search runs over.
- Arm. One setup under test, run on the same questions as every other setup,
  so results can be compared question by question. The closed-book arm gives
  the model no documents at all. It is the floor. If closed-book already
  scores high, the model knew the answers before any search, and any "RAG
  lift" is suspect.
- BM25 and dense retrieval. Two kinds of search. BM25 is keyword search. It
  scores a passage by how many of the question's words it contains, weighted
  by how rare those words are, and it needs no model. Dense retrieval uses a
  small model, called an embedding model, to turn each passage and the
  question into a list of numbers. It then picks the passages whose numbers
  sit closest to the question's. Here that small model is
  `qwen3-embedding:0.6b`.
- Knowledge graph. A third kind of search. A model reads every passage and
  writes out the named things in it (people, companies, places) and the links
  between them. Search then walks that web of names to reach passages.
- Containment. The scoring rule. An answer counts as correct if one of the
  gold answers (the labelled right answers) appears inside it after
  lower-casing and stripping punctuation and spaces. It is a substring check.
  No second model judges the answers.
- The escape hatch. A sentence in the prompt that tells the model to reply
  exactly "insufficient information" when the context lacks the answer.
  Abstention is how often the model does that.
- strict@k and hits@k. Both measure search. hits@k asks whether at least one
  needed document is in the top k results. strict@k asks whether every needed
  document is. hits@10 sat near the ceiling on both English corpora, 98.93 on
  HotpotQA and between 96.7 and 99.3% on the news corpus
  ([reports/hotpotqa-transfer.md](reports/hotpotqa-transfer.md)), so that
  measure could not tell the methods apart. This study reports strict@k
  instead, which can.
- pp, p and n. pp is percentage points, the gap between two percentages. p is
  the chance that a gap this large is luck alone: near 1 means no evidence,
  near 0 means strong evidence. n is the number of questions.

## Run it

### Offline: recompute one result

```bash
git clone https://github.com/netsatsawat/sovereign-rag && cd sovereign-rag
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Python 3.12 is the version the repo's automated check uses. Save the block
below as `recompute.py` in the repo folder, then run
`.venv/bin/python recompute.py`.

The script reads `reports/stage7_hotpot.json`. HotpotQA is a public question
set built on Wikipedia. For each of 1,500 questions and each of two search
methods, the file holds a 1 if that method found every document the question
needed in its top 10, else a 0. The script takes the difference between the
two methods and prints it under the label delta, where delta just means the
difference. It then draws 10,000 random resamples of the questions to get a
95% confidence interval, the range the difference would likely land in on a
rerun.

```python
import json, random
d = json.load(open("reports/stage7_hotpot.json"))
b, e = d["per_query_strict"], d["per_query_strict_dense"]
diffs = [ei - bi for bi, ei in zip(b, e)]
print("delta", 100 * sum(diffs) / len(diffs))
rng = random.Random(0)
boots = sorted(100 * sum(rng.choices(diffs, k=len(diffs))) / len(diffs)
               for _ in range(10_000))
print("ci95", boots[249], boots[9749])
```

Output:

```
delta 6.4
ci95 4.0 8.866666666666667
```

Read it as: on HotpotQA, dense retrieval found every document a question
needed 6.4 points more often than BM25, and the whole interval sits above
zero, so the gap is not luck. You will see exactly the two lines above,
because the script fixes its random seed to 0. That gives the interval
[4.0, 8.87]. The study's own report shows [3.93, 8.87] instead, only because
it drew its resamples with a different random seed. A different seed shifts
the interval edges by about 0.1, and the 6.4-point gap is identical either way
([reports/hotpotqa-transfer.md](reports/hotpotqa-transfer.md)).

The second offline command is the check that runs on every push to GitHub
(CI, short for continuous integration):

```bash
.venv/bin/python scripts/verify_readme_claims.py
```

It recomputes the study's headline numbers from the committed result files,
checks that `STATUS.md` quotes them, and checks that every file this README
links to exists. When nothing has been changed, its last line is
`every quoted README number matches its artifact`. An artifact is a committed
result file. The check needs no model and no network.

### With a local model: the notebook

[examples/minimal_eval.ipynb](examples/minimal_eval.ipynb) runs five checks
on a bundled sample of 15 documents and 6 questions: (1) answer with no
documents, (2) answer with retrieved documents, (3) does the model abstain
when search missed the right document, (4) what a fixed constant answer
scores, (5) the same run with the escape-hatch sentence removed. It is
committed with its outputs, so you can read the whole run on GitHub first.
Three things it needs:

1. ollama, a free program that downloads and runs models on your own
   computer. Install it from [ollama.com](https://ollama.com) and start it.
   It listens on `localhost:11434`. Then download the model with
   `ollama pull qwen3:8b`, about 5 GB (`models.manifest.json` records the
   exact byte count).
2. Jupyter, the notebook viewer, which `requirements.txt` does not install:
   `.venv/bin/pip install jupyterlab`
3. The notebook's own folder as the working directory, because it imports
   the study's code from `../ops`. Jupyter sets that for you when you open
   the file:

```bash
.venv/bin/jupyter lab examples/minimal_eval.ipynb
```

What comes out, from the committed outputs. First the two arms. Wilson95 is
the 95% interval for a percentage. Exact McNemar is a paired test that counts
only the questions where the two arms disagree. A smoke test is a quick check
that the code runs, not a real measurement, and with 6 questions p cannot get
small:

```
closed  containment  50.0%  (Wilson95 [18.76, 81.24])  abstained 0%
rag     containment  66.7%  (Wilson95 [30.0, 90.32])  abstained 17%

RAG over closed-book: 1 won by RAG only, 0 by closed only -> delta +16.7pp, exact McNemar p = 1.000
(6 questions is a smoke test; the p-value earns meaning at your real n)
```

The calibration cross and the constant-answer baseline come next. The cross
asks: when the model abstained, had search found the document holding the
right answer ("gold retrieved") or missed it ("gold missed")? A well-behaved
model abstains on the misses. The constant-answer baseline is the score you
get by giving the most common gold answer to every question, with no model:

```
abstains when gold retrieved:   0.0%  (n=5)
abstains when gold missed  : 100.0%  (n=1)

constant answer "Denver Broncos": 16.7% with no model at all
```

The escape-hatch ablation comes last. An ablation removes one part and
reruns to see what that part did. Here the part is the hatch sentence:

```
with hatch    RAG containment  66.7%  abstained 17%
without hatch RAG containment  66.7%  abstained 0%
```

Six questions prove the pipeline runs and nothing more. [TUTORIAL.md](TUTORIAL.md)
describes the two jsonl files (one JSON record per line) you swap in to run
the same five checks on your own documents.

## What the study found

### Giving the model documents helped by 23.5 points

The English corpus is MultiHop-RAG, a set of news articles with multi-hop
questions. A multi-hop question needs two or more documents to answer.
"Which suppliers are affected if regulation X changes?" is the classic shape.
The news questions come in three types. Inference questions ask for a name
or a fact. Comparison and temporal questions ask whether two articles agree
or which event came first, and their answers are mostly yes or no.

Three arms ran on the same 1,381 questions with `qwen3:8b`: closed-book (no
documents, labelled A0 in the files), BM25 RAG (keyword search, labelled
plain) and the knowledge graph. Containment came out plain 63.1% vs A0 39.5%
vs graph 46.6% at n=1,381. Keyword search lifted the score by 23.5 points
(63.07 minus 39.54 before rounding). Every pair differs by more than chance.
That holds after the Holm correction, which raises the bar because three
comparisons were made at once. Source: `reports/stage3_final.json`, prose in
[reports/generation-arms.md](reports/generation-arms.md). What to do with it:
never quote a RAG number without its closed-book floor beside it.

### The knowledge graph lost to keyword search everywhere

Documents are cut into pieces called chunks first. Building the graph, the
indexing step, means having the model read every chunk and write out the
names and links in it. On this corpus 2.9% of chunks were dropped because
the model's output ran past its length limit or could not be parsed. BM25
needs no model to index at all. On strict@10 the graph scored 24 points
below keyword search: −24.19 [−26.72, −21.72], an interval that never reaches
zero, and it lost on all three question types. Against closed-book at answer
time it still won, 46.6% vs 39.5%. So the graph helped a little and cost a
lot.

Microsoft's GraphRAG popularised this design. The version tested here only
looks up names from the question in the graph, and the 8B model did the name
extraction. A second GraphRAG design, which first summarises whole clusters
of the graph and answers from those summaries, was not tested and might score
better. This study only shows that this one did
worse, and [reports/graph-arm.md](reports/graph-arm.md) says so. Source:
`reports/stage2_graph.json`.

### The feedback loop made things slightly worse

The learning layer gives documents that helped past questions a bonus in
future rankings. It is the mechanism most teams reach for when they promise
a system that improves from feedback instead of from vibes. I wrote four
pass-or-fail gates before the run. Two failed. On questions that resembled
earlier ones, where the bonus should help most, the learned re-ranking scored
1.40 strict@10 points below plain BM25. Handing the same bonuses to random
documents (the shuffled-credit control) scored the same, so the feedback
carried no signal. I filed the result as a pre-registered negative rather
than burying it. Source:
[reports/learning-layer.md](reports/learning-layer.md),
`reports/stage4_learning.json`.

### The retrieval winner depends on the corpus

On the full news set (609 articles, 2,255 questions) keyword search found
every needed document 48.74% of the time at strict@10, 12.95 points more than
dense. On HotpotQA, the Wikipedia question set from the recompute above, the winner
flipped. Those 1,500 questions were searched over a pool of 14,526
paragraphs. BM25 scored 68.4 at strict@10 and dense scored 74.80, so dense
won by 6.40 points, interval [3.93, 8.87]. The stage 7 report puts that
Wikipedia pool at ten times the news corpus. Neither search method is the
winner. Measure on the corpus you will deploy on. What did hold on both was
the metric lesson: hits@10 was 98.93 on HotpotQA and 96.7 to 99.3 on news,
near the ceiling for every method, while strict@10 told the methods apart on
both. Sources: `reports/stage5_full_corpus.json` and
[reports/hotpotqa-transfer.md](reports/hotpotqa-transfer.md).

### One prompt sentence moved a model nearly forty points

Muse Glimmer is a 30-billion-parameter model. It ran through llama.cpp, a
second local model runner, separate from ollama. On a 240-question sample of
the news set it answered every question in both arms: 480 calls, no
failures. With documents it scored 15.83 points below `qwen3:8b`. It
abstained on 64.6% of those RAG questions. So I deleted the escape-hatch
sentence from the prompt and reran the same 240 questions with the same
retrieved text in front of the model. Containment went from 32.92 to 72.50,
a paired gain of 39.58pp (p = 5.0e-29). 95 questions went from wrong to
right, and none went the other way. Here is where those 95 came from. The
hatch arm refused 155 questions in all, and one more question it answered
wrongly:

| Under the hatch | After removing the hatch | Count |
| --- | --- | --- |
| refused | right answer | 94 |
| refused | still wrong | 61 |
| wrong answer | right answer | 1 |

Those 94 plus the 1 make the 95 that flipped to right. The 61 that stayed
wrong break down as 42 plain wrong answers plus 19 refusals worded differently,
which containment also scores as wrong. Read the other way round, keeping the
hatch sentence costs you 94 right answers in order to avoid 61 wrong ones. It
is a dial. Nobody tuning prompts from vibes knows the dial exists. Measure what the sentence does on
your own questions before you ship it. This ablation ran for Glimmer only. Sources:
[reports/glimmer-day-one.md](reports/glimmer-day-one.md) and
[reports/confound-closers.md](reports/confound-closers.md), rows in
`reports/stage10_hatch.jsonl`.

### The same two models in five languages

Thai, Japanese, Chinese, Spanish and Vietnamese, each with its own native
question set of 100 questions. Two models, two arms each: 2,000 model calls,
none failed. Glimmer with documents tied or beat the 8B in every language:
+6.0 in Thai (could be luck, p 0.286), 0.0 in Japanese, +13.0 in Chinese
(p 0.001), +16.0 in Spanish (p 0.0015) and +17.0 in Vietnamese (p 0.0005).
Closed-book collapsed in all five, so retrieval lifted Glimmer by +54 to +79
points. The same escape hatch became a good detector there. Glimmer abstained
far more often when search had missed the right document (50 to 100% of
misses) than when search had found it (1 to 8%). One caution. Compare these
numbers against the English results only for whether the effect points the
same way, not for how large it is. These questions each need one document and
have short answers, while the English ones needed two or more documents and
were mostly yes or no. Thai, Japanese and
Chinese are written without spaces. BM25 needed a word splitter for each
(newmm, janome, jieba), or it would have scored near zero. Source:
[reports/five-language-replication.md](reports/five-language-replication.md).

### Two headlines the controls caught

I retracted two claims in writing rather than deleting them. Stage 1 first
reported that merging keyword and dense results into one list scored worse
than keyword search alone. A bug explained it: the two lists had been merged
at different depths, 50 results from BM25 against 10 from dense. After the
fix the finding shrank to "does not measurably help"
([reports/retrieval.md](reports/retrieval.md)).

Stage 3 said the 8B could not combine evidence from two documents on the
yes/no questions. If that were true, a bigger model should do better. Stage 6
ran the 27-billion-parameter `qwen3.6:27b` on the same retrieved text. It
abstained on 84% of those questions and lost to the 8B. A simpler problem
showed up next: on the comparison questions a constant "Yes" scores 60.0%,
above every model arm, so those questions cannot tell models apart at all.
Glimmer's 96.67% closed-book on the inference questions showed the answers
were already in its training data, the text it learned from before anyone
asked it anything. Source:
[reports/claim-stress-tests.md](reports/claim-stress-tests.md). What to do:
run a constant-answer baseline and a closed-book arm before believing any RAG
number.

## How the study is built

![The study's pipeline: eight corpora go through one chunker, then three retrieval arms (BM25, dense, graph), then paired generation arms per model, then containment scoring and paired statistics, with the control arms along the bottom](docs/architecture.svg)

Documents are cut into chunks of at most 600 or 1,200 tokens, aligned to
sentence boundaries. Each retrieval arm ranks the same chunks. Each
generation arm gets the same prompt shape and the same retrieved text, with
the model's randomness switched off (temperature 0 and a fixed seed), so the
same input gives the same answer on a rerun. Every generated answer is
appended to a jsonl file as it lands, so a crashed run resumes by rerunning
the same command.

Repo map:

- `src/` shared code: the chunker, BM25, the embedding index, graph assembly
  and the two retrieval scorers.
- `ops/` the stage scripts, the shell drivers that sequence them, and the
  probe scripts that each produce one number quoted in a report
  (`ops/verify_bm25.py`, `ops/chunker_integrity.py` and their neighbours).
- `reports/` the per-question rows (`*.jsonl`), the summaries (`*.json`) and
  one narrative per stage (`*.md`).
- `data/` manifests with content hashes (a hash is a fingerprint of a file's
  bytes), plus the committed caches: the embedding files, the five language
  corpora and question files, and the assembled graph
  `data/graph_600.build.json`. The source tables and chunk files are
  regenerated, not committed.
- `examples/` the notebook and its 15-document sample.
- `scripts/` the CI verifier.
- `STATUS.md` the ledger of every stage's verdict and artifact. The verifier
  locks its numbers.

### Reports, stage by stage

Stage 5 is folded into stage 6, so the numbers skip.

- [Stage 0: machine measurement](reports/machine-measurement.md). Could the
  machine run the study as first specified? No. Extraction hit its length cap
  on 4.8% of calls, and hits@10 could not tell the search methods apart, so
  the main metric changed to strict@k before stage 1.
- [Stage 1, retrieval layer: BM25 / dense / hybrid](reports/retrieval.md).
  Which search method finds all the evidence? BM25, in every cell. Hybrid
  means BM25 and dense results merged into one list, and the merge finding was
  retracted inline.
- [Stage 2: the graph arm](reports/graph-arm.md). Does a knowledge graph
  beat keyword search? No, by 24 strict@10 points.
- [Stage 3: generation arms](reports/generation-arms.md). Does retrieval
  improve answers? Yes, plain 63.1% vs A0 39.5% at n=1,381.
- [Stage 4: the learning layer (pre-registered negative)](reports/learning-layer.md).
  Does feedback re-ranking help? No, 1.40 strict@10 points worse on similar
  questions.
- [Stage 6: pushing on the claims](reports/claim-stress-tests.md). Which
  findings survive a harder test? The BM25 win held on the full corpus
  (`reports/stage5_full_corpus.json`). The claim that the 8B could not
  combine evidence broke.
- [Stage 7: the cross-domain replication (HotpotQA)](reports/hotpotqa-transfer.md).
  Does the BM25 win transfer to Wikipedia? No, dense wins there. The strict@k
  lesson does transfer.
- [Stage 8: Muse Glimmer 30B, day one](reports/glimmer-day-one.md). How does
  a brand-new 30B model do? It lost by abstaining, and it had memorised the
  news.
- [Stage 9: the five-language replication](reports/five-language-replication.md).
  Do the English findings hold in five other languages? The verdict
  inverted. Glimmer tied or won everywhere.
- [Stage 10: the three confound-closers](reports/confound-closers.md). Three
  follow-ups, one per open question. The hatch caused the Glimmer loss. The
  hatch stopped hurting in the five languages because of the question type,
  not the language. Whether Glimmer's memorised news came from its size or
  from being newer could not be separated.

## Regenerating from scratch

Large regenerable files are not committed. Manifests carry their content
hashes, so a regenerated file can be checked against what the study used.

1. `python ops/fetch_data.py` downloads the source tables (parquet files, a
   compact format for storing tables) from Hugging Face, a public host for
   datasets and models, and writes `data/sources.manifest.json`.
2. `python src/chunker.py --budget 600` cuts them into chunk files with a
   hash manifest. Run again with `--budget 1200` for the second chunking.
3. `python src/embed_index.py --budget 600` builds the dense index. It needs
   ollama running with `qwen3-embedding:0.6b` pulled.
4. The stage scripts under `ops/`, in the order below. Each is checkpointed
   and resumable.

Stage order:

- stage 0: `ops/stage0.py`
- stage 1: `src/score_retrieval.py`
- stage 2: `ops/graph_index.py`, then `src/build_graph.py`, then
  `src/score_graph.py`. The index step is the long one, because the model
  reads every chunk once to write out its names and links.
- stage 3: `ops/stage3_pilot.py`. Its `--n` flag sets how many questions to
  run. The full run was 1,381.
- stage 4: `ops/stage4_learning.py`
- stage 6: `ops/stage6_27b.py`
- stage 8: `ops/stage8_glimmer.py`
- stage 9: `ops/stage9_lang.py` once per language, with
  `ops/run_multiling.sh` as the driver.
- stage 10: `ops/escape_hatch_ablation.py` deletes the hatch sentence and
  reruns Glimmer RAG, `ops/prep_en_squad.py` builds the English single-hop
  control corpus, and `ops/recency_vs_scale_27b.py` runs the 27B on the 60
  inference questions. `ops/stage10_driver.sh` sequences them.

Four of the run scripts take `--analyze` to write their summary JSON from
the rows: `stage8_glimmer.py`, `stage9_lang.py`, `escape_hatch_ablation.py`
and `recency_vs_scale_27b.py`. `stage4_learning.py` writes its summary at the
end of its run. `stage3_pilot.py` and `stage6_27b.py` write rows only.

The Glimmer arms need llama.cpp build b10353, the second local model runner,
serving the model on port 8095. The header of `ops/run_multiling.sh` has the
launch line. `models.manifest.json` records the start-up options plus the
download link and checksum of the model file, which is about 17 GB. Model
binaries are not committed.

Known gaps, four of them.

- The stage 9 language corpora are committed but the script that prepared
  them is not. See the provenance caveat (provenance means where the data
  came from) in
  [reports/five-language-replication.md](reports/five-language-replication.md).
- The stage 7 HotpotQA run was done by hand without a saved script, so only
  its result files landed. See the provenance debt note in
  [reports/hotpotqa-transfer.md](reports/hotpotqa-transfer.md).
- The graph index's raw extraction checkpoint `data/graph_600.jsonl` is not
  committed, because `.gitignore` excludes `data/*.jsonl`. `STATUS.md` has
  the three-step recipe to rebuild it.
- `reports/stage3_final.json` and `reports/stage6_27b.json`, the summaries
  behind the stage 3 and stage 6 headlines, have no committed script that
  writes them. Their per-question rows are committed, and the verifier reads
  the summaries as they stand.

## Limits

- Each architecture verdict comes from one corpus. The graph and learning
  results are from the news corpus only.
- Containment is a substring check. A correct answer phrased without the
  gold string scores wrong, and so does a free-form refusal.
- The escape-hatch ablation ran for Glimmer only. The 8B abstained on 30.9%
  of RAG questions in stage 3 and has no ablation arm.
- `qwen3.6:27b` is both bigger and newer than `qwen3:8b`, so the 27B probe
  cannot separate model size from recency.
- Every primary arm ran on one machine at temperature 0 with a fixed seed.
  One stage 8 sensitivity run used random sampling instead and reached the
  same verdict ([reports/glimmer-day-one.md](reports/glimmer-day-one.md)).
- The news benchmark's comparison and temporal questions are mostly yes/no,
  so a constant answer scores well on them. The graph verdict and the
  retracted evidence-combining claim both sit on that corpus.

## What is not built

This repo is the measurement study. The product it was meant to feed has no
code in `src/` or `ops/`: no Postgres with pgvector (a database with vector
search built in), no retrieval router, no feedback tuning loop. It is
specified in [PRD-TRACK-2-PRODUCT.md](PRD-TRACK-2-PRODUCT.md), which opens
with "Status: not started". That plan names
[agent-report-card](https://github.com/netsatsawat/agent-report-card), my
evaluation harness, as its scorer. There is no feedback loop that improves
the system here, because stage 4 tested that mechanism and it did not work.

---

Written by [Satsawat Natakarnkitkul](https://satsawat.ai), a data & AI leader in
ASEAN, author of *Why Your AI Agent Will Fail*. Companion articles are in
draft and will appear at [satsawat.ai](https://satsawat.ai) · Newsletter:
[AI in Practice](https://satsawat.ai/#newsletter)

License: MIT
