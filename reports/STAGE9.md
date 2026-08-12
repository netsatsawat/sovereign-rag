# Stage 9 — the five-language replication (2026-08-12)

Muse Glimmer's card claims 100+ languages. Stage 8 measured it in English
only, and its two sharpest findings — the escape hatch firing on
evidence-complete queries, and the contamination floor — could each be an
English artifact. This stage asks the same paired questions in Thai,
Japanese, Chinese, Spanish and Vietnamese on native extractive-QA corpora,
with qwen3:8b as the paired comparator: 100 queries × {closed, RAG@5} ×
{glimmer, qwen8b} per language, 2,000 generation rows, zero errors.

## Datasets (all native; none translated)

| lang | dataset | corpus | eval split |
|---|---|---|---|
| th | iapp wiki QA | 1,912 articles | test[::7][:100] |
| ja | JSQuAD v1.3 (JGLUE) | 2,500 paragraphs | test[::44][:100] |
| zh | CMRC 2018 | 3,000 articles | dev[::32][:100] |
| es | SQAC | 2,000 paragraphs | test[::62][:100] |
| vi | UIT-ViQuAD 2.0 | 2,500 articles | validation (answerable)[::26][:100] |

## Retrieval: segmented BM25 sweeps, the embedder's gap is largest where it
matters most

hits@1 over 100 single-gold queries (strict@k = hits@k here):

| lang | BM25 (lang-segmented) | dense qwen3-emb 0.6b | gap |
|---|---|---|---|
| th (newmm) | 90 | 86 | +4 |
| ja (janome) | 87 | 72 | +15 |
| zh (jieba) | 95 | 91 | +4 |
| es (regex) | 69 | 64 | +5 |
| vi (regex) | 72 | 56 | +16 |

The BM25 wins are conditional on proper segmentation: the study's shared
`normalize()` deletes non-Latin text (and Spanish accents) outright, so an
unsegmented BM25 arm would have silently scored near zero in th/ja/zh and
handed the win to the embedder for the wrong reason.

## Generation: the English verdict inverts

Containment % (abstention %) on the RAG arm, and paired glimmer-vs-8B:

| lang | glimmer RAG | qwen8b RAG | delta pp | p |
|---|---|---|---|---|
| th | 67.0 (10.0) | 61.0 (4.0) | +6.0 | 0.286 |
| ja | 87.0 (9.0) | 87.0 (2.0) | 0.0 | 1.0 |
| zh | 87.0 (2.0) | 74.0 (1.0) | **+13.0** | 0.001 |
| es | 61.0 (19.0) | 45.0 (10.0) | **+16.0** | 0.0015 |
| vi | 70.0 (18.0) | 53.0 (9.0) | **+17.0** | 0.0005 |

English (stage 8, multi-hop news): glimmer −15.83pp vs the same 8B,
p<0.001. Outside English, Glimmer ties or wins everywhere, significantly in
three of five.

Closed-book collapses in every language (glimmer 7–27, 8B 3–12 — single
digits everywhere except Japanese's 27/12): no contamination floor — these
corpora test knowledge the models do not have, so RAG-over-closed-book is
+54 to +79pp for glimmer (all p<0.001). The English "+6.25pp, p=0.067,
retrieval useless" result was the contaminated entity stratum talking, not
the architecture.

## The calibration reversal

Glimmer's RAG abstention crossed with whether BM25@5 actually retrieved the
gold article:

| lang | abstain \| gold retrieved | abstain \| gold missed |
|---|---|---|
| th | 8% (n=96) | 50% (n=4) |
| ja | 6% (n=95) | 60% (n=5) |
| zh | 1% (n=99) | 100% (n=1) |
| es | 7% (n=82) | 72% (n=18) |
| vi | 7% (n=86) | 86% (n=14) |

The 8B's same cross (recomputed from the same rows + deterministic index
rebuild): found → 2/1/0/4/2%, missed → 50/20/100/39/50% (th/ja/zh/es/vi).
It abstains less everywhere — including on the misses, where abstention is
correct: 20 vs 60 (ja), 39 vs 72 (es), 50 vs 86 (vi). Glimmer's hatch
discriminates better than the model that beat it in English.

In English the same model's abstention was near-chance as an
evidence-completeness detector (precision 0.63 / recall 0.71 vs a 58% base
rate). On single-hop extractive tasks it is a good detector in all five
languages. The Stage-8 over-refusal is therefore not a fixed trait of the
model: it is what strict grounding + multi-hop yes/no questions + a
containment metric jointly produce.

## Confounds, stated

- Task shape differs from Stage 8 (single-hop extractive vs multi-hop
  news; mostly entity/span answers vs 75% yes/no). Cross-language rows are
  comparable to each other; comparisons to Stage 8 are qualitative.
- Not cross-language paired: each language has its own 100 questions and
  its own corpus difficulty (es/vi retrieval is notably harder).
- k=5 (not 10) and 2,500-char contexts, for the 8k window under CJK/Thai
  token inflation. The truncation has a measured cost: th qi=51 and qi=88
  lose their gold span to the trim (verified against the committed
  corpus), so both models' Thai RAG ceiling is 98/100; paired deltas
  unaffected.
- Reasoning tax persists everywhere: glimmer mean gen tokens 161–371 per
  answer depending on language and arm (RAG arms 161–276; lowest zh,
  highest th) vs the 8B's actual means of 7–17 under its 64-token cap.

## Reading the committed numbers

- `mcnemar_p` is rounded to 4 dp, so values below 5e-5 appear as `0.0` —
  a value an exact binomial test cannot produce. All ten
  `*_rag_over_closed` pairings are in this bucket (true values
  1e-15..3e-24). Quote them as p < 0.0001, never "p = 0.0".
- Containment runs on separator-stripped text, so a purely numeric gold
  can match inside a longer digit run. Exactly one committed row benefits:
  ja closed-book 8B (qi=96, gold '4' matched inside '574族'), i.e. its
  12.0% would be 11.0% with digit boundaries.
- The same strip removes all combining marks, which for Thai deletes
  vowels-above/below and tone marks from gold and answer alike — lenient
  (one committed th RAG match rides on a dropped silent-letter mark) and
  documented in `ops/stage9_lang.py`.
- `abstained_pct` counts the PRESCRIBED escape phrase (plus the English
  fallback), not refusal in general: three committed th rows refuse
  free-form ('ไม่ทราบ') and are not counted. Read it as "answered with the
  prescribed escape phrase", not "refused".
- The per-arm summaries do not carry stage 8's `truncated_pct`, and five
  glimmer closed-book rows hit the 1,024-token budget with the ENTIRE
  budget consumed by reasoning (`done_reason: "length"`, gen_tokens 1024,
  thinking 1,867–3,375 chars, answer ""): th qi=15,36 and ja qi=4,6,37.
  They score as non-contained, non-abstained misses, indistinguishable in
  the summary JSON from a confidently wrong answer — i.e. 2pp of th and
  3pp of ja closed_glimmer's containment/abstention denominators are "the
  answer channel never emitted a token". Footnote them when quoting th/ja
  closed-book numbers. th closed_qwen8b additionally has six
  `done_reason: "length"` rows at its 64-token cap, all with non-empty
  truncated answers.

## Artifacts

`ops/stage9_lang.py` (retrieval/generation/analysis — NOT prep; see the
provenance caveat below), `ops/run_multiling.sh` (driver; generation +
analysis only), `data/multiling/{lang}/` (corpus + queries, committed),
`reports/stage9_{lang}.jsonl` (400 rows each),
`reports/stage9_{lang}.json`, `reports/stage9_{lang}_retrieval.json`,
`reports/stage9_driver.log`. Generator/embedder provenance (GGUF sha256,
ollama digests, ollama version) is in `models.manifest.json`. The
calibration cross: `python ops/stage9_lang.py --lang $L --calibration`
recomputes the reversal table from the jsonl rows plus a deterministic
LangBM25 rebuild (no server needed), writing
`reports/stage9_{lang}_calibration.json`; the table above rounds its
percentages to integers.

**Provenance caveat.** The corpora and queries are committed, so every
downstream number recomputes from the repo — but the script that built
`data/multiling/{lang}/corpus.jsonl` + `queries.jsonl` from the source
datasets was not committed, and the upstream revision each language was
pulled at was not recorded (contrast `ops/fetch_data.py` +
`data/sources.manifest.json` for the English corpus).
`data/multiling/sources.manifest.json` records what is verifiable today:
per language, the upstream dataset's canonical location (corroborated by
the id formats inside the committed files), the split and sampling
expression from the table above, and sha256 hashes of the committed
corpus/queries files. Until a prep script in the fetch_data.py pattern
lands, treat the committed corpora as the ground truth of what was
measured, not as verifiable derivations of the upstream datasets.
Dense-retrieval caches are also unevenly committed (ja/th `emb.npy`
tracked, es/vi/zh local-only). The tracked caches ship with the
`emb.meta.json` corpus fingerprint that `--retrieval`'s stale-cache guard
requires; the caches predate the guard, so the fingerprints were
back-filled from the committed corpora — sound because each `corpus.jsonl`
is unchanged in git since the commit that added its `emb.npy`, and the
committed hits@k were computed from exactly these cache/corpus pairs.
Deleting an `emb.npy` and re-running `--retrieval` regenerates it
deterministically via ollama either way.

## Reproduction

No committed script starts the servers. Prerequisites:

    # glimmer half: llama-server resident on :8095 (the stage-8 launch line)
    models/llama-b10353/llama-server -m models/muse-glimmer-30B-kquant-17gb.gguf \
        --port 8095 -c 8192 --jinja --reasoning off --reasoning-format deepseek

    # qwen half + dense retrieval: ollama on :11434
    ollama pull qwen3:8b && ollama pull qwen3-embedding:0.6b

Per language: `--retrieval` (not run by the driver), the two `--model`
arms (or `ops/run_multiling.sh` for all five), then `--analyze` — the
command sequence in `ops/stage9_lang.py`'s docstring (`--calibration`
there is optional and derived; it needs no server).
