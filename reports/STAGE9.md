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

Closed-book is single digits in every language (glimmer 7–27, 8B 3–12): no
contamination floor — these corpora test knowledge the models do not have,
so RAG-over-closed-book is +54 to +79pp for glimmer (all p<0.001). The
English "+6.25pp, p=0.067, retrieval useless" result was the contaminated
corpus talking, not the architecture.

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
  token inflation.
- Reasoning tax persists everywhere: glimmer mean gen tokens 238–371 per
  answer vs the 8B's ≤64 cap.

## Artifacts

`ops/stage9_lang.py` (prep/retrieval/generation/analysis),
`ops/run_multiling.sh` (driver), `data/multiling/{lang}/` (corpus + queries,
committed), `reports/stage9_{lang}.jsonl` (400 rows each),
`reports/stage9_{lang}.json`, `reports/stage9_{lang}_retrieval.json`,
`reports/stage9_driver.log`. The calibration cross recomputes from the
jsonl rows plus a rebuild of each LangBM25 index (deterministic).
