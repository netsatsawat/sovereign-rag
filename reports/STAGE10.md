# Stage 10 — the three confound-closers (2026-08-13)

Stages 8–9 left three named holes: whether the escape hatch *caused* the
English loss or merely labelled it; whether task shape or language drove
the calibration reversal; and whether the closed-book memorization floor
was recency or scale. One arm each. All three came back decisive.

## 1. Escape-hatch ablation: causal, and then some

The stage-8 240-query sample, identical BM25@600 k=10 contexts, identical
generator configuration, one change: the sentence "If the context does not
contain the answer, reply exactly: insufficient information." deleted from
PROMPT_RAG (`ops/stage10_hatch.py` hard-fails if the sentence drifts).

| | containment | abstention |
|---|---|---|
| plain_glimmer (hatch, stage 8) | 32.92 | 64.6 |
| plain_glimmer_nohatch | **72.50** | **0.0** |

Paired: +39.58pp, discordant 95:0 — no query got worse — p = 5.0e-29.
Fates of the 155 hatch-arm abstentions: **94 recovered** (became correct),
**42 committed wrong**, **19 free-form refusals** ("cannot be determined
from the provided context" and kin) that containment scores as wrong, 0
emit the prescribed phrase. Composition of the 94: 27 of 29 refused
open-ended entity questions recovered (hard to fake), while the 126
refused yes/no items recovered at 67/126 = 53.2% — coin-flip territory,
so part of the binary recovery is chance, not released knowledge.

Readings, both true:
- The hatch sentence caused the loss. Without it, Glimmer (72.5) beats the
  8B's 48.75 on the same 240 by ~24pp — the model that lost the stage-8
  headline was one prompt sentence away from winning it.
- The hatch bought 61 prevented hallucinations at the price of 94
  suppressed correct answers. On this benchmark that trade is ruinous; in
  a compliance setting it may be the right one. Either way it is a
  *dial*, not a defect — and nobody tuning prompts from vibes knows the
  dial exists.

Glimmer-specific caveat: this dose-response is measured for Glimmer only;
the 8B's hatch sensitivity (its abstention is 43.8% overall) has no
ablation arm.

## 2. English single-hop control: task shape, not language

SQuAD v1.1 dev (`ops/prep_en_squad.py`, provenance in
`data/multiling/sources.manifest.json`): same language as stage 8, same
task shape as stage 9. 2,067-paragraph corpus, 100 queries, both models,
closed + RAG@5.

| arm | containment | abstention |
|---|---|---|
| closed glimmer | 23.0 | 41.0 |
| closed qwen8b | 19.0 | 16.0 |
| RAG glimmer | 82.0 | 11.0 |
| RAG qwen8b | 82.0 | 5.0 |

RAG glimmer vs 8B: 0.0pp (p=1.0) — the exact tie again, as in Japanese.
RAG-over-closed: +59/+63pp, both p < 0.0001. Calibration cross for
glimmer (reports/stage9_en_calibration.json): abstain 4.4% when the gold
was retrieved (n=91) vs 77.8% when missed (n=9).

Verdict: **the calibration reversal follows task shape into English.**
Same language and near-identical prompt wording as the arm that refused
56% of evidence-complete queries; single-hop extractive shape; calibrated
behaviour. Language and corpus are exonerated as primary drivers.

Bonus datapoint: SQuAD closed-book is LOW (23/19) despite being the
most-published QA set alive — parametric answerability, not publication,
is what the closed-book arm measures. Retrieval note: BM25 79 vs dense 73
at hits@1, but dense wins at hits@10 (96 vs 92) — the first depth
crossover in the study.

## 3. 27B entity probe: scale, not recency

The 60 inference qi from the stage-8 sample, closed + RAG, qwen3.6:27b in
the stage-6 configuration.

| closed-book on the entity stratum | containment |
|---|---|
| qwen3:8b | 51.67 |
| qwen3.6:27b | **88.33** |
| glimmer 30B | 96.67 |

27B vs glimmer paired: −8.33pp, 6:1 discordant, p = 0.125 (underpowered
null, not equivalence). 27B vs 8B: +36.67pp, p = 3.0e-06.

Correction to this stage's original design gloss: qwen3.6:27b is NOT the
same generation as qwen3:8b — it is a newer Qwen line (model_family
qwen35 in the ollama config vs qwen3), contemporaneous with Glimmer. The
27B is bigger AND fresher than the 8B, so this arm cannot attribute the
floor rise to scale alone. What it does establish: two large models from
two labs share the high floor (not a Glimmer quirk), and the
grounding-suppression effect travels (RAG drops the 27B too, 88.3 → 80.0
with 18.3% abstention). The scale-vs-recency separator — a same-family
scale step, e.g. qwen3:32b on the same 60 qi — is on the follow-up list.

## Artifacts

`ops/stage10_hatch.py`, `ops/prep_en_squad.py` (+ en in
`ops/stage9_lang.py`), `ops/stage10_27b_inf.py`, `ops/stage10_driver.sh`;
rows in `reports/stage10_hatch.jsonl`, `reports/stage9_en.jsonl`,
`reports/stage10_27b_inf.jsonl`; summaries alongside as .json;
`reports/stage9_en_calibration.json`; driver log
`reports/stage10_driver.log`. All p-values full precision per the house
rule.
