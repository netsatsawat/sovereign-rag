# Stage 8 — Muse Glimmer 30B, day one (2026-08-11)

Meta released Muse Glimmer 2026-08-10: 30B, Apache 2.0, "open agentic model
that runs on your device," benchmarked by Meta against the same Qwen3.6-27B
this study already holds paired rows for. The harness gives the comparison
for free: byte-identical prompts, contexts and scoring against the committed
qwen3:8b rows (all 1,381 queries, Stage 3) and qwen3.6:27b rows (180
comparison/temporal queries, Stage 6).

## What was actually runnable on a 24 GB Apple-silicon machine

| attempt | result |
|---|---|
| `ollama pull muse-glimmer:30b-mlx` (0.32.5) | 412: requires newer Ollama |
| same, after upgrade to 0.32.7 | model loads only after `iogpu.wired_limit_mb` raised 17.8→21 GiB |
| MLX build, loaded | ~40 s/token; one 64-token call = 52m32s wall; response empty (template parser ate all 64 tokens) |
| `ollama pull muse-glimmer:30b-q4_K_M` (GGUF) | 412: requires a pre-release runtime |
| official `muse-glimmer-30B-kquant-17gb.gguf` @ llama.cpp b10353 | loads in 13 s, prefill ~73 tok/s (95-token prompt) to ~170 tok/s (5k), generation 7.5 tok/s — the run |

The MLX empty-response is explained below: the model has no non-thinking
mode, and 64 tokens of budget were consumed entirely by reasoning.

## Protocol

Sample: 240 queries — the exact Stage-6 deterministic sample (120 comparison
+ 60 temporal; asserted bit-identical to the committed stage6 qi set) plus 60
evenly-spaced inference queries. Arms: closed-book and BM25@600 k=10 RAG,
PROMPT_CLOSED/PROMPT_RAG byte-identical to Stages 3/6. Scoring: the same
normalised containment; abstention = "insufficient information" in the
normalised answer. temperature 0, seed 0. 480/480 calls completed, zero
errors, zero truncations.

Two deviations, both forced by the model and both documented in
`ops/stage8_glimmer.py`:

- **`Reasoning strength: low` system message.** `--reasoning off` is a no-op
  for this template; the model reasons unconditionally (probe: all 64 tokens
  to `reasoning_content`, content empty). "low" is Meta's documented control
  and halves the spend (~270 vs 664 gen tokens on the probe query).

  Probe record (llama-server /v1/chat/completions, PROMPT_CLOSED with
  "What is the capital of France?", temperature 0, max_tokens 64, server
  launched with `--reasoning off`): `finish_reason: "length"`,
  `content: ""`, `reasoning_content` ending `...So just "Paris". Probably
  just Paris.\n\nNo` — 64/64 tokens spent deliberating, zero on the answer
  channel.
- **max_tokens 1024 vs the others' 64**, because reasoning and answer share
  the budget. Only the answer channel is scored, as in every other arm.

A sensitivity armset at Meta's recommended sampling (T=1.0, top-p 0.95,
top-k 64) ran as `--armset rec`, complete at 480/480. Verdict: sampling is
not the story. RAG containment 31.67 (primary 32.92; paired delta −1.25pp,
p=0.68), abstention 66.2% vs 64.6%, closed-book 26.25 vs 26.67. Against the
8B the rec arm loses −17.08pp (p<0.001); RAG-over-closed-book at rec is
+5.42pp, p=0.11 — still no significant retrieval benefit.

## Results (primary, temperature 0)

Containment % (abstention %) by arm and stratum:

| | overall | comparison (n=120) | inference (n=60) | temporal (n=60) |
|---|---|---|---|---|
| closed-book | 26.67 (72.9) | 0.83 (99.2) | **96.67 (1.7)** | 8.33 (91.7) |
| RAG BM25@600 k=10 | 32.92 (64.6) | 29.17 (65.8) | 68.33 (31.7) | 5.00 (95.0) |

Paired (exact McNemar):

| comparison | n | delta pp | p |
|---|---|---|---|
| glimmer RAG vs 8B RAG | 240 | **−15.83** | <0.001 |
| glimmer closed vs 8B closed | 240 | −2.08 | 0.625 |
| glimmer RAG vs 27B RAG | 180 | +6.11 | 0.052 |
| glimmer closed vs 27B closed | 180 | −12.78 | <0.001 |
| glimmer RAG over closed-book | 240 | +6.25 | 0.067 |

Always-Yes on the comparison stratum: 60.0 (this sample) — roughly double
Glimmer's best comparison arm.

## The three findings

**1. The loss travels through compliance.** Glimmer abstains on 64.6% of
RAG queries — between the 8B and the 27B's 84% — and crossing abstention
with the Stage-1 strict@10 per-query vectors shows it is not calibration:
**56.4% abstention on queries where every gold document is in the context**
(vs 70.5% where evidence is incomplete). As a detector of genuinely-missing
evidence, its abstention has precision 0.63 / recall 0.71 against a 58%
base rate. Anchors on the identical evidence-complete rows: the 8B abstains
38.6%, the 27B 74.7%. When Glimmer does answer with complete evidence it is
right 90.9% of the time (40/44) — the loss is delivered through the escape
hatch, not through wrong answers. Note the cross-model ordering (27B: lower
IFBench, more abstention) blocks any claim that IF-training *causes* the
over-refusal; what is measured is that the failure expresses itself as
literal compliance with the escape-hatch instruction.

**2. Contamination has a floor and it is rising.** Closed-book, no context,
Glimmer scores 96.67% on the inference stratum. The 8B on the same 60
questions: 51.67 (recomputed from stage3_pilot.jsonl; 57.68 is the full
671-question stratum figure) — a 45pp rise in the memorization floor in one
model generation. Near-verbatim recall of late-2023 news entities (knowledge
cutoff 2026-01-04); for practical purposes the corpus is in the training
set. Retrieval *drops* it to 68.33% because grounding suppresses what it
already knows. Net: RAG-over-closed-book is +6.25pp, p=0.067 — **no
statistically significant retrieval benefit for this model on this
corpus**.

**3. The reasoning tax is unconditional.** No off switch exists. At the
lowest documented strength: mean 349 gen tokens per RAG answer (median
331.5, max 1020). The Qwens' actual means on the same questions under their
64-token cap: 7.0 (8B) and 5.7 (27B) — roughly 50× the output tokens per
answer.

## Artifacts

`ops/stage8_glimmer.py` (run + analysis), `reports/stage8_glimmer.jsonl`
(480 rows), `reports/stage8_glimmer.json` (summary), `reports/stage8.log`,
`reports/llama_server.log`. Model: `models/muse-glimmer-30B-kquant-17gb.gguf`
(models/ gitignored; sha256, size and runtime pin recorded in
`models.manifest.json`, alongside the ollama digests for qwen3:8b and
qwen3-embedding:0.6b — the official download URL is still to be added
there), server llama.cpp b10353 (f8def7fe1). Calibration cross: `python
ops/stage8_glimmer.py --calibration` recomputes finding 1's numbers (the
56.4/70.5 split, precision 0.63 / recall 0.71, the 38.6/74.7 anchors,
40/44) and finding 2's paired-60 8B closed-book 51.67 from
stage8_glimmer.jsonl + stage1_retrieval.json
per_query.600_k10.bm25.strict + stage3_pilot.jsonl + stage6_27b.jsonl,
writing `reports/stage8_calibration.json`.

Reading the committed summary: `mcnemar_p` is rounded to 4 dp, so values
below 5e-5 appear as `0.0` (three pairings here). An exact test never
yields 0 — quote those as p < 0.0001.
