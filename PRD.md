# sovereign-rag — PRD

## TRACK 1 — the public artifact

**This document builds the study, the article and the repo. It does not build a product.**

There are two tracks. Track 1 is this: a measured comparison of three retrieval architectures,
fully self-hosted, published as an article and a repo anyone can clone and reproduce. Its job is to
be *believed*, not to acquire users. Success is an editor reply, a vendor citation, or engineers
arguing with the numbers. Judged as revenue it is a bad investment; judged as credibility and
lead-generation for the consulting and writing lanes, it is the strongest asset available.

Track 2 — the BYOC product this could become — is specified separately in
[`PRD-TRACK-2-PRODUCT.md`](PRD-TRACK-2-PRODUCT.md). **Nothing in Track 2 is built here.** Track 1
carries exactly one obligation toward it, stated in §14: the arms sit behind a stable
`Arm.answer() -> Envelope` contract with an OpenAI-compatible adapter over it. That boundary costs
Track 1 nothing — it is also the cheapest way to get an interface — and without it none of this
work ports.

Anything else that smells like product in this document is a mistake and should be cut.

---

Produced 2026-08-08 by two workflows: 10 agents on the study (5 research briefs — corpora,
architectures, local embedders, self-learning mechanisms, evaluation fairness — a design pass, and
three adversarial critiques on rigging risk, 24 GB feasibility, and honesty of the learning claim),
then 9 agents on the interaction layer (Open WebUI internals verified from source, a 13-tool
alternatives scan, production requirements, feedback UX, three competing strategies, and two
adversarial critiques on scope and on dilution).

Hardware constants are measured from this machine and from committed artifacts in
agent-report-card, not estimated: Apple M5, 24 GB unified, qwen3.6:27b resident at 17.4 GB,
7.79 s per short-prompt LLM call.

---

## 1. One-paragraph summary

Three retrieval architectures — plain hybrid RAG, agentic RAG, and GraphRAG — are run over one shared 253-article news corpus on one 24 GB M5 MacBook Pro with no API key, no hosted call, and no second machine. Every question is answered by every arm (fully paired design), and results are reported **per question type**, never as a single aggregate. On top of that grid, a retrieval-feedback re-ranking layer is applied identically to all three architectures and evaluated separately on repeated queries and on genuinely novel ones. The answer is not a leaderboard. It is a crossover: a curve showing what share of a deployment's traffic must be aggregative or temporal-multi-hop before a 16-hour graph index and a 4-LLM-call agent loop beat a one-line BM25 baseline. The predicted shape, pre-registered: BM25 wins single-fact lookup outright, single-fact lookup is most real traffic, and the graph never repays its index until aggregative traffic clears a low-double-digit share — at which point plain RAG at k=5 does not merely lose, it cannot answer at all.

---

## 2. The headline question

> **At what question mix does each architecture start paying for itself?**

Three subordinate conditions, each stated the same way:

- **YOUR STRATUM DECIDES YOUR WINNER, and the spread is asymmetric.** Published prior art on this same corpus: GraphRAG beats plain RAG by **+19.9 points** on temporal, **+3.0** on comparison, **−0.25** on inference. An aggregate over an unstated mix can be made to say anything. A single unlabelled number is refused.
- **WHICH COMPONENT OF "AGENTIC RAG" YOU MEAN DECIDES WHETHER IT HELPS.** *Dissecting Agentic RAG* (arXiv:2606.21553, Table 2): single-pass dense 43.1 EM → agentic with adaptive routing 53.2 → **agentic with fixed hybrid retrieval 55.0**. `hybrid-only` is the full agentic pipeline with the router swapped out, not the agent removed. The scaffolding buys **+11.9 EM**; the *cleverest* component in it, the router, **loses 1.8 EM**. Dropping decomposition costs 1.4 EM and halves latency. Quoting this as "retrieval beat the agent" is wrong and is the first thing a reviewer will break.
- **YOUR QUERY REPEAT RATE DECIDES WHETHER "SELF-LEARNING" MEANS ANYTHING.** Above it, a semantic cache with no intelligence wins every metric users feel. Below it, only mechanisms that generalise matter — and on this corpus a retriever that **ignores the query entirely** and ranks documents by how often they were gold in the training half scores **Hit@10 = 65.1% [measured]**. Any learning curve must be shown to beat that.

---

## 3. Non-goals — stated first and bluntly

1. **No hosted call anywhere.** If a component requires an API key, it is disqualified, not caveated. This kills Jina v5 embeddings (CC-BY-NC), Microsoft GraphRAG's default provider path, and every LLM-judge service.
2. **No leaderboard, no single headline accuracy number.** Per-stratum always, plus exactly one explicitly-labelled weighted aggregate under a stated production mix, plus the weight slider that shows the winner changing.
3. **No LLM judge. [changed:** the design specified a judge on the global stratum plus a 150-pair calibration. Removed. If global-stratum gold is computed by pandas and by an independent NER pass, it is exactly scorable, and judging exact gold adds attenuation noise — observed ≈ κ × true — to the *one* stratum where the graph is predicted to win. The design was measuring the predicted plain-RAG win with a micrometer and the predicted graph win with a rubber band. It also removes ~2.5 h, the calibration exam, and the swap-consistency machinery.**]** `agent-report-card` is reused for the harness, exit codes 0/1/2, and CI number-assertion — not as a judge.
4. **No write-back experiment.** The contamination replay already measured it. It is **cited, not re-run**, and every number carries the tag *MS MARCO / BM25 / 20 rounds / experimenter-imposed adversarial condition / not re-run in this study*.
5. **No reranker training, no LoRA, no fine-tuning of anything.** This is the strongest learning claim available and it is deliberately traded away: MPS training is not bit-exactly reproducible, and the unload/reload cycle serialises a study already bounded by one machine. Deferred to v0.4, and the trade is stated in the README.
6. **No long-context ceiling arm. [changed:** deleted. The corpus is ~599k tokens; KV cache at that length is >100 GB. On the global stratum the arm is physically impossible, and on the others it is unaffordable. Deleting is honest; caveating is not.**]**
7. **No claim of bit-exactness for the whole pipeline.** Retrieval is bit-exact *conditional on the committed embedding matrix*. Generation is not, and is reported with a measured run-to-run noise floor.
8. **No episodic memory, no few-shot insight distillation, no generative query rewriting.** Reasons in §6.
9. **No sweep of the GraphRAG index.** One configuration. The tuning budget per arm is a reported number, not a hidden asymmetry.
10. **Not a claim about generator scale.** The primary generator is an 8B-class model. The 27B appears once, as a bounded sanity arm.

---

## 4. The corpus

### 4.1 What

**MultiHop-RAG**, restricted to the `technology` + `business` categories.
Source: `https://huggingface.co/datasets/yixuantt/MultiHopRAG` (the GitHub `dataset/corpus.json` 404s on `main`).
Licence: **ODC-BY**, per the HuggingFace dataset card. The arXiv page says CC BY-SA 4.0. The README states we relied on the HF card and links both.

| | Full | **tech + business** |
|---|---|---|
| Articles | 609 | **253** |
| Words | 1,063,319 **[measured]** | **450,381 [measured]** |
| ~Tokens | 1.41M | **599k** |
| Chunks @1024 tok / 128 overlap | — | **~669** |
| Answerable native queries retained | 2,255 | **1,381 [measured]** |

Retained native queries by type **[measured]**: inference 671, comparison 391, temporal 319, plus 301 nulls.

### 4.2 Why

Four hard filters kill nearly everything before quality is discussed. GraphRAG indexes **one shared corpus once**, so any benchmark shipping per-question retrieval contexts is structurally disqualified (CRAG, RAGBench, TAT-QA, ConvFinQA, FinQA). It must be redistributable (kills GraphRAG-Bench, BioASQ). It must fit an overnight index (kills the full 1.41M-token corpus, hence the subset). And a non-commercial licence is disqualifying for a repo whose thesis is banks, telcos and governments (kills FinanceBench and the BenchmarkQED AP data).

What survives has to satisfy the real requirement, which is not "a corpus" but "a corpus that can be stratified six ways with gold that does not come from a judge." MultiHop-RAG is the only public option shipping third-party-authored queries pre-labelled by reasoning type **with annotated evidence spans**. That matters most for credibility: the author did not write the questions he is scored on.

Fallback if news is judged too toy-ish: a ~60-contract stratified draw from **CUAD v1** (CC BY 4.0, commercial use and redistribution explicitly permitted), landing at the same chunk budget with 6,702 exact spans and 14,208 `is_impossible` cases as a free abstention stratum. Its weakness is the mirror image: contracts are self-contained, so multi-hop questions would have to be authored.

### 4.3 What is wrong with it, stated up front

Four properties were measured, not assumed, and all four change the design.

**(a) The questions name their own sources.** **98.2–100% [measured]** of retained multi-hop queries contain at least one gold outlet name verbatim ("…as reported by both The Verge and TechCrunch…"); **76.5–99.0%** name *all* of them. This is an artifact of the GPT-4 generation template and it is a lexical retrieval oracle handed directly to BM25 — which is the arm predicted to win the headline.

**[changed: FR-4]** Ship an **outlet-stripped variant** of every native query: a committed deterministic regex/NER pass over the 49 known outlets that deletes outlet names and "according to an article from X" clauses. Commit both files. Score every stratum on both. Retrieval-leg scoring is free (numpy); generation on the stripped variant is run for S1 and S3 only, on the two arms that carry the headline (A2, A3). The **as-published-vs-stripped delta is a publishable result nobody has reported**: how much of MultiHop-RAG retrieval performance is the questions naming their own publishers. If BM25's win survives stripping, the headline is earned.

**(b) The subset removes exactly the queries a graph exists to serve.** Retention into tech+business is non-uniform **[measured]**: inference **82.2%**, temporal **54.7%**, comparison **45.7%**. A query survives only if *all* its evidence is inside the subset, so every cross-domain bridge is deleted. We kept 82% of the stratum where prior art says the arms tie and deleted 54% of the stratum with the biggest published graph win.

**[changed: FR-5]** Publish the per-stratum retention rate in the README as a known selection effect. Report a **cross-category evidence covariate** — for the *full*-corpus retrieval-only runs, split every stratum by whether gold articles span ≥2 original categories. On the subset that split is degenerate; say so, and treat the graph result as bounded on this axis.

**(c) The answers are tiny and highly concentrated.** **[measured]** 107 distinct gold answers across all 2,255 answerable queries. Mean 6.1 chars, median 3, **79.6% single-token**. Comparison golds are **95.9% Yes/No** — always answering "yes" scores **60.1%**. Temporal is **89.5% Yes/No** — always-"yes" scores **46.7%**. Inference has 36 distinct answers over 671 retained, of which **"Sam Bankman-Fried" alone is 40.4%**, and the top two cover 71.8%.

**[changed: FR-6]** Add a **majority-class arm (AM)** — zero LLM cost, computed from the training split — and **refuse to report any stratum win that does not clear it**. Note plainly in the README that the prior art cited as motivation (GraphRAG 60.63 vs RAG 57.59 on comparison) sits **at or below the constant baseline**. Also: gold casing is inconsistent (`Yes` 782 / `no` 536 / `No` 25 / `True` 10) — ship the normalizer with the harness so arms do not differ on capitalisation habits.

**(d) Parametric leakage is real and stratum-specific.** The articles are Sept–Dec 2023. Published contextless accuracy is 0.41 on HotpotQA, 0.35 on 2WikiMQA. S4's dominant answer is the most heavily trained-on tech-news answer of 2023.

**[changed: FR-7]** The closed-book arm (A0) runs on every stratum, and **both raw accuracy and retrieval lift are reported for every cell**. This is not cosmetic: leakage is high on S1–S4 (the plain/agentic strata) and zero on S5 by construction (the graph stratum), so reporting lift alone gives the graph a zero floor and reporting raw alone does the reverse. The crossover slider must be recomputable under both. The article does not get to pick one.

### 4.4 The strata, with pool sizes disclosed next to the n's

**[changed:** the design specified n=300 on six strata. Two pools cannot support it and one does not exist at the assumed size. Re-cut below.**]**

| Stratum | Source | Pool | Eval n | Dev n | Gold | Primary metric |
|---|---|---|---|---|---|---|
| **S1** single-fact lookup | authored from `evidence_list.fact` fields | **420 [measured]**, not ~6,000 | **250** | 60 | exact doc + span | EM, TOST ±7 |
| **S2** comparison | native `comparison_query` | 391 | **150** | 60 | exact | EM (bounded) |
| **S3** temporal | native `temporal_query` | **319** | **180** | 59 | exact | EM |
| **S4** multi-hop inference | native `inference_query` | 671 | **150** | 100 | exact | EM (bounded) |
| **S5a** metadata-aggregative | authored off corpus metadata | unbounded | **50** | 20 | pandas | numeric match / set-F1 / Kendall τ |
| **S5b** content-aggregative | authored, gold from independent NER co-occurrence | unbounded | **50** | 20 | set-F1 | set-F1 |
| **S6** null / abstention | native `null_query` | 301 | **150** | 60 | refusal | risk-coverage curve |

Total generation questions: **980**.

**On S1 — the most consequential stratum and the most contaminated by construction.** It carries the TOST equivalence claim that actually kills GraphRAG's business case. Its 420 facts are the same ~27-word article sentences that generate S2–S4, so **S1 and S2–S4 are not independent samples**, and n=250 is 60% of the pool. Three mitigations, all committed artifacts: report question↔gold-chunk Jaccard overlap per item and **stratify the S1 result by overlap quartile** (if BM25 only wins the top quartile, say so); ship an entity-preserving **paraphrased S1 variant** generated once offline; disclose the pool size and non-independence in the README table, not a footnote.

**On S3 — kept separate from S2 deliberately.** The published gap is +19.9 on temporal versus +3.0 on comparison. Merging them averages a decisive result with an undecidable one and destroys the finding.

**On S2 and S4 — pre-registered as bounded, not measured. [changed:** the design promised n=300 on both. The effects are +3.0 and −0.25 points; resolving them needs n≈1,300 per stratum, roughly 130 machine-hours each. That budget does not exist on one laptop at any generator size.**]** The abstract will say: *comparison and inference are bounded within ±11 points at the sample size one laptop affords, and we cannot rule out zero.* Reporting them at n=150 as resolved would be the dishonest version.

**On S5 — split, and the design's internal contradiction resolved. [changed:** the design defined S5 as both "gold computed by pandas" and "requires a judge because there is no short gold span." Both cannot be true.**]**
- **S5a metadata-aggregative** ("how many articles did TechCrunch publish in November 2023"). Exact pandas gold, judge-free. Add a **trivial structured-query baseline** that reads the metadata table and expect it to win outright. That is the finding: *this class of question should never be sent to a RAG system.* The design borrowed GlobalQA's type names and applied them to database fields.
- **S5b content-aggregative** ("which companies appear in both chip-supply and antitrust coverage"). This is the real graph stratum. Gold is computed by **an independent, committed NER co-occurrence pass** (spaCy, or the proper-noun matcher already measured). **Hard constraint: the answer key must not be produced by qwen — if the same extractor writes the graph and the gold, GraphRAG is graded by its own annotator.** Scored by set-F1.

**On S6 — never scored alone.** Abstention rate is a prompt property, not an architectural one: an arm can buy S6 by being timid and pay on S1–S4. Report the paired operating point as a **risk-coverage curve** per arm. Also **[measured]**: only **21.9%** of the 301 nulls name an outlet that exists in the tech+business subset, so 78% are topically out-of-corpus and trivially rejectable. Sample **hard nulls** (top-1 retrieval score above a threshold) and report the easy/hard split.

---


---

*Section 4.5 added 8 Aug 2026 after a dedicated 8-agent chunking workflow. It corrects several decisions made earlier in this document, including a fairness-contract violation in §5.3 and a measured embedder-context error. Changes it forces are listed in 4.5.7.*

## 4.5 Chunking — the one shared transformation

*Supersedes §5.3 item 2 in full. §5.3 item 2 currently reads "1024 tokens, 128 overlap, tuned once on the held-out dev split **for the plain arm**." That sentence is a fairness-contract violation written into the fairness contract, and it is struck.*

---

### 4.5.1 The decision

**Sentence-aligned greedy token packing to a 600-token budget, article-scoped, zero overlap, `title + "\n\n"` prepended and charged against the budget. One scheme, frozen, byte-identical across all three arms — and run a second time at 1200 tokens for every arm, with the chunk-size × architecture interaction published as a primary result rather than confessed as a limitation.** Sentences are split with pySBD, packed greedily into whole-sentence chunks, never split, never spanning two articles. Tokens are counted with the Qwen3 BPE, not with a chars-per-token approximation and not with tiktoken. Expected output at 600: **~1,102–1,139 chunks** over the 253-article tech+business subset — a number that is currently *projected two different ways and measured zero times*, and which is not published until the chunker has actually run (E1, §4.5.6).

**[changed:** the decision entering review was a single scheme at a single size, with the GraphRAG arm labelled a lower bound. Both halves of that are withdrawn. One size is not defensible when chunk size moves the primary retrieval metric 4.6× more than architecture does, and "lower bound" was an unproven directional claim about a bias that measurement shows runs both ways.**]**

---

### 4.5.2 Why, against the evidence

Four things are established well enough to build on, and I have separated them by what kind of evidence they are.

**Sentence alignment is the part that does the real work, and it is nearly free.** Peer-reviewed and replicated: fixed-size chunking at sentence or paragraph boundaries is best-in-class on in-corpus retrieval (Zhou et al., SIGIR 2026, paragraph 0.4948 nDCG@10 vs fixed-size 0.4849 vs semantic 0.4726), and paragraph-group chunking wins a 36-strategy sweep at 0.459 vs 0.244 for fixed-character (Shaukat et al., preprint). The corpus cooperates: **100% of documents have blank-line paragraph breaks (mean 46.6/doc), 47,778 sentences at a mean of 28.3 tokens, and no sentence anywhere in the corpus exceeds 398 tokens** — so the hard-split path is an assertion, not a behaviour.

**The current scheme is measurably damaging, and the damage is question-type-shaped.** MultiHop-RAG ships gold evidence as verbatim sentences; all 981 unique spans locate exactly in the article bodies. Naive 2048-char chunking severs **6.83% of gold spans and costs 17.18% of queries at least one gold span** — inference 24.8%, comparison 16.4%, temporal 16.6%, null 0%. The deliverable of this study is a per-question-type curve. A scheme that destroys inference evidence at 1.5× the rate of comparison evidence puts a question-shaped thumb on the scale before any architecture runs. That is a worse threat to the headline than chunk size is.

**The unit is settled.** The 4-chars/token assumption is biased, not noisy: true value is **4.607 chars/token** on tech+business, **4.409** on all 609. Nominal "512-token" chunks were **446 real tokens at 87% fill with a 345–633 spread**. The generator and embedder tokenizers are byte-identical (merges sha256 match; first 151,669 token entries match; the 8B's extra 267 entries are `[PAD151669]`–`[PAD151935]`), differing only in that the embedder appends EOS — verified as `emb_tokens == gen_tokens + 1` on 25 texts. One tokenizer governs both arms. A 600-token Qwen3 chunk is **3% less text** than a 600-token `o200k_base` chunk (4.607 vs 4.748 chars/token), so the unit comparison with Microsoft's configuration is sound.

**600 has external provenance, but the provenance argument is demoted.** [changed:] 600 is the chunk size Microsoft used for GraphRAG's published MultiHop-RAG experiments (arXiv:2404.16130 §4.1.1) and one of the three sizes in the only published chunk-size × architecture ablation on this dataset (arXiv:2503.04338 Table 7). That is worth something. It is worth **less** than the draft claimed, for three reasons that are now stated in the section rather than discovered by a reviewer: (i) 2404.16130 is a vendor-authored preprint, never peer-reviewed, and Microsoft's *shipped* default is 1200/100 — 600 was an experimental setting; (ii) their 600 came with `overlap: 100` and `max_gleanings: 1`, and this study takes the size and drops both compensations, which forfeits any claim that "the configuration was chosen by someone else"; (iii) Table 7's "MultihopQA" is the **full 609-article / 1.43M-token corpus with different models and a different embedder** — same dataset, different experiment. Provenance is now a supporting note, not the first line of the rationale.

**Table 7's argument is restated on recall only.** [changed:] The draft leaned on LGraphRAG-over-VanillaRAG **+0.86 accuracy at 600**. At n=2,556 that is **0.61σ** — noise, and quoting it hands a reviewer a free hit. The recall figures survive: **+3.53 at 600 (2.5σ) rising to +13.51 at 1200 (9.7σ)**, and VanillaRAG's own 600→1200 drop of −7.28 accuracy against HippoRAG (5.2σ). The claim this section makes is therefore: *on recall, on this dataset, the graph-over-vanilla margin roughly quadruples between 600 and 1200 tokens.* Nothing about accuracy margins at 600.

---

### 4.5.3 What is rejected, why, and what it would have cost

| Rejected | Reason | Cost avoided |
|---|---|---|
| **Character chunking at 4 chars/token** (the current scheme) | Biased 13% low, 1.83× per-chunk spread, severs 6.83% of gold spans and 17.18% of queries with question-type asymmetry (inference 24.8% vs comparison 16.4%). Also mis-sizes the generator's evidence budget by 13% with per-query variance. | — (it is the incumbent) |
| **Semantic / embedding-breakpoint / clustering** | The strongest negative result in the field, peer-reviewed. Qu, Tu & Bao (Findings of NAACL 2025): fixed-size beats breakpoint and clustering on HotpotQA (90.59 / 87.37 / 84.79 F1@5), MSMARCO, ExpertQA, TechQA; answer BERTScore identical to two decimals. Semantic wins only on artificially *stitched* corpora, and a news article is the least stitched document type there is. **Decisive here for a second reason:** a semantic chunker embeds the corpus with the same model the retriever uses. That is a direct thumb on the scale for the dense arm and a contract violation on its face. | 4.9 min–3.09 h chunking, and the contract |
| **Late chunking (Jina)** | Architecturally unavailable, and this goes in writing now rather than at analysis time. It requires mean pooling over a bidirectional encoder. Per the Qwen3-Embedding technical report (arXiv:2506.05176), `qwen3-embedding:0.6b` uses causal attention with an appended `[EOS]` and takes that last token's hidden state. Under a causal mask token *i* has never seen token *i+1*, so mean-pooling a span does not produce the contextual property the method depends on. The 32k advertised window does not rescue it — the pooling is the problem. Evaluating it means swapping one of the five frozen variables. | Reported gain was +3.63% relative; a re-run of the entire study |
| **Contextual retrieval (Anthropic-style per-chunk LLM context)** | Affordable (~6.4 s/chunk, ~2.9 h, 15% of the graph budget) and contract-legal, since it applies identically to all arms. Rejected on provenance: vendor blog, no paper, no released harness, undisclosed corpora, and the headline −49% has never been independently replicated at that magnitude. Independent work finds it not uniformly positive — Claim Recall *decreased* in ECIR 2025's most advanced pipelines; SIGIR 2026 measured −5.16% to −53.17% on in-document retrieval. The one shared transformation that must be above suspicion is not where an unreplicated vendor technique goes. **Priced, named, deferred.** | 2.9 h |
| **Propositions / atomic facts (Dense X)** | +10.1 Recall@20 in the original EMNLP 2024 paper, but the gain concentrates on weak unsupervised retrievers (+2.7 supervised) and `qwen3-embedding:0.6b` is not weak. SIGIR 2026 ranks propositions **last of six** (0.3888 vs 0.4948). Contradicted by a controlled representation-only ablation: verbatim chunks beat LLM-extracted artifacts by 15.9 pts on LoCoMo and 22.0 on LongMemEval-S. | 15.05 h with repeated timeouts |
| **LLM-boundary chunkers (LumberChunker, MoC, HiChunk)** | LumberChunker wins in-document (0.5640 DCG@10) and *loses* in-corpus (0.4690 vs 0.4948) at 1.11 docs/s against paragraph packing's 1,854 — a 1,670× cost for a loss. Fatal here regardless: running qwen3:8b as boundary predictor makes chunking a function of the same model that powers the graph arm's extractor. An architecture-coupled chunker is exactly what the contract forbids. (MoC's *Boundary Clarity* and *Chunk Stickiness* are kept as intrinsic Stage 0 scores — scoring a scheme is not letting a model choose it.) | 8.37 h |
| **Overlap (100 tokens / 10–20%)** | Sentence alignment already drives severance to near zero. The late-chunking appendix states overlap "generally neither improves nor harms" retrieval; a 0–80% sweep raises recall 0.711→0.724 while Precision@Ω drops 15%, and >40% past 20% overlap. It is also arm-asymmetric in **both** directions: +10–24% LLM calls and duplicate entity mentions for the graph arm, and duplicate near-identical chunks consuming top-k slots for the vector arms. | +10.1% to +23.6% of every index build |
| **256 tokens** | [changed:] **rejected on graph degree and dense-arm recall, not on cost.** The draft claimed 256 costs ~24 h "because per-chunk cost falls sub-linearly" — that is arithmetically backwards under the study's own fitted k=1.43, where total cost ∝ s^0.43 makes 256 the **cheapest** candidate (~12 h vs 600's ~16 h). Shipping that sentence would have been a free hit. The real reasons are that it drives the graph below any published degree band and that MultiHop-RAG's own reference implementation at 256 tops out at Hits@10 = 0.7467. | — |
| **Per-arm chunk tuning of any kind** | The thing the contract exists to forbid. Recorded here explicitly so the rejection is on the record rather than assumed — including the struck "tuned for the plain arm" clause in §5.3. | — |

---

### 4.5.4 The fairness problem

**There is no neutral chunk size. The PRD's premise — that freezing one scheme across arms makes the comparison fair — is false on this corpus, and the proof is published on this dataset.** arXiv:2503.04338 Table 7 runs 600/1200/2400 with everything else fixed: VanillaRAG loses 13.6% recall going 600→1200 while LGraphRAG gains 9.0%; VanillaRAG vs HippoRAG **inverts** (Vanilla +7.28 accuracy at 600, Hippo +3.13 at 1200). Freezing one size does not remove the bias. It picks a point on a known interaction surface.

**[changed:] The direction of that bias was asserted in the draft and is wrong.** The draft's conservatism argument — "BM25 wants 4096–8192 tokens, so 600 starves the lexical arm and handicaps only the graph" — was imported from DAPFAM, a patents corpus. Measured on *this* corpus at a fixed 6,000-token retrieved budget:

| chunk tok | BM25 recall | dense recall | hybrid RRF recall | fusion gain |
|---|---|---|---|---|
| 90 | **74.5** | — | — | — |
| 150 | 71.7 | 64.6 | **74.3** | **+2.6** |
| 300 | 66.7 | 60.7 | 69.4 | +2.7 |
| **600** | 62.2 | 56.7 | 62.4 | **+0.2** |
| 1200 | 58.1 | 52.0 | 58.2 | +0.1 |

BM25 is **monotonically decreasing** from 90 tokens up; its optimum is ~90, not 4096. At 600 the hybrid arm is **−11.9 recall points** below its own optimum, against the graph arm's ~−10 relative to 1200 per Table 7. **The two arms are handicapped roughly equally, and the net direction is unknown.**

**Consequently: the sentence "our GraphRAG results are a lower bound" does not ship.** It is unproven, and an unproven directional claim is worse than a disclosed unknown, because the paper would be claiming to know something it does not.

**[changed:] The baseline arm is relabelled.** Fusion gain collapses from +2.6 recall points at 150 tokens to **+0.2 at 600** — a 93% loss — with the mechanism confirmed: mean Jaccard between the BM25 and dense candidate sets rises 0.214 → 0.262 as chunks grow, so the two retrievers converge and RRF has nothing to fuse. At 600, hybrid's inference-query recall (58.1) is *below* its own BM25 component (58.3), and dense never beats BM25 at any tested size at fixed budget. Every table and figure caption reads **"hybrid (BM25-dominated; RRF contributes +0.2 recall at this chunk size)"**, not "plain hybrid RAG." Identical chunking preserves the *input*; it does not preserve each architecture's *mechanism*, and at 600 it zeroes out what makes the hybrid arm hybrid. That is a finding about the fairness contract itself and it is reported as one.

**[changed:] Fixed k is replaced by a fixed retrieved-token budget, k derived.** §5.3 item 4 currently fixes k=5. At fixed k, BM25 all@10 *rises* 16.8 → 36.9 with chunk size; at a fixed 6,000-token budget it *falls* 44.3 → 22.7. Same corpus, same retriever, opposite conclusions — the contract's own choice sets the sign of the chunk-size effect. Fixed k is also not commensurable across arms: at k=5 the hybrid arm receives 1,330 / 2,568 / 4,578 evidence tokens at 300 / 600 / 1200, while the graph arm's local-search context is governed by its own `max_tokens` and does not move. This costs zero machine-hours and is non-negotiable.

**[changed:] The chunk-size sensitivity run is required, and it is costed.** Vector arms at both 600 and 1200 are ~15 minutes of machine time and are already partly done. The graph arm's second index is **+21.7 h** (639 chunks × 122.3 s, lean, ungleaned). Both graph indices must run under the *same* gleaning setting or the comparison is uncontrolled; the recommendation is **both ungleaned, plus a gleaning delta measured on a 200-chunk stratified sample at 600 (1.7 h)**. The second index also subsumes experiment E5 — Leiden ARI/NMI over 5 seeds and post-dedup degree are computed on both real indices rather than on a 50-article proxy, which removes 14.1 h from the measurement program. Net chunking-attributable budget: **16.4 h (600) + 21.7 h (1200) + 1.7 h (gleaning sample) + 4.7 h (E0–E4) = 44.5 h**, against the 16.0 h currently in §11. That converts the study's central methodological risk into a headline: *below ~1200 tokens, chunk size decides the winner by more than architecture does.* Table 7 predicts it on this corpus; this would be the first self-hosted confirmation. **If that 28.5 h increment cannot be found, the cut is the 1200 graph index only** — vector arms still run both sizes, and the graph interaction is explicitly labelled *imported from Table 7, not measured here*.

**Corpus subset coverage is disclosed as a question-mix effect, not just a retention rate.** §4.3(b) already reports retention (inference 82.2%, temporal 54.7%, comparison 45.7%). What is added: **874 of 2,255 answerable queries (38.8%) are unscorable for all three arms**, and the evaluated mix shifts comparison 38.0% → 28.3%, temporal 25.9% → 23.1%, **inference 36.2% → 48.6% (+12.4 pts)**. The deliverable is a function of question mix, so the corpus decision — which exists only because the graph arm's wall clock forced it — re-weights the study's independent variable by 12 points before anything runs. **Every headline number is reported both raw and re-weighted to the original 2,255-query mix.** Re-weighting costs zero hours. The full-609 alternative is +21.7 h on top of everything above and is not recommended.

---

### 4.5.5 The graph-specific problem

**The superlinearity finding is stated with its confidence bounded, and every k-derived number is frozen until E0 returns.**

What was measured: one chunk, one article, one sample, five cells. Fitting `cost = c·s^k` to two points with two free parameters gives **zero degrees of freedom** — the "predicted −25.9% vs observed −25.9%" agreement is arithmetic, not corroboration. The same is true of α_ent = 0.96, α_rel = 1.61, R ∝ E^1.69, and the "21.1 tok/s with ~0 s fixed overhead" solve. That last one is also physically wrong for the shipped configuration: Microsoft's extraction prompt with few-shot examples is ~1,500–2,500 tokens, so prefill at 369–1,697 tok/s is **1.8–8.1 s** — 3–13% of a 63 s chunk, and the gleaning pass re-sends the whole prior conversation for another 2.3–10.6 s. The term set to zero is larger than the gate margin being argued over.

Under the most generous noise model (Poisson, a variance *floor* — content variance across news articles dominates it), the headline edge loss is **1.9σ** (52 vs 2×17, se 9.27) and "entity count is conserved" is **0.12σ** (33 vs 34, se 8.19). Neither clears p<0.05. **"You traded 35% of the graph's edges for 26% of the time" is not yet a result and does not ship in that form.**

**[changed:] One unread config flag can erase the whole finding, and it is checked first.** Qwen3 ships thinking-on by default in Ollama. If `gen tokens` is `eval_count`, it includes reasoning, and reasoning length scales with input complexity — producing exactly this curve with no entity-pair mechanism at all. Break-even: **556 reasoning tokens at 1024 with zero at 512 (23% of that cell's output) takes k from 1.38 to 1.00.** The `format` grammar may have suppressed thinking entirely, in which case the finding survives — but nobody checked. E0 is 12 minutes and it gates everything.

**The entity-pairs framing is dropped.** Measured β is 1.69, not the 2.0 that pairs predicts. The defensible mechanism is a *proximity-limited co-occurrence window*, and the cost law reduces to `seconds ≈ generated_tokens / decode_rate`, which makes every cost claim falsifiable as an output-token count.

**What n=30 buys, stated honestly.** The paired-split experiment (n≥30 articles, 1×1200 vs 2×600, production prompt, latencies logged per call) has an MDE of ~30% relative at 80% power with a log-ratio sd of 0.5, and a CI of roughly ±18%. **That licenses "there is a loss, somewhere between about 15% and 55%." It does not license quoting 34.6% to one decimal.** Quoting the point estimate to ±10% needs n≈100 and 13.1 GPU-h. The claim is written to the sample size purchased.

**[changed:] The gate is re-derived on the tail, with the production prompt, and it fails.** The draft claimed 600 "passes with margin under both schemas (55.4 s and 76.1 s)." Those figures use mean fill (513.6 tok measured, dragged down by a left tail of short trailing chunks — median is 578, p90 596) and zero gleanings. With `max_gleanings: 1`, the default in Microsoft GraphRAG, nano-graphrag and LightRAG:

| config @600, lean | mean s | p95 s | index | % chunks over 90 s |
|---|---|---|---|---|
| no gleaning | 51.7 | 62.4 | 16.4 h | **0.0%** |
| +1 gleaning | 82.8 | 98.8 | 26.2 h | **70.9%** |
| entity-desc, +1 gleaning | 111.5 | 130.3 | 35.3 h | **83.7%** |

A per-chunk gate evaluated on the mean is not a gate. **71% of chunks fail it with gleanings on.** So the gate has not stopped dictating the configuration; it moved from dictating the schema to dictating gleanings, which is worse, because gleanings are the mechanism the GraphRAG authors added to compensate for exactly the chunk-local recall loss this study is worried about. **Resolution: the per-chunk 90 s gate is replaced by a corpus-hour budget, the two-size comparison runs ungleaned for control, and the arm is labelled "GraphRAG, gleanings disabled, no overlap" in every table and figure caption**, with the statement that both of Microsoft's chunk-local-recall compensations were removed while their chunk size was retained. The gleaning delta is measured once on a 200-chunk stratified sample (1.7 h) and reported. Verify in the logs that the setting takes effect — microsoft/graphrag#613 reports `max_gleanings: 0` not applying from `settings.yaml`.

**Cross-chunk severance may not matter at all on this benchmark, and the section says so rather than hedging.** **0 of 2,255 answerable queries have all their evidence in one article; 100% span ≥2 distinct articles** (mean 2.62, max 4), uniformly across inference/comparison/temporal — so ~97% of evidence items are the only item from their source article. The relations this benchmark scores are inter-*document*. Chunk-local extraction was never going to see them at any size, and no chunk size can sever them. **Within-chunk edge density therefore has no demonstrated bearing on this benchmark's score, and chunk size is not selected on it.** What the graph arm needs instead is normalized-name entity merging plus coreference at build time, and `chunks ∪ artifacts` retention — the graph arm keeps and can retrieve the verbatim source chunks, because a controlled representation-only ablation shows extraction-only indices losing 15.9 pts on LoCoMo and 22.0 on LongMemEval-S to the raw text they were extracted from. Discarding chunks would handicap the graph arm in the opposite direction, which is the same violation.

**Leiden degeneracy is left open, not claimed either way.** The draft rejected 512 at projected degree 2.00 because it sits below the 2.88–4.42 band where partitions are shown non-reproducible — but 600's projected 2.43 is **also below 2.88**, so the argument used to leave 512 does not license landing on 600. All these degrees are pre-dedup instance counts derived from an n=1 α fit; the theorem concerns the *merged* graph, which the design's own entity-merging step will make denser by an unknown factor. This is settled only by building a graph and running Leiden under ≥5 seeds on both indices, reporting ARI/NMI. Until then it is a disclosed unknown.

**CrossAug is priced and declined**: +0.9 to +1.99 EM on multi-hop (0.0 / −0.2 on HotpotQA) for +8.2% LLM calls and +15.3% wall clock. Roughly one EM point for 15% of an index, on relations that are mostly inter-document here anyway. Cited in limitations, including their finding that community summarization "cannot recover such precise cross-chunk relational facts" — the Leiden step does not launder away severed edges.

---

### 4.5.6 Implementation

```
tokenizer      Qwen3 BPE from Qwen/Qwen3-8B tokenizer.json (Apache-2.0)
               151,669 vocab + 151,387 merges, pre-tokenizer "qwen2", add_bos=false
               proven byte-identical to qwen3-embedding:0.6b's GGUF tokenizer;
               only difference is add_eos_token=true on the embedder
               (emb_tokens == gen_tokens + 1, verified on 25 texts)
               NOT tiktoken (cl100k/o200k mis-measure this text by 10-20%)
               NOT a chars-per-token approximation (true value 4.607, not 4.0)
budget         600 content tokens, title included; second configuration at 1200
boundaries     paragraphs on blank lines, then pySBD sentences (MIT, rule-based,
               deterministic, 97.92% on the English Golden Rule Set vs spaCy's 52.08%
               -- news is full of "U.S.", "Inc.", "Sept. 12")
               greedy whole-sentence packing; prefer a paragraph boundary falling in
               the last 15% of the budget; never split a sentence; never span a doc_id
               hard split at token offsets is dead code -- max sentence is 398 tokens
overlap        0
header         title + "\n\n", charged against the budget (17.9 tok mean, p95 30, max 57)
               NOT source, NOT published_at
metadata       doc_id, url, source, published_at, category, chunk_index,
               char_start, char_end, n_tokens, tokenizer_sha256, chunker_version
packer         ~40 lines in-repo. Not semchunk, not chonkie, not LangChain --
               a study selling bit-exact reproducibility does not let a patch bump
               change its chunk IDs
```

**Two Ollama landmines, both measured on 0.32.5, both documented in the harness.** The embedder's usable context is **4,096 tokens, not the 32,768 the GGUF advertises** (4,095 accepted, 4,096 rejected) — which **corrects §5.3 item 3's claim that the 32k context "makes chunk size a free parameter."** It does not; 8 of 609 articles exceed 4,096 tokens, which caps any one-embedding-per-article ablation. And **passing `options.num_ctx` to `/api/embed` kills the runner**: the triggering call returns normally and every subsequent embed fails with `do embedding request: … EOF`. Never pass `options` to `/api/embed`; assert on `prompt_eval_count` and fail loud.

**Determinism.** `tokenizer.json` sha256 pinned in the run manifest; `tokenizers` pinned exactly; a canary-string token count asserted at startup so a library bump cannot silently re-chunk the corpus; `n_tokens` recomputed and asserted at index time. If any of these change, every chunk ID changes and every downstream number in all three arms is invalidated together — which is the property we want.

**[changed:] The severance claim is measured, not asserted.** "~0% by construction" has two leaks: **6 of 981 gold spans (0.61%) are multi-sentence** and a sentence packer will split them, and pySBD's ~2% boundary error rate concentrates in exactly the abbreviations news is made of, putting ~20 spans at splitter risk. The honest expectation is **0.5–2%**, not 0% — still 4–14× better than 6.83%, and it costs CPU-seconds to measure rather than argue.

**Committed Stage 0 artifacts, in run order:**

| # | Experiment | What it settles | Hours |
|---|---|---|---|
| **E0** | Rerun the 5 cost cells with `think:false`; log `eval_count`, `prompt_eval_count`, `done_reason`, raw string; assert `done_reason=="stop"` | Kills or confirms the thinking-token and truncation confounds. **Precondition for every other number in §4.5.5** | 0.2 |
| **E1** | Run the real chunker at 600 and 1200; emit chunk count, token histogram, fill rate, per-question-type severance including the 6 multi-sentence spans and the pySBD-vs-gold disagreement rate | Replaces the projected chunk count, the projected index hours, and the "~0%" claim | 0.1 CPU |
| **E2/E3** | Paired split, n=30 articles stratified by capitalised-token ratio: 1×1200 + 2×600, production prompt, per-call latencies. Paired log-ratios with 95% CIs; classify every 1200-only relation as boundary-severed vs within-half-but-unemitted; p50/p95 latency | Converts k, α, the edge loss and the gate verdict from anecdote to bounded result. **The severed-vs-noise split does not exist in the literature** — CrossAug, the paper about this exact worry, runs no chunk-size sensitivity at all | 3.9 |
| **E4** | Embedder-only sweep: embed both chunk sets, run 2,255 queries, Hits@10 / MRR@10 per question type × {title, title+source, body} | Prices the chunk-size confound for both vector arms, and prices the title decision on something other than BM25 | 0.5 |
| **E6** | MoC *Boundary Clarity* and *Chunk Stickiness* over both chunkings | Intrinsic chunk-quality score, no downstream QA, defensible against "picked by vibes" | 0.2 |

**[changed:] The title-prepending decision is provisional pending E4.** The +1.0 pt gain that survives outlet-stripping (33.6 → 34.7) was measured with **BM25 alone, on the abandoned 2048-char scheme, over all 609 articles, with no significance test and no dense retriever**. The unmeasured risk runs the other way for the dense arm: prepending an identical ~18-token string to every chunk of an article compresses intra-article embedding distance, which is a known failure mode with last-token-pooled encoders and is precisely what must work when 2.62 articles each have to contribute a distinct chunk. If E4 shows dense-arm degradation, the title comes out.

**The `source` leakage finding stands and ships verbatim.** 91.6% of queries name an outlet (100% of comparison and temporal); prepending `source` gives +3.5 pts strict all@10 that **vanish entirely** when outlet names are stripped from queries (38.0 → 34.6, against title-only's 34.7). Every point is the retriever matching a string the benchmark's own query generator wrote into the question, and it is arm-asymmetric: BM25 gets it free, the graph arm would have to have extracted "TechCrunch" as a node. Primary study runs source-excluded; the source-included delta is reported as a named **outlet-name leakage** ablation. This is a finding, not a confound.

---

### 4.5.7 What this changes in the PRD

| Location | Current | Replace with |
|---|---|---|
| **§4.1 table** | "Chunks @1024 tok / 128 overlap — ~669" | Two rows: **@600 sentence-packed, 0 overlap — 1,102–1,139 (projected; E1 replaces)** and **@1200 — ~639**. Mark both projected until E1 runs. |
| **§4.1 table** | "~Tokens 599k" | **587,104 [measured]** on the 253-article subset; 1,428,283 for all 609. |
| **§4.3(b)** | retention rates only | Add the **absolute loss (874 / 2,255 = 38.8% unscorable for all arms)** and the **evaluated-mix shift (inference +12.4 pts)**. Extend FR-5: every headline number reported raw **and** re-weighted to the 2,255-query mix. |
| **§5.3 item 2** | "1024 tokens, 128 overlap. Tuned once on the held-out dev split **for the plain arm**." | **Struck in full.** Replaced by §4.5.6. The "tuned for the plain arm" clause is a contract violation and must not survive in any form; there is no per-arm chunk sweep in this study. |
| **§5.3 item 3** | "The 32k context is load-bearing… makes chunk size a free parameter." | **Corrected.** Measured usable embedder context is **4,096 tokens**, not 32,768. Chunk size is not a free parameter; 8 of 609 articles exceed the real limit. |
| **§5.3 item 4** | "Retrieval budget k = 5." | **Fixed retrieved-token budget, k derived per arm per size.** At fixed k the chunk-size effect on the primary metric has the opposite sign to the effect at fixed tokens, and k is not commensurable between the chunk-retrieving arms and the community-summary-retrieving arm. Zero hours. |
| **§5.3, arm naming** | "plain hybrid RAG" | **"hybrid (BM25-dominated; RRF contributes +0.2 recall at 600 tokens)"** in every table and figure caption, with the retriever-sensitivity table from §4.5.4 shipped as a named Stage 0 artifact. |
| **§5.3, graph arm naming** | — | **"GraphRAG, gleanings disabled, no overlap"** in every caption, with the statement that Microsoft's two chunk-local-recall compensations were removed while their chunk size was retained. |
| **§5.3, tuning budget** | "graph: 1 config… labelled LOWER BOUND in the headline" | **Strike "lower bound."** The bias direction is unknown and measurement shows the hybrid arm handicapped at least as much. Replace with the measured two-size interaction. |
| **§11 budget** | "GraphRAG index — 669 chunks: 16.0 h" | **16.4 h @600 lean ungleaned + 21.7 h @1200 + 1.7 h gleaning-delta sample + 4.7 h E0–E4 = 44.5 h.** Net **+28.5 h**; study total ~110 h, ~126 h with the 15% thermal derate ≈ **16 overnight runs**, up from 12. E5 disappears into the second index. If the increment is unaffordable, cut **only** the 1200 graph index and label the graph interaction *imported from Table 7, not measured*. |
| **§12 risks** | "if per-chunk extraction > 90 s, drop the graph arm to a 300-chunk subset" | **Replace the per-chunk 90 s gate with a corpus-hour budget.** A per-chunk gate evaluated on the mean is not a gate — 71% of chunks fail at 600 with gleanings on. Add a new row: **"E0 shows thinking tokens in `eval_count"` → detection: Stage 0, 12 min → response: every k-derived number in §4.5.5 is withdrawn and refitted before the index runs.** |
| **§13 Stage 0** | "20-chunk extraction pilot, 3.0 h" | Add **E0 first**, then E1, E2/E3, E4, E6. Stage 0 becomes ~8 h and gates the index. |
| **§13 Stage 1** | "Build the chunk file" | Build **two** chunk files (600, 1200), both hashed and committed; run the retrieval-only sweep at both sizes for both vector arms before any generation. |
| **Pre-registered headline** | unchanged | Add a second pre-registered outcome: **"below ~1200 tokens, chunk size moves the primary retrieval metric more than architecture does."** Measured swing on the hybrid arm across 150→1200 is **16.1 recall points** against Table 7's +3.53 architecture margin at 600 — a 4.6× ratio. If that holds locally it is the study's strongest result and it must be pre-registered, not discovered. |

---

### 4.5.8 What this could not settle

1. **Whether the superlinear extraction cost is real.** k is unknown until E0 rules out thinking tokens and a `num_predict` ceiling. Every number derived from k — the index projections, the 512→600 cost argument, the schema comparison — is provisional.
2. **The magnitude of the cross-chunk edge loss.** Currently 1.9σ from n=1. n=30 will license "somewhere between ~15% and ~55%." Quoting a point estimate needs n≈100 and 13.1 GPU-h that are not budgeted.
3. **Whether within-chunk edge density affects this benchmark at all.** 100% of queries are inter-document, so the answer may be "not measurably." Only the two-size graph build settles it, and if the answer is no, four paragraphs of this section become an interesting negative result rather than a design constraint.
4. **Leiden reproducibility on the merged graph.** Every published degree figure here is pre-dedup and derived from an n=1 fit. The degeneracy theorem concerns the merged graph, which does not exist yet.
5. **Whether title-prepending helps or hurts the dense arm.** The +1.0 pt evidence is BM25-only, on a discarded scheme, on the wrong corpus slice.
6. **What the right chunk size is.** Nobody knows, the optima disagree by 2–4× between lexical and dense retrievers on the same corpus, and two dense encoders disagree with each other on news (Stella peaks at 512 and loses 12.8 pts Recall@5 at 1024; Snowflake improves monotonically). This study will report the interaction it measures on two sizes on one machine with one embedder. It will not claim to have found the size.
7. **Whether the tech+business subset can support the question-mix deliverable at all.** 38.8% of answerable queries are unscorable and the evaluated mix is 12.4 points more inference-heavy than the benchmark's. Re-weighting is a mitigation, not a fix. If the re-weighted and raw curves cross, the corpus decision — not the chunking decision — is what the paper is actually about.

## 5. The arms

### 5.1 Generator — the change that makes the study exist

**[changed: primary generator is an 8B-class Q4 instruct model, not `qwen3.6:27b`.]**

Measured on this machine today, `qwen3.6:27b` Q4, `think:false` forced, nothing else running:

| Quantity | Design assumed | **[measured]** | Error |
|---|---|---|---|
| Decode | ~12 s/generation | **6.36 tok/s** @250 tok; 6.63 @1,013 tok | — |
| Prefill | not modelled | **85.3–100.4 tok/s** | — |
| One k=10 × 1024-token RAG call | ~12–20 s | **129.2 s** (105.3 prefill + 18.8 decode) | 6–10× |
| One GraphRAG extraction chunk | 5.9–7.79 s | **174.2 s** | 22–29× |
| Cold model load | not budgeted | **11.4 s** | — |

The 5.9 s and 7.79 s figures from `reports/calibration_qwen3.6_27b_1bf57252a9a4.json` are the cost of emitting ~50 verdict tokens. Using them as the unit cost of a call that emits 1,000 tokens is the single error that produced the design's 41.6-hour estimate. Re-costed with measured constants, the design as written is **2,617 machine-hours — 63× over, roughly a year of unattended overnight runs.**

`ollama ps`, model reloaded cold at each context size **[measured]**:

| `num_ctx` | resident | CPU / GPU split |
|---|---|---|
| 4,096 | 17 GB | **11% / 89%** |
| 32,768 | 19 GB | **19% / 81%** |

**The 27B never fits entirely on the GPU, not even at 4k.** You pay 4× the wall-clock for a model throttled by a ceiling you cannot raise. The study's claim is about architectures, and its power comes from n. The counter-evidence paper being engaged used **Qwen2.5-7B at n=5,000**; proposing a 27B at n=34 inverts that trade in the wrong direction.

So: **generator held fixed at an 8B-class Q4 model** (verify the exact tag with `ollama list` / `ollama pull` at build time; if no 8B point exists in the qwen3.6 family, use the largest ≤9B Q4 instruct model available and record the digest). Projected 20 tok/s decode, 350 tok/s prefill — **projected, and Stage 0 measures it before anything else runs.** The 27B appears once, as **A-27B**, a bounded sanity arm on S3 only at n=60, to put a number on the generator-scale effect rather than assume it away.

### 5.2 The arms

| ID | Arm | Built / imported | Strata |
|---|---|---|---|
| **AM** | Majority-class constant | in-repo, free | all |
| **A0** | Closed-book, no retrieval | in-repo | all |
| **A2** | BM25-only, single pass | **in-repo, already verified** against brute force (1,200 comparisons, 0 mismatches) | all |
| **A3** | Plain RAG: BM25 + dense exact cosine + RRF, fixed weights, no router | in-repo, ~300 LOC | all |
| **A3t** | Plain RAG at **token-matched k′** | in-repo | conditional, see below |
| **A4** | Agentic RAG | in-repo, ~250 LOC | all |
| **A4a** | Agentic ablation: adaptive router | in-repo | 150-question slice |
| **A4b** | Agentic ablation: no decomposition | in-repo | 150-question slice |
| **A5** | GraphRAG-local | **vendored `nano-graphrag`** (MIT, ~1,100 LOC) into `third_party/` | all except S5 |
| **A6** | GraphRAG-global, map-reduce over community reports | vendored | S5a, S5b |
| **A6b** | Plain-RAG map-reduce at **matched context tokens** | in-repo | S5a, S5b |
| **AQ** | Structured metadata query | in-repo, free | S5a |
| **A-27B** | 27B sanity arm (A3 + A5 only) | — | S3, n=60 |

**Deleted from the design:** A1 random-chunk (retained as a retrieval-only diagnostic, no generation) and A7 long-context ceiling (physically impossible on S5, unaffordable elsewhere).

**A3 must not be a strawman.** If plain RAG is a naive dense top-k, the study is worthless. It is BM25 + exact-cosine dense + RRF with fixed weights, no router. Exact brute-force cosine in NumPy over a ~669 × 1024 matrix (2.7 MB) — no LanceDB, no FAISS, no HNSW. At this scale an ANN index buys nothing and costs reproducibility.

**A5 vendoring detail.** Patch `nano-graphrag`'s single `hierarchical_leiden` call from `graspologic` to **`graspologic-native`** (the Rust wheel Microsoft moved to). That one-file diff deletes the entire `gensim → numpy<1.20 → Python<3.13` chain — the three-virtualenvs story from tsfm-bakeoff, avoided this time, and worth its own section. **Pin and record the `graspologic-native` version in the artifact:** 1.3.x changes Leiden community counts and levels. Most GraphRAG write-ups do not do this and their graphs are therefore not reproducible.

**Ollama structured outputs (`format=<JSON schema>`) on every non-prose call.** Microsoft's own docs warn about malformed JSON from non-OpenAI models; constrained decoding turns a flaky-parse problem into a non-problem. This is the highest-leverage local-RAG trick in the study and it gets a paragraph.

**Declined, with reasons in the README:** DRIFT search (report as a variant, never a stand-in for Local); LightRAG (its 32k `num_ctx` requirement is a KV-cache requirement in disguise, and it skips Leiden and community reports entirely — it removes the exact mechanism global search wins with); HippoRAG 2 (NV-Embed-v2 at ~7.85B cannot co-reside with the generator); GraphRAG-SDK (FalkorDB container plus an unverified sovereignty question); R2R (Postgres + pgvector via Docker). Refusals with reasons read as rigour.

### 5.3 The fairness contract

Held **identical** across every arm, and each is a reported field in the run manifest:

1. **Corpus.** Byte-identical chunk file, SHA-256 committed.
2. **Chunking.** **[struck per §4.5.6, which governs.]** The original text tuned chunking on the dev split *for the plain arm* — a fairness-contract violation written into the fairness contract. One scheme, §4.5.6's sentence-aligned 600/1200, applied byte-identically to all arms; there is no per-arm chunk sweep.
3. **Embedder.** `qwen3-embedding:0.6b` (q8_0, 639 MB, Apache-2.0, 1024-dim, 32k context). Held fixed, and this is a fairness requirement, not a preference: *a weak embedder manufactures a win for the clever architectures*, and "agentic RAG helps because our retriever is bad" is the mirror image of corpus-rigging. Note that the counter-evidence paper got its clever-component-hurts result with BGE-small-en-v1.5, a weak 512-token retriever — a stronger retriever should widen that conclusion, not narrow it. **[corrected:]** The GGUF advertises 32k but the measured usable context is **4,096 tokens** (4,095 accepted, 4,096 rejected — §4.5.6), so chunk size is *not* a free parameter. A 512-token embedder would have forced ~1,170 chunks and nearly doubled the graph bill. **The embedder's `max_seq_len` sets the indexing budget** — a cost nobody writes down. Monoculture caveat disclosed: same family as the generator.
4. **Retrieval budget: fixed retrieved-token budget, k derived per size.** **[changed twice: first from k=10 to k=5 for prefill cost; then Stage 1 measured the confound §4.5.7 predicted. At fixed k=10, 1200-token chunks beat 600 by ~5.6 strict@10 points; at a fixed ~6,000-token budget (600@k10 vs 1200@k5) the sign flips and 600 wins by 12.2. Every chunk-size claim must state its budget convention. Generation runs use the token budget, with k derived per size; both conventions are reported for retrieval.]**
5. **Generation params.** `temperature=0`, fixed `seed`, `top_k=1`, `num_predict=48`, **`think:false` asserted in code on every call** — not a config default. Qwen3.6 with thinking on emits hundreds of reasoning tokens before the answer; leaving it on silently triples the study. Pinned model digest and quantization.
6. **Answer-format prompt.** One shared template.
7. **`keep_alive`** set long enough that no phase boundary triggers an 11.4 s reload.

**Context tokens are matched, not just k. [changed:** the design held "retrieval budget k" constant and matched tokens only on S5. Constant k is not constant budget: agentic at ~3 sub-questions × 2 iterations × k=5 sees up to 30 chunk slots against plain's 5 — up to 6× the evidence — on exactly the strata where it is predicted to win.**]** Rule, pre-registered: **on any stratum where A4 beats A3 by more than the measured noise floor, A3t is run at a k′ chosen so plain's mean context tokens equal agentic's measured mean on that stratum.** Accuracy-per-context-token is reported on **all** strata, not only S5. This decides whether "agentic reasoning helps" or "agentic read more," and it is the cheapest experiment in the design.

**Tuning budget is a reported number. [changed:** plain got a chunk sweep and the graph got nothing, which is the untuned-baseline sin in reverse.**]** Publish per arm: *plain: 5 chunk configs + a `k1`/`b` sweep on the same dev split; BM25: same sweep* **[changed:** the design gave the dense leg the strongest available embedder and left the sparse leg's `k1`/`b`, stemmer and stopword list unspecified — while predicting BM25 would win the headline**]**; *graph: 1 config.* The graph gets exactly one knob — `max_gleanings` 0 vs 1, selected on a 100-chunk pilot by **gold-entity coverage**, not end-task accuracy, so it cannot be accused of tuning on the test. If the budget forbids even that, the graph result is labelled **LOWER BOUND in the headline**, not in a footnote.

**One retrieval metric all arms can be scored on. [changed:** the design's own rule — never put a community summary in an nDCG table — meant the cheapest, best-powered comparison in the study excluded the graph arm entirely, leaving it measurable only end-to-end where it also absorbs generation noise. Unequal measurement surface.**]** Define a **common projection**: every arm ultimately places source text in the prompt, so map each arm's assembled context back to the set of **source chunk IDs actually present in the prompt** (nano-graphrag's local search returns source text units; community reports carry constituent chunk ids). Primary retrieval metric for all arms: **supporting-chunk recall @ matched context tokens.** nDCG@10 is demoted to a chunk-arms-only appendix.

**Frozen evidence.** Retrieved context is written to disk for every arm before generation runs. This attributes retrieval loss separately from generation loss and makes the whole study re-scorable without re-running the generator.

---

## 6. The self-learning layer

**[changed: renamed. It is "retrieval-feedback re-ranking" everywhere in the repo and the article until an N2 result exists.]** The word *self-learning* appears only in the section that explains why it is not being used.

### 6.1 The pre-flight gate — run before committing to this axis at all

Pure numpy, under an hour, zero LLM calls, at **chunk** level. Ported from `scratchpad/selflearn_leak.py` and `scratchpad/popprior.py`, which computed the document-level versions **[measured, tech+business, 1,381 queries]**:

| Quantity | Document level |
|---|---|
| Distinct gold documents | 232 |
| Mean queries per gold doc | 16.9 (median 3, **max 495**) |
| Share of gold citations in the top 10% of docs | **70.4%** |
| Held-out queries with **every** gold doc already credited (random 50/50 stream, 5 seeds) | **91.9%** (89.1–94.2) |
| Held-out queries with **fully disjoint** gold docs — the only honest novel set | **3.0%** (~20 of 691) |
| Greedy maximum unit-disjoint query set | **79–82 queries** |
| **Query-blind popularity retriever** (ranks by training-half gold frequency, uses no query information) | **Hit@10 65.1%, Recall@10 43.5%** |

Three articles are gold for 495, 376 and 273 queries — 29% of all gold citations from three documents. That is a property of how MultiHop-RAG was built, not of the retrieval problem.

**Gate rule, pre-registered:** if the chunk-level query-blind popularity baseline captures most of the oracle-boost headroom, or if the unit-disjoint set stays under ~150 queries, **the axis is reported as a bound, not a demonstration**, and the README wording is fixed before a single GPU-second is spent.

### 6.2 Mechanism

Applied as a **dimension crossed with all three architectures**, not a fourth arm — every arm retrieves units, so unit-level score-side memory is the only candidate applicable identically to plain, agentic and graph. State is keyed on `unit_id`, never `chunk_id`, because GraphRAG's retrievable units are communities and relations and one shared schema is what makes the comparison fair.

**[changed: the query is put back into the boost, and the two forms are separate levels.]** The design's formula had **no query term** — the counters were global, so it improved every query identically and could not, even in principle, "get better every time a *similar* query is asked." The research brief's `(query_cluster_id, unit_id) → credit` affinity table had been dropped.

- **L0** — off. Reuses the main run at checkpoint 0. Free.
- **L1g** — **global prior**: `final = α·retrieval_score + β·log((1 + credited_u)/(1 + surfaced_u))`
- **L1c** — **cluster-conditional**: the same, with counters keyed on `(query_cluster_id, unit_id)`. Clusters by leader/greedy clustering of query embeddings at a fixed cosine radius; deterministic given seed and stream order, and stream order is permuted across 5 seeds.
- **L2** — semantic cache. The control.

Credits are weighted by `1/p(rank)` (inverse-propensity correction for position bias) with ε-greedy exploration at ε=0.1. Ablate with IPW and ε both off; the resulting rich-get-richer curve is a figure.

**The difference between L1g and L1c is the difference between "learned the corpus" and "learned the query neighbourhood."** Given the measured popularity concentration, the global prior will eat most of the apparent gain. Reporting only L1c and attributing its gain to similarity would be the dishonest version.

**Credit signal:** gold-unit-in-top-k during the learning stream — free, deterministic, and **disclosed as an upper bound** on what a deployed system gets.

### 6.3 Novelty split and controls

Three levels, not two:

- **R** — paraphrase repeats.
- **N1** — novel queries whose gold units overlap the learning stream (~97% of held-out).
- **N2** — novel queries with **fully disjoint** gold units. **~79 queries, and this is the only set that licenses the word "learning."**

Four destructive controls, all pure numpy, all cheap, and without them no gain is interpretable:

1. **Query-blind popularity control** — the 65.1% Hit@10 baseline, run as an arm. If L1 does not beat it, there is no learning, full stop.
2. **Shuffled-credit control** — permute credits across units preserving marginals.
3. **Random-boost control** — same boost magnitude, random units. Separates information from the entropy effect of perturbing a ranker.
4. **Oracle ceiling** — boost = ∞ on gold units. Run as a **pre-flight gate**: if the oracle buys only a few points on N2, no mechanism can buy more.

### 6.4 The nulls, and the power that is actually available

Pre-registered, one-sided, MDE stated in advance:

- **H0-transfer:** nDCG@10(L1) = nDCG@10(L0) on N2. The only test that licenses "learning."
- **H0-conditioning:** L1c = L1g on N1. Fail to reject ⇒ it is a corpus prior.
- **H0-cache:** L2 accuracy on novel ≤ L0, and within ±2 points by TOST.
- **H0-repeat:** L1 = L0 on paraphrases. Expected rejection; smoke test, not finding.

| n | π_d | MDE @ α=.05 | MDE @ α=.05/9 |
|---|---|---|---|
| **79** | 0.15 | **12.0 pts** | **15.4 pts** |
| 150 | 0.15 | 8.8 | 11.3 |

GAM-RAG's headline is +3.95% at 0 turns, +8.19% after 5. **At n=79 with a nine-test family you cannot detect 15 points.** The axis is underpowered by roughly 2× against the most optimistic literature effect. The honest framing, and it is in the tsfm-bakeoff register: *on a leak-controlled novel set, whatever score-side memory buys is smaller than 15 points, and we cannot rule out zero.* The README says the bound was the **pre-registered outcome, not a retreat**.

### 6.5 The cache control, corrected

**[changed:** the design pre-registered "exactly 0.0 accuracy win on the novel stratum." Wrong sign of bound. A false cache hit returns a stale answer; GPTCache at τ=0.99 drifts to ~1.7% error by 150k samples.**]** Correct pre-registration: **≤ 0**, bounded below by the false-hit rate at τ, with the τ sweep published.

**[changed:** the "~500×" latency figure was assembled from two literature numbers. It must be **measured** p50/p95 end-to-end on this machine with the embedder resident, including the query-embedding call.**]**

Two things must be measured before the repeat stratum carries any claim: **L0 accuracy on paraphrases vs originals** (if equal, the stratum has zero headroom and both L1's gain and L2's hit rate are measuring the paraphraser's laziness), and **the paraphrase↔original cosine distribution** (if τ sits below its mass, the hit rate is self-similarity within one model family, not query repetition in the wild). Paraphrases are generated by qwen and retrieved by a qwen-family embedder; entity-dense MultiHop-RAG queries will survive paraphrasing with their proper nouns intact and BM25 will nail them with no memory at all. Ship a **low-cosine adversarial paraphrase subset** (entity-substituted, abstracted, elliptical) or the cache-vs-learning table demonstrates a tautology.

### 6.6 Judge-noise sensitivity — simulated, not run

**[changed:** the design asserted "THE CEILING OF THE SELF-LEARNING LOOP IS THE JUDGE'S COHEN'S KAPPA." Deleted. The committed artifact reports `agreement: 30, false_pass: 0, false_fail: 0` — accuracy 1.000, κ = 1.00, Wilson 95% [0.886, 1.000]. On current evidence the claim reduces to "the ceiling is 1.0," and the exam is *saturated*, which means it is too easy for the task, not that the judge is perfect. It is also the wrong formula: κ×Δ attenuation governs a binary outcome contrast, not the signal-to-noise of a ranking counter that averages over rounds. And the real-judge ablation as scoped was 3 arms × 300 queries × 5 rounds × 5.9 s ≈ 7.4 h against a 2.5 h budget line — unfunded.**]**

Replacement: **simulate the credit-noise sweep.** Corrupt the gold credit signal across κ ∈ [0.6, 1.0], both symmetric **and rank-correlated**, 5 seeds, pure numpy, minutes. The dangerous case the design never named is the rank-correlated one: when judge errors correlate with retrieval score, credit becomes partly a function of rank, IPW does not correct it (IPW corrects position propensity, not content bias), and you get rich-get-richer on a *biased* signal — strictly worse than noise. Produces a curve instead of an anecdote, bit-exactly, for free.

### 6.7 Write-back — cited, not run

Five things appear verbatim in the README:

1. **"Statistically indistinguishable" is an equivalence claim and gets an equivalence test.** 58.6% vs 59.0% across 5 seeds is a *failed difference test* until a TOST with a stated margin and the CI on the difference are published. The study insists on TOST for S1; the same standard applies to its own headline.
2. **The benign/adversarial labels were assigned by the experimenter, not detected by the system.** The finding is that *the provenance statistic people propose as a monitor takes the same value in both regimes*. State the threat model — injected attacker text and compounding self-error are different papers.
3. **Separate the transferable claim from the setup-specific one.** *Exposure amplification 1.27× (1.16–1.65) under a lexical similarity function* is the mechanism. *90.3% → 99.0% / 71.0%* is MS MARCO, BM25, 20 rounds, simulated write-back. They do not travel together into a news-corpus dense-retrieval article.
4. **Say plainly that write-back was not run here.**
5. **Commit the replay artifact.** A grep for `exposure.?amplif|retrieved.?synthetic|58\\.6|1\\.27` across both working directories returns zero hits. For a repo whose ethos is "every README number asserted against a committed artifact by CI," the load-bearing citation of this whole section currently rests on prose.

One cheap **new** measurement is bought: re-run exposure amplification under `qwen3-embedding:0.6b`. The 1.27× was lexical; under a dense embedder the mechanism is semantic and the factor could differ. No generation, no judge, and it connects the BM25 result to the dense pipeline.

### 6.8 What may honestly be claimed

- Score-side memory **cannot modify the corpus**. **[changed:** the design said it "structurally cannot poison the evidence base." True of the corpus, false of the system: a biased boost table can suppress gold units permanently and reproduce the exact metric-up-system-worse pattern one layer up. It is tested by the IPW-off / ε-off ablation.**]**
- If N2 shows nothing, **the honest headline is that the most defensible learning mechanism available did not transfer** — a better article than a marginal positive.
- **The state is a persisted query log.** That is a data-governance fact a bank will ask about. Volunteering it is worth more than being asked.

### 6.9 Query pool double-booking

**[changed:** the design allocated n=300 to S3, leaving **19** of 319 temporal queries, while separately requiring a chunk-tuning dev split, a learning stream, a learning eval set, and a paraphrase pool from the same leftovers.**]** Resolution, stated in the README before anything runs: the learning stream and its eval set are drawn from the **dev columns in §4.4 plus the unallocated remainder**, never from the eval sets. Where the pools cannot cover it, the learning stream is drawn from S4 (371 spare) and S6 (151 spare), and this composition is disclosed as a limitation on the mix-shift test.

---

## 7. Metrics and statistics

**Design: fully paired.** Every arm sees every question. For a 5-point gap, an unpaired two-proportion test needs 1,565 per arm; McNemar at π_d = 0.10 needs 312 total. Same conclusion, 5× less compute.

**Tier 1, judge-free, and there is no Tier 2.** EM, token F1, answer containment, with the committed gold normalizer. S6 additionally: abstention accuracy, false-answer rate, risk-coverage curve. S5a: numeric match (counting, extremum), set-F1 (top-k), Kendall τ (sorting). S5b: set-F1 against the independent NER key.

Note that with **median gold = 3 characters and 79.6% single-token [measured]**, EM and token-F1 are an extraction test and they disagree systematically with containment as a function of answer length. Arms that synthesise are penalised by EM and rewarded by containment. **Mean answer length per arm per stratum is a reported column** so the reader can see the confound, and the primary metric per stratum is declared in advance.

**Retrieval:** **[changed after E4/Stage 1:]** primary is **strict@k — every gold document retrieved** — because Hits@10 is saturated on this corpus (96.5–97.0% across six configs, 0.59-point spread [measured]) while strict@10 spreads 30.1–43.1 over the same grid. Supporting-chunk recall @ matched context tokens stays as the cross-arm cost-normalised view; Hits@1 and MRR@10 secondary; Document-F1@k on S5.

**Faithfulness:** HHEM-2.1-open or MiniCheck as a local classifier — deterministic, sub-1 GB, coexists with the resident generator. RAGAS-style LLM metrics are not used as ground truth anywhere.

**Statistics:** McNemar exact (mid-p) on binary correctness; Wilcoxon signed-rank on continuous scores; paired bootstrap at 10,000 resamples for every CI; **Holm–Bonferroni across a pre-registered family** with one primary metric per stratum. π_d is estimated from the Stage 1 pilot before n is finalised — it is the parameter that decides feasibility.

**Equivalence, not non-significance.** S1 gets a pre-registered **TOST at ±7 points**. "The three architectures are equivalent within 7 points on single-fact lookup" is stronger and more defensible than a non-significant difference, and it is the claim that kills GraphRAG's business case for most traffic.

**Two curves that convert unfixable limitations into findings:**
- **Corpus-size curve.** Retrieval-only recall@10 per arm-leg at 253 / 609 / distractor-padded article counts. Minutes of compute. Tells a reader with 10M documents which side of the crossover they are on. The corpus is 253 articles because of a 24 GB machine, not because of science, and k=5 is 0.75% of it — that points the same direction as the predicted headline and must be bounded.
- **Community count and Leiden depth** from the 100-chunk pilot, published as a first-class number, with a **pre-registered floor: if levels ≤ 2, the S5 result is reported as a lower bound on global search, not as its performance.** 669 chunks is the smallest corpus at which global search can technically run.

**Refuse to report any per-stratum difference smaller than the measured noise floor** (§8).

**The crossover is published as a function**, not a sentence — over (traffic mix × query volume × re-index cadence), under **both raw and lift**. "Aggregative traffic must clear ~12%" is a function of an invented traffic prior and a chosen amortisation denominator and is the one line a critic will quote back.

**Per-arm verification statement, published.** **[changed:** BM25 is verified against brute force; GraphRAG gets a 50-chunk cross-check against Microsoft GraphRAG; the hand-rolled agentic arm currently gets nothing, so any loss it takes is attributable to implementation rather than architecture.**]** Add a 30-item human audit of decompositions and of graph extractions with an error taxonomy. Any unvalidated arm is labelled **implementation-bounded** in the results table.

---

## 8. Cost reporting — a headline result, not an appendix

Reported per arm per stratum, in the **same table as accuracy**, because reporting accuracy without write-path cost lets a 50×-more-expensive system claim a tie as a win:

index wall-clock · index bytes on disk · peak RSS as a fraction of 24 GB · LLM calls per query · prompt + completion tokens per query · **context tokens per query** · p50/p95 end-to-end latency · **accuracy per context token** · **zero cloud GPU-hours, zero API spend** **[changed:** "zero GPU-hours" was wrong — the M5 does Metal GPU inference. A skeptic who catches that discounts the other forty numbers.**]**

Prior art to beat, full MultiHop-RAG: construction 135 s (RAG) vs 7,702 s (KG-GraphRAG) — a **57× indexing penalty**.

**Index amortisation is reported as a curve, not a constant. [changed:** a rolling market-intelligence desk over a news archive is a corpus that changes daily, which is the graph's *worst* amortisation case, while a static contract corpus is its best. The corpus choice silently sets the denominator, and the denominator is the crossover point.**]** Report index cost per query at three query volumes × two cadences (one-shot / weekly rebuild), and **measure nano-graphrag's incremental-insert path once** to publish the per-new-article marginal index cost.

**Diagnostics that turn black-box losses into explained ones:** gold-entity coverage of the extracted graph (prior art found only **65.8%** of answer entities present in the constructed KG — that one number is the mechanism behind most GraphRAG underperformance); the closed-book leakage floor per stratum; question↔gold-chunk lexical overlap as a covariate on the two authored strata.

---

## 9. Functional requirements

**Infrastructure**
- **FR-1** No component makes a network call at benchmark time. CI asserts an empty egress allowlist.
- **FR-2** `think:false` is asserted in code on every generation call and the harness **fails loudly** if the response contains reasoning tokens.
- **FR-3** The corpus and all queries (originals, stripped, paraphrased) are embedded in **one pass with the generator unloaded**, persisted to disk. The embedder never co-resides during the query loop, and the 11.4 s reload tax never enters it.

**Corpus and questions**
- **FR-4** Ship an outlet-stripped variant of every native query; commit both files; score every stratum on both.
- **FR-5** Publish per-stratum retention rates into the subset and the cross-category evidence covariate.
- **FR-6** Ship the majority-class arm and the gold-casing normalizer; refuse any stratum win that does not clear the constant.
- **FR-7** Report raw accuracy and closed-book lift for every cell; make the crossover slider recomputable under both.
- **FR-8** Publish pool sizes next to eval n for every stratum, and state that S1 and S2–S4 are not independent samples.
- **FR-9** Ship the paraphrased S1 variant and stratify S1 by question↔chunk overlap quartile.
- **FR-10** Build S5b's answer key with an independent, committed NER pass. CI asserts the key was not produced by the generator.
- **FR-11** Sample hard nulls for S6 and report the easy/hard split and the risk-coverage curve.

**Arms and fairness**
- **FR-12** Enforce k=5, 1024/128 chunking, `num_predict=48`, `temperature=0`, fixed seed, pinned digest — read from one config, asserted per call.
- **FR-13** Freeze retrieved evidence to disk for every arm before generation.
- **FR-14** Compute supporting-chunk recall @ matched context tokens for all arms via the common source-chunk-id projection.
- **FR-15** Run A3t at token-matched k′ on any stratum where A4 beats A3 by more than the noise floor; report accuracy-per-context-token everywhere.
- **FR-16** Publish the per-arm tuning budget, the plain-arm chunk sweep, and the BM25 `k1`/`b` sweep.
- **FR-17** Vendor `nano-graphrag` into `third_party/`, patch to `graspologic-native`, pin and record the version in the run artifact.
- **FR-18** Log LLM calls per query as a first-class reported column for every arm. **[changed:** the design stated "measured mean 3.2 LLM calls/query" for an arm that does not exist yet — the repo contains three files. A fabricated "measured" in a document whose credibility rests on measurement-versus-projection discredits everything else on the page. The number is removed until the counter produces it.**]**
- **FR-19** Cap agentic at 2 retrieval iterations, and **measure the 1/2/3 iteration curve on the dev split for this corpus** rather than importing the cap from a different model on a different corpus.

**Learning layer**
- **FR-20** Run the chunk-level pre-flight gate (overlap, query-blind popularity, oracle headroom, N2 size) and publish all four numbers before any GPU-second is spent on this axis.
- **FR-21** Implement L1g and L1c as separate levels; report both.
- **FR-22** Implement all four destructive controls.
- **FR-23** Score the learning primary at the retrieval layer (recall@10 / nDCG@10 / MRR), 5 seeds, permuted stream order. Generation at one final checkpoint only.
- **FR-24** Simulate the credit-noise sweep across κ ∈ [0.6, 1.0], symmetric and rank-correlated.
- **FR-25** Report the paraphrase↔original cosine distribution and ship the low-cosine adversarial subset.
- **FR-26** Measure cache latency p50/p95 end-to-end on this machine, embedder resident, query-embedding call included.
- **FR-27** Run the mid-stream mix shift at round R/2 (single-fact-dominant → multi-hop-dominant) and report the staleness penalty.

**Reporting**
- **FR-28** No unlabelled aggregate number is emitted by any code path. The reporter refuses to print one.
- **FR-29** Every stratum table carries n, pool size, MDE, and whether the result is *resolved* or *bounded*.
- **FR-30** Publish the run-to-run noise floor and suppress any difference below it.

---

## 10. Reproducibility contract

- **Seeds** fixed and recorded for: chunk assignment, stream permutation (5 seeds), ε-greedy exploration, bootstrap resampling, query cluster assignment.
- **Pinned:** Python version, `graspologic-native` version, `nano-graphrag` vendored commit, Ollama version, model **digests** for generator and embedder, quantization.
- **Committed artifacts:** chunk file + SHA-256; **the embedding matrix itself** **[changed:** "bit-exact retrieval" is conditional on the vectors. Metal float reductions are not guaranteed bit-identical across runs, so committing the code is not enough — the matrix is committed and regeneration is verified to a stated precision.**]**; the stripped-query file; the paraphrase file; the S1 paraphrase variant; the S5a pandas gold; the S5b NER key; the Leiden graph with its community count and depth; the frozen retrieved evidence; the contamination-replay artifact from §6.7.
- **CI asserts:** every number in the README against a committed artifact; the egress allowlist is empty; `think:false`; the S5b key's provenance; that no unlabelled aggregate appears in any generated report. Exit codes 0/1/2, per `agent-report-card`'s contract.
- **Honest boundary, stated in the README:** the retrieval layer is bit-exact given the committed matrix; the generation layer is not, and is reported as N=3 repeats with a measured variance.

---

## 11. Wall-clock budget

All 8B-class figures are **projected** from the measured 27B constants and a projected 20 tok/s decode / 350 tok/s prefill. Stage 0 replaces them with measurements before anything else runs.

| Stage | Hours |
|---|---|
| Stage 0 — throughput measurement, memory wall, 20-chunk extraction pilot | 3.0 |
| Corpus + all query variants embedded in one pass | 0.5 |
| **GraphRAG index** — 669 chunks: extraction ~10.0 h, entity summarisation ~3.3 h, community reports ~2.0 h | **16.0** |
| Core generation grid — 980 questions × 5 arms (A0 3 s, A2 18 s, A3 18 s, A4 45 s, A5 20 s) | 25.4 |
| S5a + S5b global arms (A6, A6b), map capped at 3 community batches | 5.8 |
| Stripped-variant generation, S1 + S3, arms A2 + A3 | 4.3 |
| Agentic ablations A4a / A4b on a 150-question slice | 3.8 |
| Learning layer — retrieval-only primary (0.5 h) + one generation checkpoint (3.3 h) | 3.8 |
| Paraphrase artifact — 5 per call | 1.6 |
| Noise floor — A3 repeated ×2 on a 150-question spine | 1.5 |
| A-27B sanity arm, S3, n=60 | 4.6 |
| Contingency and re-runs, 15% | 10.5 |
| **Total** | **~81 h** |
| With 15% sustained-thermal derate | **~93 h ≈ 12 overnight runs** |

**Cuts already applied**, and what each bought:

| Cut | Saved |
|---|---|
| 27B → 8B-class generator | the study (2,617 h → ~81 h) |
| k=10 → k=5 | ~45% of the grid; prefill is 85% of a query |
| `num_predict` → 48 | ~19 s/query of decoding tokens nobody reads |
| Judge deleted; S5 split into S5a/S5b | ~2.5 h + the calibration exam + judge attenuation on five strata |
| A1 and A7 deleted | ~73 h |
| Learning primary moved to retrieval-only | 96 h (1 seed) / 480 h (5 seeds) → ~4 h |
| Judge-noise ablation simulated | 7.4 h unfunded → minutes |
| S2, S4 declared bounded; n cut | ~9 h |
| Paraphrases 5-per-call | 46 h → 1.6 h |
| Global map capped at 3 batches | ~8 h |

**What the original 41.6 h estimate actually bought at measured 27B speed: 206 questions total across six strata, 34 per stratum, MDE 18.6 points** — exactly enough power to detect the one effect that was already known.

**Two things to try in the first hour, both his call:** `sudo sysctl iogpu.wired_limit_mb=20480` and re-read the CPU/GPU split (11% → 0% CPU is free throughput; real OOM risk if set too high), and confirm with `ollama ps` that generator and embedder are resident simultaneously before pre-embedding.

---

## 12. Risks and what would invalidate the study

| Risk | Detection | Response |
|---|---|---|
| **Stage 0 throughput comes in below projection** | Stage 0, hour 3 | Drop to technology-only (430 chunks). **[changed:]** the per-chunk 90 s gate is replaced by a **corpus-hour budget: the 600 index must land in ≤ 30 h** — a mean-evaluated per-chunk gate is not a gate when 4.8% of calls truncate outright (E0) and per-chunk seconds vary 21–643 s (E2/E3). If the budget is blown, the graph arm drops to a 300-chunk subset and the S5 result is labelled a lower bound |
| **Leiden depth ≤ 2** | 100-chunk pilot | Pre-registered: global search reported as a lower bound, not as performance |
| **BM25's win disappears under outlet stripping** | Retrieval-only, Stage 1 | The headline changes and improves; the delta becomes the finding |
| **N2 set is too small or oracle headroom is near zero** | Pre-flight gate, §6.1 | Learning axis is re-scoped to a bound before any GPU time; README wording fixed in advance |
| **L1c does not beat L1g** | Learning run | Reported: it is a corpus prior, not similarity learning |
| **Every stratum difference falls under the noise floor** | Noise-floor run | Publish the floor and the bounds. This is a valid outcome |
| **`format=<schema>` extraction fails on 8B** | Stage 0, 20 chunks | If not 20/20 valid, GraphRAG is not in the study and the README says why |
| **Model swapping in the query loop** | `ollama ps` during Stage 1 | Pre-embedding (FR-3) removes it; if it recurs, fall back to `nomic-embed-text` + `embeddinggemma:300m-qat-q4_0` and the ~8-point retrieval gap becomes the sensitivity story |
| **Thinking tokens left on** | FR-2 assertion | Fails loudly. Silently triples the study otherwise |
| **Embedder sensitivity flips a stratum ordering** | Pre-registered retrieval-only alternate pass (S1 + S4, ~10–15 min, graph reused verbatim because extraction is embedder-independent) | Escalate to a full re-run only on a flip. MemDelta measured a +6.2 pp swing from an embedder change alone that flipped a headline conclusion |

**What would invalidate it outright:** if the outlet-stripped delta shows that retrieval on this corpus is mostly publisher-name matching *and* the stripped questions become unanswerable by every arm, the corpus cannot support the study and it moves to the CUAD fallback. That decision is made at the end of Stage 1, on retrieval-only evidence, at a cost of hours rather than nights.

---

## 13. Staging

**Stage 0 — measure the machine (3 h). Checkpoint: continue only if all four pass.**
Measure decode and prefill on the 8B candidate; `ollama ps` at 4k/8k/16k; 20 real chunks through the real extraction prompt with `format=<schema>` (need 20/20 valid payloads and a per-chunk second under 90 s); confirm generator + embedder co-residency. **Commit the measurements to the repo before writing benchmark code.** This measurement is itself publishable — nobody reports the per-chunk second for local GraphRAG extraction, and *"Microsoft indexed ~1M tokens in 281 minutes on GPT-4-turbo; this box needs 32 hours for 669 chunks on a 27B and 10 on an 8B"* is a better paragraph than anything in the plan it replaces.

**Stage 1 — corpus, chunks, embeddings, BM25, and every retrieval-only result (6 h). Checkpoint: the whole study can be abandoned here for the cost of one evening.**
Build the chunk file, all query variants, the embedding matrix. Run the pre-flight leakage gate (§6.1). Run all retrieval-only tables: as-published vs stripped, the corpus-size curve, recall@10 per arm-leg, the query-blind popularity baseline, the oracle headroom. **Zero LLM calls.** If the stripped delta guts the corpus, or N2 is under 150, or oracle headroom on N2 is negligible — re-scope now, in writing, before spending a night on generation.

**Stage 2 — plain and BM25 arms end to end, plus the noise floor (8 h, one night).** Establishes π_d, the noise floor, and whether n is right. Finalise sample sizes against the measured discordance rate.

**Stage 3 — GraphRAG index (16 h, two nights).** 100-chunk pilot first: community count, Leiden depth, gold-entity coverage, `max_gleanings` selection. Then the full index. Cross-validate the vendored graph against Microsoft GraphRAG on 50 chunks, exactly as BM25 was validated against brute force. **Checkpoint: if depth ≤ 2, S5 is relabelled a lower bound before the S5 arms run.**

**Stage 4 — agentic and graph arms across the grid (30 h, four nights).** Then the ablations and the 27B sanity arm.

**Stage 5 — learning layer (4 h).** Retrieval-only primary, five seeds, four controls, the simulated κ sweep, one generation checkpoint, the mix shift.

**Stage 6 — S5a/S5b and the crossover (8 h, one night).** Structured-query baseline first; if it wins S5a outright, that is the finding and it is written up as one.

**Stage 7 — report, article, CI assertions.** The reporter emits the per-stratum tables, both raw and lift, the cost table, the amortisation curve, the corpus-size curve, and the crossover as a function. It refuses to emit a single unlabelled number.

---

**The predicted headline, pre-registered here so it cannot be discovered after the fact:** single-fact lookup is most real traffic, BM25 wins it outright, and the graph does not repay a 16-hour index until aggregative traffic clears a low-double-digit share of the mix. If the outlet-stripped variant kills that, the finding is better. If the learning axis produces nothing on N2, the bound is published as the pre-registered outcome. If the machine cannot index the graph, the study says so and reports five arms instead of seven.

---

---

*Section 14 (a summary drafted from the first nine agents) was replaced by section 15 below when the workflow's synthesis landed. Same verdict, specified to build level, and it corrects one claim the summary made — see 15.5.*

"`[changed:]` **This section supersedes the §14 summary drafted 8 Aug in `/Users/net/Documents/001-personal-projects/sovereign-rag/PRD.md`. Same verdict, specified to build level. Delete §14 when merging.** Three adversarial critiques (scope/opportunity-cost, sovereignty integrity, dilution/honesty) were run against three competing strategies (adopt Open WebUI, purpose-built bench, no UI). All three critiques rejected adopting Open WebUI as a shipped component. They disagreed on shipped scope by a factor of five; that disagreement is resolved in 15.12 and 15.13 rather than averaged away.

---

## 15. The interaction layer

### 15.1 Verdict

**Thin build. One self-contained HTML file, one ~150-line OpenAI-compatible adapter, and one tested integration page — no chat interface is written, and none is shipped.** Open WebUI is adopted as a *documented deployment target*, not as a component: the production-shaped answer to "think about the front end" is a stable `Arm.answer() -> Envelope` contract with an OpenAI-compatible surface over it, pointed at whatever client the buyer already runs.

Three findings decide this and each is arithmetic, not taste.

**The front end cannot be the sensor for the learning loop.** §6.2 weights credits by `1/p(rank)` with ε-greedy exploration at ε=0.1. A human thumb cannot supply a logged propensity. Worse, answer-level feedback is one bit spread across k=5 units, conditioned on exposure, which is mechanically an increment of a "how often was this shown" counter — and that counter *is* §6.3's destructive control #1, the query-blind popularity retriever measured at **Hit@10 65.1%**. A UI-fed loop reproduces the control the study exists to beat. Volume finishes the argument: production rating rates run 0.6–2.6% of turns; §6.4 sizes N2 at ~79 judgments; at 3% that is ~2,600 chat turns, roughly 130 evenings, against the **0.5 h** FR-23 budgets for the retrieval-only primary.

**A live three-arm comparison takes 83 seconds.** §11's own per-query grid: A3 18 s, A4 45 s, A5 20 s, sequential because there is one Ollama instance and 24 GB. Nobody demos that, and no GIF of it travels. `[changed:` the purpose-built-bench proposal costed the same fan-out at ~47 s. That contradicts §11 by 1.8×. Corrected upward.`]`

**A chat surface is a different memory configuration from the benchmarked one.** FR-3 embeds the corpus and all queries in one pass with the generator unloaded, and states the embedder never co-resides during the query loop. A free-text box requires live query embedding with the embedder resident. Any latency or RSS observed through a UI is therefore not a number from this study and may not be reported as one.

What survives is the artifact §3 non-goal 2 and §7 already oblige: **the weight slider that shows the winner changing.** It is the finding, it responds in 0 ms, it needs no GPU, and it opens from `file://` on a bank's laptop with the stack switched off.

**Committed: 10.5 h ≈ 4–5 evenings**, of which 7.0 h is gated behind Stage 1's abandon checkpoint and 3.5 h behind Stage 7. Against ~7 evenings to wire Open WebUI properly and 12–13 to build a bespoke bench.

---

### 15.2 What the user sees

**Default surface — `reports/crossover.html`, opened from disk or from an `<iframe>` on satsawat.ai.** No server, no inference, no network.

*Pane 1, CROSSOVER.* Six stratum rows, S1–S6. Each row carries one horizontal bar per arm at that stratum's declared §4.4 primary metric, with a paired-bootstrap CI whisker. Two vertical rules cross every row: **AM**, the majority-class constant (60.1% on comparison, 46.7% on temporal — §4.3c), and **A0**, the closed-book leakage floor (§4.3d). A bar that does not clear the AM rule is drawn grey and captioned *does not clear the majority-class constant — not reported as a win* (FR-6). Any inter-arm delta whose committed `reportable` flag is false is greyed with the literal string *below measured noise floor — not reported* (FR-30).

Beneath it, six weight inputs summing to 100, with presets (`MultiHop-RAG native`, `enterprise search`, `analyst desk`). Dragging one recomputes the single explicitly-labelled weighted aggregate live, and **the winner's name changes as you drag**. Two toggles, both mandatory: `raw / lift-over-closed-book` (FR-7) and `as-published / outlet-stripped` (FR-4). All four combinations addressable. Per §7, the article does not get to pick one.

Below that, the §8 cost row per arm — index wall-clock, index bytes, peak RSS as a fraction of 24 GB, LLM calls per query, context tokens per query, p50/p95, accuracy per context token — and one sentence recomputed at the current mix: *at this mix, A5 costs 47× A3 per point of accuracy.*

Permanent badges where the PRD pre-registers them: `S2, S4 — BOUNDED ±11, cannot rule out zero`; `S5 — 1 config; LOWER BOUND if Leiden depth ≤ 2`; `A4 — implementation-bounded (§7)`.

*Pane 2, MANIFEST.* One table from the run manifest: generator digest and quantization, embedder digest, `graspologic-native` version, vendored `nano-graphrag` commit, corpus SHA-256, chunking parameters, index timestamps, every licence, and the egress test result with its timestamp and attempt count.

**Conditional surface — REPLAY.** Only if the trigger in 15.13 fires. Three columns rendered from committed run artifacts (FR-13 already freezes retrieved evidence to disk), so it is instant, reproducible by a reader with no GPU, and *is* the benchmarked configuration rather than an approximation of it. A question picker over the eval sets, each question labelled with its stratum, its provenance (`native — third-party authored` vs `authored in-repo`, per §4.4), and that stratum's AM constant.

**What the user does not see: a text box.** There is no free-text query input in v1. Live inference, if it is ever built, is a secondary mode behind a permanent banner and is covered by FR-U14.

---

### 15.3 What is adopted vs written

| Path | Adopted / written | Licence | Cost |
|---|---|---|---|
| `reports/crossover.html` | **written** — single file, no CDN, no npm, no build step, no webfont, system font stack | his own | 3.5 h |
| `reports/results.json` | **written** — emitted by the existing reporter; carries `reportable` per comparison | his own | in the reporter anyway |
| `src/sovereign_rag/arms/base.py` — `Arm.answer() -> Envelope` | **written** | his own | in the harness anyway |
| `src/sovereign_rag/trace.py` | **written** — one append-only row per answer; serves §8's cost columns, FR-13, FR-18's call counter, FR-30's noise floor | his own | in the harness anyway |
| `src/sovereign_rag/serve.py` | **written** — ~150 lines, OpenAI-compatible | his own | 1.5 h · Stage 7 |
| `ops/egress/` — pf anchor, `tcpdump` capture, logging-proxy runner, `just egress-test` | **written** | his own | 1.5 h |
| `ops/manifest.py` + `just verify-models` | **written** — digests and re-hashing | his own | 2.0 h |
| `docs/integrations/open-webui.md` | **written** — one tested page, one screenshot, the kill-list, the traps | his own | 1.0 h · Stage 7 |
| FastAPI + uvicorn | adopted | MIT / BSD-3 | — |
| stdlib `sqlite3`, stdlib `StreamingResponse`, browser `EventSource`, CSS Grid, inline `<svg>` | adopted | PSF / browser | — |
| Ollama (generator + embedder host) | adopted — **CLI binary, not the macOS `.app`** (15.7) | MIT | — |
| `agent-report-card` | adopted — harness, exit codes 0/1/2, CI number-assertion | his own | — |
| Open WebUI v0.11.0 | **documented target only — not a dependency, not in `pyproject.toml`, not in CI** | "Open WebUI License" — BSD-3 + branding clause, **not OSI-approved**, CLA required to contribute | — |

Zero npm packages. Zero `node_modules`. Zero containers on the machine that runs the study. `grep -rnE "https?://" reports/ src/sovereign_rag/serve/` returning nothing but comment URLs is one line of the egress audit.

The 30-item blind audit of decompositions and graph extractions (§7) is a shuffled CSV opened in Numbers. Fifteen minutes. Not an application.

---

### 15.4 The backend contract

**One object, three surfaces.** The harness imports `Arm.answer()` in-process — no HTTP on the benchmark path, which is what keeps FR-1 clean. The CLI and the adapter are thin wrappers over the same function. Demo and study cannot drift because there is nothing to drift.

```
Envelope:
  answer, abstained, arm_id, level_id, config_sha, model_digest, corpus_sha, trace_id
  citations: [{unit_id, unit_kind, doc_id, doc_version, chunk_ids[], rank, score,
               retriever_leg, used_in_answer: bool}]
  cost:      {llm_calls, prompt_tokens, completion_tokens, context_tokens,
              retrieval_ms, generation_ms, ttft_ms, wall_ms, peak_rss_mb}
  learning:  {level, gate_cleared: bool, boost_applied: bool, delta, n_credits, table_updated_at}
```

`unit_id`, not `chunk_id` — §6.2 keys state on units so plain, agentic and graph share one schema. `chunk_ids[]` is §5.3's common source-chunk projection, which is what makes the citation comparable across arms.

**Model ids are a 2-tuple, always.** §6.2 makes learning a dimension crossed with all arms, not a fourth arm. A dropdown entry reading `Sovereign GraphRAG` that is silently at L1c attributes a learning gain to architecture. So: `sovereign/A3-plain@L0`, `sovereign/A4-agentic@L0`, `sovereign/A5-graph-local@L0`, the `@L1g` / `@L1c` / `@L2` variants of each, plus the free controls `sovereign/A0-closed-book`, `sovereign/A2-bm25`, `sovereign/AM-majority`. Fifteen ids from one process. `GET /v1/models` returns them; each `id` is the exact row label in the results table.

**Streaming.** `POST /v1/chat/completions`, standard body, SSE `data: {...}\
\
` chunks with `choices[].delta.content`, terminated by `data: [DONE]`. The stream is ordered so the wait is legible without any client-specific extension:

1. **Provenance block first**, emitted the moment retrieval returns (~340 ms), as markdown in the message body — every retrieved unit, with `cited` and `retrieved, not cited` distinguished in text.
2. The answer.
3. **A one-line footer**: `A3-plain@L0 · trace 01J8QK… · config 7f3a2c · corpus v3 · 5 units, 3 cited · retrieve 340 ms · gen 4.1 s · 1 LLM call`.

Citations ride in the message body as markdown, **not as `source` events**. Verified in `backend/open_webui/utils/middleware.py`: there is no `data.get("sources")` in the upstream stream-parsing path, so a plain endpoint cannot attach native citation chips. The alternative is a Pipe function — his Python executing inside a third-party server, admin-installed, with a `delta.tool_calls` footgun that loops to `CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS` (default **256**) and a frontmatter `requirements:` path that runs `pip install` from PyPI (`utils/plugin.py:441`). Markdown citations render in every OpenAI-compatible client, survive every version bump, and cost no sovereignty. `[changed:` the adopt-a-shell proposal made the Pipe its single deliverable. Rejected on sovereignty grounds by the audit and on FR-28 grounds by the dilution critique.`]`

**Cost rides along** in the final chunk: a top-level `usage` object plus a `timings` object (`middleware.py:3105-3108` merges both from any upstream). Response headers carry `X-Sovereign-Arm`, `X-Sovereign-Trace-Id`, `X-Sovereign-Config-Sha`.

`serve.py` never runs on the benchmark path, and the interlock is mutual: the harness refuses to start while the adapter port is listening, and the adapter refuses to start while a run lock exists. `[changed:` proposed as a one-line interlock by the dilution critique, replacing "remember to stop the UI" as a discipline across 12 overnight runs. The A-27B arm at ~23.5 GB of 24 GB has no margin for a forgotten process.`]`

---

### 15.5 The feedback loop

**No human-feedback write path ships in v1. This is a deliberate refusal, not an omission.**

**What is collected.** §6.2's credit signal, unchanged: gold-unit-in-top-k during the learning stream, IPW-weighted by `1/p(rank)`, ε-greedy at ε=0.1 with logged propensities, five seeds, permuted stream order. Deterministic and free. Disclosed in §6.2 as an upper bound on what a deployed system gets — and that disclosure is stronger than pretending thumbs are the real thing.

**Where it is stored.** One SQLite file written by `trace.py`, one append-only row per answer, shared with the harness. The row carries `trace_id, arm_id, level_id, config_sha, unit_ids[], ranks[], scores[], retriever_leg[], used_in_answer[], injected, propensity, context_tokens, llm_calls, ttft_ms, retrieval_ms, generation_ms, peak_rss_mb, model_digest, corpus_sha`. It has **no `chat_id` and no `message_id` columns.** `[changed:` the adopt proposal built a correlation-ID scheme against Open WebUI's `feedback.meta.message_id`, and argued it must be designed early because retrofitting discards prior feedback. Discarding data that §6.2 and §6.4 pre-register as inadmissible is not a loss. Cut.`]`

**How it reaches the re-ranking layer.** Directly. The credit table is `(unit_id | (query_cluster_id, unit_id)) → credit` with a `source` column. In v1 every row is `source=harness`. The human port is *specified* — `source=human`, `propensity=NULL`, per-unit not per-answer — and *not written*. Building storage for inadmissible data invites "why isn't this in the results?" in every review.

**The pathologies it must be defended against, and the defence in each case.**

| Pathology | Measured evidence | Defence |
|---|---|---|
| Answer-level credit collapses to a popularity counter | **Hit@10 65.1%, Recall@10 43.5%** query-blind (§6.1); 70.4% of gold citations in the top 10% of docs; three articles are gold for 495/376/273 queries | §6.3 control #1 is a gate: if L1 does not beat 65.1%, there is no learning, full stop |
| Exposure conditioning — no signal ever reaches ranks 11–1000 | k=5 is 0.75% of the corpus | ε=0.1 injection with logged propensity; `injected` and `propensity` are columns, not afterthoughts |
| Rich-get-richer on a biased signal | §6.6: rank-correlated credit noise; IPW corrects position propensity, not content bias | Simulated κ sweep, symmetric and rank-correlated (FR-24); IPW-off / ε-off ablation published as a figure |
| Recycling satisfying-but-wrong answers | §6.7: 90.3% → 99.0% / 71.0% depending only on whether recycled answers were right, **under an identical provenance signature** | Write-back is cited, not run (§3 non-goal 4). No write-back path exists in the shipped system, therefore no `SYNTHETIC` chip state is rendered — see the banned list |
| Sycophancy — optimising the feedback you got | OpenAI's April 2025 GPT-4o rollback: thumbs-derived reward "may have overpowered existing safeguards", rolled back in four days | No human feedback influences any ranking in v1, and the UI says so in the present tense |
| One non-blinded rater who built all three arms | n=1 | No human judgment enters any reported number. Open WebUI's blind arena (`utils/middleware.py:2271`, `random.choice(arena_model_ids)`) is genuine methodological value and is the one thing lost by not adopting it — stated here rather than buried |

**Disagreement, surfaced.** The sovereignty audit and the dilution critique both note that Open WebUI's `data.sibling_model_ids` records pairwise preference with the comparison set, which is a strictly better signal than a lone thumb, and that arena mode is genuinely blind. That is real and it is why the integration page exists. It does not change the verdict, because the granularity is still per-answer and the volume is still ~130 evenings to N2.

---

### 15.6 Honesty requirements, and the banned list

**Shown on every answer and every comparison, in the render, not in a footnote.**

1. **Arm and level as one string** — `A3-plain@L0` — plus `config_sha` and `corpus_sha`. Never inferable from prose style alone.
2. **The closed-book lift.** A0 runs on every query (3 s, §11). When it produces the same answer: *closed-book gave the same answer — retrieval contributed nothing to this response.* FR-7 made visible. This is the most on-brand element available and it costs three seconds.
3. **The majority-class constant for that stratum, beside the result.** Comparison 60.1%, temporal 46.7% (§4.3c). A cited "Yes" that does not clear the constant is labelled as not clearing it.
4. **Citations split into `cited` and `retrieved, not cited`.** Liu/Zhang/Liang (EMNLP Findings 2023): 51.5% of generated sentences fully supported by their citations, and citation precision correlated with perceived utility at **r = −0.96**. Chips make unsupported answers look better; a `used_in_answer` column does the opposite.
5. **The cost line at the same visual weight as the answer** — retrieval ms, generation ms, LLM calls, context tokens.
6. **The noise floor beside any pair.** *On S3, n=180, A3 vs A5 differ by 2.1 pts, below the 3.4-pt noise floor. This single comparison is not evidence.* This is the only defensible reason to render a comparison at all: it converts a side-by-side from an n=1 claim generator into a teaching device.
7. **Question provenance and stratum status.** `native (third-party authored)` vs `authored in-repo` (S1, S5a, S5b are authored — §4.4). Bounded strata say bounded.
8. **Outlet-stripped is the default variant.** As-published is available and labelled: *contains publisher-name oracle — 98.2–100% of native queries name their gold outlet (§4.3a)*.

**Banned outright.**

1. **Any learning indicator, in any tense**, until §6.4's H0-transfer is rejected on N2. The default rendered state is the pre-registered negative: *retrieval-feedback re-ranking: OFF — pre-registered gate not cleared (§6.1/§6.4). Feedback is stored and affects no ranking.* Rendering the negative is a stronger product statement than any sparkle and it is the version that survives the likely outcome. `[changed:` the adopt proposal's honest-looking string *"41 ratings incorporated"* is still banned. "Incorporated" asserts direction, from 41 judgments by one rater, on an axis §6.4 pre-registers as unable to detect 15 points.`]`
2. **Any unlabelled aggregate number, anywhere, including one computed in the browser.** FR-28 says *emitted by any code path*. A hidden or captioned Elo leaderboard does not satisfy it.
3. **A confidence percentage derived from a retrieval score.** Cosine similarity is not calibrated probability of correctness.
4. **A thumbs-up count displayed as quality.** It is an exposure-conditioned popularity count — the same object as the 65.1% control.
5. **A `SYNTHETIC` provenance state.** §6.7 makes write-back cited, not run. Shipping a defence that never fires implies an exercised capability. `[changed:` proposed by the bespoke-bench strategy, cut by the dilution critique.`]`
6. **A static "air-gapped" or "sovereign" badge.** A claim. `egress test PASSED 2026-08-07 02:14 · 0 outbound attempts observed` is a measurement, and FR-1 already produces the timestamp.
7. **A free-text box with no scope banner.** 253 articles, two categories. The first out-of-corpus question produces a confident wrong answer and the judgment lands on him.
8. **A model dropdown that mixes arms and learning levels as one name.** See 15.4.

FR-28 and FR-30 are enforced in the reporter and in CI, **never in JavaScript**. The renderer reads a precomputed `reportable` flag; it does not recompute a significance decision. `[changed:` the bespoke-bench strategy proposed the noise-floor rule as a renderer branch, arguing discipline-as-code beats discipline-as-vigilance. The scope critique is right that this creates a second implementation of his own honesty rule in the one untested layer. Resolution: the *rule* lives in the reporter, the *rendering* of its verdict lives in the browser, and FR-U6 asserts no number originates in the browser.`]`

---

### 15.7 Sovereignty audit

**The claim, rewritten before any code. `[changed:` the audit found that all three strategies made a runtime claim in language a reader hears as a lifetime claim, and the lifetime claim is false for every one of them.`]`**

> **Build time requires network.** Model weights from `registry.ollama.ai` and `huggingface.co`, packages from PyPI — all pinned by digest and enumerated in `model_manifest.json` and `uv.lock`.
> **At query time**, with the index built and weights on disk, the measured configuration opens no network connection. Verified *(date)* with a default-deny packet filter and a logging proxy: **N outbound attempts observed, N = 0.** Method and logs in `ops/egress/`.
> **No component transmits telemetry, usage data, prompts or feedback to any third party** in the shipped configuration.

Naming what was turned off is more credible than claiming there was nothing to turn off.

**The largest unmitigated leak is not the front end — it is Ollama, and it is common to every strategy considered.** The macOS Ollama `.app` checks for and downloads updates on launch and has no documented environment variable to disable it. `ollama pull` resolves tags against `registry.ollama.ai`, and no proposal pinned a model by digest at pull time. Fix: run the CLI binary, never the menubar app; pin every model by manifest digest; re-hash local blobs with `just verify-models`.

**Own stack.**

| Surface | When | Action |
|---|---|---|
| `registry.ollama.ai` — generator, embedder | build | Pin by manifest digest; hash into `model_manifest.json`; `just verify-models` re-hashes |
| Ollama macOS `.app` auto-update | runtime, **no off switch** | Do not run it. CLI binary only |
| `huggingface.co` — HHEM-2.1-open / MiniCheck, spaCy | build | Pin revision SHA; cache into the repo's model dir; `HF_HUB_OFFLINE=1` at run time |
| PyPI via `uv` | build | `uv.lock` committed |
| `reports/crossover.html` | never | No CDN, no webfont, no telemetry, no update check. `grep` is the audit |
| `serve.py` | never | Binds `127.0.0.1` only |

**Open WebUI — documented target, and the doc names every one of these.** Verified against `main` at v0.11.0 on 2026-08-08.

| Surface | Default | Kill |
|---|---|---|
| `api.github.com` version check — `env.py:1126`, `main.py:2422-2434`, called from `+layout.svelte:430` on every authenticated page | **on** | `OFFLINE_MODE=true` **and** `ENABLE_VERSION_UPDATE_CHECK=False` explicitly |
| `huggingface.co` — `all-MiniLM-L6-v2` loaded per worker when `RAG_EMBEDDING_ENGINE=''` (`config.py:984/990`, `retrieval.py:142`) | **on** | `RAG_EMBEDDING_ENGINE=ollama` **before first boot** |
| `huggingface.co` — `TaylorAI/bge-micro-v2` in `evaluations.py:65-78`, called from the tag-filtered leaderboard at `:173`, **bypassing `get_model_path` and the offline guard** | **on** | Pre-cache the checkpoint or never open the tag leaderboard |
| `openwebui.com` — community sharing, `Feedbacks.svelte:127-151`, `postMessage` target origin `'*'`, ships `data` (free-text comments) and `meta` | **on** | `ENABLE_COMMUNITY_SHARING=False` |
| PyPI — `pip install` of Pipe frontmatter `requirements:`, `utils/plugin.py:441` | **on** | `ENABLE_PIP_INSTALL_FRONTMATTER_REQUIREMENTS=False`; no Pipe ships anyway |
| Gravatar, Whisper, web-search providers, MinerU | off / opt-in | Leave off |
| Five background-generation vars (title, tags, autocomplete, follow-up, retrieval-query) | **all on** | All `False` — at 7.79 s per short-prompt call this is 30–40 s of hidden contention per turn and it corrupts any latency read from the UI |

**The trap that makes the kill-list decorative: `ENABLE_PERSISTENT_CONFIG` defaults `True` (`config.py:3186`).** Startup reads the embedding engine from the *database*, not the environment (`main.py:612-622`). Every PersistentConfig var seeds its DB row on first boot and is ignored thereafter. Boot once before writing the env file and the posture is permanently wrong, invisibly, in an untracked SQLite row. `ENABLE_PERSISTENT_CONFIG=False` appears in none of the four research briefs and none of the three strategies. It is the first line of the integration page.

**Credit where due, and it belongs in the doc:** ChromaDB telemetry is explicitly disabled (`retrieval/vector/dbs/chroma.py:32-34`); there is no posthog, sentry, mixpanel or segment anywhere; OpenTelemetry defaults off and to `localhost:4317`; fonts and Pyodide are vendored, not CDN. Open WebUI is not spyware. It is a product with an update check, a share button and two HuggingFace dependencies, all containable — and a weekly release cadence that would make the containment his permanent job.

**Verify, do not trust.**

- **Every one of these calls fails into a log line.** `get_app_latest_release_version` → `except Exception: return current`. `get_ef` → `log.error`, `ef = None`. `_get_embedding_model` → `log.error`, `None`. `get_model_path` → returns the bare repo id. **A blocked-egress test whose passing condition is "it still worked" proves nothing.** The test must count attempts, not crashes.
- The test therefore runs cold-cache (fresh user; empty `~/.cache/huggingface`, `~/.cache/tiktoken`, `nltk_data`, `~/.ollama`), under a default-deny `pf` anchor with `log`, `tcpdump -i pflog0` capturing the session, **plus** a second warm run behind an `HTTPS_PROXY` that 403s everything — because `aiohttp` is constructed with `trust_env=True` (`main.py:942, 983, 2429`), so a proxy catches attempts a packet filter can only drop. Assert **zero logged attempts**. Publish the log.
- After any Open WebUI boot, fetch `/api/config` and diff against the env file. `just verify-posture` exits 1 on drift.
- Confirm `HF_HUB_OFFLINE` was set before `huggingface_hub` was first imported — `evaluations.py` depends on that ordering and nothing enforces it.
- The built-in CSV export reads `data?.chat_id` (`Feedbacks.svelte:154-160`) but `chat_id`/`message_id` live in `meta` (`models/feedbacks.py:86-90`). The CSV emits a blank column and drops `message_id`. Use the JSON export. Noted in the doc so a reader does not lose an evening to it.

---

### 15.8 What runs in 24 GB

| Component | Resident | Basis |
|---|---|---|
| macOS + terminal + editor | ~4.0 GB | projected; Stage 0 measures |
| Ollama — 8B-class Q4 generator @ 8k ctx | 5.5–6.5 GB | projected from §5.1's measured 27B constants |
| Ollama — `qwen3-embedding:0.6b` q8_0 (639 MB file) | ~0.9 GB | **[measured]** file size; FR-3 unloads it during the benchmark query loop |
| Embedding matrix, 669 × 1024 fp32 | **2.74 MB** | **[measured]** |
| All query-variant vectors (~5k) | ~20 MB | derived |
| BM25 index, 669 chunks | ~10 MB | derived |
| nano-graphrag entities / relations / community reports | 200–400 MB | projected |
| MiniCheck / HHEM-2.1-open | < 1 GB | §7 |
| Python — harness or `serve.py`, numpy, **no torch** | 0.3–0.6 GB | projected |
| Browser tab rendering `crossover.html` | 0.3–0.6 GB | projected — the cost nobody budgets, on the same 24 GB |
| **Benchmark configuration (UI processes down, FR-3)** | **~12.5–13.5 GB** | ~10.5 GB headroom |
| **Stage 7 demo (8B resident, `serve.py` up, embedder co-resident)** | **~13.5–15.0 GB** | off-benchmark by construction |
| **A-27B sanity arm** — 17.0 GB @4k **[measured]** + 4.0 + 2.5 | **~23.5 GB** | **~0.5 GB headroom. No other process may be running.** FR-U16 enforces this |

Two things to say out loud. **The vector store is 2.74 MB** — the memory cost of RAG here is 100% model weights and the "database" is a rounding error, so no shell choice can be argued on vector-store grounds. And **Open WebUI's own process, at 600 MB–1.5 GB, would be larger than the entire retrieval stack** — Python, arms, BM25, vectors, graph artifacts, trace store — combined, on a metric (§8: peak RSS as a fraction of 24 GB) printed in the same table as accuracy. `crossover.html` contributes ~120 KB of static files and zero processes.

---

### 15.9 Deployment

```
uv sync
open reports/crossover.html          # the default surface: no server, no model, no network
```

Stage 7 adds one process:

```
just serve                           # ollama (CLI binary) + uvicorn on 127.0.0.1:8080
```

Two processes total, and `ollama` is already running for the harness. No Docker on this machine: Docker Desktop on Apple silicon runs a Linux VM carving a fixed allocation out of the same 24 GB and gets **no Metal passthrough**, so the model must run natively regardless. A compose file here would be either a fiction or a measurable regression. A `compose.yaml` still ships, labelled *reference deployment for Linux/x86 buyers — not the measured configuration*.

**Air-gapped variant.** There is no separate build. Pre-fetch weights and packages on a networked machine, verify against `model_manifest.json`, copy the repo and `~/.ollama` into the enclave, then:

```
just egress-test    # cold cache, pf default-deny + pflog capture, then a 403-everything proxy run
                    # asserts ZERO outbound attempts, writes the count and timestamp into model_manifest.json
just smoke          # 20-question slice, ~10 min, exit 0/1/2
```

`just smoke` is the highest-value line in the file: a stranger can falsify the setup in ten minutes rather than twelve nights.

**Backup set:** trace DB, credit table, embedding matrix, indices, `model_manifest.json`, config. **Execute the restore once and record the date.** "When did you last test restore?" has one acceptable answer.

---

### 15.10 Non-goals

1. **No chat interface.** Not adopted as a component, not written.
2. **No free-text query box in v1.**
3. **No human-feedback write path in v1** — no rating command, no per-unit judgment table, no feedback poller, no correlation-ID scheme against a third-party schema.
4. **No Pipe function, no Pipelines service, no Open WebUI Function of any kind.** His code does not execute inside someone else's server.
5. **No Elo leaderboard, hidden, captioned or otherwise.**
6. **No auth, no multi-user, no RBAC, no SSO.** The default surface is a file; the adapter binds loopback. Anything built on forwarded user headers would be a proxy assertion, not an authenticated identity, and would have to be labelled demonstration-grade — so it is not built.
7. **No per-document or per-chunk access control, and no connectors.** Team-quarters problems. `docs/gap-analysis-permission-aware-retrieval.md` — early vs late binding, chunk-level ACL inheritance, revocation lag — is written instead, and is worth more in a pre-sales conversation than a half-built version.
8. **No document upload or ingestion through any UI.** The corpus is pre-indexed. Open WebUI's native RAG would silently bypass the entire system under test, and its default pypdf extractor has documented persistent memory leaks.
9. **No OpenTelemetry exporter, no dashboards, no alerting.**
10. **No claim that this is a product.** Track 1 does not build one; see the header of this PRD.

---

### 15.11 Functional requirements

Numbered FR-U to avoid collision with §9's FR-1..FR-30.

**The contract**
- **FR-U1** Every arm implements `Arm.answer(query, config) -> Envelope` with the fields in 15.4. The harness calls it in-process; no HTTP on the benchmark path.
- **FR-U2** Model ids are `arm@level` 2-tuples. No surface may name an arm without its learning level.
- **FR-U3** `citations[].used_in_answer` is populated by every arm. A citation list without it is a contract violation and CI fails.
- **FR-U4** `trace.py` writes one append-only row per answer, carrying `injected` and `propensity`. It has no `chat_id` and no `message_id` column.

**The renderer**
- **FR-U5** `reports/crossover.html` is a single file: no network fetch, no CDN, no npm, no webfont, no build step. It opens from `file://` with the stack off.
- **FR-U6** No number rendered in `crossover.html` originates in the browser. CI asserts every rendered number is present in the committed `results.json`.
- **FR-U7** The renderer reads a precomputed `reportable` flag and greys the comparison. It never computes a significance decision (FR-30 lives in the reporter).
- **FR-U8** Every stratum row draws the AM constant and the A0 floor as rules, and greys any bar that does not clear AM (FR-6).
- **FR-U9** The mix slider recomputes under all four combinations of `raw / lift` × `as-published / stripped` (FR-4, FR-7). Shipping fewer than four is a violation.
- **FR-U10** Every question and column carries its status labels: stratum, `native` vs `authored in-repo`, `bounded` vs `resolved`, `implementation-bounded`, `LOWER BOUND`.
- **FR-U11** No learning indicator renders in any tense until §6.4's H0-transfer is rejected on N2. The default rendered state is the pre-registered negative.
- **FR-U12** Nothing on the banned list in 15.6 appears in any rendered surface. CI greps for the banned strings.

**The adapter**
- **FR-U13** `serve.py` exposes `GET /v1/models` and `POST /v1/chat/completions` with SSE, one id per arm×level, provenance as markdown in the body, `usage` + `timings` in the final chunk, and `X-Sovereign-Arm` / `-Trace-Id` / `-Config-Sha` headers. Binds `127.0.0.1`. Stage 7.
- **FR-U14** Any live-inference mode renders a permanent banner — *off-benchmark configuration: embedder co-resident (FR-3); latency and RSS are not comparable to reported figures* — and no number displayed in live mode may appear in the article.
- **FR-U15** The harness refuses to start while the adapter port is listening; the adapter refuses to start while a run lock exists.
- **FR-U16** Before the A-27B arm runs, the harness asserts no adapter, no browser and no other model is resident (`ollama ps` + port check) and aborts otherwise. ~0.5 GB of headroom is not a margin for judgment.

**Sovereignty**
- **FR-U17** The sovereignty claim is stated as build-time vs query-time (15.7). A bare "nothing leaves the machine" is a violation.
- **FR-U18** `just egress-test` runs cold-cache under a default-deny packet filter with `pflog` capture **and** a second pass behind a 403-everything proxy, and asserts **zero outbound attempts**. Exit 0 is not the passing condition; attempt count is. Result and timestamp are written into `model_manifest.json`.
- **FR-U19** `model_manifest.json` records the Ollama manifest digest and HF revision SHA for every model artifact — generator, embedder, MiniCheck/HHEM, spaCy. `just verify-models` re-hashes local blobs and exits 1 on mismatch.
- **FR-U20** Ollama runs from the CLI binary. The macOS `.app` is not used, because its update check has no documented off switch.
- **FR-U21** Open WebUI is not a dependency: absent from `pyproject.toml`, absent from CI, absent from the reproduction path. It appears only in `docs/integrations/open-webui.md`, pinned to a version and a digest, with `ENABLE_PERSISTENT_CONFIG=False` as the first line of the kill-list and `just verify-posture` documented alongside.
- **FR-U22** No shipped artifact makes an outbound call. CI greps `reports/` and `src/sovereign_rag/serve/` for URLs.

---

### 15.12 Evening budget

One evening ≈ 2.5 focused hours.

| Item | Hours | Gate |
|---|---|---|
| `reports/crossover.html` — stratum bars, CI whiskers, AM/A0 rules, `reportable` greying, mix slider, four toggles, cost row, light/dark | 3.5 | Stage 1 |
| `ops/egress/` — pf anchor, pflog capture, proxy runner, `just egress-test` | 1.5 | Stage 1 |
| `ops/manifest.py` + `just verify-models` — digests, re-hashing, egress result | 2.0 | Stage 1 |
| MANIFEST pane (second tab in the same file) | 1.0 | Stage 7 |
| `serve.py` — ~150 lines, SSE, markdown citations, usage/timings, interlocks | 1.5 | Stage 7 |
| `docs/integrations/open-webui.md` — kill-list, PersistentConfig trap, the `evaluations.py` HF load, the community-share button, the CSV export bug, one screenshot | 1.0 | Stage 7 |
| **Committed total** | **10.5 h ≈ 4–5 evenings** | |
| *less* the matplotlib crossover figure this replaces | −2.0 | |
| **Net new** | **8.5 h** | |

**Conditional, built only if 15.13's trigger fires:** REPLAY pane, 6.0 h.

**Cut to fit, with what each cut bought:**

| Cut | Saved |
|---|---|
| Open WebUI as a shipped component — Pipe, manifold, source events, arena wiring, env tuning, memory triage, tracking a weekly release cadence | ~7 evenings, plus a permanent config-drift tax on a §8 reported column |
| A purpose-built live bench — SSE demux, three-column streaming, judgment UI, blind mode | ~19 h, and 870 lines of untested browser code in a repo whose CI asserts every number |
| Every human-feedback write path — rate command, `unit_judgments`, feedback poller, correlation-ID scheme | ~4 h, and the question *"why isn't this in the results?"* in every review |
| Auth, RBAC, SSO, multi-user | ~2 evenings of configuration, and a per-user-security claim that would have to be labelled demonstration-grade |
| A blind three-way arena | Nothing — this is the one real loss, ~13 h of Open WebUI wiring not spent, and blinding is genuine methodological value. §7's 30-item audit is blinded via a shuffled CSV instead |

**Disagreement, surfaced and resolved.** The scope critique put v1 at ~1 evening (crossover file only) plus 1.5 gated. The dilution critique put it at 4–6 evenings (crossover + manifest + replay bench). The gap is almost entirely the replay bench and the sovereignty work. **Resolution: the replay bench is conditional (15.13), and the egress/manifest work is committed because the sovereignty audit showed nobody had budgeted it and FR-1 already promises it.** That lands at 4–5 evenings, closer to dilution's number for scope's reasons.

---

### 15.13 Staging

**No interaction-layer hour is spent before Stage 1's abandon checkpoint is committed to the repo.** §13 guarantees the whole study can be abandoned at Stage 1 for the cost of one evening. That guarantee is void the moment a front end exists in front of it. `[changed:` the adopt strategy put "env kill-list, pin version, verify join keys" as evening one, before Stage 0 had measured whether the 8B hits its projected 20 tok/s. Rejected by the scope critique and it is right.`]`

**Ships after Stage 1 (7.0 h).** `reports/crossover.html` against the retrieval-only tables — as-published vs stripped, corpus-size curve, recall@10 per arm-leg, the query-blind popularity baseline, oracle headroom. Those are real numbers with real n and no generation, so the slider works before a single generation hour is spent. Plus `just egress-test` and `model_manifest.json`, which the sovereignty claim needs regardless of whether any UI exists.

**Ships in Stage 7 (3.5 h).** MANIFEST pane, `serve.py`, `docs/integrations/open-webui.md`. `[changed:` the bespoke-bench strategy argued the adapter should be written in evening one as a reversibility hedge — build the instrument, keep the shell one config change away. The door does not close: reversibility is a property of the `Arm.answer() -> Envelope` contract, which is in the harness from day one, not of the 150-line wrapper over it. The wrapper is exactly as cheap in Stage 7.`]`

**Conditional — REPLAY pane (6.0 h). Trigger, pre-registered:** built only if Stage 4 completes **and** at least one stratum shows an inter-arm difference above the measured noise floor. If every difference falls under the floor — a valid outcome per §12 — there is nothing for a comparison view to teach and it is not built.

**Never, unless both fire:** live fan-out and per-unit judgment capture. Trigger: §6.1's pre-flight gate clears **and** N2 exceeds ~150. `[changed:` the front end was the only component in this PRD with no kill condition. Adding one is what brings it under the same contract as everything else in the document.`]`

**Falsification test, recorded here so it can be checked later:** if the article lands and the most-quoted, most-linked element is neither the crossover slider nor the Stage 0 indexing-cost paragraph, the interaction layer was scoped wrong. The counterfactual — whether fifteen evenings of chat UI would have converted more inbound — is unmeasurable, which means it stays permanently arguable and he will never know he was right. Stating that is cheaper than pretending otherwise."
