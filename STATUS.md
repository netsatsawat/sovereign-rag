# Study status

Updated as stages complete. Ledger of what is DONE / RUNNING / NEXT.

| Stage | State | Artifact |
|---|---|---|
| 0: machine measurement | DONE, verified | reports/machine-measurement.md |
| 1: retrieval layer (BM25/dense/hybrid) | DONE, verified, 1 finding retracted honestly | reports/retrieval.md, reports/crossover.html |
| graph index (600) | DONE. 21.2 h, 2.9% truncated | data/graph_600.build.json, reports/graph_600.log |
| 2: graph arm | DONE. Loses to BM25 everywhere, −24.19 [−26.72, −21.72]; no crossover exists at retrieval level | reports/graph-arm.md, reports/stage2_graph.json |
| 3: generation arms | DONE at n=1,381: plain 63.1% vs A0 39.5% vs graph 46.6%; all pairs Holm-significant; synthesis bottleneck confirmed | reports/generation-arms.md, stage3_final.json |
| 4: learning layer | DONE, **pre-registered negative**: learned re-ranking −1.40 on similar queries, indistinguishable from shuffled credit | reports/learning-layer.md, stage4_learning.json |
| full-corpus replication | DONE. BM25 strict@10 48.74 on 2,255 queries, 609 articles; margin over dense grows (run within the stage-6 follow-ups) | reports/stage5_full_corpus.json |
| 6: pushing on the claims | DONE. BM25 win hardened, synthesis-bottleneck claim broken and rewritten (27B abstains 84%; comparison stratum degenerate, always-Yes 60.0%) | reports/claim-stress-tests.md, stage6_27b.json(l) |
| 7: HotpotQA transfer | DONE. BM25 strict@10 68.4 on 1,500 questions / 14,526-paragraph pool | reports/stage7_hotpot.json |
| 8: Muse Glimmer 30B, day one | DONE. 480/480 calls; loses to 8B via escape-hatch compliance (−15.83pp RAG); 96.67% closed-book inference = contamination floor; no significant RAG-over-closed | reports/glimmer-day-one.md, stage8_glimmer.json(l) |
| 9: five-language replication | DONE. 2,000 rows, zero errors; English verdict inverts (glimmer ties/wins outside English); abstention becomes a good evidence detector | reports/five-language-replication.md, stage9_{th,ja,zh,es,vi}.json(l) |
| article | drafts in progress; every number recomputes from the artifacts above | (companion posts) |

Stage numbers map to the reports as: 0 `machine-measurement`, 1 `retrieval`,
2 `graph-arm`, 3 `generation-arms`, 4 `learning-layer`, 6 `claim-stress-tests`,
7 `hotpotqa-transfer`, 8 `glimmer-day-one`, 9 `five-language-replication`,
10 `confound-closers`. The `reports/*.json` and `reports/*.jsonl` artifacts keep
their stage numbers, because committed logs cite them by those names.

Everything regenerates from committed scripts; generation stages are
checkpointed per (arm, qi) and resume by re-running the same command.
Model binaries are pinned in models.manifest.json.

Regenerates but is not committed: the graph index's raw extraction checkpoint
`data/graph_600.jsonl` (4.1 MB) falls under `.gitignore`'s `data/*.jsonl`. What is committed is what everything downstream
actually reads: the assembled graph `data/graph_600.build.json` and the build
log `reports/graph_600.log`. Getting the checkpoint back costs the full 21.2 h:

1. `python src/chunker.py --budget 600` (only if `data/chunks_600.jsonl` is absent)
2. `python ops/graph_index.py --budget 600` (21.2 h, resumable)
3. `python src/build_graph.py --budget 600` (reassembles the .build.json)
