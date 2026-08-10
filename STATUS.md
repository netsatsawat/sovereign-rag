# Study status

Updated automatically as stages complete. Ledger of what is DONE / RUNNING / NEXT.

| Stage | State | Artifact |
|---|---|---|
| 0 — machine measurement | DONE, verified | reports/STAGE0.md |
| 1 — retrieval layer (BM25/dense/hybrid) | DONE, verified, 1 finding retracted honestly | reports/STAGE1.md, reports/crossover.html |
| graph index (600) | DONE — 21.2 h, 2.9% truncated | data/graph_600.jsonl |
| 2 — graph arm | DONE — loses to BM25 everywhere, −24.19 [−26.72, −21.72]; no crossover exists at retrieval level | reports/STAGE2.md, reports/stage2_graph.json |
| 3 — generation arms | DONE at n=1,381 — plain 63.1% vs A0 39.5% vs graph 46.6%; all pairs Holm-significant; synthesis bottleneck confirmed | reports/STAGE3.md, stage3_final.json |
| 4 — learning layer | DONE — **pre-registered negative**: learned re-ranking −1.40 on similar queries, indistinguishable from shuffled credit | reports/STAGE4.md, stage4_learning.json |
| 5 — article | ALL data collection complete (stages 0–4). Thesis: cheapest retrieval wins, synthesis is the bottleneck, feedback re-ranking is a measured negative | not started |

If the machine restarted: the index resumes with the command above (checkpointed,
loses at most one chunk). Everything else regenerates from committed scripts.
