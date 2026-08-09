# Study status

Updated automatically as stages complete. Ledger of what is DONE / RUNNING / NEXT.

| Stage | State | Artifact |
|---|---|---|
| 0 — machine measurement | DONE, verified | reports/STAGE0.md |
| 1 — retrieval layer (BM25/dense/hybrid) | DONE, verified, 1 finding retracted honestly | reports/STAGE1.md, reports/crossover.html |
| graph index (600) | DONE — 21.2 h, 2.9% truncated | data/graph_600.jsonl |
| 2 — graph arm | DONE — loses to BM25 everywhere, −24.19 [−26.72, −21.72]; no crossover exists at retrieval level | reports/STAGE2.md, reports/stage2_graph.json |
| 3 — generation arms | pilot RUNNING (a0/plain/graph × 100, checkpointed): `ops/stage3_pilot.py` | reports/stage3_pilot.jsonl |
| 4 — learning layer | gated on beating the 65.1% popularity baseline | not started |
| 5 — article | after 2–3; interim two-article package already writable from Stage 1 | not started |

If the machine restarted: the index resumes with the command above (checkpointed,
loses at most one chunk). Everything else regenerates from committed scripts.
