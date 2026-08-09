# Study status

Updated automatically as stages complete. Ledger of what is DONE / RUNNING / NEXT.

| Stage | State | Artifact |
|---|---|---|
| 0 — machine measurement | DONE, verified | reports/STAGE0.md |
| 1 — retrieval layer (BM25/dense/hybrid) | DONE, verified, 1 finding retracted honestly | reports/STAGE1.md, reports/crossover.html |
| graph index (600) | **RUNNING** — resumable: `./.venv/bin/python ops/graph_index.py --budget 600` | data/graph_600.jsonl, log reports/graph_600.log |
| 2 — graph arm | code READY + smoke-tested on partial graph; runs when index lands | src/build_graph.py, src/score_graph.py |
| 3 — generation arms | pilot first (π_d decides n); needs GPU free | not started |
| 4 — learning layer | gated on beating the 65.1% popularity baseline | not started |
| 5 — article | after 2–3; interim two-article package already writable from Stage 1 | not started |

If the machine restarted: the index resumes with the command above (checkpointed,
loses at most one chunk). Everything else regenerates from committed scripts.
