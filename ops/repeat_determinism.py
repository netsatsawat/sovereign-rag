"""Producer for the 'extraction is deterministic, verified 3x' claim.

Runs one representative chunk through the extraction call three times and
asserts byte-identical entity and relationship sets. Requires the local
Ollama and therefore must NOT run while the graph index holds the GPU;
STAGE0.md cites the result of the run performed 9 Aug 2026.

    python ops/repeat_determinism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ops"))
from e0_e2e3 import call, norm, rel_key  # noqa: E402


def main() -> None:
    chunks = [json.loads(l) for l in (ROOT / "data" / "chunks_600.jsonl").open()]
    c = chunks[len(chunks) // 2]
    runs = []
    for i in range(3):
        o = call(c["text"])
        if not o["payload"]:
            print(f"run {i+1}: {o}"); return
        runs.append(({norm(e["name"]) for e in o["payload"]["entities"]},
                     {rel_key(r) for r in o["payload"]["relationships"]}))
        print(f"run {i+1}: {len(runs[-1][0])} entities, {len(runs[-1][1])} rels, "
              f"{o['gen_tokens']} tokens")
    ident = all(r == runs[0] for r in runs)
    print("byte-identical across 3 runs:", ident)
    out = ROOT / "reports" / "stage0.json"
    d = json.loads(out.read_text())
    d["determinism_check"] = {"chunk_id": f'{c["doc_id"]}#{c["chunk_index"]}',
                              "runs": 3, "identical": ident}
    out.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
