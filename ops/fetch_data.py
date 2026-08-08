"""Fetch the corpus and query parquets and record their hashes.

Note the honest caveat: parquet bytes depend on the HF backend's writer
version, so a re-download is verified by ROW CONTENT hash, not file bytes.
The chunker pipeline is deterministic from row content onward.

    python ops/fetch_data.py
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
URLS = {
    "multihoprag_corpus.parquet":
        "https://huggingface.co/api/datasets/yixuantt/MultiHopRAG/parquet/corpus/train/0.parquet",
    "multihoprag_queries.parquet":
        "https://huggingface.co/api/datasets/yixuantt/MultiHopRAG/parquet/MultiHopRAG/train/0.parquet",
}


def content_sha(path: Path) -> str:
    t = pq.read_table(path)
    h = hashlib.sha256()
    for row in t.to_pylist():
        h.update(json.dumps(row, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def main() -> None:
    manifest = {}
    for name, url in URLS.items():
        dest = ROOT / "data" / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        manifest[name] = {
            "url": url,
            "file_sha256": hashlib.sha256(dest.read_bytes()).hexdigest()[:16],
            "content_sha256": content_sha(dest),
            "rows": pq.read_table(dest).num_rows,
        }
        print(f"  {name}: {manifest[name]['rows']} rows, content {manifest[name]['content_sha256']}")
    (ROOT / "data" / "sources.manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote data/sources.manifest.json")


if __name__ == "__main__":
    main()
