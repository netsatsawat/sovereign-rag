"""Stage 0: measure the machine before writing any benchmark code.

Four gates. All four must pass or the study is re-scoped rather than started.

  G1  decode and prefill throughput on the 8B candidate
  G2  resident size and CPU/GPU split at 4k / 8k / 16k context
  G3  20 real chunks through the real extraction prompt: 20/20 valid
      structured payloads, and under 90 s per chunk
  G4  generator and embedder co-resident inside 24 GB

The measurements are the point, not a formality. The design's original
41.6-hour estimate came from costing a 1,000-token call with the unit price of
a 50-token one, and was 63x wrong. Nothing here is projected.

    python ops/stage0.py            # all four gates
    python ops/stage0.py --gate g3  # one gate
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import urllib.request

OLLAMA = "http://localhost:11434"
HERE = Path(__file__).parent
OUT = HERE.parent / "reports" / "stage0.json"

GEN = "qwen3:8b"
EMB = "qwen3-embedding:0.6b"


def post(path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{OLLAMA}{path}", timeout=30) as r:
        return json.loads(r.read())


def ps() -> list[dict]:
    """`ollama ps` via the API: resident size and the CPU/GPU split."""
    try:
        return get("/api/ps").get("models", [])
    except Exception:
        return []


def unload(model: str) -> None:
    try:
        post("/api/generate", {"model": model, "keep_alive": 0, "prompt": ""}, timeout=60)
    except Exception:
        pass
    time.sleep(2)


def gpu_split(m: dict) -> str:
    """Ollama reports size and size_vram; the gap is what spilled to CPU."""
    total, vram = m.get("size", 0), m.get("size_vram", 0)
    if not total:
        return "unknown"
    gpu = round(100 * vram / total)
    return f"{100 - gpu}% CPU / {gpu}% GPU"


# ---- G1: throughput ------------------------------------------------------
def g1_throughput() -> dict:
    """Decode and prefill, separated. Ollama returns the counts and the
    nanosecond timings, so neither has to be inferred from wall clock."""
    unload(GEN)
    t0 = time.time()
    post("/api/generate", {"model": GEN, "prompt": "hi", "stream": False,
                           "options": {"num_predict": 1}})
    cold_load_s = round(time.time() - t0, 2)

    runs = []
    for label, prompt_tokens, num_predict in (
        ("short", 50, 64),
        ("long_decode", 50, 512),
        ("long_prefill", 6000, 64),
    ):
        prompt = ("The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 9 + 1))
        r = post("/api/generate", {
            "model": GEN, "prompt": prompt, "stream": False,
            "think": False,
            "options": {"num_predict": num_predict, "temperature": 0, "seed": 0},
        })
        pe, pc = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0)
        ec, ed = r.get("eval_count", 0), r.get("eval_duration", 0)
        runs.append({
            "case": label,
            "prompt_tokens": pe,
            "gen_tokens": ec,
            "prefill_tok_s": round(pe / (pc / 1e9), 1) if pc else None,
            "decode_tok_s": round(ec / (ed / 1e9), 2) if ed else None,
            "total_s": round(r.get("total_duration", 0) / 1e9, 2),
        })
    return {"cold_load_s": cold_load_s, "runs": runs}


# ---- G2: context scaling -------------------------------------------------
def g2_context() -> dict:
    rows = []
    for ctx in (4096, 8192, 16384):
        unload(GEN)
        post("/api/generate", {"model": GEN, "prompt": "hi", "stream": False,
                               "think": False,
                               "options": {"num_ctx": ctx, "num_predict": 1}})
        time.sleep(1)
        for m in ps():
            if m.get("name", "").startswith(GEN.split(":")[0]):
                rows.append({
                    "num_ctx": ctx,
                    "resident_gb": round(m.get("size", 0) / 1e9, 1),
                    "vram_gb": round(m.get("size_vram", 0) / 1e9, 1),
                    "split": gpu_split(m),
                })
                break
    return {"rows": rows}


# ---- G4: co-residency ----------------------------------------------------
def g4_coresidency() -> dict:
    post("/api/generate", {"model": GEN, "prompt": "hi", "stream": False,
                           "think": False, "options": {"num_predict": 1}})
    t0 = time.time()
    post("/api/embed", {"model": EMB, "input": "co-residency probe"})
    embed_s = round(time.time() - t0, 2)
    resident = [{"name": m.get("name"), "gb": round(m.get("size", 0) / 1e9, 2),
                 "split": gpu_split(m)} for m in ps()]
    total = round(sum(r["gb"] for r in resident), 2)
    return {
        "resident": resident,
        "total_gb": total,
        "both_resident": len(resident) >= 2,
        "embed_call_s": embed_s,
        "fits_24gb": total < 24.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["g1", "g2", "g3", "g4", "all"], default="all")
    args = ap.parse_args()

    OUT.parent.mkdir(exist_ok=True)
    res = {
        "stage": 0,
        "machine": {
            "platform": platform.platform(),
            "processor": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True).stdout.strip(),
            "ram_gb": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                            capture_output=True, text=True).stdout) / 1e9),
        },
        "ollama_version": get("/api/version").get("version"),
        "generator": GEN,
        "embedder": EMB,
    }

    if args.gate in ("g1", "all"):
        print("G1 throughput ...", flush=True)
        res["g1_throughput"] = g1_throughput()
        for r in res["g1_throughput"]["runs"]:
            print(f"   {r['case']:14} prefill {r['prefill_tok_s']} tok/s"
                  f"  decode {r['decode_tok_s']} tok/s  ({r['total_s']}s)")

    if args.gate in ("g2", "all"):
        print("G2 context scaling ...", flush=True)
        res["g2_context"] = g2_context()
        for r in res["g2_context"]["rows"]:
            print(f"   num_ctx {r['num_ctx']:6} resident {r['resident_gb']} GB  {r['split']}")

    if args.gate in ("g4", "all"):
        print("G4 co-residency ...", flush=True)
        res["g4_coresidency"] = g4_coresidency()
        c = res["g4_coresidency"]
        print(f"   resident {c['total_gb']} GB, both={c['both_resident']}, "
              f"fits 24 GB={c['fits_24gb']}, embed {c['embed_call_s']}s")

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.update(res)
    OUT.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
