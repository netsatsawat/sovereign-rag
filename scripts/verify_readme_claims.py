#!/usr/bin/env python3
"""No number in README.md or STATUS.md without a runnable path behind it.

This repo publishes its headline figures in STATUS.md as much as in README.md,
so both files are checked. Every assertion below recomputes a value from a
committed artifact under reports/ (or from git's own record of what is
tracked) and then asserts the prose literally contains that recomputed string
so an edit to either side breaks CI instead of drifting quietly.

What it does not do: re-run a stage, load a model, read
data/*.npy, or touch the network. It reads the committed JSON summaries and
their JSONL row files and finishes in seconds.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bm25 import normalize                                  # noqa: E402

# STATUS.md sets its negative numbers with a real minus sign; the artifacts
# use ASCII. Fold one onto the other so a typographic choice is not read as
# a numeric drift.
MINUS = "−"

problems = []


def check(name, condition, detail=""):
    print(f"  {'ok ' if condition else 'FAIL'} {name}" +
          (f" ({detail})" if detail and not condition else ""))
    if not condition:
        problems.append(name)


def report(name):
    return json.loads((REPO / "reports" / name).read_text(encoding="utf-8"))


def rows(name):
    with (REPO / "reports" / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def num(x):
    """Render a computed float the way the reports write it: no trailing zeros."""
    return f"{x:g}"


def committed():
    """Every path git has under version control. Working-tree existence is the
    wrong test for 'is this published?': the tree also holds the ignored
    artifacts, which is exactly what the graph-index disclosure is about."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO, check=True,
                             capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.stdout.splitlines())


def main() -> int:
    readme = (REPO / "README.md").read_text(encoding="utf-8").replace(MINUS, "-")
    status = (REPO / "STATUS.md").read_text(encoding="utf-8").replace(MINUS, "-")
    stage6_md = (REPO / "reports" / "claim-stress-tests.md").read_text(
        encoding="utf-8").replace(MINUS, "-")

    print("stage 7, HotpotQA transfer (reports/stage7_hotpot.json):")
    s7 = report("stage7_hotpot.json")
    strict7 = round(100 * sum(s7["per_query_strict"]) / len(s7["per_query_strict"]), 2)
    check(f"strict@10 recomputes from the per-query array ({num(strict7)})",
          strict7 == s7["bm25_strict10"], f"summary says {s7['bm25_strict10']}")
    check("the per-query array is as long as the question count",
          len(s7["per_query_strict"]) == s7["questions"],
          f"{len(s7['per_query_strict'])} rows vs {s7['questions']} questions")
    sentence = (f"BM25 strict@10 {num(strict7)} on {s7['questions']:,} questions"
                f" / {s7['pool_paragraphs']:,}-paragraph pool")
    check(f"STATUS quotes it ({sentence})", sentence in status)

    print("full-corpus replication (reports/stage5_full_corpus.json):")
    s5 = report("stage5_full_corpus.json")
    strict5 = round(100 * sum(s5["per_query_strict"]) / len(s5["per_query_strict"]), 2)
    check(f"strict@10 recomputes from the per-query array ({num(strict5)})",
          strict5 == s5["bm25_strict10"], f"summary says {s5['bm25_strict10']}")
    check("the per-query array is as long as the query count",
          len(s5["per_query_strict"]) == s5["queries"],
          f"{len(s5['per_query_strict'])} rows vs {s5['queries']} queries")
    articles = int(re.search(r"(\d+) articles", s5["corpus"]).group(1))
    sentence = (f"BM25 strict@10 {num(strict5)} on {s5['queries']:,} queries,"
                f" {articles} articles")
    check(f"STATUS quotes it ({sentence})", sentence in status)

    print("stage 2, the graph arm (reports/stage2_graph.json):")
    s2 = report("stage2_graph.json")["deltas_vs_stage1"]["k10_graph_vs_bm25_strict"]
    lo, hi = s2["ci95_pts"]
    sentence = f"{s2['delta_pts']:.2f} [{lo:.2f}, {hi:.2f}]"
    check(f"STATUS quotes the graph-vs-BM25 delta and CI ({sentence})",
          sentence in status)
    check("that CI sits entirely below zero, as 'loses to BM25 everywhere' needs",
          hi < 0, f"upper bound {hi}")

    print("stage 3, generation arms (reports/stage3_final.json):")
    s3 = report("stage3_final.json")
    arm = {k: round(v["containment"], 1) for k, v in s3["arms"].items()}
    sentence = (f"plain {num(arm['plain'])}% vs A0 {num(arm['a0'])}%"
                f" vs graph {num(arm['graph'])}%")
    check(f"STATUS quotes the three arms ({sentence})", sentence in status)
    check(f"STATUS quotes the paired n (n={s3['n']:,})", f"n={s3['n']:,}" in status)
    unsigned = [p for p in ("plain_vs_a0", "plain_vs_graph", "graph_vs_a0")
                if not s3["mcnemar"][p]["significant_holm"]]
    check("every arm pair is Holm-significant in the artifact", not unsigned,
          f"not significant: {unsigned}")
    check("STATUS says all pairs Holm-significant",
          "all pairs Holm-significant" in status)

    print("stage 4, the learning layer (reports/stage4_learning.json):")
    s4 = report("stage4_learning.json")
    learned = s4["deltas"]["n1_learned_vs_base_strict"]
    sentence = f"learned re-ranking {learned['delta_pts']:.2f} on similar queries"
    check(f"STATUS quotes the learned-vs-base delta ({sentence})",
          sentence in status)
    shuffled = s4["deltas"]["n1_learned_vs_shuffled_strict"]["ci95"]
    check("the learned-vs-shuffled CI brackets zero",
          shuffled[0] <= 0 <= shuffled[1], f"ci95={shuffled}")
    check("STATUS calls it indistinguishable from shuffled credit",
          "indistinguishable from shuffled credit" in status)
    verdict = s4["gates"]["verdict"].split(":")[0]
    check(f"STATUS carries the artifact's verdict ({verdict})",
          verdict.lower() in status.lower())

    print("stage 6, the 27B abstention finding (reports/stage6_27b.json):")
    s6 = report("stage6_27b.json")
    rate = s6["verdict"]["abstention_rates"]["plain_27b"]
    check(f"STATUS quotes the 27B RAG abstention rate (27B abstains {rate:.0%})",
          f"27B abstains {rate:.0%}" in status)
    # The always-Yes baseline belongs to the comparison stratum stage 6 and
    # stage 8 share (the same 120 questions), and ops/stage8_glimmer.py is what
    # computes it. Recompute it from the committed stage-8 rows using that same
    # normalised containment, then require the stored field to agree before
    # either is matched against the prose, so a drifted artifact and a drifted
    # sentence are distinguishable rather than cancelling out. Reconstructing it
    # from the stage-3 pool instead reads a different, larger stratum (391
    # questions, 60.1%) and quietly answers a question nobody asked.
    gold = {r["qi"]: r["gold"] for r in rows("stage8_glimmer.jsonl")
            if r["type"] == "comparison_query"}
    yes = sum(1 for g in gold.values()
              if normalize(g) and normalize(g) in normalize("Yes"))
    always_yes = round(100 * yes / len(gold), 2)
    stored = report("stage8_glimmer.json").get("always_yes_comparison")
    check(f"the always-Yes baseline recomputes from the stage-8 rows "
          f"({yes}/{len(gold)})",
          stored is not None and abs(stored - always_yes) < 0.005,
          f"rows give {always_yes}, stage8_glimmer.json stores {stored}")
    # One decimal, because that is how both prose sites write this figure; num()
    # would render 60.0 as "60" and match neither.
    shown = f"{always_yes:.1f}"
    quoted = re.search(r"always-Yes [\d.]+%", status)
    check(f"STATUS quotes the always-Yes comparison baseline "
          f"(always-Yes {shown}%)",
          f"always-Yes {shown}%" in status,
          f"{yes}/{len(gold)} comparison golds are Yes -> {shown}%, "
          f"but STATUS says {quoted.group(0) if quoted else 'nothing'}")
    check(f"claim-stress-tests.md quotes the same baseline (scores {shown}%)",
          f'scores {shown}%' in stage6_md,
          "claim-stress-tests.md and STATUS.md must not disagree about the same number")

    print("stage 8, Muse Glimmer 30B (reports/stage8_glimmer.json):")
    s8 = report("stage8_glimmer.json")
    primary = [r for r in rows("stage8_glimmer.jsonl")
               if r["arm"] in ("a0_glimmer", "plain_glimmer")]
    landed = sum(1 for r in primary
                 if not str(r["done_reason"]).startswith("error"))
    check("the two primary arms hold the row count the summary sizes",
          len(primary) == 2 * s8["n"], f"{len(primary)} rows vs {2 * s8['n']}")
    check(f"STATUS quotes the day-one call count ({landed}/{len(primary)} calls)",
          f"{landed}/{len(primary)} calls" in status)
    delta8 = s8["paired"]["plain_vs_8b"]["delta_pp"]
    check(f"STATUS quotes the RAG delta against the 8B ({delta8:.2f}pp RAG)",
          f"{delta8:.2f}pp RAG" in status)
    closed = s8["arms"]["a0_glimmer"]["by_type"]["inference_query"]["containment"]
    check(f"STATUS quotes the closed-book inference figure "
          f"({num(closed)}% closed-book)", f"{num(closed)}% closed-book" in status)

    print("stage 9, the five-language replication:")
    langs = re.search(r"stage9_\{([a-z,]+)\}", status).group(1).split(",")
    missing = [ln for ln in langs
               if not (REPO / "reports" / f"stage9_{ln}.jsonl").exists()]
    check(f"every language STATUS lists has committed rows ({','.join(langs)})",
          not missing, f"no row file for {missing}")
    total = errors = 0
    for lang in [ln for ln in langs if ln not in missing]:
        rs = rows(f"stage9_{lang}.jsonl")
        total += len(rs)
        errors += sum(1 for r in rs
                      if str(r.get("done_reason", "")).startswith("error"))
    check(f"the {len(langs)} language files hold {total:,} rows, none errored",
          errors == 0, f"{errors} rows with done_reason error:*")
    check(f"STATUS quotes it ({total:,} rows, zero errors)",
          f"{total:,} rows, zero errors" in status)

    print("the disclosed graph-index gap:")
    tracked = committed()
    check("git can list what is committed", tracked is not None,
          "git ls-files failed; the disclosure below cannot be checked")
    if tracked is not None:
        check("data/graph_600.jsonl is NOT committed, as STATUS discloses",
              "data/graph_600.jsonl" not in tracked)
        for path in ("data/graph_600.build.json", "reports/graph_600.log"):
            check(f"{path} IS committed, as STATUS promises", path in tracked)
    check("STATUS names the uncommitted checkpoint",
          "data/graph_600.jsonl" in status)
    check(".gitignore's data/*.jsonl is what excludes it",
          "data/*.jsonl" in (REPO / ".gitignore").read_text(encoding="utf-8"))
    build = json.loads(
        (REPO / "data" / "graph_600.build.json").read_text(encoding="utf-8"))
    truncated = round(100 * build["excluded_truncated_or_unparseable"]
                      / build["chunks_total"], 1)
    hours = re.findall(r"elapsed\s+([\d.]+)h",
                       (REPO / "reports" / "graph_600.log").read_text(
                           encoding="utf-8"))[-1]
    check(f"STATUS quotes the build cost ({hours} h, {num(truncated)}% truncated)",
          f"{hours} h, {num(truncated)}% truncated" in status)
    check(f"STATUS quotes the same cost in the recovery recipe (the full {hours} h)",
          f"the full {hours} h" in status)

    print("README's paths:")
    linked = {t.split("#")[0] for t in re.findall(r"\]\(([^)]+)\)", readme)}
    absent = sorted(t for t in linked
                    if t and not t.startswith(("http", "mailto", "#"))
                    and not (REPO / t).exists())
    check("every path README links to exists", not absent, str(absent))

    if problems:
        print(f"\n{len(problems)} README claim(s) drifted: {problems}")
        return 1
    print("\nevery quoted README number matches its artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
