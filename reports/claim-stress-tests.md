# Stage 6: pushing on the claims until they broke or hardened

11 Aug 2026. Five follow-up experiments run because "verified against
artifacts" is not the same as "sufficient evidence." Two claims hardened, one
broke and was rewritten, and the corpus itself confessed.

## Hardened

**BM25's win replicates on the full 609-article corpus**: strict@10 48.74 on
2,255 queries (2.4× distractors, all strata restored), beating dense by
+12.95 [10.91, 14.94], a larger margin than the subset's. And it survives
the outlet-oracle control: stripping the outlet names 99.1% of queries carry
costs 1.16 pts, CI touching zero.

**The graph verdict deepens.** 95-100% of multi-doc queries are bridged in
the graph (shared entity or edge). Connectivity was never the problem. The
extractions are: an 8B on news text builds a citation graph (headlines as
EVENTs, outlet domains as LOCATIONs, dates as entities, metadata triangles at
strength 10). Community structure is excellent (modularity 0.803) and it is
communities of boilerplate; community-boosted retrieval scores 8.47 vs the
plain graph arm's 13.32, and 54.7% of multi-doc queries span communities, a
structural ceiling.

## Broken, and rewritten

**The synthesis-bottleneck claim did not survive its causal test.** The 27B
on byte-identical contexts and prompts lost to the 8B on both target strata,
22:5 on comparison (p=0.0015) and 15:1 on temporal (p=0.0005), by abstaining
on 84% of RAG calls. And the strata themselves are degenerate for containment
scoring: comparison golds are 96% yes/no, and a constant "Yes" scores 60.0%,
above every model arm. What looked like a synthesis failure decomposes into
benchmark anatomy plus abstention calibration. The clean survivor: on the
entity-answer stratum with no degenerate shortcut, RAG's lift is real and
huge (57.7 → 95.2).

## The corpus confession

MultiHop-RAG's comparison/temporal questions are META-questions about
articles ("do the TechCrunch article and the Hacker News article both
report…"), with the natural join key being the article, not a world entity.
That is simultaneously why a knowledge graph of world entities was the wrong
tool, why the outlet names leak into queries, and why the answers collapse to
yes/no. The benchmark tests article-pair verification wearing multi-hop
reasoning's clothes.

Artifacts: stage5_full_corpus.json, stage6_27b.json + stage6_27b.jsonl,
stage2_graph.json (community_analysis), stage1_retrieval.json
(outlet_strip_control). Every number recomputes.
