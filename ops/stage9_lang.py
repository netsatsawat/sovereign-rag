"""Stage 9 — the multilingual replication: five languages, same two questions.

Muse Glimmer's card claims 100+ languages. Stage 8's two live findings — the
escape hatch firing on evidence-complete queries, and the contamination floor
under closed-book — were measured in English on English news. This stage asks
the same paired questions (Glimmer vs qwen3:8b, closed-book vs BM25 RAG) in
Thai, Japanese, Chinese, Spanish and Vietnamese, each on a native-language
extractive QA corpus normalised to one schema:

    data/multiling/{lang}/corpus.jsonl   {"article_id", "context"}
    data/multiling/{lang}/queries.jsonl  {"qi", "question_id", "question",
                                          "gold_article", "golds": [...]}

Per-language machinery that has to differ, and why:
  tokeniser  th=pythainlp newmm, ja=janome, zh=jieba (no-space scripts;
             src/bm25.py's normalize() deletes non-Latin text outright);
             es/vi = unicode \\w+ regex (the shared normalize also deletes
             accents, so even Spanish cannot reuse it).
  prompts    native-language RAG/closed prompts, same structure as
             PROMPT_RAG/PROMPT_CLOSED, native escape phrase.
  scoring    containment and abstention on NFKC + casefold text with every
             [\\W_] character deleted. No internal spaces to reproduce, and
             Latin precomposed accents survive (á/ñ are word characters) —
             but the strip removes ALL combining marks, so Thai
             vowels-above/below, tone marks and thanthakhat vanish from
             gold and answer alike (NFKC first decomposes SARA AM:
             norm('น้ำ') == 'นา'). One committed verdict rides on that
             leniency — th rag_glimmer qi=28, where the model dropped a
             silent-letter mark ('เซอรนัก' vs gold 'เซอร์นัก') and still
             matches — and words differing only in tone marks collapse to
             the same normed string, an accepted false-positive channel on
             this single-gold extractive task.

Constant across languages: k=5 retrieved contexts trimmed to 2,500 chars;
glimmer via llama-server :8095 with "Reasoning strength: low" + 1024 tokens
(the Stage-8 accommodations, same reasons); qwen3:8b via ollama with
think:false + 64 tokens (the study's convention); temperature 0, seed 0;
checkpointed per (arm, qi); single-gold datasets so strict@k = hits@k.

    python ops/stage9_lang.py --lang th --retrieval
    python ops/stage9_lang.py --lang th --model glimmer
    python ops/stage9_lang.py --lang th --model qwen8b
    python ops/stage9_lang.py --lang th --analyze
    python ops/stage9_lang.py --lang th --calibration   # optional, derived:
                                        # STAGE9.md's calibration-reversal row
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
K = 5
CTX_TRIM = 2500

PROMPTS = {
    "th": {
        "rag": "จงตอบคำถามโดยใช้ข้อมูลจากบริบทด้านล่างเท่านั้น ตอบให้สั้นและตรงประเด็น: ให้ตอบเฉพาะคำตอบ ไม่ต้องมีคำอธิบายเพิ่ม หากบริบทไม่มีคำตอบ ให้ตอบว่า: ข้อมูลไม่เพียงพอ\n\nบริบท:\n{context}\n\nคำถาม: {q}\n\nคำตอบ:",
        "closed": "จงตอบคำถามให้สั้นและตรงประเด็น: ให้ตอบเฉพาะคำตอบ ไม่ต้องมีคำอธิบายเพิ่ม หากไม่ทราบคำตอบ ให้ตอบว่า: ข้อมูลไม่เพียงพอ\n\nคำถาม: {q}\n\nคำตอบ:",
        "escape": ["ข้อมูลไม่เพียงพอ"],
    },
    "ja": {
        "rag": "以下の文脈の情報のみを使って質問に答えてください。簡潔に直接答えること。答えのみを書き、前置きは不要です。文脈に答えがない場合は「情報不足」とだけ答えてください。\n\n文脈:\n{context}\n\n質問: {q}\n\n回答:",
        "closed": "質問に簡潔に直接答えてください。答えのみを書き、前置きは不要です。分からない場合は「情報不足」とだけ答えてください。\n\n質問: {q}\n\n回答:",
        "escape": ["情報不足"],
    },
    "zh": {
        "rag": "仅使用下面的上下文回答问题。回答要简短直接：只给出答案，不要多余说明。如果上下文中没有答案，请只回答：信息不足\n\n上下文:\n{context}\n\n问题: {q}\n\n回答:",
        "closed": "请简短直接地回答问题：只给出答案，不要多余说明。如果不知道答案，请只回答：信息不足\n\n问题: {q}\n\n回答:",
        "escape": ["信息不足"],
    },
    "es": {
        "rag": "Responde la pregunta usando SOLO el contexto siguiente. Sé directo y breve: da solo la respuesta, sin preámbulo. Si el contexto no contiene la respuesta, responde exactamente: información insuficiente\n\nCONTEXTO:\n{context}\n\nPREGUNTA: {q}\n\nRESPUESTA:",
        "closed": "Responde la pregunta de forma directa y breve: solo la respuesta, sin preámbulo. Si no sabes la respuesta, responde exactamente: información insuficiente\n\nPREGUNTA: {q}\n\nRESPUESTA:",
        "escape": ["información insuficiente"],
    },
    "vi": {
        "rag": "Chỉ sử dụng ngữ cảnh dưới đây để trả lời câu hỏi. Trả lời ngắn gọn và trực tiếp: chỉ đưa ra câu trả lời, không cần mở đầu. Nếu ngữ cảnh không chứa câu trả lời, hãy trả lời chính xác: không đủ thông tin\n\nNGỮ CẢNH:\n{context}\n\nCÂU HỎI: {q}\n\nTRẢ LỜI:",
        "closed": "Trả lời câu hỏi ngắn gọn và trực tiếp: chỉ đưa ra câu trả lời, không cần mở đầu. Nếu không biết câu trả lời, hãy trả lời chính xác: không đủ thông tin\n\nCÂU HỎI: {q}\n\nTRẢ LỜI:",
        "escape": ["không đủ thông tin"],
    },
}

SYSMSG_GLIMMER = "Reasoning strength: low"


def paths(lang: str) -> dict[str, Path]:
    return {"corpus": ROOT / "data" / "multiling" / lang / "corpus.jsonl",
            "queries": ROOT / "data" / "multiling" / lang / "queries.jsonl",
            "emb": ROOT / "data" / "multiling" / lang / "emb.npy",
            "rows": ROOT / "reports" / f"stage9_{lang}.jsonl",
            "retrieval": ROOT / "reports" / f"stage9_{lang}_retrieval.json",
            "summary": ROOT / "reports" / f"stage9_{lang}.json"}


def load(lang: str) -> tuple[list[dict], list[dict]]:
    p = paths(lang)
    corpus = [json.loads(l) for l in p["corpus"].open()]
    for c in corpus:
        c["context"] = c["context"][:CTX_TRIM]
    queries = [json.loads(l) for l in p["queries"].open()]
    return corpus, queries


# ------------------------------------------------------------------ tokenisers

_engines: dict[str, object] = {}


def tokens(lang: str, text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    if lang == "th":
        if "th" not in _engines:
            from pythainlp.tokenize import word_tokenize
            _engines["th"] = word_tokenize
        words = _engines["th"](text, engine="newmm")
    elif lang == "ja":
        if "ja" not in _engines:
            from janome.tokenizer import Tokenizer
            _engines["ja"] = Tokenizer(wakati=True)
        words = list(_engines["ja"].tokenize(text))
    elif lang == "zh":
        if "zh" not in _engines:
            import jieba
            jieba.setLogLevel(60)
            _engines["zh"] = jieba
        words = list(_engines["zh"].cut(text))
    else:
        words = re.findall(r"\w+", text)
    return [w for w in (x.strip() for x in words) if w and re.search(r"\w", w)]


class LangBM25:
    """src/bm25.py's scoring formula with a language-aware tokeniser (the
    shared class's normalize() deletes non-Latin text) — but NOT its
    parameters: b matches at 0.75, while k1 here is 1.5 vs src/bm25.py's
    1.2. Every committed stage-9 retrieval and RAG number was produced at
    k1=1.5, so the difference is recorded rather than repaired: stage-8 vs
    stage-9 BM25 arms share the formula, not the parameterization."""

    K1, B = 1.5, 0.75

    def __init__(self, lang: str, docs: list[str]):
        self.toks = [tokens(lang, d) for d in docs]
        self.lang = lang
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / max(self.N, 1)
        self.df: Counter = Counter()
        for t in self.toks:
            self.df.update(set(t))
        self.tf = [Counter(t) for t in self.toks]

    def top_k(self, query: str, k: int) -> list[int]:
        scores = np.zeros(self.N)
        for w in tokens(self.lang, query):
            df = self.df.get(w)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i in range(self.N):
                f = self.tf[i].get(w)
                if f:
                    dl = len(self.toks[i])
                    scores[i] += idf * f * (self.K1 + 1) / (
                        f + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order[:k] if scores[i] > 0]


# ------------------------------------------------------------------ dense

def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": "qwen3-embedding:0.6b",
                           "input": [t[:6000] for t in texts[i:i + batch]]}).encode()
        req = urllib.request.Request("http://localhost:11434/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out.extend(json.loads(r.read())["embeddings"])
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ------------------------------------------------------------------ scoring

def norm(s: str) -> str:
    # Strips every separator, so a purely numeric gold can match inside a
    # longer digit run: ja closed_qwen8b qi=96 (gold '4') was scored correct
    # against '574族', the only committed row where this decides a verdict
    # (ja closed-book 8B containment 12.0 vs 11.0 with digit boundaries).
    # A boundary-aware fix would change committed numbers — do not alter
    # without an explicit decision to re-run --analyze for every language.
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", s or "").casefold())


def contained(golds: list[str], answer: str) -> bool:
    na = norm(answer)
    return any(norm(g) and norm(g) in na for g in golds)


def abstained(lang: str, answer: str) -> bool:
    # Counts use of the PRESCRIBED escape phrase (plus the English fallback),
    # not refusal in general: free-form refusals such as th 'ไม่ทราบ' (three
    # committed th rows) are NOT counted. Report abstained_pct as
    # "answered with the prescribed escape phrase"; widening this list would
    # change committed th numbers.
    na = norm(answer)
    hatches = [norm(e) for e in PROMPTS[lang]["escape"]] + ["insufficientinformation"]
    return any(h in na for h in hatches)


# ------------------------------------------------------------------ retrieval

def run_retrieval(lang: str) -> None:
    p = paths(lang)
    corpus, queries = load(lang)
    print(f"[{lang}] corpus {len(corpus)} articles, {len(queries)} queries", flush=True)
    aid_of = [c["article_id"] for c in corpus]

    t0 = time.time()
    bm = LangBM25(lang, [c["context"] for c in corpus])
    bm_ranks = [bm.top_k(q["question"], 10) for q in queries]
    print(f"[{lang}] bm25 indexed+ranked in {time.time()-t0:.0f}s", flush=True)

    def corpus_fingerprint() -> dict:
        h = hashlib.sha256()
        for c in corpus:
            h.update((c["article_id"] + "\x00" + c["context"] + "\x01").encode())
        return {"rows": len(corpus), "corpus_sha256": h.hexdigest()[:16]}

    emb_meta = p["emb"].with_suffix(".meta.json")
    if p["emb"].exists():
        # Guard against a stale cache: emb.npy is trusted blindly otherwise,
        # and a regenerated corpus would map argsort indices to the wrong
        # article_ids with no error. (raise, not assert, for all three
        # checks below: asserts are stripped under python -O, which would
        # silently disable exactly the defense they exist to provide.)
        dv = np.load(p["emb"])
        if dv.shape[0] != len(corpus):
            raise SystemExit(
                f"emb.npy rows {dv.shape[0]} != corpus {len(corpus)} — stale "
                f"embedding cache; delete {p['emb']} and re-run --retrieval")
        # A missing fingerprint is a hard failure, not a skip: the shape
        # assert alone cannot tell a regenerated corpus with the same
        # article count from the one this cache was embedded from — and
        # because the meta file is only written on a cache miss, skipping
        # here would leave pre-guard caches unverified forever (the guard
        # could never self-heal while emb.npy exists). Re-embedding is
        # deterministic (temperature-free qwen3-embedding), so deleting the
        # cache reproduces the committed hits@k.
        if not emb_meta.exists():
            raise SystemExit(
                f"{p['emb']} has no {emb_meta.name} fingerprint, so the cache "
                f"cannot be verified against the current corpus — stale "
                f"embedding cache; delete {p['emb']} and re-run --retrieval")
        if json.loads(emb_meta.read_text()) != corpus_fingerprint():
            raise SystemExit(
                f"{emb_meta} does not match the current corpus — stale "
                f"embedding cache; delete {p['emb']} and re-run --retrieval")
    else:
        dv = embed([c["context"] for c in corpus])
        np.save(p["emb"], dv)
        emb_meta.write_text(json.dumps(corpus_fingerprint()))
    qv = embed([q["question"] for q in queries])
    dense_ranks = [list(map(int, np.argsort(-(dv @ qv[i]))[:10]))
                   for i in range(len(queries))]

    def hits(ranks: list[list[int]], k: int) -> float:
        return round(100 * sum(
            q["gold_article"] in {aid_of[i] for i in r[:k]}
            for q, r in zip(queries, ranks)) / len(queries), 2)

    res = {"lang": lang, "corpus_articles": len(corpus), "n_queries": len(queries),
           "bm25": {"hits1": hits(bm_ranks, 1), "hits5": hits(bm_ranks, 5),
                    "hits10": hits(bm_ranks, 10)},
           "dense_qwen06b": {"hits1": hits(dense_ranks, 1),
                             "hits5": hits(dense_ranks, 5),
                             "hits10": hits(dense_ranks, 10)}}
    p["retrieval"].write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ generation

def gen_glimmer(prompt: str) -> tuple[str, dict]:
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSMSG_GLIMMER},
                     {"role": "user", "content": prompt}],
        "temperature": 0, "seed": 0, "max_tokens": 1024}).encode()
    req = urllib.request.Request("http://localhost:8095/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = d.get("usage") or {}
    return (msg.get("content") or "").strip(), {
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "gen_tokens": u.get("completion_tokens", 0),
        "done_reason": ch.get("finish_reason"),
        "thinking_len": len(msg.get("reasoning_content") or "")}


def gen_qwen8b(prompt: str) -> tuple[str, dict]:
    body = json.dumps({
        "model": "qwen3:8b", "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 64},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return (d.get("response") or "").strip(), {
        "wall_s": round(time.time() - t0, 1),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "gen_tokens": d.get("eval_count", 0),
        "done_reason": d.get("done_reason"),
        "thinking_len": 0}


def run_generation(lang: str, model: str) -> None:
    gen = {"glimmer": gen_glimmer, "qwen8b": gen_qwen8b}[model]
    p = paths(lang)
    corpus, queries = load(lang)
    bm = LangBM25(lang, [c["context"] for c in corpus])

    # qi is only an index; question_id is the identity. Any existing row
    # whose question_id disagrees with the current queries.jsonl means the
    # query file changed under the checkpoint — resuming would silently mix
    # answers to different questions under one qi. Hard-fail instead.
    # (SystemExit deliberately escapes the except Exception below.)
    qid_of = {q["qi"]: q["question_id"] for q in queries}
    done: set[tuple[str, int]] = set()
    if p["rows"].exists():
        for line in p["rows"].open():
            try:
                r = json.loads(line)
                if r.get("question_id") != qid_of.get(r.get("qi")):
                    raise SystemExit(
                        f"{p['rows']}: row (arm={r.get('arm')}, qi={r.get('qi')}) "
                        f"has question_id {r.get('question_id')!r} but "
                        f"queries.jsonl says {qid_of.get(r.get('qi'))!r} — "
                        f"queries.jsonl changed under the checkpoint; resolve "
                        f"before resuming")
                if not str(r.get("done_reason", "")).startswith("error"):
                    done.add((r["arm"], r["qi"]))
            except Exception:
                pass

    t0 = time.time()
    with p["rows"].open("a") as fh:
        for n_i, q in enumerate(queries, 1):
            for kind in ("closed", "rag"):
                arm = f"{kind}_{model}"
                if (arm, q["qi"]) in done:
                    continue
                if kind == "closed":
                    prompt = PROMPTS[lang]["closed"].format(q=q["question"])
                else:
                    top = bm.top_k(q["question"], K)
                    ctx = "\n\n".join(corpus[i]["context"] for i in top)
                    prompt = PROMPTS[lang]["rag"].format(context=ctx, q=q["question"])
                try:
                    ans, meta = gen(prompt)
                except Exception as ex:
                    ans, meta = "", {"wall_s": None, "prompt_tokens": 0,
                                     "gen_tokens": 0,
                                     "done_reason": f"error:{type(ex).__name__}",
                                     "thinking_len": 0}
                fh.write(json.dumps({"qi": q["qi"], "arm": arm,
                                     "question_id": q["question_id"],
                                     "golds": q["golds"], "answer": ans, **meta},
                                    ensure_ascii=False) + "\n")
                fh.flush()
            if n_i % 10 == 0 or n_i == len(queries):
                print(f"  [{lang}/{model}] {n_i}/{len(queries)} · "
                      f"{(time.time()-t0)/60:.0f}m", flush=True)
    print(f"stage9 {lang} {model} arms complete", flush=True)


# ------------------------------------------------------------------ analysis

def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    # KNOWN DISPLAY DEFECT: rounding to 4 dp reports 0.0 whenever p < 5e-5,
    # a value an exact binomial test cannot produce. All ten committed
    # stage9 rag_over_closed pairings carry mcnemar_p = 0.0 this way (true
    # values 1e-15..3e-24) — quote them as p < 0.0001, never "p = 0.0".
    # Returning the unrounded value is the right fix but rewrites committed
    # summary JSONs on the next --analyze; change only with that decision.
    return min(1.0, 2 * p)   # full precision; display rounding is prose's job


def analyze(lang: str) -> None:
    p = paths(lang)
    # Same identity guard as the resume path: every row's question_id must
    # match what the CURRENT queries.jsonl says its qi means, or the paired
    # stats would silently compare answers to different questions.
    qid_of = {q["qi"]: q["question_id"]
              for q in (json.loads(l) for l in p["queries"].open())}
    rows: dict[tuple[str, int], dict] = {}
    for line in p["rows"].open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("question_id") != qid_of.get(r.get("qi")):
            raise SystemExit(
                f"{p['rows']}: row (arm={r.get('arm')}, qi={r.get('qi')}) has "
                f"question_id {r.get('question_id')!r} but queries.jsonl says "
                f"{qid_of.get(r.get('qi'))!r} — rows and queries.jsonl are out "
                f"of sync; refusing to analyze")
        if not str(r.get("done_reason", "")).startswith("error"):
            rows[(r["arm"], r["qi"])] = r

    arms = sorted({a for (a, _q) in rows})
    qis = sorted({q for (_a, q) in rows})

    # Completeness check: the qi universe above is whatever non-error rows
    # exist, so an interrupted run yields a summary computed on the
    # surviving subset — same shape as the committed artifact,
    # distinguishable only by its small "n" fields. Warn loudly; stderr
    # only, so the summary JSON stays byte-identical on complete data.
    # (Rows for qi absent from queries.jsonl already hard-fail the
    # question_id guard above, so only the incomplete direction is checked.)
    expected = len(qid_of)
    for arm in arms:
        n_arm = sum(1 for (a, _q) in rows if a == arm)
        if n_arm != expected:
            print(f"WARNING: [{lang}] arm {arm} has {n_arm}/{expected} "
                  f"scored rows — the summary is computed on an incomplete "
                  f"run; treat every number in it as partial",
                  file=sys.stderr)

    summary: dict = {"lang": lang, "n": len(qis), "arms": {}, "paired": {}}
    if p["retrieval"].exists():
        summary["retrieval"] = json.loads(p["retrieval"].read_text())

    for arm in arms:
        rs = [rows[(arm, qi)] for qi in qis if (arm, qi) in rows]
        ok = sum(contained(r["golds"], r["answer"]) for r in rs)
        summary["arms"][arm] = {
            "n": len(rs),
            "containment": round(100 * ok / len(rs), 2) if rs else None,
            "wilson95": wilson(ok, len(rs)),
            "abstained_pct": round(100 * sum(abstained(lang, r["answer"])
                                             for r in rs) / len(rs), 1) if rs else None,
            "mean_gen_tokens": round(sum(r["gen_tokens"] for r in rs) / len(rs), 1)
            if rs else None}

    def paired(name: str, a: str, b: str) -> None:
        common = [qi for qi in qis if (a, qi) in rows and (b, qi) in rows]
        if not common:
            return
        x = y = 0
        for qi in common:
            ra = contained(rows[(a, qi)]["golds"], rows[(a, qi)]["answer"])
            rb = contained(rows[(b, qi)]["golds"], rows[(b, qi)]["answer"])
            x += ra and not rb
            y += rb and not ra
        summary["paired"][name] = {"n": len(common), "a_only": x, "b_only": y,
                                   "delta_pp": round(100 * (x - y) / len(common), 2),
                                   "mcnemar_p": mcnemar_exact(x, y)}

    paired("rag_glimmer_vs_rag_8b", "rag_glimmer", "rag_qwen8b")
    paired("closed_glimmer_vs_closed_8b", "closed_glimmer", "closed_qwen8b")
    paired("glimmer_rag_over_closed", "rag_glimmer", "closed_glimmer")
    paired("qwen8b_rag_over_closed", "rag_qwen8b", "closed_qwen8b")

    p["summary"].write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def calibration(lang: str) -> None:
    """STAGE9.md's 'calibration reversal' table as a command, not a prose
    spec: glimmer RAG abstention crossed with whether the BM25@5 arm
    actually retrieved the gold article, recomputed from the committed
    corpus/queries/rows via a deterministic LangBM25 rebuild (no server
    needed). Purely derived — writes reports/stage9_{lang}_calibration.json
    and touches nothing --analyze depends on."""
    p = paths(lang)
    corpus, queries = load(lang)
    aid_of = [c["article_id"] for c in corpus]
    bm = LangBM25(lang, [c["context"] for c in corpus])
    hit5 = {q["qi"]: q["gold_article"] in {aid_of[i] for i in bm.top_k(q["question"], K)}
            for q in queries}

    # Same identity guard as analyze(): every row's question_id must match
    # what the CURRENT queries.jsonl says its qi means.
    qid_of = {q["qi"]: q["question_id"] for q in queries}
    rows: dict[int, dict] = {}
    for line in p["rows"].open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("question_id") != qid_of.get(r.get("qi")):
            raise SystemExit(
                f"{p['rows']}: row (arm={r.get('arm')}, qi={r.get('qi')}) has "
                f"question_id {r.get('question_id')!r} but queries.jsonl says "
                f"{qid_of.get(r.get('qi'))!r} — rows and queries.jsonl are out "
                f"of sync; refusing to compute the calibration cross")
        if r.get("arm") == "rag_glimmer" and \
                not str(r.get("done_reason", "")).startswith("error"):
            rows[r["qi"]] = r

    out: dict = {"lang": lang, "arm": "rag_glimmer", "k": K, "n": len(rows)}
    for label, flag in (("gold_retrieved", True), ("gold_missed", False)):
        qs = [qi for qi in rows if hit5[qi] == flag]
        nab = sum(abstained(lang, rows[qi]["answer"]) for qi in qs)
        out[label] = {"n": len(qs), "abstained": nab,
                      "abstain_pct": round(100 * nab / len(qs), 1) if qs else None}
    cal = ROOT / "reports" / f"stage9_{lang}_calibration.json"
    cal.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--model", choices=["glimmer", "qwen8b"])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    args = ap.parse_args()
    if args.retrieval:
        run_retrieval(args.lang)
    elif args.model:
        run_generation(args.lang, args.model)
    elif args.analyze:
        analyze(args.lang)
    elif args.calibration:
        calibration(args.lang)
    else:
        ap.error("pick one of --retrieval / --model / --analyze / --calibration")
