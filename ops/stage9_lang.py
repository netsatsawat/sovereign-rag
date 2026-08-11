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
  scoring    containment and abstention on NFKC + casefold + [\\W_]-stripped
             text, which is language-neutral (no internal spaces to
             reproduce, accents preserved).

Constant across languages: k=5 retrieved contexts trimmed to 2,500 chars;
glimmer via llama-server :8095 with "Reasoning strength: low" + 1024 tokens
(the Stage-8 accommodations, same reasons); qwen3:8b via ollama with
think:false + 64 tokens (the study's convention); temperature 0, seed 0;
checkpointed per (arm, qi); single-gold datasets so strict@k = hits@k.

    python ops/stage9_lang.py --lang th --retrieval
    python ops/stage9_lang.py --lang th --model glimmer
    python ops/stage9_lang.py --lang th --model qwen8b
    python ops/stage9_lang.py --lang th --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
    """src/bm25.py's scoring (k1=1.5, b=0.75) with a language-aware
    tokeniser; the shared class's normalize() deletes non-Latin text."""

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
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", s or "").casefold())


def contained(golds: list[str], answer: str) -> bool:
    na = norm(answer)
    return any(norm(g) and norm(g) in na for g in golds)


def abstained(lang: str, answer: str) -> bool:
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

    if p["emb"].exists():
        dv = np.load(p["emb"])
    else:
        dv = embed([c["context"] for c in corpus])
        np.save(p["emb"], dv)
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

    done: set[tuple[str, int]] = set()
    if p["rows"].exists():
        for line in p["rows"].open():
            try:
                r = json.loads(line)
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
    return round(min(1.0, 2 * p), 4)


def analyze(lang: str) -> None:
    p = paths(lang)
    rows: dict[tuple[str, int], dict] = {}
    for line in p["rows"].open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not str(r.get("done_reason", "")).startswith("error"):
            rows[(r["arm"], r["qi"])] = r

    arms = sorted({a for (a, _q) in rows})
    qis = sorted({q for (_a, q) in rows})
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--model", choices=["glimmer", "qwen8b"])
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.retrieval:
        run_retrieval(args.lang)
    elif args.model:
        run_generation(args.lang, args.model)
    elif args.analyze:
        analyze(args.lang)
    else:
        ap.error("pick one of --retrieval / --model / --analyze")
