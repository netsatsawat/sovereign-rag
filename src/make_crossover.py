"""Emit reports/crossover.html from stage1_retrieval.json.

FR-U2: one self-contained file. No network fetch, no CDN, no npm, no webfont;
the results JSON is inlined at build time, so file:// works with the stack off.
FR-U3: the renderer reads precomputed `reportable` flags; it computes no
significance decision. FR-U4: every number shown originates in the committed
JSON this script inlines; regenerate with `python src/make_crossover.py`.

    python src/make_crossover.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "reports" / "stage1_retrieval.json"
OUT = ROOT / "reports" / "crossover.html"

data = json.loads(SRC.read_text())

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sovereign-rag · retrieval-layer crossover</title>
<style>
  :root{--bg:#fff;--fg:#16181d;--dim:#5b6472;--line:#e4e7ec;--card:#f7f8fa;
        --blue:#2563eb;--gold:#b45309;--red:#b42318;--green:#0f7b45}
  @media (prefers-color-scheme: dark){
    :root{--bg:#0d0f13;--fg:#e8eaee;--dim:#98a2b3;--line:#252a33;--card:#151920;
          --blue:#7aa2f7;--gold:#e0a458;--red:#f27a70;--green:#5fd39b}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  .wrap{max-width:880px;margin:0 auto;padding:2rem 1rem 4rem}
  h1{font-size:1.5rem;margin:0 0 .2rem} h2{font-size:1.1rem;margin:2rem 0 .5rem}
  .sub{color:var(--dim);font-size:.85rem;margin:0 0 1rem}
  .banner{border-left:3px solid var(--gold);background:var(--card);
          padding:.6rem .9rem;font-size:.85rem;color:var(--dim);margin:1rem 0}
  .row{display:flex;align-items:center;gap:.5rem;margin:.18rem 0}
  .lbl{width:120px;font-size:.8rem;color:var(--dim);text-align:right;flex:none}
  .bar{height:16px;border-radius:3px;min-width:2px}
  .val{font-size:.8rem;font-variant-numeric:tabular-nums}
  .b600{opacity:1}.b1200{opacity:.55}
  .bm25{background:var(--blue)}.dense{background:var(--gold)}.hybrid{background:var(--red)}
  .stratum{margin:1.1rem 0;padding:.8rem;border:1px solid var(--line);border-radius:8px}
  .stratum h3{margin:.1rem 0 .5rem;font-size:.95rem}
  .winner{font-size:1.05rem;font-weight:700;margin:.4rem 0}
  input[type=range]{width:160px;accent-color:var(--blue)}
  .mix{display:flex;gap:1.2rem;flex-wrap:wrap;align-items:center;
       background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.8rem}
  .mix label{font-size:.8rem;color:var(--dim)}
  table{border-collapse:collapse;font-size:.82rem;width:100%}
  th,td{padding:.35rem .5rem;border-bottom:1px solid var(--line);text-align:left}
  th{color:var(--dim);font-weight:600}
  .grey{opacity:.42}
  .tag{font-size:.68rem;font-weight:700;padding:.08rem .4rem;border-radius:4px;
       background:var(--card);border:1px solid var(--line);color:var(--dim)}
  button{font-size:.75rem;background:var(--card);border:1px solid var(--line);
         color:var(--fg);border-radius:5px;padding:.2rem .55rem;cursor:pointer}
</style>
</head>
<body><div class="wrap">
<h1>At what question mix does each retriever start paying for itself?</h1>
<p class="sub">sovereign-rag · Stage 1, retrieval layer · MultiHop-RAG tech+business,
253 articles · one 24&nbsp;GB M5, no API key · metric: <b>strict@10</b>, every gold
document for the query retrieved in the top 10 chunks</p>

<div class="banner"><b>Scope labels, before any number.</b> Retrieval layer only: no
generation, no LLM judge; the agentic and graph arms are pending (graph index in progress).
<span id="satrange"></span> strict@k is the replacement declared in machine-measurement.md before Stage 1 ran
(chosen post-hoc from two candidate metrics). Deltas whose 95% CI includes zero are
greyed; the flag is computed offline, never in this page.</div>

<h2>Per-stratum, per-retriever</h2>
<div id="strata"></div>

<h2>Your traffic mix decides your winner</h2>
<div class="mix" id="mix"></div>
<div class="winner" id="winner"></div>
<div id="mixbars"></div>

<h2>The fixed-token control: the sign flips</h2>
<p class="sub">Comparing 600@k10 to 1200@k10 confounds chunk size with retrieved tokens
(k=10 of 1200-token chunks reads 2× the text). At a fixed ~6,000-token budget:</p>
<table id="ftc"></table>

<h2>Pairwise deltas (strict@10, overall)</h2>
<table id="deltas"></table>

<h2>Retrieval cost</h2>
<table id="cost"></table>

<p class="sub" id="foot"></p>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const ARMS = ["bm25","dense","hybrid"], BUDGETS=[600,1200];
const cfg=(b,a,k=10)=>D.configs[`${b}_k${k}_${a}`];
const allH10=Object.values(D.configs).filter(c=>c.k===10).map(c=>c.overall.hitsk);
const h10lo=Math.min(...allH10).toFixed(1), h10hi=Math.max(...allH10).toFixed(1);
const nice = {bm25:"BM25", dense:"dense", hybrid:"hybrid (RRF)"};
const strata = Object.keys(cfg(600,"bm25").by_type);

function bar(host, label, cls, pct, extra){
  const r=document.createElement('div'); r.className='row';
  r.innerHTML=`<span class="lbl">${label}</span>
    <div class="bar ${cls}" style="width:${pct*4.2}px"></div>
    <span class="val">${pct.toFixed(1)}%${extra||''}</span>`;
  host.appendChild(r);
}

document.getElementById('satrange').textContent=`Hits@10 is saturated on this corpus (${h10lo}-${h10hi}% across all k=10 configs) and is not a headline;`;
const sh=document.getElementById('strata');
for(const s of strata){
  const box=document.createElement('div'); box.className='stratum';
  const n=cfg(600,"bm25").by_type[s].n;
  box.innerHTML=`<h3>${s.replace('_query','')} <span class="tag">n=${n} · native, third-party authored</span></h3>`;
  for(const b of BUDGETS) for(const a of ARMS){
    bar(box, `${nice[a]} @${b}`, `${a} b${b}`, cfg(b,a).by_type[s].strict);
  }
  sh.appendChild(box);
}

const mixHost=document.getElementById('mix');
const sliders={};
const presets={"native mix":null,"lookup-heavy":{inference_query:70,comparison_query:20,temporal_query:10},
               "analyst desk":{inference_query:25,comparison_query:45,temporal_query:30}};
const totN = strata.reduce((t,s)=>t+cfg(600,"bm25").by_type[s].n,0);
const nativePct = s => Math.round(100*cfg(600,"bm25").by_type[s].n/totN);
for(const s of strata){
  const wrap=document.createElement('label');
  wrap.innerHTML=`${s.replace('_query','')} <input type="range" min="0" max="100" value="${nativePct(s)}" id="w_${s}">
                  <span id="v_${s}"></span>`;
  mixHost.appendChild(wrap);
}
const pb=document.createElement('span');
for(const [name,p] of Object.entries(presets)){
  const b=document.createElement('button'); b.textContent=name;
  b.onclick=()=>{for(const s of strata){
    const el=document.getElementById('w_'+s);
    el.value = p? (p[s]||0) : nativePct(s);} render();};
  pb.appendChild(b); pb.append(' ');
}
mixHost.appendChild(pb);

function render(){
  let tot=0; const w={};
  for(const s of strata){w[s]=+document.getElementById('w_'+s).value; tot+=w[s];}
  for(const s of strata) document.getElementById('v_'+s).textContent=
    tot? Math.round(100*w[s]/tot)+'%':'0%';
  const scores=[];
  for(const b of BUDGETS) for(const a of ARMS){
    let v=0; for(const s of strata) v+=w[s]*cfg(b,a).by_type[s].strict;
    scores.push({k:`${nice[a]} @${b}`, cls:a, v: tot? v/tot:0});
  }
  scores.sort((x,y)=>y.v-x.v);
  document.getElementById('winner').textContent=
    `Winner at this mix: ${scores[0].k}, weighted strict@10 ${scores[0].v.toFixed(1)}%`;
  const mb=document.getElementById('mixbars'); mb.innerHTML='';
  for(const s of scores) bar(mb, s.k, s.cls, s.v);
}
for(const s of strata) document.getElementById('w_'+s).oninput=render;
render();

const ftc=document.getElementById('ftc');
ftc.innerHTML='<tr><th>retriever</th><th>600 @ k=5</th><th>600 @ k=10</th><th>1200 @ k=5</th><th>1200 @ k=10</th></tr>'+
 ARMS.map(a=>`<tr><td>${nice[a]}</td>
   <td>${cfg(600,a,5).overall.strict.toFixed(2)}%</td>
   <td>${cfg(600,a,10).overall.strict.toFixed(2)}%</td>
   <td>${cfg(1200,a,5).overall.strict.toFixed(2)}%</td>
   <td>${cfg(1200,a,10).overall.strict.toFixed(2)}%</td></tr>`).join('')+
 `<tr><td colspan=5 style="color:var(--dim)">fixed ~6,000-token budget: 600@k10 − 1200@k5 (BM25) = `+
 `${D.deltas['fixed_tokens_600k10_vs_1200k5_bm25_strict'].delta_pts>0?'+':''}`+
 `${D.deltas['fixed_tokens_600k10_vs_1200k5_bm25_strict'].delta_pts} `+
 `[${D.deltas['fixed_tokens_600k10_vs_1200k5_bm25_strict'].ci95_pts}], a slot effect, not a chunk-size effect; `+
 `at matched k, 1200 wins both (+${D.deltas['matched_k5_1200_vs_600_bm25_strict'].delta_pts}, `+
 `+${D.deltas['matched_k10_1200_vs_600_bm25_strict'].delta_pts})</td></tr>`;

const dt=document.getElementById('deltas');
let rows='<tr><th>comparison</th><th>Δ pts</th><th>95% CI</th><th></th></tr>';
for(const b of BUDGETS) for(const pair of [["hybrid","bm25"],["hybrid","dense"],["bm25","dense"]]){
  const k=`${b}_k10_${pair[0]}_vs_${pair[1]}_strict`; const d=D.deltas[k]; if(!d) continue;
  const grey=d.reportable?'':' class="grey"';
  rows+=`<tr${grey}><td>@${b} ${nice[pair[0]]} − ${nice[pair[1]]}</td>
    <td>${d.delta_pts>0?'+':''}${d.delta_pts}</td>
    <td>[${d.ci95_pts[0]}, ${d.ci95_pts[1]}]</td>
    <td>${d.reportable?'':'not significant after Holm-Bonferroni, not reported'}</td></tr>`;
}
dt.innerHTML=rows;

const ct=document.getElementById('cost');
ct.innerHTML='<tr><th>config</th><th>index build</th><th>query side (incl. one-time query embedding for dense/hybrid)</th></tr>'+
 BUDGETS.flatMap(b=>ARMS.map(a=>{const c=cfg(b,a).cost_s;
   return `<tr><td>${nice[a]} @${b}</td><td>${c.index_build??'n/a'} s</td><td>${c.query_side} s</td></tr>`})).join('');

document.getElementById('foot').textContent=
  `Every number originates in reports/stage1_retrieval.json; regenerate this page with `+
  `python src/make_crossover.py. RRF fuse depth ${D.fuse_depth}, k_rrf=${D.k_rrf}, B=${D.bootstrap_resamples}, ${D.multiplicity}. Queries scored only when their `+
  `entire evidence set lies inside the corpus.`;
</script>
</div></body></html>
"""

OUT.write_text(HTML.replace("__DATA__", json.dumps(data)))
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
