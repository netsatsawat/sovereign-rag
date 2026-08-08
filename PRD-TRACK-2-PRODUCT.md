# sovereign-rag — PRD, Track 2: the product

**Status: not started, and deliberately not started.** This document exists so Track 1 does not
foreclose it, and so the strategic contradiction at the centre of it gets resolved on paper before
anyone writes code. Nothing here is scheduled. The gate that would start it is in §9.

Track 1 — the study, the article and the repo — is specified in [`PRD.md`](PRD.md).

---

## 1. The contradiction, resolved first

**Sovereignty and SaaS point in opposite directions.** Track 1's entire thesis is that nothing
leaves your building. A hosted multi-tenant service is the thing that thesis argues against, and
the first serious prospect who read the article will say so out loud. If this is not resolved
before the first sales conversation, the article becomes the strongest objection to the product it
was supposed to sell.

Three ways out, and only two survive contact with the thesis:

| Model | What it is | Survives? |
|---|---|---|
| **BYOC — customer VPC or on-prem** | the stack deployed into their infrastructure; they hold the data, the keys and the model weights | ✅ the honest one |
| **Hosted control plane, local data plane** | config, fleet management and eval dashboards hosted; inference, documents and embeddings never leave | ✅ with care |
| Multi-tenant hosted | classic SaaS, their documents in your database | ❌ contradicts the article that sold it |

**Decision: BYOC data plane, optional hosted control plane, and the control plane is optional in
the literal sense — an air-gapped install must be fully functional without it.** The moment the
control plane becomes load-bearing, the product is multi-tenant SaaS wearing a different word, and
a procurement reviewer will find that out faster than a competitor will.

This is also what regulated buyers in banking, telecom and government actually procure. The
commercial cost is real and should be named: BYOC has worse gross margins, slower onboarding, and
support burden across customer-controlled environments you cannot log into.

## 2. What the product is

A self-hostable retrieval system a regulated enterprise can deploy inside its own boundary, where
**the architecture is a configuration choice rather than a vendor's opinion** — plain, agentic or
graph retrieval per collection, chosen from measured evidence rather than a sales deck — and where
the system reports what each answer cost and what it was grounded in.

The differentiator is not the retrieval. Retrieval is commodity, and Track 1's own research found
~175k stars of self-hostable RAG already shipping. The differentiator is that **it arrives with its
own measurement harness**: the same eval that produced the published study runs against the
customer's corpus and tells them which architecture their traffic actually justifies. Everyone else
ships a default and an opinion.

## 3. What ports from Track 1, and what does not

Ports unchanged, because §14 of Track 1 makes it so:

- `Arm.answer() -> Envelope` and the three arm implementations
- The OpenAI-compatible adapter (`serve.py`)
- The retrieval-feedback layer, if its gate ever clears
- The evaluation harness, the strata design, and `agent-report-card` as the scorer
- The cost accounting — prefill/decode tokens, LLM calls, wall time, peak RSS

Does **not** port, and must be built from zero:

- Everything in §4. That is the entire product, and it is larger than Track 1.

## 4. The hard problem, named early

**Row-level access control on retrieval.** Not chat-level permissions — retrieval-level. A chunk a
user is not cleared to see must never enter the model's context, and the failure mode is
catastrophic and silent: the model summarises a document the user cannot open, and nobody notices
until an auditor does.

This is the requirement that separates a demo from a product, and it is genuinely difficult:

- Filters must apply **inside** the vector search, not after it, or top-k degrades unpredictably as
  permissions vary — a user with narrow access silently gets worse answers rather than fewer.
- Permissions change; the index does not. Re-indexing on every ACL change is infeasible, so the
  filter has to be evaluated at query time against a live authority.
- GraphRAG makes it worse. A community summary is *derived from* documents with mixed permissions.
  A single entity summary can leak the existence, and often the content, of a document the reader
  cannot access. **There is no clean answer to this** — the honest options are per-principal graphs
  (expensive), permission-uniform collections (restrictive), or not offering graph retrieval on
  mixed-permission corpora.
- Deletion must propagate: to the store, the index, the graph, the caches, the feedback log and any
  learned ranking state. GDPR and PDPA erasure requests do not exempt derived artifacts.

**Anything claiming enterprise readiness without a tested answer here is not a product.** The first
security questionnaire will ask, and "we filter after retrieval" is a failing answer.

## 5. What else is table stakes

Beyond §4, drawn from what a security review in these markets actually asks:

**Identity** — SSO via OIDC/SAML, SCIM provisioning, role-based access, service accounts.
**Ingestion** — connectors, incremental refresh, deletion propagation, OCR, and a dead-letter path
that surfaces what failed to ingest rather than silently dropping it.
**Audit** — immutable logs of who asked what, what was retrieved, what was returned; retention
policy; exportable for the customer's SIEM.
**Provenance as a compliance artifact** — every answer carries its sources with a `used_in_answer`
flag, because "the model cited it" and "the model used it" are different claims and a regulator
will ask which one you mean.
**Operations** — backup and restore including the vector index and graph, a tested upgrade path,
health and capacity signals, and a documented air-gapped install with no runtime registry pulls.
**Model governance** — pinned weights with digests, a recorded model change history, and an eval
that must pass before a model swap is allowed to reach production. This one is a natural extension
of `agent-report-card` and is the piece nobody else ships.

## 6. Non-goals, permanently

- **No multi-tenant hosted inference.** See §1. This is a positioning commitment, not a roadmap
  item deferred.
- **No training on customer data**, ever, under any consent flow. It is the one thing the thesis
  cannot survive, and offering it as an option means answering for it in every review.
- **No proprietary model.** Pinned open weights the customer can inspect and replace.
- **No "self-learning" claim** until Track 1's §6.4 gate clears. Selling a learning loop that has
  not been demonstrated is the failure mode most likely to end the credibility Track 1 built.
- **No agent marketplace, no workflow builder, no plugin ecosystem.** Those are Dify's and
  Flowise's business, and both have full-time teams.

## 7. Pricing shape

Not per-seat, and not per-token. Per-token pricing is incoherent when the customer owns the GPUs,
and per-seat punishes exactly the wide read-only deployment these buyers want.

The defensible shapes are **per-deployment annual licence with support tiers**, or **per-collection
capacity**. Both are legible to a procurement team used to buying software rather than API calls,
and neither requires metering something happening inside a boundary you cannot see.

Open-core is the natural split: the arms, the adapter and the eval harness stay open — they are the
article's credibility and its distribution — and access control, connectors, audit and the control
plane are commercial. That split has to be decided *before* the first commercial line is written,
because relicensing later is where open-core projects lose their communities.

## 8. Risks

**Contract boundary — the biggest one, and not technical.** Selling enterprise AI advisory or
product into Thailand or ASEAN is the restricted lane in `boundaries.md`, and it is the lane
closest to True Corporation's own growth area. This document does not open that lane.

**Ownership is settled.** IP was confirmed clear on 2026-08-08: True claims nothing over work
built on Net's own time, so this codebase is his to license or sell. That removes the gate that
used to sit in front of Track 2 entirely. What remains is not ownership but market: the
non-compete wording is still not on file, so the target market must not be the employer's.
**Global or non-ASEAN first is not a preference here, it is the constraint.**

**The competitive floor is high.** Onyx is MIT, 31.5k stars, and already does enterprise
self-hosted RAG with connectors and permissions. Any commercial answer has to survive "why not
Onyx plus a consultant", and the only honest answer is the measurement harness — which is precisely
why Track 1 has to land first and be good.

**Support across environments you cannot enter** is the operational tax of BYOC, and it is the
reason most BYOC vendors eventually push customers toward hosted. Resisting that is a standing
decision, not a one-time one.

**One person.** Every hour here competes with lanes paying $250–500/hour. This is not a
side-project-sized product, and pretending otherwise is how it consumes two years and ships
nothing.

## 9. The gate

**Do not start Track 2 until all three are true:**

1. **Track 1 is published** and has produced a demand signal that was not solicited — an inbound
   asking to deploy it, not a like.
2. **The contract position is settled for the lane being sold into.**

   **[updated 2026-08-08: the IP half is now confirmed clear.]** True claims nothing over work
   built on Net's own time, so the code in this repo is his to publish, license or sell. That
   removes what was the largest gate on this document — productizing no longer needs anything
   cleared first.

   The **non-compete is a different clause and is not cleared by that.** Its exact wording is
   still not filed, and it is the operative test here because it restricts precisely one lane:
   enterprise AI advisory and adjacent product sold into Thai or ASEAN telecom. So the target
   market condition stands unchanged — **global or non-ASEAN first is a constraint, not a
   preference** — and it is a commercial routing decision now rather than a legal blocker.

   Also unchanged by the IP clearance, and worth keeping separate from it: the **reputational**
   risk of selling into a category True has just invested THB 77bn in. That is not contractual and
   no clause resolves it.
3. **A named first customer** willing to be a design partner in their own environment. Building
   §4 speculatively is how this becomes a two-year unpaid project.

If (1) does not happen, Track 2 does not happen, and Track 1 was still worth it — it is a
credibility asset for the consulting and writing lanes on its own terms.

**The next action for Track 2 is not code.** It is a one-page positioning note that survives the
question *"you wrote that hosted AI is the problem — why are you selling me software?"* If that
note cannot be written convincingly, the product does not exist yet, and no amount of engineering
fixes that.
