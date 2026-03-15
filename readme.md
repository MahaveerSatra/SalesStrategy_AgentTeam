# Enterprise Account Business Development System

> This tool helps a Head of Sales at a Tier-1 Tech firm reduce account research time from 4 hours to 4 minutes.

A full-stack multi-agent AI system that automates enterprise sales research. Given a target account, it gathers intelligence in real time, identifies sales opportunities, and generates actionable reports.

**Stack:** LangGraph · FastAPI · React/TypeScript · ChromaDB · LangSmith

---

## The Problem

Enterprise sales teams spend 5-10 hours per account researching prospects: scanning job postings for technology signals, reading news for expansion plans, and manually matching customer needs to products. This work is repetitive, time-consuming, and inconsistent.

## The Solution

A multi-agent system that:
1. **Gathers intelligence** from web searches, job postings, and news
2. **Extracts requirements** using LLM-powered analysis
3. **Matches products** via semantic search against your product catalog
4. **Validates opportunities** with confidence scoring and risk assessment
5. **Generates reports** with actionable sales recommendations

Delivered through a real-time web interface that streams live agent progress.

**Result:** Validated opportunities identified from signals for the target customer in a single automated run.

---

## Architecture

The frontend streams real-time agent progress via Server-Sent Events. Clicking any completed node opens an observability panel showing run time, extracted data, and a LangSmith trace deep-link.

### Agent Workflow

```mermaid
flowchart LR
    You(["You"])
    CE["Coordinator\nEntry"]
    G["Gatherer\nweb · jobs · news"]
    I["Identifier\nChromaDB match"]
    V["Validator\nconfidence score"]
    CX["Coordinator\nExit"]
    R["Report"]

    You -->|"account + industry"| CE
    CE --> G
    CE --> I
    CE --> V
    G --> I
    I --> V
    G --> CX
    I --> CX
    V --> CX
    CX --> R
    R -->|"Human Review · interrupt()"| You
    You -.->|"feedback"| CE
```

The workflow pauses after report generation using LangGraph's `interrupt()` primitive. The user reviews the report in the browser and submits feedback (approve / dig deeper / different products / custom). The workflow resumes from the SQLite checkpoint with the feedback injected.

### Seller-Agnostic Design

Works with any company's product catalog. Index once, research unlimited accounts:

```bash
# Index your product catalog (JSON, URL, or markdown)
python -m src.cli setup-catalog --seller "YourCompany" --catalog-file products.json

# Research a target account
python -m src.cli research "Boeing" --industry aerospace --seller "YourCompany"
```

---

## Key Features

| Feature | Implementation |
|---------|----------------|
| **Multi-Agent Orchestration** | LangGraph with conditional routing and checkpointed feedback loops |
| **Real-Time Streaming** | Server-Sent Events stream node state (started / completed / waiting) to the frontend live |
| **Human-in-the-Loop** | LangGraph `interrupt()` pauses the workflow; resumes from SQLite checkpoint on feedback |
| **Agent Observability** | Click any node to inspect run time, extracted data, and LangSmith trace link |
| **Hybrid RAG Product Matching** | 3-stage pipeline: BM25 keyword + ChromaDB vector → Reciprocal Rank Fusion → cross-encoder re-ranking before LLM sees candidates. Corpus enriched with ~76 scraped solution pages for broader recall. |
| **Intelligent LLM Routing** | 3-tier routing (local Ollama → Groq API) based on task complexity |
| **Session Persistence** | Pause, stop, and resume research sessions across browser reloads via SQLite |
| **Production-Ready** | 454 tests passing, structured logging (structlog), Pydantic v2 validation |

---

## Production-Grade Engineering

### State Persistence — No Lost Work

For an enterprise customer like Disney, a system that crashes and loses progress is unusable. Every node completion is checkpointed to SQLite. If the agent fails mid-run, or the user closes the browser, the workflow resumes from exactly where it paused:

```python
# src/graph/workflow.py
conn = sqlite3.connect("data/checkpoints/checkpoints.db", check_same_thread=False)
self.app = self.graph.compile(
    checkpointer=SqliteSaver(conn),
    interrupt_before=["_wait_for_human"]  # pause for human review
)

# Resume from checkpoint — human can edit state before continuing
self.app.update_state(config, {"human_feedback": [user_input]})
result = self.app.invoke(None, config)  # None = resume from checkpoint
```

### Error Handling — Graceful Degradation

What happens when a search returns 0 results, or the LLM API rate limits? The workflow continues. Every external call uses `tenacity` retry with exponential backoff, and every data source falls back to a simpler strategy before returning an empty result:

```python
# src/data_sources/mcp_ddg_client.py
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10),
       retry=retry_if_exception_type(DataSourceTimeoutError))
async def search(self, query: str) -> list[SearchResult]: ...

# If "Boeing AI hiring news" returns nothing, try simpler queries before giving up
for query_variant in [f"{query} news", f"{query} announcement", query]:
    results = await self.search(query_variant)
    if results:
        return results
return []  # Workflow continues with partial data rather than crashing
```

### Response Caching — Cost-Conscious by Design

**Application-level caching** (all providers): Identical LLM prompts return cached results — ~0ms, $0 cost.
Search results are cached separately with a shorter TTL:

```python
# src/core/model_router.py — 24-hour TTL for LLM responses
cached = self.cache.get(model, prompt, temperature=temperature)
if cached:
    return cached  # Cache hit: instant response, zero API cost
self.cache.set(model, prompt, response)

# src/data_sources/mcp_ddg_client.py — 1-hour TTL for search results
cached = self.cache.get("search", query=query, max_results=max_results)
if cached is not None:
    return cached
```

**Anthropic prompt caching** (`cache_control`): Sales agents reuse the same system prompts
(product catalog, analysis playbook) across dozens of calls per session. Marking them with
`cache_control` caches them on Anthropic's servers — ~90% cost reduction on those tokens:

```python
# src/core/model_router.py — _call_anthropic_model()
# The system prompt (product catalog + instructions) is identical across many calls.
# Only the user prompt (account-specific query) changes each time.
messages = [
    {
        "role": "system",
        "content": [{
            "type": "text",
            "text": system_prompt,            # product catalog + analysis instructions
            "cache_control": {"type": "ephemeral"}   # cached on Anthropic's servers
        }]
    },
    {"role": "user", "content": prompt}       # account-specific query (not cached)
]
response = await litellm.acompletion(model="anthropic/claude-haiku-4-5-20251001", messages=messages)
```

### Advanced RAG Pipeline — Precision Before the LLM

Passing all ChromaDB results directly to the LLM introduces noise. Products that are semantically close but irrelevant dilute reasoning. A 3-stage pipeline filters candidates before the LLM sees them:

```
Requirements → BM25 + Vector → RRF wide pool → Cross-Encoder → top-k → LLM
```

**Stage 1 — Hybrid Retrieval**: BM25 preserves exact-match precision (product names, acronyms); ChromaDB vector search handles paraphrases and domain language. Reciprocal Rank Fusion (k=60) merges both ranked lists:

```python
# src/data_sources/product_catalog.py
rrf_score = sum(1.0 / (60 + rank) for rank in [bm25_rank, vector_rank])
```

The retrieval corpus is enriched with ~76 MathWorks solution pages (scraped via Tavily Extract API, which bypasses Cloudflare WAF). A query like "EV battery management" now matches the Electric Vehicle solution page, which shares BM25 term overlap with Simscape Battery — boosting recall for industry-specific language.

**Stage 2 — Cross-Encoder Re-ranking**: A cross-encoder scores every `(requirement, product_doc)` pair jointly — unlike bi-encoders, it attends to both sides simultaneously, capturing token-level relevance signals:

```python
# Lazy-loaded once per process, ~20-40ms for 20 pairs on CPU
ce_scores = self._get_cross_encoder().predict([[req, doc] for req in requirements for doc in candidates])
```

**Hardware-aware model selection**: Running on a 4GB laptop with Ollama already consuming ~2GB, the RAM budget for the re-ranker is ~220MB. `mxbai-rerank-xsmall-v1` (56M params, ~220MB) fits safely; BGE-v2-m3 (1.4GB) would OOM. The choice is a single config string — upgrade is trivial if hardware improves.

**Result**: The LLM receives a short, high-precision candidate list — reducing hallucinations and tightening the context window used for opportunity generation.

---

## Quick Start

**Backend:**
```bash
git clone https://github.com/MahaveerSatra/SalesStrategy_AgentTeam.git
cd SalesStrategy_AgentTeam
python -m venv venv && source venv/bin/activate  # .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env  # Add GROQ_API_KEY (free at console.groq.com)

uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install && npm run dev
# → http://localhost:5173
```

**CLI (no UI required):**
```bash
python -m src.cli research "Remora Carbon" --industry "carbon capture" --seller "MathWorks"
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Orchestration** | LangGraph, SQLite checkpointing, `interrupt()` for human-in-the-loop |
| **API Layer** | FastAPI, Server-Sent Events, asyncio |
| **Frontend** | React 18, TypeScript, ReactFlow, TanStack Query, Tailwind CSS, recharts |
| **LLMs** | Ollama (local), Groq API, LiteLLM routing |
| **Hybrid RAG** | ChromaDB (vector), rank-bm25 (BM25), sentence-transformers CrossEncoder (re-ranking) |
| **Data Sources** | DuckDuckGo MCP, Tavily (search + Extract API), BeautifulSoup, httpx |
| **Observability** | LangSmith tracing, in-app SSE callback handler, per-node trace store |
| **Resilience** | tenacity retry, TTL caching (24h LLM / 1h search), graceful degradation |
| **Validation** | Pydantic v2, structured LLM outputs |
| **Testing** | pytest, 454 tests, 100% pass rate |

---

## Project Structure

```
src/                              # Backend agents
├── agents/                       # Coordinator, Gatherer, Identifier, Validator
├── graph/                        # LangGraph workflow + SSECallbackHandler
├── data_sources/                 # MCP client, job scrapers, product catalog
└── models/                       # Pydantic schemas, ResearchState

api/                              # FastAPI layer
├── routers/research.py           # REST + SSE endpoints
├── services/workflow_service.py  # Agent lifecycle (start/pause/resume/discard)
└── sse/event_stream.py           # Per-thread asyncio queues for SSE

frontend/src/                     # React SPA
├── components/                   # WorkflowGraph, NodeTracePanel, ReportView, ResearchForm
├── hooks/                        # useSSEStream (EventSource), useResearchWorkflow (TanStack Query)
└── lib/api.ts                    # Typed fetch wrappers for all API endpoints
```

**454 tests passing**

---

## Evals

Custom eval framework for measuring and improving agent output quality across prompt iterations.

**Design**: Model-graded evaluation using Claude as judge (CoT reasoning → structured JSON) plus 9 deterministic rule-based checks. Results are tracked in `history.csv` to show score deltas after prompt changes.

**4 Metrics (1–5 scale):**
- **Accuracy** — Are talking points grounded in evidence with citations ([SIG-001], [JOB-002])?
- **Actionability** — Does the report give a sales rep concrete next steps and specific personas?
- **Alignment** — Do recommended products genuinely fit the account's industry signals?
- **Safety & Ethics** — Is the output consultative and honest, with no manipulative pressure tactics?

### Running Evals

```bash
# Fast smoke test — synthetic state, no live API or Ollama required
python -m evals.run_evals --case TC-01 --mock

# Live run (requires Ollama running) — Remora Carbon is most reliable
python -m evals.run_evals --case TC-04

# Manual judge step: paste evals/results/pending_judge_TC-04.txt into Claude Pro
# Save the JSON response to: evals/results/judge_response_TC-04.json

# Record scores
python -m evals.run_evals --ingest TC-04

# Track improvement across runs
python -m evals.run_evals --compare
```

### Score History

**TC-02: NASA / MathWorks** (expansion intent — recommend new toolboxes beyond existing MATLAB/Simulink install)

| Run | Phase | Accuracy | Actionability | Alignment | Safety | Overall | Det | Change |
|-----|-------|----------|---------------|-----------|--------|---------|-----|--------|
| Run 1 | Baseline | 4/5 | 2/5 | 1/5 | 5/5 | 3.0 | 8/13 | — |
| Run 2 | Phase 1 | 4/5 | 3/5 | 2/5 | 5/5 | **3.5** | 9/13 | +0.5 overall (+1 align) |
| Run 3 | Phase 2 | — | — | — | — | — | — | Pending eval |

**Per-agent detail (TC-02):**

| Agent | Run 1 Avg | Run 2 Avg | Delta | Phase 2 changes |
|-------|-----------|-----------|-------|-----------------|
| Gatherer | 3.0 | 3.5 | +0.5 | Free-form queries; source-authority scoring; Groq 8B |
| Identifier | 2.75 | 3.0 | +0.25 | 5-7 opportunities; abbreviation expansion; top_k=15 |
| Validator | 2.25 | 3.25 | +1.0 | Intent parsing; risk mitigations; coverage check |
| Coordinator | 2.75 | 3.5 | +0.75 | Inherited from upstream improvements |

**Phase 1 root cause fixed:** Validator scored existing products (MATLAB) as "aligned" for an expansion objective — INTENT PARSING block now applies mandatory -0.25 penalty on confirmed existing products. All risks now require paired mitigation strategies.

**Phase 2 root cause fixed:** Product taxonomy gap — catalog descriptions were one-sentence stubs ("Multi-sensor fusion") preventing BM25 from matching "GNC toolboxes" → "Sensor Fusion and Tracking Toolbox". Fixed by scraping 133 product pages (347 total docs), weighting vector search 1.5x over BM25, and having the LLM expand domain abbreviations before retrieval. Gatherer upgraded to generate objective-driven queries freely from `user_context` instead of 5 hardcoded categories.

---

## License

MIT License

---

**Built by:** Mahaveer Satra
**Contact:** [LinkedIn](https://www.linkedin.com/in/mahaveer-satra/)
