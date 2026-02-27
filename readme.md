# Enterprise Account Business Development System

A full-stack multi-agent AI system that automates enterprise sales research. Given a target account, it gathers intelligence in real time, identifies sales opportunities, and generates actionable reports — reducing hours of manual research to minutes.

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

```
React Frontend     TypeScript · ReactFlow · TanStack Query · Tailwind CSS
                   http://localhost:5173
        │  HTTP REST + Server-Sent Events
        ▼
FastAPI API Layer  async REST endpoints · SSE emitter · session management
                   http://localhost:8000
        │  Python function calls
        ▼
LangGraph Agents   Coordinator · Gatherer · Identifier · Validator
                   SQLite checkpoints · ChromaDB · DuckDuckGo MCP
```

The frontend streams real-time agent progress via Server-Sent Events. Clicking any completed node opens an observability panel showing run time, extracted data, and a LangSmith trace deep-link.

### Agent Workflow

```
CoordinatorAgent ──► GathererAgent ──► IdentifierAgent ──► ValidatorAgent
        ↑                                                           │
        └────────────── Human-in-the-Loop Feedback ◄───────────────┘
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
| **Semantic Product Matching** | ChromaDB + sentence-transformers for embedding-based opportunity identification |
| **Intelligent LLM Routing** | 3-tier routing (local Ollama → Groq API) based on task complexity |
| **Session Persistence** | Pause, stop, and resume research sessions across browser reloads via SQLite |
| **Production-Ready** | 454 tests passing, structured logging (structlog), Pydantic v2 validation |

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
| **Frontend** | React 18, TypeScript, ReactFlow, TanStack Query, Tailwind CSS |
| **LLMs** | Ollama (local), Groq API, LiteLLM routing |
| **Vector Search** | ChromaDB, sentence-transformers |
| **Data Sources** | DuckDuckGo MCP, BeautifulSoup, httpx |
| **Observability** | LangSmith tracing, in-app SSE callback handler, per-node trace store |
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

## License

MIT License

---

**Built by:** Mahaveer Satra
**Contact:** [LinkedIn](https://www.linkedin.com/in/mahaveer-satra/)
