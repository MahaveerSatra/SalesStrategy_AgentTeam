# Enterprise Account Business Development System

A production-ready multi-agent AI system that automates enterprise sales research. Given a target account, it gathers intelligence, identifies sales opportunities, and generates actionable recommendations—reducing hours of manual research to minutes.

**Built with:** LangGraph | ChromaDB | Ollama | Python

---

## The Problem

Enterprise sales teams spend 5-10 hours per account researching prospects: scanning job postings for technology signals, reading news for initiatives, and manually matching customer needs to products. This research is repetitive, time-consuming, and inconsistent.

## The Solution

A multi-agent system that:
1. **Gathers intelligence** from web searches, job postings, and news
2. **Extracts requirements** using LLM-powered analysis
3. **Matches products** via semantic search against your product catalog
4. **Validates opportunities** with confidence scoring and risk assessment
5. **Generates reports** with actionable sales recommendations

**Result:** 3 validated opportunities identified from 10 signals for a major aerospace account in a single automated run.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐                                                │
│  │  CoordinatorAgent   │  Entry point, clarifying questions,            │
│  │  (Entry/Exit)       │  feedback routing, report generation           │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │   GathererAgent     │  Web search (DuckDuckGo MCP), job boards,      │
│  │                     │  news aggregation, LLM-powered analysis        │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  IdentifierAgent    │  Requirement extraction, semantic product      │
│  │                     │  matching via ChromaDB embeddings              │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  ValidatorAgent     │  Confidence scoring, competitive risk          │
│  │                     │  assessment, opportunity filtering             │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  Human-in-the-Loop  │  Review report, request refinements,           │
│  │  (Feedback)         │  loop back to any agent                        │
│  └─────────────────────┘                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Seller-Agnostic Design

The system works with **any company's product catalog**. Index your products once, then research unlimited target accounts:

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
| **Multi-Agent Orchestration** | LangGraph with conditional routing and feedback loops |
| **Semantic Product Matching** | ChromaDB + sentence-transformers embeddings |
| **Intelligent Model Routing** | 3-tier LLM routing (local Ollama → Groq API) based on task complexity |
| **Human-in-the-Loop** | Interactive CLI with clarifying questions and report refinement |
| **Checkpointing** | SQLite persistence—pause and resume research anytime |
| **Production-Ready** | 432 tests passing, structured logging, comprehensive error handling |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/MahaveerSatra/SalesStrategy_AgentTeam.git
cd SalesStrategy_AgentTeam
python -m venv venv && source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)

# 3. Index a product catalog
python -m src.cli setup-catalog --seller "YourCompany" --catalog-file products.json

# 4. Run research
python -m src.cli research "TargetCompany" --industry "industry" --output ./reports
```

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Orchestration** | LangGraph, SQLite checkpointing |
| **LLMs** | Ollama (local), Groq API (cloud), LiteLLM |
| **Vector Search** | ChromaDB, sentence-transformers |
| **Data Sources** | DuckDuckGo MCP, BeautifulSoup, httpx |
| **Validation** | Pydantic v2, structured LLM outputs |
| **Testing** | pytest (432 tests), 100% pass rate |

---

## Project Structure

```
src/
├── agents/           # 4 specialized agents (~1,800 lines)
│   ├── coordinator.py    # Entry/exit, human-in-loop, feedback routing
│   ├── gatherer.py       # Intelligence collection with LLM analysis
│   ├── identifier.py     # Opportunity identification, product matching
│   └── validator.py      # Risk assessment, confidence scoring
├── graph/            # LangGraph workflow with conditional routing
├── data_sources/     # MCP client, job scrapers, product catalog
├── core/             # Model router, base agent, exceptions
├── cli/              # Production CLI interface
└── models/           # Pydantic schemas, state management
```

**Total:** ~6,700 lines of production code | 432 tests

---

## License

MIT License

---

**Built by:** Mahaveer Satra

**Contact:** [LinkedIn](https://www.linkedin.com/in/mahaveer-satra/)
