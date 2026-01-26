# Enterprise Account Research System - Codebase Architecture

**Last Updated**: 2026-01-26
**Status**: Phase 3 FULLY COMPLETE - Ready for Phase 4
**Test Status**: 199 tests passing

---

## Quick Context Recovery

**READ THIS FIRST** when restoring context after clearing chat:

1. **Project**: Multi-agent system for enterprise account research using LangGraph
2. **Current Phase**: Phase 3 COMPLETE ✅ - All goals achieved
3. **What's Done**: All 4 agents + LangGraph workflow + human-in-loop + 199 tests
4. **What's Next**: Phase 4 - Testing & Polish (integration tests, E2E tests, CLI)

---

## Phase 3 Goals - ALL COMPLETE ✅

| Goal | Status | Evidence |
|------|--------|----------|
| 1. Implement all 4 agents | ✅ COMPLETE | CoordinatorAgent, GathererAgent, IdentifierAgent, ValidatorAgent |
| 2. Wire into LangGraph workflow | ✅ COMPLETE | `graph/workflow.py` with conditional routing |
| 3. Human-in-the-loop implementation | ✅ COMPLETE | `_wait_for_human` interrupt nodes, feedback loops |
| 4. End-to-end flow working | ✅ COMPLETE | Full pipeline: Entry → Gather → Identify → Validate → Exit → Feedback |

---

## Phase 4 Goals (NEXT)

**Goals:**
1. Integration tests (multi-agent pipeline tests)
2. E2E tests (full workflow with mocked external services)
3. CLI interface for running research
4. Documentation and examples

### Immediate Next Action
**Write Integration Tests** (`tests/test_integration/`)
- Test Coordinator → Gatherer → Identifier → Validator pipeline
- Test feedback loop scenarios (human says "gather more data")
- Test error recovery across agents
- Test state persistence with SQLite checkpointing

---

## Executive Summary

**Total Codebase**: ~5,500+ lines of production code + ~1,400 lines of tests

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ COMPLETE | Core infrastructure (config, router, logging, exceptions) |
| Phase 2 | ✅ COMPLETE | Data layer (MCP client, scrapers, product catalog, workflow) |
| Phase 3 | ✅ COMPLETE | Agent implementations (4/4) + human-in-loop + workflow integration |
| Phase 4 | ⏳ NEXT | Testing & Polish (integration, E2E, CLI) |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Graph/Workflow Layer (graph/workflow.py) - FULLY INTEGRATED         │
│  - LangGraph orchestration with feedback loops                       │
│  - SQLite checkpointing for resume capability                        │
│  - Conditional routing based on human feedback                       │
│  - Human-in-loop interrupts at _wait_for_human node                  │
├──────────────────────────────────────────────────────────────────────┤
│  Agent Layer (agents/) - PHASE 3 COMPLETE                            │
│  ✅ CoordinatorAgent (entry/exit, human-in-loop, routing) - COMPLETE │
│  ✅ GathererAgent (collect & analyze data from sources) - COMPLETE   │
│  ✅ IdentifierAgent (find opportunities) - COMPLETE                  │
│  ✅ ValidatorAgent (confidence scoring, risk assessment) - COMPLETE  │
├──────────────────────────────────────────────────────────────────────┤
│  Core Services (core/)                                               │
│  - ModelRouter: 3-tier LLM routing with caching                      │
│  - BaseAgent: Abstract base with monitoring                          │
│  - Exceptions: Custom hierarchy                                      │
├──────────────────────────────────────────────────────────────────────┤
│  Data Source Layer (data_sources/)                                   │
│  - DuckDuckGoMCPClient: Web search via MCP                           │
│  - JobBoardScraper: Career page detection & parsing                  │
│  - ProductCatalogIndexer: ChromaDB semantic search                   │
│  - ProductMatcher: Semantic product matching (used by Identifier)    │
├──────────────────────────────────────────────────────────────────────┤
│  Models/Domain Layer (models/)                                       │
│  - ResearchState: LangGraph state TypedDict (with Coordinator fields)│
│  - Domain models: JobPosting, Opportunity, Signal, etc.              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Workflow Architecture (FULLY IMPLEMENTED)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RESEARCH WORKFLOW (workflow.py)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐                                                │
│  │  coordinator_entry  │  Validate inputs, normalize name, questions    │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼ (conditional: needs_human?)                               │
│  ┌─────────────────────┐                                                │
│  │  _wait_for_human    │  INTERRUPT: Human clarification                │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │     gatherer        │  Web search, job postings, news, LLM analysis  │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │    identifier       │  Extract requirements, match products          │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │     validator       │  Assess risks, score confidence, filter        │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  coordinator_exit   │  Format report, present to human               │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  _wait_for_human    │  INTERRUPT: Human feedback on report           │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │coordinator_feedback │  Parse feedback, determine routing             │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│    ┌────────┼────────┬────────┬────────┐                                │
│    ▼        ▼        ▼        ▼        ▼                                │
│ gatherer identifier validator  END   (feedback loops)                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Human-in-the-Loop Flow

1. **Entry Interrupt**: Coordinator can ask clarifying questions before research
2. **Exit Interrupt**: Human reviews report and provides feedback
3. **Feedback Routing**: Human can request:
   - "gather more data" → loops back to GathererAgent
   - "find different opportunities" → loops back to IdentifierAgent
   - "re-evaluate scores" → loops back to ValidatorAgent
   - "looks good" → workflow completes

### Workflow Routing Logic

| From Node | Condition | Routes To |
|-----------|-----------|-----------|
| `coordinator_entry` | `waiting_for_human=True` | `_wait_for_human` |
| `coordinator_entry` | `waiting_for_human=False` | `gatherer` |
| `_wait_for_human` | `current_report` exists | `coordinator_feedback` |
| `_wait_for_human` | No `current_report` | `gatherer` |
| `coordinator_feedback` | `next_route="gatherer"` | `gatherer` |
| `coordinator_feedback` | `next_route="identifier"` | `identifier` |
| `coordinator_feedback` | `next_route="validator"` | `validator` |
| `coordinator_feedback` | `next_route="complete"` | `END` |

---

## Complete File Inventory

### Phase 1: Core Infrastructure (COMPLETE)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `config.py` | ~150 | Pydantic settings, env vars, model routing config | ✅ |
| `core/exceptions.py` | ~120 | Custom exception hierarchy | ✅ |
| `core/model_router.py` | ~350 | 3-tier LLM routing, caching, retries | ✅ |
| `core/base_agent.py` | ~220 | Abstract BaseAgent, StatelessAgent | ✅ |
| `utils/logging.py` | ~80 | Structured logging (structlog) | ✅ |

**Total Phase 1**: ~920 lines

---

### Phase 2: Data Layer (COMPLETE)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `models/state.py` | ~180 | ResearchState TypedDict, Signal, Opportunity | ✅ |
| `models/domain.py` | ~180 | JobPosting, CompanyInfo, Product, AgentResult | ✅ |
| `data_sources/base.py` | ~150 | Abstract DataSource, CachedDataSource | ✅ |
| `data_sources/mcp_ddg_client.py` | ~400 | DuckDuckGo MCP client with rate limiting | ✅ |
| `data_sources/scraper.py` | ~250 | Web scraping utilities (BeautifulSoup, httpx) | ✅ |
| `data_sources/job_boards.py` | ~300 | Job board scraping, career page detection | ✅ |
| `data_sources/product_catalog.py` | ~350 | ChromaDB indexing, semantic product matching | ✅ |
| `graph/workflow.py` | ~520 | LangGraph workflow with feedback loops | ✅ |

**Total Phase 2**: ~2,330 lines

---

### Phase 3: Agents (COMPLETE - 4/4)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `agents/__init__.py` | ~22 | Package exports (all agents + WorkflowRoute) | ✅ |
| `agents/coordinator.py` | ~580 | Dual entry/exit, human-in-loop, feedback routing | ✅ |
| `agents/gatherer.py` | ~530 | Collect & analyze data with LLM from MCP, jobs, news | ✅ |
| `agents/identifier.py` | ~350 | LLM-based opportunity identification with ProductMatcher | ✅ |
| `agents/validator.py` | ~300 | Confidence scoring, risk assessment, filtering | ✅ |

**Total Phase 3**: ~1,780 lines

---

### Tests (199 passing)

| File | Tests | Purpose | Status |
|------|-------|---------|--------|
| `tests/test_agents/test_coordinator.py` | 31 | CoordinatorAgent full coverage | ✅ |
| `tests/test_agents/test_gatherer.py` | 16 | GathererAgent full coverage | ✅ |
| `tests/test_agents/test_identifier.py` | 31 | IdentifierAgent full coverage | ✅ |
| `tests/test_agents/test_validator.py` | 35 | ValidatorAgent full coverage | ✅ |
| Other test files | 86 | Core, data sources, model router | ✅ |

**Total Tests**: 199 passing

---

## Agent Summary

### CoordinatorAgent ✅
- **File**: `src/agents/coordinator.py` (~580 lines)
- **Tests**: 31 passing
- **Role**: Supervisor agent with 3 entry points
- **Entry Points**: `process_entry()`, `process_exit()`, `process_feedback()`
- **Human-in-Loop**: Handles interrupts, formats reports, routes feedback

### GathererAgent ✅
- **File**: `src/agents/gatherer.py` (~530 lines)
- **Tests**: 16 passing
- **Role**: Intelligence collection with LLM analysis
- **Sources**: DuckDuckGo MCP (web + news), JobBoardScraper
- **Outputs**: signals, job_postings, news_items, tech_stack

### IdentifierAgent ✅
- **File**: `src/agents/identifier.py` (~350 lines)
- **Tests**: 31 passing
- **Role**: Opportunity identification from gathered data
- **Uses**: ProductMatcher (semantic search), ModelRouter (LLM reasoning)
- **Outputs**: opportunities (list of Opportunity objects with evidence)

### ValidatorAgent ✅
- **File**: `src/agents/validator.py` (~300 lines)
- **Tests**: 35 passing
- **Role**: Risk assessment and confidence scoring
- **Features**: 5 risk categories, confidence re-scoring, 0.6 threshold filtering
- **Outputs**: validated_opportunities, competitive_risks

---

## Model Router Configuration

| Complexity | Model | Provider | Use Case |
|------------|-------|----------|----------|
| 1-3 | llama3.2:3b | LOCAL Ollama | CoordinatorAgent, GathererAgent |
| 4-7 | llama-3.1-8b-instant | Groq | IdentifierAgent, ValidatorAgent |
| 8-10 | llama-3.1-70b | Groq | Complex reasoning (if needed) |

---

## Development Checklist

### ✅ Phase 3 COMPLETE (2026-01-26)

**Goal 1: Implement all 4 agents**
- [x] CoordinatorAgent implementation (~580 lines)
- [x] CoordinatorAgent tests (31 tests)
- [x] GathererAgent implementation (~530 lines)
- [x] GathererAgent tests (16 tests)
- [x] IdentifierAgent implementation (~350 lines)
- [x] IdentifierAgent tests (31 tests)
- [x] ValidatorAgent implementation (~300 lines)
- [x] ValidatorAgent tests (35 tests)

**Goal 2: Wire into LangGraph workflow**
- [x] workflow.py with all 4 agents as nodes
- [x] Conditional routing between agents
- [x] State persistence with SQLite checkpointing

**Goal 3: Human-in-the-loop implementation**
- [x] `_wait_for_human` interrupt nodes
- [x] CoordinatorAgent handles entry/exit interrupts
- [x] Feedback parsing and routing

**Goal 4: End-to-end flow working**
- [x] Full pipeline: Entry → Gather → Identify → Validate → Exit
- [x] Feedback loops back to any agent
- [x] All 199 tests passing

### ⏳ Phase 4: Testing & Polish (NEXT)
- [ ] Integration tests (multi-agent pipeline)
- [ ] E2E tests (full workflow with mocks)
- [ ] CLI interface
- [ ] Documentation and examples

---

## Commands Reference

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run all tests
python -m pytest tests/ -v

# Run specific agent tests
python -m pytest tests/test_agents/test_coordinator.py -v
python -m pytest tests/test_agents/test_gatherer.py -v
python -m pytest tests/test_agents/test_identifier.py -v
python -m pytest tests/test_agents/test_validator.py -v

# Quick test summary
python -m pytest tests/ -v --tb=short 2>&1 | tail -5

# Check imports work
python -c "from src.agents import CoordinatorAgent, GathererAgent, IdentifierAgent, ValidatorAgent, WorkflowRoute; print('OK')"
```

---

## Key Files for Context Recovery

When restoring context, read these files in order:

1. **This file** (`CODEBASE_ARCHITECTURE.md`) - Architecture + status + next steps
2. `src/models/state.py` - State structure (ResearchState, Opportunity, Signal)
3. `src/graph/workflow.py` - LangGraph workflow definition
4. `src/agents/coordinator.py` - Human-in-loop patterns
5. `src/agents/validator.py` - Most recently implemented agent

---

## Key State Structure (`models/state.py`)

```python
class ResearchState(TypedDict):
    """Main workflow state - ALL fields are implemented"""

    # Input (from user)
    account_name: str              # Company to research
    industry: str                  # Industry vertical
    research_depth: ResearchDepth  # QUICK/STANDARD/DEEP

    # Collected data (GathererAgent)
    signals: list[Signal]          # Web search results with LLM analysis
    job_postings: list[dict]       # Scraped job postings
    news_items: list[dict]         # News articles
    tech_stack: list[str]          # Extracted technologies

    # Analysis (IdentifierAgent)
    opportunities: list[Opportunity]  # Identified opportunities

    # Validation (ValidatorAgent)
    validated_opportunities: list[Opportunity]  # Filtered (>0.6 confidence)
    competitive_risks: list[str]   # Identified risks

    # Human interaction (CoordinatorAgent)
    human_feedback: list[str]      # Conversation history
    waiting_for_human: bool        # Pause for human input
    human_question: str | None     # Question/report for human
    current_report: str | None     # Formatted report
    feedback_context: str | None   # Parsed guidance for retry
    next_route: str | None         # Routing decision

    # Progress tracking
    progress: ResearchProgress     # Tracks which agents completed
```

---

**END OF ARCHITECTURE DOCUMENT**

*Phase 3 is COMPLETE. All 4 agents implemented, workflow integrated, human-in-loop working, 199 tests passing.*
*Ready for Phase 4: Testing & Polish*
