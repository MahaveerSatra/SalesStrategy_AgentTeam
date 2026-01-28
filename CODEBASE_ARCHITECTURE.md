# Enterprise Account Research System - Codebase Architecture

**Last Updated**: 2026-01-28
**Status**: Phase 4 IN PROGRESS - Realistic Fixtures Complete, E2E Tests Next
**Test Status**: 290 tests passing

---

## Quick Context Recovery

**READ THIS FIRST** when restoring context after clearing chat:

1. **Project**: Multi-agent system for enterprise account research using LangGraph
2. **Current Phase**: Phase 4 IN PROGRESS - Realistic fixtures complete, E2E tests next
3. **What's Done**: All 4 agents + LangGraph workflow + human-in-loop + 290 tests + Realistic fixtures
4. **What's Next**: E2E tests with real Ollama LLM, then CLI interface
5. **CRITICAL**: Fixtures reveal agents need robust JSON parsing for varied LLM output formats

### Current Session Context (2026-01-28)

**Just Completed**:
- Created realistic test fixtures in `tests/fixtures/`
- Added `FixtureLoader` utility for easy fixture access
- LLM response fixtures with 6 variants (clean, markdown, extra_text, whitespace, partial, etc.)
- Search result fixtures (Acme Corp, NovaTech) with raw MCP response format
- Job posting fixtures (Greenhouse, Lever, Generic formats)
- 28 new fixture-based tests (290 total tests passing)
- JSON parsing robustness tests using `extract_json_from_llm_response` helper

**Identified Enhancement Opportunity**:
- Agent code uses raw `json.loads()` - not robust to varied LLM output
- The `extract_json_from_llm_response` helper in `tests/fixtures/loader.py` handles varied formats
- TODO: Integrate robust JSON parsing into agent code for production resilience

**Immediate Next Action**:
1. Add E2E tests with real Ollama LLM (mark as `@pytest.mark.slow`)
2. Consider integrating `extract_json_from_llm_response` into agent code
3. Build CLI interface for running research

---

## Phase 3 Goals - ALL COMPLETE ✅

| Goal | Status | Evidence |
|------|--------|----------|
| 1. Implement all 4 agents | ✅ COMPLETE | CoordinatorAgent, GathererAgent, IdentifierAgent, ValidatorAgent |
| 2. Wire into LangGraph workflow | ✅ COMPLETE | `graph/workflow.py` with conditional routing |
| 3. Human-in-the-loop implementation | ✅ COMPLETE | `_wait_for_human` interrupt nodes, feedback loops |
| 4. End-to-end flow working | ✅ COMPLETE | Full pipeline: Entry → Gather → Identify → Validate → Exit → Feedback |

---

## Phase 4 Goals (IN PROGRESS)

**Goals:**
1. ✅ Integration tests (multi-agent pipeline tests) - DONE
2. ✅ Realistic fixtures for testing - DONE
3. ⏳ E2E tests (full workflow with real Ollama LLM) - **CURRENT TASK**
4. ⏳ CLI interface for running research
5. ⏳ Documentation and examples

### Integration Tests Created (2026-01-28)

| File | Tests | Purpose | Quality |
|------|-------|---------|---------|
| `test_integration/test_pipeline.py` | 13 | Agent pipeline flow | ⚠️ Mocked |
| `test_integration/test_feedback_loops.py` | 16 | Human feedback routing | ⚠️ Mocked |
| `test_integration/test_error_recovery.py` | 17 | Error handling paths | ⚠️ Mocked |
| `test_integration/test_checkpointing.py` | 17 | SQLite checkpointing | ✅ Real LangGraph |
| `test_integration/test_realistic_fixtures.py` | 28 | Realistic data fixtures | ✅ Realistic Data |

**Total**: 91 integration tests (290 total tests passing)

### Realistic Test Fixtures (COMPLETE)

**Directory Structure:**
```
tests/fixtures/
├── __init__.py           # Package init, exports FixtureLoader
├── loader.py             # FixtureLoader utility + extract_json_from_llm_response helper
├── llm_responses/        # LLM response fixtures with format variants
│   ├── gatherer_analysis.json
│   ├── identifier_requirements.json
│   ├── identifier_opportunities.json
│   ├── validator_risks.json
│   └── validator_scoring.json
├── search_results/       # DuckDuckGo search result fixtures
│   ├── acme_corp.json
│   └── tech_startup.json
└── job_postings/         # Job board fixtures (HTML + parsed)
    ├── greenhouse.json
    ├── lever.json
    └── generic.json
```

**LLM Response Variants (for robustness testing):**
- `clean` - Perfect JSON as expected
- `markdown` - JSON wrapped in markdown code fences
- `extra_text` - JSON with explanatory text before/after
- `whitespace` - JSON with extra whitespace/newlines
- `partial` - Malformed JSON (for error handling tests)
- Additional variants per fixture (empty, missing_fields, etc.)

**Usage:**
```python
from tests.fixtures import FixtureLoader

loader = FixtureLoader()
# Get LLM response in different formats
response = loader.get_llm_response("gatherer_analysis", variant="markdown")
# Get search results
results = loader.get_search_results("acme_corp")
# Get job postings
jobs = loader.get_job_postings("greenhouse")
```

### Test Quality Assessment

**What current tests DO verify:**
- Agent pipeline flow (state passes correctly between agents)
- Error handling paths (when mocks throw exceptions)
- State mutation (progress flags, feedback lists)
- SQLite checkpointing (real LangGraph/SQLite)
- ✅ JSON parsing robustness (markdown, extra text, whitespace)
- ✅ Realistic search result data structures
- ✅ Realistic job posting data structures
- ✅ Data flow with realistic fixtures

**What current tests DON'T verify (NEXT STEPS):**
- Real Ollama LLM responses (actual model calls)
- Actual semantic matching with ChromaDB
- Full E2E workflow with live services

### Immediate Next Action
**Add E2E Tests with Real Ollama**

1. **Create Ollama-based tests**: Use real local LLM
   - Configure test to use `llama3.2:3b` via Ollama
   - Test actual prompt → response → parsing flow
   - Mark as `@pytest.mark.slow` for CI skip option

2. **Integrate robust JSON parsing into agents**:
   - Move `extract_json_from_llm_response` to `src/utils/`
   - Update agent code to use robust parsing

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

### Tests (290 passing)

| File | Tests | Purpose | Status |
|------|-------|---------|--------|
| `tests/test_agents/test_coordinator.py` | 31 | CoordinatorAgent full coverage | ✅ |
| `tests/test_agents/test_gatherer.py` | 16 | GathererAgent full coverage | ✅ |
| `tests/test_agents/test_identifier.py` | 31 | IdentifierAgent full coverage | ✅ |
| `tests/test_agents/test_validator.py` | 35 | ValidatorAgent full coverage | ✅ |
| `tests/test_integration/test_pipeline.py` | 13 | Agent pipeline flow | ⚠️ Mocked |
| `tests/test_integration/test_feedback_loops.py` | 16 | Human feedback routing | ⚠️ Mocked |
| `tests/test_integration/test_error_recovery.py` | 17 | Error handling paths | ⚠️ Mocked |
| `tests/test_integration/test_checkpointing.py` | 17 | SQLite checkpointing | ✅ Real |
| `tests/test_integration/test_realistic_fixtures.py` | 28 | Realistic fixture tests | ✅ Real Data |
| Other test files (core, router, data sources) | 86 | Infrastructure | ✅ |

**Total Tests**: 290 passing

**Test Fixture Files** (in `tests/fixtures/`):
- `loader.py` - FixtureLoader utility + `extract_json_from_llm_response` helper
- `llm_responses/*.json` - 5 LLM response fixtures with format variants
- `search_results/*.json` - 2 search result fixtures (Acme Corp, NovaTech)
- `job_postings/*.json` - 3 job board fixtures (Greenhouse, Lever, Generic)

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

### ⏳ Phase 4: Testing & Polish (IN PROGRESS)

**Step 1: Integration Tests (DONE - but mocked)**
- [x] `test_pipeline.py` - 13 tests for agent pipeline flow
- [x] `test_feedback_loops.py` - 16 tests for feedback routing
- [x] `test_error_recovery.py` - 17 tests for error handling
- [x] `test_checkpointing.py` - 17 tests for SQLite persistence
- [x] Installed `langgraph`, `langgraph-checkpoint`, `langgraph-checkpoint-sqlite`
- [x] Fixed mock interface issues (method names, spec restrictions)

**Step 2: Improve Tests with Realistic Fixtures (CURRENT - NOT STARTED)**
- [ ] Create `tests/fixtures/` directory structure
- [ ] Add realistic LLM response fixtures (varied formats)
- [ ] Add realistic search result fixtures (DuckDuckGo structure)
- [ ] Add Ollama-based tests using `llama3.2:3b`
- [ ] Test JSON parsing robustness (markdown wrapping, extra text)
- [ ] Mark slow tests with `@pytest.mark.slow`

**Step 3: E2E Tests (NOT STARTED)**
- [ ] Full workflow with real Ollama LLM
- [ ] Test actual data flow from search to opportunities
- [ ] Test checkpointing with real state changes

**Step 4: CLI & Documentation (NOT STARTED)**
- [ ] CLI interface for running research
- [ ] Usage documentation
- [ ] Example workflows

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
5. `tests/fixtures/loader.py` - FixtureLoader utility + `extract_json_from_llm_response` helper
6. `tests/test_integration/test_realistic_fixtures.py` - Fixture-based integration tests

### Current Task Context (for E2E tests)

When continuing work on E2E tests with Ollama, read:
- `tests/fixtures/loader.py` - Has `extract_json_from_llm_response` that could be moved to `src/utils/`
- `src/core/model_router.py` - ModelRouter that routes to Ollama
- `src/agents/gatherer.py` - See `_analyze_with_llm()` for JSON parsing patterns
- `config.py` - Ollama configuration (model: `llama3.2:3b`)

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

---

## Mock Interface Reference (for tests)

When mocking data sources, use these correct method names:

```python
# DuckDuckGoMCPClient (mcp_ddg_client.py)
mock_mcp_client.search.return_value = []        # NOT web_search
mock_mcp_client.search_news.return_value = []   # NOT news_search
mock_mcp_client.fetch_content.return_value = ""

# JobBoardScraper (job_boards.py)
mock_job_scraper.fetch.return_value = []        # NOT scrape_career_pages

# ProductMatcher (product_catalog.py)
mock_product_matcher.match_requirements_to_products.return_value = []

# ModelRouter (model_router.py)
mock_model_router.generate.return_value = MagicMock(content='{"key": "value"}')
```

### Fixture Pattern for Mocks

```python
@pytest.fixture
def mock_mcp_client():
    """Provide mocked MCP client with default empty returns."""
    client = AsyncMock()  # Do NOT use spec= (too restrictive)
    client.search.return_value = []
    client.search_news.return_value = []
    client.fetch_content.return_value = ""
    return client
```

---

## Ollama Configuration for Testing

The project uses local Ollama for LLM calls. Ensure Ollama is running:

```powershell
# Check Ollama is running
ollama list

# Pull required model if not present
ollama pull llama3.2:3b

# Test model works
ollama run llama3.2:3b "Return only: {\"status\": \"ok\"}"
```

### Model Router Test Configuration

For realistic integration tests, configure ModelRouter to use Ollama:

```python
# In tests, use actual Ollama instead of mocks
from src.core.model_router import ModelRouter

async def test_with_real_ollama():
    router = ModelRouter()  # Uses settings from config.py
    response = await router.generate(
        prompt="Extract company name from: 'Acme Corp is hiring'",
        complexity=2  # Routes to llama3.2:3b
    )
    # Test actual response parsing
    assert "Acme" in response.content
```

---

**END OF ARCHITECTURE DOCUMENT**

*Phase 4 IN PROGRESS: Integration tests written (262 tests passing) but using hollow mocks.*
*Next: Improve tests with realistic fixtures and local Ollama testing.*
*See "Quick Context Recovery" section for current task details.*
