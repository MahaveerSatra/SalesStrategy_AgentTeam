# Enterprise Account Research System - Codebase Architecture

**Last Updated**: 2026-01-28
**Status**: Phase 4 IN PROGRESS - Robust JSON Parsing Complete, E2E Tests Next
**Test Status**: 326 tests passing

---

## Quick Context Recovery

**READ THIS FIRST** when restoring context after clearing chat:

1. **Project**: Multi-agent system for enterprise account research using LangGraph
2. **Current Phase**: Phase 4 IN PROGRESS - Robust JSON parsing integrated, E2E tests next
3. **What's Done**: All 4 agents + LangGraph workflow + human-in-loop + 326 tests + Robust JSON parsing
4. **What's Next**: E2E tests with real Ollama LLM, then CLI interface
5. **COMPLETED**: Robust JSON parsing integrated into all agents via `src/utils/json_parsing.py`

### Current Session Context (2026-01-28)

**Just Completed**:
- ✅ Created `src/utils/json_parsing.py` with robust JSON extraction utilities
- ✅ Added `extract_json_from_llm_response()` function that handles:
  - Clean JSON (direct parsing)
  - JSON wrapped in markdown code fences (```json ... ```)
  - JSON with explanatory text before/after
  - JSON with extra whitespace/newlines
- ✅ Updated ALL 4 agents to use robust JSON parsing:
  - `GathererAgent` - 1 location updated
  - `IdentifierAgent` - 2 locations updated
  - `ValidatorAgent` - 2 locations updated
  - `CoordinatorAgent` - 3 locations updated
- ✅ Added 36 new tests for JSON parsing utility (`tests/test_utils/test_json_parsing.py`)
- ✅ All 326 tests passing

**Immediate Next Action**:
1. Create E2E tests with real Ollama LLM (mark as `@pytest.mark.slow`)
2. Build CLI interface for running research
3. Documentation and examples

---

## Phase 4 Goals (IN PROGRESS)

**Goals:**
1. ✅ Integration tests (multi-agent pipeline tests) - DONE
2. ✅ Realistic fixtures for testing - DONE
3. ✅ Robust JSON parsing integration - DONE (2026-01-28)
4. ⏳ E2E tests (full workflow with real Ollama LLM) - **CURRENT TASK**
5. ⏳ CLI interface for running research
6. ⏳ Documentation and examples

### Robust JSON Parsing Integration (COMPLETE)

**New Files Created:**
```
src/utils/
├── __init__.py           # Package exports
├── json_parsing.py       # Robust JSON extraction utilities
└── logging.py            # Existing logging module

tests/test_utils/
├── __init__.py           # Package init
└── test_json_parsing.py  # 36 tests for JSON parsing
```

**Key Functions in `src/utils/json_parsing.py`:**
```python
from src.utils.json_parsing import (
    extract_json_from_llm_response,  # Main function - extracts JSON from varied LLM output
    extract_json_with_default,        # Returns default on failure instead of raising
    safe_get_field,                   # Safely extract field with type validation
    JSONParseError,                   # Custom exception for parse failures
)
```

**Usage in Agents:**
```python
# Before (fragile):
result = json.loads(response.content)

# After (robust):
from src.utils.json_parsing import extract_json_from_llm_response, JSONParseError

result = extract_json_from_llm_response(response.content)
# Handles: markdown fences, extra text, whitespace variations
```

**Agents Updated:**
| Agent | Locations Updated | Methods |
|-------|-------------------|---------|
| GathererAgent | 1 | `_analyze_source_with_llm()` |
| IdentifierAgent | 2 | `_extract_requirements()`, `_generate_opportunities()` |
| ValidatorAgent | 2 | `_assess_risks()`, `_score_opportunities()` |
| CoordinatorAgent | 3 | `_validate_inputs()`, `_generate_clarifying_questions()`, `_parse_feedback_intent()` |

---

## Phase 3 Goals - ALL COMPLETE ✅

| Goal | Status | Evidence |
|------|--------|----------|
| 1. Implement all 4 agents | ✅ COMPLETE | CoordinatorAgent, GathererAgent, IdentifierAgent, ValidatorAgent |
| 2. Wire into LangGraph workflow | ✅ COMPLETE | `graph/workflow.py` with conditional routing |
| 3. Human-in-the-loop implementation | ✅ COMPLETE | `_wait_for_human` interrupt nodes, feedback loops |
| 4. End-to-end flow working | ✅ COMPLETE | Full pipeline: Entry → Gather → Identify → Validate → Exit → Feedback |

---

## Executive Summary

**Total Codebase**: ~5,700+ lines of production code + ~1,800 lines of tests

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ COMPLETE | Core infrastructure (config, router, logging, exceptions) |
| Phase 2 | ✅ COMPLETE | Data layer (MCP client, scrapers, product catalog, workflow) |
| Phase 3 | ✅ COMPLETE | Agent implementations (4/4) + human-in-loop + workflow integration |
| Phase 4 | ⏳ IN PROGRESS | Testing & Polish (integration, E2E, CLI) |

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
│  Agent Layer (agents/) - PHASE 3 COMPLETE + ROBUST JSON PARSING      │
│  ✅ CoordinatorAgent (entry/exit, human-in-loop, routing) - COMPLETE │
│  ✅ GathererAgent (collect & analyze data from sources) - COMPLETE   │
│  ✅ IdentifierAgent (find opportunities) - COMPLETE                  │
│  ✅ ValidatorAgent (confidence scoring, risk assessment) - COMPLETE  │
│  ✅ All agents use extract_json_from_llm_response() for robustness   │
├──────────────────────────────────────────────────────────────────────┤
│  Core Services (core/) + Utilities (utils/)                          │
│  - ModelRouter: 3-tier LLM routing with caching                      │
│  - BaseAgent: Abstract base with monitoring                          │
│  - Exceptions: Custom hierarchy                                      │
│  - json_parsing: Robust JSON extraction from LLM responses           │
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
| `utils/json_parsing.py` | ~150 | Robust JSON extraction from LLM responses | ✅ NEW |

**Total Phase 1**: ~1,070 lines

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
| `agents/coordinator.py` | ~580 | Dual entry/exit, human-in-loop, feedback routing | ✅ + JSON parsing |
| `agents/gatherer.py` | ~530 | Collect & analyze data with LLM from MCP, jobs, news | ✅ + JSON parsing |
| `agents/identifier.py` | ~350 | LLM-based opportunity identification with ProductMatcher | ✅ + JSON parsing |
| `agents/validator.py` | ~300 | Confidence scoring, risk assessment, filtering | ✅ + JSON parsing |

**Total Phase 3**: ~1,780 lines

---

### Tests (326 passing)

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
| `tests/test_utils/test_json_parsing.py` | 36 | JSON parsing utility tests | ✅ NEW |
| Other test files (core, router, data sources) | 86 | Infrastructure | ✅ |

**Total Tests**: 326 passing

**Test Fixture Files** (in `tests/fixtures/`):
- `loader.py` - FixtureLoader utility + legacy `extract_json_from_llm_response` helper
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
- **JSON Parsing**: Uses `extract_json_from_llm_response()` in 3 methods

### GathererAgent ✅
- **File**: `src/agents/gatherer.py` (~530 lines)
- **Tests**: 16 passing
- **Role**: Intelligence collection with LLM analysis
- **Sources**: DuckDuckGo MCP (web + news), JobBoardScraper
- **Outputs**: signals, job_postings, news_items, tech_stack
- **JSON Parsing**: Uses `extract_json_from_llm_response()` in `_analyze_source_with_llm()`

### IdentifierAgent ✅
- **File**: `src/agents/identifier.py` (~350 lines)
- **Tests**: 31 passing
- **Role**: Opportunity identification from gathered data
- **Uses**: ProductMatcher (semantic search), ModelRouter (LLM reasoning)
- **Outputs**: opportunities (list of Opportunity objects with evidence)
- **JSON Parsing**: Uses `extract_json_from_llm_response()` in 2 methods

### ValidatorAgent ✅
- **File**: `src/agents/validator.py` (~300 lines)
- **Tests**: 35 passing
- **Role**: Risk assessment and confidence scoring
- **Features**: 5 risk categories, confidence re-scoring, 0.6 threshold filtering
- **Outputs**: validated_opportunities, competitive_risks
- **JSON Parsing**: Uses `extract_json_from_llm_response()` in 2 methods

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
- [x] All tests passing

### ⏳ Phase 4: Testing & Polish (IN PROGRESS)

**Step 1: Integration Tests (DONE - but mocked)**
- [x] `test_pipeline.py` - 13 tests for agent pipeline flow
- [x] `test_feedback_loops.py` - 16 tests for feedback routing
- [x] `test_error_recovery.py` - 17 tests for error handling
- [x] `test_checkpointing.py` - 17 tests for SQLite persistence
- [x] Installed `langgraph`, `langgraph-checkpoint`, `langgraph-checkpoint-sqlite`
- [x] Fixed mock interface issues (method names, spec restrictions)

**Step 2: Realistic Fixtures (DONE)**
- [x] Create `tests/fixtures/` directory structure
- [x] Add realistic LLM response fixtures (varied formats)
- [x] Add realistic search result fixtures (DuckDuckGo structure)
- [x] Test JSON parsing robustness (markdown wrapping, extra text)

**Step 3: Robust JSON Parsing Integration (DONE - 2026-01-28)**
- [x] Create `src/utils/json_parsing.py` with robust extraction
- [x] Add `JSONParseError` custom exception
- [x] Update GathererAgent to use robust parsing
- [x] Update IdentifierAgent to use robust parsing (2 locations)
- [x] Update ValidatorAgent to use robust parsing (2 locations)
- [x] Update CoordinatorAgent to use robust parsing (3 locations)
- [x] Add 36 tests for JSON parsing utility
- [x] All 326 tests passing

**Step 4: E2E Tests with Ollama (CURRENT - NOT STARTED)**
- [ ] Create `tests/test_integration/test_e2e_ollama.py`
- [ ] Test ModelRouter with real Ollama calls
- [ ] Test agent JSON parsing with real LLM responses
- [ ] Test simplified E2E flow with real LLM
- [ ] Mark as `@pytest.mark.slow` for CI skip option

**Step 5: CLI & Documentation (NOT STARTED)**
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

# Run all tests quickly
python -m pytest tests/ -v --tb=short

# Run specific test categories
python -m pytest tests/test_agents/ -v           # Agent tests
python -m pytest tests/test_integration/ -v      # Integration tests
python -m pytest tests/test_utils/ -v            # Utility tests (JSON parsing)

# Run specific agent tests
python -m pytest tests/test_agents/test_coordinator.py -v
python -m pytest tests/test_agents/test_gatherer.py -v
python -m pytest tests/test_agents/test_identifier.py -v
python -m pytest tests/test_agents/test_validator.py -v

# Run JSON parsing tests only
python -m pytest tests/test_utils/test_json_parsing.py -v

# Skip slow tests (when E2E tests exist)
python -m pytest tests/ -v -m "not slow"

# Count total tests
python -m pytest tests/ --co -q 2>&1 | tail -3

# Check imports work
python -c "from src.agents import CoordinatorAgent, GathererAgent, IdentifierAgent, ValidatorAgent, WorkflowRoute; print('OK')"
python -c "from src.utils.json_parsing import extract_json_from_llm_response, JSONParseError; print('OK')"
```

---

## Key Files for Context Recovery

When restoring context, read these files in order:

1. **This file** (`CODEBASE_ARCHITECTURE.md`) - Architecture + status + next steps
2. `src/utils/json_parsing.py` - Robust JSON extraction (NEW)
3. `src/models/state.py` - State structure (ResearchState, Opportunity, Signal)
4. `src/graph/workflow.py` - LangGraph workflow definition
5. `src/agents/coordinator.py` - Human-in-loop patterns
6. `tests/test_utils/test_json_parsing.py` - JSON parsing test examples

### Current Task Context (for E2E tests)

When continuing work on E2E tests with Ollama, read:
- `src/utils/json_parsing.py` - Robust JSON parsing already integrated into agents
- `src/core/model_router.py` - ModelRouter that routes to Ollama
- `src/agents/gatherer.py` - See `_analyze_source_with_llm()` for LLM usage patterns
- `config.py` - Ollama configuration (model: `llama3.2:3b`)

**Ollama is available locally** with model `llama3.2:3b` (verified via `ollama list`)

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

## JSON Parsing Utility Reference

**Location**: `src/utils/json_parsing.py`

```python
from src.utils.json_parsing import (
    extract_json_from_llm_response,  # Main function
    extract_json_with_default,        # Returns default on failure
    safe_get_field,                   # Safe field extraction with type checking
    JSONParseError,                   # Custom exception
)

# Usage examples:
response_text = '''Here is the analysis:
```json
{"confidence": 0.85, "summary": "Acme Corp is expanding"}
```
This looks promising!'''

# Extract JSON from varied LLM output formats
result = extract_json_from_llm_response(response_text)
# Returns: {"confidence": 0.85, "summary": "Acme Corp is expanding"}

# With default fallback
result = extract_json_with_default(response_text, {"error": True})

# Safe field extraction
confidence = safe_get_field(result, "confidence", 0.0, float)
```

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

*Phase 4 IN PROGRESS: Robust JSON parsing integrated (326 tests passing).*
*Next: E2E tests with real Ollama LLM, then CLI interface.*
*See "Quick Context Recovery" section for current task details.*
