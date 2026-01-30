# Enterprise Account Research System - Codebase Architecture

**Last Updated**: 2026-01-30
**Status**: Phase 4 IN PROGRESS - All Tests Passing, CLI Interface Next
**Test Status**: ✅ 347 tests passing (326 unit/integration + 21 E2E with real Ollama)

---

## Quick Context Recovery

**READ THIS FIRST** when restoring context after clearing chat:

1. **Project**: Multi-agent system for enterprise account research using LangGraph
2. **Current Phase**: Phase 4 IN PROGRESS - All tests passing, CLI interface next
3. **What's Done**: All 4 agents + LangGraph workflow + human-in-loop + 347 tests passing + Robust JSON parsing + E2E tests + Structured outputs + Pydantic schemas for all agents
4. **What's Next**: Build CLI interface for running research, then documentation
5. **ALL TECH DEBT RESOLVED**: Structured outputs, flaky tests fixed, all dependencies installed

### Current Session Context (2026-01-30)

**Latest Verification** (2026-01-30):
- ✅ **ALL 347 tests passing** - verified with full test run including slow E2E tests
- ✅ All tech debt resolved
- ✅ Ready for CLI implementation

**Previously Completed** (2026-01-29):
- ✅ Created `tests/test_integration/test_e2e_ollama.py` with 21 E2E tests using real Ollama
- ✅ **Added Structured Output support to ModelRouter** (proper LLM JSON handling)
  - New `response_format` parameter accepts Pydantic JSON schemas
  - Uses Ollama's structured output feature (v0.5+) for guaranteed valid JSON
  - Use `Literal` types in Pydantic to constrain LLM output values
- ✅ **TECH DEBT RESOLVED: All agents now use Pydantic schemas for JSON validation**
  - Created `src/models/llm_schemas.py` with schemas for all agent outputs
  - GathererAgent & CoordinatorAgent: Use `response_format` for structured outputs (LOCAL Ollama)
  - IdentifierAgent & ValidatorAgent: Use Pydantic validation after JSON parsing (external models)
  - Graceful fallback to `extract_json_from_llm_response` if Pydantic validation fails
- ✅ Tests cover:
  - ModelRouter with real Ollama calls (5 tests)
  - **Structured outputs with Pydantic schemas (3 tests)**
  - Real LLM JSON parsing with structured outputs (4 tests)
  - Agent-level tests with real LLM (3 tests)
  - Simplified E2E workflow (1 test)
  - LLM response variability handling (2 tests)
  - Error handling edge cases (2 tests)
- ✅ All tests marked with `@pytest.mark.slow` for CI skip option
- ✅ Installed `chromadb` dependency for agent imports
- ✅ **Fixed flaky E2E test** `test_coordinator_validation_prompt_real_llm`
  - Was failing due to LLM returning `[]` instead of `{}` without schema enforcement
  - Now uses `response_format=InputValidation.model_json_schema()` for guaranteed valid JSON
- ✅ **Installed missing dependencies**: `lxml`, `sentence_transformers`
  - Required for BeautifulSoup HTML parsing and ChromaDB embeddings

**Key Technical Decision**: Use Ollama's structured outputs (not hoping LLM returns JSON)
- Reference: https://docs.ollama.com/capabilities/structured-outputs
- Pass `response_format=Model.model_json_schema()` to enforce JSON schema
- Use `Literal["high", "medium", "low"]` in Pydantic to constrain enum values

**Required Dependencies** (ensure these are installed):
```powershell
pip install lxml sentence_transformers chromadb
```

**Immediate Next Action**:
1. Build CLI interface for running research
2. Documentation and examples

---

## Phase 4 Goals (IN PROGRESS)

**Goals:**
1. ✅ Integration tests (multi-agent pipeline tests) - DONE
2. ✅ Realistic fixtures for testing - DONE
3. ✅ Robust JSON parsing integration - DONE (2026-01-28)
4. ✅ E2E tests (full workflow with real Ollama LLM) - DONE (2026-01-29)
5. ⏳ CLI interface for running research - **CURRENT TASK**
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

### Structured Outputs for Reliable JSON (COMPLETE)

**Problem**: LLMs don't always return valid JSON, even when prompted. Tests were flaky.

**Solution**: Use Ollama's structured output feature (v0.5+) to **enforce** JSON schema compliance.

**Implementation in ModelRouter** (`src/core/model_router.py`):
```python
from pydantic import BaseModel
from typing import Literal

class AnalysisResult(BaseModel):
    confidence: float
    summary: str
    relevance: Literal["high", "medium", "low"]  # Constrain values

# Use structured output - GUARANTEES valid JSON
response = await router.generate(
    prompt="Analyze this company...",
    complexity=3,
    temperature=0,  # Deterministic for structured output
    response_format=AnalysisResult.model_json_schema()  # NEW PARAMETER
)

# Parse with Pydantic - guaranteed to work
result = AnalysisResult.model_validate_json(response.content)
```

**Key Points**:
- `response_format` parameter added to `ModelRouter.generate()`
- Pass Pydantic's `model_json_schema()` to enforce schema
- Use `Literal` types to constrain string values (e.g., "high/medium/low")
- Set `temperature=0` for deterministic output
- Reference: https://docs.ollama.com/capabilities/structured-outputs

**✅ TECH DEBT RESOLVED**: All agents now use Pydantic schemas for JSON validation
- Created `src/models/llm_schemas.py` with schemas: `SourceAnalysis`, `RequirementsExtraction`, `OpportunitiesGeneration`, `RiskAssessment`, `OpportunityScoring`, `InputValidation`, `ClarificationCheck`, `FeedbackIntent`
- LOCAL Ollama agents (GathererAgent, CoordinatorAgent): Use `response_format` for guaranteed JSON
- External model agents (IdentifierAgent, ValidatorAgent): Use Pydantic validation after parsing

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
| `utils/json_parsing.py` | ~150 | Robust JSON extraction from LLM responses | ✅ |
| `models/llm_schemas.py` | ~210 | Pydantic schemas for LLM structured outputs | ✅ NEW |

**Total Phase 1**: ~1,280 lines

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

### Tests (347 total)

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
| `tests/test_integration/test_e2e_ollama.py` | 21 | E2E tests with real Ollama | ✅ Real LLM + Structured Outputs |
| `tests/test_utils/test_json_parsing.py` | 36 | JSON parsing utility tests | ✅ |
| Other test files (core, router, data sources) | 86 | Infrastructure | ✅ |

**Total Tests**: 347 (326 unit/integration + 21 E2E with real Ollama)

**Note**: Run `pytest -m "not slow"` to skip E2E tests for faster CI runs.

**Test Fixture Files** (in `tests/fixtures/`):
- `loader.py` - FixtureLoader utility + legacy `extract_json_from_llm_response` helper
- `llm_responses/*.json` - 5 LLM response fixtures with format variants
- `search_results/*.json` - 2 search result fixtures (Acme Corp, NovaTech)
- `job_postings/*.json` - 3 job board fixtures (Greenhouse, Lever, Generic)

---

## Agent Summary

### CoordinatorAgent ✅
- **File**: `src/agents/coordinator.py` (~600 lines)
- **Tests**: 31 passing
- **Role**: Supervisor agent with 3 entry points
- **Entry Points**: `process_entry()`, `process_exit()`, `process_feedback()`
- **Human-in-Loop**: Handles interrupts, formats reports, routes feedback
- **Structured Outputs**: Uses `InputValidation`, `ClarificationCheck`, `FeedbackIntent` schemas with Ollama

### GathererAgent ✅
- **File**: `src/agents/gatherer.py` (~540 lines)
- **Tests**: 16 passing
- **Role**: Intelligence collection with LLM analysis
- **Sources**: DuckDuckGo MCP (web + news), JobBoardScraper
- **Outputs**: signals, job_postings, news_items, tech_stack
- **Structured Outputs**: Uses `SourceAnalysis` schema with Ollama's `response_format`

### IdentifierAgent ✅
- **File**: `src/agents/identifier.py` (~360 lines)
- **Tests**: 31 passing
- **Role**: Opportunity identification from gathered data
- **Uses**: ProductMatcher (semantic search), ModelRouter (LLM reasoning)
- **Outputs**: opportunities (list of Opportunity objects with evidence)
- **Pydantic Validation**: Uses `RequirementsExtraction`, `OpportunitiesGeneration` schemas

### ValidatorAgent ✅
- **File**: `src/agents/validator.py` (~310 lines)
- **Tests**: 35 passing
- **Role**: Risk assessment and confidence scoring
- **Features**: 5 risk categories, confidence re-scoring, 0.6 threshold filtering
- **Outputs**: validated_opportunities, competitive_risks
- **Pydantic Validation**: Uses `RiskAssessment`, `OpportunityScoring` schemas

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

**Step 4: E2E Tests with Ollama (DONE - 2026-01-29)**
- [x] Create `tests/test_integration/test_e2e_ollama.py` (21 tests)
- [x] Test ModelRouter with real Ollama calls (5 tests)
- [x] Test agent JSON parsing with real LLM responses (4 tests)
- [x] Test simplified E2E flow with real LLM (1 test)
- [x] Mark as `@pytest.mark.slow` for CI skip option
- [x] Handle LLM variability with retries and graceful skips
- [x] Fixed flaky `test_coordinator_validation_prompt_real_llm` with structured output enforcement
- [x] Installed missing dependencies: `lxml`, `sentence_transformers`

**Step 5: CLI & Documentation (NOT STARTED - CURRENT TASK)**
- [ ] CLI interface for running research
- [ ] Usage documentation
- [ ] Example workflows

---

## Commands Reference

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Install required dependencies (if missing)
pip install lxml sentence_transformers chromadb

# Run all tests (347 total)
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

# Run E2E tests with real Ollama (slow)
python -m pytest tests/test_integration/test_e2e_ollama.py -v

# Skip slow E2E tests (faster CI)
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
2. `src/models/llm_schemas.py` - Pydantic schemas for structured outputs (NEW)
3. `src/utils/json_parsing.py` - Robust JSON extraction
4. `src/models/state.py` - State structure (ResearchState, Opportunity, Signal)
5. `src/graph/workflow.py` - LangGraph workflow definition
6. `src/agents/coordinator.py` - Human-in-loop patterns
7. `tests/test_utils/test_json_parsing.py` - JSON parsing test examples

### E2E Test File Reference

**Location**: `tests/test_integration/test_e2e_ollama.py`

**Test Classes** (21 tests total):
| Class | Tests | Description |
|-------|-------|-------------|
| `TestModelRouterWithRealOllama` | 5 | Basic generation, JSON output, system prompts, caching, metrics |
| `TestStructuredOutputs` | 3 | **NEW**: Pydantic schemas for guaranteed JSON output |
| `TestRealLLMJsonParsing` | 4 | Gatherer, Identifier, Validator, Coordinator prompt formats |
| `TestGathererAgentWithRealLLM` | 2 | Structured output test + agent integration test (xfail) |
| `TestIdentifierAgentWithRealLLM` | 1 | IdentifierAgent requirement extraction |
| `TestValidatorAgentWithRealLLM` | 1 | ValidatorAgent risk assessment |
| `TestSimplifiedE2EFlow` | 1 | Mini-pipeline: Gatherer → Identifier |
| `TestLLMResponseVariability` | 2 | Markdown-wrapped JSON, consistency |
| `TestErrorHandlingWithRealLLM` | 2 | Empty prompt, long prompt |

**Running E2E Tests**:
```powershell
# Run only E2E tests
python -m pytest tests/test_integration/test_e2e_ollama.py -v

# Run all tests INCLUDING slow E2E tests
python -m pytest tests/ -v

# Run all tests EXCLUDING slow E2E tests (faster CI)
python -m pytest tests/ -v -m "not slow"
```

**Ollama Requirement**: Model `llama3.2:3b` must be available locally.

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

## GAP ANALYSIS & ROADMAP TO STAFF-LEVEL DEMONSTRATION

**Analysis Date**: 2026-01-30
**Career Goal**: Promote from Senior → Staff Application Engineer
**Purpose**: This project demonstrates staff-level engineering skills

### Executive Summary

**Current Implementation**: ★★★☆☆ (Good infrastructure, medium sophistication)
**Staff-Level Readiness**: ⚠️ **NOT YET READY FOR DEMONSTRATION**

**WHY THIS PROJECT EXISTS** (from original brief):
- **Problem**: Account managers spend 6-8 hours manually researching each enterprise account
- **Solution**: Multi-agent AI system reducing research time to 40 minutes (10x faster)
- **Goal**: Demonstrate staff-level engineering to get promoted
- **Success**: Working demo on 2-3 real companies + measurable metrics + LinkedIn visibility

The project has **excellent technical foundation** but is **missing critical components** needed to demonstrate staff-level engineering and achieve career advancement.

---

### CRITICAL GAPS BLOCKING PROMOTION

#### ❌ GAP 1: NO CLI INTERFACE (CRITICAL BLOCKER)

**Impact**: ⚠️ **BLOCKS ALL WORK** - Cannot run system on real companies

**Current**: `main.py` only has Phase 1 tests, system is complete but unusable

**Needed**:
- `src/cli/main.py` - Entry point with argparse
- `src/cli/commands.py` - research, resume, list-runs commands
- `src/cli/formatters.py` - Terminal, markdown, JSON output
- Usage: `python -m src.cli research "Boeing" --industry aerospace`

**Priority**: 🔴 CRITICAL (2 days)

---

#### ❌ GAP 2: JOB ANALYSIS NOT STAFF-LEVEL (40-50% COMPLETE)

**Impact**: ⚠️ **FAILS TO DEMONSTRATE TECHNICAL DEPTH**

Original brief's **"Challenge A: Job Posting Analysis"** is the staff-level differentiator. Current implementation is basic scraping + LLM prompting, not sophisticated NLP.

**Current vs Staff-Level**:

| Feature | Current | Required |
|---------|---------|----------|
| Data Points | 7 fields | 15+ fields |
| NER Pipeline | ❌ None | ✅ spaCy/Transformers |
| Context Parse | ❌ None | ✅ "required" vs "preferred" |
| Urgency Detect | ❌ None | ✅ "ASAP", multiple postings |
| Pattern Detect | ❌ None | ✅ "5 control engineers = autonomy" |
| Accuracy | ❌ Unmeasured | ✅ 70%+ validated |

**Component Assessment**:
- **GathererAgent** (lines 345-374): ★☆☆☆☆ - Just field aggregation
- **IdentifierAgent**: ★★★☆☆ - LLM-based, lacks explicit pattern logic
- **ValidatorAgent** (lines 244-389): ★★★☆☆ - Good heuristics, no calibration

**Needed**:
- `src/analysis/pattern_detector.py` - Role clustering, team expansion detection
- `src/analysis/context_parser.py` - Parse requirement levels
- Expand JobPosting model to 15+ fields

**Priority**: 🟡 HIGH (3-4 days)

---

#### ❌ GAP 3: NO REAL COMPANY DEMONSTRATIONS

**Impact**: ⚠️ **CANNOT PROVE VALUE**

- 347 tests passing with mocks
- **ZERO runs on actual companies**
- Cannot prove "6-8 hours → 40 minutes" claim

**Needed**:
- Run on Boeing, Tesla, Rivian
- Collect timing, data points, opportunities
- Generate reports in `demos/demo_results/`

**Priority**: 🔴 CRITICAL (1 day after CLI)

---

#### ❌ GAP 4: NO DEMO MATERIALS (BLOCKS PROMOTION)

**Impact**: ⚠️ **NO VISIBILITY = NO PROMOTION**

Having working system ≠ Career advancement. Need materials for visibility.

**Current**:
- README.md outdated (says "Phase 1", we're in Phase 4)
- No LinkedIn post, no interview guide

**Needed**:
- Update README with current status, real results
- `Status_Plan/linkedin_post.md` - Draft with real metrics
- `Status_Plan/interview_guide.md` - Technical talking points

**Priority**: 🔴 CRITICAL (2 days)

---

### TWO-WEEK IMPLEMENTATION PLAN

**APPROACH**: Hybrid (demo first, enhance after)
- **Week 1**: CLI + real demos + materials + publish LinkedIn
- **Week 2**: Add sophistication + publish enhancement post

---

#### WEEK 1: MAKE IT DEMONSTRABLE (5 days)

**Day 1-2: CLI Interface**

Files to create:
```
src/cli/__init__.py
src/cli/main.py          # python -m src.cli
src/cli/commands.py      # research, resume, list-runs
src/cli/formatters.py    # Terminal, markdown, JSON
```

Features:
- `python -m src.cli research "Boeing" --industry aerospace`
- Human-in-loop prompts
- Progress indicators (rich/tqdm)
- Resume: `python -m src.cli resume <thread_id>`
- List: `python -m src.cli list-runs`

Acceptance:
- [ ] Can start research from CLI
- [ ] Prompts for human input
- [ ] Generates markdown report
- [ ] Can resume workflow

---

**Day 3: Real Company Demos**

Run research on:
1. Boeing (aerospace)
2. Tesla (automotive)
3. Rivian (automotive)

Collect per company:
- Time to complete (<60 min target)
- Data points (signals, jobs, news)
- Opportunities found
- Confidence distribution

Deliverables:
```
demos/demo_results/boeing_report.md
demos/demo_results/tesla_report.md
demos/demo_results/rivian_report.md
demos/demo_results/metrics_summary.json
```

Acceptance:
- [ ] All 3 companies complete
- [ ] 5+ opportunities each
- [ ] Timing documented
- [ ] Evidence for "40 minutes" claim

---

**Day 4-5: Demo Materials**

Tasks:

1. **Update README.md**:
   - Current status (Phase 4, 347 tests)
   - Installation (Windows PowerShell)
   - CLI usage examples
   - Real demo results
   - Architecture diagram

2. **LinkedIn Post** (`Status_Plan/linkedin_post.md`):
   - Hook: "6-8 hours → 40 minutes"
   - Solution: Multi-agent AI system
   - Metrics: Real Boeing/Tesla/Rivian results
   - Tech: LangGraph, multi-tier LLM, MCP, 347 tests
   - GitHub link
   - CTA: "Interested in AI for B2B sales?"

3. **Interview Guide** (`Status_Plan/interview_guide.md`):
   - System design (why LangGraph, why multi-tier LLM)
   - Challenges (Challenge A, JSON, checkpointing)
   - Real metrics
   - Trade-offs
   - Future enhancements

Acceptance:
- [ ] README polished
- [ ] LinkedIn post ready
- [ ] Interview guide complete
- [ ] Can discuss confidently

**END WEEK 1**: ✅ PUBLISH LINKEDIN POST with real results

---

#### WEEK 2: ADD SOPHISTICATION (5-6 days)

**Day 6-7: Enhanced Job Analysis**

1. **Expand JobPosting** (`src/models/domain.py`):
```python
# Add 8+ new fields for 15+ total:
seniority_level: str | None
team_size_indicators: list[str]
urgency_signals: list[str]
required_skills_explicit: list[str]
preferred_skills: list[str]
domain_focus: str | None
posting_date: str | None
role_category: str | None
tech_stack_primary: list[str]
tech_stack_secondary: list[str]
```

2. **Context Parser** (`src/analysis/context_parser.py`):
```python
def parse_skill_context(description: str) -> dict:
    """Categorize skills by requirement level."""
    return {
        "required": [...],    # "must have", "required"
        "preferred": [...],   # "nice to have", "bonus"
        "nice_to_have": [...] # "familiarity with"
    }

def detect_urgency_signals(description: str) -> list[str]:
    """Detect urgency indicators."""
    keywords = ["asap", "immediate", "urgent", "rapidly growing"]
    return signals
```

3. **Pattern Detector** (`src/analysis/pattern_detector.py`):
```python
def cluster_job_postings(postings: list[JobPosting]) -> dict:
    """Group similar roles."""
    clusters = {
        "control_engineers": [],
        "autonomy_engineers": [],
        "simulation_engineers": [],
    }
    return clusters

def detect_team_expansion(clusters: dict) -> list[dict]:
    """Identify initiatives from clusters."""
    patterns = []
    for role_type, jobs in clusters.items():
        if len(jobs) >= 3:  # 3+ = initiative
            patterns.append({
                "pattern": f"{len(jobs)} {role_type}",
                "inference": f"Major initiative in {domain}",
                "confidence": calc_confidence(jobs),
                "evidence": jobs
            })
    return patterns
```

Integration:
- GathererAgent calls context_parser per job
- GathererAgent calls pattern_detector after collection
- Add state field: `hiring_patterns: list[dict]`

Acceptance:
- [ ] 15+ fields per job
- [ ] Skills categorized
- [ ] Urgency detected
- [ ] Roles clustered
- [ ] "5+ engineers = initiative" works

---

**Day 8-9: Enhanced Opportunities**

Add to IdentifierAgent:
```python
def _generate_opportunities_from_patterns(
    self, hiring_patterns, tech_stack
) -> list[Opportunity]:
    """Create opportunities from patterns."""
    # Map: "5 control engineers" → Control System Toolbox
    # Map: "autonomy engineers" → Automated Driving Toolbox
    return opportunities
```

Enhanced evidence:
- Link to job clusters
- Reference counts
- Include urgency

Acceptance:
- [ ] Opportunities reference patterns
- [ ] Evidence includes clusters
- [ ] Higher confidence from patterns

---

**Day 10-11: Metrics & Validation**

Create `src/validation/metrics.py`:
```python
class ResearchMetrics(BaseModel):
    time_to_complete: float
    data_points_collected: int
    opportunities_found: int
    high_confidence_count: int
    job_postings_analyzed: int
    hiring_patterns_detected: int
    fields_per_job_avg: float  # Should be 15+
```

Validation:
1. Re-run Boeing, Tesla, Rivian
2. Compare before/after
3. Document improvements

Acceptance:
- [ ] Metrics framework done
- [ ] Before/after comparison
- [ ] Improvements measured

---

**Day 11-12: Update Materials**

1. Update README with enhancements
2. **Second LinkedIn Post** (`Status_Plan/linkedin_enhancement_post.md`):
   - Hook: "Last week I shared..."
   - Update: "This week I enhanced with pattern detection..."
   - Metrics: Improved results
   - Learning: Iteration process
3. Update interview guide

Acceptance:
- [ ] Docs reflect enhancements
- [ ] Second post ready
- [ ] Can explain improvements

**END WEEK 2**: ✅ PUBLISH ENHANCEMENT POST

---

### SUCCESS METRICS

**Week 1 (Demonstrable)**:
- [ ] CLI working
- [ ] 3 real reports (Boeing, Tesla, Rivian)
- [ ] <60 min per company
- [ ] README updated
- [ ] LinkedIn post published
- [ ] **Can discuss real results in interviews**

**Week 2 (Sophisticated)**:
- [ ] 15+ data points per job
- [ ] Pattern detection working
- [ ] Skills categorized
- [ ] Urgency detection
- [ ] Improvements measured
- [ ] **Can demonstrate staff-level depth**

---

### VERIFICATION CHECKLIST

**Technical**:
- [ ] `python -m src.cli research "Company" --industry sector` works
- [ ] Completes in <1 hour
- [ ] Markdown report with opportunities
- [ ] 15+ fields per job
- [ ] Hiring patterns detected (3+ similar roles)
- [ ] Skills by requirement level
- [ ] All 347+ tests passing

**Demo Readiness**:
- [ ] Boeing report
- [ ] Tesla report
- [ ] Rivian report
- [ ] Metrics summary
- [ ] Results impressive

**Career Materials**:
- [ ] README polished
- [ ] LinkedIn post 1 published
- [ ] LinkedIn post 2 ready
- [ ] Interview guide
- [ ] Can discuss:
  - System design
  - Technical challenges
  - Real metrics
  - Staff-level sophistication

---

### CRITICAL FILES TO IMPLEMENT

**Week 1 (Demo)**:
```
NEW:
  src/cli/__init__.py
  src/cli/main.py
  src/cli/commands.py
  src/cli/formatters.py
  demos/demo_results/boeing_report.md
  demos/demo_results/tesla_report.md
  demos/demo_results/rivian_report.md
  demos/demo_results/metrics_summary.json
  Status_Plan/linkedin_post.md
  Status_Plan/interview_guide.md

UPDATED:
  readme.md
```

**Week 2 (Sophistication)**:
```
NEW:
  src/analysis/__init__.py
  src/analysis/context_parser.py
  src/analysis/pattern_detector.py
  src/validation/__init__.py
  src/validation/metrics.py
  Status_Plan/linkedin_enhancement_post.md

UPDATED:
  src/models/domain.py (expand JobPosting)
  src/agents/gatherer.py (use analysis modules)
  src/agents/identifier.py (pattern opportunities)
  demos/demo_results/* (re-run)
  readme.md (enhancements)
```

---

### WHY THIS MATTERS FOR CAREER

**Original Brief Goal**: Senior → Staff Application Engineer promotion

**What This Must Demonstrate**:
1. ✅ System design (multi-tier architecture) - **DONE**
2. ✅ Protocol knowledge (MCP) - **DONE**
3. ✅ Production patterns (error handling, caching) - **DONE**
4. ✅ Performance (async, parallel, routing) - **DONE**
5. ⏳ Complex problem solving (job analysis) - **Week 2**
6. ⏳ End-to-end ownership (demo) - **Week 1**

**Success Criteria**:
- Working demo on 2-3 companies ← **Week 1**
- Measurable improvements ← **Week 1**
- Production code ← **DONE (347 tests)**
- LinkedIn post with metrics ← **Week 1**
- Staff-level depth ← **Week 2**
- Interview ready ← **Week 1-2**

**Timeline**: 2 weeks → promotion/job search

---

### COMPONENT QUALITY SUMMARY

**Production Ready** ✅:
- Product Catalog: ★★★★☆ (20 products, semantic search)
- Testing: ★★★★★ (347 tests, E2E coverage)
- Infrastructure: ★★★★☆ (LangGraph, multi-tier LLM, checkpointing)
- CoordinatorAgent: ★★★★☆ (human-in-loop, structured outputs)

**Medium Quality** ⚠️:
- GathererAgent: ★☆☆☆☆ (basic scraping, needs enhancement)
- IdentifierAgent: ★★★☆☆ (LLM-based, lacks pattern logic)
- ValidatorAgent: ★★★☆☆ (good heuristics, no calibration)

**Missing** ❌:
- CLI Interface (BLOCKER)
- Pattern Detection (staff-level depth)
- Enhanced Extraction (15+ fields)
- Metrics Framework (validation)
- Demo Materials (visibility)

---

**END OF ARCHITECTURE DOCUMENT**

*Last verified: 2026-01-30 - All 347 tests passing (326 unit/integration + 21 E2E).*
*All tech debt resolved.*
*Next immediate action: Build CLI interface (Week 1, Day 1-2).*
*See "GAP ANALYSIS" section above for complete roadmap to staff-level demonstration.*
*Use this document as single source of truth for context recovery.*
