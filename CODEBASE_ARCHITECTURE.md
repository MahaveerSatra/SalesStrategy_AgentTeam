# Enterprise Account Research System - Codebase Architecture

**Last Updated**: 2026-01-25
**Status**: Phase 3 In Progress (2/4 agents complete, workflow fully integrated)
**Test Status**: 133 tests passing

---

## Quick Context Recovery

**READ THIS FIRST** when restoring context after clearing chat:

1. **Project**: Multi-agent system for enterprise account research using LangGraph
2. **Current Phase**: Phase 3 - Agent Implementation (2/4 agents done)
3. **What's Done**: CoordinatorAgent + GathererAgent + workflow integration + all tests
4. **What's Next**: Step 4 - Implement IdentifierAgent

### Immediate Next Action
**Implement IdentifierAgent** (`src/agents/identifier.py`)
- Extract requirements from signals and job_postings
- Use ProductMatcher for semantic product matching
- Generate Opportunity objects with LLM

---

## Executive Summary

**Total Codebase**: ~4,800+ lines of production code + ~600 lines of tests

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ COMPLETE | Core infrastructure (config, router, logging, exceptions) |
| Phase 2 | ✅ COMPLETE | Data layer (MCP client, scrapers, product catalog) |
| Phase 3 | 🔄 IN PROGRESS | Agent implementations (2/4 complete) |

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
│  Agent Layer (agents/) - PHASE 3 IN PROGRESS                         │
│  ✅ CoordinatorAgent (entry/exit, human-in-loop, routing) - COMPLETE │
│  ✅ GathererAgent (collect & analyze data from sources) - COMPLETE   │
│  ⏳ IdentifierAgent (find opportunities) - NEXT                      │
│  ⏳ ValidatorAgent (confidence scoring) - TODO                       │
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

## Workflow Architecture (IMPLEMENTED)

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
│  │  _wait_for_human    │  Interrupt point for human clarification       │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼ (conditional: has current_report?)                        │
│  ┌─────────────────────┐                                                │
│  │     gatherer        │  Web search, job postings, news, LLM analysis  │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │    identifier       │  Extract requirements, match products (TODO)   │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │     validator       │  Score confidence, filter (TODO)               │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  coordinator_exit   │  Format report, present to human               │
│  └──────────┬──────────┘                                                │
│             │                                                            │
│             ▼                                                            │
│  ┌─────────────────────┐                                                │
│  │  _wait_for_human    │  Interrupt point for feedback                  │
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

### Phase 3: Agents (IN PROGRESS - 2/4 complete)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `agents/__init__.py` | ~15 | Package exports (CoordinatorAgent, GathererAgent, WorkflowRoute) | ✅ |
| `agents/coordinator.py` | ~580 | Dual entry/exit, human-in-loop, feedback routing | ✅ COMPLETE |
| `agents/gatherer.py` | ~530 | Collect & analyze data with LLM from MCP, jobs, news | ✅ COMPLETE |
| `agents/identifier.py` | ~250 | LLM-based opportunity identification | ⏳ NEXT |
| `agents/validator.py` | ~200 | Confidence scoring, risk assessment | ⏳ TODO |

### Tests

| File | Tests | Purpose | Status |
|------|-------|---------|--------|
| `tests/test_agents/test_coordinator.py` | 31 | CoordinatorAgent full coverage | ✅ |
| `tests/test_agents/test_gatherer.py` | 16 | GathererAgent full coverage | ✅ |
| Other test files | 86 | Core, data sources, model router | ✅ |

**Total Tests**: 133 passing

---

## Key State Structure (`models/state.py`)

```python
class ResearchState(TypedDict):
    """Main workflow state - ALL fields are implemented"""

    # Input (from user)
    account_name: str              # Company to research
    industry: str                  # Industry vertical
    region: str | None             # Geographic region
    user_context: str | None       # Additional context/meeting notes
    research_depth: ResearchDepth  # QUICK/STANDARD/DEEP

    # Collected data (GathererAgent populates)
    signals: list[Signal]          # Web search results with LLM analysis
    job_postings: list[dict]       # Scraped job postings
    news_items: list[dict]         # News articles
    tech_stack: list[str]          # Extracted technologies
    financial_data: dict | None    # Financial info if available

    # Analysis (IdentifierAgent populates - TODO)
    opportunities: list[Opportunity]  # Identified opportunities

    # Validation (ValidatorAgent populates - TODO)
    validated_opportunities: list[Opportunity]  # Filtered opportunities
    competitive_risks: list[str]   # Identified risks

    # Progress tracking
    progress: ResearchProgress     # Tracks which agents completed

    # Human interaction
    human_feedback: list[str]      # Conversation history
    waiting_for_human: bool        # Pause for human input
    human_question: str | None     # Question/report for human

    # Metadata
    started_at: datetime
    last_updated: datetime
    error_messages: list[str]
    confidence_scores: dict[str, float]

    # CoordinatorAgent fields (IMPLEMENTED)
    current_report: str | None     # Formatted report from process_exit()
    workflow_iteration: int        # Feedback loop count (default: 1)
    feedback_context: str | None   # Parsed guidance for retry
    next_route: str | None         # "gatherer"|"identifier"|"validator"|"complete"
```

---

## Agent Implementations

### CoordinatorAgent (COMPLETE) ✅

**File**: `src/agents/coordinator.py` (~580 lines)
**Tests**: `tests/test_agents/test_coordinator.py` (31 tests)
**Complexity**: 3 (LOCAL Ollama)

**Three Entry Points**:
1. `process_entry()` - Validate inputs, normalize name, ask questions
2. `process_exit()` - Format report, present to human
3. `process_feedback()` - Parse feedback, route to next agent

```python
class WorkflowRoute(str, Enum):
    GATHERER = "gatherer"      # Need more data
    IDENTIFIER = "identifier"  # Find different opportunities
    VALIDATOR = "validator"    # Re-evaluate scores
    COMPLETE = "complete"      # Human approved
```

---

### GathererAgent (COMPLETE) ✅

**File**: `src/agents/gatherer.py` (~530 lines)
**Tests**: `tests/test_agents/test_gatherer.py` (16 tests)
**Complexity**: 3 (LOCAL Ollama)

**Dependencies**:
- `DuckDuckGoMCPClient` (web search + content fetching)
- `JobBoardScraper` (job postings)
- `ModelRouter` (LLM analysis)

**Populates**:
- `state["signals"]` - Web search results with LLM analysis
- `state["job_postings"]` - Scraped job postings
- `state["news_items"]` - News articles
- `state["tech_stack"]` - Extracted technologies

---

### IdentifierAgent (NEXT) ⏳

**File**: `src/agents/identifier.py` (~250 lines estimated)
**Complexity**: 6-7 (Groq 8B for complex reasoning)

**Dependencies**:
- `ProductMatcher` from `data_sources/product_catalog.py`
- `ModelRouter` (LLM reasoning)

**Implementation Outline**:
```python
class IdentifierAgent(StatelessAgent):
    def __init__(self, product_matcher: ProductMatcher, model_router: ModelRouter):
        super().__init__(name="identifier")
        self.product_matcher = product_matcher
        self.model_router = model_router

    async def process(self, state: ResearchState) -> None:
        # 1. Extract requirements from job_postings and signals
        requirements = await self._extract_requirements(state)

        # 2. Match to products using semantic search
        matches = await self.product_matcher.match_requirements(requirements)

        # 3. Generate opportunity hypotheses with LLM
        opportunities = await self._generate_opportunities(state, matches)

        # 4. Create Opportunity objects with evidence
        state["opportunities"] = opportunities
        state["progress"].identifier_complete = True

    def get_complexity(self, state: ResearchState) -> int:
        return 6  # Groq 8B
```

**Populates**:
- `state["opportunities"]` - List of Opportunity objects

---

### ValidatorAgent (TODO) ⏳

**File**: `src/agents/validator.py` (~200 lines estimated)
**Complexity**: 6 (Groq 8B)

**Implementation Outline**:
```python
class ValidatorAgent(StatelessAgent):
    async def process(self, state: ResearchState) -> None:
        # 1. Assess competitive risks
        risks = await self._assess_risks(state)

        # 2. Score confidence for each opportunity
        scored = await self._score_opportunities(state["opportunities"])

        # 3. Filter: only keep confidence > 0.6
        validated = [o for o in scored if o.confidence_score > 0.6]

        # 4. Populate results
        state["validated_opportunities"] = validated
        state["competitive_risks"] = risks
        state["progress"].validator_complete = True
```

---

## Model Router Configuration

| Complexity | Model | Provider | Use Case |
|------------|-------|----------|----------|
| 1-3 | llama3.2:3b | LOCAL Ollama | CoordinatorAgent, GathererAgent |
| 4-7 | llama-3.1-8b-instant | Groq | IdentifierAgent, ValidatorAgent |
| 8-10 | llama-3.1-70b | Groq | Complex reasoning (if needed) |

---

## Current Status Summary

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Core Infrastructure | ✅ Complete | Passing | Phase 1 |
| Data Layer | ✅ Complete | 77/77 | Phase 2 |
| CoordinatorAgent | ✅ Complete | 31/31 | Entry/exit/feedback |
| GathererAgent | ✅ Complete | 16/16 | LLM analysis |
| Workflow Integration | ✅ Complete | - | Conditional routing |
| **IdentifierAgent** | ⏳ **NEXT** | TODO | Opportunity identification |
| ValidatorAgent | ⏳ TODO | TODO | Confidence scoring |

**Total Tests**: 133 passing

---

## Development Checklist

### ✅ Completed (2026-01-25)
- [x] CoordinatorAgent implementation (~580 lines)
- [x] CoordinatorAgent tests (31 tests)
- [x] GathererAgent implementation (~530 lines)
- [x] GathererAgent tests (16 tests)
- [x] state.py updated with Coordinator fields
- [x] workflow.py fully integrated with conditional routing
- [x] agents/__init__.py exports updated
- [x] All 133 tests passing

### ⏳ Next Steps
- [ ] **Step 4: IdentifierAgent implementation** ← NEXT
- [ ] Step 5: IdentifierAgent tests
- [ ] Step 6: ValidatorAgent implementation
- [ ] Step 7: ValidatorAgent tests
- [ ] Step 8: Integration tests
- [ ] Step 9: E2E tests

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

# Quick test summary
python -m pytest tests/ -v --tb=short 2>&1 | grep -E "(passed|failed)"

# Check imports work
python -c "from src.agents import CoordinatorAgent, GathererAgent, WorkflowRoute; print('OK')"
```

---

## Key Files for Context Recovery

When restoring context, read these files in order:

1. **This file** (`CODEBASE_ARCHITECTURE.md`) - Architecture + status + next steps
2. `src/models/state.py` - State structure (ResearchState TypedDict)
3. `src/agents/identifier.py` - Next file to implement
4. `src/data_sources/product_catalog.py` - ProductMatcher for Identifier
5. `src/agents/coordinator.py` - Reference for agent patterns
6. `src/graph/workflow.py` - Workflow integration reference

---

## ProductMatcher Reference (for IdentifierAgent)

The `ProductMatcher` class in `data_sources/product_catalog.py` provides semantic product matching:

```python
from src.data_sources.product_catalog import ProductMatcher

matcher = ProductMatcher(collection_name="products")

# Match requirements to products
matches = await matcher.match_requirements(
    requirements=["need ML training platform", "data visualization"],
    top_k=5,
    min_confidence=0.5
)

# Returns list of ProductMatch objects with:
# - product_name: str
# - confidence: float
# - matched_requirement: str
# - explanation: str
```

---

## Opportunity Model Reference (for IdentifierAgent)

```python
from src.models.state import Opportunity, OpportunityConfidence, Signal

opportunity = Opportunity(
    product_name="Enterprise Analytics Suite",
    rationale="Strong hiring signals in data engineering",
    evidence=[signal1, signal2],  # List of Signal objects
    target_persona="VP of Engineering",
    talking_points=["Data-driven culture", "Scaling needs"],
    estimated_value="$250K ARR",
    risks=["Competitor evaluation ongoing"],
    confidence=OpportunityConfidence.HIGH,
    confidence_score=0.85
)
```

---

**END OF ARCHITECTURE DOCUMENT**

*This document serves as the primary context recovery resource. Read it first when starting a new session.*
