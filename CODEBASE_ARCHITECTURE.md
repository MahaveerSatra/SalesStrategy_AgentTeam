# Enterprise Account Research System - Codebase Architecture

**Last Updated**: 2026-01-15
**Status**: Phase 2 Complete | Phase 3 In Progress (1/4 agents complete)

---

## 📋 Executive Summary

**Total Codebase**: ~3,200+ lines of production code
- ✅ Phase 1: Core infrastructure (config, router, logging, exceptions)
- ✅ Phase 2: Data layer (MCP client, scrapers, product catalog)
- 🔄 Phase 3: Agent implementations (1/4 complete - GathererAgent ✅)

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Graph/Workflow Layer (graph/workflow.py)                    │
│  - LangGraph orchestration                                   │
│  - SQLite checkpointing                                      │
│  - Node placeholder functions                                │
├──────────────────────────────────────────────────────────────┤
│  Agent Layer (agents/) - PHASE 3 IN PROGRESS                │
│  - CoordinatorAgent (parse input, human-in-loop) - TODO     │
│  - GathererAgent (collect data from sources) - COMPLETE ✅  │
│  - IdentifierAgent (find opportunities) - TODO              │
│  - ValidatorAgent (confidence scoring) - TODO               │
├──────────────────────────────────────────────────────────────┤
│  Core Services (core/)                                       │
│  - ModelRouter: 3-tier LLM routing with caching            │
│  - BaseAgent: Abstract base with monitoring                │
│  - Exceptions: Custom hierarchy                            │
├──────────────────────────────────────────────────────────────┤
│  Data Source Layer (data_sources/)                          │
│  - DuckDuckGoMCPClient: Web search via MCP                 │
│  - JobBoardScraper: Career page detection & parsing        │
│  - ProductCatalogIndexer: ChromaDB semantic search         │
│  - WebScraper: HTML parsing utilities                      │
├──────────────────────────────────────────────────────────────┤
│  Models/Domain Layer (models/)                              │
│  - ResearchState: LangGraph state TypedDict                │
│  - Domain models: JobPosting, Opportunity, Signal, etc.    │
├──────────────────────────────────────────────────────────────┤
│  Infrastructure                                             │
│  - Config (config.py): Pydantic settings                   │
│  - Logging (utils/logging.py): structlog                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📂 Complete File Inventory

### ✅ Phase 1: Core Infrastructure (COMPLETE)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `config.py` | ~150 | Pydantic settings, env vars, model routing config | ✅ |
| `core/exceptions.py` | ~120 | Custom exception hierarchy | ✅ |
| `core/model_router.py` | ~350 | 3-tier LLM routing, caching, retries | ✅ |
| `core/base_agent.py` | ~220 | Abstract BaseAgent, StatelessAgent | ✅ |
| `utils/logging.py` | ~80 | Structured logging (structlog) | ✅ |

**Total Phase 1**: ~920 lines

---

### ✅ Phase 2: Data Layer (COMPLETE)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `models/state.py` | ~170 | ResearchState TypedDict, Signal, Opportunity | ✅ |
| `models/domain.py` | ~180 | JobPosting, CompanyInfo, Product, AgentResult | ✅ |
| `data_sources/base.py` | ~150 | Abstract DataSource, CachedDataSource | ✅ |
| `data_sources/mcp_ddg_client.py` | ~400 | DuckDuckGo MCP client with rate limiting | ✅ |
| `data_sources/scraper.py` | ~250 | Web scraping utilities (BeautifulSoup, httpx) | ✅ |
| `data_sources/job_boards.py` | ~300 | Job board scraping, career page detection | ✅ |
| `data_sources/product_catalog.py` | ~350 | ChromaDB indexing, semantic product matching | ✅ |
| `graph/workflow.py` | ~200 | LangGraph workflow with checkpointing | ✅ |

**Total Phase 2**: ~2,000 lines

---

### 🔄 Phase 3: Agents (IN PROGRESS - 1/4 complete)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `agents/__init__.py` | 6 | Package exports | ✅ Complete |
| `agents/gatherer.py` | 316 | Collect data from MCP, jobs, news in parallel | ✅ Complete |
| `agents/coordinator.py` | ~200 | Parse input, human-in-loop, clarify requirements | ⏳ TODO |
| `agents/identifier.py` | ~250 | LLM-based opportunity identification | ⏳ NEXT |
| `agents/validator.py` | ~200 | Confidence scoring, risk assessment | ⏳ TODO |

**Total Phase 3**: 316/~950 lines (33% complete)

---

## 🔑 Key Classes & Their Roles

### 1. State Management (`models/state.py`)

```python
class ResearchState(TypedDict):
    """Main workflow state - use this structure in agents"""

    # Input (from user)
    account_name: str              # ⚠️ Use this, NOT company_name
    industry: str
    region: str | None
    user_context: str | None
    research_depth: ResearchDepth  # QUICK/STANDARD/DEEP

    # Collected data (GathererAgent populates)
    signals: list[Signal]
    job_postings: list[dict]
    news_items: list[dict]
    tech_stack: list[str]
    financial_data: dict | None

    # Analysis (IdentifierAgent populates)
    opportunities: list[Opportunity]

    # Validation (ValidatorAgent populates)
    validated_opportunities: list[Opportunity]
    competitive_risks: list[str]

    # Progress tracking
    progress: ResearchProgress     # Tracks which agents completed

    # Human interaction
    human_feedback: list[str]
    waiting_for_human: bool
    human_question: str | None

    # Metadata
    started_at: datetime
    last_updated: datetime
    error_messages: list[str]
    confidence_scores: dict[str, float]

class Signal(BaseModel):
    """Individual data point with confidence"""
    source: str
    signal_type: str  # "hiring", "tech_stack", "news", etc.
    content: str
    timestamp: datetime
    confidence: float  # 0.0-1.0
    metadata: dict

class Opportunity(BaseModel):
    """Identified upsell/cross-sell opportunity"""
    product_name: str
    rationale: str
    evidence: list[Signal]
    target_persona: str | None
    talking_points: list[str]
    estimated_value: str | None
    risks: list[str]
    confidence: OpportunityConfidence  # LOW/MEDIUM/HIGH
    confidence_score: float  # 0.0-1.0

class ResearchProgress(BaseModel):
    """Track agent completion"""
    coordinator_complete: bool = False
    gatherer_complete: bool = False
    identifier_complete: bool = False
    validator_complete: bool = False
```

**Factory Function**:
```python
create_initial_state(
    account_name: str,
    industry: str,
    region: str | None = None,
    user_context: str | None = None,
    research_depth: ResearchDepth = ResearchDepth.STANDARD
) -> ResearchState
```

---

### 2. Base Agent Pattern (`core/base_agent.py`)

```python
class StatelessAgent(BaseAgent[ResearchState, ResearchState]):
    """Inherit from this for all Phase 3 agents"""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.logger = structlog.get_logger(__name__).bind(agent=name)

    @abstractmethod
    async def process(self, state: ResearchState) -> None:
        """Override this - modify state in-place"""
        pass

    @abstractmethod
    def get_complexity(self, state: ResearchState) -> int:
        """Return 1-10 for model routing"""
        pass

# Example usage in Phase 3:
class GathererAgent(StatelessAgent):
    def __init__(self, mcp_client, job_scraper):
        super().__init__(name="gatherer")
        self.mcp_client = mcp_client
        self.job_scraper = job_scraper

    async def process(self, state: ResearchState) -> None:
        # Collect data, populate state["signals"], etc.
        pass

    def get_complexity(self, state: ResearchState) -> int:
        return 3  # Simple - data collection, no LLM
```

**Built-in Features**:
- `execute_with_monitoring()` - Automatic metrics, error handling
- `get_metrics()` - execution_count, error_count, avg_time, success_rate
- Structured logging with agent context

---

### 3. Model Router (`core/model_router.py`)

```python
router = ModelRouter()

response = await router.generate(
    prompt="Analyze this job posting...",
    complexity=6,  # 1-10 scale
    system_prompt="You are a sales analyst...",
    temperature=0.7,
    max_tokens=2000,
    use_cache=True
)
```

**Routing Logic**:
- Complexity 1-3: `local_model` (llama3.2:3b via Ollama)
- Complexity 4-7: `smart_model` (groq/llama-3.1-8b-instant)
- Complexity 8-10: `advanced_model` (groq/llama-3.1-70b)

**Features**:
- Response caching (24h TTL)
- Automatic retries (3 attempts, exponential backoff)
- Latency tracking
- Token usage tracking

---

### 4. Data Sources

#### DuckDuckGo MCP Client (`data_sources/mcp_ddg_client.py`)

```python
async with DuckDuckGoMCPClient() as client:
    # Web search
    results = await client.search("Boeing careers", max_results=10)
    # -> list[SearchResult]

    # Webpage content
    content = await client.fetch_content("https://boeing.com/careers")
    # -> str (HTML content)

    # News search
    news = await client.search_news("Boeing technology", max_results=5)
    # -> list[NewsItem]

    # Company info
    company = await client.search_company_info("Boeing")
    # -> CompanyInfo

    # Metrics
    metrics = client.get_metrics()
```

**Features**:
- MCP protocol via `uvx duckduckgo-mcp-server`
- Rate limiting (2.0s between requests) to avoid bot detection
- Caching (1h TTL)
- Retry logic (3 attempts)

---

#### Job Board Scraper (`data_sources/job_boards.py`)

```python
scraper = JobBoardScraper()

# Find career page and scrape jobs
jobs = await scraper.fetch(
    company_name="Boeing",
    company_domain="boeing.com"
)
# -> list[JobPosting]

# Parse individual job posting
job = await scraper.parse_job_posting("https://boeing.com/careers/job/12345")
# -> JobPosting
```

**Features**:
- Career page auto-detection (common patterns + MCP fallback)
- Multi-format support (Greenhouse, Lever, Generic)
- Rate limiting (0.5 req/sec)
- 6-hour cache TTL

---

#### Product Catalog (`data_sources/product_catalog.py`)

```python
# Build catalog (run once)
indexer = ProductCatalogIndexer(
    company_name="MathWorks",
    catalog_file="data/mathworks_products.json"  # optional
)
products = await indexer.build_catalog()
await indexer.index_products(products)

# Semantic search
matcher = ProductMatcher(company_name="MathWorks")
matches = await matcher.match_requirements_to_products(
    requirements=[
        "real-time embedded systems",
        "automotive control algorithms"
    ],
    top_k=5
)
# -> [(product_name, confidence_score), ...]

# Explain match
explanation = await matcher.explain_match(
    requirement="real-time embedded systems",
    product_name="Simulink Real-Time"
)
```

**Features**:
- ChromaDB vector database
- Sentence-transformers embeddings (all-MiniLM-L6-v2)
- 20+ built-in MathWorks products
- Company-agnostic (supports custom JSON catalogs)
- Semantic matching (not keyword matching)

---

## 🔄 Data Flow: How the System Works

### Workflow Execution

```
1. USER INPUT
   account_name="Boeing", industry="Aerospace", research_depth=STANDARD

2. CREATE INITIAL STATE
   state = create_initial_state(account_name, industry, ...)
   # Empty lists, progress tracking initialized

3. RUN LANGGRAPH WORKFLOW
   ResearchWorkflow.run(state, thread_id="boeing_2024")

   ├─→ COORDINATOR NODE [Phase 3 TODO]
   │   - Parse inputs
   │   - Ask clarifying questions (human-in-loop)
   │   - Update state["human_feedback"]
   │   - Mark progress.coordinator_complete = True
   │
   ├─→ GATHERER NODE [Phase 3 TODO]
   │   ├─→ MCP search: "Boeing company info"
   │   │   └─→ state["signals"].append(Signal(...))
   │   ├─→ JobBoardScraper("boeing.com")
   │   │   └─→ state["job_postings"] = [...]
   │   ├─→ MCP search: "Boeing news technology"
   │   │   └─→ state["news_items"] = [...]
   │   ├─→ Extract tech stack from jobs
   │   │   └─→ state["tech_stack"] = [...]
   │   └─→ Mark progress.gatherer_complete = True
   │
   ├─→ IDENTIFIER NODE [Phase 3 TODO]
   │   ├─→ Extract requirements from jobs (LLM)
   │   ├─→ ProductMatcher.match_requirements_to_products()
   │   ├─→ Generate hypotheses (LLM via ModelRouter)
   │   ├─→ Create Opportunity objects
   │   │   └─→ state["opportunities"] = [...]
   │   └─→ Mark progress.identifier_complete = True
   │
   ├─→ VALIDATOR NODE [Phase 3 TODO]
   │   ├─→ Assess competitive risks
   │   ├─→ Score confidence (LLM)
   │   ├─→ Filter low-confidence opportunities
   │   │   └─→ state["validated_opportunities"] = [...]
   │   └─→ Mark progress.validator_complete = True
   │
   └─→ FINALIZER NODE
       └─→ Update last_updated timestamp

4. CHECKPOINT SAVED
   SQLite database: data/checkpoints/<thread_id>.db

5. RETURN RESULTS
   Final state with all fields populated
```

---

## 🎯 Phase 3 Implementation Guide

### Agent 1: CoordinatorAgent (⏳ TODO)

**Purpose**: Parse inputs, clarify requirements, human interaction

**Key Methods**:
```python
class CoordinatorAgent(StatelessAgent):
    async def process(self, state: ResearchState) -> None:
        # 1. Validate inputs (account_name, industry)
        # 2. Generate clarifying questions if needed
        # 3. Set state["waiting_for_human"] = True if questions exist
        # 4. Store questions in state["human_question"]
        # 5. Mark progress.coordinator_complete = True
```

**Complexity**: 3-4 (simple validation, minimal LLM usage)

---

### Agent 2: GathererAgent (🔄 NEEDS ENHANCEMENT)

**Purpose**: Research Analyst - Collect, analyze, and score intelligence from multiple sources

**Status**: Basic implementation complete (316 lines) - Requires enhancement for LLM-based analysis

**Git**: [2f71469] "Added intelligence gatherer agent"

**Dependencies**:
- `DuckDuckGoMCPClient` (web search + content fetching)
- `JobBoardScraper` (job postings)
- `ModelRouter` (⚠️ MISSING - needed for LLM analysis)

---

#### **🔑 Critical Architecture Understanding**

**GathererAgent is NOT a simple data collector - it's a Research Analyst that:**

1. **Receives Rich Context from CoordinatorAgent**:
   - `account_name`, `industry`, `region`, `user_context`, `research_depth`
   - Uses ALL fields to build intelligent, context-aware search queries
   - Example: "Acme Corp enterprise software real-time data processing North America"

2. **Fetches AND Analyzes Each Source Individually**:
   - NOT just snippets - fetches full webpage content with `fetch_content()`
   - Uses LOCAL LLM (Tier 1 Ollama via ModelRouter, complexity=4) to analyze each source
   - Assigns per-source confidence based on authority + content quality + relevance

3. **Creates Rich Signal Objects with LLM-Generated Metadata**:
   ```python
   Signal(
       source="duckduckgo",
       signal_type="web_search",
       content="[LLM-generated summary, not raw snippet]",  # ← Analyzed summary
       confidence=0.87,  # ← LLM-assigned based on source quality
       metadata={
           "url": "https://acme.com/press/...",
           "title": "Acme Announces New Platform",
           "source_type": "official_company_site",  # ← LLM determines
           "key_facts": ["Uses Kafka", "Targets enterprises"],  # ← LLM extracts
           "keywords": ["real-time", "data-processing"],  # ← LLM identifies
           "relevance": "high",  # ← LLM rates
           "date_published": "2024-01-10"  # ← Extracted if available
       }
   )
   ```

4. **Respects research_depth Parameter**:
   - `ResearchDepth.QUICK`: Analyze top 5 results per source
   - `ResearchDepth.STANDARD`: Analyze top 10 results per source
   - `ResearchDepth.DEEP`: Analyze top 20 results per source

---

#### **⚠️ Current vs. Required Implementation**

**Current Implementation (INCOMPLETE)**:
```python
# ❌ Uses hardcoded confidence values
Signal(confidence=0.8)  # Same for all DuckDuckGo results

# ❌ Uses raw snippets, not analyzed content
content=result.snippet  # Not summarized by LLM

# ❌ No LLM analysis per source
# ❌ Minimal metadata (only url, title)
# ❌ Doesn't use all context fields
# ❌ No ModelRouter dependency
```

**Required Implementation (TODO)**:
```python
class GathererAgent(StatelessAgent):
    def __init__(
        self,
        mcp_client: DuckDuckGoMCPClient,
        job_scraper: JobBoardScraper,
        model_router: ModelRouter  # ← ADD THIS
    ):
        super().__init__(name="gatherer")
        self.mcp_client = mcp_client
        self.job_scraper = job_scraper
        self.model_router = model_router  # ← ADD THIS
        self._analysis_cache = {}  # Cache LLM analyses by URL

    async def process(self, state: ResearchState) -> None:
        # Extract ALL context fields
        account = state["account_name"]
        industry = state["industry"]
        region = state.get("region", "")
        user_context = state.get("user_context", "")
        depth = state["research_depth"]

        # Build rich, context-aware query
        query = self._build_query(account, industry, region, user_context)

        # Determine max_results based on depth
        max_results = {
            ResearchDepth.QUICK: 5,
            ResearchDepth.STANDARD: 10,
            ResearchDepth.DEEP: 20
        }[depth]

        # Fetch raw results
        raw_results = await self.mcp_client.search(query, max_results=max_results)

        # Analyze EACH result individually with LLM
        for result in raw_results:
            # Step 1: Fetch full webpage content
            full_content = await self.mcp_client.fetch_content(result.url)

            # Step 2: Analyze with LOCAL LLM (Tier 1 Ollama via ModelRouter)
            analyzed_signal = await self._analyze_source_with_llm(
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                full_content=full_content,
                account_name=account,
                industry=industry
            )

            # Step 3: Add analyzed Signal to state
            state["signals"].append(analyzed_signal)

        # Mark complete
        state["progress"].gatherer_complete = True

    async def _analyze_source_with_llm(
        self,
        url: str,
        title: str,
        snippet: str,
        full_content: str,
        account_name: str,
        industry: str
    ) -> Signal:
        """
        Use LLM to analyze source and assign confidence.

        LLM analyzes:
        - Source authority (official site vs blog vs news)
        - Content relevance to account research
        - Factual quality (facts vs speculation)
        - Extracts key facts, keywords
        - Assigns confidence score (0.0-1.0)
        """

        # Check cache first
        cache_key = hash(url)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        prompt = f"""Analyze this source about {account_name} ({industry}):

URL: {url}
Title: {title}
Snippet: {snippet}
Content (first 2000 chars): {full_content[:2000]}

Tasks:
1. Assess source reliability (official/news/blog)
2. Rate relevance to {account_name} research (high/medium/low)
3. Identify key facts (not speculation)
4. Extract keywords and technologies
5. Assign confidence (0.0-1.0):
   - 0.9-1.0: Official company source with facts
   - 0.7-0.8: Reputable news/industry source
   - 0.5-0.6: Blog with citations
   - 0.0-0.4: Unreliable/irrelevant
6. Summarize key information (2-3 sentences)

Return JSON:
{{
    "confidence": 0.85,
    "summary": "...",
    "source_type": "official_company_site",
    "key_facts": ["fact1", "fact2"],
    "keywords": ["keyword1", "keyword2"],
    "relevance": "high"
}}"""

        # Use ModelRouter with complexity=4 (routes to Tier 1 Ollama)
        response = await self.model_router.generate(
            prompt=prompt,
            complexity=4,  # Simple classification/summarization
            use_cache=True
        )

        # Parse LLM JSON response
        analysis = json.loads(response.content)

        # Create Signal with LLM-analyzed data
        signal = Signal(
            source="duckduckgo",
            signal_type="web_search",
            content=analysis["summary"],  # LLM summary
            confidence=analysis["confidence"],  # LLM-assigned
            metadata={
                "url": url,
                "title": title,
                "source_type": analysis["source_type"],
                "key_facts": analysis["key_facts"],
                "keywords": analysis["keywords"],
                "relevance": analysis["relevance"],
                "original_snippet": snippet
            }
        )

        # Cache for future lookups
        self._analysis_cache[cache_key] = signal
        return signal

    def get_complexity(self, state: ResearchState) -> int:
        return 4  # ← UPDATE from 3: Now uses LLM analysis
```

---

#### **🎯 Two Types of Confidence (DO NOT CONFUSE)**

**1. Signal.confidence (0.0-1.0 float)**:
- **Purpose**: Measures signal reliability (source trustworthiness)
- **Assigned by**: GathererAgent using LLM analysis
- **Example**: Official company site = 0.9, blog = 0.5
- **Field**: `Signal.confidence`

**2. Opportunity.confidence_score (0.0-1.0 float) + OpportunityConfidence (enum)**:
- **Purpose**: Measures opportunity validity (likelihood customer needs product)
- **Assigned by**: ValidatorAgent using LLM evaluation
- **Example**: Strong evidence = HIGH (0.85), weak = LOW (0.35)
- **Fields**: `Opportunity.confidence` (enum), `Opportunity.confidence_score` (float)

**⚠️ GathererAgent assigns Signal confidence, NOT Opportunity confidence**

---

#### **📦 Key Features Implemented**:
- ✅ Parallel data collection with `asyncio.gather(return_exceptions=True)`
- ✅ Creates immutable Signal objects for all data
- ✅ Graceful failure handling (continues if 1-2 sources fail)
- ✅ Tech stack extraction from job postings
- ✅ Supports optional `company_domain` in state
- ✅ 16 comprehensive tests (100% pass rate)

#### **⏳ Features Needed (Enhancement)**:
- ⏳ Add `ModelRouter` dependency
- ⏳ Fetch full content with `fetch_content()` for each result
- ⏳ LLM analysis per source (Tier 1 Ollama, complexity=4)
- ⏳ Rich metadata generation (source_type, key_facts, keywords)
- ⏳ Use ALL context fields in queries
- ⏳ Respect `research_depth` parameter (5/10/20 results)
- ⏳ Analysis caching by URL hash
- ⏳ Update tests to mock ModelRouter responses

---

### Agent 3: IdentifierAgent (⏳ TODO)

**Purpose**: Identify opportunities using LLM and semantic search

**Dependencies**:
- `ProductMatcher` (semantic product search)
- `ModelRouter` (LLM reasoning)

**Key Methods**:
```python
class IdentifierAgent(StatelessAgent):
    def __init__(self, product_matcher: ProductMatcher, model_router: ModelRouter):
        super().__init__(name="identifier")
        self.product_matcher = product_matcher
        self.model_router = model_router

    async def process(self, state: ResearchState) -> None:
        # 1. Extract requirements from job postings (LLM)
        requirements = await self._extract_requirements(
            state["job_postings"],
            state["tech_stack"]
        )

        # 2. Match to products (semantic search)
        product_matches = await self.product_matcher.match_requirements_to_products(
            requirements=requirements,
            top_k=10
        )

        # 3. Generate opportunity hypotheses (LLM)
        for product_name, confidence in product_matches:
            opportunity = await self._generate_opportunity(
                product_name=product_name,
                requirements=requirements,
                signals=state["signals"],
                confidence=confidence
            )
            state["opportunities"].append(opportunity)

        # Mark complete
        state["progress"].identifier_complete = True

    async def _extract_requirements(self, jobs, tech_stack) -> list[str]:
        # Use ModelRouter with complexity 6-7
        prompt = f"Extract technical requirements from these jobs:\n{jobs}"
        response = await self.model_router.generate(prompt, complexity=6)
        # Parse response into list of requirements
        return requirements

    async def _generate_opportunity(self, ...) -> Opportunity:
        # Use ModelRouter with complexity 7-8
        # Generate rationale, talking points, risks
        return Opportunity(...)
```

**Complexity**: 7-8 (complex LLM reasoning, multiple calls)

---

### Agent 4: ValidatorAgent (⏳ TODO)

**Purpose**: Validate opportunities, assess risks, filter by confidence

**Dependencies**:
- `ModelRouter` (LLM validation)

**Key Methods**:
```python
class ValidatorAgent(StatelessAgent):
    def __init__(self, model_router: ModelRouter):
        super().__init__(name="validator")
        self.model_router = model_router

    async def process(self, state: ResearchState) -> None:
        # 1. Assess competitive risks (LLM)
        risks = await self._assess_competitive_risks(
            state["opportunities"],
            state["industry"]
        )
        state["competitive_risks"] = risks

        # 2. Validate each opportunity
        for opp in state["opportunities"]:
            # Score confidence based on evidence strength
            validated_confidence = await self._validate_opportunity(opp)

            # Filter: only keep confidence > 0.6
            if validated_confidence > 0.6:
                opp.confidence_score = validated_confidence
                state["validated_opportunities"].append(opp)

        # Mark complete
        state["progress"].validator_complete = True

    async def _validate_opportunity(self, opp: Opportunity) -> float:
        # Use ModelRouter complexity 6
        # Analyze evidence strength, recency, relevance
        return confidence_score
```

**Complexity**: 6-7 (LLM validation, structured reasoning)

---

## 🧪 Testing Strategy

### Unit Tests (Mock everything)

```python
# tests/test_agents/test_gatherer.py
@pytest.mark.asyncio
async def test_gatherer_successful_collection():
    # Mock MCP client
    mock_mcp = AsyncMock()
    mock_mcp.search.return_value = [SearchResult(...)]

    # Mock job scraper
    mock_scraper = AsyncMock()
    mock_scraper.fetch.return_value = [JobPosting(...)]

    # Create agent
    agent = GathererAgent(mcp_client=mock_mcp, job_scraper=mock_scraper)

    # Create state
    state = create_initial_state(account_name="Boeing", industry="Aerospace")

    # Execute
    await agent.process(state)

    # Verify
    assert len(state["signals"]) > 0
    assert len(state["job_postings"]) > 0
    assert state["progress"].gatherer_complete == True
```

---

## 📊 Current Status Summary

| Component | Status | Lines | Tests | Coverage |
|-----------|--------|-------|-------|----------|
| **Core Infrastructure** | ✅ Complete | ~920 | ✅ Passing | >80% |
| **Data Layer** | ✅ Complete | ~2,000 | ✅ 77/77 | >85% |
| **Agent Framework** | ✅ Complete | ~220 | ✅ Passing | >80% |
| **GathererAgent** | ✅ Complete | 316 | ✅ 16/16 | >90% |
| **CoordinatorAgent** | ⏳ TODO | ~200 | ⏳ TODO | - |
| **IdentifierAgent** | ⏳ NEXT | ~250 | ⏳ TODO | - |
| **ValidatorAgent** | ⏳ TODO | ~200 | ⏳ TODO | - |
| **Integration Tests** | ⏳ TODO | ~150 | ⏳ TODO | - |
| **E2E Tests** | ⏳ TODO | ~100 | ⏳ TODO | - |

**Total Implemented**: ~3,456 lines
**Total Remaining**: ~900 lines
**Progress**: 79% complete (Phase 3: 1/4 agents done)

---

## 🚀 Phase 3 Progress Update

**Infrastructure Ready**:
- ✅ State models defined
- ✅ Base agent pattern established
- ✅ Data sources implemented and tested
- ✅ Model router with caching
- ✅ LangGraph workflow structure
- ✅ Testing framework ready

**Agents Completed** (2026-01-15):
1. ✅ **GathererAgent** - Parallel data collection
   - Production: 316 lines
   - Tests: 559 lines (16 tests, 100% pass)
   - Git: [2f71469]

**Next**: IdentifierAgent (LLM-based opportunity identification)

---

**END OF ARCHITECTURE DOCUMENT**
