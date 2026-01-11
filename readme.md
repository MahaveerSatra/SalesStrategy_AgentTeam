# Enterprise Account Research System

Multi-agent AI system using LangGraph to identify upsell/cross-sell opportunities for enterprise accounts.

**Status:** Phase 1 Complete ✅

## Phase 1: Foundation

### What's Built

✅ **Project Structure**
- Clean, scalable directory layout
- Separation of concerns (models, core, agents, data sources, graph)
- Production-ready from day one

✅ **Configuration Management**
- Pydantic Settings with environment variable support
- Model routing thresholds configurable
- Caching, logging, performance settings

✅ **State Models**
- Type-safe state with Pydantic
- ResearchState for LangGraph workflow
- Domain models (JobPosting, Opportunity, etc.)
- Progress tracking

✅ **Base Abstractions**
- `BaseAgent` abstract class for all agents
- Built-in monitoring and error handling
- Standardized execution interface
- Performance metrics

✅ **Intelligent Model Router**
- Multi-tier architecture (local + external APIs)
- Automatic routing based on complexity
- Response caching with TTL
- Retry logic with exponential backoff
- Comprehensive metrics

✅ **LangGraph Workflow**
- Basic workflow structure with 4 agents
- SQLite checkpointing for resumability
- State management
- Placeholder nodes (ready for implementation)

✅ **Production Features**
- Structured logging (JSON for production, console for dev)
- Custom exceptions for different failure modes
- Comprehensive error handling
- Performance tracking

**Technology Stack

**Core:**
- Python 3.11+
- LangGraph (multi-agent orchestration)
- Pydantic v2 (data validation)
- SQLite (checkpointing)

**LLMs:**
- Ollama (local): llama3.2:3b, phi3:3.8b
- Groq API (external): llama-3.1-8b-instant, llama-3.1-70b
- LiteLLM (model abstraction)

**Data Sources:**
- MCP (Model Context Protocol) - DuckDuckGo web search
- Direct web scraping (BeautifulSoup)
- Job boards (company career pages, public APIs)

**Utilities:**
- structlog (structured logging)
- tenacity (retry logic)
- pytest (testing)

## Quick Start (Windows)

### 1. Prerequisites

```powershell
# Ensure you have:
- Python 3.11+ (https://www.python.org/downloads/)
- Ollama installed (https://ollama.ai/download)
- GTX 1050Ti or better GPU (4GB+ VRAM)
- 8GB+ RAM available
- VS Code (recommended)
```

### 2. Get API Keys

**Groq API (FREE):**
1. Go to https://console.groq.com
2. Sign up (free)
3. Create API key
4. Add to `.env` file: `GROQ_API_KEY=gsk_...`

### 3. Test Phase 1

```powershell
# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# Run the test suite
python main.py

# You should see:
# - Model router tests (local + external models)
# - Workflow execution (placeholder agents)
# - Checkpoint/resume functionality

# Run unit tests
pytest tests\test_model_router\ -v
```

### 4. Verify

✅ **Check Ollama is working:**
```powershell
ollama list
# Should show llama3.2:3b and phi3:3.8b
```

✅ **Check checkpoint database created:**
```powershell
Get-ChildItem data\checkpoints\
# Should see checkpoints.db
```

✅ **Check logs are structured:**
- JSON format if LOG_FORMAT=json
- Console format if LOG_FORMAT=console

## Architecture Overview

### Model Routing Strategy

```
Task Complexity → Model Selection
─────────────────────────────────
1-3 (Simple)    → llama3.2:3b (local GPU, <1s, free)
4-7 (Medium)    → groq/llama-3.1-8b-instant (API, 2-3s, free)
8-10 (Complex)  → groq/llama-3.1-70b (API, 5s, free tier)
```

**Why this works:**
- 80% of tasks use local GPU (fast, free)
- 20% use external APIs for complex reasoning
- Total cost: $0 (within free tiers)
- Excellent output quality

### Agent Architecture (To Be Implemented)

```
┌─────────────────────┐
│  Research           │
│  Coordinator        │  ← Human-in-the-loop
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───┐    ┌───▼────┐
│ Intel │    │ Oppty  │
│ Gather│    │ Ident  │
└───┬───┘    └───┬────┘
    │            │
    └──────┬─────┘
           │
    ┌──────▼──────┐
    │  Strategy   │
    │  Validator  │
    └─────────────┘
```

### State Flow

```python
ResearchState = {
    # Input
    "account_name": str,
    "industry": str,
    "research_depth": ResearchDepth,
    
    # Collected Data (Gatherer)
    "signals": List[Signal],
    "job_postings": List[JobPosting],
    "news_items": List[NewsItem],
    
    # Analysis (Identifier)
    "opportunities": List[Opportunity],
    
    # Validation (Validator)
    "validated_opportunities": List[Opportunity],
    "competitive_risks": List[str],
    
    # Progress & Metadata
    "progress": ResearchProgress,
    "confidence_scores": Dict[str, float]
}
```

## Configuration

Key settings in `.env`:

```bash
# API Keys
GROQ_API_KEY=gsk_...  # Free from console.groq.com

# Model Selection
LOCAL_MODEL=llama3.2:3b
SMART_MODEL=groq/llama-3.1-8b-instant
ADVANCED_MODEL=groq/llama-3.1-70b

# Performance
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
MAX_PARALLEL_TASKS=5

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # or 'console' for development
```

## Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_model_router/test_router.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Test model router manually
python main.py
```

## Next Steps: Phase 2 - Data Layer

**Goals:**
1. Implement data source abstractions
2. Build web scrapers for company data
3. Integrate DuckDuckGo search
4. Job posting collection
5. Product catalog indexing
6. Caching layer

## Troubleshooting (Windows)

**PowerShell Execution Policy Error:**
```powershell
# Error: "cannot be loaded because running scripts is disabled"
# Fix:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Module Not Found Error:**
```powershell
# Error: "ModuleNotFoundError: No module named 'src'"
# Fix: Make sure main.py is in the project root, not in src/
# Correct location: account-research-system\main.py
# Verify with:
Get-Location  # Should show: ...\account-research-system
Get-ChildItem main.py  # Should exist

# If main.py is in src/, move it to root:
Move-Item src\main.py main.py
```

**Ollama Not Connecting:**
```powershell
# Check Ollama is running
ollama list

# If error, start Ollama service
# Open a new PowerShell window and run:
ollama serve

# Then try again in original window
```

**Import Errors After Installation:**
```powershell
# Ensure you're in virtual environment (should see (venv) in prompt)
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
## Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **Groq Console:** https://console.groq.com
- **Ollama Models:** https://ollama.ai/library
- **Pydantic Docs:** https://docs.pydantic.dev/

## License

MIT License - This is a portfolio/demonstration project.

---

**Built by:** Mahaveer Satra
**For:** Sales Strategy / Business Development / Multi-agent AI enthusiasts
**Contact:** [LinkedIn](https://www.linkedin.com/in/mahaveer-satra/)
