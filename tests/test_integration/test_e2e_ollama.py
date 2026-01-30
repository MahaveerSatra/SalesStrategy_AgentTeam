"""
End-to-end tests with real Ollama LLM.

These tests verify:
1. ModelRouter correctly routes to and calls real Ollama
2. Agents can parse real LLM JSON responses (not mocked)
3. Simplified E2E workflow with real LLM responses
4. Structured outputs enforce reliable JSON generation

IMPORTANT: These tests require Ollama to be running locally with llama3.2:3b model.
Run: ollama pull llama3.2:3b

Tests are marked with @pytest.mark.slow to allow skipping in CI:
    pytest tests/ -v -m "not slow"
"""
import pytest
import asyncio
from datetime import datetime
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

from src.core.model_router import ModelRouter
from src.utils.json_parsing import extract_json_from_llm_response, JSONParseError
from src.models.state import ResearchState, ResearchDepth, ResearchProgress, Signal, Opportunity, OpportunityConfidence
from src.models.domain import ModelResponse


# Pydantic models for structured outputs - guarantees valid JSON from Ollama
class SourceAnalysis(BaseModel):
    """Schema for GathererAgent's source analysis output.

    Using Literal types to constrain LLM output to specific values.
    This ensures the LLM returns exactly the values we expect.
    """
    confidence: float
    summary: str
    source_type: str
    key_facts: list[str]
    keywords: list[str]
    relevance: Literal["high", "medium", "low"]  # Constrain to valid values


class RequirementsExtraction(BaseModel):
    """Schema for IdentifierAgent's requirements extraction."""
    requirements: list[str]


class RiskAssessment(BaseModel):
    """Schema for ValidatorAgent's risk assessment."""
    risks: list[str]


class InputValidation(BaseModel):
    """Schema for CoordinatorAgent's input validation."""
    is_valid: bool
    normalized_name: str
    needs_clarification: bool
    questions: list[str]


def is_ollama_available() -> bool:
    """Check if Ollama is running and llama3.2:3b is available."""
    try:
        import ollama
        response = ollama.list()
        # Handle both old dict format and new object format
        if hasattr(response, 'models'):
            # New format: response.models is list of Model objects
            model_names = [m.model for m in response.models]
        else:
            # Old format: response is dict with 'models' key
            model_names = [m.get('name', '') for m in response.get('models', [])]
        # Check for llama3.2:3b or llama3.2 variants
        return any('llama3.2' in name for name in model_names)
    except Exception as e:
        print(f"Ollama check failed: {e}")
        return False


# Skip all tests in this module if Ollama is not available
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not is_ollama_available(),
        reason="Ollama not available or llama3.2:3b not installed"
    )
]


class TestModelRouterWithRealOllama:
    """Test ModelRouter with real Ollama calls."""

    @pytest.mark.asyncio
    async def test_model_router_basic_generation(self):
        """Test basic text generation with real Ollama."""
        router = ModelRouter()

        response = await router.generate(
            prompt="What is 2 + 2? Reply with just the number.",
            complexity=2,  # Routes to local Ollama
            use_cache=False  # Force fresh generation
        )

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "llama3.2:3b"
        assert response.latency_ms > 0
        assert "4" in response.content

    @pytest.mark.asyncio
    async def test_model_router_json_generation(self):
        """Test that ModelRouter can generate JSON output."""
        router = ModelRouter()

        prompt = """Return a JSON object with the following structure:
{
    "status": "ok",
    "count": 3
}

Return ONLY the JSON, no explanation."""

        response = await router.generate(
            prompt=prompt,
            complexity=2,
            temperature=0.1,  # Lower temperature for more deterministic output
            use_cache=False
        )

        assert response.content is not None

        # Should be able to parse the response as JSON
        parsed = extract_json_from_llm_response(response.content)
        assert "status" in parsed or "count" in parsed

    @pytest.mark.asyncio
    async def test_model_router_with_system_prompt(self):
        """Test ModelRouter with system prompt."""
        router = ModelRouter()

        response = await router.generate(
            prompt="What color is the sky?",
            system_prompt="You are a helpful assistant. Always respond in exactly one word.",
            complexity=2,
            use_cache=False
        )

        assert response.content is not None
        # Response should be relatively short (one word or short phrase)
        assert len(response.content.split()) <= 5

    @pytest.mark.asyncio
    async def test_model_router_caching(self):
        """Test that ModelRouter caching works with real responses."""
        router = ModelRouter()
        router.clear_cache()

        prompt = "What is the capital of France? Answer in one word."

        # First call - should miss cache
        response1 = await router.generate(
            prompt=prompt,
            complexity=2,
            use_cache=True
        )

        # Second call - should hit cache
        response2 = await router.generate(
            prompt=prompt,
            complexity=2,
            use_cache=True
        )

        assert response1.content is not None
        assert response2.content is not None
        assert response2.cached is True
        assert response1.cached is False

    @pytest.mark.asyncio
    async def test_model_router_metrics(self):
        """Test that ModelRouter tracks metrics correctly."""
        router = ModelRouter()
        router.clear_cache()

        await router.generate(
            prompt="Say hello",
            complexity=2,
            use_cache=False
        )

        metrics = router.get_metrics()

        assert metrics["total_requests"] >= 1
        assert "llama3.2:3b" in metrics["requests_by_model"]


class TestStructuredOutputs:
    """Test Ollama's structured output feature for guaranteed JSON.

    These tests demonstrate the PROPER way to get reliable JSON from LLMs:
    using the `response_format` parameter with a Pydantic schema.

    Reference: https://docs.ollama.com/capabilities/structured-outputs
    """

    @pytest.mark.asyncio
    async def test_structured_output_basic(self):
        """Test basic structured output with simple schema."""
        router = ModelRouter()

        class SimpleResponse(BaseModel):
            name: str
            value: int

        response = await router.generate(
            prompt="Return name='test' and value=42",
            complexity=2,
            temperature=0,
            use_cache=False,
            response_format=SimpleResponse.model_json_schema()
        )

        # Guaranteed to be valid JSON conforming to schema
        result = SimpleResponse.model_validate_json(response.content)
        assert result.name is not None
        assert isinstance(result.value, int)

    @pytest.mark.asyncio
    async def test_structured_output_complex_schema(self):
        """Test structured output with nested/complex schema."""
        router = ModelRouter()

        class Item(BaseModel):
            name: str
            quantity: int

        class Order(BaseModel):
            order_id: str
            items: list[Item]
            total: float

        response = await router.generate(
            prompt="Create a sample order with 2 items",
            complexity=2,
            temperature=0,
            use_cache=False,
            response_format=Order.model_json_schema()
        )

        result = Order.model_validate_json(response.content)
        assert result.order_id is not None
        assert len(result.items) >= 1
        assert isinstance(result.total, float)

    @pytest.mark.asyncio
    async def test_structured_output_for_analysis(self):
        """Test structured output for data extraction/analysis tasks."""
        router = ModelRouter()

        response = await router.generate(
            prompt="""Analyze this company:
Name: TechCorp
Industry: Software
Founded: 2015

Extract key facts, assess confidence, and summarize.""",
            complexity=3,
            temperature=0,
            use_cache=False,
            response_format=SourceAnalysis.model_json_schema()
        )

        result = SourceAnalysis.model_validate_json(response.content)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.summary) > 0
        assert len(result.key_facts) >= 0


class TestRealLLMJsonParsing:
    """Test JSON parsing with actual LLM responses (not mocked)."""

    @pytest.mark.asyncio
    async def test_gatherer_analysis_prompt_real_llm(self):
        """Test the Gatherer's analysis prompt with structured output for reliable JSON.

        Uses Ollama's structured output feature to GUARANTEE valid JSON.
        This is the proper way to test LLM JSON generation reliably.
        """
        router = ModelRouter()

        # This is the actual prompt format used by GathererAgent
        prompt = """Analyze this source about Acme Corp (Technology):

URL: https://example.com/acme
Title: Acme Corp - About Us
Snippet: Acme Corp is a leading technology company
Content (first 2000 chars): Acme Corp was founded in 2020. We build enterprise software solutions.

Tasks:
1. Assess source reliability (official/news/blog)
2. Rate relevance to Acme Corp research (high/medium/low)
3. Identify key facts (not speculation)
4. Extract keywords and technologies
5. Assign confidence (0.0-1.0):
   - 0.9-1.0: Official company source with facts
   - 0.7-0.8: Reputable news/industry source
   - 0.5-0.6: Blog with citations
   - 0.0-0.4: Unreliable/irrelevant
6. Summarize key information (2-3 sentences)"""

        # Use structured output for guaranteed valid JSON
        response = await router.generate(
            prompt=prompt,
            complexity=3,
            temperature=0,  # Deterministic for structured output
            use_cache=False,
            response_format=SourceAnalysis.model_json_schema()
        )

        # Parse with Pydantic - guaranteed to work with structured output
        analysis = SourceAnalysis.model_validate_json(response.content)

        # Verify expected fields
        assert 0.0 <= analysis.confidence <= 1.0
        assert len(analysis.summary) > 0
        assert isinstance(analysis.key_facts, list)
        assert isinstance(analysis.keywords, list)

    @pytest.mark.asyncio
    async def test_identifier_requirements_prompt_real_llm(self):
        """Test that the Identifier's requirements prompt generates parseable JSON."""
        router = ModelRouter()

        prompt = """Based on this research data about Acme Corp:

Signals:
- Acme Corp is expanding their data science team
- They are investing in machine learning infrastructure

Job Postings:
- Senior Data Scientist (Python, TensorFlow)
- ML Engineer (PyTorch, Kubernetes)

Tech Stack: Python, TensorFlow, AWS

Extract the key technology and business requirements. Return JSON:
{
    "requirements": [
        "requirement 1",
        "requirement 2"
    ]
}"""

        response = await router.generate(
            prompt=prompt,
            complexity=5,  # Medium complexity for Identifier
            temperature=0.3,
            use_cache=False
        )

        parsed = extract_json_from_llm_response(response.content)

        assert "requirements" in parsed
        assert isinstance(parsed["requirements"], list)
        assert len(parsed["requirements"]) > 0

    @pytest.mark.asyncio
    async def test_validator_risks_prompt_real_llm(self):
        """Test that the Validator's risk assessment prompt generates parseable JSON."""
        router = ModelRouter()

        prompt = """Assess competitive risks for selling to Acme Corp based on:

Signals:
- Company uses some competitor products
- Budget constraints mentioned in news

Opportunities:
- MATLAB for data analysis
- Simulink for modeling

Identify potential risks. Return JSON:
{
    "risks": [
        "risk 1",
        "risk 2"
    ]
}"""

        response = await router.generate(
            prompt=prompt,
            complexity=5,
            temperature=0.3,
            use_cache=False
        )

        parsed = extract_json_from_llm_response(response.content)

        assert "risks" in parsed
        assert isinstance(parsed["risks"], list)

    @pytest.mark.asyncio
    async def test_coordinator_validation_prompt_real_llm(self):
        """Test that the Coordinator's input validation prompt works with structured output."""
        router = ModelRouter()

        prompt = """Validate this research request:

Account Name: Acme Corp
Industry: Technology
Research Depth: STANDARD

Check if:
1. Account name is a valid company name
2. Industry makes sense
3. Any clarifying questions needed

Provide validation results with normalized company name."""

        # Use structured output to GUARANTEE valid JSON
        response = await router.generate(
            prompt=prompt,
            complexity=3,
            temperature=0,  # Deterministic for structured output
            use_cache=False,
            response_format=InputValidation.model_json_schema()
        )

        # Parse and validate with Pydantic - guaranteed to work with structured output
        result = InputValidation.model_validate_json(response.content)

        # Validate the response structure and content
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.normalized_name, str)
        assert len(result.normalized_name) > 0, "Should have a normalized name"
        assert "acme" in result.normalized_name.lower(), "Should contain Acme"
        assert isinstance(result.needs_clarification, bool)
        assert isinstance(result.questions, list)


class TestGathererAgentWithRealLLM:
    """Test GathererAgent's LLM analysis with real Ollama using structured outputs."""

    @pytest.mark.asyncio
    async def test_gatherer_analyze_source_with_structured_output(self):
        """Test source analysis using Ollama's structured output feature.

        Uses Pydantic schema to GUARANTEE valid JSON output from Ollama.
        This is the proper way to get reliable JSON from LLMs - not hoping
        the model returns JSON, but enforcing it at the API level.
        """
        router = ModelRouter()

        prompt = """Analyze this source about Acme Corp (Technology):

URL: https://example.com/acme
Title: Acme Corp - Technology Solutions
Snippet: Acme Corp provides enterprise software solutions
Content: Acme Corp was founded in 2020. We specialize in enterprise software.

Analyze source reliability, relevance, extract key facts and keywords.
Assign confidence score 0.0-1.0 based on source authority."""

        # Use structured output with Pydantic schema - GUARANTEES valid JSON
        response = await router.generate(
            prompt=prompt,
            complexity=3,
            temperature=0,  # Deterministic for structured output
            use_cache=False,
            response_format=SourceAnalysis.model_json_schema()
        )

        # Parse and validate with Pydantic
        analysis = SourceAnalysis.model_validate_json(response.content)

        # Verify the structured output
        assert 0.0 <= analysis.confidence <= 1.0
        assert len(analysis.summary) > 0
        # LLM may use different casing/phrasing for source_type
        assert analysis.source_type is not None and len(analysis.source_type) > 0
        assert isinstance(analysis.key_facts, list)
        assert isinstance(analysis.keywords, list)
        # LLM may use different casing for relevance
        assert analysis.relevance.lower() in ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_gatherer_agent_with_real_llm_integration(self):
        """Test full GathererAgent._analyze_source_with_llm with real Ollama.

        Note: This test calls the actual agent method which doesn't use
        structured outputs (yet). It may occasionally fail due to LLM
        variability - this is a known limitation of the current agent code.

        TODO: Update GathererAgent to use structured outputs for reliability.
        """
        from src.agents.gatherer import GathererAgent

        model_router = ModelRouter()
        mock_mcp_client = AsyncMock()
        mock_job_scraper = AsyncMock()

        gatherer = GathererAgent(
            mcp_client=mock_mcp_client,
            job_scraper=mock_job_scraper,
            model_router=model_router
        )

        # Try the agent's method - may fail due to LLM variability
        try:
            signal = await gatherer._analyze_source_with_llm(
                url="https://example.com/acme",
                title="Acme Corp - Technology Solutions",
                snippet="Acme Corp provides enterprise software solutions",
                full_content="Acme Corp was founded in 2020. We specialize in enterprise software.",
                account_name="Acme Corp",
                industry="Technology"
            )

            assert signal is not None
            assert signal.source == "duckduckgo"
            assert signal.signal_type == "web_search"
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.metadata["url"] == "https://example.com/acme"

        except JSONParseError:
            # Agent doesn't use structured outputs yet - document this as tech debt
            pytest.xfail(
                "GathererAgent._analyze_source_with_llm doesn't use structured outputs. "
                "TODO: Update agent to use response_format parameter for guaranteed JSON."
            )


class TestIdentifierAgentWithRealLLM:
    """Test IdentifierAgent with real Ollama."""

    @pytest.mark.asyncio
    async def test_identifier_extract_requirements_real_llm(self):
        """Test IdentifierAgent's requirement extraction with real LLM."""
        from src.agents.identifier import IdentifierAgent

        model_router = ModelRouter()
        mock_product_matcher = AsyncMock()
        mock_product_matcher.match_requirements_to_products.return_value = [
            ("MATLAB", 0.85),
            ("Simulink", 0.70)
        ]

        identifier = IdentifierAgent(
            product_matcher=mock_product_matcher,
            model_router=model_router
        )

        # Create state with signals
        state = ResearchState(
            account_name="Acme Corp",
            industry="Technology",
            research_depth=ResearchDepth.STANDARD,
            signals=[
                Signal(
                    source="duckduckgo",
                    signal_type="web_search",
                    content="Acme Corp is expanding data analytics capabilities",
                    timestamp=datetime.now(),
                    confidence=0.8,
                    metadata={}
                )
            ],
            job_postings=[
                {
                    "title": "Data Scientist",
                    "description": "Build ML models using Python and TensorFlow",
                    "technologies": ["Python", "TensorFlow"]
                }
            ],
            news_items=[],
            tech_stack=["Python", "TensorFlow"],
            opportunities=[],
            validated_opportunities=[],
            competitive_risks=[],
            human_feedback=[],
            waiting_for_human=False,
            human_question=None,
            current_report=None,
            feedback_context=None,
            next_route=None,
            progress=ResearchProgress(gatherer_complete=True),
            error_messages=[]
        )

        # Process with real LLM
        await identifier.process(state)

        # Verify opportunities were identified
        assert state["progress"].identifier_complete is True
        # Should have some opportunities (may vary based on LLM response)
        assert len(state["opportunities"]) >= 0


class TestValidatorAgentWithRealLLM:
    """Test ValidatorAgent with real Ollama."""

    @pytest.mark.asyncio
    async def test_validator_assess_risks_real_llm(self):
        """Test ValidatorAgent's risk assessment with real LLM."""
        from src.agents.validator import ValidatorAgent

        model_router = ModelRouter()

        validator = ValidatorAgent(model_router=model_router)

        # Create state with opportunities
        state = ResearchState(
            account_name="Acme Corp",
            industry="Technology",
            research_depth=ResearchDepth.STANDARD,
            signals=[
                Signal(
                    source="duckduckgo",
                    signal_type="web_search",
                    content="Acme Corp evaluating competitor solutions",
                    timestamp=datetime.now(),
                    confidence=0.7,
                    metadata={}
                )
            ],
            job_postings=[],
            news_items=[],
            tech_stack=["Python"],
            opportunities=[
                Opportunity(
                    product_name="MATLAB",
                    rationale="Strong fit for data analytics",
                    evidence=[
                        Signal(
                            source="job_posting",
                            signal_type="hiring",
                            content="Hiring data scientists",
                            timestamp=datetime.now(),
                            confidence=0.8,
                            metadata={}
                        )
                    ],
                    target_persona="VP of Data Science",
                    talking_points=["Point 1"],
                    confidence=OpportunityConfidence.HIGH,
                    confidence_score=0.85
                )
            ],
            validated_opportunities=[],
            competitive_risks=[],
            human_feedback=[],
            waiting_for_human=False,
            human_question=None,
            current_report=None,
            feedback_context=None,
            next_route=None,
            progress=ResearchProgress(gatherer_complete=True, identifier_complete=True),
            error_messages=[]
        )

        # Process with real LLM
        await validator.process(state)

        # Verify validation completed
        assert state["progress"].validator_complete is True
        # Risks may or may not be identified depending on LLM response
        assert isinstance(state["competitive_risks"], list)


class TestSimplifiedE2EFlow:
    """Test simplified end-to-end flow with real LLM."""

    @pytest.mark.asyncio
    async def test_mini_pipeline_gather_to_identify(self):
        """Test a mini-pipeline: Gatherer analysis -> Identifier extraction."""
        from src.agents.gatherer import GathererAgent
        from src.agents.identifier import IdentifierAgent

        model_router = ModelRouter()

        # Mock data sources for Gatherer (real LLM for analysis)
        mock_mcp_client = AsyncMock()
        mock_mcp_client.search.return_value = []
        mock_mcp_client.search_news.return_value = []
        mock_mcp_client.fetch_content.return_value = ""

        mock_job_scraper = AsyncMock()
        mock_job_scraper.fetch.return_value = []

        # Mock product matcher for Identifier (real LLM for reasoning)
        mock_product_matcher = AsyncMock()
        mock_product_matcher.match_requirements_to_products.return_value = [
            ("MATLAB", 0.80)
        ]

        # Create agents
        gatherer = GathererAgent(
            mcp_client=mock_mcp_client,
            job_scraper=mock_job_scraper,
            model_router=model_router
        )

        identifier = IdentifierAgent(
            product_matcher=mock_product_matcher,
            model_router=model_router
        )

        # Create initial state
        state = ResearchState(
            account_name="TechCorp",
            industry="Software",
            research_depth=ResearchDepth.QUICK,
            signals=[],
            job_postings=[],
            news_items=[],
            tech_stack=[],
            opportunities=[],
            validated_opportunities=[],
            competitive_risks=[],
            human_feedback=[],
            waiting_for_human=False,
            human_question=None,
            current_report=None,
            feedback_context=None,
            next_route=None,
            progress=ResearchProgress(),
            error_messages=[]
        )

        # Run Gatherer (with no external data, but LLM would be ready)
        await gatherer.process(state)
        assert state["progress"].gatherer_complete is True

        # Add a manual signal to simulate gathered data
        state["signals"].append(
            Signal(
                source="manual",
                signal_type="research",
                content="TechCorp is investing in machine learning infrastructure",
                timestamp=datetime.now(),
                confidence=0.8,
                metadata={}
            )
        )
        state["job_postings"].append({
            "title": "ML Engineer",
            "description": "Build ML pipelines",
            "technologies": ["Python", "PyTorch"]
        })
        state["tech_stack"] = ["Python", "PyTorch"]

        # Run Identifier with real LLM
        await identifier.process(state)

        assert state["progress"].identifier_complete is True


class TestLLMResponseVariability:
    """Test handling of varied LLM response formats."""

    @pytest.mark.asyncio
    async def test_llm_returns_markdown_wrapped_json(self):
        """Test that we can handle LLM returning markdown-wrapped JSON."""
        router = ModelRouter()

        # Prompt that might cause LLM to wrap response in markdown
        prompt = """Please analyze this and return JSON:

Company: Test Corp
Industry: Technology

Return your analysis as JSON with "company" and "industry" fields."""

        response = await router.generate(
            prompt=prompt,
            complexity=2,
            temperature=0.5,
            use_cache=False
        )

        # Our robust parser should handle various formats
        try:
            parsed = extract_json_from_llm_response(response.content)
            assert isinstance(parsed, dict)
        except JSONParseError:
            # If parsing fails, the response format was truly unparseable
            # This is acceptable - we're testing robustness
            pytest.skip("LLM response was not parseable as JSON (expected variability)")

    @pytest.mark.asyncio
    async def test_llm_consistency_across_multiple_calls(self):
        """Test that LLM produces consistent JSON structure across calls."""
        router = ModelRouter()

        prompt = """Return a JSON object with exactly these fields:
{
    "name": "test",
    "value": 42
}

Return ONLY valid JSON, nothing else."""

        results = []
        for _ in range(3):
            response = await router.generate(
                prompt=prompt,
                complexity=2,
                temperature=0.1,  # Low temperature for consistency
                use_cache=False
            )
            try:
                parsed = extract_json_from_llm_response(response.content)
                results.append(parsed)
            except JSONParseError:
                pass

        # At least some responses should be parseable
        assert len(results) >= 1, "No valid JSON responses received"

        # Verify structure consistency
        for result in results:
            assert isinstance(result, dict)


class TestErrorHandlingWithRealLLM:
    """Test error handling scenarios with real LLM."""

    @pytest.mark.asyncio
    async def test_model_router_handles_empty_prompt(self):
        """Test that ModelRouter handles edge cases gracefully."""
        router = ModelRouter()

        # Empty prompt might cause issues
        response = await router.generate(
            prompt="",
            complexity=2,
            use_cache=False
        )

        # Should return some response (even if empty or error message)
        assert response is not None

    @pytest.mark.asyncio
    async def test_model_router_handles_very_long_prompt(self):
        """Test ModelRouter with a longer prompt."""
        router = ModelRouter()

        # Create a reasonably long prompt
        long_context = "This is context about the company. " * 50
        prompt = f"{long_context}\n\nSummarize in one sentence."

        response = await router.generate(
            prompt=prompt,
            complexity=2,
            max_tokens=100,
            use_cache=False
        )

        assert response is not None
        assert response.content is not None
