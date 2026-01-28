"""
Integration tests using realistic fixtures.

These tests verify that the agents can handle real-world data formats,
including varied LLM response formats and realistic data source outputs.

Unlike the mocked tests, these tests verify:
- JSON parsing robustness (markdown, extra text, whitespace)
- Data flow with realistic search results
- Data flow with realistic job postings
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from tests.fixtures import FixtureLoader
from tests.fixtures.loader import extract_json_from_llm_response
from src.models.state import ResearchState, ResearchDepth, ResearchProgress
from src.models.domain import SearchResult, NewsItem, JobPosting, ModelResponse


# Initialize fixture loader
@pytest.fixture
def fixture_loader():
    """Provide fixture loader for tests."""
    return FixtureLoader()


class TestJsonParsingRobustness:
    """Test JSON parsing with varied LLM output formats."""

    def test_extract_json_clean(self, fixture_loader):
        """Test extraction from clean JSON."""
        response = fixture_loader.get_llm_response("gatherer_analysis", "clean")
        result = extract_json_from_llm_response(response)

        assert result["confidence"] == 0.85
        assert "Acme Corp" in result["summary"]
        assert result["source_type"] == "official_company_site"

    def test_extract_json_markdown(self, fixture_loader):
        """Test extraction from markdown-wrapped JSON."""
        response = fixture_loader.get_llm_response("gatherer_analysis", "markdown")
        result = extract_json_from_llm_response(response)

        assert result["confidence"] == 0.85
        assert "key_facts" in result
        assert len(result["key_facts"]) == 5

    def test_extract_json_extra_text(self, fixture_loader):
        """Test extraction from JSON with explanatory text."""
        response = fixture_loader.get_llm_response("gatherer_analysis", "extra_text")
        result = extract_json_from_llm_response(response)

        assert result["confidence"] == 0.85
        assert result["relevance"] == "high"

    def test_extract_json_whitespace(self, fixture_loader):
        """Test extraction from JSON with extra whitespace."""
        response = fixture_loader.get_llm_response("gatherer_analysis", "whitespace")
        result = extract_json_from_llm_response(response)

        assert result["confidence"] == 0.85
        assert "keywords" in result

    def test_extract_json_partial_fails(self, fixture_loader):
        """Test that partial JSON raises error."""
        response = fixture_loader.get_llm_response("gatherer_analysis", "partial")

        with pytest.raises(json.JSONDecodeError):
            extract_json_from_llm_response(response)

    def test_extract_requirements_markdown(self, fixture_loader):
        """Test requirements extraction from markdown format."""
        response = fixture_loader.get_llm_response("identifier_requirements", "markdown")
        result = extract_json_from_llm_response(response)

        assert "requirements" in result
        assert len(result["requirements"]) == 8
        assert "automated testing" in result["requirements"][0].lower()

    def test_extract_opportunities_extra_text(self, fixture_loader):
        """Test opportunities extraction with extra text."""
        response = fixture_loader.get_llm_response("identifier_opportunities", "extra_text")
        result = extract_json_from_llm_response(response)

        assert "opportunities" in result
        assert len(result["opportunities"]) == 2
        assert result["opportunities"][0]["product_name"] == "MATLAB"
        assert result["opportunities"][0]["confidence_score"] == 0.85

    def test_extract_risks_markdown(self, fixture_loader):
        """Test risk extraction from markdown format."""
        response = fixture_loader.get_llm_response("validator_risks", "markdown")
        result = extract_json_from_llm_response(response)

        assert "risks" in result
        assert len(result["risks"]) == 5
        assert "competitor" in result["risks"][0].lower()

    def test_extract_scoring_whitespace(self, fixture_loader):
        """Test scoring extraction with whitespace."""
        response = fixture_loader.get_llm_response("validator_scoring", "whitespace")
        result = extract_json_from_llm_response(response)

        assert "scored_opportunities" in result
        assert len(result["scored_opportunities"]) == 2
        assert result["scored_opportunities"][0]["new_score"] == 0.82


class TestSearchResultFixtures:
    """Test with realistic search result fixtures."""

    def test_load_acme_corp_results(self, fixture_loader):
        """Test loading Acme Corp search results."""
        results = fixture_loader.get_search_results("acme_corp")

        assert len(results) == 5
        assert results[0]["title"] == "Acme Corp | About Us"
        assert "acmecorp.com" in results[0]["url"]

    def test_load_raw_mcp_response(self, fixture_loader):
        """Test loading raw MCP response format."""
        raw_response = fixture_loader.get_search_results_raw("acme_corp")

        assert "1. Acme Corp | About Us" in raw_response
        assert "URL:" in raw_response
        assert "Summary:" in raw_response

    def test_load_news_items(self, fixture_loader):
        """Test loading news items."""
        news = fixture_loader.get_news_items("acme_corp")

        assert len(news) == 3
        assert "Series C" in news[0]["title"]
        assert news[0]["source"] == "TechCrunch"

    def test_create_search_result_models(self, fixture_loader):
        """Test creating SearchResult models from fixtures."""
        results_data = fixture_loader.get_search_results("acme_corp")

        search_results = []
        for data in results_data:
            result = SearchResult(
                title=data["title"],
                url=data["url"],
                snippet=data["snippet"],
                source=data["source"]
            )
            search_results.append(result)

        assert len(search_results) == 5
        assert search_results[0].title == "Acme Corp | About Us"

    def test_create_news_item_models(self, fixture_loader):
        """Test creating NewsItem models from fixtures."""
        news_data = fixture_loader.get_news_items("acme_corp")

        news_items = []
        for data in news_data:
            item = NewsItem(
                title=data["title"],
                source=data["source"],
                url=data.get("url"),
                summary=data["summary"],
                published_date=datetime.fromisoformat(data["published_date"].replace("Z", "+00:00")) if data.get("published_date") else None
            )
            news_items.append(item)

        assert len(news_items) == 3
        assert news_items[0].source == "TechCrunch"


class TestJobPostingFixtures:
    """Test with realistic job posting fixtures."""

    def test_load_greenhouse_postings(self, fixture_loader):
        """Test loading Greenhouse job postings."""
        postings = fixture_loader.get_job_postings("greenhouse")

        assert len(postings) == 5
        assert "Data Scientist" in postings[0]["title"]
        assert "TensorFlow" in postings[0]["technologies"]

    def test_load_lever_postings(self, fixture_loader):
        """Test loading Lever job postings."""
        postings = fixture_loader.get_job_postings("lever")

        assert len(postings) == 4
        assert "ML Engineer" in postings[0]["title"]
        assert "MATLAB" in postings[3]["technologies"]

    def test_load_greenhouse_html(self, fixture_loader):
        """Test loading Greenhouse HTML for scraper testing."""
        html = fixture_loader.get_job_postings_html("greenhouse")

        assert "opening" in html
        assert "Senior Data Scientist" in html
        assert "data-qa" in html

    def test_create_job_posting_models(self, fixture_loader):
        """Test creating JobPosting models from fixtures."""
        postings_data = fixture_loader.get_job_postings("greenhouse")

        job_postings = []
        for data in postings_data:
            posting = JobPosting(
                title=data["title"],
                company=data["company"],
                description=data["description"],
                location=data.get("location"),
                url=data.get("url"),
                required_skills=data.get("required_skills", []),
                technologies=data.get("technologies", []),
                seniority_level=data.get("seniority_level"),
                confidence=data.get("confidence", 0.5)
            )
            job_postings.append(posting)

        assert len(job_postings) == 5
        assert job_postings[0].company == "Acme Corp"
        assert "Python" in job_postings[0].required_skills

    def test_extract_tech_stack_from_postings(self, fixture_loader):
        """Test extracting tech stack from job postings."""
        postings_data = fixture_loader.get_job_postings("greenhouse")

        # Simulate the tech stack extraction logic from gatherer
        tech_stack = set()
        for job in postings_data:
            tech_stack.update(job.get("technologies", []))
            tech_stack.update(job.get("required_skills", []))

        assert "Python" in tech_stack
        assert "TensorFlow" in tech_stack
        assert "Kubernetes" in tech_stack
        assert len(tech_stack) > 15  # Should have many unique technologies


class TestRealisticGathererFlow:
    """Test gatherer agent with realistic fixtures."""

    @pytest.fixture
    def mock_research_state(self):
        """Create a research state for testing."""
        return ResearchState(
            account_name="Acme Corp",
            industry="Enterprise Software",
            research_depth=ResearchDepth.STANDARD,
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

    @pytest.mark.asyncio
    async def test_gatherer_with_realistic_search_results(self, fixture_loader, mock_research_state):
        """Test gatherer processing with realistic search results."""
        from src.agents.gatherer import GathererAgent

        # Load realistic fixtures
        search_data = fixture_loader.get_search_results("acme_corp")
        news_data = fixture_loader.get_news_items("acme_corp")
        llm_response = fixture_loader.get_llm_response("gatherer_analysis", "markdown")

        # Create mock dependencies
        mock_mcp_client = AsyncMock()
        mock_mcp_client.search.return_value = [
            SearchResult(**data) for data in search_data
        ]
        mock_mcp_client.search_news.return_value = [
            NewsItem(**{**data, "published_date": datetime.fromisoformat(data["published_date"].replace("Z", "+00:00")) if data.get("published_date") else None})
            for data in news_data
        ]
        mock_mcp_client.fetch_content.return_value = "Full page content here..."

        mock_job_scraper = AsyncMock()
        mock_job_scraper.fetch.return_value = []

        mock_model_router = AsyncMock()
        mock_model_router.generate.return_value = ModelResponse(
            content=llm_response,
            model="llama3.2:3b"
        )

        # Create agent
        gatherer = GathererAgent(
            mcp_client=mock_mcp_client,
            job_scraper=mock_job_scraper,
            model_router=mock_model_router
        )

        # Process (this tests the actual data flow)
        await gatherer.process(mock_research_state)

        # Verify signals were created
        assert len(mock_research_state["signals"]) > 0
        assert mock_research_state["progress"].gatherer_complete is True


class TestRealisticIdentifierFlow:
    """Test identifier agent with realistic fixtures.

    NOTE: These tests use "clean" JSON variants because the current agent code
    uses raw json.loads() without robust parsing. The TestJsonParsingRobustness
    tests verify the extract_json_from_llm_response helper can handle varied formats.

    TODO: Future enhancement - integrate robust JSON parsing into agent code to
    handle markdown-wrapped and extra-text responses from LLMs.
    """

    @pytest.mark.asyncio
    async def test_identifier_with_realistic_llm_responses(self, fixture_loader):
        """Test identifier processing with realistic LLM responses."""
        from src.agents.identifier import IdentifierAgent
        from src.models.state import Signal

        # Load realistic fixtures - using "clean" variant because agent code
        # currently uses raw json.loads() (see class docstring)
        requirements_response = fixture_loader.get_llm_response("identifier_requirements", "clean")
        opportunities_response = fixture_loader.get_llm_response("identifier_opportunities", "clean")

        # Create mock state with signals
        state = ResearchState(
            account_name="Acme Corp",
            industry="Enterprise Software",
            research_depth=ResearchDepth.STANDARD,
            signals=[
                Signal(
                    source="duckduckgo",
                    signal_type="web_search",
                    content="Acme Corp is hiring data scientists",
                    timestamp=datetime.now(),
                    confidence=0.8,
                    metadata={}
                )
            ],
            job_postings=[
                {
                    "title": "Senior Data Scientist",
                    "description": "Build ML models",
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

        # Create mock dependencies
        mock_product_matcher = AsyncMock()
        mock_product_matcher.match_requirements_to_products.return_value = [
            ("MATLAB", 0.85),
            ("Simulink", 0.72)
        ]

        mock_model_router = AsyncMock()
        # First call returns requirements, second returns opportunities
        mock_model_router.generate.side_effect = [
            ModelResponse(content=requirements_response, model="llama-3.1-8b-instant"),
            ModelResponse(content=opportunities_response, model="llama-3.1-8b-instant")
        ]

        # Create agent
        identifier = IdentifierAgent(
            product_matcher=mock_product_matcher,
            model_router=mock_model_router
        )

        # Process
        await identifier.process(state)

        # Verify opportunities were created
        assert len(state["opportunities"]) > 0
        assert state["progress"].identifier_complete is True
        assert state["opportunities"][0].product_name == "MATLAB"


class TestRealisticValidatorFlow:
    """Test validator agent with realistic fixtures.

    NOTE: Uses "clean" JSON variants - see TestRealisticIdentifierFlow docstring.
    """

    @pytest.mark.asyncio
    async def test_validator_with_realistic_llm_responses(self, fixture_loader):
        """Test validator processing with realistic LLM responses."""
        from src.agents.validator import ValidatorAgent
        from src.models.state import Opportunity, OpportunityConfidence, Signal

        # Load realistic fixtures - using "clean" variant (see class docstring)
        risks_response = fixture_loader.get_llm_response("validator_risks", "clean")
        scoring_response = fixture_loader.get_llm_response("validator_scoring", "clean")

        # Create mock state with opportunities
        state = ResearchState(
            account_name="Acme Corp",
            industry="Enterprise Software",
            research_depth=ResearchDepth.STANDARD,
            signals=[
                Signal(
                    source="duckduckgo",
                    signal_type="web_search",
                    content="Acme Corp using competitor tools",
                    timestamp=datetime.now(),
                    confidence=0.7,
                    metadata={}
                )
            ],
            job_postings=[],
            news_items=[],
            tech_stack=[],
            opportunities=[
                Opportunity(
                    product_name="MATLAB",
                    rationale="Strong fit for data analytics needs",
                    evidence=[],
                    target_persona="VP of Data Science",
                    talking_points=["Point 1"],
                    confidence=OpportunityConfidence.HIGH,
                    confidence_score=0.85
                ),
                Opportunity(
                    product_name="Simulink",
                    rationale="Embedded systems opportunity",
                    evidence=[],
                    target_persona="Director of Engineering",
                    talking_points=["Point 1"],
                    confidence=OpportunityConfidence.MEDIUM,
                    confidence_score=0.68
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

        # Create mock model router
        mock_model_router = AsyncMock()
        mock_model_router.generate.side_effect = [
            ModelResponse(content=risks_response, model="llama-3.1-8b-instant"),
            ModelResponse(content=scoring_response, model="llama-3.1-8b-instant")
        ]

        # Create agent
        validator = ValidatorAgent(model_router=mock_model_router)

        # Process
        await validator.process(state)

        # Verify validation completed
        assert state["progress"].validator_complete is True
        assert len(state["competitive_risks"]) > 0
        assert len(state["validated_opportunities"]) > 0


class TestFixtureLoaderUtilities:
    """Test the fixture loader utility functions."""

    def test_list_llm_fixtures(self, fixture_loader):
        """Test listing available LLM fixtures."""
        fixtures = fixture_loader.list_fixtures("llm")

        assert "gatherer_analysis" in fixtures
        assert "identifier_requirements" in fixtures
        assert "identifier_opportunities" in fixtures
        assert "validator_risks" in fixtures
        assert "validator_scoring" in fixtures

    def test_list_search_fixtures(self, fixture_loader):
        """Test listing available search fixtures."""
        fixtures = fixture_loader.list_fixtures("search")

        assert "acme_corp" in fixtures
        assert "tech_startup" in fixtures

    def test_list_job_fixtures(self, fixture_loader):
        """Test listing available job fixtures."""
        fixtures = fixture_loader.list_fixtures("jobs")

        assert "greenhouse" in fixtures
        assert "lever" in fixtures
        assert "generic" in fixtures

    def test_get_expected_parsed(self, fixture_loader):
        """Test getting expected parsed data."""
        expected = fixture_loader.get_llm_response_parsed("gatherer_analysis")

        assert expected["confidence"] == 0.85
        assert len(expected["key_facts"]) == 5

    def test_fixture_not_found_raises_error(self, fixture_loader):
        """Test that missing fixtures raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            fixture_loader.get_llm_response("nonexistent_fixture", "clean")

    def test_invalid_variant_raises_error(self, fixture_loader):
        """Test that invalid variants raise ValueError."""
        with pytest.raises(ValueError):
            fixture_loader.get_llm_response("gatherer_analysis", "invalid_variant")
