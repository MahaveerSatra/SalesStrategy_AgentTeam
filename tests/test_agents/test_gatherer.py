"""
Tests for Intelligence Gatherer Agent.
Comprehensive test coverage with mocked dependencies.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from pydantic import HttpUrl

from src.agents.gatherer import GathererAgent
from src.models.state import ResearchState, Signal, ResearchProgress, ResearchDepth, create_initial_state
from src.models.domain import SearchResult, JobPosting, NewsItem
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.job_boards import JobBoardScraper
from src.core.model_router import ModelRouter


@pytest.fixture
def mock_mcp_client():
    """Provide mocked MCP client."""
    client = AsyncMock(spec=DuckDuckGoMCPClient)
    return client


@pytest.fixture
def mock_job_scraper():
    """Provide mocked job scraper."""
    scraper = AsyncMock(spec=JobBoardScraper)
    return scraper


@pytest.fixture
def mock_model_router():
    """Provide mocked model router for LLM analysis."""
    router = AsyncMock(spec=ModelRouter)

    # Mock response for LLM analysis
    mock_response = MagicMock()
    mock_response.content = """{
        "confidence": 0.85,
        "summary": "This is an LLM-generated summary of the source content.",
        "source_type": "official_company_site",
        "key_facts": ["Fact 1", "Fact 2"],
        "keywords": ["keyword1", "keyword2"],
        "relevance": "high"
    }"""
    router.generate.return_value = mock_response

    return router


@pytest.fixture
def gatherer_agent(mock_mcp_client, mock_job_scraper, mock_model_router):
    """Provide GathererAgent instance with mocked dependencies."""
    return GathererAgent(
        mcp_client=mock_mcp_client,
        job_scraper=mock_job_scraper,
        model_router=mock_model_router
    )


@pytest.fixture
def sample_search_results():
    """Provide sample search results."""
    return [
        SearchResult(
            title="Acme Corp - Official Site",
            url=HttpUrl("https://acme.com"),
            snippet="Leading provider of enterprise solutions",
            source="duckduckgo",
            timestamp=datetime.now()
        ),
        SearchResult(
            title="Acme Corp Technology Stack",
            url=HttpUrl("https://stackshare.io/acme"),
            snippet="Acme uses Python, React, and AWS",
            source="duckduckgo",
            timestamp=datetime.now()
        )
    ]


@pytest.fixture
def sample_job_postings():
    """Provide sample job postings."""
    return [
        JobPosting(
            title="Senior Software Engineer",
            company="Acme Corp",
            location="Boston, MA",
            url=HttpUrl("https://acme.com/careers/123"),
            description="Build scalable systems using Python and AWS",
            posted_date=datetime.now(),
            technologies=["Python", "AWS", "Docker"],
            required_skills=["Python", "Microservices"],
            experience_level="Senior"
        ),
        JobPosting(
            title="Machine Learning Engineer",
            company="Acme Corp",
            location="Remote",
            url=HttpUrl("https://acme.com/careers/124"),
            description="Develop ML models using TensorFlow",
            posted_date=datetime.now(),
            technologies=["Python", "TensorFlow", "Kubernetes"],
            required_skills=["Machine Learning", "Python"],
            experience_level="Mid-Senior"
        )
    ]


@pytest.fixture
def sample_news_items():
    """Provide sample news items."""
    return [
        NewsItem(
            title="Acme Corp Expands Cloud Services",
            url=HttpUrl("https://news.example.com/acme-cloud"),
            summary="Acme announces new cloud platform for enterprises",
            source="TechNews",
            published_date=datetime.now()
        ),
        NewsItem(
            title="Acme Corp Raises $50M Series B",
            url=HttpUrl("https://news.example.com/acme-funding"),
            summary="Acme secures funding to expand AI capabilities",
            source="VentureBeat",
            published_date=datetime.now()
        )
    ]


@pytest.fixture
def initial_state():
    """Provide initial research state with company_domain for testing."""
    state = create_initial_state(
        account_name="Acme Corp",
        industry="Technology",
        region="North America",
        research_depth=ResearchDepth.STANDARD
    )
    # Add company_domain for testing purposes (not in ResearchState TypedDict by default)
    # This allows job scraper to be called in tests
    state["company_domain"] = "acme.com"  # type: ignore
    return state


class TestGathererAgentInit:
    """Test GathererAgent initialization."""

    def test_init_creates_agent(self, mock_mcp_client, mock_job_scraper, mock_model_router):
        """Test that agent initializes correctly."""
        agent = GathererAgent(
            mcp_client=mock_mcp_client,
            job_scraper=mock_job_scraper,
            model_router=mock_model_router
        )

        assert agent.name == "gatherer"
        assert agent.mcp_client == mock_mcp_client
        assert agent.job_scraper == mock_job_scraper
        assert agent.model_router == mock_model_router
        assert agent._analysis_cache == {}

    def test_get_complexity(self, gatherer_agent, initial_state):
        """Test complexity returns 3 for LOCAL Ollama LLM analysis."""
        complexity = gatherer_agent.get_complexity(initial_state)
        assert complexity == 3


class TestGathererAgentSuccessfulDataGathering:
    """Test successful data gathering scenarios."""

    @pytest.mark.asyncio
    async def test_successful_full_data_collection(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper,
        sample_search_results,
        sample_job_postings,
        sample_news_items
    ):
        """Test successful data collection from all sources."""
        # Setup mocks
        mock_mcp_client.search.return_value = sample_search_results
        mock_mcp_client.search_news.return_value = sample_news_items
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample webpage content</body></html>"
        mock_job_scraper.fetch.return_value = sample_job_postings

        # Execute
        await gatherer_agent.process(initial_state)

        # Verify signals created
        assert len(initial_state["signals"]) > 0

        # Check search signals (now with LLM analysis)
        search_signals = [s for s in initial_state["signals"] if s.signal_type == "web_search"]
        assert len(search_signals) == 2
        assert search_signals[0].source == "duckduckgo"
        assert search_signals[0].confidence == 0.85  # From mocked LLM response
        assert "source_type" in search_signals[0].metadata
        assert "key_facts" in search_signals[0].metadata
        assert "keywords" in search_signals[0].metadata

        # Check hiring signals
        hiring_signals = [s for s in initial_state["signals"] if s.signal_type == "hiring"]
        assert len(hiring_signals) == 2
        assert hiring_signals[0].source == "job_boards"
        assert hiring_signals[0].confidence == 0.9

        # Check news signals
        news_signals = [s for s in initial_state["signals"] if s.signal_type == "news"]
        assert len(news_signals) == 2
        assert news_signals[0].source == "duckduckgo_news"
        assert news_signals[0].confidence == 0.7

        # Verify job postings stored
        assert len(initial_state["job_postings"]) == 2
        assert initial_state["job_postings"][0]["title"] == "Senior Software Engineer"

        # Verify news items stored
        assert len(initial_state["news_items"]) == 2
        assert initial_state["news_items"][0]["title"] == "Acme Corp Expands Cloud Services"

        # Verify tech stack extracted
        assert len(initial_state["tech_stack"]) > 0
        assert "Python" in initial_state["tech_stack"]
        assert "AWS" in initial_state["tech_stack"]
        assert "TensorFlow" in initial_state["tech_stack"]

        # Verify progress marked complete
        assert initial_state["progress"].gatherer_complete is True

    @pytest.mark.asyncio
    async def test_search_without_job_postings(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper,
        sample_search_results,
        sample_news_items
    ):
        """Test when job postings fail but search succeeds."""
        # Setup mocks - job scraper returns empty list
        mock_mcp_client.search.return_value = sample_search_results
        mock_mcp_client.search_news.return_value = sample_news_items
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        mock_job_scraper.fetch.return_value = []

        # Execute
        await gatherer_agent.process(initial_state)

        # Verify search and news signals created
        assert len(initial_state["signals"]) == 4  # 2 search (LLM analyzed) + 2 news
        assert len(initial_state["job_postings"]) == 0
        assert len(initial_state["tech_stack"]) == 0

        # Still marks complete
        assert initial_state["progress"].gatherer_complete is True


class TestGathererAgentPartialFailures:
    """Test partial failure scenarios."""

    @pytest.mark.asyncio
    async def test_search_fails_others_succeed(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper,
        sample_job_postings,
        sample_news_items
    ):
        """Test when search fails but job/news succeed."""
        # Setup mocks - search raises exception
        mock_mcp_client.search.side_effect = Exception("Search API error")
        mock_mcp_client.search_news.return_value = sample_news_items
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        mock_job_scraper.fetch.return_value = sample_job_postings

        # Execute - should not raise, handles exception
        await gatherer_agent.process(initial_state)

        # Verify error logged in state
        assert len(initial_state["error_messages"]) > 0
        assert "Web search failed" in initial_state["error_messages"][0]

        # Verify job and news signals still created
        hiring_signals = [s for s in initial_state["signals"] if s.signal_type == "hiring"]
        news_signals = [s for s in initial_state["signals"] if s.signal_type == "news"]
        assert len(hiring_signals) == 2
        assert len(news_signals) == 2

        # Still marks complete
        assert initial_state["progress"].gatherer_complete is True

    @pytest.mark.asyncio
    async def test_jobs_fail_others_succeed(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper,
        sample_search_results,
        sample_news_items
    ):
        """Test when job fetch fails but search/news succeed."""
        # Setup mocks - job scraper raises exception
        mock_mcp_client.search.return_value = sample_search_results
        mock_mcp_client.search_news.return_value = sample_news_items
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        mock_job_scraper.fetch.side_effect = Exception("Career page not found")

        # Execute
        await gatherer_agent.process(initial_state)

        # Verify error logged
        assert len(initial_state["error_messages"]) > 0
        assert "Job posting collection failed" in initial_state["error_messages"][0]

        # Verify search and news signals created
        search_signals = [s for s in initial_state["signals"] if s.signal_type == "web_search"]
        news_signals = [s for s in initial_state["signals"] if s.signal_type == "news"]
        assert len(search_signals) == 2
        assert len(news_signals) == 2

        # Job postings empty, tech stack empty
        assert len(initial_state["job_postings"]) == 0
        assert len(initial_state["tech_stack"]) == 0

        # Still marks complete
        assert initial_state["progress"].gatherer_complete is True

    @pytest.mark.asyncio
    async def test_news_fails_others_succeed(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper,
        sample_search_results,
        sample_job_postings
    ):
        """Test when news fails but search/jobs succeed."""
        # Setup mocks - news raises exception
        mock_mcp_client.search.return_value = sample_search_results
        mock_mcp_client.search_news.side_effect = Exception("News API rate limited")
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        mock_job_scraper.fetch.return_value = sample_job_postings

        # Execute
        await gatherer_agent.process(initial_state)

        # Verify error logged
        assert len(initial_state["error_messages"]) > 0
        assert "News collection failed" in initial_state["error_messages"][0]

        # Verify search and job signals created
        search_signals = [s for s in initial_state["signals"] if s.signal_type == "web_search"]
        hiring_signals = [s for s in initial_state["signals"] if s.signal_type == "hiring"]
        assert len(search_signals) == 2
        assert len(hiring_signals) == 2

        # News items empty
        assert len(initial_state["news_items"]) == 0

        # Still marks complete
        assert initial_state["progress"].gatherer_complete is True


class TestGathererAgentCompleteFailures:
    """Test complete failure scenarios."""

    @pytest.mark.asyncio
    async def test_all_sources_fail(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper
    ):
        """Test when all data sources fail."""
        # Setup mocks - all raise exceptions
        mock_mcp_client.search.side_effect = Exception("Search failed")
        mock_mcp_client.search_news.side_effect = Exception("News failed")
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        mock_job_scraper.fetch.side_effect = Exception("Jobs failed")

        # Execute - should handle gracefully
        await gatherer_agent.process(initial_state)

        # Verify all errors logged
        assert len(initial_state["error_messages"]) == 3

        # Verify empty results
        assert len(initial_state["signals"]) == 0
        assert len(initial_state["job_postings"]) == 0
        assert len(initial_state["news_items"]) == 0
        assert len(initial_state["tech_stack"]) == 0

        # Still marks complete (agent attempted)
        assert initial_state["progress"].gatherer_complete is True


class TestGathererAgentTechStackExtraction:
    """Test tech stack extraction logic."""

    def test_extract_tech_stack_from_multiple_jobs(self, gatherer_agent):
        """Test tech stack extraction with multiple jobs."""
        jobs = [
            {
                "title": "Backend Engineer",
                "technologies": ["Python", "PostgreSQL"],
                "required_skills": ["Docker", "Kubernetes"]
            },
            {
                "title": "Frontend Engineer",
                "technologies": ["React", "TypeScript"],
                "required_skills": ["JavaScript", "React"]
            }
        ]

        tech_stack = gatherer_agent._extract_tech_stack(jobs)

        # Verify all unique technologies extracted
        assert "Python" in tech_stack
        assert "PostgreSQL" in tech_stack
        assert "Docker" in tech_stack
        assert "Kubernetes" in tech_stack
        assert "React" in tech_stack
        assert "TypeScript" in tech_stack
        assert "JavaScript" in tech_stack

        # Verify no duplicates
        assert len(tech_stack) == len(set(tech_stack))

        # Verify sorted
        assert tech_stack == sorted(tech_stack)

    def test_extract_tech_stack_empty_jobs(self, gatherer_agent):
        """Test tech stack extraction with no jobs."""
        tech_stack = gatherer_agent._extract_tech_stack([])
        assert tech_stack == []

    def test_extract_tech_stack_jobs_without_tech_fields(self, gatherer_agent):
        """Test tech stack extraction when jobs have no tech fields."""
        jobs = [
            {"title": "Manager", "location": "Boston"},
            {"title": "Analyst", "location": "Remote"}
        ]

        tech_stack = gatherer_agent._extract_tech_stack(jobs)
        assert tech_stack == []

    def test_extract_tech_stack_with_single_string_values(self, gatherer_agent):
        """Test tech stack extraction when tech fields are strings not lists."""
        jobs = [
            {
                "title": "Engineer",
                "technologies": "Python",  # String not list
                "required_skills": "AWS"   # String not list
            }
        ]

        tech_stack = gatherer_agent._extract_tech_stack(jobs)
        assert "Python" in tech_stack
        assert "AWS" in tech_stack

    def test_extract_tech_stack_handles_none_values(self, gatherer_agent):
        """Test tech stack extraction handles None values gracefully."""
        jobs = [
            {
                "title": "Engineer",
                "technologies": None,
                "required_skills": ["Python"]
            },
            {
                "title": "Engineer 2",
                "technologies": ["Java"],
                "required_skills": None
            }
        ]

        tech_stack = gatherer_agent._extract_tech_stack(jobs)
        assert "Python" in tech_stack
        assert "Java" in tech_stack
        assert len(tech_stack) == 2


class TestGathererAgentParallelExecution:
    """Test parallel execution behavior."""

    @pytest.mark.asyncio
    async def test_parallel_execution_with_asyncio_gather(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper,
        sample_search_results,
        sample_job_postings,
        sample_news_items
    ):
        """Test that all sources are called in parallel via asyncio.gather."""
        # Setup mocks
        mock_mcp_client.search.return_value = sample_search_results
        mock_mcp_client.search_news.return_value = sample_news_items
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        mock_job_scraper.fetch.return_value = sample_job_postings

        # Execute
        await gatherer_agent.process(initial_state)

        # Verify all sources were called
        mock_mcp_client.search.assert_called_once()
        mock_mcp_client.search_news.assert_called_once()
        mock_job_scraper.fetch.assert_called_once()

        # Verify results collected (2 search with LLM + 2 jobs + 2 news)
        assert len(initial_state["signals"]) == 6
        assert initial_state["progress"].gatherer_complete is True


class TestGathererAgentStateModifications:
    """Test state modifications."""

    @pytest.mark.asyncio
    async def test_state_modified_in_place(
        self,
        gatherer_agent,
        initial_state,
        mock_mcp_client,
        mock_job_scraper,
        sample_search_results
    ):
        """Test that state is modified in-place, not replaced."""
        # Get initial state reference
        original_state_id = id(initial_state)

        # Setup mocks
        mock_mcp_client.search.return_value = sample_search_results
        mock_mcp_client.search_news.return_value = []
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        mock_job_scraper.fetch.return_value = []

        # Execute
        await gatherer_agent.process(initial_state)

        # Verify same state object (modified in-place)
        assert id(initial_state) == original_state_id

        # Verify fields populated
        assert len(initial_state["signals"]) > 0
        assert initial_state["progress"].gatherer_complete is True

    @pytest.mark.asyncio
    async def test_missing_company_domain_handles_gracefully(
        self,
        gatherer_agent,
        mock_mcp_client,
        mock_job_scraper,
        sample_search_results,
        sample_news_items
    ):
        """Test when company_domain is not in state (ResearchState doesn't have this field)."""
        # Create state without company_domain (field doesn't exist in ResearchState)
        state = create_initial_state(
            account_name="Acme Corp",
            industry="Technology"
        )

        # Setup mocks
        mock_mcp_client.search.return_value = sample_search_results
        mock_mcp_client.search_news.return_value = sample_news_items
        mock_mcp_client.fetch_content.return_value = "<html><body>Sample content</body></html>"
        # Job scraper should NOT be called since domain is empty

        # Execute - should not crash
        await gatherer_agent.process(state)

        # Verify job scraper NOT called (empty domain results in early return)
        mock_job_scraper.fetch.assert_not_called()

        # Verify search and news data still collected
        assert len(state["signals"]) > 0
        search_signals = [s for s in state["signals"] if s.signal_type == "web_search"]
        news_signals = [s for s in state["signals"] if s.signal_type == "news"]
        assert len(search_signals) == 2
        assert len(news_signals) == 2

        # No job postings
        assert len(state["job_postings"]) == 0

        # Still marks complete
        assert state["progress"].gatherer_complete is True
