"""
Intelligence Gatherer Agent - Collects company intelligence from multiple sources.
Phase 3: Agent Implementation
"""
import asyncio
from typing import Any
from datetime import datetime

import structlog

from src.core.base_agent import StatelessAgent
from src.models.state import ResearchState, Signal
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.job_boards import JobBoardScraper
from src.core.exceptions import DataSourceError

logger = structlog.get_logger(__name__)


class GathererAgent(StatelessAgent):
    """
    Intelligence Gatherer Agent - Collects company data from multiple sources.

    Responsibilities:
    - Search web for company information
    - Collect job postings from career pages
    - Gather news articles about the company
    - Extract tech stack from job descriptions
    - Structure all data as Signal objects

    Modifies ResearchState in-place:
    - state["signals"] - Web search results as Signal objects
    - state["job_postings"] - Scraped job postings (dict format)
    - state["news_items"] - News articles (dict format)
    - state["tech_stack"] - Extracted technologies (list of strings)
    - state["progress"].gatherer_complete = True
    """

    def __init__(self, mcp_client: DuckDuckGoMCPClient, job_scraper: JobBoardScraper):
        """
        Initialize Gatherer Agent.

        Args:
            mcp_client: DuckDuckGo MCP client for web search
            job_scraper: Job board scraper for career pages
        """
        super().__init__(name="gatherer")
        self.mcp_client = mcp_client
        self.job_scraper = job_scraper

    async def process(self, state: ResearchState) -> None:
        """
        Collect data from multiple sources in parallel.

        Args:
            state: Current research state (modified in-place)
        """
        account = state["account_name"]
        industry = state.get("industry", "")
        region = state.get("region", "")

        self.logger.info(
            "gatherer_started",
            account=account,
            industry=industry,
            region=region
        )

        # Build search queries
        company_info_query = f"{account} company information"
        if industry:
            company_info_query += f" {industry}"

        news_query = f"{account} news technology"
        if industry:
            news_query += f" {industry}"

        # Parallel data collection with error recovery
        self.logger.info("gatherer_fetching_data", sources=3)

        # Extract company domain from state if available (not in ResearchState TypedDict)
        # Can be added dynamically for testing or future enhancement
        company_domain = state.get("company_domain", "")  # type: ignore

        try:
            search_results, job_postings, news_items = await asyncio.gather(
                self._search_company_info(company_info_query),
                self._fetch_job_postings(account, company_domain),
                self._search_news(news_query),
                return_exceptions=True  # Continue even if some fail
            )
        except Exception as e:
            self.logger.error("gatherer_parallel_fetch_failed", error=str(e))
            # Initialize with empty lists if total failure
            search_results = []
            job_postings = []
            news_items = []

        # Process search results -> Signal objects
        if isinstance(search_results, Exception):
            self.logger.warning(
                "search_failed",
                error=str(search_results),
                error_type=type(search_results).__name__
            )
            state["error_messages"].append(f"Web search failed: {search_results}")
        elif search_results:
            for result in search_results:
                try:
                    signal = Signal(
                        source="duckduckgo",
                        signal_type="web_search",
                        content=result.snippet,
                        timestamp=result.timestamp,
                        confidence=0.8,
                        metadata={
                            "url": str(result.url),
                            "title": result.title
                        }
                    )
                    state["signals"].append(signal)
                except Exception as e:
                    self.logger.warning("signal_creation_failed", error=str(e))

            self.logger.info("search_signals_added", count=len(search_results))

        # Process job postings
        if isinstance(job_postings, Exception):
            self.logger.warning(
                "job_fetch_failed",
                error=str(job_postings),
                error_type=type(job_postings).__name__
            )
            state["error_messages"].append(f"Job posting collection failed: {job_postings}")
        elif job_postings:
            # Convert JobPosting objects to dicts for state storage
            state["job_postings"] = [jp.model_dump() for jp in job_postings]
            self.logger.info("job_postings_added", count=len(job_postings))

            # Create signals from job postings
            for job in job_postings:
                try:
                    signal = Signal(
                        source="job_boards",
                        signal_type="hiring",
                        content=f"{job.title} - {job.company}",
                        timestamp=datetime.now(),
                        confidence=0.9,
                        metadata={
                            "location": job.location or "Unknown",
                            "technologies": job.technologies,
                            "url": str(job.url) if job.url else ""
                        }
                    )
                    state["signals"].append(signal)
                except Exception as e:
                    self.logger.warning("job_signal_creation_failed", error=str(e))

        # Process news items
        if isinstance(news_items, Exception):
            self.logger.warning(
                "news_fetch_failed",
                error=str(news_items),
                error_type=type(news_items).__name__
            )
            state["error_messages"].append(f"News collection failed: {news_items}")
        elif news_items:
            # Convert NewsItem objects to dicts for state storage
            state["news_items"] = [news.model_dump() for news in news_items]
            self.logger.info("news_items_added", count=len(news_items))

            # Create signals from news items
            for news in news_items:
                try:
                    signal = Signal(
                        source="duckduckgo_news",
                        signal_type="news",
                        content=news.summary,
                        timestamp=news.published_date if news.published_date else datetime.now(),
                        confidence=0.7,
                        metadata={
                            "url": str(news.url) if news.url else "",
                            "title": news.title,
                            "source": news.source
                        }
                    )
                    state["signals"].append(signal)
                except Exception as e:
                    self.logger.warning("news_signal_creation_failed", error=str(e))

        # Extract tech stack from job postings
        state["tech_stack"] = self._extract_tech_stack(state["job_postings"])
        self.logger.info("tech_stack_extracted", count=len(state["tech_stack"]))

        # Mark completion
        state["progress"].gatherer_complete = True

        # Log final metrics
        self.logger.info(
            "gatherer_completed",
            signals_count=len(state["signals"]),
            jobs_count=len(state["job_postings"]),
            news_count=len(state["news_items"]),
            tech_count=len(state["tech_stack"])
        )

    async def _search_company_info(self, query: str) -> list[Any]:
        """
        Search for company information.

        Args:
            query: Search query

        Returns:
            List of SearchResult objects

        Raises:
            Exception: If search fails (caught by caller)
        """
        try:
            results = await self.mcp_client.search(query, max_results=10)
            return results
        except Exception as e:
            self.logger.error("company_search_failed", query=query, error=str(e))
            raise

    async def _fetch_job_postings(self, company_name: str, company_domain: str) -> list[Any]:
        """
        Fetch job postings from company career page.

        Args:
            company_name: Company name
            company_domain: Company domain (may be empty)

        Returns:
            List of JobPosting objects

        Raises:
            Exception: If fetch fails (caught by caller)
        """
        try:
            # If no domain provided, try to infer from company name
            if not company_domain:
                self.logger.debug("no_domain_provided", company=company_name)
                # For now, return empty list - domain detection can be enhanced later
                return []

            jobs = await self.job_scraper.fetch(
                company_name=company_name,
                company_domain=company_domain
            )
            return jobs
        except Exception as e:
            self.logger.error(
                "job_fetch_failed",
                company=company_name,
                domain=company_domain,
                error=str(e)
            )
            raise

    async def _search_news(self, query: str) -> list[Any]:
        """
        Search for news articles.

        Args:
            query: News search query

        Returns:
            List of NewsItem objects

        Raises:
            Exception: If search fails (caught by caller)
        """
        try:
            news = await self.mcp_client.search_news(query, max_results=5)
            return news
        except Exception as e:
            self.logger.error("news_search_failed", query=query, error=str(e))
            raise

    def _extract_tech_stack(self, job_postings: list[dict]) -> list[str]:
        """
        Extract unique technologies from job postings.

        Args:
            job_postings: List of job posting dicts

        Returns:
            List of unique technology names
        """
        tech_stack = set()

        for job in job_postings:
            # Extract from technologies field if exists
            if "technologies" in job and job["technologies"]:
                if isinstance(job["technologies"], list):
                    tech_stack.update(job["technologies"])
                elif isinstance(job["technologies"], str):
                    # Handle single string
                    tech_stack.add(job["technologies"])

            # Also check required_skills
            if "required_skills" in job and job["required_skills"]:
                if isinstance(job["required_skills"], list):
                    tech_stack.update(job["required_skills"])
                elif isinstance(job["required_skills"], str):
                    tech_stack.add(job["required_skills"])

        # Return sorted list for consistent ordering
        return sorted(list(tech_stack))

    def get_complexity(self, state: ResearchState) -> int:
        """
        Get task complexity for model routing.

        Gatherer agent performs simple data collection without LLM reasoning,
        so complexity is low.

        Args:
            state: Current research state

        Returns:
            Complexity score (1-10). Gatherer returns 3 (simple data collection)
        """
        return 3  # Simple data collection, no LLM reasoning needed
