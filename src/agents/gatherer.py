"""
Intelligence Gatherer Agent - Collects company intelligence from multiple sources.
Phase 3: Agent Implementation
"""
import asyncio
import json
from typing import Any
from datetime import datetime

import structlog

from src.core.base_agent import StatelessAgent
from src.models.state import ResearchState, Signal, ResearchDepth
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.job_boards import JobBoardScraper
from src.core.model_router import ModelRouter
from src.core.exceptions import DataSourceError

logger = structlog.get_logger(__name__)


class GathererAgent(StatelessAgent):
    """
    Intelligence Gatherer Agent - Research Analyst that collects AND analyzes company intelligence.

    This agent is NOT a simple data collector - it's a Research Analyst that:
    1. Receives rich context from CoordinatorAgent
    2. Fetches AND analyzes each source individually with LLM
    3. Assigns per-source confidence based on authority + content quality + relevance
    4. Creates rich Signal objects with LLM-generated metadata

    Responsibilities:
    - Search web for company information (context-aware queries)
    - Collect job postings from career pages
    - Gather news articles about the company
    - Analyze EACH source with LOCAL LLM (Tier 1 Ollama, complexity=4)
    - Extract tech stack from job descriptions
    - Structure all data as Signal objects with confidence scores

    Modifies ResearchState in-place:
    - state["signals"] - Web search results as Signal objects (with LLM analysis)
    - state["job_postings"] - Scraped job postings (dict format)
    - state["news_items"] - News articles (dict format)
    - state["tech_stack"] - Extracted technologies (list of strings)
    - state["progress"].gatherer_complete = True
    """

    def __init__(
        self,
        mcp_client: DuckDuckGoMCPClient,
        job_scraper: JobBoardScraper,
        model_router: ModelRouter
    ):
        """
        Initialize Gatherer Agent.

        Args:
            mcp_client: DuckDuckGo MCP client for web search
            job_scraper: Job board scraper for career pages
            model_router: Model router for LLM analysis (Tier 1 Ollama)
        """
        super().__init__(name="gatherer")
        self.mcp_client = mcp_client
        self.job_scraper = job_scraper
        self.model_router = model_router
        self._analysis_cache: dict[int, Signal] = {}  # Cache LLM analyses by URL hash

    async def process(self, state: ResearchState) -> None:
        """
        Collect and analyze data from multiple sources.

        This method:
        1. Extracts ALL context fields from state
        2. Builds context-aware search queries
        3. Determines max_results based on research_depth
        4. Fetches raw data from sources
        5. Analyzes EACH source with LLM
        6. Creates rich Signal objects with metadata

        Args:
            state: Current research state (modified in-place)
        """
        # Extract ALL context fields
        account = state["account_name"]
        industry = state.get("industry", "")
        region = state.get("region", "")
        user_context = state.get("user_context", "")
        depth = state["research_depth"]

        self.logger.info(
            "gatherer_started",
            account=account,
            industry=industry,
            region=region,
            user_context=user_context[:50] if user_context else None,
            depth=depth.value
        )

        # Build rich, context-aware queries
        company_info_query = self._build_query(account, industry, region, user_context)
        news_query = f"{account} news technology"
        if industry:
            news_query += f" {industry}"

        # Determine max_results based on research_depth
        max_results = {
            ResearchDepth.QUICK: 5,
            ResearchDepth.STANDARD: 10,
            ResearchDepth.DEEP: 20
        }[depth]

        self.logger.info(
            "gatherer_fetching_data",
            sources=3,
            max_results=max_results,
            depth=depth.value
        )

        # Extract company domain from state if available (not in ResearchState TypedDict)
        # Can be added dynamically for testing or future enhancement
        company_domain = state.get("company_domain", "")  # type: ignore

        try:
            search_results, job_postings, news_items = await asyncio.gather(
                self._search_company_info(company_info_query, max_results),
                self._fetch_job_postings(account, company_domain),
                self._search_news(news_query, max_results=5),
                return_exceptions=True  # Continue even if some fail
            )
        except Exception as e:
            self.logger.error("gatherer_parallel_fetch_failed", error=str(e))
            # Initialize with empty lists if total failure
            search_results = []
            job_postings = []
            news_items = []

        # Process search results -> Analyze EACH with LLM -> Signal objects
        if isinstance(search_results, Exception):
            self.logger.warning(
                "search_failed",
                error=str(search_results),
                error_type=type(search_results).__name__
            )
            state["error_messages"].append(f"Web search failed: {search_results}")
        elif search_results:
            self.logger.info("analyzing_search_results", count=len(search_results))
            for result in search_results:
                try:
                    # Fetch full webpage content
                    full_content = await self.mcp_client.fetch_content(str(result.url))

                    # Analyze with LLM (Tier 1 Ollama)
                    analyzed_signal = await self._analyze_source_with_llm(
                        url=str(result.url),
                        title=result.title,
                        snippet=result.snippet,
                        full_content=full_content,
                        account_name=account,
                        industry=industry
                    )

                    state["signals"].append(analyzed_signal)
                except Exception as e:
                    self.logger.warning(
                        "signal_analysis_failed",
                        url=str(result.url) if result else "unknown",
                        error=str(e)
                    )
                    # Fallback: Create signal without LLM analysis
                    signal = Signal(
                        source="duckduckgo",
                        signal_type="web_search",
                        content=result.snippet,
                        timestamp=result.timestamp,
                        confidence=0.5,  # Lower confidence without analysis
                        metadata={
                            "url": str(result.url),
                            "title": result.title,
                            "analysis_failed": True
                        }
                    )
                    state["signals"].append(signal)

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

    async def _search_company_info(self, query: str, max_results: int = 10) -> list[Any]:
        """
        Search for company information.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            Exception: If search fails (caught by caller)
        """
        try:
            results = await self.mcp_client.search(query, max_results=max_results)
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

    async def _search_news(self, query: str, max_results: int = 5) -> list[Any]:
        """
        Search for news articles.

        Args:
            query: News search query
            max_results: Maximum number of results to return

        Returns:
            List of NewsItem objects

        Raises:
            Exception: If search fails (caught by caller)
        """
        try:
            news = await self.mcp_client.search_news(query, max_results=max_results)
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

    def _build_query(
        self,
        account_name: str,
        industry: str,
        region: str,
        user_context: str
    ) -> str:
        """
        Build rich, context-aware search query.

        Args:
            account_name: Company name
            industry: Industry vertical
            region: Geographic region
            user_context: Additional context from user

        Returns:
            Enriched search query string
        """
        query_parts = [account_name, "company information"]

        if industry:
            query_parts.append(industry)

        if region:
            query_parts.append(region)

        if user_context:
            # Extract key terms from user context (first 100 chars)
            context_snippet = user_context[:100].strip()
            query_parts.append(context_snippet)

        return " ".join(query_parts)

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

        Args:
            url: Source URL
            title: Page title
            snippet: Search result snippet
            full_content: Full webpage content
            account_name: Company being researched
            industry: Company industry

        Returns:
            Signal object with LLM-analyzed metadata
        """
        # Check cache first
        cache_key = hash(url)
        if cache_key in self._analysis_cache:
            self.logger.debug("analysis_cache_hit", url=url[:50])
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

        try:
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
                timestamp=datetime.now(),
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
            self.logger.debug("analysis_completed", url=url[:50], confidence=signal.confidence)

            return signal

        except json.JSONDecodeError as e:
            self.logger.warning("llm_json_parse_failed", url=url[:50], error=str(e))
            raise
        except Exception as e:
            self.logger.error("llm_analysis_failed", url=url[:50], error=str(e))
            raise

    def get_complexity(self, state: ResearchState) -> int:
        """
        Get task complexity for model routing.

        GathererAgent now performs LLM analysis for each source,
        so complexity is moderate (uses Tier 1 Ollama).

        Args:
            state: Current research state

        Returns:
            Complexity score (1-10). Gatherer returns 4 (uses LLM for analysis)
        """
        return 4  # LLM analysis per source (Tier 1 Ollama)
