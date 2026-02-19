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
from src.utils.json_parsing import extract_json_from_llm_response, JSONParseError
from src.models.state import ResearchState, Signal, ResearchDepth
from src.models.llm_schemas import (
    SourceAnalysis,
    SearchQueryGeneration,
    SalesSourceAnalysis,
    JobPostingAnalysis,
)
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.job_boards import JobBoardScraper
from src.core.model_router import ModelRouter
from src.core.exceptions import DataSourceError
from src.config import settings

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
    - Analyze EACH source with LOCAL LLM (Tier 1 Ollama, complexity=3)
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
        Collect and analyze data from multiple sources for sales intelligence.

        This method:
        1. Extracts ALL context fields from state (including seller_name)
        2. Builds MULTIPLE targeted search queries using LLM
        3. Determines max_results based on research_depth
        4. Fetches raw data from sources in parallel
        5. Analyzes EACH source with LLM for sales intelligence
        6. Creates rich Signal objects with buying signals

        Args:
            state: Current research state (modified in-place)
        """
        # Extract ALL context fields including seller context
        account = state["account_name"]
        industry = state.get("industry", "")
        region = state.get("region", "")
        user_context = state.get("user_context", "")
        seller_name = state.get("seller_name", "")  # type: ignore
        depth = state["research_depth"]

        self.logger.info(
            "gatherer_started",
            account=account,
            industry=industry,
            region=region,
            seller=seller_name,
            user_context=user_context[:50] if user_context else None,
            depth=depth.value
        )

        # Build multiple targeted search queries using LLM
        search_queries = await self._build_queries(
            account_name=account,
            industry=industry,
            seller_name=seller_name,
            user_context=user_context
        )

        # Build strategic news queries
        news_queries = self._build_news_queries(account, industry)

        # Determine max_results based on research_depth
        max_results_per_query = {
            ResearchDepth.QUICK: 3,
            ResearchDepth.STANDARD: 5,
            ResearchDepth.DEEP: 8
        }[depth]

        self.logger.info(
            "gatherer_fetching_data",
            search_queries=len(search_queries),
            news_queries=len(news_queries),
            max_results_per_query=max_results_per_query,
            depth=depth.value
        )

        # Extract company domain from state (now part of ResearchState)
        # If not provided, attempt to infer it before the parallel gather
        company_domain = state.get("company_domain") or ""
        if not company_domain:
            self.logger.info("domain_not_in_state_inferring", account=account)
            company_domain = await self._infer_company_domain(account)
            if company_domain:
                state["company_domain"] = company_domain  # Store for future use
                self.logger.info("domain_inferred_and_stored", account=account, domain=company_domain)
            else:
                self.logger.warning("domain_inference_failed_jobs_will_be_empty", account=account)

        # Execute multiple targeted searches in parallel
        try:
            # Build search tasks for each query
            search_tasks = [
                self._search_company_info(q["query"], max_results_per_query)
                for q in search_queries
            ]

            # Build news search tasks
            news_tasks = [
                self._search_news(nq, max_results=3)
                for nq in news_queries
            ]

            # Run all searches in parallel
            all_results = await asyncio.gather(
                *search_tasks,
                self._fetch_job_postings(account, company_domain),
                *news_tasks,
                return_exceptions=True
            )

            # Split results: search results, job postings, news items
            num_search = len(search_queries)
            num_news = len(news_queries)

            search_results_list = all_results[:num_search]
            job_postings = all_results[num_search]
            news_results_list = all_results[num_search + 1:]

        except Exception as e:
            self.logger.error("gatherer_parallel_fetch_failed", error=str(e))
            search_results_list = []
            job_postings = []
            news_results_list = []

        # Process search results from all queries -> Analyze EACH with LLM
        all_search_results = []
        seen_urls = set()  # Deduplicate across queries

        # Track search diagnostics
        search_success_count = 0
        search_empty_count = 0
        search_error_count = 0

        for idx, results in enumerate(search_results_list):
            query_info = search_queries[idx] if idx < len(search_queries) else {}
            if isinstance(results, Exception):
                search_error_count += 1
                self.logger.warning(
                    "search_query_failed",
                    category=query_info.get("category", "unknown"),
                    query=query_info.get("query", "unknown"),
                    error=str(results),
                    error_type=type(results).__name__
                )
                state["error_messages"].append(f"Search failed ({query_info.get('category', 'unknown')}): {results}")
                continue
            if results:
                search_success_count += 1
                for r in results:
                    url_str = str(r.url)
                    if url_str not in seen_urls:
                        seen_urls.add(url_str)
                        all_search_results.append(r)
            else:
                search_empty_count += 1
                self.logger.warning(
                    "search_query_empty",
                    category=query_info.get("category", "unknown"),
                    query=query_info.get("query", "unknown")
                )

        # Log search diagnostics summary
        self.logger.info(
            "search_diagnostics",
            total_queries=len(search_queries),
            success=search_success_count,
            empty=search_empty_count,
            errors=search_error_count,
            total_results=len(all_search_results)
        )

        self.logger.info("search_results_collected", total=len(all_search_results))

        # Analyze each search result with sales-focused LLM analysis
        if all_search_results:
            self.logger.info("analyzing_search_results", count=len(all_search_results))
            for result in all_search_results:
                try:
                    # Fetch full webpage content
                    full_content = await self.mcp_client.fetch_content(str(result.url))

                    # Analyze with sales-focused LLM (Tier 1 Ollama)
                    analyzed_signal = await self._analyze_source_with_llm(
                        url=str(result.url),
                        title=result.title,
                        snippet=result.snippet,
                        full_content=full_content,
                        account_name=account,
                        industry=industry,
                        seller_name=seller_name
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

            self.logger.info("search_signals_added", count=len(all_search_results))

        # Process job postings with LLM analysis for sales intelligence
        if isinstance(job_postings, Exception):
            self.logger.warning(
                "job_fetch_failed",
                error=str(job_postings),
                error_type=type(job_postings).__name__
            )
            state["error_messages"].append(f"Job posting collection failed: {job_postings}")
        elif job_postings:
            # Cap job postings to avoid imbalanced data (too many jobs vs news)
            original_count = len(job_postings)
            if original_count > settings.max_job_postings:
                job_postings = job_postings[:settings.max_job_postings]
                self.logger.info(
                    "job_postings_capped",
                    original=original_count,
                    capped_to=settings.max_job_postings
                )

            # Convert JobPosting objects to dicts for state storage
            state["job_postings"] = [jp.model_dump() for jp in job_postings]
            self.logger.info("job_postings_added", count=len(job_postings))

            # Analyze each job posting with LLM for sales intelligence
            for job in job_postings:
                try:
                    analyzed_signal = await self._analyze_job_posting_with_llm(
                        job=job.model_dump(),
                        account_name=account,
                        seller_name=seller_name,
                        user_context=user_context
                    )
                    state["signals"].append(analyzed_signal)
                except Exception as e:
                    self.logger.warning(
                        "job_analysis_failed",
                        job_title=job.title if hasattr(job, 'title') else "unknown",
                        error=str(e)
                    )
                    # Fallback: Create signal without LLM analysis
                    signal = Signal(
                        source="job_boards",
                        signal_type="hiring",
                        content=f"{job.title} - {job.company}",
                        timestamp=datetime.now(),
                        confidence=0.7,  # Lower confidence without analysis
                        metadata={
                            "location": job.location or "Unknown",
                            "technologies": job.technologies,
                            "url": str(job.url) if job.url else "",
                            "analysis_failed": True
                        }
                    )
                    state["signals"].append(signal)

        # Process news items from all news queries
        all_news_items = []
        seen_news_urls = set()

        # Track news diagnostics
        news_success_count = 0
        news_empty_count = 0
        news_error_count = 0

        for idx, news_results in enumerate(news_results_list):
            query = news_queries[idx] if idx < len(news_queries) else "unknown"
            if isinstance(news_results, Exception):
                news_error_count += 1
                self.logger.warning(
                    "news_query_failed",
                    query_idx=idx,
                    query=query,
                    error=str(news_results),
                    error_type=type(news_results).__name__
                )
                state["error_messages"].append(f"News search failed: {news_results}")
                continue
            if news_results:
                news_success_count += 1
                for news in news_results:
                    url_str = str(news.url) if news.url else news.title
                    if url_str not in seen_news_urls:
                        seen_news_urls.add(url_str)
                        all_news_items.append(news)
            else:
                news_empty_count += 1
                self.logger.warning(
                    "news_query_empty",
                    query_idx=idx,
                    query=query
                )

        # Log news diagnostics summary
        self.logger.info(
            "news_diagnostics",
            total_queries=len(news_queries),
            success=news_success_count,
            empty=news_empty_count,
            errors=news_error_count,
            total_results=len(all_news_items)
        )

        if all_news_items:
            # Convert NewsItem objects to dicts for state storage
            state["news_items"] = [news.model_dump() for news in all_news_items]
            self.logger.info("news_items_added", count=len(all_news_items))

            # Create signals from news items
            for news in all_news_items:
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

        Note: Domain inference is done before this method is called (in process()).
        If domain is still empty here, we skip job fetching gracefully.

        Args:
            company_name: Company name
            company_domain: Company domain (should be set by caller, empty means skip)

        Returns:
            List of JobPosting objects

        Raises:
            Exception: If fetch fails (caught by caller)
        """
        try:
            # If no domain, skip job fetching (inference was attempted earlier)
            if not company_domain:
                self.logger.debug("no_domain_skipping_jobs", company=company_name)
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

    async def _infer_company_domain(self, company_name: str) -> str | None:
        """
        Infer company domain from company name using web search verification.

        This method uses a multi-step approach:
        1. Generate candidate domain from company name (simple heuristic)
        2. Search the web for the company's official website
        3. Extract and validate the domain from search results

        This approach is more reliable than pure heuristics because:
        - Company names don't always match domains (Alphabet → abc.xyz, not alphabet.com)
        - Searches return the actual official website URL
        - Works for international companies with country-specific domains

        Args:
            company_name: Company name (e.g., "Boeing", "Microsoft")

        Returns:
            Company domain (e.g., "boeing.com") or None if inference fails
        """
        from urllib.parse import urlparse

        try:
            # Step 1: Search for the company's official website
            search_query = f"{company_name} official website"
            self.logger.debug("domain_inference_search", query=search_query)

            search_results = await self.mcp_client.search(search_query, max_results=5)

            if not search_results:
                self.logger.debug("domain_inference_no_results", company=company_name)
                # Fall back to simple heuristic
                return self._simple_domain_heuristic(company_name)

            # Step 2: Extract domains from search results
            candidate_domains: dict[str, int] = {}  # domain -> score

            for i, result in enumerate(search_results):
                try:
                    url = str(result.url)
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()

                    # Remove www. prefix
                    if domain.startswith('www.'):
                        domain = domain[4:]

                    # Skip common non-company domains
                    skip_domains = [
                        'wikipedia.org', 'linkedin.com', 'facebook.com', 'twitter.com',
                        'youtube.com', 'instagram.com', 'bloomberg.com', 'reuters.com',
                        'forbes.com', 'wsj.com', 'nytimes.com', 'crunchbase.com',
                        'glassdoor.com', 'indeed.com', 'yahoo.com', 'google.com',
                        'bing.com', 'duckduckgo.com', 'bbc.com', 'cnn.com'
                    ]

                    if any(skip in domain for skip in skip_domains):
                        continue

                    # Score based on position (higher for earlier results)
                    position_score = 5 - i

                    # Bonus score if company name appears in domain
                    company_clean = company_name.lower().replace(' ', '').replace('-', '')
                    domain_clean = domain.replace('.', '').replace('-', '')

                    name_match_score = 0
                    if company_clean in domain_clean:
                        name_match_score = 10  # Strong indicator

                    # Bonus for .com domains (often official)
                    com_score = 2 if domain.endswith('.com') else 0

                    total_score = position_score + name_match_score + com_score
                    candidate_domains[domain] = max(
                        candidate_domains.get(domain, 0),
                        total_score
                    )

                except Exception as e:
                    self.logger.debug("domain_parse_error", url=str(result.url), error=str(e))
                    continue

            # Step 3: Select best candidate
            if candidate_domains:
                # Sort by score (descending) and get best match
                best_domain = max(candidate_domains.items(), key=lambda x: x[1])[0]
                self.logger.debug(
                    "domain_inference_candidates",
                    company=company_name,
                    candidates=list(candidate_domains.keys())[:5],
                    selected=best_domain
                )
                return best_domain

            # Step 4: Fall back to simple heuristic if no good candidates
            self.logger.debug("domain_inference_fallback_to_heuristic", company=company_name)
            return self._simple_domain_heuristic(company_name)

        except Exception as e:
            self.logger.warning("domain_inference_failed", company=company_name, error=str(e))
            # Final fallback: simple heuristic
            return self._simple_domain_heuristic(company_name)

    def _simple_domain_heuristic(self, company_name: str) -> str:
        """
        Simple heuristic to convert company name to domain.

        This is a fallback when web search fails. It handles common patterns:
        - Removes common suffixes (Inc, Corp, LLC, Ltd, etc.)
        - Removes spaces and special characters
        - Adds .com suffix

        Args:
            company_name: Company name

        Returns:
            Inferred domain (e.g., "boeing.com")
        """
        name = company_name.lower().strip()

        # Remove common corporate suffixes
        suffixes = [
            ' incorporated', ' inc.', ' inc', ' corporation', ' corp.', ' corp',
            ' company', ' co.', ' co', ' limited', ' ltd.', ' ltd',
            ' llc', ' l.l.c.', ' plc', ' gmbh', ' ag', ' sa', ' nv',
            ' holdings', ' group', ' international', ' intl', ' worldwide'
        ]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        # Remove special characters and spaces, keep alphanumeric
        name = ''.join(c for c in name if c.isalnum())

        # Ensure we have something to return
        if not name:
            name = company_name.lower().replace(' ', '')[:20]

        return f"{name}.com"

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

    async def _build_queries(
        self,
        account_name: str,
        industry: str,
        seller_name: str,
        user_context: str
    ) -> list[dict]:
        """
        Generate multiple targeted search queries for sales research using LLM.

        Args:
            account_name: Company name to research
            industry: Industry vertical
            seller_name: Seller company (whose products we're trying to sell)
            user_context: Additional context from user

        Returns:
            List of query dicts with category, query, and priority
        """
        prompt = f"""Generate targeted search queries to research {account_name} as a potential customer for {seller_name}.

═══════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════
Account: {account_name}
Industry: {industry or "Not specified"}
Seller: {seller_name or "Not specified"}
User Context: {user_context or "None"}

═══════════════════════════════════════════════════════════════
QUERY CATEGORIES (generate 1 query per category)
═══════════════════════════════════════════════════════════════

1. **TECH STACK & TOOLS** (priority 1)
   Goal: Find what technologies they currently use
   Pattern: "{account_name} engineering tech stack" or "{account_name} software tools platform"

2. **HIRING SIGNALS** (priority 2)
   Goal: Find what skills/roles they're hiring for (indicates investment areas)
   Pattern: "{account_name} hiring engineers" or "{account_name} careers technology"

3. **STRATEGIC INITIATIVES** (priority 3)
   Goal: Find digital transformation, modernization, expansion projects
   Pattern: "{account_name} digital transformation" or "{account_name} technology investment"

4. **PARTNERSHIPS & VENDORS** (priority 4)
   Goal: Find existing technology partnerships (potential displacement opportunities)
   Pattern: "{account_name} partnership technology" or "{account_name} vendor software"

5. **CHALLENGES & PAIN POINTS** (priority 5)
   Goal: Find public statements about challenges they're solving
   Pattern: "{account_name} challenges engineering" or "{account_name} {industry} problems"

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return JSON:
{{
    "queries": [
        {{"category": "tech_stack", "query": "...", "priority": 1}},
        {{"category": "hiring", "query": "...", "priority": 2}},
        {{"category": "strategic", "query": "...", "priority": 3}},
        {{"category": "partnerships", "query": "...", "priority": 4}},
        {{"category": "challenges", "query": "...", "priority": 5}}
    ]
}}

RULES:
- Keep queries concise (3-6 words max after company name)
- Always start query with "{account_name}"
- DO NOT include special characters or quotes in queries
- Prioritize queries that would reveal {seller_name} sales opportunities
"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                temperature=0,
                use_cache=True,
                response_format=SearchQueryGeneration.model_json_schema()
            )

            try:
                result = SearchQueryGeneration.model_validate_json(response.content)
            except Exception as pydantic_error:
                self.logger.warning(
                    "query_generation_pydantic_failed",
                    error=str(pydantic_error)
                )
                raw_result = extract_json_from_llm_response(response.content)
                result = SearchQueryGeneration.model_validate(raw_result)

            # Convert to list of dicts
            queries = [
                {"category": q.category, "query": q.query, "priority": q.priority}
                for q in result.queries
            ]

            self.logger.info("search_queries_generated", count=len(queries))
            return queries

        except Exception as e:
            self.logger.warning("query_generation_failed", error=str(e))
            # Fallback to basic queries
            return [
                {"category": "general", "query": f"{account_name} company information {industry}", "priority": 1},
                {"category": "tech", "query": f"{account_name} technology stack", "priority": 2},
                {"category": "hiring", "query": f"{account_name} hiring jobs", "priority": 3},
            ]

    def _build_news_queries(self, account_name: str, industry: str) -> list[str]:
        """
        Generate strategic news queries for sales intelligence.

        Uses simple, broad queries that work well with DuckDuckGo's basic search.
        Avoids complex operators which aren't supported by the MCP.

        Args:
            account_name: Company name
            industry: Industry vertical

        Returns:
            List of news search queries (simple keywords, no operators)
        """
        # Keep queries simple - DuckDuckGo MCP doesn't support complex operators
        # Use broad terms that are likely to return results
        queries = [
            f"{account_name} latest news",                    # Most likely to return results
            f"{account_name} {industry} news 2026",           # Industry-specific news
            f"{account_name} technology announcement",        # Tech-focused
            f"{account_name} partnership deal",               # Business development
            f"{account_name} hiring expansion",               # Growth signals
        ]

        return queries

    async def _analyze_source_with_llm(
        self,
        url: str,
        title: str,
        snippet: str,
        full_content: str,
        account_name: str,
        industry: str,
        seller_name: str = ""
    ) -> Signal:
        """
        Use LLM to analyze source for SALES INTELLIGENCE.

        This is a Sales Research Analyst that extracts:
        - Source authority and reliability
        - Sales relevance (not just general relevance)
        - BUYING SIGNALS (technologies, hiring, budget, urgency, decision-makers)
        - Key facts and keywords
        - Confidence score based on sales value

        Args:
            url: Source URL
            title: Page title
            snippet: Search result snippet
            full_content: Full webpage content
            account_name: Company being researched (customer)
            industry: Company industry
            seller_name: Seller company (whose products we're trying to sell)

        Returns:
            Signal object with sales-focused metadata and buying signals
        """
        # Check cache first
        cache_key = hash(url)
        if cache_key in self._analysis_cache:
            self.logger.debug("analysis_cache_hit", url=url[:50])
            return self._analysis_cache[cache_key]

        prompt = f"""You are a Sales Research Analyst helping {seller_name or "the sales team"} sell to {account_name}.

═══════════════════════════════════════════════════════════════
SOURCE DATA
═══════════════════════════════════════════════════════════════
URL: {url}
Title: {title}
Snippet: {snippet}
Content (first 3000 chars): {full_content[:3000]}

═══════════════════════════════════════════════════════════════
ANALYSIS TASKS
═══════════════════════════════════════════════════════════════

1. **SOURCE AUTHORITY** (affects confidence)
   - Official company source (investor relations, press releases, careers) = HIGH
   - Reputable news (Reuters, industry publications) = MEDIUM-HIGH
   - Blog/third-party analysis = MEDIUM
   - Forum/social media = LOW

2. **SALES RELEVANCE** - Does this help sell {seller_name or "our"} products?
   - HIGH: Reveals needs, tech stack, hiring, initiatives we can address
   - MEDIUM: General company info useful for context
   - LOW: Irrelevant to sales conversation

3. **BUYING SIGNALS** - Extract any of these:
   - technologies: What technologies they use or mention needing
   - hiring_for: Roles being hired (indicates investment areas)
   - budget_indicators: Investment mentions ("allocated $X", "investing in...")
   - urgency_signals: Deadlines, acceleration, critical initiatives
   - decision_makers: Titles or names of technology decision-makers
   - pain_points: Challenges they've publicly mentioned
   - competitors_mentioned: Vendors/competitors (displacement opportunities)

4. **KEY FACTS** - Only VERIFIABLE facts, not speculation

═══════════════════════════════════════════════════════════════
CONFIDENCE SCORING
═══════════════════════════════════════════════════════════════
0.9-1.0: Official company source with recent, sales-relevant facts
0.7-0.8: Reputable news with sales-relevant insights
0.5-0.6: Third-party source with useful context
0.3-0.4: Tangentially relevant or dated information
0.0-0.2: Irrelevant or unreliable source

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return JSON:
{{
    "confidence": 0.85,
    "summary": "2-3 sentence summary focused on sales-relevant insights",
    "source_type": "official_company_site|news|blog|forum|other",
    "sales_relevance": "high|medium|low",
    "buying_signals": {{
        "technologies": ["tech1", "tech2"],
        "hiring_for": ["role1", "role2"],
        "budget_indicators": ["indicator1"],
        "urgency_signals": ["signal1"],
        "decision_makers": ["title1"],
        "pain_points": ["pain1"],
        "competitors_mentioned": ["competitor1"]
    }},
    "key_facts": ["fact1", "fact2"],
    "keywords": ["keyword1", "keyword2"]
}}"""

        try:
            # Use ModelRouter with complexity=3 (routes to Tier 1 LOCAL Ollama)
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama for cost efficiency
                temperature=0,
                use_cache=True,
                response_format=SalesSourceAnalysis.model_json_schema()
            )

            # Parse with Pydantic
            try:
                analysis = SalesSourceAnalysis.model_validate_json(response.content)
            except Exception as pydantic_error:
                self.logger.warning(
                    "pydantic_validation_failed_using_fallback",
                    url=url[:50],
                    error=str(pydantic_error)
                )
                raw_analysis = extract_json_from_llm_response(response.content)
                analysis = SalesSourceAnalysis.model_validate(raw_analysis)

            # Create Signal with sales-focused metadata
            signal = Signal(
                source="duckduckgo",
                signal_type="web_search",
                content=analysis.summary,
                timestamp=datetime.now(),
                confidence=analysis.confidence,
                metadata={
                    "url": url,
                    "title": title,
                    "source_type": analysis.source_type,
                    "sales_relevance": analysis.sales_relevance,
                    "buying_signals": {
                        "technologies": analysis.buying_signals.technologies,
                        "hiring_for": analysis.buying_signals.hiring_for,
                        "budget_indicators": analysis.buying_signals.budget_indicators,
                        "urgency_signals": analysis.buying_signals.urgency_signals,
                        "decision_makers": analysis.buying_signals.decision_makers,
                        "pain_points": analysis.buying_signals.pain_points,
                        "competitors_mentioned": analysis.buying_signals.competitors_mentioned,
                    },
                    "key_facts": analysis.key_facts,
                    "keywords": analysis.keywords,
                    "original_snippet": snippet
                }
            )

            # Cache for future lookups
            self._analysis_cache[cache_key] = signal
            self.logger.debug(
                "analysis_completed",
                url=url[:50],
                confidence=signal.confidence,
                sales_relevance=analysis.sales_relevance
            )

            return signal

        except (json.JSONDecodeError, JSONParseError) as e:
            self.logger.warning("llm_json_parse_failed", url=url[:50], error=str(e))
            raise
        except Exception as e:
            self.logger.error("llm_analysis_failed", url=url[:50], error=str(e))
            raise

    async def _analyze_job_posting_with_llm(
        self,
        job: dict,
        account_name: str,
        seller_name: str,
        user_context: str = ""
    ) -> Signal:
        """
        Use LLM to analyze job posting for sales intelligence.

        Extracts:
        - Technologies required/desired
        - Hiring urgency
        - Team size indicators
        - Seller relevance
        - Potential product champions

        Args:
            job: Job posting dict
            account_name: Company being researched
            seller_name: Seller company
            user_context: User's sales context/objectives

        Returns:
            Signal object with job analysis metadata
        """
        job_title = job.get("title", "Unknown")
        job_description = job.get("description", "")
        job_location = job.get("location", "Unknown")
        job_skills = job.get("required_skills", [])
        job_technologies = job.get("technologies", [])

        # Build context-aware prompt section
        context_section = ""
        if user_context:
            context_section = f"""
═══════════════════════════════════════════════════════════════
USER'S SALES CONTEXT (IMPORTANT - prioritize signals related to this)
═══════════════════════════════════════════════════════════════
{user_context}

PRIORITIZE: Focus your analysis on how this job relates to the user's stated objectives above.
"""

        prompt = f"""Analyze this job posting from {account_name} for sales intelligence relevant to {seller_name or "our products"}.
{context_section}
═══════════════════════════════════════════════════════════════
JOB POSTING
═══════════════════════════════════════════════════════════════
Title: {job_title}
Location: {job_location}
Required Skills: {", ".join(job_skills) if job_skills else "Not specified"}
Technologies: {", ".join(job_technologies) if job_technologies else "Not specified"}
Description: {job_description[:2000] if job_description else "Not provided"}

═══════════════════════════════════════════════════════════════
EXTRACT SALES INTELLIGENCE
═══════════════════════════════════════════════════════════════

1. **ROLE ANALYSIS**
   - What does this hiring indicate about company priorities?
   - Is this a new team/initiative or backfill?
   - Seniority level (entry/mid/senior/leadership)
   {"- Does this role align with user's stated focus area?" if user_context else ""}

2. **TECHNOLOGY SIGNALS**
   - What technologies are required vs preferred?
   - Any tools they want to adopt vs already use?
   {"- Are these technologies related to user's objectives (e.g., simulation, modeling, controls)?" if user_context else ""}

3. **URGENCY INDICATORS**
   - "Immediate start", "ASAP", "urgent" = high
   - Standard posting = medium
   - Evergreen/talent pool = low

4. **BUDGET SIGNALS**
   - Team size mentions ("join team of 50+")
   - Multiple similar roles = scaling investment

5. **{seller_name or "SELLER"} RELEVANCE**
   - Could {seller_name or "our"} products help this role/team?
   - Is this role a potential champion/user?
   {"- How well does this role match the user's target audience?" if user_context else ""}

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return JSON:
{{
    "confidence": 0.85,
    "summary": "What this hiring tells us about sales opportunity{" - especially related to user's context" if user_context else ""}",
    "technologies_required": ["tech1", "tech2"],
    "technologies_desired": ["tech3"],
    "urgency": "high|medium|low",
    "seniority": "entry|mid|senior|leadership",
    "team_indicators": "Team size/growth info if mentioned",
    "seller_relevance": "high|medium|low",
    "potential_champion": true,
    "sales_insight": "How {seller_name or "we"} could help this role/team{" based on user's objectives" if user_context else ""}"
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                temperature=0,
                use_cache=True,
                response_format=JobPostingAnalysis.model_json_schema()
            )

            try:
                analysis = JobPostingAnalysis.model_validate_json(response.content)
            except Exception as pydantic_error:
                self.logger.warning(
                    "job_analysis_pydantic_failed",
                    job_title=job_title,
                    error=str(pydantic_error)
                )
                raw_analysis = extract_json_from_llm_response(response.content)
                analysis = JobPostingAnalysis.model_validate(raw_analysis)

            # Create Signal with job analysis
            signal = Signal(
                source="job_boards",
                signal_type="hiring",
                content=analysis.summary,
                timestamp=datetime.now(),
                confidence=analysis.confidence,
                metadata={
                    "job_title": job_title,
                    "location": job_location,
                    "url": job.get("url", ""),
                    "technologies_required": analysis.technologies_required,
                    "technologies_desired": analysis.technologies_desired,
                    "urgency": analysis.urgency,
                    "seniority": analysis.seniority,
                    "team_indicators": analysis.team_indicators,
                    "seller_relevance": analysis.seller_relevance,
                    "potential_champion": analysis.potential_champion,
                    "sales_insight": analysis.sales_insight
                }
            )

            self.logger.debug(
                "job_analysis_completed",
                job_title=job_title,
                confidence=signal.confidence,
                seller_relevance=analysis.seller_relevance
            )

            return signal

        except (json.JSONDecodeError, JSONParseError) as e:
            self.logger.warning("job_llm_json_parse_failed", job_title=job_title, error=str(e))
            raise
        except Exception as e:
            self.logger.error("job_llm_analysis_failed", job_title=job_title, error=str(e))
            raise

    def get_complexity(self, state: ResearchState) -> int:
        """
        Get task complexity for model routing.

        GathererAgent performs LLM analysis for each source using LOCAL Ollama.
        Complexity=3 ensures routing to Tier 1 (local model) for zero-cost,
        fast, private analysis.

        Args:
            state: Current research state

        Returns:
            Complexity score (1-10). Gatherer returns 3 (uses LOCAL Ollama)
        """
        return 3  # LLM analysis per source (Tier 1: LOCAL Ollama)
