"""
DuckDuckGo MCP client for web search.
MCP-only implementation - no Python package fallback.
"""
import asyncio
import hashlib
import random
from datetime import datetime, timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.exceptions import DataSourceError, DataSourceTimeoutError
from src.models.domain import SearchResult, NewsItem, CompanyInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MCPCache:
    """TTL-based cache for MCP responses."""

    def __init__(self, ttl_hours: int = 1):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.hits = 0
        self.misses = 0

    def _hash_key(self, method: str, **params) -> str:
        """Generate cache key from method and parameters."""
        content = f"{method}:{str(sorted(params.items()))}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, method: str, **params) -> Any | None:
        """Get cached result if available and not expired."""
        key = self._hash_key(method, **params)

        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                logger.debug("cache_hit", method=method, key=key[:8])
                return result
            else:
                # Expired
                del self._cache[key]

        self.misses += 1
        return None

    def set(self, method: str, result: Any, **params) -> None:
        """Store in cache."""
        key = self._hash_key(method, **params)
        self._cache[key] = (result, datetime.now())
        logger.debug("cache_set", method=method, key=key[:8])

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }


class DuckDuckGoMCPClient:
    """
    DuckDuckGo search via MCP protocol.
    MCP-only - no fallback to Python packages.
    Includes aggressive rate limiting to avoid bot detection.
    """

    def __init__(
        self,
        cache_ttl_hours: int = 1,
        min_request_interval: float = 3.5,
        max_concurrent_requests: int = 1
    ):
        """
        Initialize MCP client.

        Args:
            cache_ttl_hours: Cache TTL in hours
            min_request_interval: Minimum seconds between requests (default: 2.0 to avoid bot detection)
            max_concurrent_requests: Maximum number of concurrent requests (default: 2)
        """
        self.cache = MCPCache(ttl_hours=cache_ttl_hours)
        self.request_count = 0
        self.error_count = 0
        self._latencies: list[float] = []

        self.session: ClientSession | None = None
        self._exit_stack = None

        # Rate limiting to avoid bot detection
        self.min_request_interval = min_request_interval
        self._last_request_time: datetime | None = None

        # Semaphore to limit concurrent requests (DuckDuckGo rate limits aggressively)
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._request_lock = asyncio.Lock()  # For serializing rate limit checks

        self.logger = logger.bind(component="mcp_client", source="ddg")

    async def __aenter__(self):
        """Initialize MCP connection."""
        try:
            self.logger.info("mcp_connection_starting")

            # Create server parameters for uvx duckduckgo-mcp-server
            server_params = StdioServerParameters(
                command="uvx",
                args=["duckduckgo-mcp-server"],
                env=None
            )

            # Start stdio transport - store the context manager
            self._stdio_context = stdio_client(server_params)
            read, write = await self._stdio_context.__aenter__()

            # Initialize session - store the context manager
            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()

            # Initialize the connection
            await self.session.initialize()

            self.logger.info("mcp_connection_established")
            return self

        except Exception as e:
            self.logger.error("mcp_connection_failed", error=str(e))
            raise DataSourceError(f"Failed to connect to MCP server: {e}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup MCP connection."""
        try:
            if hasattr(self, '_session_context') and self._session_context:
                await self._session_context.__aexit__(exc_type, exc_val, exc_tb)
            if hasattr(self, '_stdio_context') and self._stdio_context:
                await self._stdio_context.__aexit__(exc_type, exc_val, exc_tb)
            self.logger.info("mcp_connection_closed")
        except Exception as e:
            self.logger.error("mcp_cleanup_failed", error=str(e))

    async def _wait_for_rate_limit(self) -> None:
        """
        Enforce rate limiting by waiting if needed.
        Ensures minimum interval between requests to avoid bot detection.
        Uses a lock to properly serialize rate limit checks across concurrent requests.
        """
        async with self._request_lock:
            if self._last_request_time is None:
                self._last_request_time = datetime.now()
                return

            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                wait_time = self.min_request_interval - elapsed
                self.logger.debug(
                    "rate_limit_wait",
                    elapsed=f"{elapsed:.2f}s",
                    wait_time=f"{wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
                # Add jitter to avoid predictable request patterns
                await asyncio.sleep(random.uniform(0.5, 2.0))

            # Update last request time after waiting
            self._last_request_time = datetime.now()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(DataSourceTimeoutError)
    )
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        Search DuckDuckGo via MCP.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            DataSourceError: If search fails
        """
        # Check cache first
        cached = self.cache.get("search", query=query, max_results=max_results)
        if cached is not None:
            return cached

        if not self.session:
            raise DataSourceError("MCP session not initialized. Use 'async with' context manager.")

        # Use semaphore to limit concurrent requests (DuckDuckGo rate limits aggressively)
        async with self._request_semaphore:
            try:
                # Enforce rate limiting before making request
                await self._wait_for_rate_limit()

                start_time = datetime.now()
                self.logger.info("search_started", query=query, max_results=max_results)

                # Call MCP tool
                result = await self.session.call_tool(
                    "search",
                    arguments={"query": query, "max_results": max_results}
                )

                # Parse results using flexible parsing
                search_results = []
                if result and hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            text = item.text
                            # Debug log raw response for troubleshooting
                            if not text or text.strip() == "":
                                self.logger.warning(
                                    "mcp_returned_empty_text",
                                    query=query,
                                    content_items=len(result.content) if result.content else 0
                                )
                            else:
                                self.logger.debug(
                                    "mcp_raw_response",
                                    query=query,
                                    text_preview=text[:200] if len(text) > 200 else text
                                )
                            parsed_results = self._parse_search_results_flexible(text)
                            search_results.extend(parsed_results)
                else:
                    # Log when MCP returns nothing
                    self.logger.warning(
                        "mcp_returned_no_content",
                        query=query,
                        has_result=bool(result),
                        has_content_attr=hasattr(result, 'content') if result else False
                    )

                # Track metrics
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self._latencies.append(latency)
                self.request_count += 1

                self.logger.info(
                    "search_completed",
                    query=query,
                    result_count=len(search_results),
                    latency_ms=latency
                )

                # Cache results
                self.cache.set("search", search_results, query=query, max_results=max_results)

                return search_results

            except Exception as e:
                self.error_count += 1
                self.logger.error("search_failed", query=query, error=str(e))
                raise DataSourceError(f"DuckDuckGo search failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(DataSourceTimeoutError)
    )
    async def fetch_content(self, url: str) -> str:
        """
        Fetch webpage content via MCP.

        Args:
            url: URL to fetch

        Returns:
            Page content as string

        Raises:
            DataSourceError: If fetch fails
        """
        # Check cache first
        cached = self.cache.get("fetch_content", url=url)
        if cached is not None:
            return cached

        if not self.session:
            raise DataSourceError("MCP session not initialized. Use 'async with' context manager.")

        try:
            # Enforce rate limiting before making request
            await self._wait_for_rate_limit()

            start_time = datetime.now()
            self.logger.info("fetch_started", url=url)

            # Call MCP tool
            result = await self.session.call_tool(
                "fetch_content",
                arguments={"url": url}
            )

            # Extract content
            content = ""
            if result and hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        content += item.text

            # Track metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self._latencies.append(latency)
            self.request_count += 1

            self.logger.info(
                "fetch_completed",
                url=url,
                content_length=len(content),
                latency_ms=latency
            )

            # Cache content
            self.cache.set("fetch_content", content, url=url)

            return content

        except Exception as e:
            self.error_count += 1
            self.logger.error("fetch_failed", url=url, error=str(e))
            raise DataSourceError(f"Failed to fetch {url}: {e}")

    async def search_news(self, query: str, max_results: int = 5) -> list[NewsItem]:
        """
        Search for news articles via MCP with progressive fallback strategy.

        DuckDuckGo MCP does NOT support site: operators or boolean OR.
        Uses simple query variations with news-related keywords and falls back
        to simpler queries if results are empty.

        Args:
            query: Search query (typically company name + topic)
            max_results: Maximum number of results

        Returns:
            List of NewsItem objects with relevance scoring
        """
        # Progressive query strategy - try simpler queries if complex ones fail
        # DuckDuckGo MCP only supports basic keyword search
        query_strategies = [
            f"{query} news",                    # "Boeing technology investment news"
            f"{query} announcement",            # "Boeing technology investment announcement"
            f"{query} press release",           # "Boeing technology investment press release"
            query,                              # Just the raw query as last resort
        ]

        search_results = []
        successful_query = None

        for news_query in query_strategies:
            self.logger.debug("news_search_trying", query=news_query)
            search_results = await self.search(news_query, max_results=max_results + 5)

            if search_results:
                successful_query = news_query
                self.logger.info(
                    "news_search_succeeded",
                    original_query=query,
                    successful_query=news_query,
                    result_count=len(search_results)
                )
                break

        if not search_results:
            self.logger.warning(
                "news_search_all_strategies_failed",
                query=query,
                strategies_tried=len(query_strategies)
            )

        news_items = []
        for result in search_results:
            url_str = str(result.url).lower()

            # Calculate relevance score based on source quality
            relevance_score = 0.5  # Default

            # High-quality news sources get higher scores
            if any(source in url_str for source in ["reuters", "bloomberg", "wsj"]):
                relevance_score = 0.9
            elif any(source in url_str for source in ["businesswire", "prnewswire", "globenewswire"]):
                relevance_score = 0.85  # Press releases are often high quality
            elif any(source in url_str for source in ["techcrunch", "zdnet", "theregister", "arstechnica"]):
                relevance_score = 0.8
            elif any(source in url_str for source in ["news", "press", "article", "blog"]):
                relevance_score = 0.6

            # Extract source name from URL
            source_name = self._extract_source_name(url_str)

            news_items.append(NewsItem(
                title=result.title,
                source=source_name,
                url=result.url,
                summary=result.snippet,
                relevance_score=relevance_score
            ))

        # Sort by relevance and limit results
        news_items.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        return news_items[:max_results]

    def _extract_source_name(self, url: str) -> str:
        """
        Extract human-readable source name from URL.

        Args:
            url: URL string

        Returns:
            Source name (e.g., "Reuters", "Bloomberg")
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')

            # Map common domains to proper names
            domain_names = {
                "reuters.com": "Reuters",
                "bloomberg.com": "Bloomberg",
                "wsj.com": "Wall Street Journal",
                "businesswire.com": "Business Wire",
                "prnewswire.com": "PR Newswire",
                "techcrunch.com": "TechCrunch",
                "zdnet.com": "ZDNet",
                "theregister.com": "The Register",
                "arstechnica.com": "Ars Technica",
                "wired.com": "Wired",
                "forbes.com": "Forbes",
                "ft.com": "Financial Times",
                "cnbc.com": "CNBC",
                "bbc.com": "BBC",
                "cnn.com": "CNN",
            }

            for domain_key, name in domain_names.items():
                if domain_key in domain:
                    return name

            # Default: capitalize first part of domain
            parts = domain.split('.')
            return parts[0].title() if parts else "DuckDuckGo"

        except Exception:
            return "DuckDuckGo"

    async def search_company_info(self, company_name: str) -> CompanyInfo:
        """
        Search for company information via MCP.

        Args:
            company_name: Name of company

        Returns:
            CompanyInfo object with basic info
        """
        # Search for company
        search_results = await self.search(f"{company_name} company info", max_results=5)

        # Basic extraction - can be enhanced with LLM later
        description = search_results[0].snippet if search_results else ""

        return CompanyInfo(
            name=company_name,
            industry="Unknown",  # Would need LLM to extract
            description=description
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return client metrics."""
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0,
            "avg_latency_ms": avg_latency,
            "cache_stats": self.cache.get_stats()
        }

    def _parse_search_results_flexible(self, text: str) -> list[SearchResult]:
        """
        Parse search results with multiple fallback strategies.

        This method is designed to be robust against format variations in MCP responses.
        It tries multiple parsing strategies in order of specificity:
        1. Numbered entries with various formats (1., 1), [1])
        2. URL-based extraction (finds all URLs and extracts context)
        3. Line-by-line parsing for simple formats

        Args:
            text: Raw text response from MCP

        Returns:
            List of SearchResult objects
        """
        import re
        results = []
        seen_urls = set()  # Deduplicate results

        # Strategy 1: Numbered entries with flexible format
        # Matches: "1. Title", "1) Title", "[1] Title", "1: Title"
        # Try to split by numbered patterns
        numbered_pattern = r'\n\n(?:\d+[\.\)\]:\s]+|\[\d+\]\s*)'
        entries = re.split(numbered_pattern, text)

        for entry in entries:
            if not entry.strip():
                continue

            parsed = self._parse_single_entry(entry)
            if parsed and parsed.url not in seen_urls:
                seen_urls.add(parsed.url)
                results.append(parsed)

        # Strategy 2: If no results from numbered parsing, try URL-based extraction
        if not results:
            url_pattern = re.compile(r'https?://[^\s<>"\']+')
            urls_found = url_pattern.findall(text)

            for url in urls_found:
                # Clean up URL (remove trailing punctuation)
                url = url.rstrip('.,;:!?')

                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Try to find context around the URL
                title = self._extract_title_near_url(text, url)
                snippet = self._extract_snippet_near_url(text, url)

                results.append(SearchResult(
                    title=title or self._title_from_url(url),
                    url=url,
                    snippet=snippet or title or "",
                    source="duckduckgo"
                ))

                # Limit to max 10 results from URL extraction
                if len(results) >= 10:
                    break

        return results

    def _parse_single_entry(self, entry: str) -> SearchResult | None:
        """
        Parse a single search result entry.

        Handles various formats:
        - Title\\nURL: url\\nSummary: text
        - Title\\nurl\\ntext
        - url\\nTitle\\ntext

        Args:
            entry: Single entry text

        Returns:
            SearchResult or None if parsing fails
        """
        import re

        lines = [line.strip() for line in entry.split('\n') if line.strip()]
        if not lines:
            return None

        title = ""
        url = ""
        snippet = ""

        for line in lines:
            # Check for labeled fields (case-insensitive)
            lower_line = line.lower()

            # URL detection - multiple patterns
            if lower_line.startswith('url:') or lower_line.startswith('link:'):
                url = re.sub(r'^(?:url|link)[:\s]+', '', line, flags=re.IGNORECASE).strip()
            elif re.match(r'^https?://', line):
                if not url:  # First URL found
                    url = line.strip()
            # Snippet/Summary detection
            elif any(lower_line.startswith(prefix) for prefix in ['summary:', 'description:', 'snippet:']):
                snippet = line.split(':', 1)[1].strip() if ':' in line else ""
            # Title detection - first substantial non-URL line
            elif not title and len(line) > 5 and not line.isdigit():
                # Clean up numbering prefixes
                title = re.sub(r'^[\d\.\)\]\[:]+\s*', '', line).strip()

        # Validate we have minimum required fields
        if title and url and url.startswith('http'):
            return SearchResult(
                title=title,
                url=url,
                snippet=snippet or title,
                source="duckduckgo"
            )

        return None

    def _extract_title_near_url(self, text: str, url: str) -> str | None:
        """
        Extract potential title text near a URL in the response.

        Args:
            text: Full response text
            url: URL to find context for

        Returns:
            Title string or None
        """
        # Find the URL position
        url_pos = text.find(url)
        if url_pos == -1:
            return None

        # Look at the 200 characters before the URL for a title
        start_pos = max(0, url_pos - 200)
        before_text = text[start_pos:url_pos]

        # Find the last sentence/line before the URL
        lines = [l.strip() for l in before_text.split('\n') if l.strip()]
        if lines:
            # Return the last non-empty line as potential title
            title = lines[-1]
            # Clean up common prefixes
            import re
            title = re.sub(r'^[\d\.\)\]\[:]+\s*', '', title).strip()
            if len(title) > 5:
                return title[:200]  # Limit length

        return None

    def _extract_snippet_near_url(self, text: str, url: str) -> str | None:
        """
        Extract potential snippet text near a URL in the response.

        Args:
            text: Full response text
            url: URL to find context for

        Returns:
            Snippet string or None
        """
        # Find the URL position
        url_pos = text.find(url)
        if url_pos == -1:
            return None

        # Look at the 500 characters after the URL for a snippet
        after_text = text[url_pos + len(url):url_pos + len(url) + 500]

        # Find first meaningful text after URL
        lines = [l.strip() for l in after_text.split('\n') if l.strip()]
        for line in lines:
            # Skip if it looks like another URL
            if line.startswith('http'):
                continue
            # Skip if it's just a label
            if line.lower() in ['summary:', 'description:', 'url:', 'link:']:
                continue
            # Check for labeled snippet
            if ':' in line and any(line.lower().startswith(p) for p in ['summary:', 'description:']):
                return line.split(':', 1)[1].strip()[:500]
            # Return first substantial line
            if len(line) > 20:
                return line[:500]

        return None

    def _title_from_url(self, url: str) -> str:
        """
        Generate a basic title from URL when no title is available.

        Args:
            url: URL to generate title from

        Returns:
            Title derived from URL path or domain
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)

            # Try to get meaningful path segment
            path_parts = [p for p in parsed.path.split('/') if p and p != 'index.html']
            if path_parts:
                # Use last meaningful path segment
                title = path_parts[-1].replace('-', ' ').replace('_', ' ')
                # Clean up file extensions
                title = title.rsplit('.', 1)[0] if '.' in title else title
                return title.title()[:100]

            # Fall back to domain
            domain = parsed.netloc.replace('www.', '')
            return domain

        except Exception:
            return url[:50]
