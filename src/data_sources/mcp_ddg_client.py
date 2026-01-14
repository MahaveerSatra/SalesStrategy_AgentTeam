"""
DuckDuckGo MCP client for web search.
MCP-only implementation - no Python package fallback.
"""
import asyncio
import hashlib
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

    def __init__(self, cache_ttl_hours: int = 1, min_request_interval: float = 2.0):
        """
        Initialize MCP client.

        Args:
            cache_ttl_hours: Cache TTL in hours
            min_request_interval: Minimum seconds between requests (default: 1.0 to avoid bot detection)
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
        """
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

            # Parse results
            search_results = []
            if result and hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        # Parse formatted text response
                        # Format: "1. Title\n   URL: url\n   Summary: text\n\n2. ..."
                        text = item.text

                        # Split by numbered entries
                        import re
                        entries = re.split(r'\n\n\d+\. ', text)

                        for entry in entries:
                            if not entry.strip():
                                continue

                            lines = entry.split('\n')
                            title = ""
                            url = ""
                            snippet = ""

                            for line in lines:
                                line = line.strip()
                                if line.startswith('URL:'):
                                    url = line.replace('URL:', '').strip()
                                elif line.startswith('Summary:') or line.startswith('Description:'):
                                    snippet = line.split(':', 1)[1].strip()
                                elif not line.startswith(('URL:', 'Summary:', 'Description:')) and not url:
                                    # First non-URL/Summary line is likely the title
                                    if not title and line and not line.isdigit():
                                        title = line.lstrip('0123456789. ')

                            if title and url:
                                search_results.append(SearchResult(
                                    title=title,
                                    url=url,
                                    snippet=snippet or title,
                                    source="duckduckgo"
                                ))

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
        Search for news articles via MCP.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of NewsItem objects
        """
        # Use regular search with "news" appended
        search_results = await self.search(f"{query} news", max_results=max_results)

        news_items = []
        for result in search_results:
            news_items.append(NewsItem(
                title=result.title,
                source="duckduckgo",
                url=result.url,
                summary=result.snippet
            ))

        return news_items

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
