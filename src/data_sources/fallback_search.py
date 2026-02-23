"""
Fallback web search using direct HTTP requests.
Used when MCP DuckDuckGo server is unavailable.
"""

import asyncio
from urllib.parse import quote_plus
from typing import Any

import httpx
from bs4 import BeautifulSoup
import structlog

logger = structlog.get_logger(__name__)


class FallbackWebSearch:
    """
    Direct web search fallback when MCP is unavailable.

    Uses DuckDuckGo's HTML endpoint directly to fetch search results
    without requiring the MCP server infrastructure.
    """

    def __init__(self, timeout: float = 30.0):
        """
        Initialize fallback search.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
        self.logger = logger.bind(component="fallback_search")
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """
        Search DuckDuckGo directly via HTML endpoint.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search result dictionaries with title, snippet, url
        """
        encoded_query = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        self.logger.info("fallback_search_started", query=query, max_results=max_results)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._headers)
                response.raise_for_status()

                results = self._parse_results(response.text, max_results)

                self.logger.info(
                    "fallback_search_completed",
                    query=query,
                    results_count=len(results)
                )

                return results

        except httpx.TimeoutException:
            self.logger.warning("fallback_search_timeout", query=query)
            return []
        except httpx.HTTPStatusError as e:
            self.logger.warning(
                "fallback_search_http_error",
                query=query,
                status_code=e.response.status_code
            )
            return []
        except Exception as e:
            self.logger.error("fallback_search_error", query=query, error=str(e))
            return []

    def _parse_results(self, html: str, max_results: int) -> list[dict[str, Any]]:
        """
        Parse DuckDuckGo HTML search results.

        Args:
            html: Raw HTML response
            max_results: Maximum results to extract

        Returns:
            List of parsed search results
        """
        soup = BeautifulSoup(html, 'html.parser')
        results = []

        # DuckDuckGo HTML results are in .result divs
        for result in soup.select('.result')[:max_results]:
            try:
                # Extract title from result__a or result__title
                title_elem = result.select_one('.result__a')
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                # Extract URL
                url = title_elem.get('href', '')

                # Extract snippet
                snippet_elem = result.select_one('.result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''

                # Only add if we have meaningful content
                if title and (snippet or url):
                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'url': url,
                        'source': 'duckduckgo_fallback'
                    })

            except Exception as e:
                self.logger.debug("parse_result_failed", error=str(e))
                continue

        return results

    async def search_news(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """
        Search DuckDuckGo news directly.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of news result dictionaries
        """
        # Add "news" to the query for news-focused results
        news_query = f"{query} news recent"
        return await self.search(news_query, max_results)
