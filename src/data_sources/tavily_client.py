"""
Tavily search client — Tier 2 fallback.
Purpose-built for LLM agents. Free tier: 1,000 searches/month, no credit card.
Registration: https://app.tavily.com/sign-up
Uses httpx (already in requirements) to call the Tavily REST API directly.
"""
import httpx

from src.config import settings
from src.models.domain import SearchResult, NewsItem
from src.utils.logging import get_logger

logger = get_logger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class TavilyClient:
    """
    Tier 2 search fallback: Tavily REST API.
    Completely different infrastructure from DuckDuckGo — bypasses DDG IP blocks.
    Degrades gracefully (returns []) if TAVILY_API_KEY not set.
    """

    def __init__(self, timeout: float = 15.0):
        self.api_key = settings.tavily_api_key or ""
        self.timeout = timeout
        self.logger = logger.bind(component="tavily_client")
        if not self.api_key:
            self.logger.warning(
                "tavily_api_key_missing",
                hint="Set TAVILY_API_KEY in .env for Tier 2 search fallback",
            )

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search via Tavily REST API. Returns [] if API key not configured."""
        if not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    TAVILY_API_URL,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                        source="tavily",
                    )
                    for r in data.get("results", [])
                    if r.get("url")
                ]
                self.logger.info(
                    "tavily_search_success",
                    query=query,
                    result_count=len(results),
                )
                return results
        except Exception as e:
            self.logger.warning("tavily_search_failed", query=query, error=str(e))
            return []

    async def search_news(self, query: str, max_results: int = 5) -> list[NewsItem]:
        """Search news via Tavily using topic='news'."""
        if not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    TAVILY_API_URL,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "topic": "news",
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    NewsItem(
                        title=r.get("title", ""),
                        source="Tavily",
                        url=r.get("url", ""),
                        summary=r.get("content", ""),
                        relevance_score=r.get("score", 0.5),
                    )
                    for r in data.get("results", [])
                    if r.get("url")
                ]
        except Exception as e:
            self.logger.warning("tavily_news_failed", query=query, error=str(e))
            return []

    async def fetch_content(self, url: str) -> str:
        """Fetch page content via Tavily Extract API. Returns '' if unavailable."""
        if not self.api_key:
            return ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    TAVILY_EXTRACT_URL,
                    json={
                        "api_key": self.api_key,
                        "urls": [url],
                        "extract_depth": "advanced",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if results:
                    content = results[0].get("raw_content", "")
                    self.logger.info(
                        "tavily_extract_success", url=url, length=len(content)
                    )
                    return content
                failed = data.get("failed_results", [])
                if failed:
                    self.logger.warning(
                        "tavily_extract_url_failed",
                        url=url,
                        error=failed[0].get("error", "unknown"),
                    )
                return ""
        except Exception as e:
            self.logger.warning("tavily_extract_failed", url=url, error=str(e))
            return ""

    async def fetch_content_batch(self, urls: list[str]) -> dict[str, str]:
        """Batch fetch multiple URLs via Tavily Extract (max 20 per call).

        Returns a {url: raw_content} dict. Missing URLs (failed) are omitted.
        """
        if not self.api_key or not urls:
            return {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    TAVILY_EXTRACT_URL,
                    json={
                        "api_key": self.api_key,
                        "urls": urls,
                        "extract_depth": "advanced",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                result: dict[str, str] = {}
                for r in data.get("results", []):
                    result[r["url"]] = r.get("raw_content", "")
                for f in data.get("failed_results", []):
                    self.logger.warning(
                        "tavily_extract_url_failed",
                        url=f.get("url"),
                        error=f.get("error", "unknown"),
                    )
                self.logger.info(
                    "tavily_extract_batch_success",
                    requested=len(urls),
                    returned=len(result),
                )
                return result
        except Exception as e:
            self.logger.warning(
                "tavily_extract_batch_failed", url_count=len(urls), error=str(e)
            )
            return {}
