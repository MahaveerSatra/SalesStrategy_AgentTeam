"""
Tavily search client — Tier 2 fallback.
Purpose-built for LLM agents. Free tier: 1,000 searches/month, no credit card.
Registration: https://app.tavily.com/sign-up
Uses httpx (already in requirements) to call the Tavily REST API directly.
"""
import os

import httpx

from src.models.domain import SearchResult, NewsItem
from src.utils.logging import get_logger

logger = get_logger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"


class TavilyClient:
    """
    Tier 2 search fallback: Tavily REST API.
    Completely different infrastructure from DuckDuckGo — bypasses DDG IP blocks.
    Degrades gracefully (returns []) if TAVILY_API_KEY not set.
    """

    def __init__(self, timeout: float = 15.0):
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
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
        """Not supported — callers handle empty string gracefully."""
        return ""
