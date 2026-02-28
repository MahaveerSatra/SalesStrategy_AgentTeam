"""
SearchClient — 2-tier search facade.
Tier 1: DuckDuckGoMCPClient (primary, improved rate limiting)
Tier 2: TavilyClient (fallback — different server, bypasses DDG IP blocks)
"""
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.tavily_client import TavilyClient
from src.models.domain import SearchResult, NewsItem, CompanyInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SearchClient:
    """
    Drop-in replacement for DuckDuckGoMCPClient.
    Tier 1: MCP DDG. Tier 2: Tavily (when DDG returns empty or raises).
    """

    def __init__(self, **mcp_kwargs):
        self._mcp = DuckDuckGoMCPClient(**mcp_kwargs)
        self._tavily = TavilyClient()
        self.logger = logger.bind(component="search_client")

    async def __aenter__(self):
        await self._mcp.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._mcp.__aexit__(*args)

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            results = await self._mcp.search(query, max_results)
            if results:
                return results
        except Exception as e:
            self.logger.warning("search_ddg_failed_trying_tavily", query=query, error=str(e))
        self.logger.info("search_ddg_empty_trying_tavily", query=query)
        return await self._tavily.search(query, max_results)

    async def search_news(self, query: str, max_results: int = 5) -> list[NewsItem]:
        try:
            results = await self._mcp.search_news(query, max_results)
            if results:
                return results
        except Exception as e:
            self.logger.warning("search_news_ddg_failed_trying_tavily", query=query, error=str(e))
        self.logger.info("search_news_ddg_empty_trying_tavily", query=query)
        return await self._tavily.search_news(query, max_results)

    async def fetch_content(self, url: str) -> str:
        return await self._mcp.fetch_content(url)

    async def search_company_info(self, company_name: str) -> CompanyInfo:
        return await self._mcp.search_company_info(company_name)

    def get_metrics(self) -> dict:
        return self._mcp.get_metrics()
