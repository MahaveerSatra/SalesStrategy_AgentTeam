"""Tests for MCP DuckDuckGo client."""
import pytest
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.models.domain import SearchResult


class TestDuckDuckGoMCPClient:
    """Test MCP client with real server connection."""

    @pytest.mark.asyncio
    async def test_auto_start_and_search(self):
        """Test that MCP server auto-starts and search works."""
        async with DuckDuckGoMCPClient() as client:
            # Search should work without manual server start
            results = await client.search("Python programming", max_results=3)

            assert len(results) > 0
            assert all(isinstance(r, SearchResult) for r in results)
            assert all(r.title for r in results)
            assert all(r.url for r in results)

    @pytest.mark.asyncio
    async def test_fetch_content(self):
        """Test webpage content fetching."""
        async with DuckDuckGoMCPClient() as client:
            results = await client.search("Wikipedia", max_results=1)
            assert len(results) > 0

            content = await client.fetch_content(str(results[0].url))
            assert len(content) > 0
            assert isinstance(content, str)

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test cache hit on repeated queries."""
        async with DuckDuckGoMCPClient() as client:
            query = "test query unique"

            # First call - cache miss
            results1 = await client.search(query, max_results=2)
            metrics1 = client.get_metrics()

            # Second call - cache hit
            results2 = await client.search(query, max_results=2)
            metrics2 = client.get_metrics()

            # Same results
            assert len(results1) == len(results2)

            # Cache hit rate increased
            assert metrics2['cache_stats']['hit_rate'] > metrics1['cache_stats']['hit_rate']

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test metrics collection."""
        async with DuckDuckGoMCPClient() as client:
            await client.search("test", max_results=1)

            metrics = client.get_metrics()

            assert metrics['request_count'] >= 1
            assert metrics['error_count'] == 0
            assert 'cache_stats' in metrics
            assert 'avg_latency_ms' in metrics
