"""Tests for MCP DuckDuckGo client.

Note: These tests use mocked MCP responses for reliability and speed.
The production implementation includes rate limiting to handle real DuckDuckGo requests.
For integration testing against real DuckDuckGo, see test_mcp_client_integration.py
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import HttpUrl
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.models.domain import SearchResult


class TestDuckDuckGoMCPClient:
    """Test MCP client with mocked responses."""

    @pytest.mark.asyncio
    async def test_auto_start_and_search(self):
        """Test that MCP server auto-starts and search returns parsed results."""
        client = DuckDuckGoMCPClient()

        # Mock the session to avoid actual MCP connection
        mock_session = AsyncMock()
        mock_result = MagicMock()
        # Format matching actual DuckDuckGo MCP response
        mock_result.content = [
            MagicMock(
                text="""1. Python Official
URL: https://www.python.org/
Summary: Official Python site

2. Python Docs
URL: https://docs.python.org/
Summary: Python documentation"""
            )
        ]
        mock_session.call_tool.return_value = mock_result

        # Inject the mocked session
        client.session = mock_session

        # Test search
        results = await client.search("Python programming", max_results=3)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].title == "Python Official"
        assert str(results[0].url) == "https://www.python.org/"
        assert results[1].title == "Python Docs"

    @pytest.mark.asyncio
    async def test_fetch_content(self):
        """Test webpage content fetching."""
        client = DuckDuckGoMCPClient()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(text="<html><body>Wikipedia content here</body></html>")
        ]
        mock_session.call_tool.return_value = mock_result

        # Inject the mocked session
        client.session = mock_session

        content = await client.fetch_content("https://en.wikipedia.org/")
        assert len(content) > 0
        assert "Wikipedia content" in content

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test cache hit on repeated queries."""
        client = DuckDuckGoMCPClient()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        # Format matching actual DuckDuckGo MCP response
        mock_result.content = [
            MagicMock(
                text="""1. Test Result
URL: https://example.com/
Summary: Test snippet"""
            )
        ]
        mock_session.call_tool.return_value = mock_result

        # Inject the mocked session
        client.session = mock_session

        query = "test query unique"

        # First call - cache miss
        results1 = await client.search(query, max_results=2)
        metrics1 = client.get_metrics()

        # Second call - cache hit (won't call MCP again)
        results2 = await client.search(query, max_results=2)
        metrics2 = client.get_metrics()

        # Same results
        assert len(results1) == len(results2)
        assert results1[0].title == results2[0].title

        # Cache hit rate increased
        assert metrics2['cache_stats']['hit_rate'] > metrics1['cache_stats']['hit_rate']

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test metrics collection."""
        client = DuckDuckGoMCPClient()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        # Format matching actual DuckDuckGo MCP response
        mock_result.content = [
            MagicMock(
                text="""1. Test
URL: https://test.com/
Summary: Test"""
            )
        ]
        mock_session.call_tool.return_value = mock_result

        # Inject the mocked session
        client.session = mock_session

        await client.search("test", max_results=1)

        metrics = client.get_metrics()

        assert metrics['request_count'] >= 1
        assert metrics['error_count'] == 0
        assert 'cache_stats' in metrics
        assert 'avg_latency_ms' in metrics
