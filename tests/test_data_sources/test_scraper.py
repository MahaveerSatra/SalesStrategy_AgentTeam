"""
Unit tests for web scraping utilities.
Tests RateLimiter, fetch_url, extract_text, extract_links, extract_metadata.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.data_sources.scraper import (
    RateLimiter,
    fetch_url,
    extract_text,
    extract_links,
    extract_metadata,
    select_elements,
)
from src.core.exceptions import DataSourceError, DataSourceTimeoutError


class TestRateLimiter:
    """Test RateLimiter for polite scraping."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test rate limiter delays requests."""
        limiter = RateLimiter(requests_per_second=2.0)  # 0.5s delay

        start = datetime.now()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = (datetime.now() - start).total_seconds()

        # Second request should be delayed by ~0.5s
        assert elapsed >= 0.4  # Allow some variance

    @pytest.mark.asyncio
    async def test_rate_limiter_first_request_immediate(self):
        """Test first request is not delayed."""
        limiter = RateLimiter(requests_per_second=1.0)

        start = datetime.now()
        await limiter.acquire()
        elapsed = (datetime.now() - start).total_seconds()

        # First request should be immediate
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_respects_delay(self):
        """Test rate limiter respects configured delay."""
        limiter = RateLimiter(requests_per_second=10.0)  # 0.1s delay

        start = datetime.now()
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = (datetime.now() - start).total_seconds()

        # Three requests should take ~0.2s (2 delays)
        assert 0.15 <= elapsed <= 0.35


class TestFetchUrl:
    """Test fetch_url with retries and error handling."""

    @pytest.mark.asyncio
    async def test_fetch_url_success(self):
        """Test successful URL fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await fetch_url("https://example.com")

            assert result == "<html><body>Test content</body></html>"
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_url_with_rate_limiter(self):
        """Test fetch_url respects rate limiter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Test</html>"
        mock_response.raise_for_status = MagicMock()

        limiter = RateLimiter(requests_per_second=10.0)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            start = datetime.now()
            await fetch_url("https://example.com", rate_limiter=limiter)
            await fetch_url("https://example.com", rate_limiter=limiter)
            elapsed = (datetime.now() - start).total_seconds()

            # Should respect rate limit
            assert elapsed >= 0.08

    @pytest.mark.asyncio
    async def test_fetch_url_timeout(self):
        """Test fetch_url raises DataSourceTimeoutError on timeout."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_class.return_value = mock_client

            with pytest.raises(DataSourceTimeoutError, match="Request timeout"):
                await fetch_url("https://example.com", timeout=1)

    @pytest.mark.asyncio
    async def test_fetch_url_http_error(self):
        """Test fetch_url raises DataSourceError on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=mock_response
            ))
            mock_client_class.return_value = mock_client

            with pytest.raises(DataSourceError, match="HTTP error"):
                await fetch_url("https://example.com")

    @pytest.mark.asyncio
    async def test_fetch_url_user_agent_rotation(self):
        """Test fetch_url includes random user agent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Test</html>"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await fetch_url("https://example.com")

            # Check that User-Agent header was set
            call_kwargs = mock_client.get.call_args.kwargs
            assert "headers" in call_kwargs
            assert "User-Agent" in call_kwargs["headers"]
            assert "Mozilla" in call_kwargs["headers"]["User-Agent"]


class TestExtractText:
    """Test extract_text for clean text extraction."""

    def test_extract_text_basic(self):
        """Test basic text extraction."""
        html = "<html><body><p>Hello world</p></body></html>"
        text = extract_text(html)

        assert "Hello world" in text
        assert "<p>" not in text
        assert "<html>" not in text

    def test_extract_text_removes_scripts(self):
        """Test that script tags are removed."""
        html = """
        <html>
            <body>
                <p>Content</p>
                <script>alert('test');</script>
            </body>
        </html>
        """
        text = extract_text(html)

        assert "Content" in text
        assert "alert" not in text
        assert "script" not in text

    def test_extract_text_removes_styles(self):
        """Test that style tags are removed."""
        html = """
        <html>
            <body>
                <p>Content</p>
                <style>body { color: red; }</style>
            </body>
        </html>
        """
        text = extract_text(html)

        assert "Content" in text
        assert "color" not in text
        assert "style" not in text

    def test_extract_text_cleans_whitespace(self):
        """Test whitespace cleaning."""
        html = """
        <html>
            <body>
                <p>Line 1</p>
                <p>Line 2</p>
            </body>
        </html>
        """
        text = extract_text(html)

        assert "Line 1" in text
        assert "Line 2" in text
        # Should not have excessive whitespace
        assert "  " not in text or text.count("  ") < 3


class TestExtractLinks:
    """Test extract_links for link extraction and normalization."""

    def test_extract_links_basic(self):
        """Test basic link extraction."""
        html = """
        <html>
            <body>
                <a href="https://example.com/page1">Link 1</a>
                <a href="https://example.com/page2">Link 2</a>
            </body>
        </html>
        """
        links = extract_links(html, "https://example.com")

        assert len(links) == 2
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links

    def test_extract_links_relative_urls(self):
        """Test relative URL normalization."""
        html = """
        <html>
            <body>
                <a href="/page1">Link 1</a>
                <a href="page2">Link 2</a>
            </body>
        </html>
        """
        links = extract_links(html, "https://example.com")

        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links

    def test_extract_links_skips_anchors(self):
        """Test that anchor-only links are skipped."""
        html = """
        <html>
            <body>
                <a href="#section1">Section 1</a>
                <a href="https://example.com/page">Real link</a>
            </body>
        </html>
        """
        links = extract_links(html, "https://example.com")

        assert len(links) == 1
        assert "https://example.com/page" in links
        assert "#section1" not in str(links)

    def test_extract_links_skips_javascript(self):
        """Test that javascript: links are skipped."""
        html = """
        <html>
            <body>
                <a href="javascript:void(0)">JS Link</a>
                <a href="https://example.com/page">Real link</a>
            </body>
        </html>
        """
        links = extract_links(html, "https://example.com")

        assert len(links) == 1
        assert "https://example.com/page" in links

    def test_extract_links_deduplication(self):
        """Test link deduplication."""
        html = """
        <html>
            <body>
                <a href="https://example.com/page">Link 1</a>
                <a href="https://example.com/page">Link 2</a>
            </body>
        </html>
        """
        links = extract_links(html, "https://example.com")

        assert len(links) == 1


class TestExtractMetadata:
    """Test extract_metadata for HTML metadata extraction."""

    def test_extract_metadata_title(self):
        """Test title extraction."""
        html = "<html><head><title>Test Page</title></head></html>"
        metadata = extract_metadata(html)

        assert metadata["title"] == "Test Page"

    def test_extract_metadata_description(self):
        """Test meta description extraction."""
        html = """
        <html>
            <head>
                <meta name="description" content="Test description">
            </head>
        </html>
        """
        metadata = extract_metadata(html)

        assert metadata["description"] == "Test description"

    def test_extract_metadata_og_tags(self):
        """Test Open Graph tag extraction."""
        html = """
        <html>
            <head>
                <meta property="og:title" content="OG Title">
                <meta property="og:description" content="OG Description">
            </head>
        </html>
        """
        metadata = extract_metadata(html)

        assert metadata["og_title"] == "OG Title"
        assert metadata["og_description"] == "OG Description"

    def test_extract_metadata_canonical(self):
        """Test canonical URL extraction."""
        html = """
        <html>
            <head>
                <link rel="canonical" href="https://example.com/canonical">
            </head>
        </html>
        """
        metadata = extract_metadata(html)

        assert metadata["canonical_url"] == "https://example.com/canonical"

    def test_extract_metadata_missing_fields(self):
        """Test metadata extraction with missing fields."""
        html = "<html><head></head></html>"
        metadata = extract_metadata(html)

        # Should return empty dict or dict with no keys, not raise error
        assert isinstance(metadata, dict)


class TestSelectElements:
    """Test select_elements for CSS selector-based extraction."""

    def test_select_elements_basic(self):
        """Test basic CSS selector."""
        html = """
        <html>
            <body>
                <div class="content">
                    <p>Paragraph 1</p>
                    <p>Paragraph 2</p>
                </div>
            </body>
        </html>
        """
        elements = select_elements(html, "p")

        assert len(elements) == 2

    def test_select_elements_class_selector(self):
        """Test class selector."""
        html = """
        <html>
            <body>
                <div class="content">Content div</div>
                <div class="other">Other div</div>
            </body>
        </html>
        """
        elements = select_elements(html, ".content")

        assert len(elements) == 1

    def test_select_elements_complex_selector(self):
        """Test complex CSS selector."""
        html = """
        <html>
            <body>
                <div class="container">
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                </div>
            </body>
        </html>
        """
        elements = select_elements(html, ".container ul li")

        assert len(elements) == 2

    def test_select_elements_no_match(self):
        """Test selector with no matches."""
        html = "<html><body><p>Test</p></body></html>"
        elements = select_elements(html, ".nonexistent")

        assert len(elements) == 0
