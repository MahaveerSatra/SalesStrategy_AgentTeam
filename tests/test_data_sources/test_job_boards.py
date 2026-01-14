"""
Unit tests for job board scraping.
Tests CareerPageDetector and JobBoardScraper.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.data_sources.job_boards import CareerPageDetector, JobBoardScraper
from src.models.domain import JobPosting
from src.core.exceptions import DataSourceError


class TestCareerPageDetector:
    """Test CareerPageDetector for finding career pages."""

    @pytest.mark.asyncio
    async def test_find_career_page_common_pattern(self):
        """Test finding career page via common patterns."""
        with patch("src.data_sources.job_boards.fetch_url") as mock_fetch:
            mock_fetch.return_value = "<html>Careers</html>"

            url = await CareerPageDetector.find_career_page("example.com")

            assert url is not None
            assert "careers" in url or "jobs" in url
            assert "example.com" in url

    @pytest.mark.asyncio
    async def test_find_career_page_pattern_fails_uses_mcp(self):
        """Test fallback to MCP search when patterns fail."""
        with patch("src.data_sources.job_boards.fetch_url") as mock_fetch, \
             patch("src.data_sources.job_boards.DuckDuckGoMCPClient") as mock_mcp:

            # All pattern attempts fail
            mock_fetch.side_effect = DataSourceError("Not found")

            # MCP returns results
            mock_client = AsyncMock()
            mock_result = MagicMock()
            mock_result.url = "https://careers.example.com"
            mock_client.search.return_value = [mock_result]
            mock_mcp.return_value.__aenter__.return_value = mock_client

            url = await CareerPageDetector.find_career_page("example.com")

            assert url == "https://careers.example.com"
            mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_career_page_not_found(self):
        """Test when career page cannot be found."""
        with patch("src.data_sources.job_boards.fetch_url") as mock_fetch, \
             patch("src.data_sources.job_boards.DuckDuckGoMCPClient") as mock_mcp:

            # All pattern attempts fail
            mock_fetch.side_effect = DataSourceError("Not found")

            # MCP returns no relevant results
            mock_client = AsyncMock()
            mock_result = MagicMock()
            mock_result.url = "https://example.com/about"  # Not a careers page
            mock_client.search.return_value = [mock_result]
            mock_mcp.return_value.__aenter__.return_value = mock_client

            url = await CareerPageDetector.find_career_page("example.com")

            assert url is None

    @pytest.mark.asyncio
    async def test_find_career_page_mcp_fails(self):
        """Test when both patterns and MCP fail."""
        with patch("src.data_sources.job_boards.fetch_url") as mock_fetch, \
             patch("src.data_sources.job_boards.DuckDuckGoMCPClient") as mock_mcp:

            # All pattern attempts fail
            mock_fetch.side_effect = DataSourceError("Not found")

            # MCP raises error
            mock_mcp.return_value.__aenter__.side_effect = Exception("MCP error")

            url = await CareerPageDetector.find_career_page("example.com")

            assert url is None


class TestJobBoardScraper:
    """Test JobBoardScraper for job posting collection."""

    @pytest.fixture
    def scraper(self):
        """Provide JobBoardScraper instance."""
        return JobBoardScraper(cache_ttl_hours=1)

    @pytest.fixture
    def greenhouse_html(self):
        """Provide sample Greenhouse HTML."""
        return """
        <html>
            <body>
                <div class="opening" data-qa="opening">
                    <a href="/jobs/12345" class="opening-title">Software Engineer</a>
                    <div class="location">San Francisco, CA</div>
                </div>
                <div class="opening">
                    <a href="/jobs/67890">Senior Developer</a>
                    <div class="location">Remote</div>
                </div>
            </body>
        </html>
        """

    @pytest.fixture
    def lever_html(self):
        """Provide sample Lever HTML."""
        return """
        <html>
            <body>
                <div class="posting" data-qa="posting">
                    <a href="/jobs/abc123">
                        <h5 class="posting-title">Product Manager</h5>
                    </a>
                    <div class="posting-categories">
                        <span class="location">New York, NY</span>
                    </div>
                </div>
            </body>
        </html>
        """

    @pytest.fixture
    def generic_html(self):
        """Provide generic job page HTML."""
        return """
        <html>
            <body>
                <ul>
                    <li><a href="/jobs/1">Senior Software Engineer</a></li>
                    <li><a href="/jobs/2">Data Scientist with ML experience</a></li>
                    <li><a href="/about">About Us</a></li>
                </ul>
            </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_fetch_impl_career_page_not_found(self, scraper):
        """Test when career page is not found."""
        with patch("src.data_sources.job_boards.CareerPageDetector.find_career_page") as mock_find:
            mock_find.return_value = None

            jobs = await scraper._fetch_impl(
                company_name="Example Corp",
                company_domain="example.com"
            )

            assert jobs == []

    @pytest.mark.asyncio
    async def test_fetch_impl_fetch_fails(self, scraper):
        """Test when fetching career page fails."""
        with patch("src.data_sources.job_boards.CareerPageDetector.find_career_page") as mock_find, \
             patch("src.data_sources.job_boards.fetch_url") as mock_fetch:

            mock_find.return_value = "https://example.com/careers"
            mock_fetch.side_effect = DataSourceError("Fetch failed")

            jobs = await scraper._fetch_impl(
                company_name="Example Corp",
                company_domain="example.com"
            )

            assert jobs == []

    @pytest.mark.asyncio
    async def test_parse_greenhouse(self, scraper, greenhouse_html):
        """Test parsing Greenhouse job board format."""
        jobs = scraper._parse_job_listings(
            greenhouse_html,
            "https://example.com/careers",
            "Example Corp"
        )

        assert len(jobs) == 2
        assert all(isinstance(job, JobPosting) for job in jobs)

        # Check first job
        assert jobs[0].title == "Software Engineer"
        assert jobs[0].company == "Example Corp"
        assert jobs[0].location == "San Francisco, CA"
        assert jobs[0].confidence == 0.8

        # Check second job
        assert jobs[1].title == "Senior Developer"
        assert jobs[1].location == "Remote"

    @pytest.mark.asyncio
    async def test_parse_lever(self, scraper, lever_html):
        """Test parsing Lever job board format."""
        jobs = scraper._parse_job_listings(
            lever_html,
            "https://example.com/careers",
            "Example Corp"
        )

        assert len(jobs) == 1
        assert isinstance(jobs[0], JobPosting)
        assert jobs[0].title == "Product Manager"
        assert jobs[0].company == "Example Corp"
        assert "New York" in jobs[0].location
        assert jobs[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_parse_generic(self, scraper, generic_html):
        """Test parsing generic job page format."""
        jobs = scraper._parse_job_listings(
            generic_html,
            "https://example.com/careers",
            "Example Corp"
        )

        # Should find 2 job-like links (engineer, scientist)
        # "About Us" should be filtered out
        assert len(jobs) >= 2

        titles = [job.title for job in jobs]
        assert any("Engineer" in title for title in titles)
        assert any("Scientist" in title for title in titles)
        assert all(job.confidence == 0.5 for job in jobs)  # Generic parsing = lower confidence

    @pytest.mark.asyncio
    async def test_parse_job_listings_empty_html(self, scraper):
        """Test parsing with empty HTML."""
        jobs = scraper._parse_job_listings(
            "<html><body></body></html>",
            "https://example.com/careers",
            "Example Corp"
        )

        assert jobs == []

    @pytest.mark.asyncio
    async def test_parse_job_posting_success(self, scraper):
        """Test parsing individual job posting."""
        job_html = """
        <html>
            <head><title>Software Engineer - Example Corp</title></head>
            <body>
                <h1>Software Engineer</h1>
                <p>We are looking for a talented engineer...</p>
            </body>
        </html>
        """

        with patch("src.data_sources.job_boards.fetch_url") as mock_fetch:
            mock_fetch.return_value = job_html

            job = await scraper.parse_job_posting("https://example.com/jobs/123")

            assert job is not None
            assert isinstance(job, JobPosting)
            assert "Software Engineer" in job.title
            assert str(job.url) == "https://example.com/jobs/123"
            assert len(job.description) > 0

    @pytest.mark.asyncio
    async def test_parse_job_posting_fetch_fails(self, scraper):
        """Test parsing job posting when fetch fails."""
        with patch("src.data_sources.job_boards.fetch_url") as mock_fetch:
            mock_fetch.side_effect = DataSourceError("Fetch failed")

            job = await scraper.parse_job_posting("https://example.com/jobs/123")

            assert job is None

    @pytest.mark.asyncio
    async def test_caching_behavior(self, scraper):
        """Test that scraper caches results."""
        with patch("src.data_sources.job_boards.CareerPageDetector.find_career_page") as mock_find, \
             patch("src.data_sources.job_boards.fetch_url") as mock_fetch:

            mock_find.return_value = "https://example.com/careers"
            mock_fetch.return_value = "<html><body></body></html>"

            # First call
            jobs1 = await scraper.fetch(company_name="Example Corp", company_domain="example.com")

            # Second call with same params - should hit cache
            jobs2 = await scraper.fetch(company_name="Example Corp", company_domain="example.com")

            # Should only call find_career_page once (cache hit on second call)
            assert mock_find.call_count == 1

            # Check cache stats
            stats = scraper.get_cache_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_greenhouse_with_missing_fields(self, scraper):
        """Test Greenhouse parser handles missing fields gracefully."""
        html = """
        <html>
            <body>
                <div class="opening">
                    <a href="/jobs/123">Job Title</a>
                    <!-- No location -->
                </div>
            </body>
        </html>
        """

        jobs = scraper._parse_job_listings(html, "https://example.com", "Company")

        assert len(jobs) == 1
        assert jobs[0].location is None

    @pytest.mark.asyncio
    async def test_generic_deduplication(self, scraper):
        """Test generic parser deduplicates by title."""
        html = """
        <html>
            <body>
                <a href="/job1">Software Engineer</a>
                <a href="/job2">Software Engineer</a>
                <a href="/job3">Data Scientist</a>
            </body>
        </html>
        """

        jobs = scraper._parse_job_listings(html, "https://example.com", "Company")

        # Should have 2 unique jobs (deduplicated by title)
        titles = [job.title for job in jobs]
        assert len(set(titles)) == len(jobs)
