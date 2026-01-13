"""
Job board scraping and career page detection.
Collects job postings from company career pages.
"""
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from src.core.exceptions import DataSourceError
from src.data_sources.base import CachedDataSource
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.scraper import fetch_url, RateLimiter, extract_text, select_elements
from src.models.domain import JobPosting
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CareerPageDetector:
    """Detects career page URLs from company websites."""

    CAREER_KEYWORDS = [
        "careers", "jobs", "opportunities", "work-with-us",
        "join-us", "hiring", "employment", "job-openings"
    ]

    @staticmethod
    async def find_career_page(company_domain: str) -> str | None:
        """
        Find career page URL for a company domain.

        Args:
            company_domain: Company domain (e.g., "mathworks.com")

        Returns:
            Career page URL or None if not found
        """
        # Common career page patterns
        common_patterns = [
            f"https://{company_domain}/careers",
            f"https://{company_domain}/jobs",
            f"https://careers.{company_domain}",
            f"https://jobs.{company_domain}",
            f"https://{company_domain}/company/careers",
            f"https://{company_domain}/about/careers",
        ]

        # Try common patterns first
        rate_limiter = RateLimiter(requests_per_second=0.5)
        for url in common_patterns:
            try:
                await fetch_url(url, timeout=5, rate_limiter=rate_limiter)
                logger.info("career_page_found", url=url, method="pattern")
                return url
            except DataSourceError:
                continue

        # Search via MCP if patterns fail
        try:
            async with DuckDuckGoMCPClient() as client:
                query = f"{company_domain} careers jobs"
                results = await client.search(query, max_results=5)

                for result in results:
                    url_lower = str(result.url).lower()
                    # Check if URL contains career keywords and company domain
                    if company_domain in url_lower and any(
                        keyword in url_lower for keyword in CareerPageDetector.CAREER_KEYWORDS
                    ):
                        logger.info("career_page_found", url=result.url, method="search")
                        return str(result.url)

        except Exception as e:
            logger.warning("career_page_search_failed", error=str(e))

        return None


class JobBoardScraper(CachedDataSource):
    """Scrapes job postings from career pages."""

    def __init__(self, cache_ttl_hours: int = 6):
        super().__init__(cache_ttl_hours=cache_ttl_hours)
        self.rate_limiter = RateLimiter(requests_per_second=0.5)

    async def _fetch_impl(self, company_name: str, company_domain: str) -> list[JobPosting]:
        """
        Fetch all job postings for a company.

        Args:
            company_name: Company name
            company_domain: Company domain (e.g., "mathworks.com")

        Returns:
            List of job postings
        """
        # Find career page
        career_url = await CareerPageDetector.find_career_page(company_domain)
        if not career_url:
            self.logger.warning(
                "career_page_not_found",
                company=company_name,
                domain=company_domain
            )
            return []

        # Fetch career page HTML
        try:
            html = await fetch_url(career_url, rate_limiter=self.rate_limiter)
        except DataSourceError as e:
            self.logger.error(
                "career_page_fetch_failed",
                url=career_url,
                error=str(e)
            )
            return []

        # Parse job postings
        jobs = self._parse_job_listings(html, career_url, company_name)

        self.logger.info(
            "jobs_collected",
            company=company_name,
            count=len(jobs),
            url=career_url
        )

        return jobs

    def _parse_job_listings(
        self,
        html: str,
        base_url: str,
        company_name: str
    ) -> list[JobPosting]:
        """
        Parse job listings from career page HTML.

        Tries multiple parsing strategies:
        1. Common job board selectors (Greenhouse, Lever, etc.)
        2. Generic patterns (job title + link)
        3. Fallback to LLM extraction (future enhancement)

        Args:
            html: Career page HTML
            base_url: Base URL for resolving links
            company_name: Company name

        Returns:
            List of parsed job postings
        """
        soup = BeautifulSoup(html, "lxml")
        jobs = []

        # Strategy 1: Greenhouse job board
        jobs.extend(self._parse_greenhouse(soup, base_url, company_name))

        # Strategy 2: Lever job board
        if not jobs:
            jobs.extend(self._parse_lever(soup, base_url, company_name))

        # Strategy 3: Generic parsing
        if not jobs:
            jobs.extend(self._parse_generic(soup, base_url, company_name))

        return jobs

    def _parse_greenhouse(
        self,
        soup: BeautifulSoup,
        base_url: str,
        company_name: str
    ) -> list[JobPosting]:
        """Parse Greenhouse job board format."""
        jobs = []

        # Greenhouse uses specific class names
        job_elements = soup.select(".opening, [data-qa='opening']")

        for element in job_elements:
            try:
                # Title
                title_elem = element.select_one("a, .opening-title")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                # URL
                link = title_elem.get("href") if title_elem.name == "a" else None
                if link and not link.startswith("http"):
                    from urllib.parse import urljoin
                    link = urljoin(base_url, link)

                # Location
                location_elem = element.select_one(".location, [data-qa='opening-location']")
                location = location_elem.get_text(strip=True) if location_elem else None

                jobs.append(JobPosting(
                    title=title,
                    company=company_name,
                    description=title,  # Placeholder
                    location=location,
                    url=link,
                    confidence=0.8
                ))

            except Exception as e:
                self.logger.debug("parse_greenhouse_job_failed", error=str(e))
                continue

        if jobs:
            self.logger.info("parsed_greenhouse", count=len(jobs))

        return jobs

    def _parse_lever(
        self,
        soup: BeautifulSoup,
        base_url: str,
        company_name: str
    ) -> list[JobPosting]:
        """Parse Lever job board format."""
        jobs = []

        # Lever uses specific class names
        job_elements = soup.select(".posting, [data-qa='posting']")

        for element in job_elements:
            try:
                title_elem = element.select_one("a, .posting-title, h5")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                link = title_elem.get("href") if title_elem.name == "a" else None
                if link and not link.startswith("http"):
                    from urllib.parse import urljoin
                    link = urljoin(base_url, link)

                location_elem = element.select_one(".location, .posting-categories")
                location = location_elem.get_text(strip=True) if location_elem else None

                jobs.append(JobPosting(
                    title=title,
                    company=company_name,
                    description=title,
                    location=location,
                    url=link,
                    confidence=0.8
                ))

            except Exception as e:
                self.logger.debug("parse_lever_job_failed", error=str(e))
                continue

        if jobs:
            self.logger.info("parsed_lever", count=len(jobs))

        return jobs

    def _parse_generic(
        self,
        soup: BeautifulSoup,
        base_url: str,
        company_name: str
    ) -> list[JobPosting]:
        """Generic job parsing for custom career pages."""
        jobs = []

        # Look for common patterns: links with job-like text
        job_keywords = ["engineer", "developer", "manager", "analyst", "scientist", "specialist"]

        links = soup.find_all("a", href=True)

        for link in links:
            text = link.get_text(strip=True)
            text_lower = text.lower()

            # Check if text looks like a job title
            if any(keyword in text_lower for keyword in job_keywords) and len(text) > 10:
                href = link["href"]
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    href = urljoin(base_url, href)

                jobs.append(JobPosting(
                    title=text,
                    company=company_name,
                    description=text,
                    url=href,
                    confidence=0.5  # Lower confidence for generic parsing
                ))

        # Deduplicate by title
        seen_titles = set()
        unique_jobs = []
        for job in jobs:
            if job.title not in seen_titles:
                seen_titles.add(job.title)
                unique_jobs.append(job)

        if unique_jobs:
            self.logger.info("parsed_generic", count=len(unique_jobs))

        return unique_jobs

    async def parse_job_posting(self, url: str) -> JobPosting | None:
        """
        Parse individual job posting from URL.

        Args:
            url: Job posting URL

        Returns:
            Parsed job posting or None if failed
        """
        try:
            html = await fetch_url(url, rate_limiter=self.rate_limiter)
            text = extract_text(html)

            # Extract title from page
            soup = BeautifulSoup(html, "lxml")
            title = None
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else None

            # Basic extraction (can be enhanced with LLM)
            return JobPosting(
                title=title or "Unknown Position",
                company="Unknown",
                description=text[:1000],  # First 1000 chars
                url=url,
                confidence=0.6
            )

        except Exception as e:
            self.logger.error("parse_job_posting_failed", url=url, error=str(e))
            return None
