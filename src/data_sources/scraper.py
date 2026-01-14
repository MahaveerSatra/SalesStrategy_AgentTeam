"""
Web scraping utilities for HTML parsing and content extraction.
Provides rate limiting, retry logic, and robust error handling.
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.core.exceptions import DataSourceError, DataSourceTimeoutError
from src.utils.logging import get_logger

logger = get_logger(__name__)


# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class RateLimiter:
    """Rate limiter for polite scraping."""

    def __init__(self, requests_per_second: float = 1.0):
        self.delay = 1.0 / requests_per_second
        self.last_request: datetime | None = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            if self.last_request is not None:
                elapsed = (datetime.now() - self.last_request).total_seconds()
                if elapsed < self.delay:
                    wait_time = self.delay - elapsed
                    await asyncio.sleep(wait_time)

            self.last_request = datetime.now()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, DataSourceTimeoutError)),
    reraise=True
)
async def fetch_url(
    url: str,
    timeout: int = 10,
    rate_limiter: RateLimiter | None = None,
    headers: dict[str, str] | None = None
) -> str:
    """
    Fetch URL content with retries and rate limiting.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        rate_limiter: Optional rate limiter
        headers: Optional custom headers

    Returns:
        HTML content as string

    Raises:
        DataSourceError: If fetch fails after retries
    """
    if rate_limiter:
        await rate_limiter.acquire()

    # Default headers with random user agent
    if headers is None:
        headers = {}

    if "User-Agent" not in headers:
        headers["User-Agent"] = random.choice(USER_AGENTS)

    try:
        async with httpx.AsyncClient() as client:
            logger.debug("fetch_url_started", url=url[:100])
            response = await client.get(
                url,
                headers=headers,
                timeout=timeout,
                follow_redirects=True
            )
            response.raise_for_status()

            logger.info(
                "fetch_url_success",
                url=url[:100],
                status=response.status_code,
                size=len(response.text)
            )
            return response.text

    except httpx.TimeoutException as e:
        logger.error("fetch_url_timeout", url=url[:100], timeout=timeout)
        raise DataSourceTimeoutError(f"Request timeout for {url}: {e}")

    except httpx.HTTPStatusError as e:
        logger.error(
            "fetch_url_http_error",
            url=url[:100],
            status=e.response.status_code
        )
        raise DataSourceError(f"HTTP error {e.response.status_code} for {url}")

    except Exception as e:
        logger.error("fetch_url_failed", url=url[:100], error=str(e))
        raise DataSourceError(f"Failed to fetch {url}: {e}")


def extract_text(html: str, parser: str = "lxml") -> str:
    """
    Extract clean text from HTML.

    Args:
        html: HTML content
        parser: BeautifulSoup parser (default: lxml)

    Returns:
        Cleaned text content
    """
    try:
        soup = BeautifulSoup(html, parser)

        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    except Exception as e:
        logger.error("extract_text_failed", error=str(e))
        raise DataSourceError(f"Failed to extract text: {e}")


def extract_links(html: str, base_url: str, parser: str = "lxml") -> list[str]:
    """
    Extract and normalize all links from HTML.

    Args:
        html: HTML content
        base_url: Base URL for resolving relative links
        parser: BeautifulSoup parser (default: lxml)

    Returns:
        List of absolute URLs
    """
    try:
        soup = BeautifulSoup(html, parser)
        links = []

        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Skip empty, anchor-only, and javascript links
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)

            # Only include http/https links
            parsed = urlparse(absolute_url)
            if parsed.scheme in ("http", "https"):
                links.append(absolute_url)

        return list(set(links))  # Remove duplicates

    except Exception as e:
        logger.error("extract_links_failed", error=str(e))
        raise DataSourceError(f"Failed to extract links: {e}")


def extract_metadata(html: str, parser: str = "lxml") -> dict[str, Any]:
    """
    Extract metadata from HTML (title, description, etc.).

    Args:
        html: HTML content
        parser: BeautifulSoup parser (default: lxml)

    Returns:
        Dictionary of metadata
    """
    try:
        soup = BeautifulSoup(html, parser)
        metadata: dict[str, Any] = {}

        # Title
        if soup.title:
            metadata["title"] = soup.title.string.strip() if soup.title.string else None

        # Meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"].strip()

        # Open Graph metadata
        og_tags = soup.find_all("meta", property=lambda x: x and x.startswith("og:"))
        for tag in og_tags:
            prop = tag.get("property", "").replace("og:", "")
            content = tag.get("content")
            if prop and content:
                metadata[f"og_{prop}"] = content.strip()

        # Canonical URL
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            metadata["canonical_url"] = canonical["href"]

        return metadata

    except Exception as e:
        logger.error("extract_metadata_failed", error=str(e))
        return {}


def select_elements(
    html: str,
    selector: str,
    parser: str = "lxml"
) -> list[Any]:
    """
    Select elements using CSS selector.

    Args:
        html: HTML content
        selector: CSS selector
        parser: BeautifulSoup parser (default: lxml)

    Returns:
        List of matching elements
    """
    try:
        soup = BeautifulSoup(html, parser)
        return soup.select(selector)

    except Exception as e:
        logger.error(
            "select_elements_failed",
            selector=selector,
            error=str(e)
        )
        raise DataSourceError(f"Failed to select elements: {e}")
