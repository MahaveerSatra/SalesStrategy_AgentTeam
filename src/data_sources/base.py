"""
Base abstractions for data sources.
Provides standard interface and caching support for all data sources.
"""
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

from src.core.exceptions import DataSourceError
from src.utils.logging import get_logger


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def fetch(self, **kwargs) -> Any:
        """
        Fetch data from the source.

        Args:
            **kwargs: Query parameters specific to the data source

        Returns:
            Data from the source (type varies by implementation)

        Raises:
            DataSourceError: If fetch fails
        """
        pass

    async def fetch_with_fallback(
        self,
        fallback_sources: list["DataSource"],
        **kwargs
    ) -> Any:
        """
        Fetch with fallback to alternative sources.

        Args:
            fallback_sources: List of fallback data sources
            **kwargs: Query parameters

        Returns:
            Data from primary or fallback source

        Raises:
            DataSourceError: If all sources fail
        """
        # Try primary source
        try:
            return await self.fetch(**kwargs)
        except DataSourceError as e:
            self.logger.warning(
                "primary_source_failed",
                error=str(e),
                fallback_count=len(fallback_sources)
            )

        # Try fallback sources
        for i, fallback in enumerate(fallback_sources):
            try:
                self.logger.info("trying_fallback", fallback_index=i)
                return await fallback.fetch(**kwargs)
            except DataSourceError as e:
                self.logger.warning(
                    "fallback_failed",
                    fallback_index=i,
                    error=str(e)
                )
                continue

        raise DataSourceError("All data sources failed")


class CachedDataSource(DataSource):
    """Data source with built-in TTL caching."""

    def __init__(self, cache_ttl_hours: int = 1):
        super().__init__()
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, **kwargs) -> str:
        """Generate cache key from query parameters."""
        content = str(sorted(kwargs.items()))
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Any | None:
        """Get cached value if still valid."""
        if cache_key not in self._cache:
            self.cache_misses += 1
            return None

        data, timestamp = self._cache[cache_key]
        if datetime.now() - timestamp < self.cache_ttl:
            self.cache_hits += 1
            self.logger.debug("cache_hit", key=cache_key[:8])
            return data

        # Expired
        del self._cache[cache_key]
        self.cache_misses += 1
        return None

    def _set_cached(self, cache_key: str, data: Any) -> None:
        """Store data in cache."""
        self._cache[cache_key] = (data, datetime.now())
        self.logger.debug("cache_set", key=cache_key[:8])

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self.logger.info("cache_cleared")

    async def fetch(self, **kwargs) -> Any:
        """
        Fetch with caching.

        Checks cache first, falls back to _fetch_impl if miss.
        Subclasses implement _fetch_impl instead of fetch.
        """
        cache_key = self._cache_key(**kwargs)

        # Check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Fetch fresh data
        data = await self._fetch_impl(**kwargs)

        # Cache result
        self._set_cached(cache_key, data)

        return data

    @abstractmethod
    async def _fetch_impl(self, **kwargs) -> Any:
        """
        Implementation of data fetching logic.

        Subclasses override this instead of fetch().
        Caching is handled automatically by fetch().
        """
        pass
