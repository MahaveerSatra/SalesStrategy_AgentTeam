"""
Unit tests for data source base abstractions.
Tests DataSource and CachedDataSource classes.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.data_sources.base import DataSource, CachedDataSource
from src.core.exceptions import DataSourceError


class MockDataSource(DataSource):
    """Mock DataSource for testing."""

    def __init__(self):
        super().__init__()
        self.fetch_called = False

    async def fetch(self, **kwargs):
        self.fetch_called = True
        if kwargs.get("should_fail"):
            raise DataSourceError("Mock fetch failed")
        return {"data": "test", **kwargs}


class MockCachedDataSource(CachedDataSource):
    """Mock CachedDataSource for testing."""

    def __init__(self, cache_ttl_hours: int = 1):
        super().__init__(cache_ttl_hours=cache_ttl_hours)
        self.fetch_impl_call_count = 0

    async def _fetch_impl(self, **kwargs):
        self.fetch_impl_call_count += 1
        if kwargs.get("should_fail"):
            raise DataSourceError("Mock fetch failed")
        return {"data": "test", "call_count": self.fetch_impl_call_count, **kwargs}


class TestDataSource:
    """Test DataSource abstract base class."""

    @pytest.mark.asyncio
    async def test_fetch_basic(self):
        """Test basic fetch operation."""
        source = MockDataSource()
        result = await source.fetch(query="test")

        assert source.fetch_called
        assert result["data"] == "test"
        assert result["query"] == "test"

    @pytest.mark.asyncio
    async def test_fetch_with_fallback_primary_succeeds(self):
        """Test fetch_with_fallback when primary source succeeds."""
        primary = MockDataSource()
        fallback1 = MockDataSource()
        fallback2 = MockDataSource()

        result = await primary.fetch_with_fallback(
            fallback_sources=[fallback1, fallback2],
            query="test"
        )

        assert primary.fetch_called
        assert not fallback1.fetch_called
        assert not fallback2.fetch_called
        assert result["query"] == "test"

    @pytest.mark.asyncio
    async def test_fetch_with_fallback_primary_fails(self):
        """Test fetch_with_fallback when primary fails, first fallback succeeds."""
        primary = MockDataSource()
        fallback1 = MockDataSource()
        fallback2 = MockDataSource()

        result = await primary.fetch_with_fallback(
            fallback_sources=[fallback1, fallback2],
            should_fail=True,  # Primary will fail
            query="test"
        )

        assert primary.fetch_called
        assert fallback1.fetch_called
        assert not fallback2.fetch_called
        assert result["query"] == "test"

    @pytest.mark.asyncio
    async def test_fetch_with_fallback_all_fail(self):
        """Test fetch_with_fallback when all sources fail."""
        primary = MockDataSource()
        fallback1 = MockDataSource()
        fallback2 = MockDataSource()

        with pytest.raises(DataSourceError, match="All data sources failed"):
            await primary.fetch_with_fallback(
                fallback_sources=[fallback1, fallback2],
                should_fail=True  # All will fail
            )

        assert primary.fetch_called
        assert fallback1.fetch_called
        assert fallback2.fetch_called


class TestCachedDataSource:
    """Test CachedDataSource with TTL caching."""

    @pytest.mark.asyncio
    async def test_fetch_cache_miss(self):
        """Test fetch with cache miss."""
        source = MockCachedDataSource()

        result = await source.fetch(query="test")

        assert source.fetch_impl_call_count == 1
        assert result["data"] == "test"
        assert result["call_count"] == 1

        # Check cache stats
        stats = source.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_fetch_cache_hit(self):
        """Test fetch with cache hit on repeated query."""
        source = MockCachedDataSource()

        # First call - cache miss
        result1 = await source.fetch(query="test")
        assert source.fetch_impl_call_count == 1

        # Second call - cache hit (same params)
        result2 = await source.fetch(query="test")
        assert source.fetch_impl_call_count == 1  # Should not increase

        # Results should be identical
        assert result1 == result2

        # Check cache stats
        stats = source.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_fetch_different_params_cache_miss(self):
        """Test that different parameters create different cache entries."""
        source = MockCachedDataSource()

        result1 = await source.fetch(query="test1")
        result2 = await source.fetch(query="test2")

        assert source.fetch_impl_call_count == 2
        assert result1["query"] == "test1"
        assert result2["query"] == "test2"

        # Check cache stats
        stats = source.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        assert stats["size"] == 2

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test that cache entries expire after TTL."""
        source = MockCachedDataSource(cache_ttl_hours=1)

        # First call
        result1 = await source.fetch(query="test")
        assert source.fetch_impl_call_count == 1

        # Manually expire the cache entry
        cache_key = source._cache_key(query="test")
        data, timestamp = source._cache[cache_key]
        # Set timestamp to 2 hours ago (beyond TTL)
        source._cache[cache_key] = (data, datetime.now() - timedelta(hours=2))

        # Second call - should be cache miss due to expiration
        result2 = await source.fetch(query="test")
        assert source.fetch_impl_call_count == 2

        # Check cache stats
        stats = source.get_cache_stats()
        assert stats["hits"] == 0  # Expired entry doesn't count as hit
        assert stats["misses"] == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache clearing."""
        source = MockCachedDataSource()

        # Populate cache
        await source.fetch(query="test1")
        await source.fetch(query="test2")

        stats = source.get_cache_stats()
        assert stats["size"] == 2

        # Clear cache
        source.clear_cache()

        stats = source.get_cache_stats()
        assert stats["size"] == 0

        # Next fetch should be cache miss
        await source.fetch(query="test1")
        assert source.fetch_impl_call_count == 3  # Not served from cache

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation is consistent."""
        source = MockCachedDataSource()

        key1 = source._cache_key(query="test", limit=10)
        key2 = source._cache_key(query="test", limit=10)
        key3 = source._cache_key(query="test", limit=20)

        # Same params should generate same key
        assert key1 == key2

        # Different params should generate different key
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        source = MockCachedDataSource()

        # 2 unique queries, then repeat one
        await source.fetch(query="test1")
        await source.fetch(query="test2")
        await source.fetch(query="test1")  # Cache hit

        stats = source.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1/3)

    @pytest.mark.asyncio
    async def test_error_not_cached(self):
        """Test that errors are not cached."""
        source = MockCachedDataSource()

        # First call fails
        with pytest.raises(DataSourceError):
            await source.fetch(should_fail=True)

        # Second call should still try to fetch (error wasn't cached)
        with pytest.raises(DataSourceError):
            await source.fetch(should_fail=True)

        assert source.fetch_impl_call_count == 2

        # Cache should be empty
        stats = source.get_cache_stats()
        assert stats["size"] == 0
