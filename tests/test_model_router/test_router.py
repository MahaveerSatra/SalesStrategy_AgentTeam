"""
Unit tests for model router.
Staff-level: Comprehensive testing with mocks.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from src.core.model_router import ModelRouter, ModelCache
from src.models.domain import ModelResponse
from src.core.exceptions import ModelError, ModelTimeoutError


class TestModelCache:
    """Test the model cache."""
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ModelCache(ttl_hours=1)
        result = cache.get("model1", "test prompt")
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0
    
    def test_cache_hit(self):
        """Test cache hit returns cached response."""
        cache = ModelCache(ttl_hours=1)
        
        # Store response
        response = ModelResponse(
            content="test response",
            model="model1",
            tokens_used=10,
            latency_ms=100.0,
            cached=False
        )
        cache.set("model1", "test prompt", response)
        
        # Retrieve
        cached = cache.get("model1", "test prompt")
        assert cached is not None
        assert cached.content == "test response"
        assert cached.cached is True
        assert cache.hits == 1
    
    def test_cache_expiry(self):
        """Test expired entries are not returned."""
        cache = ModelCache(ttl_hours=0)  # Instant expiry
        
        response = ModelResponse(
            content="test",
            model="m1",
            latency_ms=100.0
        )
        cache.set("m1", "prompt", response)
        
        # Should be expired immediately
        import time
        time.sleep(0.01)
        result = cache.get("m1", "prompt")
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ModelCache()
        
        # Generate some activity
        cache.get("m1", "p1")  # miss
        response = ModelResponse(content="test", model="m1", latency_ms=100.0)
        cache.set("m1", "p1", response)
        cache.get("m1", "p1")  # hit
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1


class TestModelRouter:
    """Test the model router."""
    
    @pytest.fixture
    def router(self):
        """Create a fresh router for each test."""
        return ModelRouter()
    
    def test_router_initialization(self, router):
        """Test router initializes correctly."""
        assert router.cache is not None
        assert router.request_counts == {}
        assert router.error_counts == {}
    
    @pytest.mark.asyncio
    async def test_complexity_routing(self, router):
        """Test that complexity determines model selection."""
        with patch.object(router, '_call_ollama_model', new_callable=AsyncMock) as mock_ollama:
            mock_ollama.return_value = ModelResponse(
                content="test",
                model="llama3.2:3b",
                latency_ms=100.0
            )
            
            # Low complexity should use local model
            await router.generate("test", complexity=2)
            mock_ollama.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_caching_works(self, router):
        """Test that caching prevents duplicate calls."""
        with patch.object(router, '_call_ollama_model', new_callable=AsyncMock) as mock_ollama:
            mock_ollama.return_value = ModelResponse(
                content="test response",
                model="llama3.2:3b",
                latency_ms=100.0
            )
            
            # First call
            result1 = await router.generate("test prompt", complexity=2)
            assert mock_ollama.call_count == 1
            assert result1.cached is False
            
            # Second call - should be cached
            result2 = await router.generate("test prompt", complexity=2)
            assert mock_ollama.call_count == 1  # Not called again
            assert result2.cached is True
            assert result2.content == result1.content
    
    @pytest.mark.asyncio
    async def test_error_handling(self, router):
        """Test error handling and retry logic."""
        with patch.object(router, '_call_ollama_model', new_callable=AsyncMock) as mock_ollama:
            mock_ollama.side_effect = ModelError("Test error")
            
            with pytest.raises(ModelError):
                await router.generate("test", complexity=2)
            
            # Should have incremented error count
            assert router.error_counts.get("llama3.2:3b", 0) > 0
    
    def test_get_metrics(self, router):
        """Test metrics collection."""
        router.request_counts = {"model1": 10, "model2": 5}
        router.error_counts = {"model1": 2}
        
        metrics = router.get_metrics()
        
        assert metrics["total_requests"] == 15
        assert metrics["total_errors"] == 2
        assert metrics["success_rate"] == pytest.approx(0.8667, rel=0.01)
        assert "cache_stats" in metrics


# Run tests with: pytest tests/test_model_router/test_router.py -v