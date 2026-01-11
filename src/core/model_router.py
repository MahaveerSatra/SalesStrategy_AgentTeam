"""
Intelligent model router for cost/quality/latency optimization.
Staff-level: Multi-tier architecture with fallbacks and caching.
"""
import time
import hashlib
import json
from typing import Any, Literal
from datetime import datetime, timedelta
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from ..config import settings
from ..models.domain import ModelResponse
from ..core.exceptions import (
    ModelError, 
    ModelTimeoutError, 
    ModelRateLimitError
)

logger = structlog.get_logger(__name__)


class ModelCache:
    """Simple in-memory cache for model responses."""
    
    def __init__(self, ttl_hours: int = 24):
        self._cache: dict[str, tuple[ModelResponse, datetime]] = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.hits = 0
        self.misses = 0
    
    def _hash_key(self, model: str, prompt: str, **kwargs) -> str:
        """Create cache key from inputs."""
        key_data = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, model: str, prompt: str, **kwargs) -> ModelResponse | None:
        """Retrieve from cache if not expired."""
        key = self._hash_key(model, prompt, **kwargs)
                
        if key in self._cache:
            response, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                logger.debug("cache_hit", key=key[:8])
                # Exclude the old 'cached' value from the dump
                data = response.model_dump(exclude={'cached'})
                return ModelResponse(**data, cached=True)

                # return ModelResponse(**response.model_dump(), cached=True)
            else:
                # Expired
                del self._cache[key]
        
        self.misses += 1
        return None
    
    def set(self, model: str, prompt: str, response: ModelResponse, **kwargs) -> None:
        """Store in cache."""
        key = self._hash_key(model, prompt, **kwargs)
        self._cache[key] = (response, datetime.now())
        logger.debug("cache_set", key=key[:8])
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()
        logger.info("cache_cleared")
    
    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }


class ModelRouter:
    """
    Routes requests to appropriate model based on complexity.
    
    Tier 1 (Local): Fast, free, lower quality
    Tier 2 (External 8B): Medium speed/cost, good quality  
    Tier 3 (External 70B): Slower, free tier limited, best quality
    """
    
    def __init__(self):
        self.cache = ModelCache(ttl_hours=settings.cache_ttl_hours)
        self.request_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}
        self.logger = logger.bind(component="model_router")
        
        # Lazy-load model clients
        self._ollama_client = None
        self._litellm_available = False
        self._check_litellm()
    
    def _check_litellm(self) -> None:
        """Check if litellm is available and configured."""
        try:
            import litellm
            self._litellm_available = True
            
            # Configure API keys if available
            if settings.groq_api_key:
                import os
                os.environ["GROQ_API_KEY"] = settings.groq_api_key
                self.logger.info("groq_configured")
            
            if settings.together_api_key:
                import os
                os.environ["TOGETHER_API_KEY"] = settings.together_api_key
                self.logger.info("together_configured")
                
        except ImportError:
            self.logger.warning("litellm_not_available")
    
    def _get_ollama_client(self):
        """Lazy-load Ollama client."""
        if self._ollama_client is None:
            import ollama
            self._ollama_client = ollama
        return self._ollama_client
    
    async def generate(
        self,
        prompt: str,
        complexity: int = 5,
        model_override: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_cache: bool = True,
        **kwargs
    ) -> ModelResponse:
        """
        Generate response with automatic model routing.
        
        Args:
            prompt: User prompt
            complexity: Task complexity (1-10) for routing
            model_override: Force specific model
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            use_cache: Whether to use cache
            **kwargs: Additional model parameters
            
        Returns:
            ModelResponse with content and metadata
        """
        # Determine which model to use
        model = model_override or settings.get_model_for_complexity(complexity)
        
        self.logger.info(
            "generate_start",
            model=model,
            complexity=complexity,
            prompt_length=len(prompt)
        )
        
        # Check cache
        if use_cache and settings.enable_caching:
            cached = self.cache.get(
                model, prompt, 
                system_prompt=system_prompt,
                temperature=temperature
            )
            if cached:
                return cached
        
        # Route to appropriate backend
        try:
            if model.startswith("groq/") or model.startswith("together/"):
                response = await self._call_external_model(
                    model, prompt, system_prompt, temperature, max_tokens, **kwargs
                )
            else:
                # Local Ollama model
                response = await self._call_ollama_model(
                    model, prompt, system_prompt, temperature, max_tokens, **kwargs
                )
            
            # Cache successful response
            if use_cache and settings.enable_caching:
                self.cache.set(model, prompt, response, system_prompt=system_prompt,temperature=temperature)
            
            # Update metrics
            self.request_counts[model] = self.request_counts.get(model, 0) + 1
            
            return response
            
        except Exception as e:
            self.error_counts[model] = self.error_counts.get(model, 0) + 1
            self.logger.error(
                "generate_failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ModelTimeoutError, ModelRateLimitError))
    )
    async def _call_ollama_model(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> ModelResponse:
        """Call local Ollama model."""
        start_time = time.time()
        
        try:
            ollama = self._get_ollama_client()
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=response["message"]["content"],
                model=model,
                tokens_used=response.get("eval_count"),
                latency_ms=latency_ms,
                cached=False
            )
            
        except Exception as e:
            raise ModelError(f"Ollama error: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ModelTimeoutError, ModelRateLimitError))
    )
    async def _call_external_model(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> ModelResponse:
        """Call external model via litellm."""
        if not self._litellm_available:
            raise ModelError("litellm not available for external models")
        
        start_time = time.time()
        
        try:
            import litellm
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.request_timeout,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return ModelResponse(
                content=content,
                model=model,
                tokens_used=tokens,
                latency_ms=latency_ms,
                cached=False
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "timeout" in error_str:
                raise ModelTimeoutError(f"Model request timed out: {e}")
            elif "rate" in error_str or "limit" in error_str:
                raise ModelRateLimitError(f"Rate limit exceeded: {e}")
            else:
                raise ModelError(f"External model error: {e}")
    
    def get_metrics(self) -> dict[str, Any]:
        """Return routing metrics."""
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "success_rate": (
                (total_requests - total_errors) / total_requests
                if total_requests > 0
                else 0
            ),
            "requests_by_model": self.request_counts,
            "errors_by_model": self.error_counts,
            "cache_stats": self.cache.get_stats()
        }
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.cache.clear()


# Global router instance
router = ModelRouter()