"""
Intelligent model router for cost/quality/latency optimization.
Staff-level: Multi-tier architecture with fallbacks and caching.

Includes proactive rate limiting to avoid hitting provider limits.
"""
import time
import asyncio
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


class RateLimitTracker:
    """
    Track request rates to proactively avoid hitting provider rate limits.

    This tracker uses a sliding window approach to track requests per minute
    and tokens per minute, allowing preemptive throttling before hitting
    actual provider limits.

    Attributes:
        rpm_limit: Maximum requests per minute (after buffer applied)
        tpm_limit: Maximum tokens per minute (after buffer applied)
    """

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        buffer: float = 0.8
    ):
        """
        Initialize rate limit tracker.

        Args:
            requests_per_minute: Provider's RPM limit
            tokens_per_minute: Provider's TPM limit
            buffer: Fraction of limit to use (default 0.8 = 80% to leave headroom)
        """
        self.rpm_limit = int(requests_per_minute * buffer)
        self.tpm_limit = int(tokens_per_minute * buffer)
        self._request_timestamps: list[float] = []
        self._token_counts: list[tuple[float, int]] = []
        self._window = 60.0  # 1 minute sliding window
        self.logger = logger.bind(component="rate_limiter")

    def _clean_old_entries(self) -> None:
        """Remove entries older than the sliding window."""
        cutoff = time.time() - self._window
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
        self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]

    def can_make_request(self, estimated_tokens: int = 500) -> bool:
        """
        Check if we can make a request without hitting limits.

        Args:
            estimated_tokens: Estimated tokens for this request

        Returns:
            True if request can proceed, False if we should wait
        """
        self._clean_old_entries()

        # Check RPM
        if len(self._request_timestamps) >= self.rpm_limit:
            self.logger.debug(
                "rpm_limit_reached",
                current=len(self._request_timestamps),
                limit=self.rpm_limit
            )
            return False

        # Check TPM
        current_tokens = sum(c for _, c in self._token_counts)
        if current_tokens + estimated_tokens > self.tpm_limit:
            self.logger.debug(
                "tpm_limit_reached",
                current=current_tokens,
                estimated=estimated_tokens,
                limit=self.tpm_limit
            )
            return False

        return True

    def record_request(self, tokens_used: int = 0) -> None:
        """
        Record a completed request for rate tracking.

        Args:
            tokens_used: Actual tokens used in the request
        """
        now = time.time()
        self._request_timestamps.append(now)
        if tokens_used > 0:
            self._token_counts.append((now, tokens_used))

    async def wait_if_needed(self, estimated_tokens: int = 500) -> None:
        """
        Wait if we're approaching rate limits.

        This method will sleep in 1-second increments until the rate limit
        window clears enough to allow the request.

        Args:
            estimated_tokens: Estimated tokens for the upcoming request
        """
        wait_count = 0
        max_wait = 60  # Maximum wait time in seconds

        while not self.can_make_request(estimated_tokens):
            wait_count += 1
            if wait_count > max_wait:
                self.logger.warning(
                    "rate_limit_wait_timeout",
                    waited_seconds=wait_count
                )
                break

            self.logger.debug(
                "rate_limit_waiting",
                wait_iteration=wait_count,
                reason="approaching_limit"
            )
            await asyncio.sleep(1.0)

        if wait_count > 0:
            self.logger.info(
                "rate_limit_wait_completed",
                waited_seconds=wait_count
            )

    def get_stats(self) -> dict[str, Any]:
        """Return current rate limit statistics."""
        self._clean_old_entries()
        current_tokens = sum(c for _, c in self._token_counts)

        return {
            "current_rpm": len(self._request_timestamps),
            "rpm_limit": self.rpm_limit,
            "current_tpm": current_tokens,
            "tpm_limit": self.tpm_limit,
            "rpm_utilization": len(self._request_timestamps) / self.rpm_limit if self.rpm_limit > 0 else 0,
            "tpm_utilization": current_tokens / self.tpm_limit if self.tpm_limit > 0 else 0,
        }


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

    Includes proactive rate limiting for external providers to avoid
    hitting rate limits and incurring errors.
    """

    def __init__(self):
        self.cache = ModelCache(ttl_hours=settings.cache_ttl_hours)
        self.request_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}
        self.logger = logger.bind(component="model_router")

        # Rate limiters per provider (lazy initialization)
        self._rate_limiters: dict[str, RateLimitTracker] = {}

        # Lazy-load model clients
        self._ollama_client = None
        self._litellm_available = False
        self._check_litellm()

        # Initialize rate limiters
        self._init_rate_limiters()

    def _init_rate_limiters(self) -> None:
        """Initialize rate limiters for each external provider."""
        self._rate_limiters["groq"] = RateLimitTracker(
            requests_per_minute=settings.groq_requests_per_minute,
            tokens_per_minute=settings.groq_tokens_per_minute,
            buffer=settings.rate_limit_buffer_percent
        )
        self._rate_limiters["together"] = RateLimitTracker(
            requests_per_minute=settings.together_requests_per_minute,
            tokens_per_minute=60000,  # Together has higher token limits
            buffer=settings.rate_limit_buffer_percent
        )
        self.logger.info(
            "rate_limiters_initialized",
            providers=list(self._rate_limiters.keys())
        )
    
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
        response_format: dict | None = None,
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
            response_format: JSON schema for structured output (Ollama only).
                            Use Pydantic's model_json_schema() to generate.
                            When provided, enforces the model to return valid JSON
                            conforming to the schema.
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
            if model.startswith("anthropic/"):
                response = await self._call_anthropic_model(
                    model, prompt, system_prompt, temperature, max_tokens, **kwargs
                )
            elif model.startswith("groq/") or model.startswith("together/"):
                response = await self._call_external_model(
                    model, prompt, system_prompt, temperature, max_tokens, **kwargs
                )
            else:
                # Local Ollama model - supports structured outputs
                response = await self._call_ollama_model(
                    model, prompt, system_prompt, temperature, max_tokens,
                    response_format=response_format, **kwargs
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
        response_format: dict | None = None,
        **kwargs
    ) -> ModelResponse:
        """Call local Ollama model.

        Args:
            model: Ollama model name
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            response_format: JSON schema for structured output.
                            When provided, Ollama enforces the response
                            to conform to this schema (guaranteed valid JSON).
            **kwargs: Additional parameters
        """
        start_time = time.time()

        try:
            ollama = self._get_ollama_client()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Build chat kwargs
            chat_kwargs = {
                "model": model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            # Add structured output format if provided
            # This enforces the model to return valid JSON conforming to the schema
            if response_format is not None:
                chat_kwargs["format"] = response_format
                self.logger.debug("using_structured_output", schema_keys=list(response_format.get("properties", {}).keys()))

            response = ollama.chat(**chat_kwargs)

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
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Detect rate limit errors with provider-specific patterns.

        This method checks for various rate limit error patterns from
        different providers (Groq, Together, generic HTTP 429, etc.)

        Args:
            error: The exception to check

        Returns:
            True if this is a rate limit error
        """
        error_str = str(error).lower()

        # Generic HTTP status patterns
        generic_patterns = [
            "rate limit", "rate_limit", "ratelimit",
            "too many requests", "429",
            "quota exceeded", "quota_exceeded",
            "requests per minute", "rpm",
            "tokens per minute", "tpm",
        ]

        # Groq-specific patterns
        groq_patterns = [
            "groq", "rate limit exceeded",
            "please try again", "request limit",
        ]

        # Together-specific patterns
        together_patterns = [
            "together", "request limit",
            "credit limit", "usage limit",
        ]

        # LiteLLM wrapper patterns
        litellm_patterns = [
            "rateerror", "modelratelimiterror",
            "litellm.exceptions.ratelimiterror",
        ]

        all_patterns = generic_patterns + groq_patterns + together_patterns + litellm_patterns

        return any(pattern in error_str for pattern in all_patterns)

    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string."""
        if "/" in model:
            return model.split("/")[0]
        return "unknown"

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
        """
        Call external model via litellm with proactive rate limiting.

        This method:
        1. Determines the provider from the model name
        2. Waits if we're approaching the provider's rate limits
        3. Makes the API call
        4. Records the request for rate tracking
        5. Handles errors with improved detection

        Args:
            model: Model name (e.g., "groq/llama-3.1-8b-instant")
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters

        Returns:
            ModelResponse with generated content
        """
        if not self._litellm_available:
            raise ModelError("litellm not available for external models")

        # Determine provider for rate limiting
        provider = self._get_provider_from_model(model)

        # Apply proactive rate limiting if we have a tracker for this provider
        if provider in self._rate_limiters:
            await self._rate_limiters[provider].wait_if_needed(estimated_tokens=max_tokens)

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
            tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None

            # Record successful request for rate tracking
            if provider in self._rate_limiters:
                self._rate_limiters[provider].record_request(tokens_used=tokens or max_tokens)

            return ModelResponse(
                content=content,
                model=model,
                tokens_used=tokens,
                latency_ms=latency_ms,
                cached=False
            )

        except Exception as e:
            error_str = str(e).lower()

            # Use improved error detection
            if "timeout" in error_str:
                raise ModelTimeoutError(f"Model request timed out: {e}")
            elif self._is_rate_limit_error(e):
                self.logger.warning(
                    "rate_limit_hit",
                    provider=provider,
                    model=model,
                    error=str(e)[:200]
                )
                raise ModelRateLimitError(f"Rate limit exceeded for {provider}: {e}")
            else:
                raise ModelError(f"External model error: {e}")

    async def _call_anthropic_model(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> ModelResponse:
        """
        Call Anthropic Claude with prompt caching on the system prompt.

        Sales research agents reuse the same system prompts (product catalog,
        analysis instructions) across many calls per session. Marking the system
        prompt with cache_control caches it on Anthropic's servers, reducing
        input token costs by ~90% on cache hits.

        Requires ANTHROPIC_API_KEY in environment. The cached block must be
        at least 1024 tokens — sales agent system prompts (product catalog +
        playbook instructions) comfortably exceed this threshold.
        """
        if not self._litellm_available:
            raise ModelError("litellm not available for Anthropic models")

        messages = []
        if system_prompt:
            # cache_control marks the system prompt for server-side caching.
            # This is the content that stays constant across calls — product
            # catalog descriptions, analysis instructions, playbook rules.
            # Only the user prompt (account-specific query) changes each time.
            messages.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }]
            })
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        try:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.request_timeout,
                **kwargs
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if hasattr(response, "usage") and response.usage else None

            return ModelResponse(
                content=content,
                model=model,
                tokens_used=tokens,
                latency_ms=latency_ms,
                cached=False,
            )

        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str:
                raise ModelTimeoutError(f"Anthropic request timed out: {e}")
            elif self._is_rate_limit_error(e):
                raise ModelRateLimitError(f"Anthropic rate limit exceeded: {e}")
            else:
                raise ModelError(f"Anthropic model error: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Return routing metrics including rate limit statistics."""
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())

        # Collect rate limit stats per provider
        rate_limit_stats = {
            provider: tracker.get_stats()
            for provider, tracker in self._rate_limiters.items()
        }

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
            "cache_stats": self.cache.get_stats(),
            "rate_limit_stats": rate_limit_stats
        }
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.cache.clear()


# Global router instance
router = ModelRouter()