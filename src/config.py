"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    groq_api_key: str | None = Field(default=None, description="Groq API key")
    together_api_key: str | None = Field(default=None, description="Together.ai API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key for Claude models with prompt caching")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key for search and content extraction")

    # Model Configuration
    local_model: str = Field(default="llama3.2:3b", description="Local Ollama model")
    smart_model: str = Field(
        default="groq/llama-3.1-8b-instant",
        description="External model for complex reasoning"
    )
    advanced_model: str = Field(
        default="groq/openai/gpt-oss-20b",
        description="External model for advanced reasoning. Override via ADVANCED_MODEL in .env"
    )
    claude_model: str = Field(
        default="anthropic/claude-haiku-4-5-20251001",
        description="Claude model used for Anthropic prompt caching (opt-in via model_override)"
    )
    
    # Ollama Settings
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # Model Routing Thresholds
    complexity_threshold_medium: int = Field(
        default=3,
        description="Tasks <= this use local model"
    )
    complexity_threshold_high: int = Field(
        default=7,
        description="Tasks <= this use smart model, else advanced"
    )
    
    # Caching
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format"
    )
    
    # Performance
    max_parallel_tasks: int = Field(default=4, description="Max concurrent tasks")
    request_timeout: int = Field(default=30, description="HTTP request timeout")
    max_retries: int = Field(default=3, description="Max retry attempts")
    
    # Data Sources
    max_search_results: int = Field(default=10, description="Max search results per query")
    max_job_postings: int = Field(default=30, description="Max job postings to analyze per research")

    # Rate Limiting (for external LLM providers)
    groq_requests_per_minute: int = Field(default=30, description="Groq rate limit per minute")
    groq_tokens_per_minute: int = Field(default=6000, description="Groq token limit per minute")
    together_requests_per_minute: int = Field(default=60, description="Together rate limit per minute")
    rate_limit_buffer_percent: float = Field(
        default=0.8,
        description="Use only this fraction of rate limit to avoid hitting limits"
    )

    # Report Formatting
    report_max_evidence_signals: int = Field(default=5, description="Max evidence signals per opportunity")
    report_evidence_char_limit: int = Field(default=200, description="Character limit per evidence signal")
    report_rationale_char_limit: int = Field(default=300, description="Character limit for rationale")
    report_show_full_content: bool = Field(default=False, description="Show full content without truncation")

    # Report Context Limits (for rate limit prevention)
    report_max_opportunities: int = Field(default=5, description="Max opportunities in report context")
    report_max_signals: int = Field(default=8, description="Max signals in report context")
    report_max_jobs: int = Field(default=8, description="Max job postings in report context")
    report_max_risks: int = Field(default=5, description="Max risks in report context")
    report_signal_content_limit: int = Field(default=120, description="Char limit for signal content")
    report_target_tokens: int = Field(default=12000, description="Target token limit for report context")

    # Product Matching
    rerank_pool_size: int = Field(default=20, description="Max candidates fed to cross-encoder re-ranker (reduce to 10-15 if re-ranking is too slow)")

    # Checkpointing
    checkpoint_dir: str = Field(default="data/checkpoints", description="Checkpoint directory")
    
    def get_model_for_complexity(self, complexity: int) -> str:
        """Route to appropriate model based on task complexity."""
        if complexity <= self.complexity_threshold_medium:
            return self.local_model
        elif complexity <= self.complexity_threshold_high:
            return self.smart_model
        else:
            return self.advanced_model


# Global settings instance
settings = Settings()