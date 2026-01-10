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
    
    # Model Configuration
    local_model: str = Field(default="llama3.2:3b", description="Local Ollama model")
    smart_model: str = Field(
        default="groq/llama-3.1-8b-instant", 
        description="External model for complex reasoning"
    )
    advanced_model: str = Field(
        default="groq/llama-3.1-70b",
        description="External model for advanced reasoning"
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