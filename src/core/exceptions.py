"""
Custom exceptions for the account research system.
Staff-level: Specific exceptions for different failure modes.
"""


class AccountResearchError(Exception):
    """Base exception for all system errors."""
    pass


class ConfigurationError(AccountResearchError):
    """Configuration is invalid or missing."""
    pass


class ModelError(AccountResearchError):
    """Error from language model."""
    pass


class ModelTimeoutError(ModelError):
    """Model request timed out."""
    pass


class ModelRateLimitError(ModelError):
    """Hit rate limit for model API."""
    pass


class DataSourceError(AccountResearchError):
    """Error fetching data from external source."""
    pass


class DataSourceTimeoutError(DataSourceError):
    """Data source request timed out."""
    pass


class DataSourceRateLimitError(DataSourceError):
    """Hit rate limit for data source."""
    pass


class ParsingError(AccountResearchError):
    """Error parsing response or data."""
    pass


class ValidationError(AccountResearchError):
    """Data validation failed."""
    pass


class CheckpointError(AccountResearchError):
    """Error with checkpoint save/load."""
    pass


class AgentError(AccountResearchError):
    """Error during agent execution."""
    
    def __init__(self, agent_name: str, message: str, original_error: Exception | None = None):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}' failed: {message}")


class CircuitBreakerOpenError(AccountResearchError):
    """Circuit breaker is open, refusing requests."""
    
    def __init__(self, service: str, failure_count: int):
        self.service = service
        self.failure_count = failure_count
        super().__init__(
            f"Circuit breaker open for '{service}' after {failure_count} failures"
        )