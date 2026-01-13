"""
Domain models for business entities.
"""
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, HttpUrl, ConfigDict


class JobPosting(BaseModel):
    """Represents a job posting."""
    
    title: str
    company: str
    description: str
    posted_date: datetime | None = None
    location: str | None = None
    url: HttpUrl | None = None
    
    # Extracted/inferred fields
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    seniority_level: str | None = None
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Analysis results
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CompanyInfo(BaseModel):
    """Basic company information."""
    
    name: str
    industry: str
    website: HttpUrl | None = None
    size: str | None = None  # "1-50", "51-200", etc.
    headquarters: str | None = None
    founded_year: int | None = None
    description: str | None = None
    
    # Financial data (if public)
    is_public: bool = False
    ticker: str | None = None
    revenue: str | None = None
    funding_stage: str | None = None  # For private companies


class SearchResult(BaseModel):
    """A search result from web search."""

    title: str
    url: HttpUrl
    snippet: str
    source: str = "duckduckgo"
    timestamp: datetime = Field(default_factory=datetime.now)


class NewsItem(BaseModel):
    """A news article or press release."""

    title: str
    source: str
    published_date: datetime | None = None
    url: HttpUrl | None = None
    summary: str

    # Analysis
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sentiment: str | None = None  # "positive", "negative", "neutral"
    topics: list[str] = Field(default_factory=list)


class TechStackInfo(BaseModel):
    """Company's technology stack."""
    
    technologies: list[str] = Field(default_factory=list)
    categories: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Categorized: {'cloud': ['AWS'], 'analytics': ['Tableau']}"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str  # "website_scrape", "builtwith", etc.
    last_updated: datetime = Field(default_factory=datetime.now)


class Product(BaseModel):
    """A product from the catalog (e.g., MATLAB, Simulink)."""
    
    name: str
    category: str
    description: str
    key_features: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    target_personas: list[str] = Field(default_factory=list)
    typical_price_range: str | None = None
    
    # For semantic search
    embedding: list[float] | None = None


class AgentResult(BaseModel):
    """Result returned by an agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    agent_name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    execution_time_seconds: float = 0.0
    tokens_used: int | None = None
    model_used: str | None = None
    
    # For monitoring
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Response from a language model."""
    
    content: str
    model: str
    tokens_used: int | None = None
    latency_ms: float | None = None
    cached: bool = False
    
    # For structured outputs
    structured_output: dict[str, Any] | None = None