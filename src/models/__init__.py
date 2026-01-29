"""
Models package - Domain models and LLM schemas.
"""
from .domain import (
    JobPosting,
    CompanyInfo,
    SearchResult,
    NewsItem,
    TechStackInfo,
    Product,
    AgentResult,
    ModelResponse,
)
from .state import (
    ResearchState,
    ResearchDepth,
    ResearchProgress,
    Signal,
    Opportunity,
    OpportunityConfidence,
)
from .llm_schemas import (
    SourceAnalysis,
    RequirementsExtraction,
    OpportunityItem,
    OpportunitiesGeneration,
    RiskAssessment,
    ScoredOpportunityItem,
    OpportunityScoring,
    InputValidation,
    ClarificationCheck,
    FeedbackIntent,
)

__all__ = [
    # Domain models
    "JobPosting",
    "CompanyInfo",
    "SearchResult",
    "NewsItem",
    "TechStackInfo",
    "Product",
    "AgentResult",
    "ModelResponse",
    # State models
    "ResearchState",
    "ResearchDepth",
    "ResearchProgress",
    "Signal",
    "Opportunity",
    "OpportunityConfidence",
    # LLM Schemas for structured outputs
    "SourceAnalysis",
    "RequirementsExtraction",
    "OpportunityItem",
    "OpportunitiesGeneration",
    "RiskAssessment",
    "ScoredOpportunityItem",
    "OpportunityScoring",
    "InputValidation",
    "ClarificationCheck",
    "FeedbackIntent",
]
