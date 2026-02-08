"""
Pydantic schemas for LLM structured outputs.

These schemas are used with Ollama's structured output feature to guarantee
valid JSON responses from the LLM. Use `Model.model_json_schema()` to get
the JSON schema to pass to the `response_format` parameter.

Reference: https://docs.ollama.com/capabilities/structured-outputs

Usage:
    from src.models.llm_schemas import SourceAnalysis

    response = await router.generate(
        prompt="Analyze this source...",
        complexity=3,
        temperature=0,
        response_format=SourceAnalysis.model_json_schema()
    )
    result = SourceAnalysis.model_validate_json(response.content)
"""
from typing import Literal, Any
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# GathererAgent Schemas
# =============================================================================

class SourceAnalysis(BaseModel):
    """Schema for GathererAgent's source analysis output.

    Used in `_analyze_source_with_llm()` to analyze web sources.
    """
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    summary: str = Field(description="2-3 sentence summary of key information")
    source_type: str = Field(description="Type of source: official_company_site, news, blog, etc.")
    key_facts: list[str] = Field(default_factory=list, description="List of key facts extracted")
    keywords: list[str] = Field(default_factory=list, description="Keywords and technologies mentioned")
    relevance: Literal["high", "medium", "low"] = Field(description="Relevance to research")


class SearchQueryItem(BaseModel):
    """Individual search query with category and priority."""
    category: str = Field(description="Query category: tech_stack, hiring, strategic, partnerships, challenges")
    query: str = Field(description="The search query string")
    priority: int = Field(ge=1, le=5, description="Priority 1-5 (1=highest)")


class SearchQueryGeneration(BaseModel):
    """Schema for GathererAgent's search query generation output.

    Used in `_build_queries()` to generate multiple targeted search queries.
    """
    queries: list[SearchQueryItem] = Field(
        default_factory=list,
        description="List of targeted search queries"
    )


class BuyingSignals(BaseModel):
    """Buying signals extracted from a source for sales intelligence."""
    technologies: list[str] = Field(default_factory=list, description="Technologies mentioned")
    hiring_for: list[str] = Field(default_factory=list, description="Roles being hired")
    budget_indicators: list[str] = Field(default_factory=list, description="Budget/investment mentions")
    urgency_signals: list[str] = Field(default_factory=list, description="Urgency indicators")
    decision_makers: list[str] = Field(default_factory=list, description="Decision maker titles/names")
    pain_points: list[str] = Field(default_factory=list, description="Challenges/pain points mentioned")
    competitors_mentioned: list[str] = Field(default_factory=list, description="Competitor/vendor mentions")


class SalesSourceAnalysis(BaseModel):
    """Schema for GathererAgent's sales-focused source analysis output.

    Enhanced version of SourceAnalysis with buying signals for sales intelligence.
    """
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    summary: str = Field(description="2-3 sentence summary focused on sales-relevant insights")
    source_type: str = Field(description="Type: official_company_site, news, blog, forum, other")
    sales_relevance: Literal["high", "medium", "low"] = Field(description="Relevance to sales opportunity")
    buying_signals: BuyingSignals = Field(default_factory=BuyingSignals, description="Extracted buying signals")
    key_facts: list[str] = Field(default_factory=list, description="Verifiable facts extracted")
    keywords: list[str] = Field(default_factory=list, description="Keywords and technologies")


class JobPostingAnalysis(BaseModel):
    """Schema for GathererAgent's job posting analysis output.

    Used in `_analyze_job_posting_with_llm()` for sales intelligence from jobs.
    """
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    summary: str = Field(description="What this hiring tells us about sales opportunity")
    technologies_required: list[str] = Field(default_factory=list, description="Required technologies")
    technologies_desired: list[str] = Field(default_factory=list, description="Nice-to-have technologies")
    urgency: Literal["high", "medium", "low"] = Field(default="medium", description="Hiring urgency")
    seniority: str = Field(default="mid", description="Role seniority: entry, mid, senior, leadership")
    team_indicators: str = Field(default="", description="Team size/growth indicators if mentioned")
    seller_relevance: Literal["high", "medium", "low"] = Field(default="medium", description="Relevance to seller")
    potential_champion: bool = Field(default=False, description="Could this role be a product champion?")
    sales_insight: str = Field(default="", description="How seller could help this role/team")


# =============================================================================
# IdentifierAgent Schemas
# =============================================================================

class RequirementsExtraction(BaseModel):
    """Schema for IdentifierAgent's requirements extraction output.

    Used in `_extract_requirements()` to extract technology/business requirements.
    """
    requirements: list[Any] = Field(
        default_factory=list,
        description="List of 5-15 concise requirement statements"
    )

    @field_validator("requirements", mode="after")
    @classmethod
    def filter_and_stringify_requirements(cls, v: list[Any]) -> list[str]:
        """Filter out empty/None values and convert all to strings."""
        return [str(r) for r in v if r]


class OpportunityItem(BaseModel):
    """Individual opportunity within the opportunities list.

    Note: Some fields are optional to handle varied LLM outputs gracefully.
    The code should provide defaults for missing values.
    """
    product_name: str = Field(description="Name of the product")
    rationale: str = Field(default="", description="Why they likely need this product (2-3 sentences)")
    target_persona: str | None = Field(default=None, description="Job title of target buyer")
    talking_points: list[str] = Field(default_factory=list, description="3-5 specific conversation points")
    estimated_value: str | None = Field(default=None, description="Deal size estimate (e.g., '$50K ARR')")
    risks: list[str] = Field(default_factory=list, description="1-3 potential blockers")
    confidence: str = Field(default="medium", description="Confidence level: high, medium, or low")
    confidence_score: float = Field(default=0.5, description="Numerical confidence 0.0-1.0")

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: str) -> str:
        """Normalize confidence to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return "medium"


class OpportunitiesGeneration(BaseModel):
    """Schema for IdentifierAgent's opportunities generation output.

    Used in `_generate_opportunities()` to generate sales opportunities.
    """
    opportunities: list[OpportunityItem] = Field(
        default_factory=list,
        description="List of identified opportunities"
    )


# =============================================================================
# ValidatorAgent Schemas
# =============================================================================

class RiskAssessment(BaseModel):
    """Schema for ValidatorAgent's risk assessment output.

    Used in `_assess_risks()` to identify competitive and market risks.
    """
    risks: list[Any] = Field(
        default_factory=list,
        description="List of risk descriptions with supporting evidence"
    )

    @field_validator("risks", mode="after")
    @classmethod
    def filter_and_stringify_risks(cls, v: list[Any]) -> list[str]:
        """Filter out empty/None values and convert all to strings."""
        return [str(r) for r in v if r]


class ScoredOpportunityItem(BaseModel):
    """Individual scored opportunity.

    Note: new_score does not have range constraints because the code
    clamps the value after parsing. This allows the model to handle
    edge cases gracefully.
    """
    product_name: str = Field(description="Name of the product")
    new_score: float = Field(description="Updated confidence score")
    score_rationale: str = Field(default="", description="Explanation for the score adjustment")


class OpportunityScoring(BaseModel):
    """Schema for ValidatorAgent's opportunity scoring output.

    Used in `_score_opportunities()` to re-score opportunities with risk context.
    """
    scored_opportunities: list[ScoredOpportunityItem] = Field(
        default_factory=list,
        description="List of opportunities with updated scores"
    )


# =============================================================================
# CoordinatorAgent Schemas
# =============================================================================

class InputValidation(BaseModel):
    """Schema for CoordinatorAgent's input validation output.

    Used in `_validate_inputs()` to validate research request inputs.
    """
    is_valid: bool = Field(description="Whether the inputs are valid")
    errors: list[str] = Field(default_factory=list, description="List of validation errors")
    suggested_corrections: dict[str, str] = Field(
        default_factory=dict,
        description="Field corrections: {'field_name': 'corrected_value'}"
    )
    concerns: list[str] = Field(default_factory=list, description="Non-blocking concerns")


class ClarificationCheck(BaseModel):
    """Schema for CoordinatorAgent's clarifying questions output.

    Used in `_generate_clarifying_questions()` to determine if clarification needed.
    """
    needs_clarification: bool = Field(description="Whether clarification would help")
    questions: str | None = Field(default=None, description="1-2 focused questions if needed")
    reasoning: str = Field(default="", description="Brief explanation of the decision")


class FeedbackIntent(BaseModel):
    """Schema for CoordinatorAgent's feedback intent parsing output.

    Used in `_parse_feedback_intent()` to classify human feedback into routing.
    """
    route: str = Field(description="Routing decision: GATHERER, IDENTIFIER, VALIDATOR, or COMPLETE")
    reasoning: str = Field(default="", description="Brief explanation of classification")
    context_for_retry: str = Field(
        default="",
        description="Specific guidance for the next agent based on feedback"
    )

    @field_validator("route", mode="after")
    @classmethod
    def normalize_route(cls, v: str) -> str:
        """Normalize route to uppercase."""
        return v.upper() if isinstance(v, str) else "COMPLETE"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Gatherer
    "SourceAnalysis",
    "SearchQueryItem",
    "SearchQueryGeneration",
    "BuyingSignals",
    "SalesSourceAnalysis",
    "JobPostingAnalysis",
    # Identifier
    "RequirementsExtraction",
    "OpportunityItem",
    "OpportunitiesGeneration",
    # Validator
    "RiskAssessment",
    "ScoredOpportunityItem",
    "OpportunityScoring",
    # Coordinator
    "InputValidation",
    "ClarificationCheck",
    "FeedbackIntent",
]
