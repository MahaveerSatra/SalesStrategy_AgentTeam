"""
Pydantic models for API requests and responses.
Maps to the internal ResearchState TypedDict for the frontend.
"""
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ResearchDepthEnum(str, Enum):
    """Research depth options."""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ResearchStatusEnum(str, Enum):
    """Status of a research workflow."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    STOPPED = "stopped"  # Research paused by user, can be resumed
    ERROR = "error"


# Request Models
class ResearchRequest(BaseModel):
    """Request to start a new research workflow."""
    account_name: str = Field(..., description="Target company name", min_length=1)
    industry: str = Field(..., description="Industry vertical", min_length=1)
    seller_name: str = Field(..., description="Your company (seller)", min_length=1)
    region: str | None = Field(None, description="Geographic region")
    user_context: str | None = Field(None, description="Sales context, meeting notes, objectives")
    research_depth: ResearchDepthEnum = Field(
        ResearchDepthEnum.STANDARD,
        description="How deep should the research go"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "account_name": "Remora Carbon",
                    "industry": "carbon capture",
                    "seller_name": "MathWorks",
                    "user_context": "Sales Objective: Grow usage from current 1 license.",
                    "research_depth": "standard"
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """Human feedback on research results."""
    feedback: str = Field(..., description="Feedback text", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"feedback": "approved"},
                {"feedback": "dig deeper on their cloud initiatives"},
                {"feedback": "find other products that might fit"}
            ]
        }
    }


# Response Models
class SignalResponse(BaseModel):
    """A research signal/data point."""
    source: str
    signal_type: str
    content: str
    confidence: float
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpportunityResponse(BaseModel):
    """A validated sales opportunity."""
    product_name: str
    rationale: str
    target_persona: str | None = None
    talking_points: list[str] = Field(default_factory=list)
    estimated_value: str | None = None
    risks: list[str] = Field(default_factory=list)
    confidence: str  # "low", "medium", "high"
    confidence_score: float
    evidence_count: int = 0


class ProgressResponse(BaseModel):
    """Progress tracking for the workflow."""
    coordinator_complete: bool = False
    gatherer_complete: bool = False
    identifier_complete: bool = False
    validator_complete: bool = False
    completed_agents: list[str] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """Response when starting a new research."""
    thread_id: str
    status: ResearchStatusEnum
    message: str


class ResearchStateResponse(BaseModel):
    """Full state of a research workflow."""
    thread_id: str
    status: ResearchStatusEnum

    # Input params
    account_name: str
    industry: str
    seller_name: str
    region: str | None = None
    user_context: str | None = None
    research_depth: str

    # Progress
    progress: ProgressResponse

    # Results
    signals: list[SignalResponse] = Field(default_factory=list)
    opportunities: list[OpportunityResponse] = Field(default_factory=list)
    validated_opportunities: list[OpportunityResponse] = Field(default_factory=list)
    competitive_risks: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)

    # Human interaction
    waiting_for_human: bool = False
    human_question: str | None = None
    current_report: str | None = None

    # Metadata
    workflow_iteration: int = 1
    started_at: datetime | None = None
    last_updated: datetime | None = None
    error_messages: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    status: ResearchStatusEnum
    next_route: str | None = None
    message: str


class ThreadSummary(BaseModel):
    """Summary of a research thread."""
    thread_id: str
    account_name: str
    industry: str
    status: ResearchStatusEnum
    started_at: datetime | None = None
    progress: ProgressResponse


class ThreadListResponse(BaseModel):
    """List of all research threads."""
    threads: list[ThreadSummary]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
