"""
State models for LangGraph workflow.
Defines the structure of data flowing between agents.
"""
from datetime import datetime
from typing import TypedDict, Annotated, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# Enums for structured choices
class ResearchDepth(str, Enum):
    """How deep should the research go."""
    QUICK = "quick"          # 2-3 minutes, basic signals
    STANDARD = "standard"    # 3-5 minutes, comprehensive
    DEEP = "deep"           # 5-10 minutes, exhaustive


class OpportunityConfidence(str, Enum):
    """Confidence level for opportunities."""
    LOW = "low"         # < 40%
    MEDIUM = "medium"   # 40-70%
    HIGH = "high"       # > 70%


# Domain Models
class Signal(BaseModel):
    """A research signal/data point."""
    model_config = ConfigDict(frozen=True)
    
    source: str = Field(description="Where this signal came from")
    signal_type: str = Field(description="Type: hiring, tech_stack, news, etc.")
    content: str = Field(description="The actual signal content")
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(ge=0.0, le=1.0, description="Signal reliability")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Opportunity(BaseModel):
    """A potential upsell/cross-sell opportunity."""
    
    product_name: str = Field(description="Which product to sell")
    rationale: str = Field(description="Why they likely need it")
    evidence: list[Signal] = Field(default_factory=list, description="Supporting signals")
    target_persona: str | None = Field(default=None, description="Who to talk to")
    talking_points: list[str] = Field(default_factory=list)
    estimated_value: str | None = Field(default=None, description="Deal size estimate")
    risks: list[str] = Field(default_factory=list, description="Potential blockers")
    confidence: OpportunityConfidence = Field(default=OpportunityConfidence.MEDIUM)
    confidence_score: float = Field(ge=0.0, le=1.0, description="Numerical confidence")


class ResearchProgress(BaseModel):
    """Track which agents have completed."""
    
    coordinator_complete: bool = False
    gatherer_complete: bool = False
    identifier_complete: bool = False
    validator_complete: bool = False
    
    def get_completed_agents(self) -> list[str]:
        """Return list of completed agent names."""
        completed = []
        if self.coordinator_complete:
            completed.append("coordinator")
        if self.gatherer_complete:
            completed.append("gatherer")
        if self.identifier_complete:
            completed.append("identifier")
        if self.validator_complete:
            completed.append("validator")
        return completed
    
    def is_complete(self) -> bool:
        """Check if all agents have finished."""
        return all([
            self.coordinator_complete,
            self.gatherer_complete,
            self.identifier_complete,
            self.validator_complete
        ])


# Main state for LangGraph workflow
class ResearchState(TypedDict):
    """
    State passed between agents in LangGraph.
    Uses TypedDict for LangGraph compatibility.
    """
    
    # Input parameters
    account_name: str
    industry: str
    region: str | None
    user_context: str | None  # Optional meeting notes, etc.
    research_depth: ResearchDepth
    
    # Research data (collected by Intelligence Gatherer)
    signals: list[Signal]
    job_postings: list[dict[str, Any]]
    news_items: list[dict[str, Any]]
    tech_stack: list[str]
    financial_data: dict[str, Any] | None
    
    # Analysis results (from Opportunity Identifier)
    opportunities: list[Opportunity]
    
    # Validation results (from Strategy Validator)
    validated_opportunities: list[Opportunity]
    competitive_risks: list[str]
    
    # Progress tracking
    progress: ResearchProgress
    
    # Human interaction
    human_feedback: list[str]  # Conversation history with human
    waiting_for_human: bool
    human_question: str | None
    
    # Metadata
    started_at: datetime
    last_updated: datetime
    error_messages: list[str]
    
    # Confidence scores for different aspects
    confidence_scores: dict[str, float]

    # CoordinatorAgent fields (for workflow routing and feedback loops)
    current_report: str | None           # Formatted report from process_exit()
    workflow_iteration: int              # Track feedback loop count (default: 1)
    feedback_context: str | None         # Parsed guidance for retry
    next_route: str | None               # Routing decision: "gatherer"|"identifier"|"validator"|"complete"


# Helper function to create initial state
def create_initial_state(
    account_name: str,
    industry: str,
    region: str | None = None,
    user_context: str | None = None,
    research_depth: ResearchDepth = ResearchDepth.STANDARD
) -> ResearchState:
    """Create initial state for a new research workflow."""
    now = datetime.now()
    
    return ResearchState(
        # Input
        account_name=account_name,
        industry=industry,
        region=region,
        user_context=user_context,
        research_depth=research_depth,
        
        # Empty collections
        signals=[],
        job_postings=[],
        news_items=[],
        tech_stack=[],
        financial_data=None,
        opportunities=[],
        validated_opportunities=[],
        competitive_risks=[],
        
        # Progress
        progress=ResearchProgress(),
        
        # Human interaction
        human_feedback=[],
        waiting_for_human=False,
        human_question=None,
        
        # Metadata
        started_at=now,
        last_updated=now,
        error_messages=[],
        confidence_scores={},

        # CoordinatorAgent fields
        current_report=None,
        workflow_iteration=1,
        feedback_context=None,
        next_route=None
    )