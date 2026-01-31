"""Sample state fixtures for CLI tests."""
from datetime import datetime
from src.models.state import (
    ResearchState,
    ResearchDepth,
    ResearchProgress,
    Signal,
    Opportunity,
    OpportunityConfidence
)


def create_minimal_state() -> ResearchState:
    """Create minimal state with only required fields."""
    return ResearchState(
        account_name="Test Company",
        industry="technology",
        region=None,
        user_context=None,
        research_depth=ResearchDepth.STANDARD,
        signals=[],
        job_postings=[],
        news_items=[],
        tech_stack=[],
        financial_data=None,
        opportunities=[],
        validated_opportunities=[],
        competitive_risks=[],
        progress=ResearchProgress(),
        human_feedback=[],
        waiting_for_human=False,
        human_question=None,
        started_at=datetime(2026, 1, 30, 10, 0, 0),
        last_updated=datetime(2026, 1, 30, 10, 15, 0),
        error_messages=[],
        confidence_scores={},
        current_report=None,
        workflow_iteration=1,
        feedback_context=None,
        next_route=None
    )


def create_complete_state() -> ResearchState:
    """Create fully populated state for testing."""
    signals = [
        Signal(
            source="web_search",
            signal_type="hiring",
            content="Test Company is hiring 5 senior engineers for autonomous vehicle project",
            timestamp=datetime(2026, 1, 30, 9, 0, 0),
            confidence=0.9,
            metadata={"url": "https://example.com/careers"}
        ),
        Signal(
            source="news",
            signal_type="product_launch",
            content="Test Company announces new electric vehicle platform",
            timestamp=datetime(2026, 1, 30, 8, 0, 0),
            confidence=0.85,
            metadata={"source": "TechCrunch"}
        ),
        Signal(
            source="job_board",
            signal_type="tech_stack",
            content="Python, MATLAB, Simulink experience required",
            timestamp=datetime(2026, 1, 30, 9, 30, 0),
            confidence=0.8,
            metadata={}
        )
    ]

    opportunities = [
        Opportunity(
            product_name="Automated Driving Toolbox",
            rationale="Company is expanding autonomous vehicle development with 5 senior engineers",
            evidence=signals[:2],
            target_persona="VP of Engineering",
            talking_points=[
                "Accelerate autonomous driving development",
                "Industry-standard simulation environment",
                "Reduce testing costs by 40%"
            ],
            estimated_value="$500K-$1M",
            risks=["May already have in-house simulation tools"],
            confidence=OpportunityConfidence.HIGH,
            confidence_score=0.85
        ),
        Opportunity(
            product_name="Simulink",
            rationale="Job postings mention MATLAB/Simulink experience, indicating existing usage",
            evidence=signals[2:],
            target_persona="Engineering Manager",
            talking_points=[
                "Scale existing MATLAB workflows",
                "Model-based design for control systems"
            ],
            estimated_value="$200K-$500K",
            risks=[],
            confidence=OpportunityConfidence.MEDIUM,
            confidence_score=0.65
        )
    ]

    return ResearchState(
        account_name="Test Company",
        industry="automotive",
        region="North America",
        user_context="Met at conference, expressed interest in simulation tools",
        research_depth=ResearchDepth.DEEP,
        signals=signals,
        job_postings=[
            {
                "title": "Senior Autonomous Vehicle Engineer",
                "company": "Test Company",
                "description": "Work on cutting-edge AV technology using Python, MATLAB, Simulink",
                "location": "California",
                "url": "https://example.com/job1"
            },
            {
                "title": "Control Systems Engineer",
                "company": "Test Company",
                "description": "Design and test control algorithms for electric vehicles",
                "location": "Texas",
                "url": "https://example.com/job2"
            }
        ],
        news_items=[
            {
                "title": "Test Company Raises $500M for EV Platform",
                "source": "TechCrunch",
                "published_date": "2026-01-25",
                "url": "https://example.com/news1",
                "summary": "Major funding round for electric vehicle expansion"
            }
        ],
        tech_stack=["Python", "MATLAB", "Simulink", "ROS", "TensorFlow"],
        financial_data={"funding_round": "$500M Series D", "valuation": "$5B"},
        opportunities=opportunities,
        validated_opportunities=opportunities,
        competitive_risks=[
            "Competitor XYZ already has partnership in place",
            "Budget constraints due to recent funding round"
        ],
        progress=ResearchProgress(
            coordinator_complete=True,
            gatherer_complete=True,
            identifier_complete=True,
            validator_complete=True
        ),
        human_feedback=["Looks good, but focus more on cost savings"],
        waiting_for_human=False,
        human_question=None,
        started_at=datetime(2026, 1, 30, 10, 0, 0),
        last_updated=datetime(2026, 1, 30, 10, 45, 0),
        error_messages=[],
        confidence_scores={"overall": 0.75, "data_quality": 0.8},
        current_report="# Test Report\n\nThis is a test report.",
        workflow_iteration=2,
        feedback_context="Focus on cost savings metrics",
        next_route="complete"
    )


def create_paused_state() -> ResearchState:
    """Create state that's waiting for human input."""
    state = create_minimal_state()
    state['waiting_for_human'] = True
    state['human_question'] = "Should I gather more data about their tech stack?"
    state['progress'] = ResearchProgress(
        coordinator_complete=True,
        gatherer_complete=False,
        identifier_complete=False,
        validator_complete=False
    )
    return state


def create_empty_opportunities_state() -> ResearchState:
    """Create state with no opportunities found."""
    state = create_complete_state()
    state['opportunities'] = []
    state['validated_opportunities'] = []
    return state


def create_partial_progress_state() -> ResearchState:
    """Create state with partial progress."""
    state = create_complete_state()
    state['progress'] = ResearchProgress(
        coordinator_complete=True,
        gatherer_complete=True,
        identifier_complete=False,
        validator_complete=False
    )
    state['opportunities'] = []
    state['validated_opportunities'] = []
    state['competitive_risks'] = []
    return state


def create_state_with_risks() -> ResearchState:
    """Create state with multiple competitive risks."""
    state = create_complete_state()
    state['competitive_risks'] = [
        "Competitor A has 5-year partnership",
        "Budget frozen until Q3",
        "Recent executive turnover in engineering",
        "Existing investment in alternative platform",
        "Regulatory concerns in primary market"
    ]
    return state
