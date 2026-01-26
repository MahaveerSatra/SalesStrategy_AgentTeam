"""
Tests for Coordinator Agent.
Comprehensive test coverage with mocked dependencies.

Tests cover:
1. Entry point (process_entry) - validation, normalization, questioning
2. Exit point (process_exit) - report formatting, human-in-loop
3. Feedback routing (process_feedback) - routing decisions
4. Edge cases - JSON parsing failures, empty inputs
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.agents.coordinator import CoordinatorAgent, WorkflowRoute
from src.models.state import (
    ResearchState,
    Signal,
    Opportunity,
    ResearchProgress,
    ResearchDepth,
    OpportunityConfidence,
    create_initial_state
)
from src.core.model_router import ModelRouter


@pytest.fixture
def mock_model_router():
    """Provide mocked model router for LLM calls."""
    router = AsyncMock(spec=ModelRouter)
    return router


@pytest.fixture
def coordinator_agent(mock_model_router):
    """Provide CoordinatorAgent instance with mocked dependencies."""
    return CoordinatorAgent(model_router=mock_model_router)


@pytest.fixture
def initial_state():
    """Provide initial research state for testing."""
    return create_initial_state(
        account_name="Acme Corp",
        industry="Technology",
        region="North America",
        research_depth=ResearchDepth.STANDARD
    )


@pytest.fixture
def state_with_opportunities():
    """Provide state with validated opportunities for exit testing."""
    state = create_initial_state(
        account_name="Acme Corp",
        industry="Technology",
        research_depth=ResearchDepth.STANDARD
    )

    # Mark gatherer, identifier, validator as complete
    state["progress"].coordinator_complete = True
    state["progress"].gatherer_complete = True
    state["progress"].identifier_complete = True
    state["progress"].validator_complete = True

    # Add validated opportunities
    state["validated_opportunities"] = [
        Opportunity(
            product_name="Enterprise Analytics Suite",
            rationale="Strong hiring in data engineering and analytics roles",
            evidence=[
                Signal(
                    source="job_boards",
                    signal_type="hiring",
                    content="Hiring 5 data engineers",
                    confidence=0.9,
                    metadata={}
                )
            ],
            target_persona="VP of Engineering",
            talking_points=["Data-driven culture", "Scaling analytics team"],
            estimated_value="$250K ARR",
            risks=["Competitor evaluation"],
            confidence=OpportunityConfidence.HIGH,
            confidence_score=0.85
        ),
        Opportunity(
            product_name="Cloud Migration Toolkit",
            rationale="Job postings mention AWS and cloud modernization",
            evidence=[
                Signal(
                    source="duckduckgo",
                    signal_type="web_search",
                    content="Acme Corp cloud strategy",
                    confidence=0.75,
                    metadata={}
                )
            ],
            target_persona="CTO",
            talking_points=["Cost optimization", "Modernization roadmap"],
            estimated_value="$150K ARR",
            risks=["Budget constraints"],
            confidence=OpportunityConfidence.MEDIUM,
            confidence_score=0.72
        )
    ]

    state["competitive_risks"] = [
        "Competitor XYZ has existing relationship",
        "Upcoming budget review in Q2"
    ]

    # Add some signals
    state["signals"] = [
        Signal(
            source="duckduckgo",
            signal_type="web_search",
            content="Acme Corp expanding cloud infrastructure",
            confidence=0.8,
            metadata={}
        )
    ]

    state["job_postings"] = [
        {"title": "Senior Data Engineer", "company": "Acme Corp"}
    ]

    return state


@pytest.fixture
def state_with_feedback(state_with_opportunities):
    """Provide state with human feedback for routing tests."""
    state = state_with_opportunities
    state["current_report"] = "# Sales Report\n\nTest report content."
    state["human_feedback"] = ["dig deeper on their cloud initiatives"]
    state["waiting_for_human"] = False
    return state


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinatorAgentInit:
    """Test CoordinatorAgent initialization."""

    def test_init_creates_agent(self, mock_model_router):
        """Test that agent initializes correctly."""
        agent = CoordinatorAgent(model_router=mock_model_router)

        assert agent.name == "coordinator"
        assert agent.model_router == mock_model_router

    def test_get_complexity(self, coordinator_agent, initial_state):
        """Test complexity returns 3 for LOCAL Ollama."""
        complexity = coordinator_agent.get_complexity(initial_state)
        assert complexity == 3


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT TESTS (process_entry)
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinatorEntryValidInputs:
    """Test entry point with valid inputs."""

    @pytest.mark.asyncio
    async def test_valid_inputs_pass_through(
        self,
        coordinator_agent,
        initial_state,
        mock_model_router
    ):
        """Test that valid inputs proceed without errors."""
        # Mock LLM responses
        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {}, "concerns": []}'

        normalization_response = MagicMock()
        normalization_response.content = 'Acme Corporation'

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": false, "questions": null, "reasoning": "Inputs are clear"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        # Execute
        await coordinator_agent.process_entry(initial_state)

        # Verify state updated
        assert initial_state["progress"].coordinator_complete is True
        assert initial_state["waiting_for_human"] is False
        assert initial_state["human_question"] is None
        assert len(initial_state["error_messages"]) == 0

    @pytest.mark.asyncio
    async def test_company_name_normalized(
        self,
        coordinator_agent,
        mock_model_router
    ):
        """Test company name normalization."""
        state = create_initial_state(
            account_name="msft",
            industry="Technology"
        )

        # Mock responses
        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {}, "concerns": []}'

        normalization_response = MagicMock()
        normalization_response.content = 'Microsoft'

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": false, "questions": null, "reasoning": "Clear"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        # Execute
        await coordinator_agent.process_entry(state)

        # Verify name normalized
        assert state["account_name"] == "Microsoft"
        assert state["progress"].coordinator_complete is True


class TestCoordinatorEntryInvalidInputs:
    """Test entry point with invalid inputs."""

    @pytest.mark.asyncio
    async def test_missing_account_name_triggers_error(
        self,
        coordinator_agent,
        mock_model_router
    ):
        """Test that missing account_name triggers validation error."""
        state = create_initial_state(
            account_name="",
            industry="Technology"
        )

        # Execute
        await coordinator_agent.process_entry(state)

        # Verify error state
        assert len(state["error_messages"]) > 0
        assert "Account name is required" in state["error_messages"][0]
        assert state["waiting_for_human"] is True
        assert state["human_question"] is not None
        # Should NOT mark complete - need human to fix
        assert state["progress"].coordinator_complete is False

    @pytest.mark.asyncio
    async def test_missing_industry_triggers_error(
        self,
        coordinator_agent,
        mock_model_router
    ):
        """Test that missing industry triggers validation error."""
        state = create_initial_state(
            account_name="Acme Corp",
            industry=""
        )

        # Execute
        await coordinator_agent.process_entry(state)

        # Verify error state
        assert len(state["error_messages"]) > 0
        assert "Industry is required" in state["error_messages"][0]
        assert state["waiting_for_human"] is True

    @pytest.mark.asyncio
    async def test_llm_validation_finds_issues(
        self,
        coordinator_agent,
        initial_state,
        mock_model_router
    ):
        """Test LLM validation can catch issues."""
        validation_response = MagicMock()
        validation_response.content = '{"is_valid": false, "errors": ["Company name appears to be gibberish"], "suggested_corrections": {}, "concerns": []}'

        mock_model_router.generate.return_value = validation_response

        # Execute
        await coordinator_agent.process_entry(initial_state)

        # Verify error from LLM validation
        assert len(initial_state["error_messages"]) > 0
        assert "gibberish" in initial_state["error_messages"][0]
        assert initial_state["waiting_for_human"] is True

    @pytest.mark.asyncio
    async def test_validation_applies_corrections(
        self,
        coordinator_agent,
        mock_model_router
    ):
        """Test that suggested corrections are applied."""
        state = create_initial_state(
            account_name="Microsft",  # Typo
            industry="Techology"  # Typo
        )

        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {"account_name": "Microsoft", "industry": "Technology"}, "concerns": []}'

        normalization_response = MagicMock()
        normalization_response.content = 'Microsoft'

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": false, "questions": null, "reasoning": "Clear"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        # Execute
        await coordinator_agent.process_entry(state)

        # Verify corrections applied
        assert state["account_name"] == "Microsoft"
        assert state["industry"] == "Technology"


class TestCoordinatorEntrySmartQuestioning:
    """Test smart questioning logic."""

    @pytest.mark.asyncio
    async def test_generates_questions_when_needed(
        self,
        coordinator_agent,
        mock_model_router
    ):
        """Test that questions are generated when LLM determines they would help."""
        state = create_initial_state(
            account_name="Amazon",  # Ambiguous - AWS or Retail?
            industry="Technology"
        )

        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {}, "concerns": []}'

        normalization_response = MagicMock()
        normalization_response.content = 'Amazon'

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": true, "questions": "Are you interested in Amazon Web Services (AWS) or Amazon Retail?", "reasoning": "Company name is ambiguous"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        # Execute
        await coordinator_agent.process_entry(state)

        # Verify question generated
        assert state["waiting_for_human"] is True
        assert state["human_question"] is not None
        assert "AWS" in state["human_question"] or "Retail" in state["human_question"]
        # Still marks complete since we can proceed after human responds
        assert state["progress"].coordinator_complete is True

    @pytest.mark.asyncio
    async def test_no_questions_when_inputs_clear(
        self,
        coordinator_agent,
        initial_state,
        mock_model_router
    ):
        """Test that no questions generated when inputs are sufficiently clear."""
        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {}, "concerns": []}'

        normalization_response = MagicMock()
        normalization_response.content = 'Acme Corporation'

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": false, "questions": null, "reasoning": "Inputs are sufficiently clear for research"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        # Execute
        await coordinator_agent.process_entry(initial_state)

        # Verify no questions
        assert initial_state["waiting_for_human"] is False
        assert initial_state["human_question"] is None
        assert initial_state["progress"].coordinator_complete is True


class TestCoordinatorEntryEdgeCases:
    """Test entry point edge cases."""

    @pytest.mark.asyncio
    async def test_llm_json_parse_failure_continues(
        self,
        coordinator_agent,
        initial_state,
        mock_model_router
    ):
        """Test that JSON parse failure in validation doesn't crash."""
        validation_response = MagicMock()
        validation_response.content = 'Invalid JSON response from LLM'

        normalization_response = MagicMock()
        normalization_response.content = 'Acme Corporation'

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": false, "questions": null, "reasoning": "Clear"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        # Execute - should not crash
        await coordinator_agent.process_entry(initial_state)

        # Should continue without error (graceful degradation)
        assert initial_state["progress"].coordinator_complete is True

    @pytest.mark.asyncio
    async def test_llm_call_failure_continues(
        self,
        coordinator_agent,
        initial_state,
        mock_model_router
    ):
        """Test that LLM call failure doesn't crash."""
        # First call succeeds for validation, then fails for normalization
        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {}, "concerns": []}'

        mock_model_router.generate.side_effect = [
            validation_response,
            Exception("LLM service unavailable"),  # Normalization fails
            Exception("LLM service unavailable")   # Questioning fails
        ]

        # Execute - should not crash
        await coordinator_agent.process_entry(initial_state)

        # Should continue (uses original name)
        assert initial_state["account_name"] == "Acme Corp"
        assert initial_state["progress"].coordinator_complete is True

    @pytest.mark.asyncio
    async def test_normalization_returns_unreasonable_name(
        self,
        coordinator_agent,
        mock_model_router
    ):
        """Test that unreasonably long normalized names are rejected."""
        state = create_initial_state(
            account_name="MSFT",
            industry="Technology"
        )

        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {}, "concerns": []}'

        # Return absurdly long name
        normalization_response = MagicMock()
        normalization_response.content = 'A' * 1000  # Very long name

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": false, "questions": null, "reasoning": "Clear"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        # Execute
        await coordinator_agent.process_entry(state)

        # Should keep original name (sanity check in _normalize_company_name)
        assert state["account_name"] == "MSFT"


# ─────────────────────────────────────────────────────────────────────────────
# EXIT POINT TESTS (process_exit)
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinatorExitReportFormatting:
    """Test exit point report formatting."""

    @pytest.mark.asyncio
    async def test_report_formatted_successfully(
        self,
        coordinator_agent,
        state_with_opportunities,
        mock_model_router
    ):
        """Test that report is formatted and stored."""
        report_response = MagicMock()
        report_response.content = """## Executive Summary
Acme Corp shows strong signals for analytics and cloud products.

## Top Opportunities
1. **Enterprise Analytics Suite** (85%)
   - Strong hiring in data engineering

## Competitive Landscape
- Competitor XYZ relationship

## Recommended Next Steps
1. Schedule discovery call

---
Please review the analysis above. You can:
- Reply 'approved' or 'looks good' to finalize"""

        mock_model_router.generate.return_value = report_response

        # Execute
        await coordinator_agent.process_exit(state_with_opportunities)

        # Verify report stored
        assert state_with_opportunities["current_report"] is not None
        assert "Executive Summary" in state_with_opportunities["current_report"]
        assert "Top Opportunities" in state_with_opportunities["current_report"]

        # Verify human-in-loop flags
        assert state_with_opportunities["waiting_for_human"] is True
        assert state_with_opportunities["human_question"] == state_with_opportunities["current_report"]

    @pytest.mark.asyncio
    async def test_fallback_report_on_llm_failure(
        self,
        coordinator_agent,
        state_with_opportunities,
        mock_model_router
    ):
        """Test that fallback report is generated when LLM fails."""
        mock_model_router.generate.side_effect = Exception("LLM service down")

        # Execute
        await coordinator_agent.process_exit(state_with_opportunities)

        # Verify fallback report generated
        assert state_with_opportunities["current_report"] is not None
        assert "Sales Intelligence Report" in state_with_opportunities["current_report"]
        assert "Enterprise Analytics Suite" in state_with_opportunities["current_report"]
        assert "85%" in state_with_opportunities["current_report"]

        # Verify human-in-loop flags
        assert state_with_opportunities["waiting_for_human"] is True

    @pytest.mark.asyncio
    async def test_fallback_report_with_no_opportunities(
        self,
        coordinator_agent,
        mock_model_router
    ):
        """Test fallback report when no opportunities exist."""
        state = create_initial_state(
            account_name="Acme Corp",
            industry="Technology"
        )
        state["progress"].validator_complete = True
        state["validated_opportunities"] = []
        state["competitive_risks"] = []

        mock_model_router.generate.side_effect = Exception("LLM fail")

        # Execute
        await coordinator_agent.process_exit(state)

        # Verify fallback report generated
        assert state["current_report"] is not None
        assert "No validated opportunities found" in state["current_report"]

    @pytest.mark.asyncio
    async def test_workflow_iteration_preserved(
        self,
        coordinator_agent,
        state_with_opportunities,
        mock_model_router
    ):
        """Test that workflow iteration counter is preserved."""
        state_with_opportunities["workflow_iteration"] = 2

        report_response = MagicMock()
        report_response.content = "Test report"
        mock_model_router.generate.return_value = report_response

        # Execute
        await coordinator_agent.process_exit(state_with_opportunities)

        # Verify iteration preserved
        assert state_with_opportunities["workflow_iteration"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK ROUTING TESTS (process_feedback)
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinatorFeedbackRouting:
    """Test feedback routing decisions."""

    @pytest.mark.asyncio
    async def test_dig_deeper_routes_to_gatherer(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that 'dig deeper' routes to GATHERER."""
        state_with_feedback["human_feedback"] = ["dig deeper on their cloud initiatives"]

        routing_response = MagicMock()
        routing_response.content = '{"route": "GATHERER", "reasoning": "User wants more data", "context_for_retry": "Focus on cloud infrastructure"}'

        context_response = MagicMock()
        context_response.content = "Research more about cloud initiatives and infrastructure plans"

        mock_model_router.generate.side_effect = [routing_response, context_response]

        # Execute
        route = await coordinator_agent.process_feedback(state_with_feedback)

        # Verify routing
        assert route == WorkflowRoute.GATHERER
        assert state_with_feedback["next_route"] == "gatherer"

        # Verify progress flags reset
        assert state_with_feedback["progress"].gatherer_complete is False
        assert state_with_feedback["progress"].identifier_complete is False
        assert state_with_feedback["progress"].validator_complete is False

        # Verify iteration incremented
        assert state_with_feedback["workflow_iteration"] == 2

        # Verify report cleared
        assert state_with_feedback["current_report"] is None

    @pytest.mark.asyncio
    async def test_different_opportunities_routes_to_identifier(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that 'different opportunities' routes to IDENTIFIER."""
        state_with_feedback["human_feedback"] = ["find different opportunities, maybe other products"]

        routing_response = MagicMock()
        routing_response.content = '{"route": "IDENTIFIER", "reasoning": "User wants different products", "context_for_retry": "Look for other product matches"}'

        context_response = MagicMock()
        context_response.content = "Identify opportunities for different products"

        mock_model_router.generate.side_effect = [routing_response, context_response]

        # Execute
        route = await coordinator_agent.process_feedback(state_with_feedback)

        # Verify routing
        assert route == WorkflowRoute.IDENTIFIER
        assert state_with_feedback["next_route"] == "identifier"

        # Verify only identifier/validator reset (not gatherer)
        assert state_with_feedback["progress"].gatherer_complete is True
        assert state_with_feedback["progress"].identifier_complete is False
        assert state_with_feedback["progress"].validator_complete is False

    @pytest.mark.asyncio
    async def test_recheck_confidence_routes_to_validator(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that 'recheck confidence' routes to VALIDATOR."""
        state_with_feedback["human_feedback"] = ["the confidence for analytics seems too high"]

        routing_response = MagicMock()
        routing_response.content = '{"route": "VALIDATOR", "reasoning": "User questions confidence scores", "context_for_retry": "Re-evaluate analytics opportunity confidence"}'

        context_response = MagicMock()
        context_response.content = "Re-evaluate confidence scores, especially for analytics"

        mock_model_router.generate.side_effect = [routing_response, context_response]

        # Execute
        route = await coordinator_agent.process_feedback(state_with_feedback)

        # Verify routing
        assert route == WorkflowRoute.VALIDATOR
        assert state_with_feedback["next_route"] == "validator"

        # Verify only validator reset
        assert state_with_feedback["progress"].gatherer_complete is True
        assert state_with_feedback["progress"].identifier_complete is True
        assert state_with_feedback["progress"].validator_complete is False

    @pytest.mark.asyncio
    async def test_approved_routes_to_complete(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that 'approved' routes to COMPLETE."""
        state_with_feedback["human_feedback"] = ["looks good, approved"]

        routing_response = MagicMock()
        routing_response.content = '{"route": "COMPLETE", "reasoning": "User approved the report", "context_for_retry": ""}'

        mock_model_router.generate.return_value = routing_response

        # Execute
        route = await coordinator_agent.process_feedback(state_with_feedback)

        # Verify routing
        assert route == WorkflowRoute.COMPLETE
        assert state_with_feedback["next_route"] == "complete"

        # Verify nothing reset (workflow complete)
        assert state_with_feedback["progress"].gatherer_complete is True
        assert state_with_feedback["progress"].identifier_complete is True
        assert state_with_feedback["progress"].validator_complete is True

        # Verify waiting flag cleared
        assert state_with_feedback["waiting_for_human"] is False


class TestCoordinatorFeedbackContextUpdate:
    """Test context updates for retry loops."""

    @pytest.mark.asyncio
    async def test_feedback_context_stored(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that feedback context is stored in state."""
        state_with_feedback["human_feedback"] = ["need more info on their hiring plans"]

        routing_response = MagicMock()
        routing_response.content = '{"route": "GATHERER", "reasoning": "Need more data", "context_for_retry": "Focus on hiring"}'

        context_response = MagicMock()
        context_response.content = "Focus on hiring initiatives and workforce expansion plans"

        mock_model_router.generate.side_effect = [routing_response, context_response]

        # Execute
        await coordinator_agent.process_feedback(state_with_feedback)

        # Verify feedback_context stored
        assert state_with_feedback["feedback_context"] is not None
        assert "hiring" in state_with_feedback["feedback_context"].lower()

    @pytest.mark.asyncio
    async def test_user_context_updated_for_visibility(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that user_context is updated with feedback for downstream agents."""
        state_with_feedback["user_context"] = "Initial context"
        state_with_feedback["human_feedback"] = ["dig deeper"]

        routing_response = MagicMock()
        routing_response.content = '{"route": "GATHERER", "reasoning": "More data needed", "context_for_retry": "Dig deeper"}'

        context_response = MagicMock()
        context_response.content = "Perform deeper research on all areas"

        mock_model_router.generate.side_effect = [routing_response, context_response]

        # Execute
        await coordinator_agent.process_feedback(state_with_feedback)

        # Verify user_context updated
        assert "Initial context" in state_with_feedback["user_context"]
        assert "[Feedback Round" in state_with_feedback["user_context"]


class TestCoordinatorFeedbackEdgeCases:
    """Test feedback routing edge cases."""

    @pytest.mark.asyncio
    async def test_empty_feedback_list_returns_complete(
        self,
        coordinator_agent,
        state_with_opportunities,
        mock_model_router
    ):
        """Test that empty feedback list returns COMPLETE."""
        state_with_opportunities["human_feedback"] = []

        # Execute
        route = await coordinator_agent.process_feedback(state_with_opportunities)

        # Verify defaults to complete
        assert route == WorkflowRoute.COMPLETE

    @pytest.mark.asyncio
    async def test_json_parse_failure_returns_complete(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that JSON parse failure defaults to COMPLETE."""
        routing_response = MagicMock()
        routing_response.content = 'Not valid JSON at all'

        mock_model_router.generate.return_value = routing_response

        # Execute
        route = await coordinator_agent.process_feedback(state_with_feedback)

        # Verify defaults to complete
        assert route == WorkflowRoute.COMPLETE

    @pytest.mark.asyncio
    async def test_llm_failure_returns_complete(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that LLM failure defaults to COMPLETE."""
        mock_model_router.generate.side_effect = Exception("LLM unavailable")

        # Execute
        route = await coordinator_agent.process_feedback(state_with_feedback)

        # Verify defaults to complete
        assert route == WorkflowRoute.COMPLETE

    @pytest.mark.asyncio
    async def test_context_update_failure_uses_raw_feedback(
        self,
        coordinator_agent,
        state_with_feedback,
        mock_model_router
    ):
        """Test that context update failure stores raw feedback."""
        state_with_feedback["human_feedback"] = ["dig deeper please"]

        routing_response = MagicMock()
        routing_response.content = '{"route": "GATHERER", "reasoning": "More data", "context_for_retry": ""}'

        mock_model_router.generate.side_effect = [
            routing_response,
            Exception("LLM failed for context")  # Second call fails
        ]

        # Execute
        await coordinator_agent.process_feedback(state_with_feedback)

        # Verify raw feedback stored as fallback
        assert state_with_feedback["feedback_context"] == "dig deeper please"


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PROCESS METHOD TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinatorDefaultProcess:
    """Test default process method delegation."""

    @pytest.mark.asyncio
    async def test_delegates_to_entry_when_not_complete(
        self,
        coordinator_agent,
        initial_state,
        mock_model_router
    ):
        """Test that process() delegates to process_entry when coordinator not complete."""
        # Mock responses for entry
        validation_response = MagicMock()
        validation_response.content = '{"is_valid": true, "errors": [], "suggested_corrections": {}, "concerns": []}'

        normalization_response = MagicMock()
        normalization_response.content = 'Acme Corporation'

        questioning_response = MagicMock()
        questioning_response.content = '{"needs_clarification": false, "questions": null, "reasoning": "Clear"}'

        mock_model_router.generate.side_effect = [
            validation_response,
            normalization_response,
            questioning_response
        ]

        assert initial_state["progress"].coordinator_complete is False

        # Execute
        await coordinator_agent.process(initial_state)

        # Verify entry was processed
        assert initial_state["progress"].coordinator_complete is True

    @pytest.mark.asyncio
    async def test_delegates_to_exit_when_validator_complete(
        self,
        coordinator_agent,
        state_with_opportunities,
        mock_model_router
    ):
        """Test that process() delegates to process_exit when validator complete."""
        # Ensure current_report is None to trigger exit
        state_with_opportunities["current_report"] = None

        report_response = MagicMock()
        report_response.content = "Test Report"
        mock_model_router.generate.return_value = report_response

        # Execute
        await coordinator_agent.process(state_with_opportunities)

        # Verify exit was processed
        assert state_with_opportunities["current_report"] is not None
        assert state_with_opportunities["waiting_for_human"] is True


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW ROUTE ENUM TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkflowRouteEnum:
    """Test WorkflowRoute enum."""

    def test_route_values(self):
        """Test that route enum has expected values."""
        assert WorkflowRoute.GATHERER.value == "gatherer"
        assert WorkflowRoute.IDENTIFIER.value == "identifier"
        assert WorkflowRoute.VALIDATOR.value == "validator"
        assert WorkflowRoute.COMPLETE.value == "complete"

    def test_route_is_string_enum(self):
        """Test that route enum values are strings."""
        assert isinstance(WorkflowRoute.GATHERER, str)
        assert WorkflowRoute.GATHERER == "gatherer"
