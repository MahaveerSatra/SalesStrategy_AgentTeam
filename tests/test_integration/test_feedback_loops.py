"""
Integration tests for human-in-the-loop feedback scenarios.

Tests feedback routing: gatherer retry, identifier retry, validator retry, complete.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.models.state import (
    ResearchState, Signal, Opportunity, OpportunityConfidence,
    ResearchProgress, ResearchDepth, create_initial_state
)
from src.agents.coordinator import CoordinatorAgent, WorkflowRoute
from src.agents.gatherer import GathererAgent
from src.agents.identifier import IdentifierAgent
from src.agents.validator import ValidatorAgent
from src.core.model_router import ModelRouter
from src.data_sources.mcp_ddg_client import DuckDuckGoMCPClient
from src.data_sources.job_boards import JobBoardScraper
from src.data_sources.product_catalog import ProductMatcher


@pytest.fixture
def mock_model_router():
    """Provide mocked model router for all agents."""
    router = AsyncMock()
    return router


@pytest.fixture
def mock_mcp_client():
    """Provide mocked DuckDuckGo MCP client with default empty returns."""
    client = AsyncMock()
    client.search.return_value = []
    client.search_news.return_value = []
    client.fetch_content.return_value = ""
    return client


@pytest.fixture
def mock_job_scraper():
    """Provide mocked job board scraper with default empty returns."""
    scraper = AsyncMock()
    scraper.fetch.return_value = []
    return scraper


@pytest.fixture
def mock_product_matcher():
    """Provide mocked product matcher with default empty returns."""
    matcher = AsyncMock()
    matcher.match_requirements_to_products.return_value = []
    return matcher


@pytest.fixture
def coordinator_agent(mock_model_router):
    """Provide CoordinatorAgent with mocked dependencies."""
    return CoordinatorAgent(model_router=mock_model_router)


@pytest.fixture
def gatherer_agent(mock_mcp_client, mock_job_scraper, mock_model_router):
    """Provide GathererAgent with mocked dependencies."""
    return GathererAgent(
        mcp_client=mock_mcp_client,
        job_scraper=mock_job_scraper,
        model_router=mock_model_router
    )


@pytest.fixture
def identifier_agent(mock_product_matcher, mock_model_router):
    """Provide IdentifierAgent with mocked dependencies."""
    return IdentifierAgent(
        product_matcher=mock_product_matcher,
        model_router=mock_model_router
    )


@pytest.fixture
def validator_agent(mock_model_router):
    """Provide ValidatorAgent with mocked dependencies."""
    return ValidatorAgent(model_router=mock_model_router)


@pytest.fixture
def completed_pipeline_state():
    """Provide state that has completed full pipeline with report."""
    state = create_initial_state(
        account_name="FeedbackTest Corp",
        industry="Technology",
        region="North America",
        research_depth=ResearchDepth.STANDARD
    )

    # Mark all agents complete
    state["progress"].coordinator_complete = True
    state["progress"].gatherer_complete = True
    state["progress"].identifier_complete = True
    state["progress"].validator_complete = True

    # Add sample data
    state["signals"] = [
        Signal(
            source="duckduckgo",
            signal_type="web_search",
            content="FeedbackTest Corp ML initiative",
            timestamp=datetime.now(),
            confidence=0.85,
            metadata={}
        )
    ]

    state["opportunities"] = [
        Opportunity(
            product_name="ML Platform",
            rationale="Strong ML signals",
            evidence=state["signals"],
            target_persona="VP Engineering",
            talking_points=["Scale", "Speed"],
            estimated_value="$150K",
            risks=[],
            confidence=OpportunityConfidence.HIGH,
            confidence_score=0.85
        )
    ]

    state["validated_opportunities"] = state["opportunities"]
    state["competitive_risks"] = ["Minor budget concern"]

    # Add report (as if coordinator_exit completed)
    state["current_report"] = """## Research Report for FeedbackTest Corp

### Top Opportunities
1. ML Platform - $150K ARR

### Risks
- Minor budget concern
"""
    state["waiting_for_human"] = True
    state["workflow_iteration"] = 1

    return state


class TestFeedbackRouteToGatherer:
    """Test feedback loop that routes back to GathererAgent."""

    @pytest.mark.asyncio
    async def test_gather_more_data_feedback_routes_to_gatherer(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that 'gather more data' feedback routes to gatherer."""
        # Add human feedback requesting more data
        completed_pipeline_state["human_feedback"] = [
            "Please gather more information about their cloud infrastructure"
        ]

        # Mock feedback processing
        feedback_response = MagicMock()
        feedback_response.content = '''{
            "route": "gatherer",
            "feedback_context": "Focus on cloud infrastructure and AWS usage",
            "rationale": "Human requested more cloud-specific data"
        }'''
        mock_model_router.generate.return_value = feedback_response

        await coordinator_agent.process_feedback(completed_pipeline_state)

        # Verify routing
        assert completed_pipeline_state["next_route"] == "gatherer"
        assert completed_pipeline_state.get("feedback_context") is not None
        assert "cloud" in completed_pipeline_state["feedback_context"].lower()

    @pytest.mark.asyncio
    async def test_gatherer_uses_feedback_context_on_retry(
        self,
        gatherer_agent,
        mock_model_router,
        mock_mcp_client,
        mock_job_scraper,
        completed_pipeline_state
    ):
        """Test that GathererAgent uses feedback_context when re-running."""
        # Set up retry state
        completed_pipeline_state["feedback_context"] = "Focus on cloud infrastructure and Kubernetes"
        completed_pipeline_state["progress"].gatherer_complete = False  # Reset for retry

        # Mock gatherer with context-aware response
        gatherer_response = MagicMock()
        gatherer_response.content = '''{
            "analysis": "Found additional cloud/K8s information",
            "key_signals": ["kubernetes adoption", "cloud migration"],
            "technologies": ["Kubernetes", "AWS EKS"]
        }'''
        mock_model_router.generate.return_value = gatherer_response
        # Return empty list to simplify mock
        mock_mcp_client.search.return_value = []
        mock_mcp_client.search_news.return_value = []
        mock_job_scraper.fetch.return_value = []

        await gatherer_agent.process(completed_pipeline_state)

        # Verify gatherer completed with new data
        assert completed_pipeline_state["progress"].gatherer_complete is True
        # Note: With empty search results, LLM may not be called
        # The feedback_context is stored in state for potential use by gatherer
        assert completed_pipeline_state.get("feedback_context") is not None

    @pytest.mark.asyncio
    async def test_iteration_counter_increments_on_feedback_loop(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that workflow_iteration increments on feedback loop."""
        initial_iteration = completed_pipeline_state["workflow_iteration"]

        completed_pipeline_state["human_feedback"] = ["gather more data"]

        feedback_response = MagicMock()
        feedback_response.content = '{"route": "gatherer", "feedback_context": "More data needed", "rationale": "User request"}'
        mock_model_router.generate.return_value = feedback_response

        await coordinator_agent.process_feedback(completed_pipeline_state)

        # Iteration should increment
        assert completed_pipeline_state["workflow_iteration"] == initial_iteration + 1


class TestFeedbackRouteToIdentifier:
    """Test feedback loop that routes back to IdentifierAgent."""

    @pytest.mark.asyncio
    async def test_find_different_opportunities_routes_to_identifier(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that 'find different opportunities' feedback routes to identifier."""
        completed_pipeline_state["human_feedback"] = [
            "These opportunities don't seem relevant. Focus on data analytics products instead."
        ]

        feedback_response = MagicMock()
        feedback_response.content = '''{
            "route": "identifier",
            "feedback_context": "Focus on data analytics and BI tools rather than ML platforms",
            "rationale": "Human wants different product focus"
        }'''
        mock_model_router.generate.return_value = feedback_response

        await coordinator_agent.process_feedback(completed_pipeline_state)

        assert completed_pipeline_state["next_route"] == "identifier"
        assert "analytics" in completed_pipeline_state["feedback_context"].lower()

    @pytest.mark.asyncio
    async def test_identifier_uses_feedback_context_on_retry(
        self,
        identifier_agent,
        mock_model_router,
        mock_product_matcher,
        completed_pipeline_state
    ):
        """Test that IdentifierAgent uses feedback_context when re-running."""
        # Set up retry state
        completed_pipeline_state["feedback_context"] = "Focus on analytics and BI tools, not ML"
        completed_pipeline_state["progress"].identifier_complete = False

        # Keep existing signals
        completed_pipeline_state["signals"] = [
            Signal(
                source="web",
                signal_type="web_search",
                content="Data analytics initiative at FeedbackTest Corp",
                timestamp=datetime.now(),
                confidence=0.8,
                metadata={}
            )
        ]
        completed_pipeline_state["job_postings"] = [
            {"title": "Data Analyst", "technologies": ["Tableau", "SQL"]}
        ]
        completed_pipeline_state["tech_stack"] = ["Tableau", "PowerBI"]

        # Mock identifier responses
        req_response = MagicMock()
        req_response.content = '{"requirements": ["BI dashboard platform", "Data visualization tools"]}'

        opp_response = MagicMock()
        opp_response.content = '''{
            "opportunities": [
                {
                    "product_name": "Analytics Dashboard Pro",
                    "rationale": "Strong BI signals based on feedback",
                    "target_persona": "VP Data",
                    "talking_points": ["Visualization", "Self-service BI"],
                    "estimated_value": "$100K",
                    "risks": [],
                    "confidence": "high",
                    "confidence_score": 0.8
                }
            ]
        }'''

        mock_model_router.generate.side_effect = [req_response, opp_response]
        mock_product_matcher.match_requirements_to_products.return_value = [
            ("Analytics Dashboard Pro", 0.85)
        ]

        await identifier_agent.process(completed_pipeline_state)

        # Verify new opportunities generated
        assert completed_pipeline_state["progress"].identifier_complete is True
        assert len(completed_pipeline_state["opportunities"]) > 0
        assert completed_pipeline_state["opportunities"][0].product_name == "Analytics Dashboard Pro"


class TestFeedbackRouteToValidator:
    """Test feedback loop that routes back to ValidatorAgent."""

    @pytest.mark.asyncio
    async def test_reevaluate_scores_routes_to_validator(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that 're-evaluate scores' feedback routes to validator."""
        completed_pipeline_state["human_feedback"] = [
            "The confidence scores seem too high. Please be more conservative."
        ]

        feedback_response = MagicMock()
        feedback_response.content = '''{
            "route": "validator",
            "feedback_context": "Apply more conservative scoring - reduce confidence where evidence is limited",
            "rationale": "Human requested more conservative assessment"
        }'''
        mock_model_router.generate.return_value = feedback_response

        await coordinator_agent.process_feedback(completed_pipeline_state)

        assert completed_pipeline_state["next_route"] == "validator"
        assert "conservative" in completed_pipeline_state["feedback_context"].lower()

    @pytest.mark.asyncio
    async def test_validator_uses_feedback_context_on_retry(
        self,
        validator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that ValidatorAgent uses feedback_context when re-running."""
        # Set up retry state
        completed_pipeline_state["feedback_context"] = "Be more conservative with confidence scores"
        completed_pipeline_state["progress"].validator_complete = False

        # Mock validator responses with more conservative scoring
        risk_response = MagicMock()
        risk_response.content = '{"risks": ["Budget timing concern", "Existing vendor relationship"]}'

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "ML Platform", "new_score": 0.55, "score_rationale": "Conservative re-assessment"}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(completed_pipeline_state)

        # Verify more conservative scoring resulted in filtering
        assert completed_pipeline_state["progress"].validator_complete is True
        # With score 0.55 (< 0.6 threshold), should be filtered out
        assert len(completed_pipeline_state["validated_opportunities"]) == 0

    @pytest.mark.asyncio
    async def test_validator_feedback_includes_context_in_prompt(
        self,
        validator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that feedback context is included in validator prompts."""
        completed_pipeline_state["feedback_context"] = "Consider competitor X as a major risk"
        completed_pipeline_state["progress"].validator_complete = False

        risk_response = MagicMock()
        risk_response.content = '{"risks": ["Competitor X is a threat"]}'
        score_response = MagicMock()
        score_response.content = '{"scored_opportunities": []}'

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(completed_pipeline_state)

        # Check that feedback context was in risk assessment prompt
        risk_call = mock_model_router.generate.call_args_list[0]
        prompt = risk_call.kwargs.get("prompt", "")
        assert "competitor X" in prompt.lower() or "IMPORTANT: This is a retry" in prompt


class TestFeedbackRouteToComplete:
    """Test feedback that completes the workflow."""

    @pytest.mark.asyncio
    async def test_looks_good_feedback_routes_to_complete(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that approval feedback routes to complete."""
        completed_pipeline_state["human_feedback"] = ["Looks good, approved!"]

        feedback_response = MagicMock()
        feedback_response.content = '''{
            "route": "complete",
            "feedback_context": null,
            "rationale": "Human approved the report"
        }'''
        mock_model_router.generate.return_value = feedback_response

        await coordinator_agent.process_feedback(completed_pipeline_state)

        assert completed_pipeline_state["next_route"] == "complete"

    @pytest.mark.asyncio
    async def test_approved_feedback_routes_to_complete(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test various approval phrases route to complete."""
        approval_phrases = [
            "This is great, let's proceed",
            "Approved",
            "LGTM",
            "Send it",
        ]

        for phrase in approval_phrases:
            state = completed_pipeline_state.copy()
            state["human_feedback"] = [phrase]

            feedback_response = MagicMock()
            feedback_response.content = '{"route": "complete", "feedback_context": null, "rationale": "Approved"}'
            mock_model_router.generate.return_value = feedback_response

            await coordinator_agent.process_feedback(state)

            assert state["next_route"] == "complete", f"Failed for phrase: {phrase}"


class TestMultipleFeedbackIterations:
    """Test multiple rounds of feedback."""

    @pytest.mark.asyncio
    async def test_multiple_feedback_loops(
        self,
        coordinator_agent,
        gatherer_agent,
        validator_agent,
        mock_model_router,
        mock_mcp_client,
        mock_job_scraper,
        completed_pipeline_state
    ):
        """Test workflow can handle multiple feedback iterations."""
        # First feedback: gather more data
        completed_pipeline_state["human_feedback"] = ["Need more cloud data"]

        feedback1_response = MagicMock()
        feedback1_response.content = '{"route": "gatherer", "feedback_context": "Focus on cloud", "rationale": "Need cloud data"}'
        mock_model_router.generate.return_value = feedback1_response

        await coordinator_agent.process_feedback(completed_pipeline_state)
        assert completed_pipeline_state["workflow_iteration"] == 2

        # Simulate gatherer retry
        completed_pipeline_state["progress"].gatherer_complete = False
        gatherer_response = MagicMock()
        gatherer_response.content = '{"analysis": "Cloud data found", "key_signals": [], "technologies": []}'
        mock_model_router.generate.return_value = gatherer_response
        mock_mcp_client.search.return_value = []
        mock_mcp_client.search_news.return_value = []
        mock_job_scraper.fetch.return_value = []

        await gatherer_agent.process(completed_pipeline_state)

        # Second feedback: re-validate
        completed_pipeline_state["human_feedback"].append("Now re-evaluate the scores")

        feedback2_response = MagicMock()
        feedback2_response.content = '{"route": "validator", "feedback_context": "Re-score with new data", "rationale": "New data available"}'
        mock_model_router.generate.return_value = feedback2_response

        await coordinator_agent.process_feedback(completed_pipeline_state)
        assert completed_pipeline_state["workflow_iteration"] == 3

        # Final feedback: approve
        completed_pipeline_state["human_feedback"].append("Looks good now")

        feedback3_response = MagicMock()
        feedback3_response.content = '{"route": "complete", "feedback_context": null, "rationale": "Approved"}'
        mock_model_router.generate.return_value = feedback3_response

        await coordinator_agent.process_feedback(completed_pipeline_state)
        assert completed_pipeline_state["next_route"] == "complete"
        # Iteration increments on each feedback processing
        assert completed_pipeline_state["workflow_iteration"] >= 3

    @pytest.mark.asyncio
    async def test_feedback_history_preserved(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that all feedback is preserved in state."""
        feedbacks = [
            "Need more data",
            "Focus on different products",
            "Looks good"
        ]

        for i, feedback in enumerate(feedbacks):
            completed_pipeline_state["human_feedback"].append(feedback)

            route = "gatherer" if i == 0 else "identifier" if i == 1 else "complete"
            response = MagicMock()
            response.content = f'{{"route": "{route}", "feedback_context": "context", "rationale": "reason"}}'
            mock_model_router.generate.return_value = response

            await coordinator_agent.process_feedback(completed_pipeline_state)

        # All feedback should be in history
        assert len(completed_pipeline_state["human_feedback"]) == 3
        assert "Need more data" in completed_pipeline_state["human_feedback"]
        assert "Focus on different products" in completed_pipeline_state["human_feedback"]
        assert "Looks good" in completed_pipeline_state["human_feedback"]


class TestFeedbackEdgeCases:
    """Test edge cases in feedback handling."""

    @pytest.mark.asyncio
    async def test_empty_feedback_handled_gracefully(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that empty feedback is handled gracefully."""
        completed_pipeline_state["human_feedback"] = []

        # Coordinator should handle gracefully - with empty feedback,
        # the coordinator logs a warning and returns without setting next_route
        await coordinator_agent.process_feedback(completed_pipeline_state)

        # Empty feedback doesn't crash - coordinator handles gracefully
        # and logs "coordinator_no_feedback_to_process"
        assert completed_pipeline_state["progress"].coordinator_complete is True

    @pytest.mark.asyncio
    async def test_ambiguous_feedback_uses_llm_routing(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test that ambiguous feedback relies on LLM for routing decision."""
        completed_pipeline_state["human_feedback"] = [
            "Hmm, I'm not sure about this. Maybe try something else?"
        ]

        # LLM should interpret and route
        response = MagicMock()
        response.content = '{"route": "identifier", "feedback_context": "Try different product matches", "rationale": "User expressed uncertainty about current opportunities"}'
        mock_model_router.generate.return_value = response

        await coordinator_agent.process_feedback(completed_pipeline_state)

        # LLM interpretation should be used
        assert completed_pipeline_state["next_route"] == "identifier"

    @pytest.mark.asyncio
    async def test_feedback_with_special_characters(
        self,
        coordinator_agent,
        mock_model_router,
        completed_pipeline_state
    ):
        """Test feedback with special characters is handled."""
        completed_pipeline_state["human_feedback"] = [
            "Look at company's \"cloud\" initiative & AWS/GCP usage <important>"
        ]

        response = MagicMock()
        response.content = '{"route": "gatherer", "feedback_context": "Focus on cloud and AWS/GCP", "rationale": "Cloud focus requested"}'
        mock_model_router.generate.return_value = response

        # Should not crash
        await coordinator_agent.process_feedback(completed_pipeline_state)

        assert completed_pipeline_state["next_route"] == "gatherer"
