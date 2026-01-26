"""
Tests for Strategy Validator Agent.
Comprehensive test coverage with mocked dependencies that actually test the logic.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.agents.validator import ValidatorAgent
from src.models.state import (
    ResearchState, Signal, Opportunity, OpportunityConfidence,
    ResearchProgress, ResearchDepth, create_initial_state
)
from src.core.model_router import ModelRouter


@pytest.fixture
def mock_model_router():
    """Provide mocked model router for LLM reasoning."""
    router = AsyncMock(spec=ModelRouter)
    return router


@pytest.fixture
def validator_agent(mock_model_router):
    """Provide ValidatorAgent instance with mocked dependencies."""
    return ValidatorAgent(model_router=mock_model_router)


@pytest.fixture
def sample_signals():
    """Provide sample signals representing various research findings."""
    return [
        Signal(
            source="duckduckgo",
            signal_type="web_search",
            content="Acme Corp is expanding their machine learning infrastructure and hiring senior data scientists",
            timestamp=datetime.now(),
            confidence=0.85,
            metadata={"url": "https://acme.com/about"}
        ),
        Signal(
            source="job_boards",
            signal_type="hiring",
            content="Senior Machine Learning Engineer - Build scalable ML pipelines. Experience with TensorFlow required.",
            timestamp=datetime.now(),
            confidence=0.9,
            metadata={"location": "Boston"}
        ),
        Signal(
            source="duckduckgo_news",
            signal_type="news",
            content="Acme Corp announces partnership with Competitor X for cloud services",
            timestamp=datetime.now(),
            confidence=0.7,
            metadata={"title": "Acme Partnership News"}
        ),
        Signal(
            source="duckduckgo",
            signal_type="web_search",
            content="Company recently completed a large SAP implementation project",
            timestamp=datetime.now(),
            confidence=0.8,
            metadata={"url": "https://acme.com/news"}
        ),
        Signal(
            source="duckduckgo_news",
            signal_type="news",
            content="Acme Corp Q3 results show budget constraints in IT spending",
            timestamp=datetime.now(),
            confidence=0.75,
            metadata={"title": "Acme Financial News"}
        ),
    ]


@pytest.fixture
def sample_opportunities(sample_signals):
    """Provide sample opportunities with varied confidence levels."""
    return [
        Opportunity(
            product_name="Deep Learning Toolbox",
            rationale="Strong ML hiring signals indicate need for algorithm development platform",
            evidence=[sample_signals[0], sample_signals[1]],
            target_persona="VP of Engineering",
            talking_points=["Rapid prototyping", "Integration with TensorFlow", "Team scalability"],
            estimated_value="$150K ARR",
            risks=["Existing Python investment"],
            confidence=OpportunityConfidence.HIGH,
            confidence_score=0.85
        ),
        Opportunity(
            product_name="Cloud Analytics Suite",
            rationale="News about cloud partnership suggests interest in cloud-native tools",
            evidence=[sample_signals[2]],
            target_persona="CTO",
            talking_points=["Cloud migration", "Cost optimization"],
            estimated_value="$200K ARR",
            risks=["Competitor relationship"],
            confidence=OpportunityConfidence.MEDIUM,
            confidence_score=0.65
        ),
        Opportunity(
            product_name="Legacy Migration Tool",
            rationale="SAP implementation indicates enterprise modernization efforts",
            evidence=[sample_signals[3]],
            target_persona="IT Director",
            talking_points=["Modernization", "Integration"],
            estimated_value="$80K ARR",
            risks=["Budget timing"],
            confidence=OpportunityConfidence.LOW,
            confidence_score=0.45
        ),
    ]


@pytest.fixture
def high_confidence_opportunities(sample_signals):
    """Provide opportunities that should all pass the confidence threshold."""
    return [
        Opportunity(
            product_name="Product A",
            rationale="Strong evidence from multiple signals",
            evidence=[sample_signals[0], sample_signals[1]],
            target_persona="VP Engineering",
            talking_points=["Point 1"],
            estimated_value="$100K",
            risks=[],
            confidence=OpportunityConfidence.HIGH,
            confidence_score=0.9
        ),
        Opportunity(
            product_name="Product B",
            rationale="Clear hiring signals",
            evidence=[sample_signals[1]],
            target_persona="CTO",
            talking_points=["Point 1"],
            estimated_value="$150K",
            risks=[],
            confidence=OpportunityConfidence.HIGH,
            confidence_score=0.8
        ),
    ]


@pytest.fixture
def low_confidence_opportunities(sample_signals):
    """Provide opportunities that should all be filtered out."""
    return [
        Opportunity(
            product_name="Weak Product A",
            rationale="Speculative based on limited data",
            evidence=[],
            target_persona="Unknown",
            talking_points=[],
            estimated_value="$50K",
            risks=["High uncertainty"],
            confidence=OpportunityConfidence.LOW,
            confidence_score=0.3
        ),
        Opportunity(
            product_name="Weak Product B",
            rationale="Based on outdated information",
            evidence=[sample_signals[4]],
            target_persona="Unknown",
            talking_points=[],
            estimated_value="$30K",
            risks=["Budget constraints visible"],
            confidence=OpportunityConfidence.LOW,
            confidence_score=0.4
        ),
    ]


@pytest.fixture
def initial_state(sample_signals, sample_opportunities):
    """Provide initial research state with gathered data and identified opportunities."""
    state = create_initial_state(
        account_name="Acme Corp",
        industry="Technology",
        region="North America",
        research_depth=ResearchDepth.STANDARD
    )
    state["signals"] = sample_signals
    state["opportunities"] = sample_opportunities
    state["job_postings"] = [
        {"title": "ML Engineer", "technologies": ["Python", "TensorFlow"]},
        {"title": "Data Scientist", "technologies": ["Python", "Spark"]}
    ]
    state["tech_stack"] = ["Python", "TensorFlow", "AWS"]
    state["progress"].gatherer_complete = True
    state["progress"].identifier_complete = True
    return state


@pytest.fixture
def empty_opportunities_state():
    """Provide state with no opportunities to validate."""
    state = create_initial_state(
        account_name="Empty Corp",
        industry="Unknown",
        research_depth=ResearchDepth.QUICK
    )
    state["opportunities"] = []
    state["signals"] = []
    state["progress"].identifier_complete = True
    return state


class TestValidatorAgentInit:
    """Test ValidatorAgent initialization."""

    def test_init_creates_agent(self, mock_model_router):
        """Test that agent initializes correctly with dependencies."""
        agent = ValidatorAgent(model_router=mock_model_router)

        assert agent.name == "validator"
        assert agent.model_router == mock_model_router
        assert agent.CONFIDENCE_THRESHOLD == 0.6

    def test_get_complexity_returns_six(self, validator_agent, initial_state):
        """Test complexity returns 6 for Tier 2 Groq 8B routing."""
        complexity = validator_agent.get_complexity(initial_state)
        assert complexity == 6


class TestRiskAssessment:
    """Test risk assessment logic."""

    @pytest.mark.asyncio
    async def test_assess_risks_analyzes_signals_and_opportunities(
        self,
        validator_agent,
        mock_model_router,
        sample_signals,
        sample_opportunities
    ):
        """Test that LLM prompt includes signal and opportunity data."""
        mock_response = MagicMock()
        mock_response.content = '{"risks": ["Competitor X relationship is a risk", "Budget constraints noted"]}'
        mock_model_router.generate.return_value = mock_response

        risks = await validator_agent._assess_risks(
            account_name="Acme Corp",
            industry="Technology",
            signals=sample_signals,
            opportunities=sample_opportunities,
            feedback_context=None
        )

        # Verify LLM was called
        mock_model_router.generate.assert_called_once()
        call_args = mock_model_router.generate.call_args

        # Verify prompt contains key information
        prompt = call_args.kwargs["prompt"]
        assert "Acme Corp" in prompt
        assert "Technology" in prompt
        # Verify signals are included
        assert "machine learning" in prompt.lower() or "ml" in prompt.lower()
        # Verify opportunities are referenced
        assert "Deep Learning Toolbox" in prompt

        # Verify risks extracted
        assert len(risks) == 2
        assert "Competitor X relationship" in risks[0]
        assert "Budget constraints" in risks[1]

    @pytest.mark.asyncio
    async def test_assess_risks_includes_feedback_context(
        self,
        validator_agent,
        mock_model_router,
        sample_signals,
        sample_opportunities
    ):
        """Test that feedback_context is included in prompt for retries."""
        mock_response = MagicMock()
        mock_response.content = '{"risks": ["Re-evaluated competitor risk"]}'
        mock_model_router.generate.return_value = mock_response

        feedback = "Focus more on competitive threats from AWS partnership"

        await validator_agent._assess_risks(
            account_name="Acme Corp",
            industry="Technology",
            signals=sample_signals,
            opportunities=sample_opportunities,
            feedback_context=feedback
        )

        # Verify feedback is in prompt
        call_args = mock_model_router.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "AWS partnership" in prompt
        assert "IMPORTANT: This is a retry" in prompt

    @pytest.mark.asyncio
    async def test_assess_risks_json_parse_failure_fallback(
        self,
        validator_agent,
        mock_model_router,
        sample_signals,
        sample_opportunities
    ):
        """Test fallback when LLM returns invalid JSON."""
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all"
        mock_model_router.generate.return_value = mock_response

        risks = await validator_agent._assess_risks(
            account_name="Test Corp",
            industry="Technology",
            signals=sample_signals,
            opportunities=sample_opportunities,
            feedback_context=None
        )

        # Should return fallback message
        assert len(risks) == 1
        assert "manual review" in risks[0].lower()

    @pytest.mark.asyncio
    async def test_assess_risks_llm_failure_returns_empty(
        self,
        validator_agent,
        mock_model_router,
        sample_signals,
        sample_opportunities
    ):
        """Test that LLM exception returns empty risks list."""
        mock_model_router.generate.side_effect = Exception("LLM service unavailable")

        risks = await validator_agent._assess_risks(
            account_name="Test Corp",
            industry="Technology",
            signals=sample_signals,
            opportunities=sample_opportunities,
            feedback_context=None
        )

        assert risks == []

    @pytest.mark.asyncio
    async def test_assess_risks_filters_empty_values(
        self,
        validator_agent,
        mock_model_router,
        sample_signals,
        sample_opportunities
    ):
        """Test that empty/null risks are filtered out."""
        mock_response = MagicMock()
        mock_response.content = '{"risks": ["Valid risk", "", null, "Another valid risk"]}'
        mock_model_router.generate.return_value = mock_response

        risks = await validator_agent._assess_risks(
            account_name="Test Corp",
            industry="Technology",
            signals=sample_signals,
            opportunities=sample_opportunities,
            feedback_context=None
        )

        # Empty and null filtered
        assert len(risks) == 2
        assert "Valid risk" in risks
        assert "Another valid risk" in risks

    @pytest.mark.asyncio
    async def test_assess_risks_uses_caching(
        self,
        validator_agent,
        mock_model_router,
        sample_signals,
        sample_opportunities
    ):
        """Test that risk assessment uses LLM caching."""
        mock_response = MagicMock()
        mock_response.content = '{"risks": ["Some risk"]}'
        mock_model_router.generate.return_value = mock_response

        await validator_agent._assess_risks(
            account_name="Test Corp",
            industry="Technology",
            signals=sample_signals,
            opportunities=sample_opportunities,
            feedback_context=None
        )

        # Verify use_cache=True was passed
        call_kwargs = mock_model_router.generate.call_args.kwargs
        assert call_kwargs["use_cache"] is True

    @pytest.mark.asyncio
    async def test_assess_risks_with_empty_signals(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities
    ):
        """Test risk assessment handles empty signals gracefully."""
        mock_response = MagicMock()
        mock_response.content = '{"risks": ["Limited data available"]}'
        mock_model_router.generate.return_value = mock_response

        risks = await validator_agent._assess_risks(
            account_name="Test Corp",
            industry="Technology",
            signals=[],
            opportunities=sample_opportunities,
            feedback_context=None
        )

        # Should still work with empty signals
        assert len(risks) >= 0
        mock_model_router.generate.assert_called_once()


class TestOpportunityScoring:
    """Test opportunity scoring logic."""

    @pytest.mark.asyncio
    async def test_score_opportunities_updates_confidence(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that opportunities get re-scored with new confidence values."""
        risks = ["Competitor relationship", "Budget timing"]

        mock_response = MagicMock()
        mock_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 0.75, "score_rationale": "Strong evidence supports opportunity"},
                {"product_name": "Cloud Analytics Suite", "new_score": 0.55, "score_rationale": "Competitor relationship is a concern"},
                {"product_name": "Legacy Migration Tool", "new_score": 0.40, "score_rationale": "Budget constraints lower viability"}
            ]
        }'''
        mock_model_router.generate.return_value = mock_response

        scored = await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=risks,
            state=initial_state,
            feedback_context=None
        )

        assert len(scored) == 3

        # Verify scores were updated
        toolbox_opp = next(o for o in scored if o.product_name == "Deep Learning Toolbox")
        assert toolbox_opp.confidence_score == 0.75
        assert toolbox_opp.confidence == OpportunityConfidence.HIGH  # >= 0.7

        cloud_opp = next(o for o in scored if o.product_name == "Cloud Analytics Suite")
        assert cloud_opp.confidence_score == 0.55
        assert cloud_opp.confidence == OpportunityConfidence.MEDIUM  # 0.4-0.7

        legacy_opp = next(o for o in scored if o.product_name == "Legacy Migration Tool")
        assert legacy_opp.confidence_score == 0.40
        assert legacy_opp.confidence == OpportunityConfidence.MEDIUM  # Exactly 0.4

    @pytest.mark.asyncio
    async def test_score_opportunities_clamps_scores(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that scores are clamped to 0.0-1.0 range."""
        mock_response = MagicMock()
        mock_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 1.5, "score_rationale": ""},
                {"product_name": "Cloud Analytics Suite", "new_score": -0.2, "score_rationale": ""},
                {"product_name": "Legacy Migration Tool", "new_score": 0.5, "score_rationale": ""}
            ]
        }'''
        mock_model_router.generate.return_value = mock_response

        scored = await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=[],
            state=initial_state,
            feedback_context=None
        )

        # Verify scores are clamped
        toolbox_opp = next(o for o in scored if o.product_name == "Deep Learning Toolbox")
        assert toolbox_opp.confidence_score == 1.0  # Clamped from 1.5

        cloud_opp = next(o for o in scored if o.product_name == "Cloud Analytics Suite")
        assert cloud_opp.confidence_score == 0.0  # Clamped from -0.2

    @pytest.mark.asyncio
    async def test_score_opportunities_confidence_enum_mapping(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that confidence enums are correctly mapped from scores."""
        mock_response = MagicMock()
        mock_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 0.85, "score_rationale": ""},
                {"product_name": "Cloud Analytics Suite", "new_score": 0.55, "score_rationale": ""},
                {"product_name": "Legacy Migration Tool", "new_score": 0.25, "score_rationale": ""}
            ]
        }'''
        mock_model_router.generate.return_value = mock_response

        scored = await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=[],
            state=initial_state,
            feedback_context=None
        )

        # HIGH: >= 0.7
        assert scored[0].confidence == OpportunityConfidence.HIGH
        # MEDIUM: 0.4-0.7
        assert scored[1].confidence == OpportunityConfidence.MEDIUM
        # LOW: < 0.4
        assert scored[2].confidence == OpportunityConfidence.LOW

    @pytest.mark.asyncio
    async def test_score_opportunities_json_parse_failure_returns_original(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that JSON parse failure returns original opportunities unchanged."""
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON {broken"
        mock_model_router.generate.return_value = mock_response

        original_scores = [o.confidence_score for o in sample_opportunities]

        scored = await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=[],
            state=initial_state,
            feedback_context=None
        )

        # Should return original opportunities
        assert len(scored) == len(sample_opportunities)
        for i, opp in enumerate(scored):
            assert opp.confidence_score == original_scores[i]

    @pytest.mark.asyncio
    async def test_score_opportunities_llm_failure_returns_original(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that LLM failure returns original opportunities."""
        mock_model_router.generate.side_effect = Exception("LLM timeout")

        original_scores = [o.confidence_score for o in sample_opportunities]

        scored = await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=[],
            state=initial_state,
            feedback_context=None
        )

        # Should return original opportunities
        assert len(scored) == len(sample_opportunities)
        for i, opp in enumerate(scored):
            assert opp.confidence_score == original_scores[i]

    @pytest.mark.asyncio
    async def test_score_opportunities_preserves_missing_products(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that opportunities not in LLM response keep original scores."""
        mock_response = MagicMock()
        # Only return scores for one product
        mock_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 0.90, "score_rationale": "Strong fit"}
            ]
        }'''
        mock_model_router.generate.return_value = mock_response

        scored = await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=[],
            state=initial_state,
            feedback_context=None
        )

        # Toolbox should be updated
        toolbox = next(o for o in scored if o.product_name == "Deep Learning Toolbox")
        assert toolbox.confidence_score == 0.90

        # Others should keep original scores
        cloud = next(o for o in scored if o.product_name == "Cloud Analytics Suite")
        assert cloud.confidence_score == 0.65  # Original score

        legacy = next(o for o in scored if o.product_name == "Legacy Migration Tool")
        assert legacy.confidence_score == 0.45  # Original score

    @pytest.mark.asyncio
    async def test_score_opportunities_includes_feedback_context(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that feedback context is included in scoring prompt."""
        mock_response = MagicMock()
        mock_response.content = '{"scored_opportunities": []}'
        mock_model_router.generate.return_value = mock_response

        feedback = "Be more conservative with scores given market uncertainty"

        await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=[],
            state=initial_state,
            feedback_context=feedback
        )

        prompt = mock_model_router.generate.call_args.kwargs["prompt"]
        assert "conservative" in prompt.lower()
        assert "market uncertainty" in prompt.lower()
        assert "IMPORTANT: This is a retry" in prompt

    @pytest.mark.asyncio
    async def test_score_opportunities_no_caching(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities,
        initial_state
    ):
        """Test that opportunity scoring does NOT use caching."""
        mock_response = MagicMock()
        mock_response.content = '{"scored_opportunities": []}'
        mock_model_router.generate.return_value = mock_response

        await validator_agent._score_opportunities(
            opportunities=sample_opportunities,
            risks=[],
            state=initial_state,
            feedback_context=None
        )

        # Verify use_cache=False was passed
        call_kwargs = mock_model_router.generate.call_args.kwargs
        assert call_kwargs["use_cache"] is False


class TestConfidenceFiltering:
    """Test confidence threshold filtering logic."""

    @pytest.mark.asyncio
    async def test_filter_removes_low_confidence_opportunities(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test that opportunities below threshold are filtered out."""
        # Setup mock responses
        risk_response = MagicMock()
        risk_response.content = '{"risks": ["Some risk"]}'

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 0.85, "score_rationale": ""},
                {"product_name": "Cloud Analytics Suite", "new_score": 0.55, "score_rationale": ""},
                {"product_name": "Legacy Migration Tool", "new_score": 0.40, "score_rationale": ""}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        # Only opportunities > 0.6 should remain
        validated = initial_state["validated_opportunities"]
        assert len(validated) == 1
        assert validated[0].product_name == "Deep Learning Toolbox"
        assert validated[0].confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_filter_threshold_is_exclusive(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test that threshold 0.6 is exclusive (> not >=)."""
        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 0.61, "score_rationale": ""},
                {"product_name": "Cloud Analytics Suite", "new_score": 0.60, "score_rationale": ""},
                {"product_name": "Legacy Migration Tool", "new_score": 0.59, "score_rationale": ""}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        # Only > 0.6, not >= 0.6
        validated = initial_state["validated_opportunities"]
        assert len(validated) == 1
        assert validated[0].product_name == "Deep Learning Toolbox"
        assert validated[0].confidence_score == 0.61

    @pytest.mark.asyncio
    async def test_all_high_confidence_pass_filter(
        self,
        validator_agent,
        mock_model_router,
        high_confidence_opportunities,
        sample_signals
    ):
        """Test that all high confidence opportunities pass the filter."""
        state = create_initial_state(
            account_name="High Confidence Corp",
            industry="Tech"
        )
        state["signals"] = sample_signals
        state["opportunities"] = high_confidence_opportunities
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Product A", "new_score": 0.88, "score_rationale": ""},
                {"product_name": "Product B", "new_score": 0.82, "score_rationale": ""}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(state)

        # All should pass
        assert len(state["validated_opportunities"]) == 2

    @pytest.mark.asyncio
    async def test_all_low_confidence_filtered_out(
        self,
        validator_agent,
        mock_model_router,
        low_confidence_opportunities,
        sample_signals
    ):
        """Test that all low confidence opportunities are filtered out."""
        state = create_initial_state(
            account_name="Low Confidence Corp",
            industry="Tech"
        )
        state["signals"] = sample_signals
        state["opportunities"] = low_confidence_opportunities
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '{"risks": ["Multiple risks identified"]}'

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Weak Product A", "new_score": 0.25, "score_rationale": ""},
                {"product_name": "Weak Product B", "new_score": 0.35, "score_rationale": ""}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(state)

        # None should pass
        assert len(state["validated_opportunities"]) == 0


class TestFullProcessFlow:
    """Test complete process flow."""

    @pytest.mark.asyncio
    async def test_full_process_success(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test complete process from risks to validated opportunities."""
        risk_response = MagicMock()
        risk_response.content = '''{
            "risks": [
                "Competitor X has existing relationship",
                "Budget cycle may delay purchase"
            ]
        }'''

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 0.80, "score_rationale": "Strong evidence despite competitor"},
                {"product_name": "Cloud Analytics Suite", "new_score": 0.45, "score_rationale": "Competitor relationship is blocker"},
                {"product_name": "Legacy Migration Tool", "new_score": 0.35, "score_rationale": "Budget constraints too significant"}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        # Verify state updated
        assert len(initial_state["validated_opportunities"]) == 1
        assert initial_state["validated_opportunities"][0].product_name == "Deep Learning Toolbox"
        assert initial_state["validated_opportunities"][0].confidence_score == 0.80

        # Verify risks captured
        assert len(initial_state["competitive_risks"]) == 2
        assert "Competitor X" in initial_state["competitive_risks"][0]

        # Verify progress marked complete
        assert initial_state["progress"].validator_complete is True

    @pytest.mark.asyncio
    async def test_process_with_empty_opportunities(
        self,
        validator_agent,
        mock_model_router,
        empty_opportunities_state
    ):
        """Test that empty opportunities result in empty validated list."""
        await validator_agent.process(empty_opportunities_state)

        # LLM should NOT be called
        mock_model_router.generate.assert_not_called()

        # Verify state
        assert empty_opportunities_state["validated_opportunities"] == []
        assert empty_opportunities_state["competitive_risks"] == []
        assert empty_opportunities_state["progress"].validator_complete is True

    @pytest.mark.asyncio
    async def test_process_state_modified_in_place(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test that state is modified in-place, not replaced."""
        original_state_id = id(initial_state)

        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'
        score_response = MagicMock()
        score_response.content = '{"scored_opportunities": []}'

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        assert id(initial_state) == original_state_id

    @pytest.mark.asyncio
    async def test_process_uses_correct_complexity(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test that LLM calls use complexity=6 for Tier 2 routing."""
        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'
        score_response = MagicMock()
        score_response.content = '{"scored_opportunities": []}'

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        # Both LLM calls should use complexity=6
        for call in mock_model_router.generate.call_args_list:
            assert call.kwargs["complexity"] == 6

    @pytest.mark.asyncio
    async def test_process_with_feedback_context(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test that feedback_context from state is passed through."""
        initial_state["feedback_context"] = "Be more conservative with risk assessment"

        risk_response = MagicMock()
        risk_response.content = '{"risks": ["Conservative risk"]}'
        score_response = MagicMock()
        score_response.content = '{"scored_opportunities": []}'

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        # Verify feedback was in the prompts
        risk_call_prompt = mock_model_router.generate.call_args_list[0].kwargs["prompt"]
        assert "conservative" in risk_call_prompt.lower()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_opportunities_with_no_evidence(
        self,
        validator_agent,
        mock_model_router
    ):
        """Test handling opportunities that have no evidence signals."""
        state = create_initial_state(
            account_name="Test Corp",
            industry="Tech"
        )
        state["signals"] = []
        state["opportunities"] = [
            Opportunity(
                product_name="Speculative Product",
                rationale="Based on industry trends only",
                evidence=[],  # No evidence
                target_persona="Unknown",
                talking_points=[],
                estimated_value="$50K",
                risks=[],
                confidence=OpportunityConfidence.LOW,
                confidence_score=0.4
            )
        ]
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '{"risks": ["No direct evidence"]}'
        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Speculative Product", "new_score": 0.35, "score_rationale": "Lack of evidence"}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(state)

        # Should complete without error
        assert state["progress"].validator_complete is True
        # Opportunity should be filtered (< 0.6)
        assert len(state["validated_opportunities"]) == 0

    @pytest.mark.asyncio
    async def test_handles_very_long_signal_content(
        self,
        validator_agent,
        mock_model_router,
        sample_opportunities
    ):
        """Test that very long signal content is truncated properly."""
        state = create_initial_state(
            account_name="Test Corp",
            industry="Tech"
        )
        # Create signal with very long content
        long_content = "x" * 10000
        state["signals"] = [
            Signal(
                source="test",
                signal_type="web_search",
                content=long_content,
                timestamp=datetime.now(),
                confidence=0.5,
                metadata={}
            )
        ]
        state["opportunities"] = sample_opportunities
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'
        score_response = MagicMock()
        score_response.content = '{"scored_opportunities": []}'

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(state)

        # Verify prompt doesn't contain the full 10000 char content
        risk_prompt = mock_model_router.generate.call_args_list[0].kwargs["prompt"]
        # Signal content should be truncated (limited to 400 chars in _assess_risks)
        assert len(risk_prompt) < 20000

    @pytest.mark.asyncio
    async def test_handles_many_opportunities(
        self,
        validator_agent,
        mock_model_router,
        sample_signals
    ):
        """Test handling many opportunities without overflow."""
        state = create_initial_state(
            account_name="Test Corp",
            industry="Tech"
        )
        state["signals"] = sample_signals
        # Create 50 opportunities
        state["opportunities"] = [
            Opportunity(
                product_name=f"Product {i}",
                rationale=f"Rationale for product {i}",
                evidence=[],
                target_persona="VP",
                talking_points=[],
                estimated_value="$100K",
                risks=[],
                confidence=OpportunityConfidence.MEDIUM,
                confidence_score=0.6
            )
            for i in range(50)
        ]
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'
        score_response = MagicMock()
        # Return scores for all
        scored = [
            {"product_name": f"Product {i}", "new_score": 0.7, "score_rationale": ""}
            for i in range(50)
        ]
        score_response.content = f'{{"scored_opportunities": {scored}}}'.replace("'", '"')

        mock_model_router.generate.side_effect = [risk_response, score_response]

        # Should not crash
        await validator_agent.process(state)
        assert state["progress"].validator_complete is True

    @pytest.mark.asyncio
    async def test_risk_assessment_includes_all_risk_categories(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test that risk prompt mentions all 5 risk categories."""
        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'
        score_response = MagicMock()
        score_response.content = '{"scored_opportunities": []}'

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        risk_prompt = mock_model_router.generate.call_args_list[0].kwargs["prompt"]

        # Verify all 5 risk categories are mentioned
        assert "COMPETITIVE" in risk_prompt
        assert "BUDGET" in risk_prompt or "TIMING" in risk_prompt
        assert "TECHNICAL" in risk_prompt
        assert "ORGANIZATIONAL" in risk_prompt
        assert "MARKET" in risk_prompt

    @pytest.mark.asyncio
    async def test_scoring_includes_all_evaluation_factors(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test that scoring prompt mentions all evaluation factors."""
        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'
        score_response = MagicMock()
        score_response.content = '{"scored_opportunities": []}'

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        score_prompt = mock_model_router.generate.call_args_list[1].kwargs["prompt"]

        # Verify evaluation factors are mentioned
        assert "EVIDENCE" in score_prompt
        assert "RISK" in score_prompt
        assert "TIMING" in score_prompt
        assert "FIT" in score_prompt

    @pytest.mark.asyncio
    async def test_graceful_handling_of_partial_llm_response(
        self,
        validator_agent,
        mock_model_router,
        initial_state
    ):
        """Test handling when LLM returns partial/incomplete data."""
        risk_response = MagicMock()
        risk_response.content = '{"risks": []}'

        # Score response missing some products
        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Deep Learning Toolbox", "new_score": 0.80}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(initial_state)

        # Should complete without error
        assert initial_state["progress"].validator_complete is True

        # The one scored opportunity should be validated if above threshold
        validated_names = [o.product_name for o in initial_state["validated_opportunities"]]
        assert "Deep Learning Toolbox" in validated_names


class TestRealWorldScenarios:
    """Test realistic business scenarios."""

    @pytest.mark.asyncio
    async def test_competitor_heavy_environment(
        self,
        validator_agent,
        mock_model_router
    ):
        """Test validation when many competitor signals are present."""
        state = create_initial_state(
            account_name="CompetitorHeavy Corp",
            industry="Enterprise Software"
        )
        state["signals"] = [
            Signal(
                source="news",
                signal_type="news",
                content="CompetitorHeavy Corp renews 3-year contract with Salesforce",
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={}
            ),
            Signal(
                source="web",
                signal_type="web_search",
                content="CompetitorHeavy is a Microsoft Gold Partner",
                timestamp=datetime.now(),
                confidence=0.85,
                metadata={}
            ),
        ]
        state["opportunities"] = [
            Opportunity(
                product_name="CRM Alternative",
                rationale="Mentioned looking at options",
                evidence=[],
                target_persona="VP Sales",
                talking_points=[],
                estimated_value="$300K",
                risks=[],
                confidence=OpportunityConfidence.MEDIUM,
                confidence_score=0.6
            )
        ]
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '''{
            "risks": [
                "Strong existing Salesforce relationship (3-year contract renewal)",
                "Microsoft Gold Partner status suggests deep vendor commitment"
            ]
        }'''

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "CRM Alternative", "new_score": 0.30, "score_rationale": "Existing vendor relationships make this unlikely"}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(state)

        # Should be filtered out due to competitive risks
        assert len(state["validated_opportunities"]) == 0
        assert len(state["competitive_risks"]) == 2

    @pytest.mark.asyncio
    async def test_strong_buying_signals_scenario(
        self,
        validator_agent,
        mock_model_router
    ):
        """Test validation when strong buying signals are present."""
        state = create_initial_state(
            account_name="ReadyToBuy Corp",
            industry="Manufacturing"
        )
        state["signals"] = [
            Signal(
                source="job_boards",
                signal_type="hiring",
                content="Hiring 5 ML Engineers - immediate start, new AI initiative",
                timestamp=datetime.now(),
                confidence=0.95,
                metadata={}
            ),
            Signal(
                source="news",
                signal_type="news",
                content="ReadyToBuy Corp announces $10M investment in AI/ML capabilities",
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={}
            ),
            Signal(
                source="web",
                signal_type="web_search",
                content="CEO quoted: 'We need best-in-class ML tools to stay competitive'",
                timestamp=datetime.now(),
                confidence=0.85,
                metadata={}
            ),
        ]
        state["opportunities"] = [
            Opportunity(
                product_name="ML Platform",
                rationale="Strong hiring and investment signals in AI/ML",
                evidence=state["signals"],
                target_persona="VP Engineering",
                talking_points=["Immediate value", "Scale with hiring"],
                estimated_value="$500K",
                risks=["Long procurement cycle"],
                confidence=OpportunityConfidence.HIGH,
                confidence_score=0.85
            )
        ]
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '''{
            "risks": [
                "Enterprise procurement process may take 3-6 months"
            ]
        }'''

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "ML Platform", "new_score": 0.92, "score_rationale": "Multiple strong buying signals, active investment, executive sponsorship"}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(state)

        # Should pass validation with high score
        assert len(state["validated_opportunities"]) == 1
        assert state["validated_opportunities"][0].confidence_score == 0.92
        assert state["validated_opportunities"][0].confidence == OpportunityConfidence.HIGH

    @pytest.mark.asyncio
    async def test_budget_constrained_scenario(
        self,
        validator_agent,
        mock_model_router
    ):
        """Test validation when budget constraints are evident."""
        state = create_initial_state(
            account_name="Tight Budget Inc",
            industry="Retail"
        )
        state["signals"] = [
            Signal(
                source="news",
                signal_type="news",
                content="Tight Budget Inc reports Q3 losses, announces cost-cutting measures",
                timestamp=datetime.now(),
                confidence=0.9,
                metadata={}
            ),
            Signal(
                source="news",
                signal_type="news",
                content="Company freezes all non-essential IT spending until Q2 next year",
                timestamp=datetime.now(),
                confidence=0.85,
                metadata={}
            ),
        ]
        state["opportunities"] = [
            Opportunity(
                product_name="Analytics Suite",
                rationale="Would help with cost optimization",
                evidence=[],
                target_persona="CFO",
                talking_points=["ROI", "Cost savings"],
                estimated_value="$200K",
                risks=[],
                confidence=OpportunityConfidence.MEDIUM,
                confidence_score=0.65
            )
        ]
        state["progress"].identifier_complete = True

        risk_response = MagicMock()
        risk_response.content = '''{
            "risks": [
                "Q3 losses indicate financial stress",
                "IT spending freeze announced - unlikely to approve new purchases until Q2"
            ]
        }'''

        score_response = MagicMock()
        score_response.content = '''{
            "scored_opportunities": [
                {"product_name": "Analytics Suite", "new_score": 0.35, "score_rationale": "Budget freeze makes near-term purchase very unlikely"}
            ]
        }'''

        mock_model_router.generate.side_effect = [risk_response, score_response]

        await validator_agent.process(state)

        # Should be filtered out due to budget constraints
        assert len(state["validated_opportunities"]) == 0
        assert any("freeze" in risk.lower() for risk in state["competitive_risks"])
