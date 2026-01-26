"""
Strategy Validator Agent - Validates and scores opportunities.
Phase 3: Agent Implementation
"""
import json
from typing import Any

import structlog

from src.core.base_agent import StatelessAgent
from src.models.state import ResearchState, Signal, Opportunity, OpportunityConfidence
from src.core.model_router import ModelRouter

logger = structlog.get_logger(__name__)


class ValidatorAgent(StatelessAgent):
    """
    Strategy Validator Agent - Validates opportunities and assesses risks.

    This agent analyzes identified opportunities to:
    1. Assess competitive risks from signals and market context
    2. Re-score confidence for each opportunity with risk factors
    3. Filter opportunities by confidence threshold (>0.6)
    4. Provide validated, high-quality opportunities for final report

    Uses Tier 2 model (Groq 8B, complexity=6) for nuanced risk assessment.

    Responsibilities:
    - Identify competitive risks from signals and opportunities
    - Re-evaluate confidence scores with additional context
    - Filter low-confidence opportunities (<0.6 threshold)
    - Handle feedback loops for re-validation

    Modifies ResearchState in-place:
    - state["validated_opportunities"] - Filtered list of high-confidence opportunities
    - state["competitive_risks"] - List of identified risk strings
    - state["progress"].validator_complete = True
    """

    # Confidence threshold for opportunity validation
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, model_router: ModelRouter):
        """
        Initialize Validator Agent.

        Args:
            model_router: Model router for LLM reasoning (Tier 2 Groq 8B)
        """
        super().__init__(name="validator")
        self.model_router = model_router

    async def process(self, state: ResearchState) -> None:
        """
        Validate opportunities and assess competitive risks.

        This method:
        1. Gets opportunities from IdentifierAgent
        2. Assesses competitive risks using LLM
        3. Re-scores confidence for each opportunity with risk context
        4. Filters opportunities by confidence threshold (>0.6)
        5. Populates validated results

        Args:
            state: Current research state (modified in-place)
        """
        account = state["account_name"]
        industry = state.get("industry", "")
        opportunities = state.get("opportunities", [])
        signals = state.get("signals", [])
        feedback_context = state.get("feedback_context")

        self.logger.info(
            "validator_started",
            account=account,
            opportunities_count=len(opportunities),
            signals_count=len(signals),
            has_feedback=feedback_context is not None
        )

        # Handle empty opportunities
        if not opportunities:
            self.logger.warning("no_opportunities_to_validate", account=account)
            state["validated_opportunities"] = []
            state["competitive_risks"] = []
            state["progress"].validator_complete = True
            return

        # Step 1: Assess competitive risks
        risks = await self._assess_risks(
            account_name=account,
            industry=industry,
            signals=signals,
            opportunities=opportunities,
            feedback_context=feedback_context
        )

        self.logger.info("risks_assessed", count=len(risks))

        # Step 2: Re-score opportunities with risk context
        scored_opportunities = await self._score_opportunities(
            opportunities=opportunities,
            risks=risks,
            state=state,
            feedback_context=feedback_context
        )

        self.logger.info("opportunities_scored", count=len(scored_opportunities))

        # Step 3: Filter by confidence threshold
        validated = [
            opp for opp in scored_opportunities
            if opp.confidence_score > self.CONFIDENCE_THRESHOLD
        ]

        self.logger.info(
            "opportunities_filtered",
            total=len(scored_opportunities),
            validated=len(validated),
            filtered_out=len(scored_opportunities) - len(validated)
        )

        # Step 4: Store results
        state["validated_opportunities"] = validated
        state["competitive_risks"] = risks
        state["progress"].validator_complete = True

        self.logger.info(
            "validator_completed",
            validated_count=len(validated),
            risks_count=len(risks),
            high_confidence=sum(1 for o in validated if o.confidence == OpportunityConfidence.HIGH),
            medium_confidence=sum(1 for o in validated if o.confidence == OpportunityConfidence.MEDIUM),
            low_confidence=sum(1 for o in validated if o.confidence == OpportunityConfidence.LOW)
        )

    async def _assess_risks(
        self,
        account_name: str,
        industry: str,
        signals: list[Signal],
        opportunities: list[Opportunity],
        feedback_context: str | None = None
    ) -> list[str]:
        """
        Assess competitive and market risks using LLM.

        Analyzes signals and opportunities to identify:
        - Competitor mentions and existing relationships
        - Budget constraints and timing issues
        - Technical blockers or integration challenges
        - Market/industry-specific risks

        Args:
            account_name: Company being researched
            industry: Company industry
            signals: Research signals from gatherer
            opportunities: Identified opportunities
            feedback_context: Optional feedback for retry

        Returns:
            List of risk description strings
        """
        # Build signal context (focus on potentially risky signals)
        signal_summaries = []
        for signal in signals[:20]:
            content = signal.content[:400] if isinstance(signal.content, str) else str(signal.content)[:400]
            signal_summaries.append(f"- [{signal.signal_type}] {content}")

        # Build opportunity context
        opportunity_summaries = []
        for opp in opportunities[:10]:
            opportunity_summaries.append(
                f"- {opp.product_name}: {opp.rationale[:200]}... "
                f"(Confidence: {opp.confidence.value}, Score: {opp.confidence_score:.2f})"
            )

        # Build feedback instruction if retrying
        feedback_instruction = ""
        if feedback_context:
            feedback_instruction = f"""
IMPORTANT: This is a retry based on human feedback:
{feedback_context}

Adjust your risk analysis to address this feedback specifically.
"""

        prompt = f"""Analyze risks for sales opportunities at {account_name} ({industry}).

RESEARCH SIGNALS:
{chr(10).join(signal_summaries) if signal_summaries else "No signals available"}

IDENTIFIED OPPORTUNITIES:
{chr(10).join(opportunity_summaries) if opportunity_summaries else "No opportunities identified"}
{feedback_instruction}
Identify risks that could prevent or delay sales success:

1. COMPETITIVE RISKS: Existing vendor relationships, competitor evaluations, switching costs
2. BUDGET/TIMING RISKS: Budget cycles, recent purchases, procurement freezes
3. TECHNICAL RISKS: Integration challenges, tech stack incompatibilities
4. ORGANIZATIONAL RISKS: Restructuring, leadership changes, competing priorities
5. MARKET RISKS: Industry trends, regulatory changes affecting {industry}

Be specific and cite evidence from signals where possible.
Only include risks that have supporting evidence or strong indicators.

Return JSON:
{{
    "risks": [
        "Strong competitor presence: Signals indicate active relationship with Competitor X based on [evidence]",
        "Budget timing concern: Recent large infrastructure investment suggests limited budget for new tools",
        "Integration complexity: Current tech stack includes legacy systems that may complicate deployment",
        ...
    ]
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B
                use_cache=True
            )

            result = json.loads(response.content)
            risks = result.get("risks", [])

            # Validate risks are strings
            risks = [str(r) for r in risks if r]

            return risks

        except json.JSONDecodeError as e:
            self.logger.warning("risks_json_parse_failed", error=str(e))
            # Fallback: return generic risks
            return ["Unable to fully assess competitive landscape - recommend manual review"]
        except Exception as e:
            self.logger.error("risk_assessment_failed", error=str(e))
            return []

    async def _score_opportunities(
        self,
        opportunities: list[Opportunity],
        risks: list[str],
        state: ResearchState,
        feedback_context: str | None = None
    ) -> list[Opportunity]:
        """
        Re-score opportunities with risk context.

        For each opportunity, uses LLM to:
        - Factor in identified risks
        - Consider evidence quality
        - Evaluate market timing
        - Adjust confidence score

        Args:
            opportunities: Original opportunities from identifier
            risks: Identified competitive/market risks
            state: Current research state
            feedback_context: Optional feedback for retry

        Returns:
            List of Opportunity objects with updated confidence scores
        """
        account_name = state["account_name"]
        industry = state.get("industry", "")

        # Build opportunity data for prompt
        opportunities_data = []
        for opp in opportunities:
            evidence_summary = "; ".join(
                sig.content[:100] for sig in opp.evidence[:3]
            ) if opp.evidence else "No direct evidence"

            opportunities_data.append({
                "product_name": opp.product_name,
                "rationale": opp.rationale[:300],
                "current_confidence": opp.confidence.value,
                "current_score": opp.confidence_score,
                "evidence_count": len(opp.evidence),
                "evidence_summary": evidence_summary,
                "existing_risks": opp.risks[:3]
            })

        # Build feedback instruction
        feedback_instruction = ""
        if feedback_context:
            feedback_instruction = f"""
IMPORTANT: This is a retry based on human feedback:
{feedback_context}

Adjust your scoring accordingly.
"""

        prompt = f"""Re-evaluate confidence scores for opportunities at {account_name} ({industry}).

OPPORTUNITIES TO SCORE:
{json.dumps(opportunities_data, indent=2)}

IDENTIFIED RISKS:
{chr(10).join(f"- {r}" for r in risks) if risks else "No significant risks identified"}
{feedback_instruction}
For each opportunity, provide an updated confidence score (0.0-1.0) considering:

1. EVIDENCE QUALITY: How strong/reliable is the supporting evidence?
2. RISK IMPACT: How do identified risks affect this specific opportunity?
3. TIMING: Is the timing favorable based on signals?
4. FIT STRENGTH: How well does the product match the actual need?

Scoring guidelines:
- 0.8-1.0: Strong evidence, minimal risks, clear need, good timing
- 0.6-0.8: Moderate evidence, manageable risks, likely need
- 0.4-0.6: Limited evidence, significant risks, uncertain need
- 0.0-0.4: Weak evidence, high risks, speculative

Return JSON with product_name and new_score for each:
{{
    "scored_opportunities": [
        {{"product_name": "Product Name", "new_score": 0.75, "score_rationale": "Strong evidence from hiring patterns..."}},
        ...
    ]
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B
                use_cache=False  # Don't cache scoring
            )

            result = json.loads(response.content)
            scored_data = result.get("scored_opportunities", [])

            # Create a lookup for new scores
            score_lookup = {
                item["product_name"]: (
                    float(item.get("new_score", 0.5)),
                    item.get("score_rationale", "")
                )
                for item in scored_data
            }

            # Update opportunities with new scores
            updated_opportunities = []
            for opp in opportunities:
                if opp.product_name in score_lookup:
                    new_score, rationale = score_lookup[opp.product_name]
                    # Clamp score to valid range
                    new_score = max(0.0, min(1.0, new_score))

                    # Determine new confidence enum based on score
                    if new_score >= 0.7:
                        new_confidence = OpportunityConfidence.HIGH
                    elif new_score >= 0.4:
                        new_confidence = OpportunityConfidence.MEDIUM
                    else:
                        new_confidence = OpportunityConfidence.LOW

                    # Create updated opportunity
                    updated_opp = Opportunity(
                        product_name=opp.product_name,
                        rationale=opp.rationale,
                        evidence=opp.evidence,
                        target_persona=opp.target_persona,
                        talking_points=opp.talking_points,
                        estimated_value=opp.estimated_value,
                        risks=opp.risks + ([rationale] if rationale and rationale not in opp.risks else []),
                        confidence=new_confidence,
                        confidence_score=new_score
                    )
                    updated_opportunities.append(updated_opp)
                else:
                    # Keep original if not in LLM response
                    updated_opportunities.append(opp)

            return updated_opportunities

        except json.JSONDecodeError as e:
            self.logger.warning("scoring_json_parse_failed", error=str(e))
            # Fallback: return original opportunities unchanged
            return opportunities
        except Exception as e:
            self.logger.error("opportunity_scoring_failed", error=str(e))
            # Graceful degradation: return originals
            return opportunities

    def get_complexity(self, state: ResearchState) -> int:
        """
        Get task complexity for model routing.

        ValidatorAgent performs nuanced reasoning to:
        - Assess competitive and market risks
        - Re-evaluate confidence with multiple factors
        - Make filtering decisions

        This requires Tier 2 (Groq 8B) for quality reasoning.

        Args:
            state: Current research state

        Returns:
            Complexity score (1-10). Validator returns 6 (Tier 2: Groq 8B)
        """
        return 6  # Nuanced reasoning (Tier 2: Groq 8B)
