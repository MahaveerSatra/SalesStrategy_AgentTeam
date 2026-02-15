"""
Strategy Validator Agent - Validates and scores opportunities.
Phase 3: Agent Implementation
"""
import json
from typing import Any

import structlog

from src.core.base_agent import StatelessAgent
from src.utils.json_parsing import extract_json_from_llm_response, JSONParseError
from src.models.state import ResearchState, Signal, Opportunity, OpportunityConfidence
from src.models.llm_schemas import RiskAssessment, OpportunityScoring
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

    def _format_signals_with_ids(self, signals: list[Signal], limit: int = 15) -> str:
        """
        Format signals with [SIG-xxx] IDs for citation tracking.

        Args:
            signals: List of Signal objects
            limit: Max signals to include

        Returns:
            Formatted string with numbered signals
        """
        if not signals:
            return "No signals available"

        formatted = []
        for i, sig in enumerate(signals[:limit], 1):
            content = sig.content[:300] if isinstance(sig.content, str) else str(sig.content)[:300]
            formatted.append(f"[SIG-{i:03d}] ({sig.signal_type}) {content}")

        return "\n".join(formatted)

    def _format_opportunities_with_ids(self, opportunities: list[Opportunity], limit: int = 10) -> str:
        """
        Format opportunities with [OPP-xxx] IDs for citation tracking.

        Args:
            opportunities: List of Opportunity objects
            limit: Max opportunities to include

        Returns:
            Formatted string with numbered opportunities
        """
        if not opportunities:
            return "No opportunities identified"

        formatted = []
        for i, opp in enumerate(opportunities[:limit], 1):
            rationale_short = opp.rationale[:200] if opp.rationale else "No rationale"
            formatted.append(
                f"[OPP-{i:03d}] {opp.product_name}\n"
                f"    Rationale: {rationale_short}\n"
                f"    Confidence: {opp.confidence.value} ({opp.confidence_score:.2f})\n"
                f"    Persona: {opp.target_persona or 'Unknown'}"
            )

        return "\n\n".join(formatted)

    def _format_risks_with_ids(self, risks: list[str], limit: int = 10) -> str:
        """
        Format risks with [RISK-xxx] IDs for citation in talking points.

        Args:
            risks: List of risk strings
            limit: Max risks to include

        Returns:
            Formatted string with numbered risks
        """
        if not risks:
            return "No significant risks identified"

        formatted = []
        for i, risk in enumerate(risks[:limit], 1):
            formatted.append(f"[RISK-{i:03d}] {risk}")

        return "\n".join(formatted)

    def _get_seller_context(self, seller_name: str) -> str:
        """
        Get seller company context for risk/opportunity framing.

        Args:
            seller_name: Name of the seller company

        Returns:
            Formatted string with seller context
        """
        if seller_name.lower() == "mathworks":
            return f"""**SELLER: {seller_name}**
MathWorks develops mathematical computing software for engineers and scientists.
Core products: MATLAB, Simulink, and domain-specific toolboxes.
Key domains: Simulation, AI/ML, Embedded Systems, Test & Verification."""
        else:
            return f"""**SELLER: {seller_name}**
Enterprise software/solutions provider."""

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
        user_context = state.get("user_context", "")
        seller_name = state.get("seller_name", "Our Company")

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
            feedback_context=feedback_context,
            user_context=user_context,
            seller_name=seller_name
        )

        self.logger.info("risks_assessed", count=len(risks))

        # Step 2: Re-score opportunities with risk context
        scored_opportunities = await self._score_opportunities(
            opportunities=opportunities,
            risks=risks,
            state=state,
            feedback_context=feedback_context,
            user_context=user_context,
            seller_name=seller_name
        )

        self.logger.info("opportunities_scored", count=len(scored_opportunities))

        # Step 2.5: Enhance talking points with objection handling
        enhanced_opportunities = await self._enhance_talking_points(
            opportunities=scored_opportunities,
            risks=risks,
            signals=signals,
            account_name=account,
            industry=industry,
            user_context=user_context,
            seller_name=seller_name
        )

        self.logger.info("talking_points_enhanced", count=len(enhanced_opportunities))

        # Step 3: Filter by confidence threshold
        validated = [
            opp for opp in enhanced_opportunities
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
        feedback_context: str | None = None,
        user_context: str = "",
        seller_name: str = "Our Company"
    ) -> list[str]:
        """
        Assess competitive and market risks using LLM with evidence grounding.

        Uses role-based framing and source citations to identify:
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
            user_context: User's sales context/objectives
            seller_name: Seller company name

        Returns:
            List of risk description strings with source citations
        """
        # Format signals and opportunities with IDs for citation
        signals_formatted = self._format_signals_with_ids(signals, limit=20)
        opportunities_formatted = self._format_opportunities_with_ids(opportunities, limit=10)
        seller_context = self._get_seller_context(seller_name)

        # Build feedback section if retrying
        feedback_section = ""
        if feedback_context:
            feedback_section = f"""
═══════════════════════════════════════════════════════════════
COORDINATOR FEEDBACK (Address this in your risk analysis)
═══════════════════════════════════════════════════════════════
{feedback_context}
"""

        prompt = f"""### ROLE
You are a Risk Assessment Analyst at {seller_name}. Your mission is to identify evidence-grounded risks that could block or delay sales success at {account_name}.

### STRATEGIC ALIGNMENT
═══════════════════════════════════════════════════════════════

**YOUR SALES OBJECTIVE:**
{user_context if user_context else "Identify all relevant sales risks for opportunities"}

{seller_context}

**TARGET ACCOUNT:**
- Company: {account_name}
- Industry: {industry}
{feedback_section}
═══════════════════════════════════════════════════════════════
EVIDENCE DATA (Cite these using IDs)
═══════════════════════════════════════════════════════════════

**OPPORTUNITIES TO ASSESS [OPP-xxx]:**
{opportunities_formatted}

**INTELLIGENCE SIGNALS [SIG-xxx]:**
{signals_formatted}

═══════════════════════════════════════════════════════════════
RISK ASSESSMENT PROTOCOL
═══════════════════════════════════════════════════════════════

For each risk category, search evidence for indicators:

1. **COMPETITIVE RISKS** [cite SIG-xxx or OPP-xxx]
   - Existing vendor relationships
   - Competitor product mentions
   - Switching cost indicators

2. **BUDGET/TIMING RISKS** [cite evidence]
   - Budget cycle indicators
   - Recent large purchases
   - Cost-cutting mentions

3. **TECHNICAL RISKS** [cite evidence]
   - Integration challenges
   - Tech stack incompatibilities
   - Legacy system mentions

4. **ORGANIZATIONAL RISKS** [cite evidence]
   - Restructuring signals
   - Leadership changes
   - Competing priorities

5. **MARKET RISKS** [cite INDUSTRY knowledge]
   - Industry trends affecting {industry}
   - Regulatory changes

═══════════════════════════════════════════════════════════════
GROUNDING RULES (CRITICAL)
═══════════════════════════════════════════════════════════════

You are PROHIBITED from:
- Inventing risks without supporting evidence from signals
- Making up competitor names not mentioned in the evidence
- Assuming budget constraints without evidence
- Generic risks that apply to any company

If you cannot find evidence for a risk, you MUST:
- Tag it as [INDUSTRY] and frame it as "Companies in {industry} typically face..."
- Mark it as lower priority

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

❌ BAD (generic, no evidence):
"They probably have budget constraints this quarter"

❌ BAD (invented competitor):
"Strong presence of Competitor X" (when no competitor mentioned in signals)

✅ GOOD (grounded with citation):
"[SIG-003] Recent earnings call mentioned cost-cutting initiatives - budget approval may require executive sponsorship"

✅ GOOD (grounded with opportunity reference):
"[OPP-002] The Simulink opportunity faces integration risk - [SIG-007] indicates legacy FORTRAN codebase that may complicate deployment"

✅ GOOD (industry knowledge tagged):
"[INDUSTRY] Aerospace companies typically have 12-18 month procurement cycles - timing may extend sales cycle"

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return 3-7 evidence-grounded risks. Quality over quantity.

{{
    "risks": [
        "[SIG-xxx] Risk description with specific evidence citation",
        "[OPP-xxx] Risk affecting specific opportunity with evidence",
        "[INDUSTRY] Industry-level risk (only if no direct evidence)"
    ]
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B
                use_cache=True
            )

            # Use robust JSON extraction to handle varied LLM output formats
            raw_result = extract_json_from_llm_response(response.content)

            # Validate with Pydantic for type safety
            result = RiskAssessment.model_validate(raw_result)

            # Already validated as list[str] by Pydantic
            return result.risks

        except (json.JSONDecodeError, JSONParseError) as e:
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
        feedback_context: str | None = None,
        user_context: str = "",
        seller_name: str = "Our Company"
    ) -> list[Opportunity]:
        """
        Re-score opportunities with risk context and user objective alignment.

        For each opportunity, uses LLM to:
        - Factor in identified risks
        - Consider evidence quality
        - Evaluate market timing
        - Score alignment with user's sales objective
        - Adjust confidence score

        Args:
            opportunities: Original opportunities from identifier
            risks: Identified competitive/market risks
            state: Current research state
            feedback_context: Optional feedback for retry
            user_context: User's sales context/objectives
            seller_name: Seller company name

        Returns:
            List of Opportunity objects with updated confidence scores
        """
        account_name = state["account_name"]
        industry = state.get("industry", "")

        # Build opportunity data for prompt
        opportunities_data = []
        for i, opp in enumerate(opportunities, 1):
            evidence_summary = "; ".join(
                sig.content[:100] for sig in opp.evidence[:3]
            ) if opp.evidence else "No direct evidence"

            opportunities_data.append({
                "id": f"OPP-{i:03d}",
                "product_name": opp.product_name,
                "rationale": opp.rationale[:300],
                "current_confidence": opp.confidence.value,
                "current_score": opp.confidence_score,
                "evidence_count": len(opp.evidence),
                "evidence_summary": evidence_summary,
                "existing_risks": opp.risks[:3]
            })

        # Format risks with IDs
        risks_formatted = self._format_risks_with_ids(risks)
        seller_context = self._get_seller_context(seller_name)

        # Build feedback section
        feedback_section = ""
        if feedback_context:
            feedback_section = f"""
═══════════════════════════════════════════════════════════════
COORDINATOR FEEDBACK (Address this in your scoring)
═══════════════════════════════════════════════════════════════
{feedback_context}
"""

        prompt = f"""### ROLE
You are a Sales Strategy Analyst at {seller_name}. Your mission is to objectively re-score opportunities based on evidence quality, risk impact, and alignment with the user's sales objectives.

### STRATEGIC ALIGNMENT
═══════════════════════════════════════════════════════════════

**USER'S SALES OBJECTIVE (CRITICAL FOR SCORING):**
{user_context if user_context else "No specific objective provided - score based on general fit"}

{seller_context}

**TARGET ACCOUNT:**
- Company: {account_name}
- Industry: {industry}
{feedback_section}
═══════════════════════════════════════════════════════════════
OPPORTUNITIES TO SCORE
═══════════════════════════════════════════════════════════════
{json.dumps(opportunities_data, indent=2)}

═══════════════════════════════════════════════════════════════
IDENTIFIED RISKS [RISK-xxx]
═══════════════════════════════════════════════════════════════
{risks_formatted}

═══════════════════════════════════════════════════════════════
SCORING CRITERIA (Apply ALL factors)
═══════════════════════════════════════════════════════════════

For each opportunity, evaluate and adjust score based on:

1. **EVIDENCE QUALITY** (Base score impact: ±0.2)
   - Strong evidence (multiple sources, specific quotes): +0.1 to +0.2
   - Weak/generic evidence: -0.1 to -0.2

2. **RISK IMPACT** (Adjustment: -0.1 to -0.3)
   - Check if any [RISK-xxx] directly affects this opportunity
   - High-impact risk: -0.2 to -0.3
   - Moderate risk: -0.1

3. **TIMING SIGNALS** (Adjustment: ±0.1)
   - Active hiring in relevant area: +0.1
   - Budget freeze indicators: -0.1

4. **PRODUCT-NEED FIT** (Adjustment: ±0.1)
   - Clear technical match: +0.1
   - Tangential fit: -0.1

5. **USER OBJECTIVE ALIGNMENT** (CRITICAL - Adjustment: ±0.15)
   - HIGH alignment with user's stated objective: +0.1 to +0.15 BONUS
   - MODERATE alignment: no adjustment
   - LOW/NO alignment with user objective: -0.1 to -0.15 PENALTY

═══════════════════════════════════════════════════════════════
SCORING GUIDELINES
═══════════════════════════════════════════════════════════════

Final score interpretation:
- 0.8-1.0: Pursue immediately - strong evidence, minimal risks, aligned with objective
- 0.6-0.8: Qualified opportunity - good fit, manageable risks
- 0.4-0.6: Needs development - limited evidence or significant risks
- 0.0-0.4: Deprioritize - weak evidence, high risks, or misaligned

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return scores with rationale explaining the adjustment factors applied:

{{
    "scored_opportunities": [
        {{
            "product_name": "Product Name",
            "new_score": 0.75,
            "score_rationale": "Evidence quality: +0.1 (3 job posting citations). Risk impact: -0.1 ([RISK-002] budget concerns). User objective alignment: +0.15 (directly matches simulation focus). Final: 0.75"
        }}
    ]
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B
                use_cache=False  # Don't cache scoring
            )

            # Use robust JSON extraction to handle varied LLM output formats
            raw_result = extract_json_from_llm_response(response.content)

            # Validate with Pydantic for type safety
            result = OpportunityScoring.model_validate(raw_result)

            # Create a lookup for new scores from validated Pydantic models
            score_lookup = {
                item.product_name: (item.new_score, item.score_rationale)
                for item in result.scored_opportunities
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

        except (json.JSONDecodeError, JSONParseError) as e:
            self.logger.warning("scoring_json_parse_failed", error=str(e))
            # Fallback: return original opportunities unchanged
            return opportunities
        except Exception as e:
            self.logger.error("opportunity_scoring_failed", error=str(e))
            # Graceful degradation: return originals
            return opportunities

    async def _enhance_talking_points(
        self,
        opportunities: list[Opportunity],
        risks: list[str],
        signals: list[Signal],
        account_name: str,
        industry: str,
        user_context: str = "",
        seller_name: str = "Our Company"
    ) -> list[Opportunity]:
        """
        Enhance talking points with evidence-grounded objection handling.

        Uses strict source citation requirements to prevent hallucination.
        For each opportunity:
        - Links talking points to supporting evidence with [SIG-xxx] citations
        - Adds objection handling based on [RISK-xxx] citations
        - Generates persona-specific messaging aligned with user objectives

        Args:
            opportunities: Scored opportunities
            risks: Identified competitive/market risks
            signals: Research signals for evidence linking
            account_name: Target company name
            industry: Target company industry
            user_context: User's sales context/objectives
            seller_name: Seller company name

        Returns:
            Opportunities with enhanced, evidence-grounded talking points
        """
        if not opportunities:
            return opportunities

        # Format with IDs for citation
        signals_formatted = self._format_signals_with_ids(signals, limit=15)
        risks_formatted = self._format_risks_with_ids(risks)
        seller_context = self._get_seller_context(seller_name)

        # Build opportunity data for enhancement
        opps_data = []
        for i, opp in enumerate(opportunities, 1):
            # Get evidence snippets for context
            evidence_texts = [
                f"[{sig.signal_type}] {sig.content[:150]}"
                for sig in opp.evidence[:3]
            ]

            opps_data.append({
                "id": f"OPP-{i:03d}",
                "product_name": opp.product_name,
                "current_talking_points": opp.talking_points[:5],
                "target_persona": opp.target_persona or "Unknown",
                "evidence_snippets": evidence_texts
            })

        prompt = f"""### ROLE
You are an Enterprise Account Executive at {seller_name}. Your mission is to create evidence-grounded talking points that will resonate with {account_name} stakeholders.

### STRATEGIC ALIGNMENT
═══════════════════════════════════════════════════════════════

**YOUR SALES OBJECTIVE:**
{user_context if user_context else "Create compelling talking points for all opportunities"}

{seller_context}

**TARGET ACCOUNT:**
- Company: {account_name}
- Industry: {industry}

═══════════════════════════════════════════════════════════════
OPPORTUNITIES TO ENHANCE
═══════════════════════════════════════════════════════════════
{json.dumps(opps_data, indent=2)}

═══════════════════════════════════════════════════════════════
IDENTIFIED RISKS [RISK-xxx] (use for objection handling)
═══════════════════════════════════════════════════════════════
{risks_formatted}

═══════════════════════════════════════════════════════════════
INTELLIGENCE SIGNALS [SIG-xxx] (use for evidence linking)
═══════════════════════════════════════════════════════════════
{signals_formatted}

═══════════════════════════════════════════════════════════════
GROUNDING RULES (CRITICAL)
═══════════════════════════════════════════════════════════════

You are PROHIBITED from:
- Inventing quotes not found in the provided evidence
- Making up statistics or ROI numbers without [INDUSTRY] tag
- Referencing documents, reports, or statements not in the signals
- Creating talking points that cannot be traced to evidence

EACH talking point MUST include a source tag:
- [SIG-xxx] - Referencing specific intelligence signal
- [RISK-xxx] - Addressing identified risk (for objection handling)
- [INDUSTRY] - General industry knowledge (use sparingly, max 1 per opportunity)

═══════════════════════════════════════════════════════════════
ENHANCEMENT PROTOCOL
═══════════════════════════════════════════════════════════════

For each opportunity, generate 2-3 ADDITIONAL talking points:

1. **EVIDENCE-LINKED POINT** [SIG-xxx]
   - Connect specific signal to product value
   - Quote or reference actual evidence

2. **OBJECTION HANDLING** [RISK-xxx]
   - Proactively address identified risks
   - Provide concrete mitigation approach

3. **PERSONA-TAILORED POINT** [SIG-xxx or INDUSTRY]
   - Decision-makers: ROI, strategic value
   - Influencers: Technical benefits, adoption ease
   - End-users: Productivity gains

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

❌ BAD (hallucinated - DO NOT DO THIS):
"Your CEO mentioned in a recent interview that automation is a priority"

❌ BAD (generic, no citation):
"Our product will help you be more efficient"

❌ BAD (invented statistic):
"Companies using our tool see 47% improvement"

✅ GOOD (grounded with signal citation):
"[SIG-003] Your job posting for Simulation Engineer requires Simulink experience - our training program accelerates onboarding"

✅ GOOD (addressing risk):
"[RISK-001] Regarding the integration concerns with legacy systems - we provide dedicated migration support and parallel operation capability"

✅ GOOD (industry knowledge tagged):
"[INDUSTRY] Aerospace companies using Model-Based Design typically reduce certification time by 30-40%"

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return JSON with citation-tagged talking points:
{{
    "enhanced_opportunities": [
        {{
            "product_name": "Product Name",
            "additional_talking_points": [
                "[SIG-xxx] Evidence-grounded point with specific citation",
                "[RISK-xxx] Objection handling addressing specific risk",
                "[INDUSTRY] Industry benchmark (if no direct evidence)"
            ]
        }}
    ]
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B
                use_cache=False  # Don't cache enhancement
            )

            raw_result = extract_json_from_llm_response(response.content)
            enhanced_data = raw_result.get("enhanced_opportunities", [])

            # Create lookup for additional talking points
            enhanced_lookup = {
                item.get("product_name", ""): item.get("additional_talking_points", [])
                for item in enhanced_data
                if isinstance(item, dict)
            }

            # Update opportunities with enhanced talking points
            enhanced_opportunities = []
            for opp in opportunities:
                if opp.product_name in enhanced_lookup:
                    additional_points = enhanced_lookup[opp.product_name]

                    # Merge: keep original points, add enhanced ones (avoid duplicates)
                    existing_points_lower = {p.lower() for p in opp.talking_points}
                    new_points = [
                        p for p in additional_points
                        if p.lower() not in existing_points_lower
                    ]

                    # Combine original + new (limit to 7 total)
                    merged_points = list(opp.talking_points) + new_points
                    merged_points = merged_points[:7]

                    enhanced_opp = Opportunity(
                        product_name=opp.product_name,
                        rationale=opp.rationale,
                        evidence=opp.evidence,
                        target_persona=opp.target_persona,
                        talking_points=merged_points,
                        estimated_value=opp.estimated_value,
                        risks=opp.risks,
                        confidence=opp.confidence,
                        confidence_score=opp.confidence_score
                    )
                    enhanced_opportunities.append(enhanced_opp)
                else:
                    # Keep original if not in LLM response
                    enhanced_opportunities.append(opp)

            self.logger.debug(
                "talking_points_enhancement_completed",
                enhanced_count=len(enhanced_lookup)
            )

            return enhanced_opportunities

        except (json.JSONDecodeError, JSONParseError) as e:
            self.logger.warning("talking_points_enhancement_json_failed", error=str(e))
            return opportunities  # Return original on failure
        except Exception as e:
            self.logger.warning("talking_points_enhancement_failed", error=str(e))
            return opportunities  # Return original on failure

    def get_complexity(self, state: ResearchState) -> int:
        """
        Get task complexity for model routing.

        ValidatorAgent performs nuanced reasoning to:
        - Assess competitive and market risks
        - Re-evaluate confidence with multiple factors
        - Enhance talking points with objection handling
        - Make filtering decisions

        This requires Tier 2 (Groq 8B) for quality reasoning.

        Args:
            state: Current research state

        Returns:
            Complexity score (1-10). Validator returns 6 (Tier 2: Groq 8B)
        """
        return 6  # Nuanced reasoning (Tier 2: Groq 8B)
