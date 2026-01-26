"""
Opportunity Identifier Agent - Identifies sales opportunities from gathered intelligence.
Phase 3: Agent Implementation
"""
import json
from typing import Any
from datetime import datetime

import structlog

from src.core.base_agent import StatelessAgent
from src.models.state import ResearchState, Signal, Opportunity, OpportunityConfidence
from src.data_sources.product_catalog import ProductMatcher
from src.core.model_router import ModelRouter

logger = structlog.get_logger(__name__)


class IdentifierAgent(StatelessAgent):
    """
    Opportunity Identifier Agent - Finds sales opportunities from research data.

    This agent analyzes gathered intelligence to:
    1. Extract implicit and explicit requirements from signals and job postings
    2. Match requirements to products using semantic search (ProductMatcher)
    3. Generate opportunity hypotheses with LLM reasoning
    4. Create structured Opportunity objects with evidence

    Uses Tier 2 model (Groq 8B, complexity=6) for nuanced reasoning.

    Responsibilities:
    - Extract requirements from job_postings, signals, tech_stack
    - Use ProductMatcher for semantic product matching
    - Generate opportunity rationale with LLM
    - Identify target personas and talking points
    - Structure results as Opportunity objects

    Modifies ResearchState in-place:
    - state["opportunities"] - List of Opportunity objects
    - state["progress"].identifier_complete = True
    """

    def __init__(
        self,
        product_matcher: ProductMatcher,
        model_router: ModelRouter
    ):
        """
        Initialize Identifier Agent.

        Args:
            product_matcher: ProductMatcher for semantic product matching
            model_router: Model router for LLM reasoning (Tier 2 Groq 8B)
        """
        super().__init__(name="identifier")
        self.product_matcher = product_matcher
        self.model_router = model_router

    async def process(self, state: ResearchState) -> None:
        """
        Identify opportunities from gathered intelligence.

        This method:
        1. Extracts requirements from signals and job_postings
        2. Matches requirements to products via semantic search
        3. Generates opportunity hypotheses with LLM
        4. Creates Opportunity objects with evidence

        Args:
            state: Current research state (modified in-place)
        """
        account = state["account_name"]
        industry = state.get("industry", "")
        signals = state.get("signals", [])
        job_postings = state.get("job_postings", [])
        tech_stack = state.get("tech_stack", [])
        feedback_context = state.get("feedback_context")

        self.logger.info(
            "identifier_started",
            account=account,
            signals_count=len(signals),
            job_postings_count=len(job_postings),
            tech_stack_count=len(tech_stack),
            has_feedback=feedback_context is not None
        )

        # Step 1: Extract requirements from all sources
        requirements = await self._extract_requirements(
            signals=signals,
            job_postings=job_postings,
            tech_stack=tech_stack,
            account_name=account,
            industry=industry,
            feedback_context=feedback_context
        )

        self.logger.info("requirements_extracted", count=len(requirements))

        if not requirements:
            self.logger.warning("no_requirements_found", account=account)
            state["opportunities"] = []
            state["progress"].identifier_complete = True
            return

        # Step 2: Match requirements to products
        product_matches = await self.product_matcher.match_requirements_to_products(
            requirements=requirements,
            top_k=10
        )

        self.logger.info("products_matched", count=len(product_matches))

        if not product_matches:
            self.logger.warning("no_product_matches", account=account)
            state["opportunities"] = []
            state["progress"].identifier_complete = True
            return

        # Step 3: Generate opportunities with LLM
        opportunities = await self._generate_opportunities(
            state=state,
            requirements=requirements,
            product_matches=product_matches,
            signals=signals,
            job_postings=job_postings,
            feedback_context=feedback_context
        )

        self.logger.info("opportunities_generated", count=len(opportunities))

        # Step 4: Store results
        state["opportunities"] = opportunities
        state["progress"].identifier_complete = True

        self.logger.info(
            "identifier_completed",
            opportunities_count=len(opportunities),
            high_confidence=sum(1 for o in opportunities if o.confidence == OpportunityConfidence.HIGH),
            medium_confidence=sum(1 for o in opportunities if o.confidence == OpportunityConfidence.MEDIUM),
            low_confidence=sum(1 for o in opportunities if o.confidence == OpportunityConfidence.LOW)
        )

    async def _extract_requirements(
        self,
        signals: list[Signal],
        job_postings: list[dict],
        tech_stack: list[str],
        account_name: str,
        industry: str,
        feedback_context: str | None = None
    ) -> list[str]:
        """
        Extract implicit and explicit requirements from gathered data.

        Uses LLM to analyze signals and job postings to identify:
        - Technical needs (tools, platforms, capabilities)
        - Business needs (efficiency, scaling, compliance)
        - Pain points (gaps, challenges mentioned)

        Args:
            signals: List of Signal objects from gatherer
            job_postings: List of job posting dicts
            tech_stack: List of identified technologies
            account_name: Company being researched
            industry: Company industry
            feedback_context: Optional feedback from coordinator for retry

        Returns:
            List of requirement strings
        """
        # Build context from signals
        signal_summaries = []
        for signal in signals[:15]:  # Limit to avoid token overflow
            content = signal.content[:500] if isinstance(signal.content, str) else str(signal.content)[:500]
            signal_summaries.append(f"- [{signal.signal_type}] {content}")

        # Build context from job postings
        job_summaries = []
        for job in job_postings[:10]:  # Limit to avoid token overflow
            title = job.get("title", "Unknown")
            desc = job.get("description", "")[:300]
            techs = job.get("technologies", [])
            techs_str = ", ".join(techs[:5]) if techs else "none listed"
            job_summaries.append(f"- {title}: {desc}... (Technologies: {techs_str})")

        # Build feedback instruction if retrying
        feedback_instruction = ""
        if feedback_context:
            feedback_instruction = f"""
IMPORTANT: This is a retry based on human feedback:
{feedback_context}

Adjust your analysis to address this feedback specifically.
"""

        prompt = f"""Analyze this research data for {account_name} ({industry}) to extract their likely technology and business requirements.

SIGNALS (research findings):
{chr(10).join(signal_summaries) if signal_summaries else "No signals available"}

JOB POSTINGS:
{chr(10).join(job_summaries) if job_summaries else "No job postings available"}

CURRENT TECH STACK: {', '.join(tech_stack) if tech_stack else "Unknown"}
{feedback_instruction}
Tasks:
1. Identify EXPLICIT requirements (directly stated needs, job requirements)
2. Identify IMPLICIT requirements (inferred from hiring patterns, tech stack gaps)
3. Identify PAIN POINTS (challenges, inefficiencies mentioned)
4. Consider industry-specific needs for {industry}

Return 5-15 concise requirement statements that could map to software products.
Focus on actionable needs, not vague statements.

Return JSON:
{{
    "requirements": [
        "Need for automated testing solution for embedded systems",
        "Requirement for data visualization and reporting platform",
        "Looking for ML model deployment infrastructure",
        ...
    ]
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B for nuanced reasoning
                use_cache=True
            )

            result = json.loads(response.content)
            requirements = result.get("requirements", [])

            # Validate requirements are strings
            requirements = [str(r) for r in requirements if r]

            return requirements

        except json.JSONDecodeError as e:
            self.logger.warning("requirements_json_parse_failed", error=str(e))
            # Fallback: extract from tech stack
            return [f"Need for {tech} capabilities" for tech in tech_stack[:5]]
        except Exception as e:
            self.logger.error("requirements_extraction_failed", error=str(e))
            return []

    async def _generate_opportunities(
        self,
        state: ResearchState,
        requirements: list[str],
        product_matches: list[tuple[str, float]],
        signals: list[Signal],
        job_postings: list[dict],
        feedback_context: str | None = None
    ) -> list[Opportunity]:
        """
        Generate opportunity objects with LLM reasoning.

        For each matched product, uses LLM to:
        - Generate rationale for why they need it
        - Identify target persona
        - Create talking points
        - Assess confidence level
        - Link supporting evidence

        Args:
            state: Current research state
            requirements: Extracted requirements
            product_matches: List of (product_name, confidence) tuples
            signals: Original signals for evidence linking
            job_postings: Original job postings for evidence linking
            feedback_context: Optional feedback for retry

        Returns:
            List of Opportunity objects
        """
        account_name = state["account_name"]
        industry = state.get("industry", "")

        # Build context
        requirements_text = "\n".join(f"- {r}" for r in requirements)
        products_text = "\n".join(f"- {name} (match score: {score:.2f})" for name, score in product_matches[:10])

        # Build feedback instruction
        feedback_instruction = ""
        if feedback_context:
            feedback_instruction = f"""
IMPORTANT: This is a retry based on human feedback:
{feedback_context}

Adjust your opportunity analysis accordingly.
"""

        prompt = f"""Generate sales opportunities for {account_name} ({industry}).

IDENTIFIED REQUIREMENTS:
{requirements_text}

MATCHING PRODUCTS (from our catalog):
{products_text}
{feedback_instruction}
For EACH relevant product match, create an opportunity with:
1. rationale: WHY they need this product (2-3 sentences, specific to their situation)
2. target_persona: WHO to talk to (job title)
3. talking_points: 3-5 specific points for the sales conversation
4. estimated_value: Deal size estimate (e.g., "$50K ARR", "$100K-200K ARR")
5. risks: 1-3 potential blockers or objections
6. confidence: "high" (>70%), "medium" (40-70%), or "low" (<40%)
7. confidence_score: Numerical score 0.0-1.0

Only include products that have a genuine fit. Quality over quantity.

Return JSON:
{{
    "opportunities": [
        {{
            "product_name": "Product Name",
            "rationale": "They are scaling their ML operations and need...",
            "target_persona": "VP of Engineering",
            "talking_points": ["Point 1", "Point 2", "Point 3"],
            "estimated_value": "$150K ARR",
            "risks": ["Existing competitor relationship", "Budget cycle timing"],
            "confidence": "high",
            "confidence_score": 0.85
        }},
        ...
    ]
}}"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B
                use_cache=False  # Don't cache opportunity generation
            )

            result = json.loads(response.content)
            raw_opportunities = result.get("opportunities", [])

            # Convert to Opportunity objects with evidence linking
            opportunities = []
            for raw_opp in raw_opportunities:
                try:
                    # Map confidence string to enum
                    confidence_str = raw_opp.get("confidence", "medium").lower()
                    confidence_enum = {
                        "high": OpportunityConfidence.HIGH,
                        "medium": OpportunityConfidence.MEDIUM,
                        "low": OpportunityConfidence.LOW
                    }.get(confidence_str, OpportunityConfidence.MEDIUM)

                    # Find relevant evidence signals
                    evidence = self._find_evidence(
                        product_name=raw_opp.get("product_name", ""),
                        rationale=raw_opp.get("rationale", ""),
                        signals=signals
                    )

                    opportunity = Opportunity(
                        product_name=raw_opp.get("product_name", "Unknown"),
                        rationale=raw_opp.get("rationale", ""),
                        evidence=evidence,
                        target_persona=raw_opp.get("target_persona"),
                        talking_points=raw_opp.get("talking_points", []),
                        estimated_value=raw_opp.get("estimated_value"),
                        risks=raw_opp.get("risks", []),
                        confidence=confidence_enum,
                        confidence_score=float(raw_opp.get("confidence_score", 0.5))
                    )
                    opportunities.append(opportunity)

                except Exception as e:
                    self.logger.warning(
                        "opportunity_creation_failed",
                        product=raw_opp.get("product_name", "unknown"),
                        error=str(e)
                    )
                    continue

            return opportunities

        except json.JSONDecodeError as e:
            self.logger.warning("opportunities_json_parse_failed", error=str(e))
            return []
        except Exception as e:
            self.logger.error("opportunity_generation_failed", error=str(e))
            return []

    def _find_evidence(
        self,
        product_name: str,
        rationale: str,
        signals: list[Signal]
    ) -> list[Signal]:
        """
        Find signals that support this opportunity.

        Uses simple keyword matching to link evidence.
        Could be enhanced with semantic similarity.

        Args:
            product_name: Product being recommended
            rationale: Rationale for the opportunity
            signals: Available signals

        Returns:
            List of relevant Signal objects (max 5)
        """
        # Extract keywords from product name and rationale
        keywords = set()
        for word in (product_name + " " + rationale).lower().split():
            if len(word) > 3:  # Skip short words
                keywords.add(word)

        # Score each signal by keyword overlap
        scored_signals = []
        for signal in signals:
            content_lower = signal.content.lower() if isinstance(signal.content, str) else ""
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scored_signals.append((signal, score))

        # Sort by score and return top 5
        scored_signals.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_signals[:5]]

    def get_complexity(self, state: ResearchState) -> int:
        """
        Get task complexity for model routing.

        IdentifierAgent performs nuanced reasoning to:
        - Extract implicit requirements
        - Generate sales opportunity hypotheses
        - Assess confidence levels

        This requires Tier 2 (Groq 8B) for quality reasoning.

        Args:
            state: Current research state

        Returns:
            Complexity score (1-10). Identifier returns 6 (Tier 2: Groq 8B)
        """
        return 6  # Nuanced reasoning (Tier 2: Groq 8B)
