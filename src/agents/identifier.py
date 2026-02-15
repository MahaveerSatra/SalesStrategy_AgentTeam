"""
Opportunity Identifier Agent - Identifies sales opportunities from gathered intelligence.
Phase 3: Agent Implementation
"""
import json
from typing import Any
from datetime import datetime

import structlog

from src.core.base_agent import StatelessAgent
from src.utils.json_parsing import extract_json_from_llm_response, JSONParseError
from src.models.state import ResearchState, Signal, Opportunity, OpportunityConfidence
from src.models.llm_schemas import RequirementsExtraction, OpportunitiesGeneration
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
        user_context = state.get("user_context", "")

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
            feedback_context=feedback_context,
            user_context=user_context
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
            feedback_context=feedback_context,
            user_context=user_context
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

    def _get_product_categories(self) -> list[str]:
        """Get unique product categories from the catalog."""
        try:
            # Check if product_matcher has collection attribute (might be mocked in tests)
            if not hasattr(self.product_matcher, 'collection'):
                return []
            # Query ChromaDB for all unique categories
            results = self.product_matcher.collection.get(include=["metadatas"])
            if results and results.get("metadatas"):
                categories = set()
                for metadata in results["metadatas"]:
                    if metadata and metadata.get("category"):
                        categories.add(metadata["category"])
                return sorted(list(categories))
        except Exception as e:
            self.logger.warning("failed_to_get_categories", error=str(e))
        return []

    def _format_signals_with_ids(self, signals: list[Signal], limit: int = 15) -> str:
        """Format signals with IDs for traceability."""
        formatted = []
        for i, signal in enumerate(signals[:limit]):
            sig_id = f"SIG-{i+1:03d}"
            confidence = signal.confidence if hasattr(signal, 'confidence') else 0.5
            content = signal.content[:400] if isinstance(signal.content, str) else str(signal.content)[:400]
            formatted.append(f"[{sig_id}] (confidence: {confidence:.1f}, type: {signal.signal_type}) \"{content}\"")
        return "\n".join(formatted) if formatted else "No signals available"

    def _format_jobs_with_ids(self, job_postings: list[dict], limit: int = 10) -> str:
        """Format job postings with IDs for traceability."""
        formatted = []
        for i, job in enumerate(job_postings[:limit]):
            job_id = f"JOB-{i+1:03d}"
            title = job.get("title", "Unknown")
            techs = job.get("technologies", [])
            techs_str = ", ".join(techs[:5]) if techs else "none listed"
            desc = job.get("description", "")[:300]
            formatted.append(f"[{job_id}] {title} | Technologies: {techs_str}\n    \"{desc}...\"")
        return "\n".join(formatted) if formatted else "No job postings available"

    def _get_product_details(self, product_matches: list[tuple[str, float]], limit: int = 10) -> str:
        """
        Get detailed product information from ChromaDB for matched products.

        Args:
            product_matches: List of (product_name, confidence) tuples
            limit: Maximum number of products to include

        Returns:
            Formatted string with product details
        """
        try:
            if not hasattr(self.product_matcher, 'collection'):
                # Fallback for mocked tests
                return "\n".join(
                    f"- {name} (relevance: {score:.0%})"
                    for name, score in product_matches[:limit]
                )

            # Get all products from ChromaDB
            all_products = self.product_matcher.collection.get(
                include=["documents", "metadatas"]
            )

            if not all_products or not all_products.get("metadatas"):
                return "\n".join(
                    f"- {name} (relevance: {score:.0%})"
                    for name, score in product_matches[:limit]
                )

            # Build lookup dict: product_name -> document (description)
            product_docs = {}
            for i, metadata in enumerate(all_products["metadatas"]):
                if metadata and metadata.get("name"):
                    product_docs[metadata["name"]] = {
                        "document": all_products["documents"][i] if all_products.get("documents") else "",
                        "category": metadata.get("category", "General")
                    }

            # Format matched products with details
            formatted = []
            for name, score in product_matches[:limit]:
                if name in product_docs:
                    info = product_docs[name]
                    # Extract description (first sentence or truncate)
                    doc = info["document"]
                    desc_end = doc.find(". ", 50)  # Find first sentence after 50 chars
                    short_desc = doc[:desc_end + 1] if desc_end > 0 else doc[:150]
                    formatted.append(
                        f"[PROD-{len(formatted)+1:02d}] {name} ({info['category']})\n"
                        f"    Relevance: {score:.0%}\n"
                        f"    Description: {short_desc}"
                    )
                else:
                    formatted.append(f"[PROD-{len(formatted)+1:02d}] {name} (relevance: {score:.0%})")

            return "\n\n".join(formatted) if formatted else "No matching products"

        except Exception as e:
            self.logger.warning("failed_to_get_product_details", error=str(e))
            # Fallback to simple format
            return "\n".join(
                f"- {name} (relevance: {score:.0%})"
                for name, score in product_matches[:limit]
            )

    def _get_seller_context(self) -> str:
        """
        Get seller company context for opportunity framing.

        Returns information about the seller (our company) to help
        the LLM understand what we sell and our value proposition.

        Returns:
            Formatted string with seller context
        """
        seller_name = getattr(self.product_matcher, 'company_name', 'Our Company')

        # Get product categories to understand our portfolio
        categories = self._get_product_categories()

        if seller_name.lower() == "mathworks":
            # MathWorks-specific context (since we have hardcoded products)
            return f"""**SELLER: {seller_name}**
MathWorks is the leading developer of mathematical computing software.
Our products enable engineers and scientists to analyze data, develop algorithms,
and create models for applications in automotive, aerospace, communications,
electronics, industrial automation, and other industries.

**Our Core Competencies:**
- Technical Computing & Simulation (MATLAB, Simulink)
- AI/ML & Deep Learning
- Model-Based Design & Code Generation
- Embedded Systems Development
- Test & Verification (Polyspace, Simulink Test)

**Product Domains:** {', '.join(categories) if categories else 'Various technical computing solutions'}"""
        else:
            # Generic seller context
            return f"""**SELLER: {seller_name}**
We provide solutions in the following domains:
{', '.join(categories) if categories else 'Various software solutions'}

*Note: For more accurate seller positioning, configure seller profile in product catalog.*"""

    async def _extract_requirements(
        self,
        signals: list[Signal],
        job_postings: list[dict],
        tech_stack: list[str],
        account_name: str,
        industry: str,
        feedback_context: str | None = None,
        user_context: str = ""
    ) -> list[str]:
        """
        Extract Sales-Qualified Requirements (SQRs) using Chain-of-Verification.

        Uses role-based framing and evidence grounding to extract requirements
        that are relevant to our product portfolio.

        Args:
            signals: List of Signal objects from gatherer
            job_postings: List of job posting dicts
            tech_stack: List of identified technologies
            account_name: Company being researched
            industry: Company industry
            feedback_context: Optional feedback from coordinator for retry
            user_context: User's sales context/objectives for prioritization

        Returns:
            List of requirement strings
        """
        # Get product categories for seller context
        product_categories = self._get_product_categories()
        categories_text = ", ".join(product_categories) if product_categories else "General software solutions"

        # Format signals and jobs with IDs
        signals_formatted = self._format_signals_with_ids(signals)
        jobs_formatted = self._format_jobs_with_ids(job_postings)

        # Get seller name from product matcher (with fallback for mocks/tests)
        seller_name = getattr(self.product_matcher, 'company_name', 'Our Company')

        # Build feedback instruction if retrying
        feedback_section = ""
        if feedback_context:
            feedback_section = f"""
═══════════════════════════════════════════════════════════════
COORDINATOR FEEDBACK (Address this in your analysis)
═══════════════════════════════════════════════════════════════
{feedback_context}
"""

        prompt = f"""### ROLE
You are a Senior Solutions Architect at {seller_name}. Your mission is to extract "Sales-Qualified Requirements" (SQRs) from research data - requirements that could be addressed by our product portfolio.

### USER'S SALES OBJECTIVE
{user_context if user_context else "No specific focus provided - extract all relevant technical requirements"}

### TARGET ACCOUNT
- Company: {account_name}
- Industry: {industry}
- Detected Tech Stack: {', '.join(tech_stack) if tech_stack else "Unknown"}

### OUR PRODUCT DOMAINS (Only extract requirements we can address)
{categories_text}

*Ignore requirements that fall outside these technical domains.*
{feedback_section}
═══════════════════════════════════════════════════════════════
RESEARCH DATA
═══════════════════════════════════════════════════════════════

**JOB POSTINGS:**
{jobs_formatted}

**INTELLIGENCE SIGNALS:**
{signals_formatted}

═══════════════════════════════════════════════════════════════
EXTRACTION PROTOCOL (Chain-of-Verification)
═══════════════════════════════════════════════════════════════

For each potential requirement:
1. **QUOTE** - Find exact text from a source (JOB-xxx or SIG-xxx)
2. **INTERPRET** - What technical need does this imply?
3. **VERIFY RELEVANCE** - Does this relate to our product domains? (If NO → skip)
4. **VERIFY PRIORITY** - Does this align with user's sales objective? (If YES → high priority)

**NEGATIVE GUIDANCE (Skip these):**
- Soft skills ("team player", "communication skills", "leadership")
- Generic tech everyone uses ("email", "Microsoft Office", "Git")
- Requirements outside our product domains
- Duplicates (consolidate similar needs into one)
- Signals with confidence < 0.5

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

❌ BAD (vague, no evidence):
{{"requirement": "Need for better software", "evidence_quote": "", "source_id": ""}}

❌ BAD (outside our domain):
{{"requirement": "Need for HR management system", "evidence_quote": "Hiring HR Manager", "source_id": "JOB-005"}}

❌ BAD (generic tech):
{{"requirement": "Need for version control", "evidence_quote": "Must know Git", "source_id": "JOB-002"}}

✅ GOOD (specific, grounded, relevant):
{{
  "requirement": "Need for fluid dynamics simulation to optimize aircraft aerodynamics",
  "evidence_quote": "Seeking engineer with CFD experience for wing aerodynamics team, Simscape or similar tools preferred",
  "source_id": "JOB-003",
  "priority": "high"
}}

✅ GOOD (inferred from hiring pattern):
{{
  "requirement": "Need for embedded systems code generation and verification",
  "evidence_quote": "Hiring 3 embedded software engineers with AUTOSAR and ISO 26262 experience",
  "source_id": "JOB-007",
  "priority": "medium"
}}

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (CRITICAL - JSON ONLY)
═══════════════════════════════════════════════════════════════

**RESPOND WITH VALID JSON ONLY. NO markdown, NO explanatory text, NO code fences.**

Return 5-15 requirements. Put HIGH priority (aligned with user objective) FIRST.

Your ENTIRE response must be this exact JSON structure:
{{
  "requirements": [
    {{
      "requirement": "Specific technical need statement",
      "evidence_quote": "Exact text from source that proves this need",
      "source_id": "JOB-xxx or SIG-xxx",
      "priority": "high|medium|low"
    }}
  ]
}}

**IMPORTANT: Start your response with {{ and end with }}. Nothing else.**"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B for nuanced reasoning
                use_cache=True
            )

            # Use robust JSON extraction to handle varied LLM output formats
            raw_result = extract_json_from_llm_response(response.content)

            # Extract requirements from the structured response
            raw_requirements = raw_result.get("requirements", [])
            requirements = []

            for item in raw_requirements:
                if isinstance(item, dict):
                    # New structured format: extract requirement text
                    req_text = item.get("requirement", "")
                    if req_text:
                        requirements.append(req_text)
                        # Log evidence for traceability (optional debug)
                        self.logger.debug(
                            "requirement_extracted",
                            requirement=req_text[:100],
                            source_id=item.get("source_id", "unknown"),
                            priority=item.get("priority", "medium")
                        )
                elif isinstance(item, str) and item:
                    # Legacy simple string format
                    requirements.append(item)

            self.logger.info(
                "requirements_extracted_with_cove",
                total=len(requirements),
                high_priority=sum(1 for r in raw_requirements if isinstance(r, dict) and r.get("priority") == "high")
            )

            return requirements

        except (json.JSONDecodeError, JSONParseError) as e:
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
        feedback_context: str | None = None,
        user_context: str = ""
    ) -> list[Opportunity]:
        """
        Generate opportunity objects with evidence-grounded LLM reasoning.

        Uses role-based framing (Enterprise AE), seller context, and strict
        evidence grounding to prevent confabulation. Each talking point must
        cite source evidence [JOB-xxx], [SIG-xxx], or [INDUSTRY].

        Args:
            state: Current research state
            requirements: Extracted requirements
            product_matches: List of (product_name, confidence) tuples
            signals: Original signals for evidence linking
            job_postings: Original job postings for evidence linking
            feedback_context: Optional feedback from coordinator for retry
            user_context: User's sales context/objectives for prioritization

        Returns:
            List of Opportunity objects
        """
        account_name = state["account_name"]
        industry = state.get("industry", "")

        # Get seller name and context
        seller_name = getattr(self.product_matcher, 'company_name', 'Our Company')
        seller_context = self._get_seller_context()

        # Build formatted context using helpers (consistent IDs with requirements)
        requirements_text = "\n".join(f"- {r}" for r in requirements)
        products_text = self._get_product_details(product_matches, limit=8)
        signals_formatted = self._format_signals_with_ids(signals, limit=12)
        jobs_formatted = self._format_jobs_with_ids(job_postings, limit=8)

        # Build feedback section (consistent with requirements prompt)
        feedback_section = ""
        if feedback_context:
            feedback_section = f"""
═══════════════════════════════════════════════════════════════
COORDINATOR FEEDBACK (Address this in your opportunity analysis)
═══════════════════════════════════════════════════════════════
{feedback_context}
"""

        prompt = f"""### ROLE
You are an Enterprise Account Executive at {seller_name}. Your mission is to create evidence-grounded sales opportunities for {account_name}.

### STRATEGIC ALIGNMENT
═══════════════════════════════════════════════════════════════

**YOUR SALES OBJECTIVE:**
{user_context if user_context else "Identify all relevant opportunities where our products can address their needs"}

{seller_context}

**TARGET ACCOUNT:**
- Company: {account_name}
- Industry: {industry}
{feedback_section}
═══════════════════════════════════════════════════════════════
EVIDENCE DATA (Cite these using IDs in your talking points)
═══════════════════════════════════════════════════════════════

**IDENTIFIED REQUIREMENTS:**
{requirements_text}

**OUR MATCHING PRODUCTS:**
{products_text}

**JOB POSTINGS [JOB-xxx]:**
{jobs_formatted}

**INTELLIGENCE SIGNALS [SIG-xxx]:**
{signals_formatted}

═══════════════════════════════════════════════════════════════
OPPORTUNITY GENERATION PROTOCOL
═══════════════════════════════════════════════════════════════

For each product with GENUINE fit (relevance > 50%), create an opportunity:

**1. VERIFY FIT** - Does evidence support this product need?
   - Match product capabilities to specific requirements
   - Skip products without clear evidence of need

**2. DERIVE PERSONA** - Who's the buyer?
   - Look at job postings: if hiring [Title], buyer is likely [+1 level above]
   - Example: Hiring "ML Engineer" → Target "Director of ML Engineering"
   - Include department and role type: decision-maker/influencer/end-user

**3. BUILD CASE** - Create evidence-grounded talking points
   - EACH talking point MUST cite a source: [JOB-xxx], [SIG-xxx], or [INDUSTRY]
   - 3-5 points connecting their specific situation to our product value
   - Include at least one ROI/business impact point

**4. ASSESS RISK** - What could block this deal?
   - 1-3 realistic blockers with mitigation strategies
   - Format: "Risk (mitigation: approach)"

═══════════════════════════════════════════════════════════════
GROUNDING RULES (CRITICAL)
═══════════════════════════════════════════════════════════════

You are PROHIBITED from:
- Inventing quotes not found in the provided evidence
- Making up statistics or ROI numbers without [INDUSTRY] tag
- Referencing documents, reports, or statements not in the evidence

If you cannot find evidence for a talking point, you MUST:
- Tag it as [INDUSTRY] and frame it as "Companies in {industry} typically..."
- Mark confidence as "medium" or "low"

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

BAD (hallucinated quote - DO NOT DO THIS):
"talking_points": ["Your CTO mentioned in a recent blog that cloud migration is a priority"]

BAD (generic, no evidence):
"talking_points": ["Our product will help you be more efficient"]

GOOD (grounded in evidence):
"talking_points": [
    "[JOB-003] You're hiring ML Platform Engineers with Kubernetes experience - Simulink integrates natively with container orchestration",
    "[SIG-005] Your partnership with AWS indicates cloud infrastructure investment - our Cloud Solutions deploy seamlessly to AWS",
    "[INDUSTRY] Aerospace companies using Model-Based Design typically see 40% faster certification cycles"
]

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (CRITICAL - JSON ONLY)
═══════════════════════════════════════════════════════════════

**RESPOND WITH VALID JSON ONLY. NO markdown, NO explanatory text, NO code fences.**

Return 2-5 opportunities. Quality over quantity. Only products with genuine fit.

Your ENTIRE response must be this exact JSON structure:
{{
    "opportunities": [
        {{
            "product_name": "Exact Product Name from OUR MATCHING PRODUCTS",
            "rationale": "2-3 sentences explaining WHY they need this, referencing specific evidence",
            "target_persona": "Title, Department/Team (decision-maker|influencer|end-user)",
            "talking_points": [
                "[JOB-xxx] Evidence-grounded point about their hiring/needs",
                "[SIG-xxx] Point connecting signal to product value",
                "[INDUSTRY] Industry benchmark or typical use case"
            ],
            "estimated_value": "$XXK-$XXXK ARR",
            "risks": [
                "Risk description (mitigation: approach)"
            ],
            "confidence": "high|medium|low",
            "confidence_score": 0.0-1.0
        }}
    ]
}}

**IMPORTANT: Start your response with {{ and end with }}. Nothing else.**"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=6,  # Tier 2 Groq 8B
                use_cache=False  # Don't cache opportunity generation
            )

            # Use robust JSON extraction to handle varied LLM output formats
            raw_result = extract_json_from_llm_response(response.content)

            # Try Pydantic validation first, fall back to raw dict if it fails
            # This handles cases where some entries in the list are malformed
            try:
                result = OpportunitiesGeneration.model_validate(raw_result)
                raw_opportunities = [opp.model_dump() for opp in result.opportunities]
            except Exception as validation_error:
                self.logger.warning(
                    "pydantic_validation_failed_using_fallback",
                    error=str(validation_error)
                )
                # Fall back to raw dict - will validate individual items below
                raw_opportunities = raw_result.get("opportunities", [])

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

        except (json.JSONDecodeError, JSONParseError) as e:
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
