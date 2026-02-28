"""
Coordinator Agent - Supervisor agent with dual entry/exit roles and feedback routing.

This agent runs at TWO points in the workflow:
1. ENTRY: Validates inputs, asks clarifying questions, minimal enrichment
2. EXIT: Presents report to human, handles feedback, routes next action

Phase 3: Agent Implementation
"""
import json
from enum import Enum
from typing import Any
from datetime import datetime

import structlog

from src.core.base_agent import StatelessAgent
from src.utils.json_parsing import extract_json_from_llm_response, JSONParseError
from src.models.state import ResearchState, ResearchProgress, Opportunity
from src.models.llm_schemas import InputValidation, ClarificationCheck, FeedbackIntent
from src.core.model_router import ModelRouter


logger = structlog.get_logger(__name__)


class WorkflowRoute(str, Enum):
    """Routing decisions after human feedback."""
    GATHERER = "gatherer"       # Need more data collection/analysis
    IDENTIFIER = "identifier"   # Find different opportunities
    VALIDATOR = "validator"     # Re-evaluate confidence scores
    COMPLETE = "complete"       # Workflow finished, human approved


class CoordinatorAgent(StatelessAgent):
    """
    Supervisor agent with dual entry/exit roles and feedback routing.

    This agent runs at TWO points in the workflow:
    1. ENTRY (process_entry): Validates inputs, asks clarifying questions
    2. EXIT (process_exit): Presents report, handles human feedback

    Responsibilities:
    - Validate initial inputs (account_name, industry) with moderate LLM validation
    - Normalize company names (minimal enrichment)
    - Generate smart clarifying questions when LLM determines it would help
    - Format validated opportunities as readable report
    - Present analysis to human and collect feedback
    - Route to appropriate agent based on feedback (GATHERER/IDENTIFIER/VALIDATOR/COMPLETE)

    Complexity: 3 (routes to LOCAL Ollama for all LLM calls)
    """

    def __init__(self, model_router: ModelRouter):
        """
        Initialize Coordinator Agent.

        Args:
            model_router: Model router for LLM calls (Tier 1 Ollama, complexity=3)
        """
        super().__init__(name="coordinator")
        self.model_router = model_router
        self.logger = logger.bind(agent="coordinator")

    # ─────────────────────────────────────────────────────────────────────────
    # DEFAULT PROCESS METHOD (delegates to appropriate phase)
    # ─────────────────────────────────────────────────────────────────────────

    async def process(self, state: ResearchState) -> None:
        """
        Default process method - delegates to appropriate phase based on state.

        Checks state to determine which phase:
        - If coordinator entry not complete -> process_entry()
        - If validator complete but no report -> process_exit()
        - If has new human feedback -> process_feedback()

        Args:
            state: Current research state (modified in-place)
        """
        progress = state["progress"]

        # Check if we have human feedback to process
        human_feedback = state.get("human_feedback", [])
        current_report = state.get("current_report")  # type: ignore

        # Phase 1: Entry - validate inputs
        if not progress.coordinator_complete:
            self.logger.info("coordinator_delegating_to_entry")
            await self.process_entry(state)
            return

        # Phase 2: Exit - present report (after validator completes)
        if progress.validator_complete and not current_report:
            self.logger.info("coordinator_delegating_to_exit")
            await self.process_exit(state)
            return

        # Phase 3: Process feedback (if human has responded)
        if human_feedback and current_report and state.get("waiting_for_human") is False:
            self.logger.info("coordinator_delegating_to_feedback")
            await self.process_feedback(state)
            return

        self.logger.warning("coordinator_no_action_needed")

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY POINT (Before Gatherer)
    # ─────────────────────────────────────────────────────────────────────────

    async def process_entry(self, state: ResearchState) -> None:
        """
        Entry point processing - validates and prepares for research.

        Steps:
        1. Validate required inputs (account_name, industry)
        2. Minimal enrichment (normalize company name)
        3. Smart questioning (LLM decides if clarification needed)
        4. Set human-in-loop flags if questions exist
        5. Mark entry phase complete

        Args:
            state: Current research state (modified in-place)
        """
        self.logger.info(
            "coordinator_entry_started",
            account=state.get("account_name"),
            industry=state.get("industry")
        )

        # Step 1: Validate inputs
        validation_errors = await self._validate_inputs(state)

        if validation_errors:
            self.logger.warning(
                "coordinator_validation_failed",
                errors=validation_errors
            )
            # Graceful degradation: store errors, pause for human
            state["error_messages"].extend(validation_errors)
            state["waiting_for_human"] = True
            state["human_question"] = (
                "I found some issues with the research request:\n\n"
                + "\n".join(f"- {err}" for err in validation_errors)
                + "\n\nPlease provide corrected information."
            )
            # Don't mark complete - need human to fix
            return

        # Step 2: Minimal enrichment - normalize company name
        original_name = state["account_name"]
        normalized_name = await self._normalize_company_name(original_name)

        if normalized_name != original_name:
            self.logger.info(
                "coordinator_name_normalized",
                original=original_name,
                normalized=normalized_name
            )
            state["account_name"] = normalized_name

        # Step 3: Smart questioning - LLM decides if clarification needed
        # BUT skip if we already have human feedback (user already answered questions)
        has_prior_feedback = bool(state.get("human_feedback"))

        if has_prior_feedback:
            self.logger.info(
                "coordinator_skipping_questions",
                reason="human_feedback_exists",
                feedback_count=len(state.get("human_feedback", []))
            )
            # User already provided feedback, proceed without more questions
            state["waiting_for_human"] = False
            state["human_question"] = None
            state["progress"].coordinator_complete = True
            return

        clarifying_question = await self._generate_clarifying_questions(state)

        if clarifying_question:
            self.logger.info(
                "coordinator_needs_clarification",
                question=clarifying_question[:100]
            )
            state["waiting_for_human"] = True
            state["human_question"] = clarifying_question
            # Mark complete even with questions - we can proceed after human responds
            state["progress"].coordinator_complete = True
            return

        # Step 4: No questions needed, ready to proceed
        state["waiting_for_human"] = False
        state["human_question"] = None
        state["progress"].coordinator_complete = True

        self.logger.info(
            "coordinator_entry_completed",
            account=state["account_name"],
            needs_human=False
        )

    async def _validate_inputs(self, state: ResearchState) -> list[str]:
        """
        Moderate validation using LLM.

        Checks:
        - Required fields present and non-empty
        - Company name format (not gibberish)
        - Industry makes sense
        - Suggests corrections for typos

        Args:
            state: Current research state

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        # Basic validation - check required fields exist
        account_name = state.get("account_name", "")
        industry = state.get("industry", "")

        if not account_name or not account_name.strip():
            errors.append("Account name is required but was not provided.")
            return errors  # Can't continue without account name

        if not industry or not industry.strip():
            errors.append("Industry is required but was not provided.")
            return errors  # Can't continue without industry

        # LLM-based validation for quality and researchability
        seller_name = state.get("seller_name", "our company")  # type: ignore
        prompt = f"""You are validating a sales research request. Your job is to ensure we can deliver HIGH-QUALITY, ACTIONABLE intelligence.

═══════════════════════════════════════════════════════════════
RESEARCH REQUEST
═══════════════════════════════════════════════════════════════
Account Name: {account_name}
Industry: {industry}
Region: {state.get("region", "Not specified")}
Seller Company: {seller_name}
Additional Context: {state.get("user_context", "Not provided")}

═══════════════════════════════════════════════════════════════
VALIDATION CHECKS
═══════════════════════════════════════════════════════════════

1. **COMPANY VALIDITY**
   - Is this a real, researchable company? (not gibberish, placeholder, or test data)
   - Is it specific enough? ("Amazon" is ambiguous - AWS? Retail? Which subsidiary?)
   - Common typos to catch: "Microsft" → "Microsoft", "Gogle" → "Google"

2. **INDUSTRY ALIGNMENT**
   - Is the industry correctly matched to the company?
   - Example error: "Boeing" with industry "retail" should suggest "aerospace"

3. **RESEARCHABILITY ASSESSMENT**
   - Can we find meaningful public data about this company?
   - Is it a private company with limited public info? (flag as concern, don't block)
   - Is it a subsidiary that should be researched as parent company?

4. **SELLER-CUSTOMER FIT CHECK**
   - Does {seller_name}'s typical customer profile match this account?
   - Flag if the account seems outside typical target market (but don't block)

5. **CONTEXT QUALITY**
   - If context provided, does it make sense?
   - Flag contradictions (e.g., "new prospect" but mentions "renewal")

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return JSON:
{{
    "is_valid": true/false,
    "errors": ["List of blocking errors that prevent research"],
    "suggested_corrections": {{
        "account_name": "Corrected name if typo detected",
        "industry": "Corrected industry if mismatched"
    }},
    "concerns": ["Non-blocking concerns to note (e.g., 'Private company - limited public data')"],
    "enrichment_suggestions": ["Helpful additions (e.g., 'Consider specifying AWS vs Amazon Retail')"]
}}

DECISION RULES:
- Only set is_valid=false for BLOCKING issues (gibberish input, clearly fake company)
- Typos should be auto-corrected, not block validation
- Concerns are informational - research proceeds but user is informed
- When in doubt, ALLOW the research to proceed (false negatives are worse than false positives)
"""

        try:
            # Use structured output for guaranteed valid JSON
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                temperature=0,  # Deterministic for structured output
                use_cache=True,
                response_format=InputValidation.model_json_schema()
            )

            # Parse with Pydantic - guaranteed to work with structured output
            try:
                result = InputValidation.model_validate_json(response.content)
            except Exception as pydantic_error:
                # Fallback to robust JSON extraction if Pydantic validation fails
                self.logger.warning(
                    "pydantic_validation_failed_using_fallback",
                    error=str(pydantic_error)
                )
                raw_result = extract_json_from_llm_response(response.content)
                result = InputValidation.model_validate(raw_result)

            if not result.is_valid:
                errors.extend(result.errors)

            # Apply suggested corrections to state (only if non-empty)
            if result.suggested_corrections:
                corrected_account = result.suggested_corrections.get("account_name", "")
                if corrected_account and corrected_account.strip():
                    state["account_name"] = corrected_account
                    self.logger.info(
                        "coordinator_applied_correction",
                        field="account_name",
                        corrected=corrected_account
                    )
                corrected_industry = result.suggested_corrections.get("industry", "")
                if corrected_industry and corrected_industry.strip():
                    state["industry"] = corrected_industry
                    self.logger.info(
                        "coordinator_applied_correction",
                        field="industry",
                        corrected=corrected_industry
                    )

            # Log concerns but don't block
            if result.concerns:
                self.logger.info(
                    "coordinator_validation_concerns",
                    concerns=result.concerns
                )

        except (json.JSONDecodeError, JSONParseError) as e:
            self.logger.warning(
                "coordinator_validation_json_parse_failed",
                error=str(e)
            )
            # Continue without LLM validation if parsing fails
        except Exception as e:
            self.logger.warning(
                "coordinator_validation_llm_failed",
                error=str(e)
            )
            # Continue without LLM validation if LLM call fails

        return errors

    # Well-known stock ticker to company name mappings
    # Only these abbreviations will be expanded - conservative approach
    TICKER_TO_COMPANY = {
        "msft": "Microsoft",
        "aapl": "Apple",
        "googl": "Google",
        "goog": "Google",
        "amzn": "Amazon",
        "meta": "Meta",
        "fb": "Meta",
        "tsla": "Tesla",
        "nvda": "NVIDIA",
        "ibm": "IBM",
        "intc": "Intel",
        "amd": "AMD",
        "crm": "Salesforce",
        "orcl": "Oracle",
        "sap": "SAP",
        "adbe": "Adobe",
        "csco": "Cisco",
        "ba": "Boeing",
        "ge": "General Electric",
        "gm": "General Motors",
        "f": "Ford",
        "tm": "Toyota",
        "rivn": "Rivian",
    }

    async def _normalize_company_name(self, name: str) -> str:
        """
        Normalize company name using RULE-BASED approach (no LLM hallucination risk).

        Strategy:
        1. Check if it's a known stock ticker -> expand to full name
        2. Apply rule-based cleanup (remove suffixes, fix caps)
        3. Return original if already looks normal

        This approach is reliable and deterministic - no LLM hallucinations.

        Examples:
        - "msft" -> "Microsoft" (ticker expansion)
        - "BOEING CO" -> "Boeing" (caps fix + suffix removal)
        - "amazon.com" -> "Amazon" (domain removal)
        - "Boeing" -> "Boeing" (already normal, no change)

        Args:
            name: Original company name

        Returns:
            Normalized company name
        """
        if not name or not name.strip():
            return name

        original = name.strip()

        # Step 1: Check if it's a known stock ticker
        name_lower = original.lower().strip()
        if name_lower in self.TICKER_TO_COMPANY:
            normalized = self.TICKER_TO_COMPANY[name_lower]
            self.logger.info(
                "coordinator_ticker_expanded",
                original=original,
                normalized=normalized
            )
            return normalized

        # Step 2: Rule-based cleanup
        normalized = original

        # Remove common legal suffixes
        suffixes_to_remove = [
            ", Inc.", " Inc.", " Inc",
            ", Corp.", " Corp.", " Corp",
            ", LLC", " LLC",
            ", Ltd.", " Ltd.", " Ltd",
            ", Co.", " Co.", " Co",
            ", Corporation", " Corporation",
            ", Company", " Company",
            ", Incorporated", " Incorporated",
        ]
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break

        # Remove domain extensions
        domain_extensions = [".com", ".io", ".ai", ".co", ".org", ".net"]
        for ext in domain_extensions:
            if normalized.lower().endswith(ext):
                normalized = normalized[:-len(ext)]
                break

        # Fix all-caps names (but preserve intentional acronyms like IBM, AMD)
        if normalized.isupper() and len(normalized) > 4:
            # Title case for longer all-caps names
            normalized = normalized.title()

        # Strip whitespace
        normalized = normalized.strip()

        # If nothing changed, return original
        if normalized == original:
            return original

        self.logger.info(
            "coordinator_name_normalized",
            original=original,
            normalized=normalized,
            method="rule_based"
        )
        return normalized

    async def _generate_clarifying_questions(self, state: ResearchState) -> str | None:
        """
        MODERATE questioning - ask when it would significantly improve output quality.

        PHILOSOPHY: Balance between action and quality. Ask 1-2 focused questions
        when the answer would materially change our research approach.

        Args:
            state: Current research state

        Returns:
            Question string if clarification would significantly improve results, else None
        """
        user_context = state.get("user_context") or ""
        seller_name = state.get("seller_name", "our company")  # type: ignore
        account_name = state["account_name"]
        industry = state["industry"]

        # FAST PATH: If user provided rich context, skip questions
        # Rich context = mentions specific objectives, products, or situations
        rich_context_signals = [
            len(user_context.strip()) > 100,  # Substantial context provided
            "renewal" in user_context.lower() and "expansion" in user_context.lower(),
            "competitor" in user_context.lower(),
            "pain" in user_context.lower() and "point" in user_context.lower(),
            "qbr" in user_context.lower(),
            "demo" in user_context.lower(),
        ]

        if any(rich_context_signals):
            self.logger.info(
                "coordinator_skipping_questions_rich_context",
                context_length=len(user_context)
            )
            return None

        prompt = f"""You help sales reps prepare for customer meetings. Decide if 1-2 quick questions would significantly improve the research.

═══════════════════════════════════════════════════════════════
RESEARCH REQUEST
═══════════════════════════════════════════════════════════════
Account: {account_name}
Industry: {industry}
Seller: {seller_name}
Region: {state.get("region") or "Not specified"}
Context: {user_context or "None provided"}

═══════════════════════════════════════════════════════════════
DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════

**ASK QUESTIONS if (any of these):**
1. Company name is ambiguous (Amazon AWS vs Retail, GE divisions, etc.)
2. No context provided AND knowing the sales stage would change the research
   - Discovery call vs QBR vs Renewal = very different research focus
3. Industry seems mismatched with company (may be a typo or error)

**DO NOT ASK if:**
- Context already mentions meeting type, objective, or specific focus
- Company and industry are clear
- User just wants general research (that's fine, we can deliver value)

═══════════════════════════════════════════════════════════════
QUESTION GUIDELINES
═══════════════════════════════════════════════════════════════

If asking, keep it to 1-2 QUICK questions max:
- Make them multiple choice when possible (faster to answer)
- Focus on what would CHANGE the research output
- Don't ask for nice-to-have info

Good questions:
- "Quick clarification: Is this AWS or Amazon Retail?"
- "What's the meeting type? (Discovery / QBR / Renewal / Other)"
- "Any specific product areas to focus on, or should I research broadly?"

Bad questions (don't ask these):
- "What's your relationship history?" (we'll research either way)
- "What's their budget?" (that's for the sales call)
- "What are their pain points?" (we'll FIND those)

═══════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════

Return JSON:
{{
    "needs_clarification": true/false,
    "questions": "1-2 quick questions if needed, null otherwise",
    "reasoning": "Brief explanation"
}}

Example when context is empty and question would help:
{{
    "needs_clarification": true,
    "questions": "Quick question to focus the research:\\n\\nWhat type of meeting is this for?\\n- Discovery call (new prospect)\\n- QBR (existing customer)\\n- Renewal discussion\\n- Expansion opportunity\\n- General research (I'll cover all angles)",
    "reasoning": "No context provided - knowing the sales stage will focus the research on relevant opportunities"
}}

Example when we can proceed:
{{
    "needs_clarification": false,
    "questions": null,
    "reasoning": "Company and industry are clear, will provide comprehensive research"
}}
"""

        try:
            # Use structured output for guaranteed valid JSON
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                temperature=0,  # Deterministic for structured output
                use_cache=True,
                response_format=ClarificationCheck.model_json_schema()
            )

            # Parse with Pydantic - guaranteed to work with structured output
            try:
                result = ClarificationCheck.model_validate_json(response.content)
            except Exception as pydantic_error:
                # Fallback to robust JSON extraction if Pydantic validation fails
                self.logger.warning(
                    "pydantic_validation_failed_using_fallback",
                    error=str(pydantic_error)
                )
                raw_result = extract_json_from_llm_response(response.content)
                result = ClarificationCheck.model_validate(raw_result)

            if result.needs_clarification and result.questions:
                self.logger.info(
                    "coordinator_question_generated",
                    reasoning=result.reasoning[:100] if result.reasoning else ""
                )
                return result.questions

            return None

        except (json.JSONDecodeError, JSONParseError) as e:
            self.logger.warning(
                "coordinator_question_json_parse_failed",
                error=str(e)
            )
            return None
        except Exception as e:
            self.logger.warning(
                "coordinator_question_generation_failed",
                error=str(e)
            )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # EXIT POINT (After Validator)
    # ─────────────────────────────────────────────────────────────────────────

    async def process_exit(self, state: ResearchState) -> None:
        """
        Exit point processing - formats and presents results to human.

        Steps:
        1. Format validated_opportunities as readable report
        2. Include confidence scores and evidence summary
        3. Highlight competitive risks
        4. Set human_question with report + feedback prompt
        5. Set waiting_for_human = True

        Args:
            state: Current research state (modified in-place)
        """
        self.logger.info(
            "coordinator_exit_started",
            opportunities_count=len(state.get("validated_opportunities", [])),
            risks_count=len(state.get("competitive_risks", []))
        )

        # Format the report
        report = await self._format_report(state)

        # Store report in state for later reference
        state["current_report"] = report  # type: ignore

        # Present to human with simple feedback prompt (not the full report)
        state["human_question"] = (
            "Please review the research report above. You can:\n\n"
            "- Say **'approved'** to finalize the report\n"
            "- Ask to **'dig deeper on [topic]'** for more research\n"
            "- Request **'different products'** to explore other opportunities\n"
            "- Provide **specific feedback** to refine the analysis"
        )
        state["waiting_for_human"] = True

        # Increment workflow iteration counter
        current_iteration = state.get("workflow_iteration", 1)  # type: ignore
        state["workflow_iteration"] = current_iteration  # type: ignore

        self.logger.info(
            "coordinator_exit_completed",
            report_length=len(report),
            iteration=current_iteration
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return len(text) // 4

    def _build_compact_context(
        self,
        opportunities: list,
        signals: list,
        job_postings: list,
        risks: list,
    ) -> tuple[str, str, str, str, int]:
        """
        Build compact JSON context for report generation, staying within token limits.

        Uses progressive truncation if context exceeds target_tokens config setting.

        Returns:
            Tuple of (opps_json, signals_json, jobs_json, risks_json, estimated_tokens)
        """
        from src.config import settings
        from src.models.state import Opportunity

        max_opps = settings.report_max_opportunities
        max_signals = settings.report_max_signals
        max_jobs = settings.report_max_jobs
        max_risks = settings.report_max_risks
        content_limit = settings.report_signal_content_limit
        target_tokens = settings.report_target_tokens
        rationale_limit = settings.report_rationale_char_limit

        # Build compact opportunities (essential fields only)
        opps_data = []
        for opp in opportunities[:max_opps]:
            if isinstance(opp, Opportunity):
                opp_compact = {
                    "product": opp.product_name,
                    "confidence": opp.confidence_score,
                    "rationale": opp.rationale[:rationale_limit] if opp.rationale else "",
                    "persona": opp.target_persona or "Unknown",
                    "evidence": [
                        f"[{e.signal_type}] {e.content[:content_limit]}"
                        for e in (opp.evidence or [])[:2]  # Top 2 evidence only
                    ]
                }
                opps_data.append(opp_compact)
            elif isinstance(opp, dict):
                opps_data.append({
                    "product": opp.get("product_name", "Unknown"),
                    "confidence": opp.get("confidence_score", 0.5),
                    "rationale": opp.get("rationale", "")[:rationale_limit],
                    "persona": opp.get("target_persona", "Unknown"),
                    "evidence": []
                })

        # Build compact signals
        signals_data = []
        for sig in signals[:max_signals]:
            if hasattr(sig, 'signal_type'):
                signals_data.append({
                    "type": sig.signal_type,
                    "src": sig.source[:50] if sig.source else "",
                    "content": sig.content[:content_limit] if sig.content else ""
                })
            elif isinstance(sig, dict):
                signals_data.append({
                    "type": sig.get("signal_type", "unknown"),
                    "src": sig.get("source", "")[:50],
                    "content": sig.get("content", "")[:content_limit]
                })

        # Build compact jobs (title and department only)
        jobs_data = []
        for job in job_postings[:max_jobs]:
            if hasattr(job, 'title'):
                jobs_data.append({
                    "title": job.title,
                    "dept": getattr(job, 'department', 'Unknown')
                })
            elif isinstance(job, dict):
                jobs_data.append({
                    "title": job.get("title", "Unknown"),
                    "dept": job.get("department", "Unknown")
                })

        # Build compact risks
        risks_data = []
        for risk in (risks or [])[:max_risks]:
            if isinstance(risk, dict):
                risks_data.append({
                    "type": risk.get("risk_type", "unknown"),
                    "desc": risk.get("description", "")[:150]
                })
            elif hasattr(risk, 'risk_type'):
                risks_data.append({
                    "type": risk.risk_type,
                    "desc": (risk.description[:150] if hasattr(risk, 'description') else "")
                })

        # Convert to compact JSON (no indent)
        opps_json = json.dumps(opps_data, default=str)
        signals_json = json.dumps(signals_data, default=str)
        jobs_json = json.dumps(jobs_data, default=str)
        risks_json = json.dumps(risks_data, default=str) if risks_data else "[]"

        # Estimate total tokens
        total_context = opps_json + signals_json + jobs_json + risks_json
        estimated_tokens = self._estimate_tokens(total_context)

        # Progressive truncation if still over limit
        if estimated_tokens > target_tokens and len(opps_data) > 3:
            # Reduce to top 3 opportunities
            opps_json = json.dumps(opps_data[:3], default=str)
            estimated_tokens = self._estimate_tokens(opps_json + signals_json + jobs_json + risks_json)

        if estimated_tokens > target_tokens and len(signals_data) > 5:
            # Reduce signals
            signals_json = json.dumps(signals_data[:5], default=str)
            estimated_tokens = self._estimate_tokens(opps_json + signals_json + jobs_json + risks_json)

        self.logger.info(
            "coordinator_context_built",
            opportunities=len(opps_data),
            signals=len(signals_data),
            jobs=len(jobs_data),
            risks=len(risks_data),
            estimated_tokens=estimated_tokens,
            target_tokens=target_tokens
        )

        return opps_json, signals_json, jobs_json, risks_json, estimated_tokens

    async def _format_report(self, state: ResearchState) -> str:
        """
        Format validated opportunities as human-readable sales briefing.

        This is the PRIMARY OUTPUT the sales rep sees. Quality here = quality of entire system.

        Incorporates:
        - Seller product expertise (what problems our products solve)
        - Signal quality gates (explicit when evidence is weak)
        - Persona-level decision makers for each opportunity
        - Consultative discovery questions that challenge status quo
        - Cialdini's 6 principles of persuasion throughout

        Report structure:
        - Executive Summary (with Liking principle - shared values)
        - Top Opportunities (with Authority, Unity principles)
        - Signal Quality Assessment (transparency builds trust)
        - Competitive Landscape
        - Discovery Questions (with Consistency principle)
        - Pre-Meeting Checklist (with Social Proof principle)
        - Recommended Next Steps (with Scarcity, Reciprocity principles)

        Args:
            state: Current research state

        Returns:
            Formatted sales briefing string
        """
        opportunities = state.get("validated_opportunities", [])
        risks = state.get("competitive_risks", [])
        signals = state.get("signals", [])
        job_postings = state.get("job_postings", [])
        account = state["account_name"]
        industry = state["industry"]
        user_context = state.get("user_context", "")
        seller_name = state.get("seller_name", "our company")  # type: ignore

        # Build compact context to stay within token limits (rate limit prevention)
        opps_json, signals_json, jobs_json, risks_json, est_tokens = self._build_compact_context(
            opportunities=opportunities,
            signals=signals,
            job_postings=job_postings,
            risks=risks,
        )

        prompt = f"""You are an elite sales strategist creating a battle-ready brief for a sales rep. The rep will read this before walking into a meeting. Make every word count.

═══════════════════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════════════════
CUSTOMER: {account} ({industry})
SELLER: {seller_name}
SALES CONTEXT: {user_context or "General research - initial outreach"}

INTELLIGENCE GATHERED:
- {len(signals)} market signals analyzed
- {len(job_postings)} job postings scanned
- {len(opportunities)} opportunities identified

RAW SIGNALS (use as evidence):
{signals_json}

JOB POSTINGS (hiring = pain points):
{jobs_json}

VALIDATED OPPORTUNITIES:
{opps_json}

COMPETITIVE RISKS:
{risks_json}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════
1. SPECIFIC NOT GENERIC - Every claim MUST reference actual data above. Quote it.
   BAD: "They're investing in cloud"
   GOOD: "Their job posting for 'AWS Solutions Architect' (posted 3 days ago) signals cloud migration"

2. NO HALLUCINATIONS - If you can't cite evidence from the data, don't say it.

3. EVIDENCE QUALITY MATTERS - Be transparent:
   - STRONG: Multiple signals confirm (job + news + tech stack)
   - MODERATE: 1-2 signals suggest
   - WEAK: Inference without direct evidence - flag for verification

4. WHY NOW - Every opportunity needs a timing trigger. What changed?

5. TALKING POINTS = ACTUAL STATEMENTS the rep can say in a meeting. Not summaries.

═══════════════════════════════════════════════════════════════════════════════
REPORT FORMAT
═══════════════════════════════════════════════════════════════════════════════

## 🎯 Executive Summary

**At a Glance:**
- **Best Opportunity:** [Product] — [X]% confidence
- **Key Decision Maker:** [Likely title based on signals]
- **Why Now:** [The specific trigger from signals/jobs that creates urgency]
- **Caveat:** [If evidence is weak, state it honestly]

---

## 💡 Top Opportunities

For each (max 3, ranked by confidence):

### 1. [Product Name] — [X]% Confidence

**The Signal:** "[Direct quote from job posting, news, or signal]"
— Source: [job posting/news/etc]

**Target Persona:** [Job title] — They care because [specific reason tied to their role]

**Why Now:** [What changed that makes this urgent? Reference the signal timestamp or trigger]

**Talking Points (say these in the meeting):**
1. "I noticed you recently posted for [actual job title]. What's driving that investment?"
2. "[Specific observation from signal] — how is that initiative progressing?"
3. "Our [product] helps [industry] teams [specific outcome]. Would that align with your [initiative from signal]?"

**If Signal is WEAK:**
⚠️ *Needs verification* — Ask: "[Specific question to confirm the hypothesis]"

---

## 🎤 Discovery Questions

Based on your research, ask these:

1. **About their hiring:** "You're hiring [actual job titles]. What capability are you building?"
2. **About their initiatives:** "[Reference actual signal] — what problem are you trying to solve?"
3. **Challenge status quo:** "How are you currently handling [pain point from signal]?"
4. **Next step opener:** "If we could [specific value prop], would that be worth a deeper conversation?"

---

## ⚠️ Risks to Watch

{risks_json if risks else "No major competitive risks identified in signals."}

**Evidence gaps to fill:** [List any opportunities where signals are weak]

---

## 🚀 Recommended Next Steps

1. **Before the meeting:** Research [specific person/topic] to fill evidence gaps
2. **Opener:** Lead with [most compelling signal/trigger]
3. **Ask:** "[Best discovery question from above]"

---
*💬 Feedback: 'approved' to finalize | 'dig deeper on [topic]' | 'find other products'*
"""

        try:
            # Use higher complexity for report generation - this is the PRIMARY OUTPUT
            # the sales rep sees, so quality here matters most
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=7,  # Use more capable model for critical output
                use_cache=False,  # Don't cache reports - each should be fresh
                max_tokens=5000  # Crisp report - user can ask for more detail
            )

            return response.content

        except Exception as e:
            self.logger.error(
                "coordinator_report_generation_failed",
                error=str(e)
            )
            # Fallback to simple report
            return self._generate_fallback_report(state)

    def _generate_fallback_report(self, state: ResearchState) -> str:
        """
        Generate simple fallback report if LLM fails.

        Args:
            state: Current research state

        Returns:
            Basic formatted report
        """
        opportunities = state.get("validated_opportunities", [])
        risks = state.get("competitive_risks", [])

        report_lines = [
            f"# Sales Intelligence Report: {state['account_name']}",
            f"Industry: {state['industry']}",
            "",
            "## Validated Opportunities",
            ""
        ]

        if opportunities:
            for i, opp in enumerate(opportunities, 1):
                if isinstance(opp, Opportunity):
                    report_lines.append(
                        f"{i}. **{opp.product_name}** (Confidence: {opp.confidence_score:.0%})"
                    )
                    report_lines.append(f"   Rationale: {opp.rationale}")
                    report_lines.append("")
                elif isinstance(opp, dict):
                    report_lines.append(
                        f"{i}. **{opp.get('product_name', 'Unknown')}** "
                        f"(Confidence: {opp.get('confidence_score', 0):.0%})"
                    )
                    report_lines.append(f"   Rationale: {opp.get('rationale', 'N/A')}")
                    report_lines.append("")
        else:
            report_lines.append("No validated opportunities found.")
            report_lines.append("")

        if risks:
            report_lines.append("## Competitive Risks")
            report_lines.append("")
            for risk in risks:
                report_lines.append(f"- {risk}")
            report_lines.append("")

        report_lines.extend([
            "---",
            "",
            "Please review the analysis above. You can:",
            "- Reply 'approved' or 'looks good' to finalize",
            "- Ask me to 'dig deeper' on specific areas",
            "- Request 'different opportunities' if these don't fit",
            "- Share any concerns for me to address"
        ])

        return "\n".join(report_lines)

    # ─────────────────────────────────────────────────────────────────────────
    # FEEDBACK ROUTING (After Human Responds)
    # ─────────────────────────────────────────────────────────────────────────

    async def process_feedback(self, state: ResearchState) -> WorkflowRoute:
        """
        Process human feedback and determine next action.

        Analyzes feedback to route to:
        - GATHERER: "dig deeper", "need more info", "research X more"
        - IDENTIFIER: "find other opportunities", "different products"
        - VALIDATOR: "re-check confidence", "seems too high/low"
        - COMPLETE: "looks good", "approved", "done"

        Args:
            state: Current research state (modified in-place)

        Returns:
            WorkflowRoute enum indicating next agent or completion
        """
        human_feedback = state.get("human_feedback", [])

        if not human_feedback:
            self.logger.warning("coordinator_no_feedback_to_process")
            return WorkflowRoute.COMPLETE

        # Get the latest feedback
        latest_feedback = human_feedback[-1] if human_feedback else ""

        self.logger.info(
            "coordinator_processing_feedback",
            feedback=latest_feedback[:100]
        )

        # Parse feedback intent using LLM
        route = await self._parse_feedback_intent(latest_feedback)

        # Update context for retry if not complete
        if route != WorkflowRoute.COMPLETE:
            await self._update_context_for_retry(state, route, latest_feedback)

            # Reset appropriate progress flags for retry
            if route == WorkflowRoute.GATHERER:
                state["progress"].gatherer_complete = False
                state["progress"].identifier_complete = False
                state["progress"].validator_complete = False
            elif route == WorkflowRoute.IDENTIFIER:
                state["progress"].identifier_complete = False
                state["progress"].validator_complete = False
            elif route == WorkflowRoute.VALIDATOR:
                state["progress"].validator_complete = False

            # Increment iteration counter
            current_iteration = state.get("workflow_iteration", 1)  # type: ignore
            state["workflow_iteration"] = current_iteration + 1  # type: ignore

            # Clear current report for re-generation
            state["current_report"] = None  # type: ignore

        # Store routing decision in state for workflow
        state["next_route"] = route.value  # type: ignore

        # Reset waiting flag
        state["waiting_for_human"] = False

        self.logger.info(
            "coordinator_feedback_processed",
            route=route.value,
            iteration=state.get("workflow_iteration", 1)  # type: ignore
        )

        return route

    async def _parse_feedback_intent(self, feedback: str) -> WorkflowRoute:
        """
        Use LLM to parse human feedback into routing decision.

        Args:
            feedback: Human feedback text

        Returns:
            WorkflowRoute based on LLM classification
        """
        prompt = f"""You are a feedback router. Analyze the user's feedback and determine the SINGLE best action.

═══════════════════════════════════════════════════════════════
USER FEEDBACK
═══════════════════════════════════════════════════════════════
"{feedback}"

═══════════════════════════════════════════════════════════════
ROUTING OPTIONS (Choose ONE)
═══════════════════════════════════════════════════════════════

**COMPLETE** - User EXPLICITLY approves the work (STRICT criteria)
- ONLY trigger words: "approved", "looks good", "perfect", "done", "ship it", "good to go", "accepted", "finalize", "save this", "that's it"
- Must be EXPLICIT approval, not just absence of complaints
- NEVER mark as complete if user is asking questions or providing information

**GATHERER** - User wants MORE DATA or provides new context (DEFAULT for unclear)
- Trigger words: "dig deeper", "more info", "research more", "find out about", "what about their [X]", "explore", "investigate"
- User identifies a TOPIC they want more information about
- User provides CONTEXT about what they already have or what they need
- User ASKS QUESTIONS about the account or products
- User mentions specific technologies, products, or use cases they want to focus on

**IDENTIFIER** - User wants DIFFERENT OPPORTUNITIES
- Trigger words: "different products", "other opportunities", "new angle", "what else", "alternatives", "not these products", "instead"
- User is not satisfied with the PRODUCTS suggested
- Wants to explore different product matches

**VALIDATOR** - User questions CONFIDENCE SCORES
- Trigger words: "confidence too high", "confidence too low", "re-evaluate", "re-score", "seems off", "disagree with rating"
- User specifically mentions scores or confidence levels
- Rare - only use when explicitly about scoring

═══════════════════════════════════════════════════════════════
DECISION PRIORITY (IMPORTANT - Follow strictly!)
═══════════════════════════════════════════════════════════════

1. If user asks a QUESTION → GATHERER (they want more info)
2. If user provides NEW CONTEXT about account → GATHERER (use this to refine search)
3. If user mentions specific technology/product they want → IDENTIFIER
4. If user questions scores → VALIDATOR
5. If EXPLICIT approval words like "approved", "done", "ship it" → COMPLETE
6. If unclear or just informational → GATHERER (NOT COMPLETE - gather more data)

CRITICAL: Only use COMPLETE if user EXPLICITLY says they approve.
Providing information ≠ approval. Asking questions ≠ approval.

═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

"looks good, approved" → COMPLETE (explicit approval)
"approved" → COMPLETE (explicit approval)
"ship it" → COMPLETE (explicit approval)
"They already use X, can you focus on Y?" → GATHERER (providing context + question)
"What about their simulation needs?" → GATHERER (asking question)
"Is there a product for CFD simulation?" → IDENTIFIER (asking about specific product)
"dig deeper on their cloud initiatives" → GATHERER
"tell me more about their hiring in AI" → GATHERER
"what about opportunities for Simulink instead?" → IDENTIFIER
"find different products to pitch" → IDENTIFIER
"the 85% confidence seems too high" → VALIDATOR
"interesting" → GATHERER (not explicit approval - gather more data)
"ok but can you also check..." → GATHERER (has a follow-up request)

═══════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════

Return JSON:
{{
    "route": "COMPLETE" | "GATHERER" | "IDENTIFIER" | "VALIDATOR",
    "reasoning": "One sentence explaining why this route was chosen",
    "context_for_retry": "If not COMPLETE, specific instructions for the agent (e.g., 'Research their cloud migration initiatives')"
}}
"""

        try:
            # Use structured output for guaranteed valid JSON
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                temperature=0,  # Deterministic for structured output
                use_cache=False,  # Don't cache feedback parsing
                response_format=FeedbackIntent.model_json_schema()
            )

            # Parse with Pydantic - guaranteed to work with structured output
            try:
                result = FeedbackIntent.model_validate_json(response.content)
            except Exception as pydantic_error:
                # Fallback to robust JSON extraction if Pydantic validation fails
                self.logger.warning(
                    "pydantic_validation_failed_using_fallback",
                    error=str(pydantic_error)
                )
                raw_result = extract_json_from_llm_response(response.content)
                result = FeedbackIntent.model_validate(raw_result)

            route_str = result.route.upper()

            self.logger.info(
                "coordinator_intent_parsed",
                route=route_str,
                reasoning=result.reasoning[:100] if result.reasoning else ""
            )

            # Map to enum
            route_map = {
                "GATHERER": WorkflowRoute.GATHERER,
                "IDENTIFIER": WorkflowRoute.IDENTIFIER,
                "VALIDATOR": WorkflowRoute.VALIDATOR,
                "COMPLETE": WorkflowRoute.COMPLETE
            }

            return route_map.get(route_str, WorkflowRoute.COMPLETE)

        except (json.JSONDecodeError, JSONParseError) as e:
            self.logger.warning(
                "coordinator_intent_json_parse_failed",
                error=str(e),
                feedback=feedback[:50]
            )
            # Default to complete on parse failure
            return WorkflowRoute.COMPLETE
        except Exception as e:
            self.logger.warning(
                "coordinator_intent_parsing_failed",
                error=str(e),
                feedback=feedback[:50]
            )
            return WorkflowRoute.COMPLETE

    async def _update_context_for_retry(
        self,
        state: ResearchState,
        route: WorkflowRoute,
        feedback: str
    ) -> None:
        """
        Update state with feedback context for retry loop.

        Adds context to help downstream agents adjust their behavior:
        - What the human didn't like
        - What they want to see different
        - Specific areas to focus on

        Uses ACTUAL signals and opportunities from state - no hardcoded examples.

        Args:
            state: Current research state (modified in-place)
            route: Determined routing decision
            feedback: Original human feedback
        """
        # Get actual data from state to provide context
        account_name = state.get("account_name", "the target company")
        industry = state.get("industry", "")
        signals = state.get("signals", [])
        opportunities = state.get("validated_opportunities", []) or state.get("opportunities", [])

        # Build signal summary from ACTUAL data
        signal_summary = ""
        if signals:
            signal_items = []
            for sig in signals[:5]:
                if hasattr(sig, 'signal_type'):
                    signal_items.append(f"- {sig.signal_type}: {sig.content[:150] if sig.content else ''}")
                elif isinstance(sig, dict):
                    signal_items.append(f"- {sig.get('signal_type', 'signal')}: {sig.get('content', '')[:150]}")
            signal_summary = "\n".join(signal_items)

        # Build opportunity summary from ACTUAL data
        opp_summary = ""
        if opportunities:
            opp_items = []
            for opp in opportunities[:3]:
                if hasattr(opp, 'product_name'):
                    opp_items.append(f"- {opp.product_name}: {opp.rationale[:100] if opp.rationale else ''}")
                elif isinstance(opp, dict):
                    opp_items.append(f"- {opp.get('product_name', 'product')}: {opp.get('rationale', '')[:100]}")
            opp_summary = "\n".join(opp_items)

        # Agent-specific guidance (no hardcoded company examples)
        agent_guidance = {
            WorkflowRoute.GATHERER: f"""
The GATHERER agent collects data from web searches, job postings, and news.
Generate instructions specific to {account_name} based on their industry ({industry}) and the signals already found.
Focus on topics the user wants to explore deeper.""",
            WorkflowRoute.IDENTIFIER: f"""
The IDENTIFIER agent matches seller products to customer needs.
Generate instructions to find different/better product matches for {account_name}.
Consider what products might address the signals already identified.""",
            WorkflowRoute.VALIDATOR: f"""
The VALIDATOR agent scores opportunities and assesses risks.
Generate instructions to re-evaluate scores for {account_name}'s opportunities.
Reference specific products and why their confidence might need adjustment."""
        }

        prompt = f"""You are translating user feedback into SPECIFIC INSTRUCTIONS for the {route.value} agent.

═══════════════════════════════════════════════════════════════
CURRENT RESEARCH TARGET
═══════════════════════════════════════════════════════════════
Account: {account_name}
Industry: {industry}

═══════════════════════════════════════════════════════════════
SIGNALS ALREADY FOUND
═══════════════════════════════════════════════════════════════
{signal_summary if signal_summary else "No signals gathered yet."}

═══════════════════════════════════════════════════════════════
OPPORTUNITIES IDENTIFIED
═══════════════════════════════════════════════════════════════
{opp_summary if opp_summary else "No opportunities identified yet."}

═══════════════════════════════════════════════════════════════
USER FEEDBACK
═══════════════════════════════════════════════════════════════
"{feedback}"

═══════════════════════════════════════════════════════════════
AGENT CONTEXT
═══════════════════════════════════════════════════════════════
{agent_guidance.get(route, "Provide specific actionable guidance.")}

═══════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════

Convert the user's feedback into a CLEAR, ACTIONABLE instruction for {account_name}.

CRITICAL RULES:
1. ONLY reference {account_name} - do NOT use example companies
2. Use the actual signals and opportunities shown above as context
3. Be SPECIFIC - name exact topics from the signals or user feedback
4. Be DIRECTIVE - use imperative language ("Research X", "Focus on Y")
5. Be CONCISE - one paragraph max

═══════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════

Return ONLY the instruction text (no JSON, no preamble). Start directly with the action.
Reference {account_name} specifically, NOT example companies.
"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                use_cache=False
            )

            context = response.content.strip()

            # Store in state
            state["feedback_context"] = context  # type: ignore

            # Also append to user_context for visibility to all agents
            existing_context = state.get("user_context") or ""
            feedback_addition = f"\n[Feedback Round {state.get('workflow_iteration', 1)}]: {context}"  # type: ignore

            if existing_context:
                state["user_context"] = existing_context + feedback_addition
            else:
                state["user_context"] = feedback_addition.strip()

            self.logger.info(
                "coordinator_context_updated",
                route=route.value,
                context=context[:100]
            )

        except Exception as e:
            self.logger.warning(
                "coordinator_context_update_failed",
                error=str(e)
            )
            # Store raw feedback as fallback
            state["feedback_context"] = feedback  # type: ignore

    # ─────────────────────────────────────────────────────────────────────────
    # BASE CLASS REQUIREMENTS
    # ─────────────────────────────────────────────────────────────────────────

    def get_complexity(self, state: ResearchState) -> int:
        """
        Get task complexity for model routing.

        CoordinatorAgent uses LOCAL Ollama (complexity=3) for:
        - Input validation
        - Name normalization
        - Question generation
        - Report formatting
        - Feedback parsing

        All tasks are classification/formatting - no complex reasoning needed.

        Args:
            state: Current research state

        Returns:
            Complexity score: 3 (routes to Tier 1 LOCAL Ollama)
        """
        return 3
