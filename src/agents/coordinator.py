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

        # LLM-based validation for quality
        prompt = f"""Validate this research request for potential issues:

Account Name: {account_name}
Industry: {industry}
Region: {state.get("region", "Not specified")}
Additional Context: {state.get("user_context", "Not provided")}

Check for:
1. Is the account name a plausible company name? (not gibberish like "asdfgh")
2. Is the industry a recognized business category?
3. Any obvious typos that should be corrected?
4. Any red flags or concerns?

Return JSON:
{{
    "is_valid": true,
    "errors": [],
    "suggested_corrections": {{}},
    "concerns": []
}}

If there are issues, set is_valid to false and populate errors array.
If you detect typos, add them to suggested_corrections as {{"field": "corrected_value"}}.
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

            # Apply suggested corrections to state
            if result.suggested_corrections:
                if "account_name" in result.suggested_corrections:
                    state["account_name"] = result.suggested_corrections["account_name"]
                    self.logger.info(
                        "coordinator_applied_correction",
                        field="account_name",
                        corrected=result.suggested_corrections["account_name"]
                    )
                if "industry" in result.suggested_corrections:
                    state["industry"] = result.suggested_corrections["industry"]
                    self.logger.info(
                        "coordinator_applied_correction",
                        field="industry",
                        corrected=result.suggested_corrections["industry"]
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
        Smart questioning - LLM decides if clarification needed for strategic advice.

        Considers:
        - Missing strategic context (sales objective, relationship, competitive situation)
        - Ambiguous inputs (e.g., "Amazon" - AWS or Retail?)
        - Whether we have enough info to provide ACTIONABLE recommendations

        The goal is to provide PRACTICAL STRATEGIC ADVICE, not generic research.
        Without context, we can only provide generic outputs that aren't useful.

        Args:
            state: Current research state

        Returns:
            Question string if clarification would improve results, else None
        """
        user_context = state.get("user_context") or ""
        has_meaningful_context = len(user_context.strip()) > 50  # More than a sentence

        prompt = f"""Evaluate if clarifying questions are needed to provide PRACTICAL STRATEGIC ADVICE.

Account Name: {state["account_name"]}
Industry: {state["industry"]}
Region: {state.get("region") or "Not specified"}
Additional Context: {user_context or "None provided"}
Research Depth: {state["research_depth"].value}

Your goal is to help a sales professional prepare for customer engagement. Generic research
without context produces generic advice that isn't actionable.

KEY STRATEGIC CONTEXT CHECKLIST (check what's missing):
1. SALES OBJECTIVE - What's the purpose? (discovery call, QBR, renewal, expansion, new logo)
2. RELATIONSHIP STATUS - New prospect or existing customer? How long?
3. CURRENT PRODUCTS - What do they already own? (helps identify upsell vs cross-sell)
4. KNOWN INITIATIVES - Any specific projects, pain points, or goals mentioned?
5. COMPETITIVE SITUATION - Any competitor products being evaluated or in use?
6. BUDGET/TIMING - Any known budget cycles or decision timelines?

DECISION RULES:
- If "Additional Context" is "None provided" or very sparse → ASK questions (we need context for strategic advice)
- If context mentions specific products, initiatives, or sales objectives → DON'T ask (we have enough)
- If company name is ambiguous (Amazon, GE, etc.) → ASK for clarification
- If research is clearly just exploratory with no sales objective → ASK what they want to achieve

Return JSON:
{{
    "needs_clarification": true/false,
    "questions": "Your strategic questions here (2-4 questions, focused on sales context)",
    "reasoning": "Why this context would help provide actionable advice"
}}

Example questions to ask when context is sparse:
"To provide actionable strategic advice for {state["account_name"]}, I'd like to understand:

1. What's your sales objective? (e.g., preparing for discovery call, QBR, expansion opportunity)
2. What's your current relationship? (new prospect, existing customer - if so, what products do they have?)
3. Are there any specific initiatives, pain points, or competitive threats you're aware of?
4. Any particular product areas you want me to focus on?

This context will help me identify the most relevant opportunities and talking points."

If context IS sufficient (mentions objectives, products, or specific focus areas), return:
{{
    "needs_clarification": false,
    "questions": null,
    "reasoning": "Context provides enough information for strategic research"
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

        # Present to human with feedback prompt
        state["human_question"] = report
        state["waiting_for_human"] = True

        # Increment workflow iteration counter
        current_iteration = state.get("workflow_iteration", 1)  # type: ignore
        state["workflow_iteration"] = current_iteration  # type: ignore

        self.logger.info(
            "coordinator_exit_completed",
            report_length=len(report),
            iteration=current_iteration
        )

    async def _format_report(self, state: ResearchState) -> str:
        """
        Format validated opportunities as human-readable report.

        Report structure:
        - Executive Summary
        - Top Opportunities (sorted by confidence)
        - Evidence for each opportunity
        - Competitive Risks
        - Feedback prompt

        Args:
            state: Current research state

        Returns:
            Formatted report string
        """
        opportunities = state.get("validated_opportunities", [])
        risks = state.get("competitive_risks", [])
        account = state["account_name"]
        industry = state["industry"]
        signals_count = len(state.get("signals", []))
        jobs_count = len(state.get("job_postings", []))

        # Build opportunities JSON for LLM
        opps_data = []
        for opp in opportunities:
            if isinstance(opp, Opportunity):
                opps_data.append(opp.model_dump())
            elif isinstance(opp, dict):
                opps_data.append(opp)

        prompt = f"""Create a professional sales intelligence report for {account} ({industry}).

Research Summary:
- Signals collected: {signals_count}
- Job postings analyzed: {jobs_count}
- Validated opportunities: {len(opportunities)}

Validated Opportunities:
{json.dumps(opps_data, indent=2, default=str)}

Competitive Risks:
{json.dumps(risks, indent=2) if risks else "None identified"}

Create a report with these sections:

## Executive Summary
(2-3 sentences summarizing the key findings and top recommendation)

## Top Opportunities
(For each opportunity, include:
- Product name and confidence score
- Why they likely need this product (rationale)
- Key evidence supporting this opportunity
- Suggested talking points)

## Competitive Landscape
(Any risks or competitive concerns to be aware of)

## Recommended Next Steps
(2-3 actionable next steps for the sales team)

---

After the report, include this exact text:
"Please review the analysis above. You can:
- Reply 'approved' or 'looks good' to finalize
- Ask me to 'dig deeper' on specific areas
- Request 'different opportunities' if these don't fit
- Share any concerns for me to address"
"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                use_cache=False,  # Don't cache reports
                max_tokens=3000
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
        prompt = f"""Analyze this human feedback and classify the appropriate action:

Feedback: "{feedback}"

Classify as ONE of:
- GATHERER: User wants more data, deeper research, additional sources, more information about specific topics
- IDENTIFIER: User wants different opportunities, other products, new angles, alternative suggestions
- VALIDATOR: User questions confidence scores, thinks ratings are too high/low, wants re-evaluation
- COMPLETE: User is satisfied, approves the report, says looks good, done, accepted

Examples:
- "dig deeper on their cloud initiatives" -> GATHERER
- "find opportunities for different products" -> IDENTIFIER
- "the confidence seems too high for Simulink" -> VALIDATOR
- "looks good, approved" -> COMPLETE
- "need more information about their hiring" -> GATHERER
- "what about other toolboxes?" -> IDENTIFIER

Return JSON:
{{
    "route": "GATHERER" | "IDENTIFIER" | "VALIDATOR" | "COMPLETE",
    "reasoning": "Brief explanation of classification",
    "context_for_retry": "Specific guidance for the next agent based on feedback"
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

        Args:
            state: Current research state (modified in-place)
            route: Determined routing decision
            feedback: Original human feedback
        """
        # Build context based on route
        prompt = f"""Extract specific guidance for the {route.value} agent from this feedback:

Feedback: "{feedback}"
Route: {route.value}

What specific adjustments should the {route.value} agent make?
Be concise and actionable.

Return only the guidance text, no JSON.
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
