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
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                use_cache=True
            )

            # Use robust JSON extraction to handle varied LLM output formats
            result = extract_json_from_llm_response(response.content)

            if not result.get("is_valid", True):
                errors.extend(result.get("errors", []))

            # Apply suggested corrections to state
            corrections = result.get("suggested_corrections", {})
            if corrections:
                if "account_name" in corrections:
                    state["account_name"] = corrections["account_name"]
                    self.logger.info(
                        "coordinator_applied_correction",
                        field="account_name",
                        corrected=corrections["account_name"]
                    )
                if "industry" in corrections:
                    state["industry"] = corrections["industry"]
                    self.logger.info(
                        "coordinator_applied_correction",
                        field="industry",
                        corrected=corrections["industry"]
                    )

            # Log concerns but don't block
            concerns = result.get("concerns", [])
            if concerns:
                self.logger.info(
                    "coordinator_validation_concerns",
                    concerns=concerns
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

    async def _normalize_company_name(self, name: str) -> str:
        """
        Minimal enrichment - normalize company name.

        Examples:
        - "msft" -> "Microsoft"
        - "BOEING CO" -> "Boeing"
        - "amazon.com" -> "Amazon"

        Uses LLM for intelligent normalization.

        Args:
            name: Original company name

        Returns:
            Normalized company name
        """
        prompt = f"""Normalize this company name to its standard, commonly-used form:

Input: "{name}"

Rules:
1. Expand common abbreviations (MSFT -> Microsoft, AAPL -> Apple)
2. Remove legal suffixes unless important (Inc, Corp, LLC)
3. Fix capitalization to standard form
4. Remove domain extensions (.com, .io)
5. Keep the name recognizable and professional

Return ONLY the normalized name, nothing else. If the name is already normal, return it unchanged.
"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                use_cache=True
            )

            normalized = response.content.strip().strip('"').strip("'")

            # Sanity check - don't return empty or very different names
            if not normalized or len(normalized) > len(name) * 3:
                return name

            return normalized

        except Exception as e:
            self.logger.warning(
                "coordinator_normalization_failed",
                name=name,
                error=str(e)
            )
            return name  # Return original on failure

    async def _generate_clarifying_questions(self, state: ResearchState) -> str | None:
        """
        Smart questioning - LLM decides if clarification needed.

        Considers:
        - Missing optional fields (region, user_context)
        - Ambiguous inputs (e.g., "Amazon" - AWS or Retail?)
        - Research depth appropriateness

        Args:
            state: Current research state

        Returns:
            Question string if clarification would improve results, else None
        """
        prompt = f"""Evaluate if clarifying questions would SIGNIFICANTLY improve research results:

Account Name: {state["account_name"]}
Industry: {state["industry"]}
Region: {state.get("region") or "Not specified"}
Additional Context: {state.get("user_context") or "Not specified"}
Research Depth: {state["research_depth"].value}

Consider:
1. Is the company name ambiguous? (e.g., "Amazon" could be AWS, Retail, or both)
2. Would knowing the geographic region substantially help focus the research?
3. Is additional context needed to identify relevant sales opportunities?
4. Is the research depth appropriate for the company size?

IMPORTANT: Only ask if clarification would SIGNIFICANTLY improve results.
If the inputs are reasonably clear, do NOT ask questions - just return null.

Return JSON:
{{
    "needs_clarification": false,
    "questions": null,
    "reasoning": "Inputs are sufficiently clear for research"
}}

OR if clarification needed:
{{
    "needs_clarification": true,
    "questions": "Your 1-2 focused questions here",
    "reasoning": "Why this clarification would significantly help"
}}
"""

        try:
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                use_cache=True
            )

            # Use robust JSON extraction to handle varied LLM output formats
            result = extract_json_from_llm_response(response.content)

            if result.get("needs_clarification", False):
                questions = result.get("questions")
                if questions:
                    self.logger.info(
                        "coordinator_question_generated",
                        reasoning=result.get("reasoning", "")[:100]
                    )
                    return questions

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
            response = await self.model_router.generate(
                prompt=prompt,
                complexity=3,  # LOCAL Ollama
                use_cache=False  # Don't cache feedback parsing
            )

            # Use robust JSON extraction to handle varied LLM output formats
            result = extract_json_from_llm_response(response.content)
            route_str = result.get("route", "COMPLETE").upper()

            self.logger.info(
                "coordinator_intent_parsed",
                route=route_str,
                reasoning=result.get("reasoning", "")[:100]
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
