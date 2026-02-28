"""
LangGraph workflow definition.
Defines the multi-agent research workflow with checkpointing and feedback loops.

Workflow Architecture:
1. coordinator_entry: Validate inputs, ask clarifying questions
2. gatherer: Collect data from multiple sources (using MCP tools)
3. identifier: Analyze data, find opportunities
4. validator: Validate opportunities, assess risks
5. coordinator_exit: Present results, set up for human feedback
6. (Human Feedback)
7. coordinator_feedback: Route based on feedback (loop or complete)
"""
from typing import Literal
from datetime import datetime
import asyncio
import threading

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from ..models.state import ResearchState
from ..config import settings
from ..core.model_router import ModelRouter
from ..agents.coordinator import CoordinatorAgent, WorkflowRoute
from ..agents.gatherer import GathererAgent
from ..agents.identifier import IdentifierAgent
from ..agents.validator import ValidatorAgent
from ..data_sources.search_client import SearchClient
from ..data_sources.job_boards import JobBoardScraper
from ..data_sources.product_catalog import ProductMatcher

import structlog
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sse_callbacks import SSECallbackHandler

logger = structlog.get_logger(__name__)


def _print_stage(stage: str, description: str, status: str = "running") -> None:
    """
    Print a user-friendly workflow stage indicator.

    Args:
        stage: Short stage name (e.g., "GATHERING")
        description: What the stage does
        status: "running", "complete", or "waiting"
    """
    # Status indicators
    indicators = {
        "running": "[...]",
        "complete": "[OK]",
        "waiting": "[?]",
        "error": "[!]",
    }
    indicator = indicators.get(status, "[...]")

    # Print to stdout for user visibility
    print(f"\n{indicator} Stage: {stage}")
    print(f"    {description}")
    sys.stdout.flush()  # Ensure immediate output


class ResearchWorkflow:
    """
    Main research workflow orchestrating all agents.

    Workflow with feedback loops:
    ┌─────────────────────────────────────────────────────────────────┐
    │  coordinator_entry -> gatherer -> identifier -> validator      │
    │         │                                           │          │
    │         v                                           v          │
    │  (human clarification)                    coordinator_exit     │
    │                                                     │          │
    │                                                     v          │
    │                                            (human feedback)    │
    │                                                     │          │
    │                                                     v          │
    │                                          coordinator_feedback  │
    │                                                     │          │
    │                    ┌──────────┬──────────┬──────────┤          │
    │                    v          v          v          v          │
    │              gatherer   identifier  validator     END          │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        seller_name: str = "MathWorks",
        model_router: ModelRouter | None = None,
        mcp_client: SearchClient | None = None,
        job_scraper: JobBoardScraper | None = None
    ):
        """
        Initialize workflow with dependencies.

        Args:
            seller_name: Name of the seller/vendor company whose products to match.
                         This is YOUR company (e.g., "MathWorks", "Salesforce").
                         Products must be indexed first using ProductCatalogIndexer.
            model_router: Optional ModelRouter instance (creates default if None)
            mcp_client: Optional MCP client for web search
            job_scraper: Optional job board scraper
        """
        # Store seller name - this is the company whose products we're selling
        self.seller_name = seller_name

        # Initialize dependencies
        self.model_router = model_router or ModelRouter()
        self.mcp_client = mcp_client or SearchClient()
        self.job_scraper = job_scraper or JobBoardScraper()

        # Initialize agents (Identifier created lazily with company name)
        self.coordinator = CoordinatorAgent(model_router=self.model_router)
        self.gatherer = GathererAgent(
            mcp_client=self.mcp_client,
            job_scraper=self.job_scraper,
            model_router=self.model_router
        )
        self.validator = ValidatorAgent(
            model_router=self.model_router
        )

        # Single ProductMatcher for the seller (shared across all account analyses)
        self._product_matcher: ProductMatcher | None = None
        self._identifier: IdentifierAgent | None = None

        self.graph = self._build_graph()
        self.checkpointer = None
        self.app = None
        self._setup_checkpointing()

        # SSE callback handler for real-time frontend updates
        # Set by workflow_service before running
        self._sse_callback: "SSECallbackHandler | None" = None

        # Cancellation event for stopping workflow mid-execution
        # Set by workflow_service when stop is requested
        self._cancel_event: threading.Event | None = None

        # LangSmith shareable run URL (set after run() completes if LangSmith configured)
        self._langsmith_url: str | None = None

    def _setup_checkpointing(self) -> None:
        """Initialize SQLite checkpointing (synchronous version)."""
        import os

        # Ensure checkpoint directory exists
        os.makedirs(settings.checkpoint_dir, exist_ok=True)

        # Create checkpoint database path (Windows-compatible)
        db_path = os.path.join(settings.checkpoint_dir, "checkpoints.db")

        # Create connection
        conn = sqlite3.connect(db_path, check_same_thread=False)

        # Create checkpointer
        self.checkpointer = SqliteSaver(conn)

        # Compile graph with checkpointing
        # interrupt_before allows human-in-loop at specific nodes
        self.app = self.graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["_wait_for_human"]
        )

        logger.info("checkpointing_enabled", db_path=db_path)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with feedback loops."""
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("coordinator_entry", self._coordinator_entry_node)
        workflow.add_node("gatherer", self._gatherer_node)
        workflow.add_node("identifier", self._identifier_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("coordinator_exit", self._coordinator_exit_node)
        workflow.add_node("coordinator_feedback", self._coordinator_feedback_node)
        workflow.add_node("_wait_for_human", self._wait_for_human_node)

        # Set entry point
        workflow.set_entry_point("coordinator_entry")

        # Define edges with conditional routing

        # After coordinator_entry: check if we need human clarification
        workflow.add_conditional_edges(
            "coordinator_entry",
            self._route_after_entry,
            {
                "wait_for_human": "_wait_for_human",
                "continue": "gatherer"
            }
        )

        # After waiting for human: route based on context
        # - If from entry (no current_report): continue to gatherer
        # - If from exit (has current_report): go to feedback processing
        workflow.add_conditional_edges(
            "_wait_for_human",
            self._route_after_human_input,
            {
                "gatherer": "gatherer",
                "coordinator_feedback": "coordinator_feedback"
            }
        )

        # Main flow: gatherer -> identifier -> validator -> coordinator_exit
        workflow.add_edge("gatherer", "identifier")
        workflow.add_edge("identifier", "validator")
        workflow.add_edge("validator", "coordinator_exit")

        # After coordinator_exit: always wait for human feedback
        workflow.add_edge("coordinator_exit", "_wait_for_human")

        # After human feedback is processed, route based on decision
        workflow.add_conditional_edges(
            "coordinator_feedback",
            self._route_after_feedback,
            {
                "gatherer": "gatherer",
                "identifier": "identifier",
                "validator": "validator",
                "complete": END
            }
        )

        return workflow

    # ─────────────────────────────────────────────────────────────────────────
    # CANCELLATION SUPPORT
    # ─────────────────────────────────────────────────────────────────────────

    def _check_cancelled(self, state: ResearchState) -> bool:
        """
        Check if workflow cancellation was requested.

        If cancelled, marks the state appropriately for later resumption.

        Returns:
            True if cancelled, False otherwise
        """
        if self._cancel_event and self._cancel_event.is_set():
            logger.info("workflow_cancellation_detected")
            # Mark state for later resumption
            state["status"] = "stopped"
            state["waiting_for_human"] = True  # Allows resume via feedback
            state["human_question"] = "Research was paused. Click Resume to continue."
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # NODE FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _coordinator_entry_node(self, state: ResearchState) -> ResearchState:
        """
        Coordinator entry point - validate inputs and ask clarifying questions.

        Uses CoordinatorAgent.process_entry() to:
        - Validate required fields
        - Normalize company name
        - Generate clarifying questions if needed
        """
        # Check for cancellation at start of node
        if self._check_cancelled(state):
            _print_stage("CANCELLED", "Research stopped by user", "complete")
            return state

        account = state.get("account_name", "company")
        _print_stage("INITIALIZING", f"Validating inputs for {account}...", "running")

        logger.info(
            "coordinator_entry_started",
            account=account
        )

        # Emit SSE event for frontend
        if self._sse_callback:
            self._sse_callback.on_node_start("coordinator_entry", state)

        # Run async process in sync context
        asyncio.run(self.coordinator.process_entry(state))

        needs_human = state.get("waiting_for_human", False)
        if needs_human:
            _print_stage("INITIALIZING", f"Need clarification for {account}", "waiting")
        else:
            _print_stage("INITIALIZING", f"Ready to research {account}", "complete")

        logger.info(
            "coordinator_entry_completed",
            needs_human=needs_human
        )

        # Emit SSE completion event
        if self._sse_callback:
            self._sse_callback.on_node_end("coordinator_entry", state)

        return state

    def _gatherer_node(self, state: ResearchState) -> ResearchState:
        """
        Gatherer agent - collect data from multiple sources.

        Uses GathererAgent.process() to:
        - Search web for company info
        - Collect job postings
        - Gather news articles
        - Analyze each source with LLM
        """
        # Check for cancellation at start of node
        if self._check_cancelled(state):
            _print_stage("CANCELLED", "Stopping at gatherer", "complete")
            return state

        account = state.get("account_name", "company")
        _print_stage("GATHERING DATA", f"Searching web, jobs, and news for {account}...", "running")

        logger.info(
            "gatherer_started",
            account=account,
            feedback_context=state.get("feedback_context")
        )

        # Emit SSE event for frontend
        if self._sse_callback:
            self._sse_callback.on_node_start("gatherer", state)

        # Run async process in sync context with proper MCP session initialization
        # The MCP client requires async context manager to initialize the session
        async def run_gatherer_with_mcp():
            async with self.mcp_client:
                await self.gatherer.process(state)

        asyncio.run(run_gatherer_with_mcp())

        signals_count = len(state.get("signals", []))
        jobs_count = len(state.get("job_postings", []))
        news_count = len(state.get("news_items", []))

        _print_stage(
            "GATHERING DATA",
            f"Found {signals_count} signals, {jobs_count} jobs, {news_count} news items",
            "complete"
        )

        logger.info(
            "gatherer_completed",
            signals_count=signals_count,
            jobs_count=jobs_count
        )

        # Emit SSE completion event
        if self._sse_callback:
            self._sse_callback.on_node_end("gatherer", state)

        return state

    def _identifier_node(self, state: ResearchState) -> ResearchState:
        """
        Identifier agent - find opportunities from gathered data.

        Uses IdentifierAgent.process() to:
        - Extract requirements from signals and job postings
        - Match to products using semantic search
        - Generate opportunity hypotheses
        """
        # Check for cancellation at start of node
        if self._check_cancelled(state):
            _print_stage("CANCELLED", "Stopping at identifier", "complete")
            return state

        account_name = state.get("account_name", "unknown")
        _print_stage("IDENTIFYING OPPORTUNITIES", f"Analyzing data and matching products for {account_name}...", "running")

        logger.info(
            "identifier_started",
            account=account_name,
            feedback_context=state.get("feedback_context")
        )

        # Emit SSE event for frontend
        if self._sse_callback:
            self._sse_callback.on_node_start("identifier", state)

        # Create IdentifierAgent with seller's product catalog (lazily initialized)
        # The ProductMatcher uses the SELLER's products (e.g., MathWorks),
        # not the target account's products (e.g., Boeing)
        if self._identifier is None:
            try:
                self._product_matcher = ProductMatcher(company_name=self.seller_name)
                self._identifier = IdentifierAgent(
                    product_matcher=self._product_matcher,
                    model_router=self.model_router
                )
                logger.info(
                    "identifier_initialized",
                    seller=self.seller_name,
                    collection=self._product_matcher.collection_name
                )
            except Exception as e:
                logger.error(
                    "product_catalog_not_indexed",
                    seller=self.seller_name,
                    error=str(e)
                )
                raise ValueError(
                    f"Product catalog not indexed for seller '{self.seller_name}'. "
                    f"Run: python -m src.cli.setup_catalog --seller {self.seller_name}"
                ) from e

        identifier = self._identifier

        # Run async process in sync context
        asyncio.run(identifier.process(state))

        opportunities_count = len(state.get("opportunities", []))
        _print_stage(
            "IDENTIFYING OPPORTUNITIES",
            f"Found {opportunities_count} potential opportunities",
            "complete"
        )

        logger.info(
            "identifier_completed",
            opportunities_count=opportunities_count
        )

        # Emit SSE completion event
        if self._sse_callback:
            self._sse_callback.on_node_end("identifier", state)

        return state

    def _validator_node(self, state: ResearchState) -> ResearchState:
        """
        Validator agent - validate and score opportunities.

        Uses ValidatorAgent.process() to:
        - Assess competitive risks
        - Score confidence for each opportunity
        - Filter low-confidence opportunities
        """
        # Check for cancellation at start of node
        if self._check_cancelled(state):
            _print_stage("CANCELLED", "Stopping at validator", "complete")
            return state

        account = state.get("account_name", "company")
        opp_count = len(state.get("opportunities", []))
        _print_stage("VALIDATING", f"Scoring {opp_count} opportunities and assessing risks...", "running")

        logger.info(
            "validator_started",
            account=account,
            opportunities_count=opp_count
        )

        # Emit SSE event for frontend
        if self._sse_callback:
            self._sse_callback.on_node_start("validator", state)

        # Run async process in sync context
        asyncio.run(self.validator.process(state))

        validated_count = len(state.get("validated_opportunities", []))
        risks_count = len(state.get("competitive_risks", []))

        _print_stage(
            "VALIDATING",
            f"Validated {validated_count} opportunities, identified {risks_count} risks",
            "complete"
        )

        logger.info(
            "validator_completed",
            validated_count=validated_count,
            risks_count=risks_count
        )

        # Emit SSE completion event
        if self._sse_callback:
            self._sse_callback.on_node_end("validator", state)

        return state

    def _coordinator_exit_node(self, state: ResearchState) -> ResearchState:
        """
        Coordinator exit point - format report and prepare for human feedback.

        Uses CoordinatorAgent.process_exit() to:
        - Format validated opportunities as report
        - Set up human-in-loop for feedback
        """
        # Check for cancellation at start of node
        if self._check_cancelled(state):
            _print_stage("CANCELLED", "Stopping at coordinator exit", "complete")
            return state

        validated_count = len(state.get("validated_opportunities", []))
        _print_stage("PREPARING REPORT", f"Formatting {validated_count} validated opportunities...", "running")

        logger.info(
            "coordinator_exit_started",
            opportunities=validated_count
        )

        # Emit SSE event for frontend
        if self._sse_callback:
            self._sse_callback.on_node_start("coordinator_exit", state)

        # Run async process in sync context
        asyncio.run(self.coordinator.process_exit(state))

        _print_stage("PREPARING REPORT", "Report ready for review", "complete")

        logger.info(
            "coordinator_exit_completed",
            report_length=len(state.get("current_report") or "")
        )

        # Emit SSE completion event
        if self._sse_callback:
            self._sse_callback.on_node_end("coordinator_exit", state)

        return state

    def _coordinator_feedback_node(self, state: ResearchState) -> ResearchState:
        """
        Process human feedback and determine next routing.

        Uses CoordinatorAgent.process_feedback() to:
        - Parse feedback intent
        - Determine routing (gatherer/identifier/validator/complete)
        - Update context for retry if needed
        """
        _print_stage("PROCESSING FEEDBACK", "Analyzing your feedback...", "running")

        logger.info(
            "coordinator_feedback_started",
            feedback_count=len(state.get("human_feedback", []))
        )

        # Emit SSE event for frontend (re-use coordinator_entry for feedback processing)
        if self._sse_callback:
            self._sse_callback.on_node_start("coordinator_feedback", state)

        # Run async process in sync context
        asyncio.run(self.coordinator.process_feedback(state))

        next_route = state.get("next_route", "complete")
        route_descriptions = {
            "gatherer": "Will gather more data",
            "identifier": "Will find different opportunities",
            "validator": "Will re-evaluate scores",
            "complete": "Research complete!"
        }
        desc = route_descriptions.get(next_route, f"Routing to {next_route}")
        status = "complete" if next_route == "complete" else "running"
        _print_stage("PROCESSING FEEDBACK", desc, status)

        logger.info(
            "coordinator_feedback_completed",
            next_route=next_route
        )

        # Emit SSE completion event
        if self._sse_callback:
            self._sse_callback.on_node_end("coordinator_feedback", state)

        return state

    def _wait_for_human_node(self, state: ResearchState) -> ResearchState:
        """
        Placeholder node for human-in-loop interrupts.

        The graph will interrupt before this node when waiting_for_human is True.
        After human provides input, the workflow resumes.
        """
        _print_stage("AWAITING INPUT", "Waiting for your response...", "waiting")

        logger.info(
            "wait_for_human",
            question=state.get("human_question", "")[:100] if state.get("human_question") else None
        )

        # Emit SSE event for frontend - waiting for human
        if self._sse_callback:
            self._sse_callback.on_node_start("_wait_for_human_node", state)
            # Also emit waiting_for_human event
            self._sse_callback.on_node_end("_wait_for_human_node", state)

        # If we have feedback and came from coordinator_exit, process it
        if state.get("human_feedback") and state.get("current_report"):
            # Route to feedback processing
            state["waiting_for_human"] = False
            return state

        # Otherwise just clear the waiting flag
        state["waiting_for_human"] = False

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTING FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _route_after_entry(
        self, state: ResearchState
    ) -> Literal["wait_for_human", "continue"]:
        """
        Route after coordinator entry.

        Returns:
            "wait_for_human" if clarification needed
            "continue" to proceed to gatherer
        """
        if state.get("waiting_for_human", False):
            return "wait_for_human"
        return "continue"

    def _route_after_human_input(
        self, state: ResearchState
    ) -> Literal["gatherer", "coordinator_feedback"]:
        """
        Route after human provides input.

        Returns:
            "gatherer" if came from entry (starting research)
            "coordinator_feedback" if came from exit (processing feedback)
        """
        # If we have a current_report, we came from coordinator_exit
        # and need to process feedback
        if state.get("current_report"):
            return "coordinator_feedback"

        # Otherwise, we came from coordinator_entry and continue to gatherer
        return "gatherer"

    def _route_after_feedback(
        self, state: ResearchState
    ) -> Literal["gatherer", "identifier", "validator", "complete"]:
        """
        Route after coordinator processes feedback.

        Returns:
            Route based on state["next_route"] set by CoordinatorAgent
        """
        next_route = state.get("next_route", "complete")

        # Validate route
        valid_routes = {"gatherer", "identifier", "validator", "complete"}
        if next_route not in valid_routes:
            logger.warning(
                "invalid_route_defaulting_to_complete",
                invalid_route=next_route
            )
            return "complete"

        return next_route  # type: ignore

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        state: ResearchState,
        thread_id: str | None = None,
        sse_callback: "SSECallbackHandler | None" = None,
        cancel_event: threading.Event | None = None
    ) -> ResearchState:
        """
        Run the research workflow (synchronous version).

        The workflow will pause at human-in-loop points and can be resumed.

        Args:
            state: Initial research state
            thread_id: Optional thread ID for checkpointing
            sse_callback: Optional SSE callback handler for frontend updates
            cancel_event: Optional threading event to signal cancellation

        Returns:
            Research state (may be incomplete if waiting for human or cancelled)
        """
        # Store SSE callback and cancel event for node functions to use
        self._sse_callback = sse_callback
        self._cancel_event = cancel_event

        import os
        from uuid import uuid4

        # Generate a unique run ID for LangSmith tracing
        run_id = str(uuid4())
        self._langsmith_url = None  # Reset from any previous run

        # Create config for checkpointing + LangSmith run ID
        config = {
            "configurable": {
                "thread_id": thread_id or f"research_{state['account_name']}"
            },
            "run_id": run_id,
        }

        logger.info(
            "workflow_started",
            account=state["account_name"],
            thread_id=config["configurable"]["thread_id"],
            run_id=run_id,
        )

        try:
            # Run workflow (synchronous)
            result = self.app.invoke(state, config)

            # Check if waiting for human
            if result.get("waiting_for_human"):
                logger.info(
                    "workflow_paused_for_human",
                    question=result.get("human_question", "")[:100] if result.get("human_question") else None
                )
            else:
                logger.info("workflow_completed", account=state["account_name"])

            # If LangSmith tracing is enabled, share the run and capture the public URL
            if (os.environ.get("LANGCHAIN_TRACING_V2") == "true" and
                    os.environ.get("LANGSMITH_API_KEY")):
                try:
                    from langsmith import Client as LangSmithClient
                    ls_client = LangSmithClient()
                    shared_url = ls_client.share_run(run_id)
                    self._langsmith_url = str(shared_url)
                    logger.info("langsmith_run_shared", url=self._langsmith_url, run_id=run_id)
                except Exception as ls_err:
                    logger.warning("langsmith_share_run_failed", error=str(ls_err), run_id=run_id)

            return result
        finally:
            # Clear callback and cancel event after run completes
            self._sse_callback = None
            self._cancel_event = None

    def resume(
        self,
        thread_id: str,
        human_input: str | None = None,
        sse_callback: "SSECallbackHandler | None" = None,
        cancel_event: threading.Event | None = None
    ) -> ResearchState:
        """
        Resume a workflow from checkpoint with optional human input.

        Args:
            thread_id: Thread ID to resume
            human_input: Optional human feedback/response
            sse_callback: Optional SSE callback handler for frontend updates
            cancel_event: Optional threading event to signal cancellation

        Returns:
            Updated research state
        """
        # Store SSE callback and cancel event for node functions to use
        self._sse_callback = sse_callback
        self._cancel_event = cancel_event

        config = {"configurable": {"thread_id": thread_id}}

        # Get current state
        current_state = self.app.get_state(config)

        # Validate checkpoint exists and has meaningful data
        if current_state is None or current_state.values is None:
            raise ValueError(f"No checkpoint found for thread_id: {thread_id}")

        state_values = current_state.values

        # Check for required fields to ensure this is a valid checkpoint
        # An empty or incomplete checkpoint should not be resumed
        if not state_values.get("account_name"):
            raise ValueError(
                f"No valid checkpoint found for thread_id: {thread_id}. "
                "The checkpoint exists but contains no research data."
            )

        # Add human input if provided - use update_state to modify checkpoint
        if human_input:
            feedback_list = list(state_values.get("human_feedback", []))
            feedback_list.append(human_input)

            # Update checkpoint state using LangGraph's update_state API
            # This modifies the checkpoint without restarting the workflow
            self.app.update_state(
                config,
                {
                    "human_feedback": feedback_list,
                    "waiting_for_human": False
                }
            )

            logger.info(
                "human_input_added",
                thread_id=thread_id,
                feedback=human_input[:100]
            )

        logger.info("workflow_resumed", thread_id=thread_id)

        try:
            # Resume execution from checkpoint by passing None
            # This continues from where the workflow was interrupted
            result = self.app.invoke(None, config)

            return result
        finally:
            # Clear callback and cancel event after run completes
            self._sse_callback = None
            self._cancel_event = None

    def get_state(self, thread_id: str) -> ResearchState | None:
        """
        Get current state for a thread.

        Args:
            thread_id: Thread ID to query

        Returns:
            Current state or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.app.get_state(config)

        if state_snapshot and state_snapshot.values:
            return state_snapshot.values

        return None


# Example usage:
"""
from src.graph.workflow import ResearchWorkflow
from src.models.state import create_initial_state, ResearchDepth

# Create workflow
workflow = ResearchWorkflow()

# Create initial state
state = create_initial_state(
    account_name="Boeing",
    industry="aerospace",
    region="North America",
    research_depth=ResearchDepth.STANDARD
)

# Run workflow - may pause for human input
result = workflow.run(state)

# Check if waiting for human
if result.get("waiting_for_human"):
    print(f"Question: {result.get('human_question')}")

    # Resume with human feedback
    result = workflow.resume(
        thread_id="research_Boeing",
        human_input="looks good, approved"
    )

# Or get current state
state = workflow.get_state("research_Boeing")
"""
