"""
Workflow service wrapping the ResearchWorkflow for API use.
Provides async methods for starting, monitoring, and controlling research workflows.
"""
import asyncio
import threading
from datetime import datetime
from typing import Any
import structlog

from src.graph.workflow import ResearchWorkflow
from src.graph.sse_callbacks import SSECallbackHandler
from src.models.state import (
    ResearchState,
    ResearchDepth,
    ResearchProgress,
    Signal,
    Opportunity,
    create_initial_state,
)
from api.schemas.api_models import (
    ResearchRequest,
    ResearchStatusEnum,
    ResearchStateResponse,
    SignalResponse,
    OpportunityResponse,
    ProgressResponse,
)

logger = structlog.get_logger(__name__)


class WorkflowService:
    """
    Service layer for managing research workflows.
    Wraps ResearchWorkflow for use with FastAPI.
    """

    def __init__(self):
        """Initialize the workflow service."""
        self._workflows: dict[str, ResearchWorkflow] = {}
        self._states: dict[str, ResearchState] = {}
        self._running: set[str] = set()
        self._cancel_events: dict[str, threading.Event] = {}  # Cancellation signals for workflows

    def _generate_thread_id(self, account_name: str) -> str:
        """Generate a unique thread ID for a research workflow."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = account_name.replace(" ", "_").replace("/", "_")
        return f"research_{safe_name}_{timestamp}"

    def _get_or_create_workflow(self, seller_name: str, thread_id: str) -> ResearchWorkflow:
        """Get an existing workflow or create a new one."""
        if thread_id not in self._workflows:
            self._workflows[thread_id] = ResearchWorkflow(seller_name=seller_name)
        return self._workflows[thread_id]

    def _map_research_depth(self, depth: str) -> ResearchDepth:
        """Map API depth enum to internal ResearchDepth."""
        mapping = {
            "quick": ResearchDepth.QUICK,
            "standard": ResearchDepth.STANDARD,
            "deep": ResearchDepth.DEEP,
        }
        return mapping.get(depth, ResearchDepth.STANDARD)

    def _determine_status(self, state: ResearchState, thread_id: str) -> ResearchStatusEnum:
        """Determine the current status of a workflow."""
        if thread_id in self._running:
            return ResearchStatusEnum.RUNNING
        # Check for stopped status BEFORE other checks
        if state.get("status") == "stopped":
            return ResearchStatusEnum.STOPPED
        if state.get("error_messages"):
            return ResearchStatusEnum.ERROR
        if state.get("waiting_for_human"):
            return ResearchStatusEnum.WAITING_FOR_HUMAN
        if state.get("progress", ResearchProgress()).is_complete():
            return ResearchStatusEnum.COMPLETED
        return ResearchStatusEnum.PENDING

    def _serialize_signal(self, signal: Signal | dict) -> SignalResponse:
        """Convert a Signal to API response format. Handles both Signal objects and dicts."""
        # Handle dict format (from checkpoint deserialization)
        if isinstance(signal, dict):
            return SignalResponse(
                source=signal.get("source", "unknown"),
                signal_type=signal.get("signal_type", "unknown"),
                content=signal.get("content", ""),
                confidence=signal.get("confidence", 0.5),
                timestamp=signal.get("timestamp"),
                metadata=signal.get("metadata", {}),
            )
        # Handle Signal object
        return SignalResponse(
            source=signal.source,
            signal_type=signal.signal_type,
            content=signal.content,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            metadata=signal.metadata,
        )

    def _serialize_opportunity(self, opp: Opportunity | dict) -> OpportunityResponse:
        """Convert an Opportunity to API response format. Handles both Opportunity objects and dicts."""
        # Handle dict format (from checkpoint deserialization)
        if isinstance(opp, dict):
            confidence = opp.get("confidence", "medium")
            if hasattr(confidence, 'value'):
                confidence = confidence.value
            return OpportunityResponse(
                product_name=opp.get("product_name", "Unknown"),
                rationale=opp.get("rationale", ""),
                target_persona=opp.get("target_persona"),
                talking_points=opp.get("talking_points", []),
                estimated_value=opp.get("estimated_value"),
                risks=opp.get("risks", []),
                confidence=str(confidence),
                confidence_score=opp.get("confidence_score", 0.5),
                evidence_count=len(opp.get("evidence", [])) if opp.get("evidence") else 0,
            )
        # Handle Opportunity object
        return OpportunityResponse(
            product_name=opp.product_name,
            rationale=opp.rationale,
            target_persona=opp.target_persona,
            talking_points=opp.talking_points,
            estimated_value=opp.estimated_value,
            risks=opp.risks,
            confidence=opp.confidence.value if hasattr(opp.confidence, 'value') else str(opp.confidence),
            confidence_score=opp.confidence_score,
            evidence_count=len(opp.evidence) if opp.evidence else 0,
        )

    def _serialize_progress(self, progress: ResearchProgress | dict) -> ProgressResponse:
        """Convert ResearchProgress to API response format. Handles both objects and dicts."""
        # Handle dict format (from checkpoint deserialization)
        if isinstance(progress, dict):
            completed = []
            if progress.get("coordinator_complete"):
                completed.append("coordinator")
            if progress.get("gatherer_complete"):
                completed.append("gatherer")
            if progress.get("identifier_complete"):
                completed.append("identifier")
            if progress.get("validator_complete"):
                completed.append("validator")
            return ProgressResponse(
                coordinator_complete=progress.get("coordinator_complete", False),
                gatherer_complete=progress.get("gatherer_complete", False),
                identifier_complete=progress.get("identifier_complete", False),
                validator_complete=progress.get("validator_complete", False),
                completed_agents=completed,
            )
        # Handle ResearchProgress object
        return ProgressResponse(
            coordinator_complete=progress.coordinator_complete,
            gatherer_complete=progress.gatherer_complete,
            identifier_complete=progress.identifier_complete,
            validator_complete=progress.validator_complete,
            completed_agents=progress.get_completed_agents(),
        )

    def serialize_state(self, state: ResearchState, thread_id: str) -> ResearchStateResponse:
        """Convert ResearchState to API response format."""
        progress = state.get("progress", ResearchProgress())

        # Serialize signals
        signals = [
            self._serialize_signal(s) for s in state.get("signals", [])
        ]

        # Serialize opportunities
        opportunities = [
            self._serialize_opportunity(o) for o in state.get("opportunities", [])
        ]
        validated_opportunities = [
            self._serialize_opportunity(o) for o in state.get("validated_opportunities", [])
        ]

        return ResearchStateResponse(
            thread_id=thread_id,
            status=self._determine_status(state, thread_id),
            account_name=state.get("account_name", ""),
            industry=state.get("industry", ""),
            seller_name=state.get("seller_name", ""),
            region=state.get("region"),
            user_context=state.get("user_context"),
            research_depth=state.get("research_depth", ResearchDepth.STANDARD).value,
            progress=self._serialize_progress(progress),
            signals=signals,
            opportunities=opportunities,
            validated_opportunities=validated_opportunities,
            competitive_risks=state.get("competitive_risks", []),
            tech_stack=state.get("tech_stack", []),
            waiting_for_human=state.get("waiting_for_human", False),
            human_question=state.get("human_question"),
            current_report=state.get("current_report"),
            workflow_iteration=state.get("workflow_iteration", 1),
            started_at=state.get("started_at"),
            last_updated=state.get("last_updated"),
            error_messages=state.get("error_messages", []),
        )

    async def start_research(self, request: ResearchRequest) -> tuple[str, ResearchState]:
        """
        Start a new research workflow.

        Args:
            request: Research request parameters

        Returns:
            Tuple of (thread_id, initial_state)
        """
        thread_id = self._generate_thread_id(request.account_name)

        logger.info(
            "starting_research",
            thread_id=thread_id,
            account=request.account_name,
            industry=request.industry,
            seller=request.seller_name,
        )

        # Create initial state
        state = create_initial_state(
            account_name=request.account_name,
            industry=request.industry,
            seller_name=request.seller_name,
            region=request.region,
            user_context=request.user_context,
            research_depth=self._map_research_depth(request.research_depth.value),
        )

        # Store state
        self._states[thread_id] = state

        # Create workflow
        self._get_or_create_workflow(request.seller_name, thread_id)

        return thread_id, state

    async def run_workflow(self, thread_id: str) -> ResearchState:
        """
        Run the workflow for a given thread.
        This is a blocking operation that runs the full workflow.

        Args:
            thread_id: The thread ID to run

        Returns:
            Final state after workflow completion
        """
        if thread_id not in self._states:
            raise ValueError(f"Thread {thread_id} not found")

        state = self._states[thread_id]
        seller_name = state["seller_name"]
        workflow = self._get_or_create_workflow(seller_name, thread_id)

        # Create cancellation event for this workflow
        cancel_event = threading.Event()
        self._cancel_events[thread_id] = cancel_event

        self._running.add(thread_id)

        # PRE-CREATE SSE QUEUE before workflow starts
        # This ensures events emitted before frontend connects are buffered
        from api.sse.event_stream import event_emitter
        event_emitter.create_queue(thread_id)
        logger.info("sse_queue_pre_created", thread_id=thread_id)

        # Create SSE callback handler for real-time frontend updates
        sse_callback = SSECallbackHandler(thread_id)

        try:
            # Run workflow in thread pool with SSE callback and cancel event
            result = await asyncio.to_thread(
                workflow.run, state, thread_id, sse_callback, cancel_event
            )
            self._states[thread_id] = result
            return result
        finally:
            self._running.discard(thread_id)
            self._cancel_events.pop(thread_id, None)  # Clean up cancel event

    async def get_state(self, thread_id: str) -> ResearchState | None:
        """
        Get the current state of a workflow.

        Args:
            thread_id: The thread ID

        Returns:
            Current state or None if not found
        """
        # First check our local cache
        if thread_id in self._states:
            return self._states[thread_id]

        # Try to get from workflow checkpoint
        for seller_name in ["MathWorks"]:  # Default seller, could be dynamic
            workflow = self._get_or_create_workflow(seller_name, thread_id)
            try:
                state = workflow.get_state(thread_id)
                if state:
                    self._states[thread_id] = state
                    return state
            except Exception as e:
                logger.warning("failed_to_get_state", thread_id=thread_id, error=str(e))

        return None

    async def submit_feedback(self, thread_id: str, feedback: str) -> ResearchState:
        """
        Submit human feedback and resume the workflow.

        Args:
            thread_id: The thread ID
            feedback: Human feedback text

        Returns:
            Updated state after processing feedback
        """
        state = await self.get_state(thread_id)
        if not state:
            raise ValueError(f"Thread {thread_id} not found")

        # Clear stopped status when resuming
        if state.get("status") == "stopped":
            state["status"] = "running"

        seller_name = state["seller_name"]
        workflow = self._get_or_create_workflow(seller_name, thread_id)

        # Create cancellation event for resumed workflow
        cancel_event = threading.Event()
        self._cancel_events[thread_id] = cancel_event

        self._running.add(thread_id)

        # PRE-CREATE SSE QUEUE before workflow resumes
        # This ensures events emitted before frontend reconnects are buffered
        from api.sse.event_stream import event_emitter
        event_emitter.create_queue(thread_id)
        logger.info("sse_queue_pre_created_for_feedback", thread_id=thread_id)

        # Create SSE callback handler for real-time frontend updates
        sse_callback = SSECallbackHandler(thread_id)

        try:
            # Resume workflow with feedback, SSE callback, and cancel event
            result = await asyncio.to_thread(
                workflow.resume, thread_id, feedback, sse_callback, cancel_event
            )
            self._states[thread_id] = result
            return result
        finally:
            self._running.discard(thread_id)
            self._cancel_events.pop(thread_id, None)  # Clean up cancel event

    async def list_threads(self) -> list[tuple[str, ResearchState]]:
        """
        List all known research threads.

        Returns:
            List of (thread_id, state) tuples
        """
        return [(tid, state) for tid, state in self._states.items()]

    def is_running(self, thread_id: str) -> bool:
        """Check if a workflow is currently running."""
        return thread_id in self._running

    async def stop_research(self, thread_id: str) -> bool:
        """
        Stop a running research workflow with proper cancellation.
        Preserves state so research can be resumed later from Previous Sessions.

        Args:
            thread_id: The thread ID to stop

        Returns:
            True if stopped, False if not running
        """
        if thread_id not in self._running:
            return False

        # Signal cancellation to the workflow thread
        cancel_event = self._cancel_events.get(thread_id)
        if cancel_event:
            logger.info("signaling_workflow_cancellation", thread_id=thread_id)
            cancel_event.set()  # Signal the workflow to stop

        # Remove from running set
        self._running.discard(thread_id)

        # Update state to indicate stopped - PRESERVE for later resumption
        if thread_id in self._states:
            self._states[thread_id]["status"] = "stopped"
            self._states[thread_id]["waiting_for_human"] = True  # Allows resume via feedback
            self._states[thread_id]["human_question"] = "Research was paused. Click Resume to continue."
            self._states[thread_id]["last_updated"] = datetime.now().isoformat()

        logger.info("research_stopped_preserving_state", thread_id=thread_id)
        return True

    async def discard_research(self, thread_id: str) -> bool:
        """
        Stop a running research workflow and DISCARD all state.
        Unlike stop_research(), this permanently removes the research.
        It will NOT appear in Previous Sessions and cannot be resumed.

        Args:
            thread_id: The thread ID to discard

        Returns:
            True if discarded, False if not found
        """
        # Signal cancellation if running
        cancel_event = self._cancel_events.get(thread_id)
        if cancel_event:
            logger.info("signaling_workflow_cancellation_for_discard", thread_id=thread_id)
            cancel_event.set()

        # Remove from running set
        self._running.discard(thread_id)

        # DISCARD state completely (unlike stop which preserves)
        if thread_id in self._states:
            del self._states[thread_id]

        # Clean up workflow instance
        if thread_id in self._workflows:
            del self._workflows[thread_id]

        # Clean up cancel event
        self._cancel_events.pop(thread_id, None)

        logger.info("research_discarded", thread_id=thread_id)
        return True


# Singleton instance for dependency injection
workflow_service = WorkflowService()
