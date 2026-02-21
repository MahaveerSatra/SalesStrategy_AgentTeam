"""
Research workflow API endpoints.
Provides REST API for starting, monitoring, and controlling research workflows.
"""
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import structlog

from api.schemas.api_models import (
    ResearchRequest,
    ResearchResponse,
    ResearchStateResponse,
    FeedbackRequest,
    FeedbackResponse,
    ThreadListResponse,
    ThreadSummary,
    ResearchStatusEnum,
)
from api.services.workflow_service import workflow_service
from api.sse.event_stream import create_event_generator, event_emitter, WorkflowEvent, WorkflowEventType

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/start", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    """
    Start a new research workflow.

    This endpoint initiates a research workflow and returns immediately.
    The workflow runs in the background. Use the /stream endpoint to
    receive real-time updates, or poll /state for current status.
    """
    try:
        thread_id, state = await workflow_service.start_research(request)

        # Run workflow in background
        background_tasks.add_task(workflow_service.run_workflow, thread_id)

        logger.info(
            "research_started",
            thread_id=thread_id,
            account=request.account_name,
        )

        return ResearchResponse(
            thread_id=thread_id,
            status=ResearchStatusEnum.RUNNING,
            message=f"Research started for {request.account_name}",
        )

    except Exception as e:
        logger.error("start_research_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{thread_id}/stream")
async def stream_research(thread_id: str) -> StreamingResponse:
    """
    Stream real-time workflow updates via Server-Sent Events (SSE).

    Connect to this endpoint to receive live updates as agents work:
    - node_started: An agent has started processing
    - node_completed: An agent has finished
    - state_update: New signals/opportunities found
    - waiting_human: Workflow paused for feedback
    - complete: Workflow finished

    Example usage (JavaScript):
    ```javascript
    const eventSource = new EventSource('/api/research/{thread_id}/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.event, data.data);
    };
    ```
    """
    # Verify thread exists
    state = await workflow_service.get_state(thread_id)
    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Research thread {thread_id} not found"
        )

    # Create event generator
    event_generator = create_event_generator(thread_id)

    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/{thread_id}/state", response_model=ResearchStateResponse)
async def get_research_state(thread_id: str) -> ResearchStateResponse:
    """
    Get the current state of a research workflow.

    Returns all collected signals, opportunities, risks, and progress.
    """
    state = await workflow_service.get_state(thread_id)

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Research thread {thread_id} not found"
        )

    return workflow_service.serialize_state(state, thread_id)


@router.post("/{thread_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    thread_id: str,
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> FeedbackResponse:
    """
    Submit human feedback on research results.

    Feedback options:
    - "approved" or "looks good": Complete the workflow
    - "dig deeper on X": Re-run gatherer with focus on X
    - "find other products": Re-run identifier
    - "confidence seems off": Re-run validator

    The workflow will resume in the background.
    """
    state = await workflow_service.get_state(thread_id)

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Research thread {thread_id} not found"
        )

    if not state.get("waiting_for_human"):
        raise HTTPException(
            status_code=400,
            detail="Workflow is not waiting for feedback"
        )

    try:
        # Submit feedback and resume in background
        background_tasks.add_task(
            workflow_service.submit_feedback,
            thread_id,
            request.feedback,
        )

        logger.info(
            "feedback_submitted",
            thread_id=thread_id,
            feedback=request.feedback[:50],
        )

        return FeedbackResponse(
            status=ResearchStatusEnum.RUNNING,
            next_route=None,  # Will be determined by coordinator
            message="Feedback received, workflow resuming",
        )

    except Exception as e:
        logger.error("submit_feedback_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{thread_id}/report")
async def get_report(thread_id: str) -> dict:
    """
    Get the formatted research report.

    Returns the markdown report generated by the coordinator.
    """
    state = await workflow_service.get_state(thread_id)

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Research thread {thread_id} not found"
        )

    report = state.get("current_report")

    if not report:
        raise HTTPException(
            status_code=400,
            detail="Report not yet generated"
        )

    return {
        "thread_id": thread_id,
        "account_name": state.get("account_name"),
        "report": report,
        "opportunities_count": len(state.get("validated_opportunities", [])),
        "risks_count": len(state.get("competitive_risks", [])),
    }


@router.get("/list", response_model=ThreadListResponse)
async def list_threads() -> ThreadListResponse:
    """
    List all research threads.

    Returns summary information for all known research workflows.
    """
    threads = await workflow_service.list_threads()

    summaries = []
    for thread_id, state in threads:
        from api.services.workflow_service import workflow_service as ws
        progress = ws._serialize_progress(state.get("progress", {}))

        summaries.append(
            ThreadSummary(
                thread_id=thread_id,
                account_name=state.get("account_name", ""),
                industry=state.get("industry", ""),
                status=ws._determine_status(state, thread_id),
                started_at=state.get("started_at"),
                progress=progress,
            )
        )

    return ThreadListResponse(
        threads=summaries,
        total=len(summaries),
    )
