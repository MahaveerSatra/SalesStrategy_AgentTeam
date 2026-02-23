"""
SSE (Server-Sent Events) streaming for real-time workflow updates.
Streams workflow progress to the frontend as agents complete their work.
"""
import asyncio
import json
from collections import deque
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Any
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)


class WorkflowEventType(str, Enum):
    """Types of events streamed to the frontend."""
    WORKFLOW_STARTED = "workflow_started"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    STATE_UPDATE = "state_update"
    SIGNAL_FOUND = "signal_found"
    OPPORTUNITY_FOUND = "opportunity_found"
    WAITING_FOR_HUMAN = "waiting_human"
    FEEDBACK_RECEIVED = "feedback_received"
    WORKFLOW_COMPLETE = "complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class WorkflowEvent(BaseModel):
    """A single workflow event for SSE streaming."""
    event: WorkflowEventType
    data: dict[str, Any]
    timestamp: datetime = None

    def __init__(self, **data):
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.now()
        super().__init__(**data)

    def to_sse(self) -> str:
        """Format as SSE message."""
        event_data = {
            "event": self.event.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        return f"data: {json.dumps(event_data)}\n\n"


class EventEmitter:
    """
    Event emitter for workflow progress.
    Used to push events to SSE streams.

    Includes event buffering to capture events emitted before
    the client connects to the SSE stream.
    """

    def __init__(self, buffer_size: int = 50):
        self._queues: dict[str, asyncio.Queue] = {}
        self._active: set[str] = set()
        self._event_buffer: dict[str, deque] = {}  # Buffer for early events
        self.buffer_size = buffer_size

    def create_queue(self, thread_id: str) -> asyncio.Queue:
        """Create a new event queue for a thread."""
        if thread_id not in self._queues:
            self._queues[thread_id] = asyncio.Queue()
        if thread_id not in self._event_buffer:
            self._event_buffer[thread_id] = deque(maxlen=self.buffer_size)
        self._active.add(thread_id)
        return self._queues[thread_id]

    def get_queue(self, thread_id: str) -> asyncio.Queue | None:
        """Get the event queue for a thread."""
        return self._queues.get(thread_id)

    async def emit(self, thread_id: str, event: WorkflowEvent):
        """
        Emit an event to all listeners of a thread.

        If no queue exists yet (client hasn't connected), the event
        is buffered and will be sent when the client connects.
        """
        # Always buffer the event (for late-connecting clients)
        if thread_id not in self._event_buffer:
            self._event_buffer[thread_id] = deque(maxlen=self.buffer_size)
        self._event_buffer[thread_id].append(event)

        # Also queue if listener exists
        queue = self._queues.get(thread_id)
        if queue:
            await queue.put(event)
        else:
            logger.debug("sse_event_buffered", thread_id=thread_id, event=event.event.value)

    def get_buffered_events(self, thread_id: str) -> list[WorkflowEvent]:
        """Get all buffered events for a thread."""
        return list(self._event_buffer.get(thread_id, []))

    def clear_buffer(self, thread_id: str):
        """Clear the event buffer for a thread."""
        if thread_id in self._event_buffer:
            self._event_buffer[thread_id].clear()

    def close(self, thread_id: str):
        """Close the event stream for a thread."""
        self._active.discard(thread_id)
        if thread_id in self._queues:
            # Put a sentinel value to signal stream end
            try:
                self._queues[thread_id].put_nowait(None)
            except asyncio.QueueFull:
                pass
        # Clean up buffer
        if thread_id in self._event_buffer:
            del self._event_buffer[thread_id]

    def is_active(self, thread_id: str) -> bool:
        """Check if a thread has active listeners."""
        return thread_id in self._active


# Global event emitter instance
event_emitter = EventEmitter()


async def create_event_generator(
    thread_id: str,
    timeout_seconds: float = 300.0,
) -> AsyncGenerator[str, None]:
    """
    Create an async generator that yields SSE events for a workflow.

    Args:
        thread_id: The thread ID to stream events for
        timeout_seconds: Maximum time to keep the stream open

    Yields:
        SSE-formatted event strings
    """
    queue = event_emitter.create_queue(thread_id)
    start_time = asyncio.get_event_loop().time()

    logger.info("sse_stream_started", thread_id=thread_id)

    try:
        # SEND BUFFERED EVENTS FIRST (events emitted before client connected)
        buffered_events = event_emitter.get_buffered_events(thread_id)
        if buffered_events:
            logger.info("sse_sending_buffered_events",
                       thread_id=thread_id, count=len(buffered_events))
            for event in buffered_events:
                yield event.to_sse()

        # Clear buffer after sending (avoid duplicates on reconnect)
        event_emitter.clear_buffer(thread_id)

        while True:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                logger.info("sse_stream_timeout", thread_id=thread_id)
                break

            try:
                # Wait for event with timeout for heartbeat
                event = await asyncio.wait_for(queue.get(), timeout=15.0)

                if event is None:
                    # Stream closed
                    logger.info("sse_stream_closed", thread_id=thread_id)
                    break

                yield event.to_sse()

                # If workflow complete or error, close stream
                if event.event in (WorkflowEventType.WORKFLOW_COMPLETE, WorkflowEventType.ERROR):
                    break

            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                heartbeat = WorkflowEvent(
                    event=WorkflowEventType.HEARTBEAT,
                    data={"thread_id": thread_id},
                )
                yield heartbeat.to_sse()

    finally:
        event_emitter.close(thread_id)
        logger.info("sse_stream_ended", thread_id=thread_id)


# Helper functions for emitting common events
async def emit_node_started(thread_id: str, node: str, description: str):
    """Emit a node started event."""
    await event_emitter.emit(
        thread_id,
        WorkflowEvent(
            event=WorkflowEventType.NODE_STARTED,
            data={
                "node": node,
                "description": description,
                "status": "running",
            },
        ),
    )


async def emit_node_completed(thread_id: str, node: str, metrics: dict | None = None):
    """Emit a node completed event."""
    await event_emitter.emit(
        thread_id,
        WorkflowEvent(
            event=WorkflowEventType.NODE_COMPLETED,
            data={
                "node": node,
                "status": "complete",
                "metrics": metrics or {},
            },
        ),
    )


async def emit_state_update(thread_id: str, state_data: dict):
    """Emit a state update event."""
    await event_emitter.emit(
        thread_id,
        WorkflowEvent(
            event=WorkflowEventType.STATE_UPDATE,
            data=state_data,
        ),
    )


async def emit_waiting_for_human(thread_id: str, question: str):
    """Emit a waiting for human event."""
    await event_emitter.emit(
        thread_id,
        WorkflowEvent(
            event=WorkflowEventType.WAITING_FOR_HUMAN,
            data={
                "question": question,
                "status": "waiting",
            },
        ),
    )


async def emit_workflow_complete(thread_id: str, summary: dict):
    """Emit a workflow complete event."""
    await event_emitter.emit(
        thread_id,
        WorkflowEvent(
            event=WorkflowEventType.WORKFLOW_COMPLETE,
            data=summary,
        ),
    )


async def emit_error(thread_id: str, error: str):
    """Emit an error event."""
    await event_emitter.emit(
        thread_id,
        WorkflowEvent(
            event=WorkflowEventType.ERROR,
            data={"error": error},
        ),
    )
