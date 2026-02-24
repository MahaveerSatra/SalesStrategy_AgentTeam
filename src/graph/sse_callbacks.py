"""
LangGraph callback handler that emits SSE events for workflow progress.

This module provides a callback handler that integrates with LangGraph's
callback system to emit real-time SSE events to the frontend as workflow
nodes execute.
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
from datetime import datetime
from uuid import UUID
import structlog

logger = structlog.get_logger(__name__)


# Human-readable labels for agent nodes (used in trace panel)
NODE_LABELS = {
    "coordinator_entry": "Coordinator",
    "gatherer": "Gatherer",
    "identifier": "Identifier",
    "validator": "Validator",
    "coordinator_exit": "Report Coordinator",
}

# Module-level trace store: thread_id → {node_id → trace dict}
_node_traces: dict[str, dict[str, dict]] = {}


def get_node_traces(thread_id: str) -> dict[str, dict]:
    """Get all node traces for a thread (returns empty dict if none)."""
    return _node_traces.get(thread_id, {})


def clear_node_traces(thread_id: str) -> None:
    """Clear all node traces for a thread (call on discard to free memory)."""
    _node_traces.pop(thread_id, None)


# Node name mapping: LangGraph internal names → Frontend node IDs
NODE_MAP = {
    "coordinator_entry": "coordinator_entry",
    "_coordinator_entry_node": "coordinator_entry",
    "gatherer": "gatherer",
    "_gatherer_node": "gatherer",
    "identifier": "identifier",
    "_identifier_node": "identifier",
    "validator": "validator",
    "_validator_node": "validator",
    "coordinator_exit": "coordinator_exit",
    "_coordinator_exit_node": "coordinator_exit",
    "_wait_for_human_node": "human_feedback",
    "coordinator_feedback": "coordinator_entry",
    "_coordinator_feedback_node": "coordinator_entry",
}


class SSECallbackHandler:
    """
    Callback handler that emits SSE events during workflow execution.

    This integrates with LangGraph's callback system to capture
    node start/end events and emit them to the frontend via SSE.

    Note: This is a simplified callback handler that doesn't inherit from
    LangChain's BaseCallbackHandler to avoid import complexity. Instead,
    it provides direct methods to emit SSE events from workflow nodes.
    """

    def __init__(self, thread_id: str):
        """
        Initialize SSE callback handler.

        Args:
            thread_id: The thread ID for SSE event routing
        """
        self.thread_id = thread_id
        self._current_node: Optional[str] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for async operations."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._event_loop is None:
                self._event_loop = asyncio.new_event_loop()
            return self._event_loop

    def _emit_sync(self, coro) -> None:
        """
        Run async emit in sync context.

        Args:
            coro: Coroutine to execute
        """
        try:
            loop = self._get_event_loop()
            if loop.is_running():
                # We're in an async context, create task
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except Exception as e:
            # Don't let SSE errors break the workflow
            logger.warning("sse_emit_failed", error=str(e), thread_id=self.thread_id)

    async def _emit_node_started(self, node: str, description: str) -> None:
        """Emit node started event."""
        try:
            from api.sse.event_stream import emit_node_started
            await emit_node_started(self.thread_id, node, description)
            logger.debug("sse_node_started_emitted", node=node, thread_id=self.thread_id)
        except Exception as e:
            logger.warning("sse_emit_node_started_failed", error=str(e), node=node)

    async def _emit_node_completed(self, node: str, metrics: Dict[str, Any] = None) -> None:
        """Emit node completed event."""
        try:
            from api.sse.event_stream import emit_node_completed
            await emit_node_completed(self.thread_id, node, metrics)
            logger.debug("sse_node_completed_emitted", node=node, thread_id=self.thread_id)
        except Exception as e:
            logger.warning("sse_emit_node_completed_failed", error=str(e), node=node)

    async def _emit_waiting_for_human(self, question: str) -> None:
        """Emit waiting for human event."""
        try:
            from api.sse.event_stream import emit_waiting_for_human
            await emit_waiting_for_human(self.thread_id, question)
            logger.debug("sse_waiting_for_human_emitted", thread_id=self.thread_id)
        except Exception as e:
            logger.warning("sse_emit_waiting_for_human_failed", error=str(e))

    def on_node_start(self, node_name: str, inputs: Dict[str, Any]) -> None:
        """
        Called when a workflow node starts.

        Args:
            node_name: Name of the node starting
            inputs: Input state to the node
        """
        mapped_node = NODE_MAP.get(node_name)
        if not mapped_node:
            return

        self._current_node = mapped_node
        account = inputs.get("account_name", "company") if isinstance(inputs, dict) else "company"
        description = self._get_description(mapped_node, account)

        # Record trace start for observable nodes
        if mapped_node in NODE_LABELS:
            _node_traces.setdefault(self.thread_id, {})[mapped_node] = {
                "node_id": mapped_node,
                "node_label": NODE_LABELS[mapped_node],
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration_ms": None,
                "summary": {},
            }

        logger.info(
            "sse_callback_node_start",
            node=mapped_node,
            description=description,
            thread_id=self.thread_id
        )

        self._emit_sync(self._emit_node_started(mapped_node, description))

    def on_node_end(self, node_name: str, outputs: Dict[str, Any]) -> None:
        """
        Called when a workflow node completes.

        Args:
            node_name: Name of the node completing
            outputs: Output state from the node
        """
        mapped_node = NODE_MAP.get(node_name)
        if not mapped_node:
            return

        # Check if waiting for human
        if isinstance(outputs, dict) and outputs.get("waiting_for_human"):
            question = outputs.get("human_question", "Please review the results")
            logger.info(
                "sse_callback_waiting_human",
                question=question[:50] if question else None,
                thread_id=self.thread_id
            )
            self._emit_sync(self._emit_waiting_for_human(question))
        else:
            metrics = self._extract_metrics(mapped_node, outputs)
            logger.info(
                "sse_callback_node_end",
                node=mapped_node,
                metrics=metrics,
                thread_id=self.thread_id
            )
            self._emit_sync(self._emit_node_completed(mapped_node, metrics))

        # Update trace with completion data
        if mapped_node in NODE_LABELS:
            existing = _node_traces.get(self.thread_id, {}).get(mapped_node, {})
            end_time = datetime.now()
            start_time_str = existing.get("start_time")
            duration_ms = None
            if start_time_str:
                try:
                    start_dt = datetime.fromisoformat(start_time_str)
                    duration_ms = int((end_time - start_dt).total_seconds() * 1000)
                except (ValueError, TypeError):
                    pass
            _node_traces.setdefault(self.thread_id, {})[mapped_node].update({
                "status": "complete",
                "end_time": end_time.isoformat(),
                "duration_ms": duration_ms,
                "summary": self._extract_summary(mapped_node, outputs),
            })

        self._current_node = None

    def _get_description(self, node: str, account: str) -> str:
        """
        Get human-readable description for node.

        Args:
            node: Frontend node ID
            account: Account name being researched

        Returns:
            Description string
        """
        descriptions = {
            "coordinator_entry": f"Validating inputs for {account}",
            "gatherer": f"Gathering signals for {account}",
            "identifier": f"Identifying opportunities for {account}",
            "validator": f"Validating findings for {account}",
            "coordinator_exit": f"Generating report for {account}",
            "human_feedback": "Awaiting your feedback",
        }
        return descriptions.get(node, f"Processing {node}")

    def _extract_metrics(self, node: str, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metrics from node outputs.

        Args:
            node: Frontend node ID
            outputs: Output state from node

        Returns:
            Metrics dictionary
        """
        if not isinstance(outputs, dict):
            return {}

        metrics = {}
        if node == "gatherer":
            metrics["signals_count"] = len(outputs.get("signals", []))
        elif node == "identifier":
            metrics["opportunities_count"] = len(outputs.get("opportunities", []))
        elif node == "validator":
            metrics["validated_count"] = len(outputs.get("validated_opportunities", []))
            metrics["risks_count"] = len(outputs.get("competitive_risks", []))
        return metrics

    def _extract_summary(self, node: str, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a rich state summary from node outputs for the trace panel.

        Returns a node-specific dict with counts, previews, and key data points.
        """
        if not isinstance(outputs, dict):
            return {}

        summary: Dict[str, Any] = {}

        if node == "coordinator_entry":
            summary["account_name"] = outputs.get("account_name", "")
            summary["industry"] = outputs.get("industry", "")
            summary["company_domain"] = outputs.get("company_domain")
            user_context = outputs.get("user_context")
            if user_context:
                summary["user_context"] = user_context[:200]

        elif node == "gatherer":
            signals = outputs.get("signals", [])
            summary["signals_count"] = len(signals)
            summary["job_postings_count"] = len(outputs.get("job_postings", []))
            summary["news_items_count"] = len(outputs.get("news_items", []))
            previews = []
            for s in signals[:3]:
                if isinstance(s, dict):
                    meta = s.get("metadata", {}) or {}
                    url = meta.get("url", "") or meta.get("source_url", "")
                    content = (s.get("content", "") or "")[:100]
                    previews.append({
                        "source": content,
                        "signal_type": url,
                        "confidence": round(float(s.get("confidence", 0.5)), 2),
                    })
                elif hasattr(s, "content"):
                    meta = getattr(s, "metadata", {}) or {}
                    url = meta.get("url", "") or meta.get("source_url", "")
                    content = (getattr(s, "content", "") or "")[:100]
                    previews.append({
                        "source": content,
                        "signal_type": url,
                        "confidence": round(float(s.confidence), 2),
                    })
            summary["signal_previews"] = previews

        elif node == "identifier":
            opportunities = outputs.get("opportunities", [])
            summary["opportunities_count"] = len(opportunities)
            previews = []
            for o in opportunities[:3]:
                if isinstance(o, dict):
                    previews.append({
                        "product_name": o.get("product_name", ""),
                        "confidence_score": round(float(o.get("confidence_score", 0.5)), 2),
                    })
                elif hasattr(o, "product_name"):
                    previews.append({
                        "product_name": o.product_name,
                        "confidence_score": round(float(o.confidence_score), 2),
                    })
            summary["opportunity_previews"] = previews

        elif node == "validator":
            validated = outputs.get("validated_opportunities", [])
            risks = outputs.get("competitive_risks", [])
            summary["validated_count"] = len(validated)
            summary["risks_count"] = len(risks)
            summary["tech_stack"] = outputs.get("tech_stack", [])[:10]
            summary["risk_previews"] = [str(r) for r in risks[:3]]

        elif node == "coordinator_exit":
            report = outputs.get("current_report")
            if report:
                summary["report_length"] = len(report)
                summary["report_preview"] = report[:300]
            summary["workflow_iteration"] = outputs.get("workflow_iteration", 1)

        return summary
