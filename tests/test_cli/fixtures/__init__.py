"""Test fixtures for CLI tests."""
from .sample_states import (
    create_minimal_state,
    create_complete_state,
    create_paused_state,
    create_empty_opportunities_state,
    create_partial_progress_state,
    create_state_with_risks
)

__all__ = [
    "create_minimal_state",
    "create_complete_state",
    "create_paused_state",
    "create_empty_opportunities_state",
    "create_partial_progress_state",
    "create_state_with_risks"
]
