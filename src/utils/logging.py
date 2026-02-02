"""
Structured logging configuration.
Staff-level: JSON logs for production, readable logs for development.
"""
import os
import sys
import warnings
import logging
import structlog
from typing import Any

from ..config import settings


def _suppress_noisy_libraries() -> None:
    """
    Suppress verbose output from third-party libraries.

    This prevents noise in the CLI output from:
    - sentence-transformers (tqdm progress bars, model loading warnings)
    - transformers (model loading warnings like 'position_ids UNEXPECTED')
    - litellm (verbose INFO logging)
    - chromadb (telemetry and internal logging)
    - pydantic (serialization warnings)
    - huggingface_hub (authentication warnings)
    """
    # Suppress tqdm progress bars from sentence-transformers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Suppress transformers logging (position_ids warnings, etc.)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Suppress sentence-transformers logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Suppress HuggingFace Hub logging (authentication warnings)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*unauthenticated.*")

    # Suppress LiteLLM verbose logging
    os.environ["LITELLM_LOG"] = "ERROR"
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    # Suppress httpx logging (used by LiteLLM)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Suppress chromadb logging
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)

    # Suppress pydantic serialization warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    # Suppress urllib3 warnings
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def setup_logging() -> None:
    """
    Configure structured logging based on settings.

    JSON format for production (machine-readable)
    Console format for development (human-readable)
    """
    # First, suppress noisy third-party libraries
    _suppress_noisy_libraries()

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        # Production: JSON logs
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: Pretty console logs
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set log level for root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level)
    )


def get_logger(name: str) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Initialize on import
setup_logging()