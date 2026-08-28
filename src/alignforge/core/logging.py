"""Structured logging with request-id propagation (connector C8)."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog

# ── Context variables: bound once per request, visible in all log lines ──
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)


def _add_context(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject ContextVar values into every log event automatically."""
    rid = request_id_var.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    run = run_id_var.get()
    if run is not None:
        event_dict.setdefault("run_id", run)
    return event_dict


def setup_logging(
    level: str = "INFO",
    fmt: str = "console",
    log_dir: Path | None = None,
) -> structlog.stdlib.BoundLogger:
    """Configure structlog once at process start. Returns the root logger."""

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Console handler.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler — JSON lines, always, regardless of console format.
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        fh = logging.FileHandler(log_dir / "alignforge.log")
        fh.setFormatter(json_formatter)
        root.addHandler(fh)

    return structlog.get_logger()  # type: ignore[no-any-return]


def get_logger(**initial_binds: Any) -> structlog.stdlib.BoundLogger:
    """Get a logger with optional initial key bindings."""
    return structlog.get_logger(**initial_binds)  # type: ignore[no-any-return]
