"""Structured logging tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from alignforge.core.logging import get_logger, request_id_var, run_id_var, setup_logging


def test_json_log_format(tmp_path: Path) -> None:
    """JSON format produces parseable JSON lines."""
    setup_logging(level="DEBUG", fmt="json", log_dir=tmp_path)
    log = get_logger(component="test")
    log.info("test_event", key="value")

    log_file = tmp_path / "alignforge.log"
    assert log_file.exists()
    lines = [line for line in log_file.read_text().splitlines() if line.strip()]
    assert len(lines) >= 1
    parsed = json.loads(lines[-1])
    assert parsed["event"] == "test_event"
    assert parsed["key"] == "value"

    # Cleanup handlers to avoid interference with other tests.
    logging.getLogger().handlers.clear()


def test_context_var_propagation(tmp_path: Path) -> None:
    """ContextVar values appear in log output."""
    setup_logging(level="DEBUG", fmt="json", log_dir=tmp_path)
    token_req = request_id_var.set("req-abc123")
    token_run = run_id_var.set("sft-test")
    try:
        log = get_logger()
        log.info("with_context")
        log_file = tmp_path / "alignforge.log"
        lines = log_file.read_text().splitlines()
        last = json.loads(lines[-1])
        assert last.get("request_id") == "req-abc123"
        assert last.get("run_id") == "sft-test"
    finally:
        request_id_var.reset(token_req)
        run_id_var.reset(token_run)
        logging.getLogger().handlers.clear()
