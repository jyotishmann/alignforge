"""SQLite run registry — the spine of the system (connector C3).

Connects offline training to online serving. Every run, artifact, dataset,
and served model is a row here. The API reads it; training writes it.
They never import each other.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from alignforge.core.errors import RegistryError

log = structlog.get_logger()

# ── Schema ──────────────────────────────────────────────────────────────

SCHEMA_VERSION = 1

DDL = """\
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,              -- 'sft', 'dpo', 'eval', 'export'
    status        TEXT NOT NULL DEFAULT 'running',  -- 'running','completed','failed'
    config_hash   TEXT NOT NULL,
    dataset_hash  TEXT,
    git_sha       TEXT,
    hardware_json TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    metrics_json  TEXT,                       -- final metrics snapshot
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id  TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    kind         TEXT NOT NULL,               -- 'lora_adapter','merged_model','gguf','ollama_tag'
    path         TEXT NOT NULL,
    sha256       TEXT,
    size_bytes   INTEGER,
    created_at   TEXT NOT NULL,
    meta_json    TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id       TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    kind             TEXT NOT NULL,           -- 'sft', 'preference'
    split            TEXT,
    path             TEXT NOT NULL,
    sha256           TEXT NOT NULL,
    n_rows           INTEGER NOT NULL,
    source_json      TEXT,                    -- [{hf_id, revision, split, n_raw}]
    filter_stats_json TEXT,                   -- the funnel
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS served_models (
    model_id     TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    backend      TEXT NOT NULL,               -- 'transformers','ollama','echo'
    weights_ref  TEXT NOT NULL,               -- path, tag, or 'echo'
    run_id       TEXT REFERENCES runs(run_id),
    enabled      INTEGER NOT NULL DEFAULT 1,
    sort_order   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS human_votes (
    vote_id     TEXT PRIMARY KEY,
    session_id  TEXT,
    prompt      TEXT NOT NULL,
    response_a  TEXT NOT NULL,
    response_b  TEXT NOT NULL,
    model_a     TEXT NOT NULL,
    model_b     TEXT NOT NULL,
    winner      TEXT NOT NULL,                -- 'a','b','tie'
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_served_enabled ON served_models(enabled, sort_order);
CREATE INDEX IF NOT EXISTS idx_votes_session ON human_votes(session_id);
"""

# ── Connection and lifecycle ────────────────────────────────────────────


class Registry:
    """SQLite-backed run registry. Thread-safe for concurrent readers."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open the database and initialise the schema if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        log.info("registry_connected", db=str(self.db_path))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RegistryError("Registry not connected. Call .connect() first.")
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for an atomic write. Rolls back on exception."""
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Create tables if they don't exist; track schema version."""
        self.conn.executescript(DDL)
        cur = self.conn.execute("SELECT value FROM schema_meta WHERE key = 'version'")
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.commit()
        log.debug("registry_schema_ready", version=SCHEMA_VERSION)

    # ── Run operations ──────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def make_run_id(kind: str, config_hash: str) -> str:
        """Generate a human-readable run ID: {kind}-{date}-{hash_prefix}."""
        date = datetime.now(UTC).strftime("%Y%m%d")
        return f"{kind}-{date}-{config_hash[:4]}"

    def create_run(
        self,
        run_id: str,
        kind: str,
        config_hash: str,
        dataset_hash: str | None = None,
        git_sha: str | None = None,
        hardware: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> str:
        """Insert a new run with status 'running'. Returns the run_id."""
        with self.transaction() as cur:
            cur.execute(
                """INSERT INTO runs
                   (run_id, kind, status, config_hash, dataset_hash, git_sha,
                    hardware_json, started_at, notes)
                   VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    kind,
                    config_hash,
                    dataset_hash,
                    git_sha,
                    json.dumps(hardware) if hardware else None,
                    self._now(),
                    notes,
                ),
            )
        log.info("run_created", run_id=run_id, kind=kind)
        return run_id

    def complete_run(self, run_id: str, metrics: dict[str, Any] | None = None) -> None:
        """Mark a run as completed with optional final metrics."""
        with self.transaction() as cur:
            cur.execute(
                """UPDATE runs SET status = 'completed', finished_at = ?,
                   metrics_json = ? WHERE run_id = ?""",
                (self._now(), json.dumps(metrics) if metrics else None, run_id),
            )
        log.info("run_completed", run_id=run_id)

    def fail_run(self, run_id: str, error: str | None = None) -> None:
        """Mark a run as failed."""
        with self.transaction() as cur:
            cur.execute(
                """UPDATE runs SET status = 'failed', finished_at = ?,
                   notes = COALESCE(notes || ' | ', '') || ? WHERE run_id = ?""",
                (self._now(), error or "unknown error", run_id),
            )
        log.warning("run_failed", run_id=run_id, error=error)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a single run by ID."""
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_runs(self, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List runs, optionally filtered by kind, newest first."""
        query = "SELECT * FROM runs"
        params: list[Any] = []
        if kind:
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # ── Artifact operations ─────────────────────────────────────────────

    def record_artifact(
        self,
        run_id: str,
        kind: str,
        path: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Register an artifact produced by a run. Returns artifact_id."""
        import uuid

        artifact_id = f"art-{uuid.uuid4().hex[:8]}"
        with self.transaction() as cur:
            cur.execute(
                """INSERT INTO artifacts
                   (artifact_id, run_id, kind, path, sha256, size_bytes, created_at, meta_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id,
                    run_id,
                    kind,
                    path,
                    sha256,
                    size_bytes,
                    self._now(),
                    json.dumps(meta) if meta else None,
                ),
            )
        log.info("artifact_recorded", artifact_id=artifact_id, run_id=run_id, kind=kind)
        return artifact_id

    def get_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        """All artifacts for a given run."""
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
        ]

    # ── Dataset operations ──────────────────────────────────────────────

    def record_dataset(
        self,
        dataset_id: str,
        name: str,
        kind: str,
        path: str,
        sha256: str,
        n_rows: int,
        split: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        filter_stats: dict[str, Any] | None = None,
    ) -> str:
        with self.transaction() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO datasets
                   (dataset_id, name, kind, split, path, sha256, n_rows,
                    source_json, filter_stats_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    name,
                    kind,
                    split,
                    path,
                    sha256,
                    n_rows,
                    json.dumps(sources) if sources else None,
                    json.dumps(filter_stats) if filter_stats else None,
                    self._now(),
                ),
            )
        log.info("dataset_recorded", dataset_id=dataset_id, n_rows=n_rows)
        return dataset_id

    def get_dataset(self, sha256: str) -> dict[str, Any] | None:
        """Look up a dataset by its content hash."""
        cur = self.conn.execute("SELECT * FROM datasets WHERE sha256 = ?", (sha256,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ── Served model operations ─────────────────────────────────────────

    def publish_model(
        self,
        model_id: str,
        display_name: str,
        backend: str,
        weights_ref: str,
        run_id: str | None = None,
        sort_order: int = 0,
    ) -> None:
        """Make a model available for serving. Overwrites if model_id exists."""
        with self.transaction() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO served_models
                   (model_id, display_name, backend, weights_ref, run_id, enabled, sort_order)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                (model_id, display_name, backend, weights_ref, run_id, sort_order),
            )
        log.info("model_published", model_id=model_id, backend=backend)

    def list_served_models(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """List models available for serving, ordered by sort_order."""
        query = "SELECT * FROM served_models"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY sort_order"
        return [dict(r) for r in self.conn.execute(query).fetchall()]

    def get_served_model(self, model_id: str) -> dict[str, Any] | None:
        cur = self.conn.execute("SELECT * FROM served_models WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ── Human vote operations ───────────────────────────────────────────

    def record_vote(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        model_a: str,
        model_b: str,
        winner: str,
        session_id: str | None = None,
    ) -> str:
        """Store a pairwise human preference vote. Returns vote_id."""
        import uuid

        vote_id = f"vote-{uuid.uuid4().hex[:8]}"
        with self.transaction() as cur:
            cur.execute(
                """INSERT INTO human_votes
                   (vote_id, session_id, prompt, response_a, response_b,
                    model_a, model_b, winner, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    vote_id,
                    session_id,
                    prompt,
                    response_a,
                    response_b,
                    model_a,
                    model_b,
                    winner,
                    self._now(),
                ),
            )
        log.info("vote_recorded", vote_id=vote_id, winner=winner)
        return vote_id

    def list_votes(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM human_votes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        ]


# ── Singleton accessor ──────────────────────────────────────────────────

_registry: Registry | None = None


def get_registry(db_path: Path | None = None) -> Registry:
    """Get or create the singleton Registry instance."""
    global _registry
    if _registry is None:
        from alignforge.core.paths import get_paths

        path = db_path or get_paths().registry_db
        _registry = Registry(path)
        _registry.connect()
    return _registry


def reset_registry() -> None:
    """Close and discard the singleton. Used by tests."""
    global _registry
    if _registry is not None:
        _registry.close()
        _registry = None
