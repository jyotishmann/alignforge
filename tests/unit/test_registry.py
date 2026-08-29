"""Registry CRUD tests. Each test gets a fresh SQLite database."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from alignforge.core.registry import Registry


@pytest.fixture
def reg(tmp_path: Path) -> Generator[Registry, None, None]:
    """Fresh registry per test."""
    r = Registry(tmp_path / "test.db")
    r.connect()
    yield r
    r.close()


class TestRuns:
    def test_create_and_get(self, reg: Registry) -> None:
        run_id = reg.create_run(
            run_id="sft-20250312-abcd",
            kind="sft",
            config_hash="abcd1234abcd",
        )
        row = reg.get_run(run_id)
        assert row is not None
        assert row["status"] == "running"
        assert row["kind"] == "sft"

    def test_complete(self, reg: Registry) -> None:
        run_id = reg.create_run("test-run", "sft", "hash123")
        reg.complete_run(run_id, metrics={"loss": 0.42})
        row = reg.get_run(run_id)
        assert row is not None
        assert row["status"] == "completed"
        assert row["finished_at"] is not None
        assert "0.42" in (row["metrics_json"] or "")

    def test_fail(self, reg: Registry) -> None:
        run_id = reg.create_run("fail-run", "dpo", "hash456")
        reg.fail_run(run_id, error="OOM at step 300")
        row = reg.get_run(run_id)
        assert row is not None
        assert row["status"] == "failed"
        assert "OOM" in (row["notes"] or "")

    def test_list_runs(self, reg: Registry) -> None:
        reg.create_run("a", "sft", "h1")
        reg.create_run("b", "dpo", "h2")
        reg.create_run("c", "sft", "h3")
        all_runs = reg.list_runs()
        assert len(all_runs) == 3
        sft_runs = reg.list_runs(kind="sft")
        assert len(sft_runs) == 2


class TestArtifacts:
    def test_record_and_get(self, reg: Registry) -> None:
        reg.create_run("run-1", "sft", "h1")
        art_id = reg.record_artifact("run-1", "lora_adapter", "/path/to/adapter")
        arts = reg.get_artifacts("run-1")
        assert len(arts) == 1
        assert arts[0]["artifact_id"] == art_id


class TestServedModels:
    def test_publish_and_list(self, reg: Registry) -> None:
        reg.publish_model("base", "Base Model", "echo", "echo", sort_order=1)
        reg.publish_model("sft", "SFT Model", "echo", "echo", sort_order=2)
        models = reg.list_served_models()
        assert len(models) == 2
        assert models[0]["model_id"] == "base"

    def test_get_model(self, reg: Registry) -> None:
        reg.publish_model("dpo", "DPO", "ollama", "alignforge-dpo:v1")
        m = reg.get_served_model("dpo")
        assert m is not None
        assert m["backend"] == "ollama"


class TestVotes:
    def test_record_and_list(self, reg: Registry) -> None:
        vid = reg.record_vote(
            prompt="How do I reverse a list?",
            response_a="Use reversed().",
            response_b="Use [::-1].",
            model_a="base",
            model_b="dpo",
            winner="b",
            session_id="sess-1",
        )
        votes = reg.list_votes()
        assert len(votes) == 1
        assert votes[0]["vote_id"] == vid
        assert votes[0]["winner"] == "b"


class TestTransactions:
    def test_rollback_on_error(self, reg: Registry) -> None:
        """A failed write must not leave partial state."""
        import sqlite3

        reg.create_run("clean-run", "sft", "h1")

        with pytest.raises(sqlite3.IntegrityError), reg.transaction() as cur:
            cur.execute(
                "INSERT INTO runs (run_id, kind, status, config_hash, started_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("clean-run", "sft", "running", "h1", "now"),
            )
            cur.execute(
                "INSERT INTO artifacts (artifact_id, run_id, kind, path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("ghost", "clean-run", "test", "/ghost", "now"),
            )

        # The artifact must not exist — rollback must have discarded it.
        arts = reg.get_artifacts("clean-run")
        assert len(arts) == 0
