"""Registry integration tests — full lifecycle including concurrency."""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from alignforge.core.registry import Registry


@pytest.fixture
def reg(tmp_path: Path) -> Generator[Registry, None, None]:
    r = Registry(tmp_path / "test.db")
    r.connect()
    yield r
    r.close()


def test_full_run_lifecycle(reg: Registry) -> None:
    """Create → complete → verify metrics."""
    run_id = reg.create_run(
        "sft-20250312-abc1",
        "sft",
        "aabbcc001122",
        dataset_hash="dd001122",
        notes="test run",
    )
    run = reg.get_run(run_id)
    assert run is not None
    assert run["status"] == "running"

    reg.complete_run(run_id, metrics={"loss": 0.312, "reward_margin": 0.44})

    completed = reg.get_run(run_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert "0.312" in run["metrics_json"]
    assert run["finished_at"] is not None


def test_fail_run_preserves_notes(reg: Registry) -> None:
    run_id = reg.create_run("fail-run", "dpo", "hash123", notes="initial note")
    reg.fail_run(run_id, error="OOM at step 300")
    run = reg.get_run(run_id)
    assert run is not None
    assert "OOM" in run["notes"]
    assert "initial note" in run["notes"]


def test_artifact_chain(reg: Registry) -> None:
    """Verify full artifact chain: adapter → gguf → ollama_tag."""
    reg.create_run("dpo-chain", "dpo", "chainhash001")
    reg.record_artifact("dpo-chain", "lora_adapter", "/artifacts/adapter", sha256="aa11")
    reg.record_artifact("dpo-chain", "gguf_quantised", "/models/q4km.gguf", sha256="bb22")
    reg.record_artifact("dpo-chain", "ollama_tag", "alignforge-dpo:v1")

    arts = reg.get_artifacts("dpo-chain")
    kinds = {a["kind"] for a in arts}
    assert kinds == {"lora_adapter", "gguf_quantised", "ollama_tag"}


def test_concurrent_read_during_write(reg: Registry) -> None:
    """WAL mode: a SELECT must not block while a write is in progress."""
    import time

    # Start a long write in a background thread.
    results: list[str] = []
    barrier = threading.Barrier(2)

    def writer():
        with reg.transaction() as cur:
            cur.execute(
                "INSERT INTO runs (run_id,kind,status,config_hash,started_at)"
                " VALUES ('bg-run','sft','running','h1','2025-01-01T00:00:00')"
            )
            barrier.wait()  # signal reader to proceed
            time.sleep(0.1)  # hold the write lock briefly
        results.append("write_done")

    t = threading.Thread(target=writer)
    t.start()
    barrier.wait()

    # Read should succeed immediately despite the open write.
    runs = reg.list_runs()
    results.append(f"read_returned_{len(runs)}")
    t.join()

    # Both should have completed without deadlock.
    assert any("read_returned" in r for r in results)
    assert "write_done" in results


def test_dataset_hash_lookup(reg: Registry) -> None:
    reg.record_dataset(
        "dev-sft-abcdef",
        name="dev_assistant_sft",
        kind="sft",
        path="/data/processed/dev/latest",
        sha256="abcdef123456",
        n_rows=17902,
    )
    found = reg.get_dataset("abcdef123456")
    assert found is not None
    assert found["n_rows"] == 17902


def test_served_models_sort_order(reg: Registry) -> None:
    """Models are returned in ascending sort_order."""
    reg.publish_model("dpo", "DPO", "echo", "echo", sort_order=3)
    reg.publish_model("base", "Base", "echo", "echo", sort_order=1)
    reg.publish_model("sft", "SFT", "echo", "echo", sort_order=2)

    models = reg.list_served_models()
    ids = [m["model_id"] for m in models]
    assert ids == ["base", "sft", "dpo"]
