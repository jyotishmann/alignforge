"""Unit tests for notebooks/_colab.py: pure filesystem/SQLite logic, no Colab, no GPU."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def af() -> ModuleType:
    """Load notebooks/_colab.py by path (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location("_colab", ROOT / "notebooks" / "_colab.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_colab"] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def layout(af: ModuleType, tmp_path: Path) -> Any:
    lay = af.Layout(
        repo=tmp_path / "repo", drive_root=tmp_path / "drive", local_root=tmp_path / "local"
    )
    lay.ensure()
    lay.repo.mkdir()
    return lay


def test_state_roundtrip(af: ModuleType, tmp_path: Path) -> None:
    s = af.State(tmp_path / "state.json")
    assert s.get("sft_hash") is None
    s.set(sft_hash="abc123def456")
    assert s.require("sft_hash") == "abc123def456"
    assert "updated_at" in s.load()
    with pytest.raises(RuntimeError, match="dpo_run"):
        s.require("dpo_run")


def test_find_runs_skips_torn_checkpoint(af: ModuleType, tmp_path: Path) -> None:
    done = tmp_path / "sft-aaa" / "adapter"
    done.mkdir(parents=True)
    (done / "adapter_config.json").write_text("{}")
    ck = tmp_path / "sft-bbb" / "checkpoints"
    (ck / "checkpoint-10").mkdir(parents=True)
    (ck / "checkpoint-10" / "trainer_state.json").write_text("{}")
    (ck / "checkpoint-20").mkdir()  # torn by a disconnect: no trainer_state.json
    runs = {r.run_id: r for r in af.find_runs(tmp_path, "sft")}
    assert runs["sft-aaa"].complete
    assert not runs["sft-bbb"].complete
    assert runs["sft-bbb"].latest_checkpoint.name == "checkpoint-10"
    assert af.latest_resumable(tmp_path, "sft").run_id == "sft-bbb"
    assert af.latest_complete(tmp_path, "sft").run_id == "sft-aaa"


def test_dataset_hash_from_new_manifest(af: ModuleType, tmp_path: Path) -> None:
    before = af.list_manifests(tmp_path)
    m = tmp_path / "processed" / "sft" / "v1" / "manifest.json"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"content_hash": "d7c2f0aabbcc"}))
    assert af.detect_new_dataset_hash(before, tmp_path, "") == "d7c2f0aabbcc"


def test_dataset_hash_falls_back_to_output(af: ModuleType, tmp_path: Path) -> None:
    out = "config_hash: 111111111111\nDataset content hash: \x1b[32m9c1e00ff9c1e\x1b[0m\n"
    assert af.detect_new_dataset_hash({}, tmp_path, out) == "9c1e00ff9c1e"
    with pytest.raises(RuntimeError):
        af.detect_new_dataset_hash({}, tmp_path, "no hashes here")


def test_parse_run_id(af: ModuleType) -> None:
    assert af.parse_run_id("...\nSFT complete. Run ID: sft-20260925-9c1e\n") == "sft-20260925-9c1e"
    assert af.parse_run_id("nothing") is None


def test_copy_tree_respects_overwrite(af: ModuleType, tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    (src / "a" / "f.txt").write_text("new")
    (dst / "a").mkdir(parents=True)
    (dst / "a" / "f.txt").write_text("git-version")
    assert af.copy_tree(src, dst, overwrite=False) == 0
    assert (dst / "a" / "f.txt").read_text() == "git-version"
    assert af.copy_tree(src, dst, overwrite=True) == 1
    assert af.copy_tree(src, dst, overwrite=True) == 0  # unchanged -> skipped


def test_registry_snapshot_and_restore(af: ModuleType, layout: Any) -> None:
    conn = sqlite3.connect(layout.registry_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE runs (run_id TEXT)")
    conn.execute("INSERT INTO runs VALUES ('sft-x')")
    conn.commit()  # connection stays open: snapshot must work against a live WAL DB
    assert af.snapshot_registry(layout) == layout.registry_snapshot
    conn.close()
    layout.registry_db.unlink()
    assert "restored" in af.restore_registry(layout)
    rows = sqlite3.connect(layout.registry_db).execute("SELECT run_id FROM runs").fetchall()
    assert rows == [("sft-x",)]


def test_log_records_filters(af: ModuleType, tmp_path: Path) -> None:
    lines = [
        {"event": "vram_step", "run_id": "sft-a", "peak_gb": 8.1},
        {"event": "vram_step", "run_id": "sft-b", "peak_gb": 9.0},
        {"event": "dpo_step", "run_id": "dpo-a", "implicit_kl": 1.2},
    ]
    (tmp_path / "alignforge.log").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\nnot json\n"
    )
    assert len(af.log_records(tmp_path, event="vram_step", run_id="sft-a")) == 1
    assert af.log_records(tmp_path, key="implicit_kl")[0]["run_id"] == "dpo-a"


def test_audit_flags_known_runtime_bugs(af: ModuleType, tmp_path: Path) -> None:
    train = tmp_path / "src" / "alignforge" / "train"
    train.mkdir(parents=True)
    (train / "callbacks.py").write_text(
        "def wrap_callbacks(cbs): ...\nx = props.total_mem\np = Path('a')\n"
    )
    (train / "sft.py").write_text("SFTTrainer(callbacks=callbacks or [])\n")
    levels = sorted(f.level for f in af.audit(tmp_path))
    assert levels.count("FIX") == 3  # total_mem, missing Path import, never wrapped
    (train / "callbacks.py").write_text(
        "from pathlib import Path\ndef wrap_callbacks(cbs): ...\nx = props.total_memory\n"
    )
    (train / "sft.py").write_text("SFTTrainer(callbacks=wrap_callbacks(callbacks or []))\n")
    assert af.audit(tmp_path) == []
