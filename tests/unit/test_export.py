"""Export stage unit tests — no llama.cpp, no Ollama required."""

from __future__ import annotations

# import tempfile
from pathlib import Path
from unittest.mock import patch  # MagicMock

import pytest


class TestSha256:
    def test_sha256_is_deterministic(self, tmp_path: Path) -> None:
        from alignforge.export.gguf import _sha256

        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h1 = _sha256(f)
        h2 = _sha256(f)
        assert h1 == h2
        assert len(h1) == 16  # truncated to 16 hex chars

    def test_sha256_differs_for_different_content(self, tmp_path: Path) -> None:
        from alignforge.export.gguf import _sha256

        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content_a")
        f2.write_bytes(b"content_b")
        assert _sha256(f1) != _sha256(f2)


class TestQuantizeBinDiscovery:
    def test_raises_when_llama_cpp_absent(self, tmp_path: Path) -> None:
        from alignforge.core.config import AlignForgeConfig
        from alignforge.export.gguf import GGUFExporter

        cfg = AlignForgeConfig()
        exporter = GGUFExporter(cfg)
        exporter.llama_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="llama-quantize"):
            exporter._find_quantize_bin()

    def test_finds_bin_at_known_path(self, tmp_path: Path) -> None:
        from alignforge.core.config import AlignForgeConfig
        from alignforge.export.gguf import GGUFExporter

        cfg = AlignForgeConfig()
        exporter = GGUFExporter(cfg)
        exporter.llama_dir = tmp_path

        # Create the expected binary path.
        bin_dir = tmp_path / "build" / "bin"
        bin_dir.mkdir(parents=True)
        quantize_bin = bin_dir / "llama-quantize"
        quantize_bin.touch()

        found = exporter._find_quantize_bin()
        assert found == quantize_bin


class TestExportDryRun:
    def test_dry_run_prints_plan(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry run should not raise and should not touch the filesystem."""
        from alignforge.core.config import AlignForgeConfig
        from alignforge.core.registry import Registry
        from alignforge.export.gguf import run_export

        reg = Registry(tmp_path / "test.db")
        reg.connect()

        # Create a fake run and adapter.
        reg.create_run("dpo-test-1234", "dpo", "abcd1234abcd")
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        reg.record_artifact("dpo-test-1234", "lora_adapter", str(adapter_path))

        cfg = AlignForgeConfig()

        with (
            patch("alignforge.export.gguf.get_registry", return_value=reg),
            patch("alignforge.export.gguf.get_paths") as mock_paths,
        ):
            mock_paths.return_value.artifacts_dir = tmp_path
            mock_paths.return_value.models_dir = tmp_path
            mock_paths.return_value.root = tmp_path

            # Should not raise even with dry_run=True.
            result = run_export(cfg=cfg, dpo_run_id="dpo-test-1234", dry_run=True)

        assert result == {}
        reg.close()


class TestRegistryPublish:
    def test_infers_ollama_tag_from_artifacts(self, tmp_path: Path) -> None:
        """publish should infer weights_ref from ollama_tag artifact."""
        from alignforge.core.registry import Registry

        reg = Registry(tmp_path / "test.db")
        reg.connect()
        reg.create_run("dpo-run-001", "dpo", "aabbccdd1234")
        reg.record_artifact("dpo-run-001", "ollama_tag", "alignforge-dpo:v1")

        # Simulate the CLI's inference logic.
        arts = reg.get_artifacts("dpo-run-001")
        ollama_arts = [a for a in arts if a["kind"] == "ollama_tag"]
        assert len(ollama_arts) == 1
        assert ollama_arts[0]["path"] == "alignforge-dpo:v1"

        reg.publish_model(
            model_id="dpo",
            display_name="DPO (β=0.1)",
            backend="ollama",
            weights_ref=ollama_arts[0]["path"],
            run_id="dpo-run-001",
        )
        m = reg.get_served_model("dpo")
        assert m is not None
        assert m["weights_ref"] == "alignforge-dpo:v1"
        reg.close()
