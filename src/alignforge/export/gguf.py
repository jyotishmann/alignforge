"""GGUF export pipeline: merge → convert → quantise → Modelfile → ollama create."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import structlog

from alignforge.core.config import AlignForgeConfig
from alignforge.core.paths import get_paths

log = structlog.get_logger()


class GGUFExporter:
    """Orchestrates the full GGUF export pipeline.

    Steps:
      1. Merge LoRA adapter into fp16 base weights.
      2. Convert fp16 safetensors → f16.gguf (llama.cpp script).
      3. Quantise f16.gguf → Q4_K_M.gguf (llama-quantize).
      4. Generate Ollama Modelfile from ChatFormat.
      5. `ollama create` registers the model.
      6. Smoke test: one generation via Ollama HTTP.
    """

    def __init__(self, cfg: AlignForgeConfig) -> None:
        self.cfg = cfg
        self.paths = get_paths()
        self.llama_dir = (self.paths.root / cfg.export.llama_cpp_dir).resolve()

    # ── Step 1: Merge ───────────────────────────────────────────────────

    def merge(self, adapter_path: Path, run_id: str) -> Path:
        """Merge adapter into fp16 base. Returns merged_model_dir."""
        from alignforge.models.merge import merge_adapter_into_base

        output_dir = self.paths.artifacts_dir / run_id / "merged_fp16"
        if output_dir.exists() and (output_dir / "config.json").exists():
            log.info("merge_skipped_already_exists", path=str(output_dir))
            return output_dir

        log.info("export_merging", adapter=str(adapter_path))
        return merge_adapter_into_base(
            cfg=self.cfg,
            adapter_path=adapter_path,
            output_path=output_dir,
        )

    # ── Step 2: Convert HF → f16 GGUF ──────────────────────────────────

    def convert(self, merged_dir: Path, run_id: str) -> Path:
        """Run convert_hf_to_gguf.py. Returns path to f16.gguf."""
        gguf_dir = self.paths.models_dir / "gguf" / run_id
        gguf_dir.mkdir(parents=True, exist_ok=True)
        f16_path = gguf_dir / "f16.gguf"

        if f16_path.exists():
            log.info("convert_skipped_already_exists", path=str(f16_path))
            return f16_path

        convert_script = self.llama_dir / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found at {convert_script}. "
                f"Run: bash scripts/setup_llama_cpp.sh"
            )

        log.info("export_converting", merged=str(merged_dir), output=str(f16_path))
        result = subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"GGUF conversion failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        if not f16_path.exists():
            raise RuntimeError(f"Conversion succeeded but f16.gguf not found at {f16_path}.")

        size_gb = f16_path.stat().st_size / (1024**3)
        log.info("convert_complete", path=str(f16_path), size_gb=round(size_gb, 2))
        return f16_path

    # ── Step 3: Quantise f16 → Q4_K_M ──────────────────────────────────

    def quantise(self, f16_path: Path, run_id: str) -> Path:
        """Run llama-quantize to produce Q4_K_M.gguf. Returns path."""
        quant_type = self.cfg.export.quant_type
        gguf_dir = f16_path.parent
        q_path = gguf_dir / f"{quant_type}.gguf"

        if q_path.exists():
            log.info("quantise_skipped_already_exists", path=str(q_path))
            return q_path

        quantize_bin = self._find_quantize_bin()
        log.info("export_quantising", quant_type=quant_type, input=str(f16_path))

        result = subprocess.run(
            [str(quantize_bin), str(f16_path), str(q_path), quant_type],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Quantisation failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        if not q_path.exists():
            raise RuntimeError(f"Quantise succeeded but output not found at {q_path}.")

        size_gb = q_path.stat().st_size / (1024**3)
        log.info("quantise_complete", path=str(q_path), size_gb=round(size_gb, 2))
        return q_path

    def _find_quantize_bin(self) -> Path:
        """Locate the llama-quantize binary."""
        for candidate in [
            self.llama_dir / "build" / "bin" / "llama-quantize",
            self.llama_dir / "build" / "bin" / "quantize",
            self.llama_dir / "llama-quantize",
        ]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "llama-quantize binary not found. Run: bash scripts/setup_llama_cpp.sh"
        )

    # ── Step 4: Modelfile ────────────────────────────────────────────────

    def write_modelfile(self, q_path: Path, run_id: str, tokenizer: Any) -> Path:
        """Generate Ollama Modelfile from ChatFormat (connector C10)."""
        from alignforge.models.chat_format import ChatFormat

        modelfile_path = q_path.parent / "Modelfile"

        chat_format = ChatFormat.from_tokenizer(tokenizer)
        template_block = chat_format.to_modelfile_block()

        # Relative path from Modelfile location to GGUF.
        rel_gguf = os.path.relpath(q_path, modelfile_path.parent)

        content = "\n".join(
            [
                f"FROM {rel_gguf}",
                "",
                template_block,
                "",
                "PARAMETER num_ctx 2048",
                "PARAMETER num_predict 512",
                f"# Generated by AlignForge export — run_id: {run_id}",
            ]
        )

        modelfile_path.write_text(content)
        log.info("modelfile_written", path=str(modelfile_path))
        return modelfile_path

    # ── Step 5: ollama create ────────────────────────────────────────────

    def ollama_create(self, modelfile_path: Path, run_id: str) -> str:
        """Register the model with a local Ollama daemon. Returns the tag."""
        prefix = self.cfg.export.ollama_tag_prefix
        version = self.cfg.export.ollama_version
        tag = f"{prefix}-dpo:v{version}"

        log.info("ollama_create_starting", tag=tag)
        result = subprocess.run(
            ["ollama", "create", tag, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"`ollama create` failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}\n"
                f"Is Ollama running? Try: ollama serve &"
            )

        log.info("ollama_create_complete", tag=tag)
        return tag

    # ── Step 6: smoke test ───────────────────────────────────────────────

    def smoke_test(self, tag: str) -> str:
        """Generate one response via Ollama HTTP to verify the model works."""
        import httpx

        prompt = self.cfg.export.smoke_test_prompt
        log.info("smoke_test_start", tag=tag, prompt=prompt)

        ollama_host = os.environ.get("ALIGNFORGE_OLLAMA_HOST", "http://localhost:11434")
        r = httpx.post(
            f"{ollama_host}/api/generate",
            json={
                "model": tag,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 128},
            },
            timeout=60.0,
        )
        r.raise_for_status()
        response = r.json().get("response", "")

        if not response.strip():
            raise RuntimeError("Smoke test returned empty response — model may be broken.")

        log.info("smoke_test_passed", tag=tag, response_preview=response[:150])
        return str(response)


def run_export(
    cfg: AlignForgeConfig,
    dpo_run_id: str,
    dry_run: bool = False,
    skip_smoke_test: bool = False,
    skip_gguf_eval: bool = False,
) -> dict[str, str]:
    """Full export pipeline. Returns {artifact_kind: path_or_tag}."""
    from alignforge.core.registry import get_registry
    from alignforge.models.loading import load_tokenizer

    reg = get_registry()
    paths = get_paths()

    # Resolve the DPO adapter path.
    adapter_path = _resolve_adapter(reg, dpo_run_id, paths)

    if dry_run:
        _print_export_dry_run(cfg, dpo_run_id, adapter_path)
        return {}

    # Run the parity test before doing anything expensive.
    _run_parity_test_or_warn()

    exporter = GGUFExporter(cfg)
    artifacts: dict[str, str] = {}

    try:
        # Step 1: Merge.
        merged_dir = exporter.merge(adapter_path, dpo_run_id)
        reg.record_artifact(dpo_run_id, "merged_fp16", str(merged_dir))

        # Load tokenizer once for Modelfile generation.
        tokenizer = load_tokenizer(cfg)

        # Step 2: Convert.
        f16_path = exporter.convert(merged_dir, dpo_run_id)
        reg.record_artifact(dpo_run_id, "gguf_f16", str(f16_path))
        artifacts["gguf_f16"] = str(f16_path)

        # Step 3: Quantise.
        q_path = exporter.quantise(f16_path, dpo_run_id)
        sha256 = _sha256(q_path)
        size = q_path.stat().st_size
        reg.record_artifact(
            dpo_run_id,
            "gguf_quantised",
            str(q_path),
            sha256=sha256,
            size_bytes=size,
            meta={"quant_type": cfg.export.quant_type},
        )
        artifacts["gguf_quantised"] = str(q_path)

        # Step 4: Modelfile.
        modelfile_path = exporter.write_modelfile(q_path, dpo_run_id, tokenizer)
        artifacts["modelfile"] = str(modelfile_path)

        # Step 5: ollama create.
        tag = exporter.ollama_create(modelfile_path, dpo_run_id)
        reg.record_artifact(dpo_run_id, "ollama_tag", tag)
        artifacts["ollama_tag"] = tag

        # Step 6: Smoke test.
        if not skip_smoke_test:
            response = exporter.smoke_test(tag)
            artifacts["smoke_test_response"] = response[:200]

        # Step 7: Register for serving.
        run_record = reg.get_run(dpo_run_id)  # noqa: F841
        display = f"DPO GGUF ({cfg.export.quant_type})"
        reg.publish_model(
            model_id="dpo_gguf",
            display_name=display,
            backend="ollama",
            weights_ref=tag,
            run_id=dpo_run_id,
            sort_order=4,
        )
        log.info("export_complete", run_id=dpo_run_id, artifacts=artifacts)
        return artifacts

    except Exception as exc:
        log.error("export_failed", run_id=dpo_run_id, error=str(exc))
        raise


def _resolve_adapter(reg: Any, run_id: str, paths: Any) -> Path:
    """Find the DPO adapter directory from the run ID."""
    arts = reg.get_artifacts(run_id)
    for art in arts:
        if art["kind"] == "lora_adapter":
            p = Path(art["path"])
            if p.exists():
                return p
    candidate = paths.artifacts_dir / run_id / "adapter"
    if candidate.exists():
        return Path(candidate)
    from alignforge.core.errors import RegistryError

    raise RegistryError(
        f"No lora_adapter artifact found for run {run_id!r}. "
        f"Check `alignforge registry list --kind dpo`.",
        run_id=run_id,
    )


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _run_parity_test_or_warn() -> None:
    """Run the chat format parity test before export. Warn but don't block on failure."""
    try:
        import subprocess as sp

        result = sp.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/integration/test_chat_format_parity.py",
                "-m",
                "integration",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            log.info("parity_test_passed_pre_export")
        else:
            log.warning(
                "parity_test_failed_pre_export",
                msg="The chat format parity test failed. "
                "The GGUF model may receive different prompts than the evaluated model. "
                "Investigate before deploying.",
                output=result.stdout[-500:],
            )
    except Exception as exc:
        log.warning("parity_test_skipped", error=str(exc))


def _print_export_dry_run(cfg: AlignForgeConfig, run_id: str, adapter_path: Path) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Export dry run: {run_id}")
    table.add_column("Step")
    table.add_column("Output")
    paths = get_paths()
    gguf_dir = paths.models_dir / "gguf" / run_id
    tag = f"{cfg.export.ollama_tag_prefix}-dpo:v{cfg.export.ollama_version}"
    for step, output in [
        ("1. Merge fp16", str(paths.artifacts_dir / run_id / "merged_fp16")),
        ("2. Convert to f16.gguf", str(gguf_dir / "f16.gguf")),
        ("3. Quantise Q4_K_M", str(gguf_dir / f"{cfg.export.quant_type}.gguf")),
        ("4. Modelfile", str(gguf_dir / "Modelfile")),
        ("5. ollama create", tag),
        ("6. Smoke test", cfg.export.smoke_test_prompt[:60] + "..."),
    ]:
        table.add_row(step, output)
    console.print(table)
    console.print(f"\nAdapter source: {adapter_path}")
    console.print("[yellow]Dry run — no files written.[/yellow]")
