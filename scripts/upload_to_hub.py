"""Upload AlignForge artifacts to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_...
    uv run python scripts/upload_to_hub.py \
        --dpo-run dpo-20250312-a3f9 \
        --repo-id <your-username>/alignforge-dpo-1.5b
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    dpo_run: str = typer.Option(..., "--dpo-run", help="DPO run ID from registry."),
    repo_id: str = typer.Option(..., "--repo-id", help="HF Hub repo, e.g. 'user/alignforge-dpo'."),
    upload_adapter: bool = typer.Option(True, help="Upload the LoRA adapter."),
    upload_gguf: bool = typer.Option(True, help="Upload the Q4_K_M GGUF."),
    upload_card: bool = typer.Option(True, help="Upload the model card."),
    private: bool = typer.Option(False, help="Create as a private repository."),
) -> None:
    """Upload AlignForge artifacts to HuggingFace Hub."""
    from huggingface_hub import HfApi

    from alignforge.core.paths import get_paths
    from alignforge.core.registry import get_registry

    api = HfApi()
    paths = get_paths()
    reg = get_registry()

    # Create (or verify) the repository.
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    typer.echo(f"Repository: https://huggingface.co/{repo_id}")

    # Resolve artifact paths from the registry.
    artifacts = reg.get_artifacts(dpo_run)
    adapter_path: Path | None = None
    gguf_path: Path | None = None

    for art in artifacts:
        if art["kind"] == "lora_adapter":
            adapter_path = Path(art["path"])
        if art["kind"] == "gguf_quantised":
            gguf_path = Path(art["path"])

    # Upload LoRA adapter.
    if upload_adapter and adapter_path and adapter_path.exists():
        typer.echo(f"Uploading LoRA adapter from {adapter_path}...")
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="adapter",
        )
        typer.secho("  ✓ Adapter uploaded.", fg=typer.colors.GREEN)
    elif upload_adapter:
        typer.secho("  ⚠ Adapter not found — skipping.", fg=typer.colors.YELLOW)

    # Upload GGUF.
    if upload_gguf and gguf_path and gguf_path.exists():
        typer.echo(f"Uploading GGUF ({gguf_path.name}, {gguf_path.stat().st_size/1e9:.1f}GB)...")
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
            repo_type="model",
        )
        # Upload the Modelfile alongside the GGUF.
        modelfile = gguf_path.parent / "Modelfile"
        if modelfile.exists():
            api.upload_file(
                path_or_fileobj=str(modelfile),
                path_in_repo="Modelfile",
                repo_id=repo_id,
                repo_type="model",
            )
        typer.secho("  ✓ GGUF + Modelfile uploaded.", fg=typer.colors.GREEN)
    elif upload_gguf:
        typer.secho("  ⚠ GGUF not found — skipping.", fg=typer.colors.YELLOW)

    # Upload model card.
    if upload_card:
        card_path = paths.root / "docs" / "model_card.md"
        if card_path.exists():
            typer.echo("Uploading model card...")
            api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            typer.secho("  ✓ Model card uploaded.", fg=typer.colors.GREEN)

    typer.secho(f"\nDone. View at: https://huggingface.co/{repo_id}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
