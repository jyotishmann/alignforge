"""Interactive labelling tool for tuning the embedding filter threshold.

Run AFTER loading and keyword-filtering the data:
    uv run python scripts/label_filter_sample.py --n-samples 200

Labels are saved to data/filter_labels.json and consumed by `alignforge data build`
to tune the embedding filter threshold and report F1.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer


def main(
    n_samples: int = typer.Option(200, help="Number of candidates to label."),
    output: Path = typer.Option(Path("data/filter_labels.json")),
    seed: int = typer.Option(42),
) -> None:
    from alignforge.data.filters import keyword_filter
    from alignforge.data.loaders import load_all_sources
    from alignforge.data.sources import SFT_SOURCES

    # Load all SFT sources with keyword filter applied.
    specs = list(SFT_SOURCES.values())
    examples, _ = load_all_sources(specs)

    # Apply keyword filter (Stage 1).
    filtered = [
        ex
        for ex in examples
        if keyword_filter(ex.prompt_text if hasattr(ex, "prompt_text") else "")
    ]

    # Sample.
    random.seed(seed)
    sample = random.sample(filtered, min(n_samples, len(filtered)))

    labels: list[dict[str, object]] = []
    typer.echo(f"\nLabelling {len(sample)} candidates. Type 'y' for in-domain, 'n' for out.")
    typer.echo("Domain: developer support assistant — technical questions, code, tools.\n")

    for i, ex in enumerate(sample):
        text = ex.prompt_text if hasattr(ex, "prompt_text") else str(ex)
        typer.echo(f"--- [{i+1}/{len(sample)}] ---")
        typer.echo(text[:500])
        while True:
            resp = typer.prompt("In-domain? [y/n/skip]").strip().lower()
            if resp in ("y", "n", "skip"):
                break
        if resp == "skip":
            continue
        labels.append({"text": text[:500], "in_domain": resp == "y"})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(labels, f, indent=2)

    n_pos = sum(1 for ln in labels if ln["in_domain"])
    typer.echo(f"\nDone. {len(labels)} labelled: {n_pos} in-domain, {len(labels)-n_pos} out.")
    typer.echo(f"Saved to {output}")


if __name__ == "__main__":
    typer.run(main)
