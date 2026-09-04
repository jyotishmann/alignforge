"""Write the final parquet artifact with a manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger()


def write_dataset_artifact(
    records: list[dict[str, Any]],
    output_dir: Path,
    name: str,
    kind: str,
    val_ratio: float = 0.05,
    quarantine: list[Any] | None = None,
    funnel: dict[str, Any] | None = None,
    filter_eval: dict[str, Any] | None = None,
    token_stats: dict[str, Any] | None = None,
    decontamination: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
    seed: int = 42,
) -> tuple[Path, str]:
    """Write train/val parquet + manifest. Returns (artifact_dir, content_hash)."""
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Train/val split.
    n_val = max(1, int(len(df) * val_ratio))
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    # Write.
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    # Quarantine.
    if quarantine:
        q_path = output_dir / "quarantine.jsonl"
        with q_path.open("w") as f:
            for q in quarantine:
                f.write(json.dumps(q.model_dump() if hasattr(q, "model_dump") else q) + "\n")

    # Content hash: SHA-256 of both parquet files concatenated.
    hasher = hashlib.sha256()
    for p in [train_path, val_path]:
        hasher.update(p.read_bytes())
    content_hash = hasher.hexdigest()[:12]

    # Manifest.
    manifest = {
        "name": name,
        "kind": kind,
        "content_hash": content_hash,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_total": len(df),
        "val_ratio": val_ratio,
        "sources": sources or [],
        "funnel": funnel or {},
        "filter_eval": filter_eval or {},
        "token_stats": token_stats or {},
        "decontamination": decontamination or {},
        "seed": seed,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        "dataset_written",
        path=str(output_dir),
        n_train=len(train_df),
        n_val=len(val_df),
        content_hash=content_hash,
    )
    return output_dir, content_hash
