"""Dataset build orchestrator — wires the full pipeline (connector C2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from alignforge.core.config import AlignForgeConfig
from alignforge.core.paths import get_paths
from alignforge.data.decontaminate import decontaminate
from alignforge.data.dedup import deduplicate
from alignforge.data.filters import apply_domain_filter, load_eval_prompts, tune_threshold
from alignforge.data.loaders import load_all_sources
from alignforge.data.normalise import normalise_examples
from alignforge.data.sources import PREF_SOURCES, SFT_SOURCES
from alignforge.data.writer import write_dataset_artifact

log = structlog.get_logger()


def build_dataset(
    cfg: AlignForgeConfig,
    limit: int | None = None,
    skip_embedding: bool = False,
    labels_path: Path | None = None,
) -> tuple[Path, str]:
    """Execute the full data pipeline. Returns (artifact_dir, content_hash).

    Steps (matching the master document §4.2):
      1. Load from HF sources
      2. Validate (quarantine rejects)
      3. Normalise
      4. Domain filter (keyword → embedding)
      5. Deduplicate (exact → MinHash)
      6. Decontaminate against eval prompts
      7. Format into chat template
      8. Filter by token length
      9. Write parquet + manifest
    """
    paths = get_paths()
    kind = cfg.data.kind

    # Resolve sources.
    source_registry = SFT_SOURCES if kind == "sft" else PREF_SOURCES
    if cfg.data.sources:
        specs = [source_registry[s] for s in cfg.data.sources if s in source_registry]
    else:
        specs = list(source_registry.values())

    funnel: dict[str, int] = {}

    # ── Step 1: Load ────────────────────────────────────────────────────
    examples, quarantine = load_all_sources(specs, limit=limit)
    funnel["loaded"] = len(examples) + len(quarantine)
    funnel["validated"] = len(examples)
    funnel["quarantined"] = len(quarantine)

    # ── Step 2&3: Normalise ─────────────────────────────────────────────
    examples, norm_q = normalise_examples(examples)
    quarantine.extend(norm_q)

    # ── Step 4: Domain filter ───────────────────────────────────────────
    eval_prompts = load_eval_prompts(paths.evals_dir)

    # Threshold tuning from hand labels, if available.
    filter_eval: dict[str, Any] = {}
    threshold = 0.45
    if labels_path and labels_path.exists():
        with labels_path.open() as f:
            labels_data = json.load(f)
        label_texts = [label["text"] for label in labels_data]
        label_values = [label["in_domain"] for label in labels_data]
        from alignforge.data.filters import EmbeddingFilter

        ef = EmbeddingFilter(eval_prompts, threshold=0.0)
        threshold, threshold_results = tune_threshold(ef, label_texts, label_values)
        filter_eval = {
            "threshold": threshold,
            "f1_by_threshold": threshold_results,
            "n_labelled": len(labels_data),
        }

    examples, filter_stats = apply_domain_filter(
        examples,
        eval_prompts,
        threshold=threshold,
        skip_embedding=skip_embedding,
    )
    funnel["after_keyword"] = filter_stats.get("after_keyword", len(examples))
    funnel["after_embedding"] = filter_stats.get("after_embedding", len(examples))

    # ── Step 5: Dedup ───────────────────────────────────────────────────
    examples, _dedup_stats = deduplicate(examples, jaccard_threshold=cfg.data.dedup_jaccard)
    funnel["after_dedup"] = len(examples)

    # ── Step 6: Decontaminate ───────────────────────────────────────────
    decontam_n = 0
    if cfg.data.decontaminate and eval_prompts:
        all_eval = load_eval_prompts(paths.evals_dir, suites=cfg.eval.suites)
        examples, decontam_n = decontaminate(
            examples,
            all_eval,
            n=cfg.data.decontaminate_ngram,
        )
    funnel["after_decontamination"] = len(examples)
    decontamination_info = {"n_dropped": decontam_n, "ngram": cfg.data.decontaminate_ngram}

    # ── Step 7: Format ──────────────────────────────────────────────────
    # We need a chat template. For now, use a simple placeholder that
    # will be replaced when the model's tokenizer is available (Part 04).
    # In the data build, we store the raw fields and format at train time
    # if no tokenizer is available here.
    records: list[dict[str, Any]] = []
    for ex in examples:
        if kind == "sft":
            records.append(
                {
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "response": ex.response,
                    "source": ex.source,
                }
            )
        else:
            records.append(
                {
                    "prompt": ex.prompt,
                    "chosen": ex.chosen,
                    "rejected": ex.rejected,
                    "source": ex.source,
                }
            )

    # ── Step 8: (Length filter deferred to train time when tokenizer available) ─

    funnel["final"] = len(records)

    # ── Step 9: Write ───────────────────────────────────────────────────
    output_dir = paths.data_dir / "processed" / cfg.data.name
    source_info = [{"name": s.name, "hf_id": s.hf_id, "split": s.split} for s in specs]

    artifact_dir, content_hash = write_dataset_artifact(
        records=records,
        output_dir=output_dir / "latest",
        name=cfg.data.name,
        kind=kind,
        val_ratio=cfg.data.val_ratio,
        quarantine=quarantine,
        funnel=funnel,
        filter_eval=filter_eval,
        token_stats={},  # filled at train time when tokenizer is available
        decontamination=decontamination_info,
        sources=source_info,
        seed=cfg.project.seed,
    )

    # ── Step 10: Register ───────────────────────────────────────────────
    from alignforge.core.registry import get_registry

    reg = get_registry()
    reg.record_dataset(
        dataset_id=f"{cfg.data.name}-{content_hash}",
        name=cfg.data.name,
        kind=kind,
        path=str(artifact_dir),
        sha256=content_hash,
        n_rows=len(records),
        sources=source_info,
        filter_stats=funnel,
    )

    # Print the funnel table.
    log.info("data_build_complete", name=cfg.data.name, hash=content_hash, funnel=funnel)
    return artifact_dir, content_hash
