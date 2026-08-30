"""HuggingFace dataset loaders. Each yields validated schema instances."""

from __future__ import annotations

from typing import Any

import structlog

from alignforge.data.schemas import PreferenceExample, QuarantineRecord, SFTExample
from alignforge.data.sources import SourceSpec

log = structlog.get_logger()


def _validate_sft(raw: dict[str, Any], spec: SourceSpec) -> SFTExample | QuarantineRecord:
    """Map raw HF row → SFTExample via the spec's field_map."""
    try:
        mapped = {our: raw.get(their, "") for our, their in spec.field_map.items()}
        mapped["source"] = spec.name
        return SFTExample.model_validate(mapped)
    except Exception as exc:
        return QuarantineRecord(raw=raw, source=spec.name, error=str(exc))


def _validate_preference(
    raw: dict[str, Any], spec: SourceSpec
) -> PreferenceExample | QuarantineRecord:
    """Map raw HF row → PreferenceExample."""
    try:
        mapped = {our: raw.get(their, "") for our, their in spec.field_map.items()}
        mapped["source"] = spec.name
        return PreferenceExample.model_validate(mapped)
    except Exception as exc:
        return QuarantineRecord(raw=raw, source=spec.name, error=str(exc))


def _parse_anthropic_hh(text: str) -> tuple[str, str]:
    """Extract the last (Human, Assistant) turn from Anthropic HH format."""
    turns = text.split("\n\nHuman: ")
    if len(turns) < 2:
        raise ValueError("Could not parse Anthropic HH conversation format.")
    last_human_block = turns[-1]
    parts = last_human_block.split("\n\nAssistant: ", 1)
    if len(parts) < 2:
        raise ValueError("No assistant reply found in last turn.")
    return parts[0].strip(), parts[1].strip()


def _extract_ultrafeedback_text(msg_list: list[dict[str, str]] | str) -> str:
    """Extract assistant text from UltraFeedback chosen/rejected message lists."""
    if isinstance(msg_list, str):
        return msg_list
    # It's a list of {"role": ..., "content": ...} dicts.
    for msg in reversed(msg_list):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return str(msg_list)


def load_source(
    spec: SourceSpec, limit: int | None = None
) -> tuple[list[SFTExample] | list[PreferenceExample], list[QuarantineRecord]]:
    """Load and validate one HF source. Returns (valid, quarantined)."""
    from datasets import load_dataset

    log.info("loading_source", name=spec.name, hf_id=spec.hf_id, split=spec.split)

    ds = load_dataset(spec.hf_id, split=spec.split, revision=spec.revision)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    valid: list[Any] = []
    quarantined: list[QuarantineRecord] = []

    for row in ds:
        raw = dict(row)

        # ── Source-specific pre-processing ───────────────────────────
        if spec.name == "oasst1":
            # Only keep English, top-ranked assistant replies.
            if raw.get("lang") != "en" or raw.get("role") != "assistant":
                continue
            if raw.get("rank") is not None and raw["rank"] != 0:
                continue
            # parent_text must be populated by a join — for simplicity we use
            # the message_tree_role approach: skip if no parent text available.
            parent = raw.get("parent_id")
            if not parent:
                continue

        if spec.name == "anthropic_hh":
            try:
                chosen_text = raw.get("chosen", "")
                rejected_text = raw.get("rejected", "")
                prompt_c, chosen = _parse_anthropic_hh(chosen_text)
                _, rejected = _parse_anthropic_hh(rejected_text)
                # Use the prompt from chosen side (they should match).
                raw = {
                    "prompt": prompt_c,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": spec.name,
                }
            except ValueError as exc:
                quarantined.append(QuarantineRecord(raw=raw, source=spec.name, error=str(exc)))
                continue

        if spec.name == "ultrafeedback":
            raw_chosen = raw.get("chosen", "")
            raw_rejected = raw.get("rejected", "")
            raw = {
                "prompt": raw.get("prompt", ""),
                "chosen": _extract_ultrafeedback_text(raw_chosen),
                "rejected": _extract_ultrafeedback_text(raw_rejected),
                "source": spec.name,
            }

        # ── Category filter ──────────────────────────────────────────
        if spec.categories_keep:
            cat = raw.get(spec.category_field, "")
            if cat not in spec.categories_keep:
                continue

        # ── Validate ─────────────────────────────────────────────────
        result = _validate_sft(raw, spec) if spec.kind == "sft" else _validate_preference(raw, spec)

        if isinstance(result, QuarantineRecord):
            quarantined.append(result)
        else:
            valid.append(result)

    log.info(
        "source_loaded",
        name=spec.name,
        valid=len(valid),
        quarantined=len(quarantined),
    )
    return valid, quarantined


def load_all_sources(
    specs: list[SourceSpec], limit: int | None = None
) -> tuple[list[Any], list[QuarantineRecord]]:
    """Load and merge multiple sources. Per-source limit applies to each."""
    all_valid: list[Any] = []
    all_quarantined: list[QuarantineRecord] = []
    for spec in specs:
        v, q = load_source(spec, limit=limit)
        all_valid.extend(v)
        all_quarantined.extend(q)
    log.info(
        "all_sources_loaded",
        total_valid=len(all_valid),
        total_quarantined=len(all_quarantined),
    )
    return all_valid, all_quarantined
