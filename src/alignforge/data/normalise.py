"""Text normalisation: whitespace, control characters, encoding artefacts."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from alignforge.data.schemas import PreferenceExample, QuarantineRecord, SFTExample


def normalise_text(text: str) -> str:
    """Apply all normalisations in a fixed order."""
    text = _strip_control_chars(text)
    text = _normalise_unicode(text)
    text = _collapse_whitespace(text)
    text = _strip_html_entities(text)
    return text.strip()


def _strip_control_chars(text: str) -> str:
    """Remove non-printable characters except newlines and tabs."""
    return "".join(
        c for c in text if c in ("\n", "\t") or not unicodedata.category(c).startswith("C")
    )


def _normalise_unicode(text: str) -> str:
    """NFC normalisation — canonical decomposition then composition."""
    return unicodedata.normalize("NFC", text)


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs to single space; limit consecutive newlines to 2."""
    text = re.sub(r"[^\S\n]+", " ", text)  # horizontal whitespace → single space
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines → 2
    return text


def _strip_html_entities(text: str) -> str:
    """Decode common HTML entities left over from web scraping."""
    import html

    return html.unescape(text)


def normalise_examples(
    examples: list[SFTExample] | list[PreferenceExample],
) -> tuple[list[Any], list[QuarantineRecord]]:
    """Normalise text fields. Returns (cleaned, quarantined-after-normalisation)."""
    cleaned: list[Any] = []
    quarantined: list[QuarantineRecord] = []

    for ex in examples:
        try:
            new: SFTExample | PreferenceExample
            if isinstance(ex, SFTExample):
                new = SFTExample(
                    instruction=normalise_text(ex.instruction),
                    input=normalise_text(ex.input),
                    response=normalise_text(ex.response),
                    source=ex.source,
                )
            else:
                new = PreferenceExample(
                    prompt=normalise_text(ex.prompt),
                    chosen=normalise_text(ex.chosen),
                    rejected=normalise_text(ex.rejected),
                    source=ex.source,
                )
            cleaned.append(new)
        except Exception as exc:
            quarantined.append(
                QuarantineRecord(
                    raw=ex.model_dump(),
                    source=ex.source,
                    error=str(exc),
                    stage="normalisation",
                )
            )

    return cleaned, quarantined
