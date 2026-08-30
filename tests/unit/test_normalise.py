"""Text normalisation tests."""

from __future__ import annotations

from alignforge.data.normalise import normalise_text


def test_collapse_whitespace() -> None:
    assert normalise_text("hello   world") == "hello world"


def test_limit_newlines() -> None:
    assert normalise_text("a\n\n\n\n\nb") == "a\n\nb"


def test_strip_control_chars() -> None:
    assert normalise_text("hello\x00world") == "helloworld"


def test_html_entities() -> None:
    assert normalise_text("& <") == "& <"


def test_unicode_nfc() -> None:
    # e + combining accent → precomposed é
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "é")
    assert normalise_text(decomposed) == "é"
