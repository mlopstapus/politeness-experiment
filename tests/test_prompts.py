import pytest
from src.prompts import build_prompt

CONTENT = "Acme Corp reported record revenue in Q3."


def test_bare_prompt():
    assert build_prompt(CONTENT, "bare") == f"Summarize the following text:\n\n{CONTENT}"


def test_polite_prompt():
    assert build_prompt(CONTENT, "polite") == f"Please summarize the following text:\n\n{CONTENT}"


def test_overly_polite_contains_framing_and_content():
    result = build_prompt(CONTENT, "overly_polite")
    assert result.startswith("Hey, when you get a chance")
    assert CONTENT in result


def test_content_identical_across_variants():
    from src.schema import VARIANTS
    for variant in VARIANTS:
        assert CONTENT in build_prompt(CONTENT, variant)


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        build_prompt(CONTENT, "rude")
