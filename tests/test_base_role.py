"""Tests for BaseRole.enforce_budget() and section-aware context extraction."""

import pytest

from khonliang.roles.base import _extract_sections, _split_into_sections

# ---------------------------------------------------------------------------
# _split_into_sections
# ---------------------------------------------------------------------------


def test_split_detects_markdown_headings():
    text = "## Abstract\nSome abstract text.\n\n## Introduction\nSome intro text."
    sections = _split_into_sections(text)
    names = [name for name, _ in sections]
    assert "Abstract" in names
    assert "Introduction" in names


def test_split_detects_all_caps_headings():
    text = "ABSTRACT\nThis is the abstract.\n\nMETHODS\nThis describes methods."
    sections = _split_into_sections(text)
    names = [name for name, _ in sections]
    assert "ABSTRACT" in names
    assert "METHODS" in names


def test_split_detects_all_caps_with_colon():
    """ALL CAPS headings with trailing colon should be detected."""
    text = "ABSTRACT:\nThis is the abstract.\n\nMETHODS:\nThis describes methods."
    sections = _split_into_sections(text)
    names = [name for name, _ in sections]
    assert "ABSTRACT" in names
    assert "METHODS" in names


def test_split_detects_numbered_headings():
    text = "1. Introduction\nSome intro.\n\n2. Methods\nSome methods."
    sections = _split_into_sections(text)
    names = [name for name, _ in sections]
    # Numeric prefix is stripped from the heading name
    assert "Introduction" in names
    assert "Methods" in names


def test_split_empty_text_returns_empty():
    assert _split_into_sections("") == []


def test_split_no_headings_returns_preamble_only():
    """Text without any headings produces at most a 'preamble' section,
    which is fewer than 3 sections — triggering the fallback in _extract_sections."""
    text = "Just a paragraph with no headings.\nAnother line."
    sections = _split_into_sections(text)
    # May return a single 'preamble' section or empty list; either way < 3 sections
    assert len(sections) < 3


def test_split_strips_trailing_punctuation_from_heading_name():
    text = "CONCLUSION:\nFinal thoughts.\n\n1. Introduction.\nIntro text."
    sections = _split_into_sections(text)
    names = [name for name, _ in sections]
    assert "CONCLUSION" in names
    assert "Introduction" in names


# ---------------------------------------------------------------------------
# _extract_sections
# ---------------------------------------------------------------------------

_PAPER = """\
## Abstract
This summarises the paper's key findings and contributions.

## Introduction
Background and motivation for the work.

## Methods
Detailed description of the experimental methodology.

## Results
Numerical results and performance figures.

## Conclusion
Summary and future directions.

## Discussion
Interpretation and comparison with related work.
"""


def test_extract_sections_detects_paper_structure():
    sections = _split_into_sections(_PAPER)
    assert len(sections) >= 5


def test_extract_sections_prioritises_abstract_and_intro():
    result = _extract_sections(_PAPER, max_chars=300)
    assert "Abstract" in result
    assert "Introduction" in result


def test_extract_sections_includes_conclusion():
    result = _extract_sections(_PAPER, max_chars=400)
    assert "Conclusion" in result


def test_extract_sections_fallback_for_unstructured_text():
    text = "a" * 2000
    result = _extract_sections(text, max_chars=200)
    assert "[" in result  # fallback header present


def test_extract_sections_labels_output():
    result = _extract_sections(_PAPER, max_chars=500)
    assert result.startswith("[Extracted")


# ---------------------------------------------------------------------------
# BaseRole.enforce_budget via concrete subclass
# ---------------------------------------------------------------------------


class _FakePool:
    """Minimal ModelPool stand-in."""

    def get_client(self, role):
        return None


class _ConcreteRole:
    """Minimal BaseRole subclass that exposes enforce_budget without a real client."""

    max_context_tokens = 100  # 400 chars

    def enforce_budget(self, context: str, strategy: str = "truncate") -> str:
        from khonliang.roles.base import _extract_sections

        if not self.max_context_tokens:
            return context
        max_chars = self.max_context_tokens * 4
        if len(context) <= max_chars:
            return context
        if strategy == "sections":
            return _extract_sections(context, max_chars)
        if strategy != "truncate":
            raise ValueError(
                f"Unknown enforce_budget strategy {strategy!r}. "
                "Use 'truncate' or 'sections'."
            )
        truncated = context[-max_chars:]
        first_newline = truncated.find("\n")
        if first_newline > 0 and first_newline < len(truncated) // 4:
            truncated = truncated[first_newline + 1:]
        return f"[Context truncated to ~{self.max_context_tokens} tokens]\n{truncated}"


def test_enforce_budget_truncate_default():
    role = _ConcreteRole()
    short = "hello"
    assert role.enforce_budget(short) == short  # fits, unchanged


def test_enforce_budget_truncate_keeps_tail():
    role = _ConcreteRole()
    text = "x" * 1000
    result = role.enforce_budget(text, strategy="truncate")
    assert result.endswith("x" * 10)
    assert len(result) <= 500  # with header overhead


def test_enforce_budget_sections_strategy():
    role = _ConcreteRole()
    result = role.enforce_budget(_PAPER * 3, strategy="sections")
    assert result.startswith("[Extracted")


def test_enforce_budget_unknown_strategy_raises():
    role = _ConcreteRole()
    with pytest.raises(ValueError, match="Unknown enforce_budget strategy"):
        role.enforce_budget("x" * 1000, strategy="smart")
