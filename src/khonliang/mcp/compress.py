"""Post-compression helper for MCP tool outputs.

Uses a local model (cheap, fast) to compress raw LLM output into structured
artifacts before sending to external coding agents (expensive context).

Falls back to rule-based extraction when no model is available.

Usage:

    # Async — uses local model
    compact = await compress_for_agent(raw_text, CompactSynthesis)

    # Sync — rule-based only (no model call)
    compact = compress_rule_based(raw_text, CompactSynthesis)
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from khonliang.mcp.artifacts import CompactConcept, CompactFR, CompactSynthesis
from khonliang.mcp.budget import ContextBudget, fit_to_budget

T = TypeVar("T", CompactConcept, CompactFR, CompactSynthesis)

# Prompt template for local-model compression
_COMPRESS_PROMPT = """Extract structured data from the text below.
Return ONLY a JSON object matching this schema — no explanation, no markdown.

Schema: {schema}

Text:
{text}"""

# Schemas for each artifact type
_SCHEMAS: dict[type, str] = {
    CompactConcept: (
        '{"name": str, "relevance": float 0-1, '
        '"paper_count": int, "top_paper": str, "actionable": bool}'
    ),
    CompactFR: (
        '{"id": str, "title": str, '
        '"priority": "high"|"medium"|"low", "target": str, '
        '"concept": str, "depends_on": [str]}'
    ),
    CompactSynthesis: (
        '{"topic": str, "paper_count": int, '
        '"key_findings": [str max 5], '
        '"relevance": {"project": float}, "suggested_frs": [str]}'
    ),
}

_DEFAULT_MODEL = "llama3.2:3b"


async def compress_for_agent(
    raw_text: str,
    artifact_type: type[T],
    budget: ContextBudget | None = None,
    model: str = _DEFAULT_MODEL,
    base_url: str = "http://localhost:11434",
) -> T:
    """Compress raw text into a structured artifact using a local model.

    Strategy:
    1. Try to parse raw_text as JSON directly
    2. If that fails, invoke a local model to extract fields
    3. Apply budget constraints if provided
    4. Falls back to rule-based extraction on any model error

    Args:
        raw_text: Raw LLM output or unstructured text.
        artifact_type: Target artifact class (CompactConcept, CompactFR, etc.).
        budget: Optional budget to constrain list fields.
        model: Local model to use for compression.
        base_url: Ollama base URL.

    Returns:
        An instance of artifact_type.
    """
    # Fast path: try direct JSON parse
    parsed = _try_parse_json(raw_text)
    if parsed is not None:
        return _build_artifact(parsed, artifact_type, budget)

    # Model path: use local model to extract
    try:
        from khonliang.client import OllamaClient

        schema = _SCHEMAS.get(artifact_type, "{}")
        prompt = _COMPRESS_PROMPT.format(schema=schema, text=raw_text[:2000])

        async with OllamaClient(model=model, base_url=base_url) as client:
            result = await client.generate(prompt, model=model)
            parsed = _try_parse_json(result)
            if parsed is not None:
                return _build_artifact(parsed, artifact_type, budget)
    except Exception:
        pass  # Fall through to rule-based

    # Fallback: rule-based extraction
    return _rule_based_extract(raw_text, artifact_type)


def compress_rule_based(
    raw_text: str,
    artifact_type: type[T],
    budget: ContextBudget | None = None,
) -> T:
    """Synchronous rule-based compression — no model call.

    Useful when you need compression but can't await, or when
    the local model is unavailable.
    """
    parsed = _try_parse_json(raw_text)
    if parsed is not None:
        return _build_artifact(parsed, artifact_type, budget)
    return _rule_based_extract(raw_text, artifact_type)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from text."""
    # Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON block in markdown
    match = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _build_artifact(
    data: dict[str, Any],
    artifact_type: type[T],
    budget: ContextBudget | None = None,
) -> T:
    """Construct an artifact from parsed data, applying budget to list fields."""
    if budget is not None:
        # Trim list fields to budget
        for key, val in data.items():
            if isinstance(val, list):
                if all(isinstance(v, dict) for v in val):
                    data[key] = fit_to_budget(val, budget)
                else:
                    data[key] = val[: budget.max_items]
    return artifact_type.from_dict(data)


def _rule_based_extract(raw_text: str, artifact_type: type[T]) -> T:
    """Extract fields from raw text using simple heuristics."""
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]

    if artifact_type is CompactConcept:
        return CompactConcept.from_dict({
            "name": lines[0] if lines else "unknown",
            "relevance": 0.5,
            "paper_count": 0,
            "top_paper": lines[1] if len(lines) > 1 else "",
            "actionable": False,
        })  # type: ignore[return-value]
    elif artifact_type is CompactFR:
        return CompactFR.from_dict({
            "id": "",
            "title": lines[0] if lines else "untitled",
            "priority": "medium",
            "target": "",
        })  # type: ignore[return-value]
    else:  # CompactSynthesis
        return CompactSynthesis.from_dict({
            "topic": lines[0] if lines else "unknown",
            "paper_count": 0,
            "key_findings": lines[1:6] if len(lines) > 1 else [],
        })  # type: ignore[return-value]
