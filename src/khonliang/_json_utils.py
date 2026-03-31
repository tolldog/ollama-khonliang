"""
Shared JSON cleanup utilities for LLM responses.

LLMs frequently produce invalid JSON — Python booleans (True/False/None),
trailing commas, markdown fences, comments. These utilities clean common
issues before parsing.
"""

import json
import re
from typing import Dict


def clean_llm_json(text: str) -> str:
    """Clean common LLM JSON issues.

    Handles: markdown fences, Python booleans, trailing commas,
    leading/trailing non-JSON content.
    """
    text = text.strip()

    # Strip markdown fences
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence) :]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Extract JSON object/array if surrounded by non-JSON text
    first = min((text.find(c) for c in "{[" if text.find(c) >= 0), default=0)
    last = max(text.rfind("}"), text.rfind("]"))
    if last >= 0:
        text = text[first : last + 1]

    # Fix Python-style literals
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # Remove trailing commas before closing brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


def parse_llm_json(text: str) -> Dict:
    """Parse JSON from an LLM response, cleaning common issues.

    Tries raw parse first, falls back to cleanup if needed.

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    text = text.strip()

    # Strip markdown fences for first attempt
    stripped = text
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence) :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    stripped = stripped.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Full cleanup
    cleaned = clean_llm_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {e}") from e
