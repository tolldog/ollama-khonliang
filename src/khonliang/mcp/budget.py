"""Context budget framework for MCP tool outputs.

External coding agents (Claude, Codex) pay per token. MCP tools should
declare an output budget and mechanically trim results to fit.

Usage:

    budget = ContextBudget(max_items=5, max_preview_chars=60)
    trimmed = fit_to_budget(items, budget)
"""

from dataclasses import dataclass
from typing import Any

from khonliang.mcp.compact import truncate


@dataclass
class ContextBudget:
    """Declares the output ceiling for an MCP tool response.

    Attributes:
        max_tokens: Rough target ceiling (chars / 4 estimate).
        max_items: Maximum list items to return.
        max_preview_chars: Per-item preview truncation length.
        priority_field: Sort/trim by this field (descending).
    """

    max_tokens: int = 500
    max_items: int = 10
    max_preview_chars: int = 80
    priority_field: str = "score"


# Presets for common tool patterns
BUDGET_COMPACT = ContextBudget(max_tokens=300, max_items=5, max_preview_chars=40)
BUDGET_BRIEF = ContextBudget(max_tokens=800, max_items=10, max_preview_chars=80)
BUDGET_FULL = ContextBudget(max_tokens=2000, max_items=25, max_preview_chars=200)


def fit_to_budget(
    items: list[dict[str, Any]],
    budget: ContextBudget,
    preview_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Sort by priority, truncate to budget, trim preview fields.

    Args:
        items: List of dicts to constrain.
        budget: The output ceiling to enforce.
        preview_fields: Dict keys whose string values should be truncated
            to ``budget.max_preview_chars``. Defaults to
            ``["preview", "description", "summary"]``.

    Returns:
        A new list of dicts, sorted and trimmed. Original dicts are not mutated.
    """
    if not items:
        return []

    preview_fields = preview_fields or ["preview", "description", "summary"]

    # Sort by priority field (descending), missing values go last
    pf = budget.priority_field
    sorted_items = sorted(
        items,
        key=lambda x: (pf in x, x.get(pf, 0) if isinstance(x.get(pf), (int, float)) else 0),
        reverse=True,
    )

    # Truncate list length
    trimmed = sorted_items[: budget.max_items]

    # Trim preview fields
    result = []
    for item in trimmed:
        row = dict(item)  # shallow copy
        for pfield in preview_fields:
            if pfield in row and isinstance(row[pfield], str):
                row[pfield] = truncate(row[pfield], budget.max_preview_chars)
        result.append(row)

    return result
