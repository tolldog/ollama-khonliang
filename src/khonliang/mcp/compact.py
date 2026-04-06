"""
Compact MCP response helpers — minimize token usage in agent-facing tools.

Problem: MCP tools that return verbose markdown eat agent context windows.
An agent calling reading_list + paper_context + feature_requests can burn
thousands of tokens on formatting before doing any real work.

Solution: compact-by-default responses with opt-in expansion.

Usage in MCP tools:

    from khonliang.mcp.compact import compact_list, compact_entry, truncate

    @mcp.tool()
    def my_search(query: str, limit: int = 5, detail: str = "brief") -> str:
        results = store.search(query, limit=limit)
        if detail == "full":
            return full_format(results)
        return compact_list(
            items=results,
            format_fn=lambda r: f"{r.id} | {r.title}",
            header=f"{len(results)} results",
        )

Design principles:
    1. Brief by default — IDs + titles + scores, not full content
    2. Explicit expansion — detail="full" to get rich output
    3. Consistent limits — small defaults (5-10), not 50-100
    4. Structured over narrative — "id | title | score" not paragraphs
    5. Truncation-safe — always cap per-item length
"""

from typing import Any, Callable, List

# Default limits for MCP tools
DEFAULT_LIST_LIMIT = 10
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CONTEXT_CHARS = 2000
DEFAULT_PREVIEW_CHARS = 80


def compact_list(
    items: List[Any],
    format_fn: Callable[[Any], str],
    header: str = "",
    limit: int = DEFAULT_LIST_LIMIT,
    empty_msg: str = "None found.",
) -> str:
    """Format a list of items compactly for agent consumption.

    Each item rendered as one line via format_fn. Shows count + items.

    Args:
        items: List of objects to format
        format_fn: Callable that produces one line per item
        header: Optional header line (e.g., "5 results for 'query'")
        limit: Max items to show
        empty_msg: Message when items is empty
    """
    if not items:
        return empty_msg

    lines = []
    if header:
        lines.append(header)

    for item in items[:limit]:
        lines.append(format_fn(item))

    remaining = len(items) - limit
    if remaining > 0:
        lines.append(f"... +{remaining} more")

    return "\n".join(lines)


def compact_entry(
    entry_id: str,
    title: str,
    status: str = "",
    score: float = 0.0,
    preview: str = "",
    max_preview: int = DEFAULT_PREVIEW_CHARS,
) -> str:
    """Format a single entry as a compact one-liner.

    Format: "id | title [status] (score)"
    """
    parts = [entry_id, title]
    if status:
        parts.append(f"[{status}]")
    if score > 0:
        parts.append(f"({score:.0%})")
    line = " | ".join(parts[:2])
    if status:
        line += f" [{status}]"
    if score > 0:
        line += f" ({score:.0%})"
    if preview:
        line += f" — {truncate(preview, max_preview)}"
    return line


def truncate(text: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Truncate text to max_chars, adding ellipsis if truncated."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compact_kv(data: dict, max_value_len: int = 60) -> str:
    """Format a dict as compact key=value pairs on one line."""
    parts = []
    for k, v in data.items():
        v_str = str(v)
        if len(v_str) > max_value_len:
            v_str = v_str[:max_value_len - 3] + "..."
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


def brief_or_full(
    brief_fn: Callable[[], str],
    full_fn: Callable[[], str],
    detail: str = "brief",
) -> str:
    """Switch between brief and full output based on detail parameter.

    Args:
        brief_fn: Callable returning compact output
        full_fn: Callable returning rich output
        detail: "brief" (default) or "full"
    """
    if detail == "full":
        return full_fn()
    return brief_fn()
