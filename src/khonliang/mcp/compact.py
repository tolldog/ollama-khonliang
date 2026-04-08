"""
Compact MCP response helpers — minimize token usage in agent-facing tools.

Problem: MCP tools that return verbose markdown eat agent context windows.
Solution: three response modes with compact-by-default for agent loops.

Response modes:
    compact — key=value pairs, pipe-delimited, no prose. ~10 fields max.
              For agent control loops where every token costs money.
    brief   — structured one-line-per-item with small headers.
              For monitoring and human-scannable output.
    full    — rich detail with context. For humans.

Usage:

    from khonliang.mcp.compact import format_response, compact_summary

    @mcp.tool()
    def my_status(detail: str = "compact") -> str:
        return format_response(
            compact_fn=lambda: compact_summary({
                "agents": 6, "active": 3, "pending": 12,
            }),
            brief_fn=lambda: "6 agents, 3 active, 12 tasks pending",
            full_fn=lambda: render_full_status(),
            detail=detail,
        )

Design principles:
    1. Compact by default for agent loops — data only, no prose
    2. Brief for monitoring — structured, one-line-per-item
    3. Full for humans — rich detail with context
    4. Deterministic field order — dict insertion order preserved
    5. Bounded field count — compact mode defaults to max 10 fields
    6. Separator-safe — values containing | or = are escaped
    7. Caveman rule — every token costs money. Data only.
"""

from typing import Any, Callable, List, Optional

# Default limits for MCP tools
DEFAULT_LIST_LIMIT = 10
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_PREVIEW_CHARS = 80

# Separator characters for compact_summary
_FIELD_SEP = "|"
_KV_SEP = "="


def format_response(
    compact_fn: Optional[Callable[[], str]] = None,
    brief_fn: Optional[Callable[[], str]] = None,
    full_fn: Optional[Callable[[], str]] = None,
    detail: str = "compact",
) -> str:
    """Switch between compact/brief/full output modes.

    Args:
        compact_fn: Returns dense key=value string for agent loops
        brief_fn: Returns structured one-line-per-item output
        full_fn: Returns rich detail for humans
        detail: "compact", "brief", or "full"

    Defaults to compact — external agents pay per token.
    Falls back gracefully based on the requested detail:
        - compact -> brief -> full
        - brief -> compact -> full
        - full -> brief -> compact
    """
    if detail == "compact":
        if compact_fn is not None:
            return compact_fn()
        if brief_fn is not None:
            return brief_fn()
        if full_fn is not None:
            return full_fn()
    elif detail == "full":
        if full_fn is not None:
            return full_fn()
        if brief_fn is not None:
            return brief_fn()
        if compact_fn is not None:
            return compact_fn()
    else:  # brief
        if brief_fn is not None:
            return brief_fn()
        if compact_fn is not None:
            return compact_fn()
        if full_fn is not None:
            return full_fn()
    return ""


def brief_or_full(
    brief_fn: Callable[[], str],
    full_fn: Callable[[], str],
    detail: str = "brief",
) -> str:
    """Switch between brief/full output. Backward-compatible alias.

    .. deprecated::
        Use :func:`format_response` which also supports compact mode.
    """
    import warnings

    warnings.warn(
        "brief_or_full() is deprecated — use format_response() "
        "with compact_fn/brief_fn/full_fn instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return format_response(brief_fn=brief_fn, full_fn=full_fn, detail=detail)


def compact_summary(
    data: dict,
    sep: str = _FIELD_SEP,
    max_fields: int = 10,
) -> str:
    """Format dict as dense key=value string for agent consumption.

    Values containing separators are escaped. Empty/None values are skipped.
    Field order follows dict insertion order (deterministic in Python 3.7+).

    Args:
        data: Key-value pairs to format
        sep: Field separator (default "|")
        max_fields: Maximum fields to include

    Example:
        >>> compact_summary({"caps": 158, "hotspots": "multi-agent:24,rl:11"})
        'caps=158|hotspots=multi-agent:24,rl:11'
    """
    parts = []
    for k, v in data.items():
        if len(parts) >= max_fields:
            break
        if v is None or v == "":
            continue
        parts.append(
            f"{_escape_value(str(k), sep)}{_KV_SEP}{_escape_value(str(v), sep)}"
        )
    return sep.join(parts)


def _escape_value(value: str, field_sep: str = _FIELD_SEP) -> str:
    """Escape separator characters in values.

    Replaces the field separator with ¦ and = with ≈ to prevent
    parsing ambiguity.
    """
    result = value.replace(field_sep, "¦")
    return result.replace(_KV_SEP, "≈")


def compact_list(
    items: List[Any],
    format_fn: Callable[[Any], str],
    header: str = "",
    limit: int = DEFAULT_LIST_LIMIT,
    empty_msg: str = "None found.",
) -> str:
    """Format a list compactly. One line per item via format_fn."""
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
    """Format a single entry as "id | title [status] (score) — preview"."""
    line = f"{entry_id} | {title}"
    if status:
        line += f" [{status}]"
    if score > 0:
        line += f" ({score:.0%})"
    if preview:
        line += f" — {truncate(preview, max_preview)}"
    return line


def truncate(text: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Truncate text to max_chars with ellipsis. Normalizes newlines first."""
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
    if max_chars < 4:
        return text[:max_chars]
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compact_kv(data: dict, max_value_len: int = 60) -> str:
    """Format a dict as compact key=value pairs on one line.

    Uses comma separator. For pipe-delimited format, use compact_summary().
    """
    parts = []
    for k, v in data.items():
        parts.append(f"{k}={truncate(str(v), max_value_len)}")
    return ", ".join(parts)
