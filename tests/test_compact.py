"""Tests for MCP compact response helpers."""

from khonliang.mcp.compact import (
    brief_or_full,
    compact_entry,
    compact_kv,
    compact_list,
    compact_summary,
    format_response,
    truncate,
)

# ---------------------------------------------------------------------------
# format_response (three-mode)
# ---------------------------------------------------------------------------


class TestFormatResponse:
    def test_compact_mode(self):
        result = format_response(
            compact_fn=lambda: "c=1|d=2",
            brief_fn=lambda: "brief output",
            full_fn=lambda: "full output",
            detail="compact",
        )
        assert result == "c=1|d=2"

    def test_brief_mode(self):
        result = format_response(
            compact_fn=lambda: "c=1",
            brief_fn=lambda: "brief output",
            full_fn=lambda: "full output",
            detail="brief",
        )
        assert result == "brief output"

    def test_full_mode(self):
        result = format_response(
            compact_fn=lambda: "c=1",
            brief_fn=lambda: "brief",
            full_fn=lambda: "full output",
            detail="full",
        )
        assert result == "full output"

    def test_compact_falls_back_to_brief(self):
        result = format_response(
            compact_fn=None,
            brief_fn=lambda: "brief fallback",
            full_fn=lambda: "full",
            detail="compact",
        )
        assert result == "brief fallback"

    def test_compact_falls_back_to_full(self):
        result = format_response(
            compact_fn=None,
            brief_fn=None,
            full_fn=lambda: "full fallback",
            detail="compact",
        )
        assert result == "full fallback"

    def test_brief_falls_back_to_compact(self):
        result = format_response(
            compact_fn=lambda: "compact fallback",
            brief_fn=None,
            full_fn=None,
            detail="brief",
        )
        assert result == "compact fallback"

    def test_full_falls_back_to_compact(self):
        result = format_response(
            compact_fn=lambda: "compact fallback",
            brief_fn=None,
            full_fn=None,
            detail="full",
        )
        assert result == "compact fallback"

    def test_all_none_returns_empty(self):
        result = format_response(detail="brief")
        assert result == ""

    def test_default_is_compact(self):
        result = format_response(
            compact_fn=lambda: "default compact",
            brief_fn=lambda: "brief",
            full_fn=lambda: "full",
        )
        assert result == "default compact"

    def test_default_falls_back_to_brief_when_no_compact(self):
        result = format_response(
            brief_fn=lambda: "brief fallback",
            full_fn=lambda: "full",
        )
        assert result == "brief fallback"


# ---------------------------------------------------------------------------
# brief_or_full (backward compat)
# ---------------------------------------------------------------------------


class TestBriefOrFull:
    def test_brief(self):
        result = brief_or_full(
            brief_fn=lambda: "brief",
            full_fn=lambda: "full",
            detail="brief",
        )
        assert result == "brief"

    def test_full(self):
        result = brief_or_full(
            brief_fn=lambda: "brief",
            full_fn=lambda: "full",
            detail="full",
        )
        assert result == "full"

    def test_default_is_brief(self):
        result = brief_or_full(
            brief_fn=lambda: "brief",
            full_fn=lambda: "full",
        )
        assert result == "brief"


# ---------------------------------------------------------------------------
# compact_summary
# ---------------------------------------------------------------------------


class TestCompactSummary:
    def test_basic(self):
        result = compact_summary({"caps": 158, "agents": 6})
        assert result == "caps=158|agents=6"

    def test_skips_none_and_empty(self):
        result = compact_summary({"a": 1, "b": "", "c": None, "e": "yes"})
        assert result == "a=1|e=yes"

    def test_preserves_zero(self):
        result = compact_summary({"hits": 0, "pending": 3})
        assert "hits=0" in result

    def test_preserves_false(self):
        result = compact_summary({"active": False, "count": 5})
        assert "active=False" in result

    def test_custom_separator(self):
        result = compact_summary({"a": 1, "b": 2}, sep=",")
        assert result == "a=1,b=2"

    def test_max_fields(self):
        data = {f"k{i}": i for i in range(1, 20)}
        result = compact_summary(data, max_fields=3)
        assert result.count("|") == 2  # 3 fields = 2 separators

    def test_deterministic_order(self):
        """Field order should match dict insertion order."""
        data = {"z": 1, "a": 2, "m": 3}
        result = compact_summary(data)
        assert result == "z=1|a=2|m=3"

    def test_escapes_pipe_in_value(self):
        result = compact_summary({"data": "a|b|c"})
        assert "|" not in result.split("=", 1)[1].split("|")[0]  # no raw pipe in value
        assert "¦" in result  # escaped

    def test_escapes_equals_in_value(self):
        result = compact_summary({"expr": "x=5"})
        assert "≈" in result  # escaped

    def test_escapes_pipe_in_key(self):
        result = compact_summary({"k|ey": "val"})
        assert "k¦ey" in result

    def test_empty_dict(self):
        assert compact_summary({}) == ""

    def test_string_values(self):
        result = compact_summary({"hotspots": "multi-agent:24,rl:11"})
        assert result == "hotspots=multi-agent:24,rl:11"


# ---------------------------------------------------------------------------
# compact_list
# ---------------------------------------------------------------------------


class TestCompactList:
    def test_basic(self):
        result = compact_list(
            items=[1, 2, 3],
            format_fn=lambda x: f"item {x}",
        )
        assert "item 1" in result
        assert "item 3" in result

    def test_with_header(self):
        result = compact_list(
            items=["a"], format_fn=str, header="Results:"
        )
        assert result.startswith("Results:")

    def test_limit(self):
        result = compact_list(
            items=list(range(20)),
            format_fn=str,
            limit=3,
        )
        assert "+17 more" in result

    def test_empty(self):
        result = compact_list(items=[], format_fn=str)
        assert result == "None found."

    def test_custom_empty_msg(self):
        result = compact_list(items=[], format_fn=str, empty_msg="nada")
        assert result == "nada"


# ---------------------------------------------------------------------------
# compact_entry
# ---------------------------------------------------------------------------


class TestCompactEntry:
    def test_basic(self):
        result = compact_entry("abc123", "My Title")
        assert result == "abc123 | My Title"

    def test_with_status_and_score(self):
        result = compact_entry("id1", "Title", status="active", score=0.95)
        assert "[active]" in result
        assert "(95%)" in result

    def test_with_preview(self):
        result = compact_entry("id1", "Title", preview="Some long preview text")
        assert "—" in result


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text(self):
        assert truncate("hello", 10) == "hello"

    def test_long_text(self):
        result = truncate("a" * 100, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_empty(self):
        assert truncate("") == ""

    def test_newlines_normalized(self):
        result = truncate("line1\nline2\r\nline3")
        assert "\n" not in result


# ---------------------------------------------------------------------------
# compact_kv
# ---------------------------------------------------------------------------


class TestCompactKV:
    def test_basic(self):
        result = compact_kv({"a": 1, "b": "two"})
        assert result == "a=1, b=two"

    def test_truncates_long_values(self):
        result = compact_kv({"k": "x" * 200}, max_value_len=10)
        assert len(result) < 50
