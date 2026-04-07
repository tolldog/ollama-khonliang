"""Tests for MCP context budget framework."""

from khonliang.mcp.budget import (
    BUDGET_BRIEF,
    BUDGET_COMPACT,
    BUDGET_FULL,
    ContextBudget,
    fit_to_budget,
)


class TestContextBudget:
    def test_defaults(self):
        b = ContextBudget()
        assert b.max_tokens == 500
        assert b.max_items == 10
        assert b.max_preview_chars == 80
        assert b.priority_field == "score"

    def test_presets_exist(self):
        assert BUDGET_COMPACT.max_items == 5
        assert BUDGET_BRIEF.max_items == 10
        assert BUDGET_FULL.max_items == 25


class TestFitToBudget:
    def test_empty_list(self):
        assert fit_to_budget([], ContextBudget()) == []

    def test_truncates_to_max_items(self):
        items = [{"name": f"item_{i}", "score": i} for i in range(20)]
        result = fit_to_budget(items, ContextBudget(max_items=5))
        assert len(result) == 5

    def test_sorts_by_priority_field(self):
        items = [
            {"name": "low", "score": 0.1},
            {"name": "high", "score": 0.9},
            {"name": "mid", "score": 0.5},
        ]
        result = fit_to_budget(items, ContextBudget(max_items=10, priority_field="score"))
        assert result[0]["name"] == "high"
        assert result[-1]["name"] == "low"

    def test_truncates_preview_fields(self):
        items = [{"name": "x", "score": 1, "description": "a" * 200}]
        result = fit_to_budget(items, ContextBudget(max_preview_chars=20))
        assert len(result[0]["description"]) == 20
        assert result[0]["description"].endswith("...")

    def test_custom_preview_fields(self):
        items = [{"name": "x", "score": 1, "body": "a" * 200}]
        result = fit_to_budget(
            items, ContextBudget(max_preview_chars=30), preview_fields=["body"]
        )
        assert len(result[0]["body"]) == 30

    def test_does_not_mutate_originals(self):
        original = {"name": "x", "score": 1, "description": "a" * 200}
        items = [original]
        fit_to_budget(items, ContextBudget(max_preview_chars=20))
        assert len(original["description"]) == 200

    def test_missing_priority_field_goes_last(self):
        items = [
            {"name": "no_score"},
            {"name": "has_score", "score": 0.5},
        ]
        result = fit_to_budget(items, ContextBudget(priority_field="score"))
        assert result[0]["name"] == "has_score"

    def test_non_string_preview_fields_untouched(self):
        items = [{"name": "x", "score": 1, "description": 42}]
        result = fit_to_budget(items, ContextBudget(max_preview_chars=5))
        assert result[0]["description"] == 42
