"""Tests for MCP post-compression helper."""

import json

import pytest

from khonliang.mcp.artifacts import CompactConcept, CompactFR, CompactSynthesis
from khonliang.mcp.budget import ContextBudget
from khonliang.mcp.compress import compress_for_agent, compress_rule_based, _try_parse_json


class TestTryParseJson:
    def test_valid_json(self):
        assert _try_parse_json('{"a": 1}') == {"a": 1}

    def test_json_in_markdown(self):
        text = '```json\n{"a": 1}\n```'
        assert _try_parse_json(text) == {"a": 1}

    def test_json_in_generic_fence(self):
        text = '```\n{"a": 1}\n```'
        assert _try_parse_json(text) == {"a": 1}

    def test_bare_json_in_text(self):
        text = 'Here is the result: {"a": 1} and more text'
        assert _try_parse_json(text) == {"a": 1}

    def test_no_json(self):
        assert _try_parse_json("just plain text") is None

    def test_array_not_dict(self):
        assert _try_parse_json("[1, 2, 3]") is None

    def test_invalid_json(self):
        assert _try_parse_json("{not: valid}") is None


class TestCompressRuleBased:
    def test_concept_from_json(self):
        data = json.dumps({
            "name": "rl",
            "relevance": 0.8,
            "paper_count": 3,
            "top_paper": "PG Methods",
            "actionable": True,
        })
        result = compress_rule_based(data, CompactConcept)
        assert result.name == "rl"
        assert result.relevance == 0.8

    def test_concept_from_plain_text(self):
        result = compress_rule_based("reinforcement learning\nsome paper title", CompactConcept)
        assert result.name == "reinforcement learning"
        assert result.top_paper == "some paper title"

    def test_fr_from_json(self):
        data = json.dumps({
            "id": "fr_abc",
            "title": "Add cache",
            "priority": "high",
            "target": "khonliang",
        })
        result = compress_rule_based(data, CompactFR)
        assert result.id == "fr_abc"
        assert result.priority == "high"

    def test_fr_from_plain_text(self):
        result = compress_rule_based("Add embedding cache", CompactFR)
        assert result.title == "Add embedding cache"
        assert result.priority == "medium"  # default

    def test_synthesis_from_json(self):
        data = json.dumps({
            "topic": "test",
            "paper_count": 5,
            "key_findings": ["finding 1"],
        })
        result = compress_rule_based(data, CompactSynthesis)
        assert result.topic == "test"
        assert result.paper_count == 5

    def test_synthesis_from_plain_text(self):
        result = compress_rule_based("token optimization\nfinding one\nfinding two", CompactSynthesis)
        assert result.topic == "token optimization"
        assert len(result.key_findings) == 2

    def test_with_budget(self):
        data = json.dumps({
            "topic": "test",
            "paper_count": 5,
            "key_findings": [f"f{i}" for i in range(20)],
        })
        budget = ContextBudget(max_items=3)
        result = compress_rule_based(data, CompactSynthesis, budget)
        assert len(result.key_findings) <= 5  # capped by from_dict

    def test_empty_text(self):
        result = compress_rule_based("", CompactConcept)
        assert result.name == "unknown"


class TestCompressForAgent:
    @pytest.mark.asyncio
    async def test_json_input_skips_model(self):
        """When input is valid JSON, no model call needed."""
        data = json.dumps({
            "name": "rl",
            "relevance": 0.8,
            "paper_count": 3,
            "top_paper": "PG",
        })
        result = await compress_for_agent(data, CompactConcept)
        assert result.name == "rl"

    @pytest.mark.asyncio
    async def test_plain_text_falls_back_to_rules(self):
        """When model is unavailable, falls back to rule-based."""
        result = await compress_for_agent(
            "some concept\nsome paper",
            CompactConcept,
            base_url="http://localhost:99999",  # unreachable
        )
        assert result.name == "some concept"
