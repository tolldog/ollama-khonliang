"""Tests for MCP compressed artifact types."""

from khonliang.mcp.artifacts import CompactConcept, CompactFR, CompactSynthesis


class TestCompactConcept:
    def test_to_compact(self):
        c = CompactConcept("rl", 0.82, 5, "Policy Gradient Methods", True)
        result = c.to_compact()
        assert "rl|0.82|5|Policy Gradient Methods|yes" == result

    def test_to_brief(self):
        c = CompactConcept("rl", 0.82, 5, "Policy Gradient Methods", True)
        result = c.to_brief()
        assert "rl (82%)" in result
        assert "[actionable]" in result

    def test_to_brief_not_actionable(self):
        c = CompactConcept("rl", 0.5, 3, "Some Paper", False)
        assert "[actionable]" not in c.to_brief()

    def test_from_dict(self):
        c = CompactConcept.from_dict({
            "name": "rl",
            "relevance": 0.82,
            "paper_count": 5,
            "top_paper": "PG Methods",
            "actionable": True,
        })
        assert c.name == "rl"
        assert c.relevance == 0.82
        assert c.actionable is True

    def test_from_dict_defaults(self):
        c = CompactConcept.from_dict({})
        assert c.name == ""
        assert c.relevance == 0.0
        assert c.actionable is False


class TestCompactFR:
    def test_to_compact(self):
        fr = CompactFR("fr_abc", "Add cache", "high", "khonliang", "caching", ["fr_xyz"])
        result = fr.to_compact()
        assert "fr_abc|high|khonliang|Add cache|deps=fr_xyz" == result

    def test_to_compact_no_deps(self):
        fr = CompactFR("fr_abc", "Add cache", "high", "khonliang")
        assert "deps=none" in fr.to_compact()

    def test_to_brief(self):
        fr = CompactFR("fr_abc", "Add cache", "high", "khonliang", "caching", ["fr_xyz"])
        result = fr.to_brief()
        assert "[high]" in result
        assert "-> khonliang" in result
        assert "[caching]" in result
        assert "blocks: fr_xyz" in result

    def test_from_dict(self):
        fr = CompactFR.from_dict({
            "id": "fr_abc",
            "title": "Add cache",
            "priority": "high",
            "target": "khonliang",
            "depends_on": ["fr_1", "fr_2"],
        })
        assert fr.id == "fr_abc"
        assert len(fr.depends_on) == 2

    def test_from_dict_string_deps(self):
        fr = CompactFR.from_dict({
            "id": "fr_abc",
            "title": "X",
            "priority": "low",
            "target": "t",
            "depends_on": "fr_1, fr_2",
        })
        assert fr.depends_on == ["fr_1", "fr_2"]


class TestCompactSynthesis:
    def test_to_compact(self):
        s = CompactSynthesis(
            topic="token opt",
            paper_count=3,
            key_findings=["finding 1", "finding 2"],
            relevance={"autostock": 0.4, "khonliang": 0.9},
            suggested_frs=["fr_abc"],
        )
        result = s.to_compact()
        assert "token opt|3|" in result
        assert "finding 1; finding 2" in result
        assert "khonliang:0.90" in result
        assert "frs=fr_abc" in result

    def test_to_compact_no_frs(self):
        s = CompactSynthesis("t", 1)
        assert "frs=none" in s.to_compact()

    def test_to_brief(self):
        s = CompactSynthesis(
            topic="token opt",
            paper_count=3,
            key_findings=["finding 1"],
            relevance={"khonliang": 0.9},
            suggested_frs=["fr_abc"],
        )
        result = s.to_brief()
        assert "token opt (3 papers)" in result
        assert "- finding 1" in result
        assert "khonliang: 90%" in result
        assert "FRs: fr_abc" in result

    def test_key_findings_capped_at_5(self):
        s = CompactSynthesis.from_dict({
            "topic": "t",
            "paper_count": 1,
            "key_findings": [f"f{i}" for i in range(10)],
        })
        assert len(s.key_findings) == 5

    def test_from_dict(self):
        s = CompactSynthesis.from_dict({
            "topic": "test",
            "paper_count": 2,
            "key_findings": ["a", "b"],
            "relevance": {"proj": 0.5},
            "suggested_frs": ["fr_1"],
        })
        assert s.topic == "test"
        assert s.relevance == {"proj": 0.5}
