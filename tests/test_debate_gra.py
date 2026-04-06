"""Tests for debate/GRA framework — adjudicator, pipeline, orchestrator wiring."""

import pytest

from khonliang.consensus.models import AgentVote, ConsensusResult
from khonliang.debate.adjudicator import (
    AdjudicationResult,
    BaseAdjudicator,
    CriterionScore,
)
from khonliang.debate.orchestrator import DebateConfig, DebateOrchestrator
from khonliang.debate.pipeline import GRAPipeline, GRAResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vote(agent_id: str, action: str, confidence: float = 0.8) -> AgentVote:
    return AgentVote(
        agent_id=agent_id,
        action=action,
        confidence=confidence,
        reasoning=f"{agent_id} reasoning for {action}",
    )


class AlwaysApproveAdjudicator(BaseAdjudicator):
    """Test adjudicator that always returns APPROVE."""

    def adjudicate(self, votes, subject, context=None):
        return AdjudicationResult(
            action="APPROVE",
            confidence=0.85,
            reason="Test adjudicator: always approve",
            criteria={"test": 1.0},
        )


class ContextAwareAdjudicator(BaseAdjudicator):
    """Adjudicator that reads context to decide."""

    def adjudicate(self, votes, subject, context=None):
        if context and context.get("force_reject"):
            return AdjudicationResult(
                action="REJECT",
                confidence=0.9,
                reason="Context says reject",
                criteria={"context_signal": 1.0},
            )
        return AdjudicationResult(
            action="APPROVE",
            confidence=0.7,
            reason="Default approve",
            criteria={"default": 1.0},
        )


class StubAgent:
    """Minimal agent for debate testing."""

    def __init__(self, agent_id, reconsider_action=None):
        self.agent_id = agent_id
        self._reconsider_action = reconsider_action

    async def reconsider(self, original_vote, debate_context, round_num):
        if self._reconsider_action:
            return AgentVote(
                agent_id=self.agent_id,
                action=self._reconsider_action,
                confidence=original_vote.confidence + 0.05,
                reasoning=f"Reconsidered to {self._reconsider_action}",
            )
        return original_vote


# ---------------------------------------------------------------------------
# AdjudicationResult tests
# ---------------------------------------------------------------------------


class TestAdjudicationResult:
    def test_to_dict(self):
        result = AdjudicationResult(
            action="APPROVE",
            confidence=0.85,
            reason="test",
            criteria={"score": 1.0},
        )
        d = result.to_dict()
        assert d["action"] == "APPROVE"
        assert d["confidence"] == 0.85
        assert d["criteria"]["score"] == 1.0

    def test_to_consensus_result(self):
        votes = [_vote("a", "APPROVE"), _vote("b", "REJECT")]
        result = AdjudicationResult(
            action="APPROVE",
            confidence=0.85,
            reason="adjudicated",
            criteria={"score": 1.0},
        )
        cr = result.to_consensus_result(votes, debate_rounds=2)

        assert isinstance(cr, ConsensusResult)
        assert cr.action == "APPROVE"
        assert cr.debate_rounds == 2
        assert "[adjudicated]" in cr.reason
        assert len(cr.votes) == 2
        # scores should be per-action, not per-criterion
        assert cr.scores == {"APPROVE": 0.85}
        # criteria should be in reason string
        assert "score=1.00" in cr.reason


# ---------------------------------------------------------------------------
# BaseAdjudicator tests
# ---------------------------------------------------------------------------


class TestBaseAdjudicator:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseAdjudicator()

    def test_subclass_works(self):
        adj = AlwaysApproveAdjudicator()
        votes = [_vote("a", "APPROVE"), _vote("b", "REJECT")]
        result = adj.adjudicate(votes, "test subject")

        assert result.action == "APPROVE"
        assert result.confidence == 0.85

    def test_context_passed_through(self):
        adj = ContextAwareAdjudicator()
        votes = [_vote("a", "APPROVE")]

        result_default = adj.adjudicate(votes, "test", context=None)
        assert result_default.action == "APPROVE"

        result_reject = adj.adjudicate(votes, "test", context={"force_reject": True})
        assert result_reject.action == "REJECT"


# ---------------------------------------------------------------------------
# CriterionScore tests
# ---------------------------------------------------------------------------


class TestCriterionScore:
    def test_creation(self):
        cs = CriterionScore(name="RSI", action="BUY", score=1.5, reason="oversold")
        assert cs.name == "RSI"
        assert cs.score == 1.5


# ---------------------------------------------------------------------------
# DebateOrchestrator + adjudicator wiring
# ---------------------------------------------------------------------------


class TestDebateOrchestratorAdjudicator:
    @pytest.mark.asyncio
    async def test_no_disagreement_returns_none(self):
        orch = DebateOrchestrator(adjudicator=AlwaysApproveAdjudicator())
        votes = [_vote("a", "APPROVE", 0.9), _vote("b", "APPROVE", 0.8)]

        updated, adjudicated = await orch.run_debate(votes, "test")

        assert adjudicated is None
        assert len(updated) == 2

    @pytest.mark.asyncio
    async def test_resolved_debate_no_adjudication(self):
        """When agents agree after debate, adjudicator should not fire."""
        agents = {
            "a": StubAgent("a"),
            "b": StubAgent("b", reconsider_action="APPROVE"),
        }
        orch = DebateOrchestrator(
            agents=agents,
            adjudicator=AlwaysApproveAdjudicator(),
            config=DebateConfig(max_rounds=1),
        )
        votes = [_vote("a", "APPROVE", 0.9), _vote("b", "REJECT", 0.8)]

        updated, adjudicated = await orch.run_debate(votes, "test")

        # b reconsidered to APPROVE → resolved
        assert adjudicated is None

    @pytest.mark.asyncio
    async def test_unresolved_triggers_adjudicator(self):
        """When debate doesn't resolve, adjudicator should fire."""
        agents = {
            "a": StubAgent("a"),
            "b": StubAgent("b"),  # doesn't change mind
        }
        orch = DebateOrchestrator(
            agents=agents,
            adjudicator=AlwaysApproveAdjudicator(),
            config=DebateConfig(max_rounds=1),
        )
        votes = [_vote("a", "APPROVE", 0.9), _vote("b", "REJECT", 0.8)]

        updated, adjudicated = await orch.run_debate(votes, "test")

        assert adjudicated is not None
        assert adjudicated.action == "APPROVE"
        assert "[adjudicated]" in adjudicated.reason
        assert adjudicated.debate_rounds == 1

    @pytest.mark.asyncio
    async def test_no_adjudicator_returns_none(self):
        """Without adjudicator, unresolved debate returns None."""
        agents = {
            "a": StubAgent("a"),
            "b": StubAgent("b"),
        }
        orch = DebateOrchestrator(
            agents=agents,
            adjudicator=None,
            config=DebateConfig(max_rounds=1),
        )
        votes = [_vote("a", "APPROVE", 0.9), _vote("b", "REJECT", 0.8)]

        updated, adjudicated = await orch.run_debate(votes, "test")

        assert adjudicated is None

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self):
        orch = DebateOrchestrator(
            adjudicator=AlwaysApproveAdjudicator(),
            config=DebateConfig(enabled=False),
        )
        votes = [_vote("a", "APPROVE", 0.9), _vote("b", "REJECT", 0.8)]

        updated, adjudicated = await orch.run_debate(votes, "test")

        assert adjudicated is None
        assert updated == votes

    def test_stats_include_adjudication(self):
        orch = DebateOrchestrator(adjudicator=AlwaysApproveAdjudicator())
        orch._debate_history = [
            {"resolved": True, "adjudicated": False},
            {"resolved": False, "adjudicated": True},
            {"resolved": False, "adjudicated": False},
        ]
        stats = orch.get_stats()

        assert stats["resolved"] == 1
        assert stats["adjudicated"] == 1
        assert stats["unresolved"] == 1
        assert stats["adjudicator_enabled"] is True


# ---------------------------------------------------------------------------
# GRAPipeline tests
# ---------------------------------------------------------------------------


class TestGRAPipeline:
    @pytest.mark.asyncio
    async def test_consensus_path(self):
        async def generate(subject, context=None):
            return {"verdict": "match", "confidence": 0.9, "reasoning": "strong"}

        async def review(subject, assessment, context=None):
            return {"agrees": True, "verdict": "match", "confidence": 0.85}

        pipeline = GRAPipeline(
            generate_fn=generate,
            review_fn=review,
            adjudicator=AlwaysApproveAdjudicator(),
        )

        result = await pipeline.evaluate("test subject")

        assert result.resolved_by == "consensus"
        assert result.verdict == "match"
        assert result.adjudication is None
        assert 0.87 <= result.confidence <= 0.88  # avg of 0.9 and 0.85

    @pytest.mark.asyncio
    async def test_adjudicator_path(self):
        async def generate(subject, context=None):
            return {"verdict": "match", "confidence": 0.9, "reasoning": "strong"}

        async def review(subject, assessment, context=None):
            return {"agrees": False, "verdict": "no_match", "confidence": 0.7}

        pipeline = GRAPipeline(
            generate_fn=generate,
            review_fn=review,
            adjudicator=AlwaysApproveAdjudicator(),
        )

        result = await pipeline.evaluate("test subject")

        assert result.resolved_by == "adjudicator"
        assert result.verdict == "APPROVE"
        assert result.adjudication is not None
        assert result.adjudication.action == "APPROVE"

    @pytest.mark.asyncio
    async def test_context_passed_to_all_stages(self):
        ctx_log = []

        async def generate(subject, context=None):
            ctx_log.append(("generate", context))
            return {"verdict": "yes", "confidence": 0.8, "reasoning": "ok"}

        async def review(subject, assessment, context=None):
            ctx_log.append(("review", context))
            return {"agrees": False, "verdict": "no", "confidence": 0.7}

        pipeline = GRAPipeline(
            generate_fn=generate,
            review_fn=review,
            adjudicator=ContextAwareAdjudicator(),
        )

        result = await pipeline.evaluate("test", context={"force_reject": True})

        assert ctx_log[0] == ("generate", {"force_reject": True})
        assert ctx_log[1] == ("review", {"force_reject": True})
        assert result.verdict == "REJECT"

    @pytest.mark.asyncio
    async def test_generator_only_path(self):
        async def generate(subject, context=None):
            return {"verdict": "go", "confidence": 0.9, "reasoning": "ok"}

        pipeline = GRAPipeline(generate_fn=generate)

        result = await pipeline.evaluate("test")

        assert result.resolved_by == "generator"
        assert result.verdict == "go"
        assert result.review is None
        assert result.adjudication is None

    @pytest.mark.asyncio
    async def test_no_adjudicator_returns_reviewer_verdict(self):
        async def generate(subject, context=None):
            return {"verdict": "yes", "confidence": 0.9, "reasoning": "ok"}

        async def review(subject, assessment, context=None):
            return {"agrees": False, "verdict": "no", "confidence": 0.7}

        pipeline = GRAPipeline(generate_fn=generate, review_fn=review)

        result = await pipeline.evaluate("test")

        assert result.resolved_by == "reviewer"
        assert result.verdict == "no"
        assert result.adjudication is None

    @pytest.mark.asyncio
    async def test_custom_keys(self):
        async def generate(subject, context=None):
            return {"decision": "go", "score": 0.9, "reasoning": "ok"}

        async def review(subject, assessment, context=None):
            return {"ok": True, "decision": "go", "score": 0.85}

        pipeline = GRAPipeline(
            generate_fn=generate,
            review_fn=review,
            adjudicator=AlwaysApproveAdjudicator(),
            verdict_key="decision",
            agrees_key="ok",
            confidence_key="score",
        )

        result = await pipeline.evaluate("test")

        assert result.resolved_by == "consensus"
        assert result.verdict == "go"


class TestGRAResult:
    def test_to_dict(self):
        result = GRAResult(
            verdict="match",
            confidence=0.9,
            assessment={"verdict": "match"},
            review=None,
            adjudication=None,
            resolved_by="consensus",
        )
        d = result.to_dict()
        assert d["verdict"] == "match"
        assert d["resolved_by"] == "consensus"
        assert d["review"] is None
        assert d["adjudication"] is None

    def test_to_dict_with_adjudication(self):
        adj = AdjudicationResult(
            action="APPROVE", confidence=0.85, reason="test"
        )
        result = GRAResult(
            verdict="APPROVE",
            confidence=0.85,
            assessment={"verdict": "match"},
            review={"agrees": False},
            adjudication=adj,
            resolved_by="adjudicator",
        )
        d = result.to_dict()
        assert d["adjudication"]["action"] == "APPROVE"
