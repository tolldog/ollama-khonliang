"""Tests for OutcomeTracker and vote validation (KH-1, KH-6)."""

import os
import tempfile

from khonliang.consensus.engine import ConsensusEngine
from khonliang.consensus.models import AgentVote, ConsensusResult
from khonliang.consensus.outcomes import OutcomeTracker


def _temp_tracker():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return OutcomeTracker(path), path


def _sample_votes():
    return [
        AgentVote(agent_id="analyst", action="APPROVE", confidence=0.8,
                  reasoning="Strong evidence supports approval"),
        AgentVote(agent_id="reviewer", action="REJECT", confidence=0.6,
                  reasoning="Minor concerns about methodology"),
        AgentVote(agent_id="skeptic", action="APPROVE", confidence=0.7,
                  reasoning="Acceptable with caveats"),
    ]


def _sample_result(votes=None):
    votes = votes or _sample_votes()
    return ConsensusResult(
        action="APPROVE", confidence=0.72, votes=votes,
        scores={"APPROVE": 0.72, "REJECT": 0.28},
    )


# === OutcomeTracker Tests ===


class TestRecordConsensus:
    def test_basic_record(self):
        tracker, path = _temp_tracker()
        result = _sample_result()
        cid = tracker.record_consensus(result, subject="task-42")
        assert cid
        assert len(cid) > 0
        os.unlink(path)

    def test_custom_consensus_id(self):
        tracker, path = _temp_tracker()
        result = _sample_result()
        cid = tracker.record_consensus(result, consensus_id="custom-123")
        assert cid == "custom-123"
        os.unlink(path)

    def test_duplicate_consensus_ignored(self):
        """INSERT OR IGNORE should not overwrite existing record."""
        tracker, path = _temp_tracker()
        result = _sample_result()
        cid = tracker.record_consensus(result, consensus_id="dup-1", subject="first")

        # Record outcome on first
        tracker.record_outcome(cid, outcome=0.5)

        # Try to re-record consensus with same ID
        tracker.record_consensus(result, consensus_id="dup-1", subject="second")

        # Outcome should still be preserved
        history = tracker.get_history(with_outcome_only=True)
        assert len(history) == 1
        assert history[0].outcome == 0.5
        os.unlink(path)


class TestRecordOutcome:
    def test_basic_outcome(self):
        tracker, path = _temp_tracker()
        cid = tracker.record_consensus(_sample_result(), subject="task-1")
        found = tracker.record_outcome(cid, outcome=0.85, metadata={"reason": "success"})
        assert found is True
        os.unlink(path)

    def test_outcome_not_found(self):
        tracker, path = _temp_tracker()
        found = tracker.record_outcome("nonexistent", outcome=0.5)
        assert found is False
        os.unlink(path)

    def test_idempotent_outcome(self):
        """Recording outcome twice should succeed (update)."""
        tracker, path = _temp_tracker()
        cid = tracker.record_consensus(_sample_result())
        tracker.record_outcome(cid, outcome=0.5)
        found = tracker.record_outcome(cid, outcome=0.8)
        assert found is True
        history = tracker.get_history(with_outcome_only=True)
        assert history[0].outcome == 0.8
        os.unlink(path)


class TestGetHistory:
    def test_empty(self):
        tracker, path = _temp_tracker()
        assert tracker.get_history() == []
        os.unlink(path)

    def test_with_outcome_filter(self):
        tracker, path = _temp_tracker()
        cid1 = tracker.record_consensus(_sample_result(), subject="a")
        tracker.record_consensus(_sample_result(), subject="b")
        tracker.record_outcome(cid1, outcome=0.5)

        all_records = tracker.get_history()
        assert len(all_records) == 2

        with_outcome = tracker.get_history(with_outcome_only=True)
        assert len(with_outcome) == 1
        assert with_outcome[0].consensus_id == cid1
        os.unlink(path)

    def test_agent_filter(self):
        tracker, path = _temp_tracker()
        tracker.record_consensus(_sample_result(), subject="a")
        tracker.record_outcome(
            tracker.record_consensus(_sample_result(), subject="b"),
            outcome=0.5,
        )

        analyst_history = tracker.get_history(agent_id="analyst")
        assert len(analyst_history) == 2

        # Non-existent agent
        nobody = tracker.get_history(agent_id="nobody")
        assert len(nobody) == 0
        os.unlink(path)

    def test_action_filter(self):
        tracker, path = _temp_tracker()
        tracker.record_consensus(_sample_result())

        approve = tracker.get_history(action="APPROVE")
        assert len(approve) == 1

        reject = tracker.get_history(action="REJECT")
        assert len(reject) == 0
        os.unlink(path)


class TestAgentStats:
    def test_aligned_vs_opposed(self):
        tracker, path = _temp_tracker()
        cid = tracker.record_consensus(_sample_result())
        tracker.record_outcome(cid, outcome=0.1)

        # analyst voted APPROVE, consensus was APPROVE → aligned
        stats = tracker.get_agent_stats("analyst")
        assert stats["aligned_count"] == 1
        assert stats["opposed_count"] == 0
        assert stats["mean_outcome_aligned"] == 0.1

        # reviewer voted REJECT, consensus was APPROVE → opposed
        stats2 = tracker.get_agent_stats("reviewer")
        assert stats2["aligned_count"] == 0
        assert stats2["opposed_count"] == 1
        assert stats2["mean_outcome_opposed"] == 0.1
        os.unlink(path)

    def test_no_data(self):
        tracker, path = _temp_tracker()
        stats = tracker.get_agent_stats("nobody")
        assert stats["sample_count"] == 0
        os.unlink(path)


class TestOutcomeRecord:
    def test_agent_voted(self):
        tracker, path = _temp_tracker()
        cid = tracker.record_consensus(_sample_result())
        tracker.record_outcome(cid, outcome=0.5)
        record = tracker.get_history()[0]

        vote = record.agent_voted("analyst")
        assert vote is not None
        assert vote["action"] == "APPROVE"

        assert record.agent_voted("nobody") is None
        os.unlink(path)

    def test_agents_who_voted(self):
        tracker, path = _temp_tracker()
        tracker.record_consensus(_sample_result())
        record = tracker.get_history()[0]

        approvers = record.agents_who_voted("APPROVE")
        assert "analyst" in approvers
        assert "skeptic" in approvers
        assert "reviewer" not in approvers
        os.unlink(path)


class TestOverallStats:
    def test_stats(self):
        tracker, path = _temp_tracker()
        cid1 = tracker.record_consensus(_sample_result())
        tracker.record_consensus(_sample_result())
        tracker.record_outcome(cid1, outcome=0.5)

        stats = tracker.get_stats()
        assert stats["total_consensuses"] == 2
        assert stats["with_outcomes"] == 1
        assert stats["pending_outcomes"] == 1
        os.unlink(path)


# === Vote Validation Tests (KH-6) ===


class TestVoteValidation:
    def setup_method(self):
        self.engine = ConsensusEngine()

    def test_clean_votes_pass(self):
        votes = [
            AgentVote(agent_id="a", action="BUY", confidence=0.8,
                      reasoning="RSI oversold, bullish MACD crossover"),
        ]
        assert self.engine.validate_votes(votes) == []

    def test_empty_reasoning(self):
        votes = [
            AgentVote(agent_id="a", action="BUY", confidence=0.8, reasoning=""),
        ]
        issues = self.engine.validate_votes(votes)
        assert len(issues) == 1
        assert issues[0].issue_type == "empty_reasoning"

    def test_zero_confidence(self):
        votes = [
            AgentVote(agent_id="a", action="HOLD", confidence=0.0,
                      reasoning="Not sure about this"),
        ]
        issues = self.engine.validate_votes(votes)
        assert any(i.issue_type == "zero_confidence" for i in issues)

    def test_zero_confidence_veto_ok(self):
        """VETO with 0.0 confidence should not be flagged."""
        votes = [
            AgentVote(agent_id="a", action="VETO", confidence=0.0,
                      reasoning="Blocked for compliance"),
        ]
        issues = self.engine.validate_votes(votes)
        assert not any(i.issue_type == "zero_confidence" for i in issues)

    def test_reasoning_action_mismatch_bearish_buy(self):
        votes = [
            AgentVote(agent_id="a", action="BUY", confidence=0.8,
                      reasoning="Bearish divergence, overbought conditions, "
                                "resistance level approaching"),
        ]
        issues = self.engine.validate_votes(votes)
        assert len(issues) == 1
        assert issues[0].issue_type == "reasoning_action_mismatch"

    def test_reasoning_action_mismatch_bullish_sell(self):
        votes = [
            AgentVote(agent_id="a", action="SELL", confidence=0.8,
                      reasoning="Strong bullish breakout with oversold recovery"),
        ]
        issues = self.engine.validate_votes(votes)
        assert len(issues) == 1
        assert issues[0].issue_type == "reasoning_action_mismatch"

    def test_mixed_signals_not_flagged(self):
        """One bearish word + one bullish word should not trigger mismatch."""
        votes = [
            AgentVote(agent_id="a", action="BUY", confidence=0.8,
                      reasoning="Slight bearish signal but overall bullish trend"),
        ]
        assert self.engine.validate_votes(votes) == []

    def test_multiple_issues(self):
        votes = [
            AgentVote(agent_id="a", action="BUY", confidence=0.0, reasoning=""),
            AgentVote(agent_id="b", action="SELL", confidence=0.8,
                      reasoning="Bullish breakout, strong support, oversold"),
        ]
        issues = self.engine.validate_votes(votes)
        types = {i.issue_type for i in issues}
        assert "empty_reasoning" in types
        assert "zero_confidence" in types
        assert "reasoning_action_mismatch" in types
