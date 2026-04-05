"""Tests for ConsensusEngine vote validation, OutcomeTracker, debate, and sampling."""

import os
import tempfile

import pytest

from khonliang.consensus.engine import ConsensusEngine, ValidationIssue
from khonliang.consensus.models import AgentVote, ConsensusResult
from khonliang.consensus.outcomes import OutcomeTracker
from khonliang.consensus.team import _select_best_vote

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_tracker():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return OutcomeTracker(path), path


def _vote(agent_id: str, action: str, confidence: float, reasoning: str) -> AgentVote:
    return AgentVote(
        agent_id=agent_id,
        action=action,
        confidence=confidence,
        reasoning=reasoning,
    )


def _result(votes, action="APPROVE", confidence=0.8) -> ConsensusResult:
    return ConsensusResult(
        action=action,
        confidence=confidence,
        votes=votes,
        scores={action: confidence},
        reason="test",
    )


# ---------------------------------------------------------------------------
# OutcomeTracker tests
# ---------------------------------------------------------------------------

def test_record_consensus_returns_id():
    tracker, path = _temp_tracker()
    try:
        votes = [_vote("analyst", "APPROVE", 0.9, "looks good")]
        cid = tracker.record_consensus(_result(votes), subject="task-1")
        assert isinstance(cid, str)
        assert len(cid) > 0
    finally:
        os.unlink(path)


def test_record_outcome_links_to_consensus():
    tracker, path = _temp_tracker()
    try:
        votes = [_vote("analyst", "APPROVE", 0.9, "looks good")]
        cid = tracker.record_consensus(_result(votes))

        found = tracker.record_outcome(cid, outcome=0.75, metadata={"task": "t1"})

        assert found is True
        history = tracker.get_history(with_outcome_only=True)
        assert len(history) == 1
        assert history[0].outcome == pytest.approx(0.75)
        assert history[0].outcome_metadata == {"task": "t1"}
    finally:
        os.unlink(path)


def test_record_outcome_missing_id_returns_false():
    tracker, path = _temp_tracker()
    try:
        found = tracker.record_outcome("nonexistent-id", outcome=0.5)
        assert found is False
    finally:
        os.unlink(path)


def test_record_outcome_idempotent_returns_true():
    """record_outcome with same values should not return False (no-op update)."""
    tracker, path = _temp_tracker()
    try:
        votes = [_vote("analyst", "APPROVE", 0.9, "looks good")]
        cid = tracker.record_consensus(_result(votes))

        assert tracker.record_outcome(cid, outcome=0.5) is True
        # Same call again — rowcount would be 0 without the SELECT 1 fix
        assert tracker.record_outcome(cid, outcome=0.5) is True
    finally:
        os.unlink(path)


def test_record_consensus_preserves_outcome_on_re_record():
    """Re-recording the same consensus_id must NOT wipe out an already-stored outcome."""
    tracker, path = _temp_tracker()
    try:
        votes = [_vote("analyst", "APPROVE", 0.9, "looks good")]
        res = _result(votes)
        cid = tracker.record_consensus(res, consensus_id="stable-cid")

        tracker.record_outcome(cid, outcome=0.9)

        # Re-record (e.g. idempotency / retry scenario)
        tracker.record_consensus(res, consensus_id="stable-cid")

        history = tracker.get_history(with_outcome_only=True)
        assert len(history) == 1
        assert history[0].outcome == pytest.approx(0.9)  # must survive the re-record
    finally:
        os.unlink(path)


def test_get_history_filters_by_action_subject_agent():
    tracker, path = _temp_tracker()
    try:
        votes_a = [_vote("agent1", "APPROVE", 0.9, "good")]
        votes_b = [_vote("agent1", "REJECT", 0.7, "bad")]

        cid_a = tracker.record_consensus(_result(votes_a, action="APPROVE"), subject="topic-a")
        tracker.record_consensus(_result(votes_b, action="REJECT"), subject="topic-b")
        tracker.record_outcome(cid_a, outcome=0.8)

        assert len(tracker.get_history(action="APPROVE")) == 1
        assert len(tracker.get_history(subject="topic-b")) == 1
        assert len(tracker.get_history(agent_id="agent1")) == 2
        assert len(tracker.get_history(with_outcome_only=True)) == 1
    finally:
        os.unlink(path)


def test_get_agent_stats():
    tracker, path = _temp_tracker()
    try:
        agent_id = "agent1"

        # Aligned: agent voted APPROVE, consensus is APPROVE
        votes_a = [_vote(agent_id, "APPROVE", 0.9, "good")]
        cid_a = tracker.record_consensus(_result(votes_a, action="APPROVE"))
        tracker.record_outcome(cid_a, outcome=0.8)

        # Opposed: agent voted REJECT, consensus is APPROVE
        votes_b = [
            _vote(agent_id, "REJECT", 0.8, "bad"),
            _vote("agent2", "APPROVE", 0.9, "good"),
        ]
        cid_b = tracker.record_consensus(
            ConsensusResult(action="APPROVE", confidence=0.9, votes=votes_b)
        )
        tracker.record_outcome(cid_b, outcome=0.2)

        stats = tracker.get_agent_stats(agent_id)

        assert stats["agent_id"] == agent_id
        assert stats["sample_count"] == 2
        assert stats["aligned_count"] == 1
        assert stats["opposed_count"] == 1
        assert stats["mean_outcome_aligned"] == pytest.approx(0.8)
        assert stats["mean_outcome_opposed"] == pytest.approx(0.2)
        assert stats["outcome_delta"] == pytest.approx(0.6)
    finally:
        os.unlink(path)


def test_get_agent_stats_no_data():
    tracker, path = _temp_tracker()
    try:
        stats = tracker.get_agent_stats("nobody")
        assert stats["agent_id"] == "nobody"
        assert stats["sample_count"] == 0
    finally:
        os.unlink(path)


def test_get_stats():
    tracker, path = _temp_tracker()
    try:
        votes = [_vote("a", "APPROVE", 0.9, "ok")]
        cid = tracker.record_consensus(_result(votes))

        stats = tracker.get_stats()
        assert stats["total_consensuses"] == 1
        assert stats["with_outcomes"] == 0
        assert stats["pending_outcomes"] == 1

        tracker.record_outcome(cid, outcome=0.5)
        stats = tracker.get_stats()
        assert stats["with_outcomes"] == 1
        assert stats["pending_outcomes"] == 0
    finally:
        os.unlink(path)


def test_outcome_record_helpers():
    tracker, path = _temp_tracker()
    try:
        votes = [
            _vote("agent1", "APPROVE", 0.9, "positive"),
            _vote("agent2", "REJECT", 0.6, "negative"),
        ]
        cid = tracker.record_consensus(_result(votes, action="APPROVE"))
        tracker.record_outcome(cid, outcome=0.7)

        records = tracker.get_history()
        assert len(records) == 1
        rec = records[0]

        assert rec.has_outcome
        assert rec.agent_voted("agent1") is not None
        assert rec.agent_voted("nobody") is None
        assert "agent1" in rec.agents_who_voted("APPROVE")
        assert "agent2" not in rec.agents_who_voted("APPROVE")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Vote validation tests
# ---------------------------------------------------------------------------

def test_validate_clean_votes_pass():
    engine = ConsensusEngine()
    votes = [
        _vote("analyst", "APPROVE", 0.9, "The code looks clean and well-tested."),
        _vote("skeptic", "REJECT", 0.7, "There are unresolved edge cases."),
    ]
    assert engine.validate_votes(votes) == []


def test_validate_empty_reasoning():
    engine = ConsensusEngine()
    votes = [_vote("analyst", "APPROVE", 0.9, "")]
    issues = engine.validate_votes(votes)
    assert len(issues) == 1
    assert issues[0].issue_type == "empty_reasoning"
    assert issues[0].agent_id == "analyst"


def test_validate_whitespace_only_reasoning():
    engine = ConsensusEngine()
    votes = [_vote("analyst", "APPROVE", 0.8, "   ")]
    issues = engine.validate_votes(votes)
    types = {i.issue_type for i in issues}
    assert "empty_reasoning" in types


def test_validate_zero_confidence():
    engine = ConsensusEngine()
    votes = [_vote("analyst", "APPROVE", 0.0, "Not sure about this.")]
    issues = engine.validate_votes(votes)
    assert len(issues) == 1
    assert issues[0].issue_type == "zero_confidence"
    assert issues[0].agent_id == "analyst"


def test_validate_veto_zero_confidence_not_flagged():
    """VETO action with 0.0 confidence must not trigger zero_confidence."""
    engine = ConsensusEngine()
    votes = [_vote("gatekeeper", "VETO", 0.0, "Hard block: policy violation.")]
    issues = engine.validate_votes(votes)
    confidence_issues = [i for i in issues if i.issue_type == "zero_confidence"]
    assert confidence_issues == []


def test_validate_mismatch_sell_with_bullish_reasoning():
    engine = ConsensusEngine()
    votes = [
        _vote(
            "analyst", "SELL", 0.8,
            "Uptrend with clear breakout and bullish momentum. Strong buy signal everywhere.",
        ),
    ]
    issues = engine.validate_votes(votes)
    mismatch = [i for i in issues if i.issue_type == "reasoning_action_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].agent_id == "analyst"


def test_validate_mismatch_buy_with_bearish_reasoning():
    engine = ConsensusEngine()
    votes = [
        _vote(
            "skeptic", "BUY", 0.7,
            "The stock is bearish with a clear breakdown. Negative momentum and weak outlook.",
        ),
    ]
    issues = engine.validate_votes(votes)
    mismatch = [i for i in issues if i.issue_type == "reasoning_action_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].agent_id == "skeptic"


def test_validate_borderline_mixed_signals_not_flagged():
    """Mixed bullish + bearish signals in the same reasoning must NOT trigger a mismatch."""
    engine = ConsensusEngine()
    votes = [
        _vote(
            "analyst", "SELL", 0.7,
            "Stock shows some bullish setup but the overall trend is bearish with breakdown.",
        ),
    ]
    issues = engine.validate_votes(votes)
    mismatch = [i for i in issues if i.issue_type == "reasoning_action_mismatch"]
    assert mismatch == []


def test_validate_single_keyword_not_flagged():
    """A single bullish keyword in a SELL vote should not trigger mismatch (threshold is 2)."""
    engine = ConsensusEngine()
    votes = [_vote("analyst", "SELL", 0.8, "There is a minor bullish signal but outlook is poor.")]
    issues = engine.validate_votes(votes)
    mismatch = [i for i in issues if i.issue_type == "reasoning_action_mismatch"]
    assert mismatch == []


def test_validate_multiple_issues_on_one_vote():
    engine = ConsensusEngine()
    votes = [_vote("agent1", "APPROVE", 0.0, "")]  # empty_reasoning + zero_confidence
    issues = engine.validate_votes(votes)
    types = {i.issue_type for i in issues}
    assert "empty_reasoning" in types
    assert "zero_confidence" in types


def test_validate_empty_vote_list():
    engine = ConsensusEngine()
    assert engine.validate_votes([]) == []


def test_validation_issue_is_dataclass():
    issue = ValidationIssue(
        agent_id="a1",
        issue_type="empty_reasoning",
        detail="No reasoning provided",
    )
    assert issue.agent_id == "a1"
    assert issue.issue_type == "empty_reasoning"


# ---------------------------------------------------------------------------
# _select_best_vote (team.py)
# ---------------------------------------------------------------------------

def test_select_best_vote_single_candidate():
    v = _vote("a", "APPROVE", 0.8, "good")
    assert _select_best_vote([v]) is v


def test_select_best_vote_plurality_action_wins():
    candidates = [
        _vote("a", "APPROVE", 0.7, "r"),
        _vote("b", "APPROVE", 0.9, "r"),
        _vote("c", "REJECT", 0.6, "r"),
    ]
    best = _select_best_vote(candidates)
    assert best.action == "APPROVE"


def test_select_best_vote_highest_confidence_in_plurality():
    candidates = [
        _vote("a", "APPROVE", 0.7, "r"),
        _vote("b", "APPROVE", 0.9, "r"),
        _vote("c", "REJECT", 0.6, "r"),
    ]
    best = _select_best_vote(candidates)
    assert best.confidence == pytest.approx(0.9)


def test_select_best_vote_unanimous():
    candidates = [
        _vote("a", "APPROVE", 0.5, "r"),
        _vote("b", "APPROVE", 0.8, "r"),
        _vote("c", "APPROVE", 0.6, "r"),
    ]
    best = _select_best_vote(candidates)
    assert best.action == "APPROVE"
    assert best.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# ConsensusEngine.needs_debate
# ---------------------------------------------------------------------------

def test_needs_debate_no_orchestrator():
    engine = ConsensusEngine()
    assert engine.needs_debate({"APPROVE": 0.6, "REJECT": 0.4}) is False


def test_needs_debate_single_action():
    engine = ConsensusEngine(debate_orchestrator=object(), debate_threshold=0.15)
    assert engine.needs_debate({"APPROVE": 1.0}) is False


def test_needs_debate_gap_above_threshold():
    engine = ConsensusEngine(debate_orchestrator=object(), debate_threshold=0.15)
    # Gap = 0.6 - 0.4 = 0.2 > 0.15 → no debate needed
    assert engine.needs_debate({"APPROVE": 0.6, "REJECT": 0.4}) is False


def test_needs_debate_gap_below_threshold():
    engine = ConsensusEngine(debate_orchestrator=object(), debate_threshold=0.15)
    # Gap = 0.525 - 0.475 = 0.05 < 0.15 → debate needed
    assert engine.needs_debate({"APPROVE": 0.525, "REJECT": 0.475}) is True


def test_needs_debate_gap_exactly_at_threshold_not_triggered():
    """Strict less-than: gap clearly above threshold should NOT trigger debate."""
    engine = ConsensusEngine(debate_orchestrator=object(), debate_threshold=0.15)
    # Gap = 0.7 - 0.3 = 0.4, well above threshold of 0.15
    assert engine.needs_debate({"APPROVE": 0.7, "REJECT": 0.3}) is False


# ---------------------------------------------------------------------------
# ConsensusEngine.calculate_consensus_with_debate
# ---------------------------------------------------------------------------

async def test_calculate_consensus_with_debate_no_orchestrator():
    """Returns standard consensus result when no orchestrator is configured."""
    engine = ConsensusEngine()
    votes = [
        _vote("a", "APPROVE", 0.9, "looks good"),
        _vote("b", "REJECT", 0.3, "marginal concern"),
    ]
    result = await engine.calculate_consensus_with_debate(votes, subject="test")
    assert result.action == "APPROVE"


async def test_calculate_consensus_with_debate_skips_when_not_close():
    """Debate is not triggered when scores are far apart."""
    debate_called = []

    class MockOrchestrator:
        async def run_debate(self, votes, subject, context):
            debate_called.append(True)
            raise AssertionError("should not be called")

    engine = ConsensusEngine(
        debate_orchestrator=MockOrchestrator(),
        debate_threshold=0.05,  # tight threshold
    )
    votes = [
        _vote("a", "APPROVE", 0.9, "clearly good"),
        _vote("b", "APPROVE", 0.85, "also good"),
        _vote("c", "REJECT", 0.05, "marginal"),
    ]
    result = await engine.calculate_consensus_with_debate(votes)
    assert result.action == "APPROVE"
    assert not debate_called


async def test_calculate_consensus_with_debate_triggers_and_recalculates():
    """Debate is triggered on close scores and consensus recalculates from updated votes."""
    post_debate_votes = [
        _vote("a", "APPROVE", 0.9, "convinced"),
        _vote("b", "APPROVE", 0.75, "changed mind"),
    ]

    class MockOrchestrator:
        _debate_history = [{"rounds": 2}]

        async def run_debate(self, votes, subject, context):
            return post_debate_votes

    engine = ConsensusEngine(
        debate_orchestrator=MockOrchestrator(),
        debate_threshold=0.5,  # wide threshold — debate always triggered
    )
    votes = [
        _vote("a", "APPROVE", 0.55, "slight lean"),
        _vote("b", "REJECT", 0.45, "slight lean other way"),
    ]
    result = await engine.calculate_consensus_with_debate(votes, subject="close call")
    assert result.action == "APPROVE"
    assert result.debate_rounds == 2


async def test_calculate_consensus_with_debate_empty_history_defaults_to_one():
    """debate_rounds defaults to 1 when orchestrator._debate_history is empty."""

    class MockOrchestrator:
        _debate_history: list = []

        async def run_debate(self, votes, subject, context):
            return votes  # return original votes unchanged

    engine = ConsensusEngine(
        debate_orchestrator=MockOrchestrator(),
        debate_threshold=0.5,
    )
    votes = [
        _vote("a", "APPROVE", 0.55, "slight lean"),
        _vote("b", "REJECT", 0.45, "other lean"),
    ]
    result = await engine.calculate_consensus_with_debate(votes)
    assert result.debate_rounds == 1


async def test_calculate_consensus_with_debate_no_history_attr():
    """debate_rounds defaults to 1 when orchestrator has no _debate_history attribute."""

    class MockOrchestrator:
        async def run_debate(self, votes, subject, context):
            return votes

    engine = ConsensusEngine(
        debate_orchestrator=MockOrchestrator(),
        debate_threshold=0.5,
    )
    votes = [
        _vote("a", "APPROVE", 0.55, "lean"),
        _vote("b", "REJECT", 0.45, "other lean"),
    ]
    result = await engine.calculate_consensus_with_debate(votes)
    assert result.debate_rounds == 1
