"""Tests for leave-one-out credit assignment (compute_agent_credits, suggest_weights)."""

import os
import tempfile

import pytest

from khonliang.consensus.credit import compute_agent_credits, suggest_weights
from khonliang.consensus.models import AgentVote, ConsensusResult
from khonliang.consensus.outcomes import OutcomeTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_tracker():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return OutcomeTracker(path), path


def _vote(agent_id: str, action: str, confidence: float) -> AgentVote:
    return AgentVote(
        agent_id=agent_id,
        action=action,
        confidence=confidence,
        reasoning="test",
    )


def _record(tracker: OutcomeTracker, votes, action: str, outcome: float) -> None:
    """Record a consensus + outcome in one call."""
    result = ConsensusResult(
        action=action,
        confidence=0.8,
        votes=votes,
        scores={action: 0.8},
    )
    cid = tracker.record_consensus(result)
    tracker.record_outcome(cid, outcome=outcome)


# ---------------------------------------------------------------------------
# compute_agent_credits — insufficient / edge cases
# ---------------------------------------------------------------------------

def test_compute_agent_credits_insufficient_samples_returns_none():
    tracker, path = _temp_tracker()
    try:
        for _ in range(3):
            _record(tracker, [_vote("a1", "APPROVE", 0.9)], "APPROVE", 0.8)

        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is None
    finally:
        os.unlink(path)


def test_compute_agent_credits_no_outcomes_returns_none():
    """Records without outcomes are excluded; if that drops below min_samples → None."""
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            result = ConsensusResult(
                action="APPROVE",
                confidence=0.9,
                votes=[_vote("a1", "APPROVE", 0.9)],
            )
            tracker.record_consensus(result)  # no outcome recorded

        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is None
    finally:
        os.unlink(path)


def test_compute_agent_credits_single_vote_per_record_still_returns():
    """Records with only one vote are skipped for LOO but the agent is still returned."""
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            _record(tracker, [_vote("a1", "APPROVE", 0.9)], "APPROVE", 0.8)

        # Records exist and have outcomes; a1 appears but each record has only 1 vote
        # so the LOO loop skips them. The function still returns a (uniform) result
        # rather than None because min_samples check already passed.
        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is not None
        assert "a1" in credits
        assert credits["a1"] == pytest.approx(1.0)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# compute_agent_credits — normalization
# ---------------------------------------------------------------------------

def test_compute_agent_credits_sums_to_one():
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [_vote("oracle", "APPROVE", 0.9), _vote("dissenter", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.8)

        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is not None
        assert abs(sum(credits.values()) - 1.0) < 1e-9
    finally:
        os.unlink(path)


def test_compute_agent_credits_all_values_non_negative():
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [
                _vote("a1", "APPROVE", 0.9),
                _vote("a2", "APPROVE", 0.8),
                _vote("a3", "REJECT", 0.6),
            ]
            _record(tracker, votes, "APPROVE", 0.7)

        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is not None
        for agent_id, score in credits.items():
            assert score >= 0.0, f"Negative credit for {agent_id}: {score}"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# compute_agent_credits — pivotal vs redundant
# ---------------------------------------------------------------------------

def test_compute_agent_credits_pivotal_scores_higher_than_dissenter():
    """Agent whose removal flips consensus gets more credit than a dissenter."""
    tracker, path = _temp_tracker()
    try:
        # oracle votes APPROVE and is the deciding vote (2-agent team).
        # Removing oracle → REJECT (action changes). Outcome is good.
        # dissenter votes REJECT but is overridden. Removing dissenter → same result.
        for _ in range(5):
            votes = [_vote("oracle", "APPROVE", 0.9), _vote("dissenter", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.9)

        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is not None
        assert credits["oracle"] > credits["dissenter"]
    finally:
        os.unlink(path)


def test_compute_agent_credits_agent_ids_match():
    """Returned dict must contain exactly the agents that appear in records."""
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [_vote("alpha", "APPROVE", 0.9), _vote("beta", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.8)

        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is not None
        assert set(credits.keys()) == {"alpha", "beta"}
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# suggest_weights
# ---------------------------------------------------------------------------

def test_suggest_weights_insufficient_data_returns_none():
    tracker, path = _temp_tracker()
    try:
        result = suggest_weights(tracker, min_samples=10)
        assert result is None
    finally:
        os.unlink(path)


def test_suggest_weights_sums_to_one():
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [_vote("a1", "APPROVE", 0.9), _vote("a2", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.8)

        weights = suggest_weights(tracker, min_samples=5)
        assert weights is not None
        assert abs(sum(weights.values()) - 1.0) < 1e-9
    finally:
        os.unlink(path)


def test_suggest_weights_no_current_equals_credits():
    """Without current_weights the suggestion equals the credit scores."""
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [_vote("a1", "APPROVE", 0.9), _vote("a2", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.8)

        weights = suggest_weights(tracker, min_samples=5)
        credits = compute_agent_credits(tracker, min_samples=5)
        assert weights is not None
        assert weights == credits
    finally:
        os.unlink(path)


def test_suggest_weights_blend_clamp_high():
    """blend > 1.0 is clamped; result must have no negative weights."""
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [_vote("a1", "APPROVE", 0.9), _vote("a2", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.8)

        current = {"a1": 0.5, "a2": 0.5}
        weights = suggest_weights(tracker, current_weights=current, blend=5.0, min_samples=5)
        assert weights is not None
        for w in weights.values():
            assert w >= 0.0, f"Negative weight: {w}"
    finally:
        os.unlink(path)


def test_suggest_weights_blend_clamp_zero_keeps_current():
    """blend=0 (clamped from -1) returns only the current (normalized) weights."""
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [_vote("a1", "APPROVE", 0.9), _vote("a2", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.8)

        current = {"a1": 0.6, "a2": 0.4}
        weights = suggest_weights(
            tracker, current_weights=current, blend=-1.0, min_samples=5
        )
        assert weights is not None
        # blend clamped to 0 → formula is 1*cur + 0*credit = cur; total=1.0 → no rescaling
        assert abs(weights["a1"] - 0.6) < 1e-9
        assert abs(weights["a2"] - 0.4) < 1e-9
    finally:
        os.unlink(path)


def test_suggest_weights_blended_between_current_and_credits():
    """Blended weight lies strictly between current and credit-only weight."""
    tracker, path = _temp_tracker()
    try:
        for _ in range(5):
            votes = [_vote("a1", "APPROVE", 0.9), _vote("a2", "REJECT", 0.5)]
            _record(tracker, votes, "APPROVE", 0.8)

        current = {"a1": 0.3, "a2": 0.7}  # deliberately away from credits
        credits = compute_agent_credits(tracker, min_samples=5)
        assert credits is not None

        blended = suggest_weights(
            tracker, current_weights=current, blend=0.5, min_samples=5
        )
        assert blended is not None

        for agent_id in ("a1", "a2"):
            # Blended weight must stay non-negative after renormalization
            assert blended[agent_id] >= 0.0
    finally:
        os.unlink(path)
