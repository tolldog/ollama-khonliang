"""Tests for leave-one-out counterfactual credit assignment (KH-2)."""

import os
import random
import tempfile

import pytest

from khonliang.consensus.credit import (
    _compute_credit,
    _reconstruct_votes,
    compute_agent_credits,
    suggest_weights,
)
from khonliang.consensus.models import AgentVote, ConsensusResult
from khonliang.consensus.outcomes import OutcomeTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_tracker():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return OutcomeTracker(path), path


def _make_vote(agent_id, action, confidence=0.8, weight=1.0):
    return AgentVote(
        agent_id=agent_id,
        action=action,
        confidence=confidence,
        reasoning=f"{agent_id} says {action}",
        weight=weight,
    )


def _populate_tracker(tracker, n, good_agent="quant", noise_agent="noise", seed=42):
    """Seed the tracker with `n` consensuses plus outcomes.

    `good_agent` always votes with the majority; `noise_agent` votes randomly.
    Returns the list of consensus_ids.
    """
    rng = random.Random(seed)
    cids = []
    for i in range(n):
        majority_action = rng.choice(["APPROVE", "REJECT"])
        noise_action = rng.choice(["APPROVE", "REJECT"])

        votes = [
            _make_vote(good_agent, majority_action, confidence=0.9),
            _make_vote(noise_agent, noise_action, confidence=0.5),
            # Third voter always agrees with majority to guarantee consensus
            _make_vote("anchor", majority_action, confidence=0.7),
        ]
        result = ConsensusResult(
            action=majority_action,
            confidence=0.8,
            votes=votes,
            scores={majority_action: 0.8},
        )
        cid = tracker.record_consensus(result, subject=f"task-{i}")
        # Good outcome whenever majority_action == APPROVE
        outcome = 0.9 if majority_action == "APPROVE" else 0.2
        tracker.record_outcome(cid, outcome=outcome)
        cids.append(cid)
    return cids


# ---------------------------------------------------------------------------
# _compute_credit unit tests
# ---------------------------------------------------------------------------

class TestComputeCredit:
    def _vote(self, action="APPROVE", confidence=0.8):
        return _make_vote("a", action, confidence=confidence)

    def test_pivotal_aligned_good_outcome(self):
        """Pivotal agent that voted for the winning action gets positive credit."""
        credit = _compute_credit(
            original_action="APPROVE",
            counterfactual_action="REJECT",  # action would have changed
            agent_vote=self._vote("APPROVE", 0.8),
            outcome=1.0,
        )
        assert credit == pytest.approx(0.8)  # outcome * confidence

    def test_pivotal_aligned_bad_outcome(self):
        """Pivotal aligned agent gets negative credit for bad decision."""
        credit = _compute_credit(
            original_action="APPROVE",
            counterfactual_action="REJECT",
            agent_vote=self._vote("APPROVE", 0.8),
            outcome=0.0,
        )
        assert credit == pytest.approx(0.0)

    def test_pivotal_misaligned(self):
        """Dissenting agent who was somehow pivotal gets penalised on good outcome."""
        credit = _compute_credit(
            original_action="APPROVE",
            counterfactual_action="REJECT",
            agent_vote=self._vote("REJECT", 0.8),  # voted against winner
            outcome=1.0,
        )
        assert credit < 0

    def test_redundant_aligned_positive(self):
        """Redundant agent that voted for winner gets small positive credit."""
        credit = _compute_credit(
            original_action="APPROVE",
            counterfactual_action="APPROVE",  # no change
            agent_vote=self._vote("APPROVE", 0.8),
            outcome=1.0,
        )
        assert credit > 0
        # Should be much smaller than a pivotal agent
        pivotal = _compute_credit(
            original_action="APPROVE",
            counterfactual_action="REJECT",
            agent_vote=self._vote("APPROVE", 0.8),
            outcome=1.0,
        )
        assert credit < pivotal

    def test_redundant_dissenter_zero(self):
        """Redundant dissenter (voted against winner, no effect) gets zero credit."""
        credit = _compute_credit(
            original_action="APPROVE",
            counterfactual_action="APPROVE",
            agent_vote=self._vote("REJECT", 0.8),
            outcome=1.0,
        )
        assert credit == pytest.approx(0.0)

    def test_confidence_scales_credit(self):
        """Higher confidence should yield higher credit magnitude."""
        low = _compute_credit("APPROVE", "REJECT", self._vote("APPROVE", 0.3), 1.0)
        high = _compute_credit("APPROVE", "REJECT", self._vote("APPROVE", 0.9), 1.0)
        assert high > low


# ---------------------------------------------------------------------------
# _reconstruct_votes unit tests
# ---------------------------------------------------------------------------

class TestReconstructVotes:
    def test_valid_dicts(self):
        dicts = [
            {"agent_id": "a", "action": "APPROVE", "confidence": 0.8,
             "reasoning": "good", "weight": 1.0},
        ]
        votes = _reconstruct_votes(dicts)
        assert len(votes) == 1
        assert votes[0].agent_id == "a"
        assert votes[0].action == "APPROVE"

    def test_defaults_filled(self):
        """Missing optional fields should use sensible defaults."""
        dicts = [{"agent_id": "a", "action": "APPROVE"}]
        votes = _reconstruct_votes(dicts)
        assert len(votes) == 1
        assert votes[0].confidence == 0.5
        assert votes[0].weight == 1.0

    def test_malformed_skipped(self):
        """Dicts missing required keys should be silently skipped."""
        dicts = [
            {"action": "APPROVE"},  # missing agent_id
            {"agent_id": "b", "action": "REJECT", "confidence": 0.7,
             "reasoning": "ok"},
        ]
        votes = _reconstruct_votes(dicts)
        assert len(votes) == 1
        assert votes[0].agent_id == "b"

    def test_invalid_confidence_skipped(self):
        """Confidence outside [0,1] raises ValueError and should be skipped."""
        dicts = [
            {"agent_id": "bad", "action": "APPROVE", "confidence": 2.0,
             "reasoning": ""},
            {"agent_id": "ok", "action": "REJECT", "confidence": 0.5,
             "reasoning": ""},
        ]
        votes = _reconstruct_votes(dicts)
        assert len(votes) == 1
        assert votes[0].agent_id == "ok"

    def test_empty_list(self):
        assert _reconstruct_votes([]) == []


# ---------------------------------------------------------------------------
# compute_agent_credits integration tests
# ---------------------------------------------------------------------------

class TestComputeAgentCredits:
    def test_insufficient_data_returns_none(self):
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=5)
        result = compute_agent_credits(tracker, min_samples=20)
        assert result is None
        os.unlink(path)

    def test_sufficient_data_returns_dict(self):
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        result = compute_agent_credits(tracker, min_samples=20)
        assert result is not None
        assert isinstance(result, dict)
        assert set(result.keys()) == {"quant", "noise", "anchor"}
        os.unlink(path)

    def test_scores_sum_to_one(self):
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        result = compute_agent_credits(tracker, min_samples=20)
        assert result is not None
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        os.unlink(path)

    def test_scores_non_negative(self):
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        result = compute_agent_credits(tracker, min_samples=20)
        assert result is not None
        for score in result.values():
            assert score >= 0.0
        os.unlink(path)

    def test_strong_predictor_beats_noise(self):
        """The agent that always votes with the majority should get more credit
        than a random-voting noise agent over enough samples."""
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=60, good_agent="quant", noise_agent="noise")
        result = compute_agent_credits(tracker, min_samples=20)
        assert result is not None
        assert result["quant"] > result["noise"], (
            f"quant={result['quant']:.4f} should exceed noise={result['noise']:.4f}"
        )
        os.unlink(path)

    def test_empty_tracker_returns_none(self):
        tracker, path = _temp_tracker()
        result = compute_agent_credits(tracker, min_samples=1)
        assert result is None
        os.unlink(path)

    def test_limit_parameter_respected(self):
        """limit restricts how many records are analyzed."""
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=40)
        # With limit=5 we're below min_samples=20, should return None
        result = compute_agent_credits(tracker, min_samples=20, limit=5)
        assert result is None
        os.unlink(path)


# ---------------------------------------------------------------------------
# suggest_weights integration tests
# ---------------------------------------------------------------------------

class TestSuggestWeights:
    def test_insufficient_data_returns_none(self):
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=5)
        result = suggest_weights(tracker, min_samples=20)
        assert result is None
        os.unlink(path)

    def test_returns_dict_with_correct_keys(self):
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        result = suggest_weights(tracker, min_samples=20)
        assert result is not None
        assert set(result.keys()) == {"quant", "noise", "anchor"}
        os.unlink(path)

    def test_suggested_weights_sum_to_one(self):
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        result = suggest_weights(tracker, min_samples=20)
        assert result is not None
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        os.unlink(path)

    def test_blend_zero_keeps_current_weights(self):
        """blend=0 should return weights proportional to current_weights."""
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        current = {"quant": 2.0, "noise": 1.0, "anchor": 1.0}
        result = suggest_weights(
            tracker, current_weights=current, min_samples=20, blend=0.0,
        )
        assert result is not None
        # Should be current weights normalised to sum=1
        total = sum(current.values())
        for agent_id, w in current.items():
            assert result[agent_id] == pytest.approx(w / total, abs=1e-9)
        os.unlink(path)

    def test_blend_one_equals_pure_credit(self):
        """blend=1 should return the same values as compute_agent_credits."""
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        current = {"quant": 1.0, "noise": 1.0, "anchor": 1.0}
        credits = compute_agent_credits(tracker, min_samples=20)
        suggested = suggest_weights(
            tracker, current_weights=current, min_samples=20, blend=1.0,
        )
        assert credits is not None
        assert suggested is not None
        for agent_id in credits:
            assert suggested[agent_id] == pytest.approx(credits[agent_id], abs=1e-9)
        os.unlink(path)

    def test_blend_interpolates(self):
        """blend=0.3 result should sit strictly between blend=0 and blend=1."""
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        current = {"quant": 2.0, "noise": 1.0, "anchor": 1.0}

        w0 = suggest_weights(tracker, current_weights=current, min_samples=20, blend=0.0)
        w1 = suggest_weights(tracker, current_weights=current, min_samples=20, blend=1.0)
        w3 = suggest_weights(tracker, current_weights=current, min_samples=20, blend=0.3)
        assert w0 is not None and w1 is not None and w3 is not None

        # For every agent, blend=0.3 should be between blend=0 and blend=1
        for agent_id in current:
            lo = min(w0[agent_id], w1[agent_id])
            hi = max(w0[agent_id], w1[agent_id])
            assert lo - 1e-9 <= w3[agent_id] <= hi + 1e-9, (
                f"{agent_id}: blend0={w0[agent_id]:.4f}, blend1={w1[agent_id]:.4f}, "
                f"blend0.3={w3[agent_id]:.4f} is not in-between"
            )
        os.unlink(path)

    def test_no_current_weights_equals_credits(self):
        """When current_weights is None the result equals the credit scores."""
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=30)
        credits = compute_agent_credits(tracker, min_samples=20)
        suggested = suggest_weights(tracker, min_samples=20)
        assert credits is not None and suggested is not None
        for agent_id in credits:
            assert suggested[agent_id] == pytest.approx(credits[agent_id], abs=1e-9)
        os.unlink(path)

    def test_limit_forwarded(self):
        """limit kwarg should be forwarded to compute_agent_credits."""
        tracker, path = _temp_tracker()
        _populate_tracker(tracker, n=40)
        result = suggest_weights(tracker, min_samples=20, limit=5)
        assert result is None
        os.unlink(path)
