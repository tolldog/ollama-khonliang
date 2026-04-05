"""
Consensus engine — aggregates agent votes into a final decision.

Uses weighted voting: each vote's score = confidence × weight.
A VETO from any agent blocks the decision regardless of other votes.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from khonliang.consensus.models import AgentVote, ConsensusResult

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """
    Calculates consensus from a list of AgentVotes.

    The action with the highest total weighted score wins.
    Any VETO vote blocks the decision (returns action="VETO").

    Args:
        agent_weights:   Per-agent weight overrides {agent_id: weight}
        veto_blocks:     If True, a VETO vote from any agent blocks consensus
        min_confidence:  Minimum overall confidence to consider actionable
    """

    def __init__(
        self,
        agent_weights: Optional[Dict[str, float]] = None,
        veto_blocks: bool = True,
        min_confidence: float = 0.0,
        judge_fn: Optional[Callable[[List[AgentVote]], Optional[AgentVote]]] = None,
        debate_orchestrator: Optional[object] = None,
        debate_threshold: float = 0.15,
    ):
        """
        Args:
            agent_weights: Per-agent weight overrides
            veto_blocks: If True, VETO blocks consensus
            min_confidence: Minimum confidence threshold
            judge_fn: Optional sync/async function that reviews votes after
                      aggregation. Signature: (votes) -> Optional[AgentVote].
                      Returns None to accept consensus, or an AgentVote to override.
            debate_orchestrator: Optional DebateOrchestrator. When provided,
                auto-triggers debate when top 2 action scores are within
                debate_threshold of each other.
            debate_threshold: Score gap below which debate is triggered (0-1).
                Default 0.15 means debate when winner leads by < 15%.
        """
        self.agent_weights = agent_weights or {}
        self.veto_blocks = veto_blocks
        self.min_confidence = min_confidence
        self.judge_fn = judge_fn
        self.debate_orchestrator = debate_orchestrator
        self.debate_threshold = debate_threshold

    def _effective_weight(self, vote: AgentVote) -> float:
        return self.agent_weights.get(vote.agent_id, vote.weight)

    def calculate_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Aggregate votes into a single ConsensusResult using weighted scoring."""
        if not votes:
            return ConsensusResult(
                action="DEFER",
                confidence=0.0,
                reason="No votes received",
            )

        # Check for veto
        if self.veto_blocks:
            veto_votes = [v for v in votes if v.action == "VETO"]
            if veto_votes:
                vetoers = ", ".join(v.agent_id for v in veto_votes)
                return ConsensusResult(
                    action="VETO",
                    confidence=1.0,
                    votes=votes,
                    reason=f"Vetoed by: {vetoers}",
                )

        # Aggregate weighted scores per action
        scores: Dict[str, float] = {}
        total_weight = 0.0
        for vote in votes:
            w = self._effective_weight(vote)
            total_weight += w
            score = vote.confidence * w
            scores[vote.action] = scores.get(vote.action, 0.0) + score

        if total_weight == 0:
            return ConsensusResult(
                action="DEFER",
                confidence=0.0,
                votes=votes,
                reason="All vote weights are zero",
            )

        # Normalise scores
        normalised = {k: v / total_weight for k, v in scores.items()}

        # Pick winner
        winner = max(normalised, key=normalised.get)
        confidence = normalised[winner]

        result = ConsensusResult(
            action=winner,
            confidence=confidence,
            votes=votes,
            scores=normalised,
            reason=self._build_reason(votes, winner, normalised),
        )

        # Judge step: optional review of the consensus
        if self.judge_fn is not None:
            result = self._apply_judge(result, votes)

        return result

    def needs_debate(self, scores: Dict[str, float]) -> bool:
        """Check if scores are close enough to warrant debate.

        Returns True when the gap between the top 2 action scores
        is within debate_threshold and a debate_orchestrator is set.
        """
        if self.debate_orchestrator is None:
            return False
        if len(scores) < 2:
            return False

        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        return gap < self.debate_threshold

    async def calculate_consensus_with_debate(
        self,
        votes: List[AgentVote],
        subject: str = "",
        context: Optional[dict] = None,
    ) -> ConsensusResult:
        """Async consensus with auto-debate on close scores.

        Same as calculate_consensus() but triggers debate when the
        top 2 action scores are within debate_threshold. Falls back
        to standard consensus if no debate_orchestrator is set.

        Args:
            votes: Agent votes
            subject: What is being evaluated (passed to debate)
            context: Optional context (passed to debate)
        """
        # First pass: standard consensus
        result = self.calculate_consensus(votes)

        # Check if debate is warranted
        if not self.needs_debate(result.scores):
            return result

        logger.info(
            "Auto-debate triggered for '%s': scores %s (gap < %.0f%%)",
            subject[:40],
            {k: f"{v:.2f}" for k, v in result.scores.items()},
            self.debate_threshold * 100,
        )

        # Run debate to let agents reconsider
        orchestrator = self.debate_orchestrator
        updated_votes = await orchestrator.run_debate(
            votes, subject, context,
        )

        # Recalculate consensus without debate to avoid recursion.
        # Use a local engine copy rather than mutating self to stay thread-safe.
        local_engine = ConsensusEngine(
            agent_weights=self.agent_weights,
            veto_blocks=self.veto_blocks,
            min_confidence=self.min_confidence,
            judge_fn=self.judge_fn,
        )
        final = local_engine.calculate_consensus(updated_votes)

        history = getattr(orchestrator, "_debate_history", None)
        if history:
            final.debate_rounds = history[-1].get("rounds", 1)
        else:
            final.debate_rounds = 1

        return final

    def _apply_judge(
        self, result: ConsensusResult, votes: List[AgentVote]
    ) -> ConsensusResult:
        """Run the judge function to optionally override consensus."""
        try:
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(self.judge_fn):
                try:
                    _ = asyncio.get_running_loop()  # raises RuntimeError if no loop
                    # Already inside a running event loop — asyncio.run() would
                    # raise RuntimeError, so skip the judge gracefully.
                    logger.warning(
                        "Async judge_fn skipped: calculate_consensus() was called "
                        "from within a running event loop. Use a sync judge_fn, or "
                        "invoke calculate_consensus() outside an async context."
                    )
                    return result
                except RuntimeError:
                    pass  # No running loop — safe to call asyncio.run().
                override = asyncio.run(self.judge_fn(votes))
            else:
                override = self.judge_fn(votes)

            if override is not None and isinstance(override, AgentVote):
                logger.info(
                    f"Judge overrode consensus: "
                    f"{result.action} -> {override.action} "
                    f"({override.reasoning[:60]})"
                )
                return ConsensusResult(
                    action=override.action,
                    confidence=override.confidence,
                    votes=votes + [override],
                    scores={},
                    reason=(
                        f"Judge override: {override.reasoning}. "
                        f"Original: {result.reason}"
                    ),
                    judge_overridden=True,
                    original_action=result.action,
                    original_scores=result.scores,
                )
        except Exception:
            logger.exception("Judge function failed")

        return result

    def _build_reason(
        self,
        votes: List[AgentVote],
        winner: str,
        scores: Dict[str, float],
    ) -> str:
        winner_votes = [v for v in votes if v.action == winner]
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items()))
        agents = ", ".join(v.agent_id for v in winner_votes)
        return (
            f"{winner} ({len(winner_votes)}/{len(votes)} votes, "
            f"agents: {agents}, scores: {score_str})"
        )

    def get_vote_summary(self, votes: List[AgentVote]) -> dict:
        """Return a summary of votes grouped by action."""
        by_action: Dict[str, List[str]] = {}
        for vote in votes:
            by_action.setdefault(vote.action, []).append(vote.agent_id)
        return {
            "total_votes": len(votes),
            "by_action": by_action,
            "agents": [v.agent_id for v in votes],
        }

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Merge new per-agent weight overrides into the current weights."""
        self.agent_weights.update(weights)

    # ------------------------------------------------------------------
    # Vote validation (KH-6)
    # ------------------------------------------------------------------

    def validate_votes(self, votes: List[AgentVote]) -> List["ValidationIssue"]:
        """Pre-consensus validation. Checks for common multi-agent failure modes.

        Currently detects:
        - reasoning_action_mismatch: reasoning sentiment contradicts the vote action
        - empty_reasoning: vote has no explanation
        - zero_confidence: agent voted but with 0.0 confidence

        Returns a list of issues (empty = all votes OK).
        """
        issues: List[ValidationIssue] = []
        for vote in votes:
            # Empty reasoning
            if not vote.reasoning or not vote.reasoning.strip():
                issues.append(ValidationIssue(
                    agent_id=vote.agent_id,
                    issue_type="empty_reasoning",
                    detail=f"Agent {vote.agent_id} voted {vote.action} with no reasoning",
                ))

            # Zero confidence
            action_upper = str(vote.action).upper()
            if vote.confidence == 0.0 and action_upper != "VETO":
                issues.append(ValidationIssue(
                    agent_id=vote.agent_id,
                    issue_type="zero_confidence",
                    detail=f"Agent {vote.agent_id} voted {vote.action} with 0.0 confidence",
                ))

            # Reasoning-action mismatch
            mismatch = _detect_reasoning_action_mismatch(vote)
            if mismatch:
                issues.append(ValidationIssue(
                    agent_id=vote.agent_id,
                    issue_type="reasoning_action_mismatch",
                    detail=mismatch,
                ))

        if issues:
            logger.warning(
                "Vote validation found %d issues: %s",
                len(issues),
                ", ".join(f"{i.agent_id}:{i.issue_type}" for i in issues),
            )
        return issues


@dataclass
class ValidationIssue:
    """A problem detected during vote validation."""

    agent_id: str
    issue_type: str  # "reasoning_action_mismatch", "empty_reasoning", "zero_confidence"
    detail: str


# Sentiment keywords for reasoning-action mismatch detection
_BULLISH = re.compile(
    r"\b(bullish|oversold|buy signal|uptrend|breakout|support|recovery|positive|strong)\b",
    re.IGNORECASE,
)
_BEARISH = re.compile(
    r"\b(bearish|overbought|sell signal|downtrend|breakdown|resistance|decline|negative|weak)\b",
    re.IGNORECASE,
)
_BUY_ACTIONS = {"BUY", "APPROVE", "LONG"}
_SELL_ACTIONS = {"SELL", "REJECT", "SHORT"}


def _detect_reasoning_action_mismatch(vote: AgentVote) -> Optional[str]:
    """Check if reasoning sentiment contradicts the vote action."""
    if not vote.reasoning:
        return None

    action_upper = vote.action.upper() if isinstance(vote.action, str) else str(vote.action).upper()
    reasoning = vote.reasoning

    bullish_hits = len(_BULLISH.findall(reasoning))
    bearish_hits = len(_BEARISH.findall(reasoning))

    # Only flag clear mismatches (2+ signals in wrong direction)
    if action_upper in _SELL_ACTIONS and bullish_hits >= 2 and bearish_hits == 0:
        return (
            f"Agent {vote.agent_id} voted {vote.action} but reasoning is bullish: "
            f"'{reasoning[:80]}...'"
        )
    if action_upper in _BUY_ACTIONS and bearish_hits >= 2 and bullish_hits == 0:
        return (
            f"Agent {vote.agent_id} voted {vote.action} but reasoning is bearish: "
            f"'{reasoning[:80]}...'"
        )

    return None
