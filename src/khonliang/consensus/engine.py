"""
Consensus engine — aggregates agent votes into a final decision.

Uses weighted voting: each vote's score = confidence × weight.
A VETO from any agent blocks the decision regardless of other votes.
"""

import logging
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
    ):
        """
        Args:
            agent_weights: Per-agent weight overrides
            veto_blocks: If True, VETO blocks consensus
            min_confidence: Minimum confidence threshold
            judge_fn: Optional sync/async function that reviews votes after
                      aggregation. Signature: (votes) -> Optional[AgentVote].
                      Returns None to accept consensus, or an AgentVote to override.
        """
        self.agent_weights = agent_weights or {}
        self.veto_blocks = veto_blocks
        self.min_confidence = min_confidence
        self.judge_fn = judge_fn

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
