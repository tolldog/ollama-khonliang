"""
Debate orchestration for agent disagreement resolution.

Detects high-confidence disagreements among agent votes and orchestrates
structured debate rounds. Agents challenge each other directly with
specific arguments rather than receiving flat vote summaries.

Debate trigger conditions:
    - Two agents vote differently, both with confidence > threshold
    - An agent votes VETO
    - Max 2 rounds (configurable). No convergence -> Adjudicator decides (if configured)

Usage:
    orchestrator = DebateOrchestrator(agents=agents, adjudicator=my_adj)
    updated_votes, adjudicated = await orchestrator.run_debate(votes, subject, ctx)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from khonliang.consensus.models import AgentVote, ConsensusResult
from khonliang.gateway.messages import AgentMessage

if TYPE_CHECKING:
    from khonliang.debate.adjudicator import BaseAdjudicator

logger = logging.getLogger(__name__)


@dataclass
class DebateConfig:
    """Configuration for debate orchestration."""

    disagreement_threshold: float = 0.6
    max_rounds: int = 2
    challenge_timeout: float = 30.0
    enabled: bool = True
    # Assigned-stance debate: agents argue assigned positions
    assigned_stances: bool = False
    stance_labels: Tuple[str, str] = ("advocate", "skeptic")


class DebateOrchestrator:
    """
    Orchestrates structured debates between disagreeing agents.

    Detects disagreements in votes, identifies the most productive
    debate pair, and runs challenge/response rounds using the gateway.

    Example:
        >>> orchestrator = DebateOrchestrator(agents={"a": agent_a, "b": agent_b})
        >>> updated_votes, adjudicated = await orchestrator.run_debate(votes, "review this PR")
    """

    def __init__(
        self,
        gateway: Optional[Any] = None,
        agents: Optional[Dict[str, Any]] = None,
        config: Optional[DebateConfig] = None,
        adjudicator: Optional["BaseAdjudicator"] = None,
    ):
        self.gateway = gateway
        self.agents = agents or {}
        self.config = config or DebateConfig()
        self.adjudicator = adjudicator
        self._debate_history: List[Dict[str, Any]] = []

    def detect_disagreement(
        self,
        votes: List[AgentVote],
    ) -> Optional[Tuple[AgentVote, AgentVote]]:
        """
        Detect if votes contain a high-confidence disagreement worth debating.

        Returns the two highest-confidence votes that disagree, or None
        if no debate is warranted.
        """
        if len(votes) < 2:
            return None

        # Check for veto first — highest priority
        veto_votes = [v for v in votes if v.action == "VETO"]
        if veto_votes:
            veto = veto_votes[0]
            opposing = [
                v
                for v in votes
                if v.action != "VETO"
                and v.action != "DEFER"
                and v.agent_id != veto.agent_id
            ]
            if opposing:
                challenger = max(opposing, key=lambda v: v.confidence)
                return (veto, challenger)

        # Find highest-confidence disagreement
        threshold = self.config.disagreement_threshold
        high_confidence = [v for v in votes if v.confidence >= threshold]

        if len(high_confidence) < 2:
            return None

        # Group by action
        by_action: Dict[str, List[AgentVote]] = {}
        for vote in high_confidence:
            by_action.setdefault(vote.action, []).append(vote)

        if len(by_action) < 2:
            return None

        # Find the two strongest opposing votes, excluding DEFER
        action_votes = [
            (action, max(vlist, key=lambda v: v.confidence))
            for action, vlist in by_action.items()
            if action != "DEFER"
        ]

        if len(action_votes) < 2:
            return None

        action_votes.sort(key=lambda x: x[1].confidence, reverse=True)
        return (action_votes[0][1], action_votes[1][1])

    def build_challenge(
        self,
        challenger: AgentVote,
        target: AgentVote,
        challenger_stance: str = "",
        target_stance: str = "",
    ) -> str:
        """
        Build a challenge text from one agent to another.

        When assigned_stances is enabled, agents are told to argue from
        a specific position regardless of their natural vote.
        """
        if challenger_stance and target_stance:
            # Assigned-stance mode: tell the target to argue their assigned position
            return (
                f"You are assigned the role of {target_stance}. "
                f"Argue this position thoroughly, even if it differs from your "
                f"initial assessment.\n\n"
                f"The {challenger_stance} argues: "
                f'"{challenger.reasoning}"\n\n'
                f"As the {target_stance}, counter this argument. "
                f"What are the strongest points against their position?"
            )

        # Default: natural disagreement
        return (
            f"You voted {target.action} with {target.confidence:.0%} confidence, "
            f'saying: "{target.reasoning}". '
            f"I voted {challenger.action} with {challenger.confidence:.0%} confidence "
            f'because: "{challenger.reasoning}". '
            f"What's your response to my position?"
        )

    async def run_debate(
        self,
        votes: List[AgentVote],
        subject: str,
        context: Optional[Any] = None,
    ) -> tuple[List[AgentVote], Optional[ConsensusResult]]:
        """
        Run a debate if disagreement is detected.

        If no disagreement or debate is disabled, returns (original_votes, None).
        Otherwise runs up to max_rounds of challenge/response. If debate
        fails to resolve and an adjudicator is configured, returns an
        adjudicated ConsensusResult as the second element.

        Args:
            votes: Original agent votes
            subject: What is being evaluated
            context: Optional additional context

        Returns:
            Tuple of (updated_votes, optional_adjudicated_result).
            Second element is non-None only when adjudicator resolved the conflict.
        """
        if not self.config.enabled:
            return votes, None

        disagreement = self.detect_disagreement(votes)
        if disagreement is None:
            logger.debug(f"No disagreement detected for '{subject[:40]}'")
            return votes, None

        vote_a, vote_b = disagreement
        logger.info(
            f"Debate triggered for '{subject[:40]}': "
            f"{vote_a.agent_id}({vote_a.action}) vs "
            f"{vote_b.agent_id}({vote_b.action})"
        )

        updated_votes = list(votes)
        rounds_completed = 0

        # Determine stance assignments
        stance_a = ""
        stance_b = ""
        if self.config.assigned_stances:
            if len(self.config.stance_labels) != 2:
                raise ValueError(
                    f"stance_labels must have exactly 2 elements, "
                    f"got {len(self.config.stance_labels)}: {self.config.stance_labels}"
                )
            stance_a, stance_b = self.config.stance_labels
            logger.info(
                f"Assigned stances: {vote_a.agent_id}={stance_a}, "
                f"{vote_b.agent_id}={stance_b}"
            )

        for round_num in range(1, self.config.max_rounds + 1):
            # A challenges B
            challenge_text = self.build_challenge(
                vote_a, vote_b,
                challenger_stance=stance_a,
                target_stance=stance_b,
            )
            response_b = await self._send_challenge(
                challenger_id=vote_a.agent_id,
                target_id=vote_b.agent_id,
                challenge_text=challenge_text,
                challenger_vote=vote_a,
                target_vote=vote_b,
                round_num=round_num,
            )

            if response_b is not None:
                vote_b = response_b
                self._update_vote_in_list(updated_votes, vote_b)

            # B challenges A
            challenge_text = self.build_challenge(
                vote_b, vote_a,
                challenger_stance=stance_b,
                target_stance=stance_a,
            )
            response_a = await self._send_challenge(
                challenger_id=vote_b.agent_id,
                target_id=vote_a.agent_id,
                challenge_text=challenge_text,
                challenger_vote=vote_b,
                target_vote=vote_a,
                round_num=round_num,
            )

            if response_a is not None:
                vote_a = response_a
                self._update_vote_in_list(updated_votes, vote_a)

            rounds_completed += 1

            if vote_a.action == vote_b.action:
                logger.info(
                    f"Debate resolved in round {round_num}: "
                    f"both {vote_a.action}"
                )
                break

        resolved = vote_a.action == vote_b.action
        adjudicated_result = None

        # If unresolved and adjudicator available, apply domain criteria
        if not resolved and self.adjudicator is not None:
            logger.info(
                f"Debate unresolved for '{subject[:40]}', invoking adjudicator"
            )
            adj_result = self.adjudicator.adjudicate(
                updated_votes, subject, context
            )
            adjudicated_result = adj_result.to_consensus_result(
                updated_votes, debate_rounds=rounds_completed
            )

        self._debate_history.append(
            {
                "subject": subject[:80],
                "agent_a": vote_a.agent_id,
                "agent_b": vote_b.agent_id,
                "rounds": rounds_completed,
                "resolved": resolved,
                "adjudicated": adjudicated_result is not None,
            }
        )

        return updated_votes, adjudicated_result

    async def _send_challenge(
        self,
        challenger_id: str,
        target_id: str,
        challenge_text: str,
        challenger_vote: AgentVote,
        target_vote: AgentVote,
        round_num: int,
    ) -> Optional[AgentVote]:
        """Send a challenge and get a reconsidered vote.

        Uses config.challenge_timeout as the deadline for the reconsider call.
        """
        import asyncio

        target_agent = self.agents.get(target_id)
        if target_agent is None:
            logger.warning(f"Agent {target_id} not found for debate")
            return None

        challenge_msg = AgentMessage.create_request(
            source_agent=challenger_id,
            target_agent=target_id,
            payload={
                "type": "debate_challenge",
                "challenge": challenge_text,
                "challenger_vote": challenger_vote.to_dict(),
                "round_num": round_num,
            },
        )

        try:
            updated_vote = await asyncio.wait_for(
                target_agent.reconsider(
                    original_vote=target_vote,
                    debate_context=challenge_msg,
                    round_num=round_num,
                ),
                timeout=self.config.challenge_timeout,
            )
            logger.debug(
                f"Debate round {round_num}: {target_id} "
                f"{target_vote.action} -> {updated_vote.action} "
                f"(confidence {target_vote.confidence:.2f}"
                f" -> {updated_vote.confidence:.2f})"
            )
            return updated_vote
        except Exception as e:
            logger.error(f"Debate challenge to {target_id} failed: {e}")
            return None

    @staticmethod
    def _update_vote_in_list(
        votes: List[AgentVote],
        updated_vote: AgentVote,
    ) -> None:
        """Replace a vote in the list with an updated version."""
        for i, vote in enumerate(votes):
            if vote.agent_id == updated_vote.agent_id:
                votes[i] = updated_vote
                return

    def get_debate_history(self) -> List[Dict[str, Any]]:
        """Return a copy of all recorded debate summaries."""
        return list(self._debate_history)

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate debate statistics and current config."""
        total = len(self._debate_history)
        resolved = sum(1 for d in self._debate_history if d["resolved"])
        adjudicated = sum(1 for d in self._debate_history if d.get("adjudicated"))
        return {
            "total_debates": total,
            "resolved": resolved,
            "adjudicated": adjudicated,
            "unresolved": total - resolved - adjudicated,
            "resolution_rate": (resolved + adjudicated) / total if total > 0 else 0,
            "enabled": self.config.enabled,
            "adjudicator_enabled": self.adjudicator is not None,
            "max_rounds": self.config.max_rounds,
            "disagreement_threshold": self.config.disagreement_threshold,
        }
