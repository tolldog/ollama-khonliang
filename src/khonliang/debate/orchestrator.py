"""
Debate orchestration for agent disagreement resolution.

Detects high-confidence disagreements among agent votes and orchestrates
structured debate rounds. Agents challenge each other directly with
specific arguments rather than receiving flat vote summaries.

Debate trigger conditions:
    - Two agents vote differently, both with confidence > threshold
    - An agent votes VETO
    - Max 2 rounds (configurable). No convergence -> ConsensusEngine decides

Usage:
    orchestrator = DebateOrchestrator(gateway, agents)
    updated_votes = await orchestrator.run_debate(votes, subject, context)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from khonliang.consensus.models import AgentVote
from khonliang.gateway.messages import AgentMessage

logger = logging.getLogger(__name__)


@dataclass
class DebateConfig:
    """Configuration for debate orchestration."""

    disagreement_threshold: float = 0.6
    max_rounds: int = 2
    challenge_timeout: float = 30.0
    enabled: bool = True


class DebateOrchestrator:
    """
    Orchestrates structured debates between disagreeing agents.

    Detects disagreements in votes, identifies the most productive
    debate pair, and runs challenge/response rounds using the gateway.

    Example:
        >>> orchestrator = DebateOrchestrator(agents={"a": agent_a, "b": agent_b})
        >>> updated_votes = await orchestrator.run_debate(votes, "review this PR")
    """

    def __init__(
        self,
        gateway: Optional[Any] = None,
        agents: Optional[Dict[str, Any]] = None,
        config: Optional[DebateConfig] = None,
    ):
        self.gateway = gateway
        self.agents = agents or {}
        self.config = config or DebateConfig()
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
    ) -> str:
        """Build a challenge text from one agent to another."""
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
    ) -> List[AgentVote]:
        """
        Run a debate if disagreement is detected.

        If no disagreement or debate is disabled, returns original votes.
        Otherwise runs up to max_rounds of challenge/response, updating
        votes with each agent's reconsidered position.

        Args:
            votes: Original agent votes
            subject: What is being evaluated
            context: Optional additional context

        Returns:
            Updated votes (may be unchanged if no debate or no mind changes)
        """
        if not self.config.enabled:
            return votes

        disagreement = self.detect_disagreement(votes)
        if disagreement is None:
            logger.debug(f"No disagreement detected for '{subject[:40]}'")
            return votes

        vote_a, vote_b = disagreement
        logger.info(
            f"Debate triggered for '{subject[:40]}': "
            f"{vote_a.agent_id}({vote_a.action}) vs "
            f"{vote_b.agent_id}({vote_b.action})"
        )

        updated_votes = list(votes)
        rounds_completed = 0

        for round_num in range(1, self.config.max_rounds + 1):
            # A challenges B
            challenge_text = self.build_challenge(vote_a, vote_b)
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
            challenge_text = self.build_challenge(vote_b, vote_a)
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

        self._debate_history.append(
            {
                "subject": subject[:80],
                "agent_a": vote_a.agent_id,
                "agent_b": vote_b.agent_id,
                "rounds": rounds_completed,
                "resolved": vote_a.action == vote_b.action,
            }
        )

        return updated_votes

    async def _send_challenge(
        self,
        challenger_id: str,
        target_id: str,
        challenge_text: str,
        challenger_vote: AgentVote,
        target_vote: AgentVote,
        round_num: int,
    ) -> Optional[AgentVote]:
        """Send a challenge and get a reconsidered vote."""
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
            updated_vote = await target_agent.reconsider(
                original_vote=target_vote,
                debate_context=challenge_msg,
                round_num=round_num,
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
        return list(self._debate_history)

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._debate_history)
        resolved = sum(1 for d in self._debate_history if d["resolved"])
        return {
            "total_debates": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "resolution_rate": resolved / total if total > 0 else 0,
            "enabled": self.config.enabled,
            "max_rounds": self.config.max_rounds,
            "disagreement_threshold": self.config.disagreement_threshold,
        }
