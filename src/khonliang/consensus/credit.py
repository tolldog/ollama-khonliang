"""
Leave-one-out counterfactual credit assignment for consensus agents.

Computes per-agent contribution by replaying past consensuses with
each agent removed. Agents whose votes change the outcome get credit;
agents whose votes are redundant get less.

Based on:
  - C3: Contextual Counterfactual Credit Assignment (arxiv:2603.06859)
  - Shapley-inspired causal influence (Lazy Agents paper)

Usage:
    from khonliang.consensus.credit import compute_agent_credits

    credits = compute_agent_credits(tracker, engine, min_samples=30)
    # {"analyst": 0.35, "reviewer": 0.28, "skeptic": 0.22, ...}
"""

import logging
from typing import Any, Dict, List, Optional

from khonliang.consensus.models import AgentVote, ConsensusResult
from khonliang.consensus.outcomes import OutcomeTracker

logger = logging.getLogger(__name__)


def compute_agent_credits(
    tracker: OutcomeTracker,
    agent_weights: Optional[Dict[str, float]] = None,
    veto_blocks: bool = True,
    min_samples: int = 20,
    limit: int = 500,
) -> Optional[Dict[str, float]]:
    """Compute per-agent credit using leave-one-out counterfactual analysis.

    For each past consensus with a recorded outcome:
      1. Replay the consensus with each agent's vote removed
      2. If removing agent X changes the winning action, X gets credit
         proportional to the outcome quality
      3. If removing agent X doesn't change the action, X is redundant

    Args:
        tracker: OutcomeTracker with recorded consensuses and outcomes
        agent_weights: Current weights for consensus replay (or equal weights)
        veto_blocks: Whether VETO blocks consensus in replay
        min_samples: Minimum outcome records required (returns None if fewer)
        limit: Max records to analyze

    Returns:
        Dict of agent_id -> credit score (0-1, normalized to sum to 1),
        or None if insufficient data.
    """
    records = tracker.get_history(with_outcome_only=True, limit=limit)
    if len(records) < min_samples:
        logger.info(
            "Insufficient data for credit assignment: %d < %d",
            len(records), min_samples,
        )
        return None

    # Collect all agent IDs seen across records
    all_agents: set = set()
    for r in records:
        for v in r.votes:
            aid = v.get("agent_id")
            if aid:
                all_agents.add(aid)

    if not all_agents:
        return None

    # Per-agent accumulators
    credit_scores: Dict[str, float] = {a: 0.0 for a in all_agents}
    participation: Dict[str, int] = {a: 0 for a in all_agents}

    for record in records:
        if record.outcome is None:
            continue

        votes = _reconstruct_votes(record.votes)
        if len(votes) < 2:
            continue

        original_action = record.action
        outcome = record.outcome

        for agent_id in all_agents:
            agent_vote = _find_vote(votes, agent_id)
            if agent_vote is None:
                continue

            participation[agent_id] += 1

            # Replay without this agent
            remaining = [v for v in votes if v.agent_id != agent_id]
            if not remaining:
                continue

            counterfactual = _replay_consensus(
                remaining, agent_weights, veto_blocks,
            )

            # Credit assignment
            credit = _compute_credit(
                original_action=original_action,
                counterfactual_action=counterfactual.action,
                agent_vote=agent_vote,
                outcome=outcome,
            )
            credit_scores[agent_id] += credit

    # Normalize by participation count
    normalized: Dict[str, float] = {}
    for agent_id in all_agents:
        n = participation[agent_id]
        if n > 0:
            normalized[agent_id] = credit_scores[agent_id] / n
        else:
            normalized[agent_id] = 0.0

    # Shift to positive range and normalize to sum to 1
    min_score = min(normalized.values()) if normalized else 0.0
    shifted = {a: s - min_score + 0.01 for a, s in normalized.items()}
    total = sum(shifted.values())
    if total > 0:
        shifted = {a: s / total for a, s in shifted.items()}

    logger.info(
        "Credit assignment from %d records: %s",
        len(records),
        {a: f"{s:.3f}" for a, s in shifted.items()},
    )
    return shifted


def suggest_weights(
    tracker: OutcomeTracker,
    current_weights: Optional[Dict[str, float]] = None,
    veto_blocks: bool = True,
    min_samples: int = 20,
    blend: float = 0.3,
) -> Optional[Dict[str, float]]:
    """Suggest updated agent weights based on outcome history.

    Blends current weights with LOO credit scores. Higher blend
    means more influence from the credit analysis.

    Args:
        tracker: OutcomeTracker with recorded outcomes
        current_weights: Starting weights (equal if not provided)
        veto_blocks: Whether VETO blocks consensus
        min_samples: Minimum samples required
        blend: How much credit influences the result (0=keep current, 1=all credit)

    Returns:
        Suggested weights (sum to 1.0), or None if insufficient data.
    """
    credits = compute_agent_credits(
        tracker, current_weights, veto_blocks, min_samples,
    )
    if credits is None:
        return None

    if current_weights:
        # Blend: new = (1-blend)*current + blend*credit
        suggested = {}
        all_agents = set(current_weights) | set(credits)
        for agent_id in all_agents:
            cur = current_weights.get(agent_id, 1.0 / len(all_agents))
            cred = credits.get(agent_id, cur)
            suggested[agent_id] = (1 - blend) * cur + blend * cred
    else:
        suggested = credits

    # Normalize
    total = sum(suggested.values())
    if total > 0:
        suggested = {a: w / total for a, w in suggested.items()}

    return suggested


def _reconstruct_votes(vote_dicts: List[Dict[str, Any]]) -> List[AgentVote]:
    """Reconstruct AgentVote objects from serialized dicts."""
    votes = []
    for v in vote_dicts:
        try:
            votes.append(AgentVote(
                agent_id=v["agent_id"],
                action=v["action"],
                confidence=v.get("confidence", 0.5),
                reasoning=v.get("reasoning", ""),
                weight=v.get("weight", 1.0),
            ))
        except (KeyError, ValueError):
            continue
    return votes


def _find_vote(
    votes: List[AgentVote], agent_id: str
) -> Optional[AgentVote]:
    """Find a specific agent's vote."""
    for v in votes:
        if v.agent_id == agent_id:
            return v
    return None


def _replay_consensus(
    votes: List[AgentVote],
    agent_weights: Optional[Dict[str, float]],
    veto_blocks: bool,
) -> ConsensusResult:
    """Replay consensus with a subset of votes."""
    from khonliang.consensus.engine import ConsensusEngine

    engine = ConsensusEngine(
        agent_weights=agent_weights,
        veto_blocks=veto_blocks,
    )
    return engine.calculate_consensus(votes)


def _compute_credit(
    original_action: str,
    counterfactual_action: str,
    agent_vote: AgentVote,
    outcome: float,
) -> float:
    """Compute credit for one agent on one decision.

    Credit is positive when:
      - Removing the agent would have changed the action AND
        the original outcome was good (agent helped)
      - Agent voted with the winning action and outcome was good

    Credit is negative when:
      - Agent voted with the winning action but outcome was bad
      - Removing agent wouldn't have changed anything (redundant)
        and outcome was bad
    """
    action_changed = counterfactual_action != original_action
    agent_aligned = agent_vote.action == original_action

    if action_changed:
        # Agent's vote was pivotal — flipped the decision
        if agent_aligned:
            # Agent was on the winning side and was decisive
            return outcome * agent_vote.confidence
        else:
            # Agent was on the losing side but still pivotal
            # (removing a dissenter changed the outcome)
            return -outcome * agent_vote.confidence * 0.5
    else:
        # Agent's vote didn't change the outcome (redundant)
        if agent_aligned:
            # Redundant agreement — small credit proportional to outcome
            return outcome * agent_vote.confidence * 0.1
        else:
            # Dissented but didn't matter — no credit
            return 0.0
