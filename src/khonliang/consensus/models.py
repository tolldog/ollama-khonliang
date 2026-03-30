"""
Data models for multi-agent consensus voting.

Domain-agnostic: works for any voting scenario (support triage,
content moderation, code review, trading, medical triage, etc.).

Subclass or extend AgentAction with your own decision labels.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentAction(str, Enum):
    """
    Default voting actions. Override or extend for your domain.

    Example — medical triage:
        class TriageAction(str, Enum):
            ADMIT = "ADMIT"
            OBSERVE = "OBSERVE"
            DISCHARGE = "DISCHARGE"
            ESCALATE = "ESCALATE"
    """

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    DEFER = "DEFER"
    VETO = "VETO"


@dataclass
class AgentVote:
    """
    A single agent's vote on a decision.

    Attributes:
        agent_id:   Unique identifier for the voting agent
        action:     The vote (use AgentAction or your own enum/str)
        confidence: Confidence level 0.0–1.0
        reasoning:  Explanation of the vote
        factors:    Arbitrary factor scores that influenced the vote
        weight:     Optional override for this vote's weight in consensus
        created_at: Vote timestamp
    """

    agent_id: str
    action: str
    confidence: float
    reasoning: str
    factors: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0–1.0, got {self.confidence}")

    def weighted_score(self) -> float:
        """Return confidence multiplied by weight."""
        return self.confidence * self.weight

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vote to a plain dict."""
        return {
            "agent_id": self.agent_id,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "factors": self.factors,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentVote":
        """Deserialize a vote from a dict."""
        return cls(
            agent_id=data["agent_id"],
            action=data["action"],
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            factors=data.get("factors", {}),
            weight=data.get("weight", 1.0),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
        )


@dataclass
class ConsensusResult:
    """
    The aggregated result of all agent votes.

    Attributes:
        action:         Final consensus action (majority or weighted winner)
        confidence:     Overall confidence 0.0–1.0
        votes:          All individual votes
        scores:         Per-action weighted scores; empty ({}) when judge_overridden
                        is True (original scores are moved to original_scores)
        reason:         Summary explanation
        debate_rounds:  Number of deliberation rounds (if any)
        judge_overridden: True when a judge_fn overrode the aggregated result
        original_action: Pre-override action (set when judge_overridden is True)
        original_scores: Pre-override weighted scores (set when judge_overridden is True)
    """

    action: str
    confidence: float
    votes: List[AgentVote] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    reason: Optional[str] = None
    summary: Optional[str] = None
    debate_rounds: int = 0
    judge_overridden: bool = False
    original_action: Optional[str] = None
    original_scores: Optional[Dict[str, float]] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_blocked(self) -> bool:
        """True if the consensus action is VETO or any vote is VETO."""
        return self.action == "VETO" or any(v.action == "VETO" for v in self.votes)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize consensus result to a plain dict."""
        return {
            "action": self.action,
            "confidence": self.confidence,
            "votes": [v.to_dict() for v in self.votes],
            "scores": self.scores,
            "reason": self.reason,
            "summary": self.summary,
            "debate_rounds": self.debate_rounds,
            "judge_overridden": self.judge_overridden,
            "original_action": self.original_action,
            "original_scores": self.original_scores,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsensusResult":
        """Deserialize a consensus result from a dict."""
        return cls(
            action=data["action"],
            confidence=data["confidence"],
            votes=[AgentVote.from_dict(v) for v in data.get("votes", [])],
            scores=data.get("scores", {}),
            reason=data.get("reason"),
            summary=data.get("summary"),
            debate_rounds=data.get("debate_rounds", 0),
            judge_overridden=data.get("judge_overridden", False),
            original_action=data.get("original_action"),
            original_scores=data.get("original_scores"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
        )


@dataclass
class AgentPerformance:
    """Tracks prediction accuracy for a voting agent over time."""

    agent_id: str
    total_predictions: int = 0
    correct_predictions: int = 0
    recent_correct: int = 0
    recent_total: int = 0
    high_conf_correct: int = 0
    high_conf_total: int = 0

    @property
    def accuracy(self) -> float:
        """Overall prediction accuracy (0.0-1.0)."""
        return self.correct_predictions / max(1, self.total_predictions)

    @property
    def recent_accuracy(self) -> float:
        """Accuracy over the recent prediction window (0.0-1.0)."""
        return self.recent_correct / max(1, self.recent_total)

    @property
    def calibration_score(self) -> float:
        """Positive when high confidence correlates with high accuracy."""
        high_acc = self.high_conf_correct / max(1, self.high_conf_total)
        baseline = self.accuracy
        return high_acc - baseline

    def to_dict(self) -> Dict[str, Any]:
        """Serialize performance metrics to a plain dict."""
        return {
            "agent_id": self.agent_id,
            "total_predictions": self.total_predictions,
            "accuracy": self.accuracy,
            "recent_accuracy": self.recent_accuracy,
            "calibration_score": self.calibration_score,
        }
