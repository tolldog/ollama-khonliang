"""
Base Adjudicator for criteria-based conflict resolution (GRA framework).

When agents disagree and debate fails to resolve the conflict, the
Adjudicator applies domain-specific criteria to make a final ruling.
This follows the GRA framework (Generator-Reviewer-Adjudicator) from
"A Strategic Coordination Framework of Small LLMs."

Subclass BaseAdjudicator and implement adjudicate() with your domain's
decision criteria. The adjudicator is typically rule-based and
deterministic — no LLM calls needed.

Examples:
    - Trading: RSI, trend, volatility, portfolio exposure
    - Genealogy: name similarity, date proximity, place overlap
    - Code review: test coverage, complexity, security flags
    - Medical triage: symptom severity, vitals, history

Usage:
    class MyAdjudicator(BaseAdjudicator):
        def adjudicate(self, votes, subject, context):
            # Apply domain criteria
            return AdjudicationResult(...)

    orchestrator = DebateOrchestrator(adjudicator=MyAdjudicator())
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from khonliang.consensus.models import AgentVote, ConsensusResult

logger = logging.getLogger(__name__)


@dataclass
class CriterionScore:
    """Score from a single adjudication criterion."""

    name: str
    action: str
    score: float
    reason: str


@dataclass
class AdjudicationResult:
    """Result from an adjudicator's ruling.

    Attributes:
        action:     The adjudicated action (domain-specific string)
        confidence: Confidence in the ruling 0.0-1.0
        reason:     Explanation of the ruling
        criteria:   Per-criterion scores that led to the decision
    """

    action: str
    confidence: float
    reason: str
    criteria: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reason": self.reason,
            "criteria": self.criteria,
        }

    def to_consensus_result(
        self,
        votes: List[AgentVote],
        debate_rounds: int = 0,
    ) -> ConsensusResult:
        """Convert to a ConsensusResult for downstream compatibility."""
        return ConsensusResult(
            action=self.action,
            confidence=self.confidence,
            votes=votes,
            scores=self.criteria,
            reason=f"[adjudicated] {self.reason}",
            debate_rounds=debate_rounds,
        )


class BaseAdjudicator(ABC):
    """Abstract base class for domain-specific adjudicators.

    Subclass this and implement adjudicate() with your domain's
    decision criteria. The method receives the conflicting votes,
    the subject being evaluated, and optional context.

    Example:
        class TradingAdjudicator(BaseAdjudicator):
            def adjudicate(self, votes, subject, context):
                rsi = context.rsi
                if rsi < 30:
                    return AdjudicationResult(action="BUY", ...)
                ...
    """

    @abstractmethod
    def adjudicate(
        self,
        votes: List[AgentVote],
        subject: str,
        context: Optional[Any] = None,
    ) -> AdjudicationResult:
        """Apply domain criteria to resolve a vote conflict.

        Args:
            votes: The conflicting agent votes
            subject: What is being evaluated
            context: Domain-specific context (MarketContext, Person pair, etc.)

        Returns:
            AdjudicationResult with the ruling
        """
        ...
