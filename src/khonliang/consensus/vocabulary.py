"""
Action vocabulary — user-defined action sets for consensus voting.

Users define their own action enums that register with ConsensusEngine.
Provides validation, default weights, and blocking action identification.

Usage:
    # Define domain-specific actions
    vocab = ActionVocabulary(
        actions=["approve", "reject", "defer", "veto"],
        blocking={"veto"},
        default_weights={"approve": 1.0, "reject": 1.0, "defer": 0.5},
    )

    # Validate and normalize votes
    assert vocab.is_valid("approve")
    assert vocab.is_blocking("veto")
    normalized = vocab.validate_vote("Approve")  # -> "approve"

    # Use built-in vocabularies
    from khonliang.consensus.vocabulary import DEFAULT_VOCABULARY, BINARY_VOCABULARY

Note: ConsensusEngine integration is planned but not yet implemented.
Standalone validation and weight lookup work today.
"""

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class ActionVocabulary:
    """
    Defines valid actions for a consensus domain.

    Subclass or instantiate with your action set.
    ConsensusEngine uses this for validation and blocking detection.
    """

    actions: list = field(default_factory=lambda: [
        "approve", "reject", "defer", "veto"
    ])
    blocking: Set[str] = field(default_factory=lambda: {"veto"})
    default_weights: Dict[str, float] = field(default_factory=dict)
    display_names: Dict[str, str] = field(default_factory=dict)

    def is_valid(self, action: str) -> bool:
        """Check if an action is in the vocabulary."""
        return action.lower() in {a.lower() for a in self.actions}

    def is_blocking(self, action: str) -> bool:
        """Check if an action blocks consensus (e.g. VETO)."""
        return action.lower() in {b.lower() for b in self.blocking}

    def get_weight(self, action: str) -> float:
        """Get the default weight for an action."""
        return self.default_weights.get(action.lower(), 1.0)

    def get_display(self, action: str) -> str:
        """Get display name for an action."""
        return self.display_names.get(
            action.lower(), action.upper()
        )

    def validate_vote(self, action: str) -> str:
        """Validate and normalize an action string. Raises ValueError if invalid."""
        normalized = action.lower()
        if not self.is_valid(normalized):
            valid = ", ".join(self.actions)
            raise ValueError(
                f"Invalid action '{action}'. Valid: {valid}"
            )
        return normalized


# Common vocabularies

DEFAULT_VOCABULARY = ActionVocabulary(
    actions=["approve", "reject", "defer", "veto"],
    blocking={"veto"},
)

BINARY_VOCABULARY = ActionVocabulary(
    actions=["approve", "reject", "veto"],
    blocking={"veto"},
)
