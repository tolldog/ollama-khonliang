"""
Data models for multi-agent conversations.

Domain-agnostic: supports any multi-agent discussion scenario
(brainstorming, code review, research debate, collaborative planning).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Stance(str, Enum):
    """How firmly an agent holds their position in a conversation.

    Used in subsequent rounds to signal willingness to compromise.
    Helps the manager detect convergence and enables structured negotiation.
    """

    FIRM = "firm"              # Not willing to change position
    FLEXIBLE = "flexible"      # Open to compromise
    CONCEDING = "conceding"    # Moving toward another agent's position
    NEUTRAL = "neutral"        # No strong position yet (default for round 1)


class TurnPolicy(str, Enum):
    """Determines how agents take turns in a conversation.

    ROUND_ROBIN:       Each agent speaks in order, cycling through the list.
    DIRECTED:          The current speaker nominates the next speaker via
                       ``next_speaker`` key in their response metadata.
    MODERATOR:         A designated moderator agent speaks first each round
                       to set direction, then other agents respond sequentially.
    ALL_THEN_ORGANIZE: Round 1 is parallel fan-out (all agents respond
                       simultaneously), subsequent rounds are sequential
                       with full visibility of prior contributions.
    ANY:               Any agent can respond; first valid responder wins the turn.
    """

    ROUND_ROBIN = "round_robin"
    DIRECTED = "directed"
    MODERATOR = "moderator"
    ALL_THEN_ORGANIZE = "all_then_organize"
    ANY = "any"


@dataclass
class ConversationMessage:
    """A single message in a multi-agent conversation.

    Attributes:
        agent_id:  Who sent this message
        content:   The message text
        round_num: Which conversation round (1-indexed)
        metadata:  Arbitrary extra data (e.g. confidence, factors)
        reply_to:  Optional agent_id this message responds to
    """

    agent_id: str
    content: str
    round_num: int = 1
    stance: str = "neutral"       # Expressed stance (what agent shows to group)
    conviction: float = 0.5       # Internal conviction 0.0-1.0 (how sure agent really is)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "round_num": self.round_num,
            "stance": self.stance,
            "conviction": self.conviction,
            "metadata": self.metadata,
            "reply_to": self.reply_to,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ConversationHistory:
    """Full history of a multi-agent conversation.

    Provides helpers for building context prompts from history.
    """

    topic: str
    messages: List[ConversationMessage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    def add(self, message: ConversationMessage) -> None:
        self.messages.append(message)

    def get_round(self, round_num: int) -> List[ConversationMessage]:
        """Get all messages from a specific round."""
        return [m for m in self.messages if m.round_num == round_num]

    def get_agent_messages(self, agent_id: str) -> List[ConversationMessage]:
        """Get all messages from a specific agent."""
        return [m for m in self.messages if m.agent_id == agent_id]

    @property
    def current_round(self) -> int:
        """The highest round number seen so far."""
        if not self.messages:
            return 0
        return max(m.round_num for m in self.messages)

    @property
    def participants(self) -> List[str]:
        """Unique agent IDs that have contributed."""
        seen = []
        for m in self.messages:
            if m.agent_id not in seen:
                seen.append(m.agent_id)
        return seen

    def build_context(self, for_agent: Optional[str] = None) -> str:
        """Build a conversation context string for injection into prompts.

        Args:
            for_agent: If set, labels the agent's own messages as "You"
        """
        lines = [f"Topic: {self.topic}\n"]
        for msg in self.messages:
            label = "You" if msg.agent_id == for_agent else msg.agent_id
            reply = f" (replying to {msg.reply_to})" if msg.reply_to else ""
            lines.append(f"[Round {msg.round_num}] {label}{reply}: {msg.content}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "messages": [m.to_dict() for m in self.messages],
            "rounds": self.current_round,
            "participants": self.participants,
        }


@dataclass
class ConversationConfig:
    """Configuration for a conversation session.

    Attributes:
        turn_policy:   How agents take turns
        max_rounds:    Maximum conversation rounds before termination
        agent_timeout: Timeout per agent response (seconds)
        moderator_id:  Agent ID of the moderator (required for MODERATOR policy)
        terminate_on_consensus: Stop early if all agents agree
    """

    turn_policy: TurnPolicy = TurnPolicy.ROUND_ROBIN
    max_rounds: int = 3
    agent_timeout: float = 30.0
    moderator_id: Optional[str] = None
    terminate_on_consensus: bool = False
