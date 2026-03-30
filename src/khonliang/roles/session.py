"""
Session context for multi-turn conversations.

Provides conversation history and entity tracking that roles can use
to give coherent follow-up answers.

Usage:
    session = SessionContext(session_id="abc123")
    session.add_exchange("Who was Roger Tolle?", "Roger Tolle was born in 1642...", "researcher")
    session.add_exchange("Where did he live?", "He lived in Maryland...", "researcher")

    # Roles see the history
    context = session.build_context(max_turns=5)
    # "Previous conversation:
    #  User: Who was Roger Tolle?
    #  Agent (researcher): Roger Tolle was born in 1642...
    #  User: Where did he live?
    #  Agent (researcher): He lived in Maryland..."
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Exchange:
    """A single user-agent exchange."""

    user_message: str
    agent_response: str
    role: str
    timestamp: float = field(default_factory=time.time)


class SessionContext:
    """
    Conversation context for a session.

    Tracks exchanges, extracted entities, and active topics.
    Passed to roles so they can give coherent multi-turn answers.
    """

    def __init__(self, session_id: str, max_history: int = 20):
        self.session_id = session_id
        self.max_history = max_history
        self.exchanges: List[Exchange] = []
        self.entities: Dict[str, Set[str]] = {
            "names": set(),
            "places": set(),
            "dates": set(),
            "topics": set(),
        }
        self.active_topic: str = ""
        self.created_at: float = time.time()

    def add_exchange(
        self,
        user_message: str,
        agent_response: str,
        role: str,
    ) -> None:
        """Record a conversation exchange."""
        self.exchanges.append(Exchange(
            user_message=user_message,
            agent_response=agent_response,
            role=role,
        ))
        if len(self.exchanges) > self.max_history:
            self.exchanges = self.exchanges[-self.max_history:]

    def add_entities(
        self,
        names: Optional[List[str]] = None,
        places: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
    ) -> None:
        """Track entities mentioned in the conversation."""
        if names:
            self.entities["names"].update(names)
        if places:
            self.entities["places"].update(places)
        if dates:
            self.entities["dates"].update(dates)
        if topics:
            self.entities["topics"].update(topics)

    def set_topic(self, topic: str) -> None:
        """Set the active conversation topic."""
        self.active_topic = topic
        self.entities["topics"].add(topic)

    def build_context(self, max_turns: int = 5) -> str:
        """
        Build a context string from recent conversation history.

        Suitable for injection into an LLM prompt.
        """
        recent = self.exchanges[-max_turns:]
        if not recent:
            return ""

        lines = ["Previous conversation:"]
        for ex in recent:
            lines.append(f"  User: {ex.user_message}")
            # Truncate long responses
            resp = ex.agent_response
            if len(resp) > 200:
                resp = resp[:200] + "..."
            lines.append(f"  Agent ({ex.role}): {resp}")

        if self.active_topic:
            lines.append(f"\nCurrent topic: {self.active_topic}")

        entity_parts = []
        for key, values in self.entities.items():
            if values:
                entity_parts.append(f"{key}: {', '.join(sorted(values)[:5])}")
        if entity_parts:
            lines.append(f"Entities discussed: {'; '.join(entity_parts)}")

        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self.exchanges)

    @property
    def last_role(self) -> str:
        return self.exchanges[-1].role if self.exchanges else ""

    @property
    def last_response(self) -> str:
        return self.exchanges[-1].agent_response if self.exchanges else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "active_topic": self.active_topic,
            "entities": {
                k: sorted(v) for k, v in self.entities.items() if v
            },
            "recent": [
                {
                    "user": ex.user_message[:60],
                    "role": ex.role,
                }
                for ex in self.exchanges[-5:]
            ],
        }
