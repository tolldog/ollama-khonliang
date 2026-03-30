"""
Message types for the agent gateway.

Defines the AgentMessage dataclass (messages flowing through the gateway)
and GatewayMetrics (runtime counters for monitoring).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentMessage:
    """
    A message routed through the gateway to/from an agent.

    Attributes:
        agent_id:    Target or source agent identifier
        content:     The message payload (text, JSON, etc.)
        session_id:  Conversation/session identifier
        metadata:    Arbitrary key-value metadata
        timestamp:   Unix timestamp of creation
        message_id:  Unique message identifier (auto-generated if omitted)
        reply_to:    ID of the message this is replying to
        priority:    0 = normal, higher = more urgent
    """

    agent_id: str
    content: str
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""
    reply_to: Optional[str] = None
    priority: int = 0

    def __post_init__(self):
        if not self.message_id:
            import uuid
            self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "reply_to": self.reply_to,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Deserialize from a dict."""
        return cls(
            agent_id=data["agent_id"],
            content=data["content"],
            session_id=data.get("session_id", ""),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id", ""),
            reply_to=data.get("reply_to"),
            priority=data.get("priority", 0),
        )

    @classmethod
    def create_request(
        cls,
        source_agent: str,
        target_agent: str,
        payload: Dict[str, Any],
    ) -> "AgentMessage":
        """Factory for directed request messages."""
        import json

        return cls(
            agent_id=target_agent,
            content=json.dumps(payload),
            metadata={"source_agent": source_agent, "type": "request"},
        )


class GatewayMetrics:
    """
    Simple in-process counters for gateway activity.

    Thread-safe via GIL for single-increment operations.
    """

    def __init__(self):
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.messages_failed: int = 0
        self.active_sessions: int = 0
        self.agents_connected: int = 0
        self._start_time: float = time.time()

    def record_sent(self) -> None:
        """Increment the sent message counter."""
        self.messages_sent += 1

    def record_received(self) -> None:
        """Increment the received message counter."""
        self.messages_received += 1

    def record_failure(self) -> None:
        """Increment the failed message counter."""
        self.messages_failed += 1

    @property
    def uptime_seconds(self) -> float:
        """Seconds elapsed since this metrics instance was created or reset."""
        return time.time() - self._start_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all counters and uptime to a dict."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_failed": self.messages_failed,
            "active_sessions": self.active_sessions,
            "agents_connected": self.agents_connected,
            "uptime_seconds": self.uptime_seconds,
        }

    def reset(self) -> None:
        """Reset all counters and restart the uptime clock."""
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_failed = 0
        self.active_sessions = 0
        self.agents_connected = 0
        self._start_time = time.time()
