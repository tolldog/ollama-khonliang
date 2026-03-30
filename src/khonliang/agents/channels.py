"""
Typed channels for organized agent-to-agent communication.

Channels are named topics that agents subscribe to. Messages are typed
so agents can filter by what they care about.

Usage:
    manager = ChannelManager()
    manager.create_channel("analysis", description="Research analysis")

    manager.subscribe("analysis", "researcher")
    manager.subscribe("analysis", "fact_checker")

    manager.publish("analysis", Message(
        type=MessageType.ANALYSIS,
        sender="researcher",
        content="Found new records for Roger Tolle",
    ))
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Generic message types for agent communication."""

    ANALYSIS = "analysis"
    VOTE = "vote"
    QUESTION = "question"
    CHALLENGE = "challenge"
    RESPONSE = "response"
    SIGNAL = "signal"
    SUMMARY = "summary"
    ALERT = "alert"


@dataclass
class ChannelMessage:
    """A message in a channel."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    channel: str = ""
    message_type: MessageType = MessageType.ANALYSIS
    sender: str = ""
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    reply_to: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "message_id": self.message_id,
            "channel": self.channel,
            "message_type": self.message_type.value,
            "sender": self.sender,
            "content": self.content,
            "data": self.data,
            "reply_to": self.reply_to,
            "timestamp": self.timestamp,
        }


@dataclass
class Channel:
    """A named communication channel."""

    name: str
    description: str = ""
    subscribers: Set[str] = field(default_factory=set)
    allowed_types: Optional[Set[MessageType]] = None  # None = all
    history: List[ChannelMessage] = field(default_factory=list)
    max_history: int = 100


class ChannelManager:
    """
    Manages channels and message routing between agents.

    Agents subscribe to channels and receive messages published to them.
    Handlers are called when messages arrive on subscribed channels.
    """

    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        # Key: (agent_id, channel_name) so handlers are channel-specific
        self._handlers: Dict[tuple, List[Callable]] = {}

    def create_channel(
        self,
        name: str,
        description: str = "",
        allowed_types: Optional[Set[MessageType]] = None,
    ) -> Channel:
        """Create a new channel."""
        channel = Channel(
            name=name,
            description=description,
            allowed_types=allowed_types,
        )
        self._channels[name] = channel
        return channel

    def get_channel(self, name: str) -> Optional[Channel]:
        """Return a channel by name, or None if not found."""
        return self._channels.get(name)

    def list_channels(self) -> List[str]:
        """Return all channel names."""
        return list(self._channels.keys())

    def subscribe(
        self,
        channel_name: str,
        agent_id: str,
        handler: Optional[Callable[[ChannelMessage], None]] = None,
    ) -> bool:
        """Subscribe an agent to a channel."""
        channel = self._channels.get(channel_name)
        if not channel:
            return False
        channel.subscribers.add(agent_id)
        if handler:
            key = (agent_id, channel_name)
            self._handlers.setdefault(key, []).append(handler)
        return True

    def unsubscribe(self, channel_name: str, agent_id: str) -> bool:
        """Unsubscribe an agent from a channel. Returns False if channel not found."""
        channel = self._channels.get(channel_name)
        if not channel:
            return False
        channel.subscribers.discard(agent_id)
        self._handlers.pop((agent_id, channel_name), None)
        return True

    def publish(self, channel_name: str, message: ChannelMessage) -> int:
        """
        Publish a message to a channel.

        Returns number of subscribers notified.
        """
        channel = self._channels.get(channel_name)
        if not channel:
            return 0

        # Check allowed types
        if (
            channel.allowed_types
            and message.message_type not in channel.allowed_types
        ):
            logger.debug(
                f"Message type {message.message_type} not allowed "
                f"on channel {channel_name}"
            )
            return 0

        message.channel = channel_name

        # Store in history
        channel.history.append(message)
        if len(channel.history) > channel.max_history:
            channel.history = channel.history[-channel.max_history:]

        # Notify subscribers (count unique subscribers, not handler invocations)
        notified = 0
        for agent_id in channel.subscribers:
            key = (agent_id, channel_name)
            handlers = self._handlers.get(key, [])
            if not handlers:
                continue
            agent_notified = False
            for handler in handlers:
                try:
                    handler(message)
                    agent_notified = True
                except Exception as e:
                    logger.warning(
                        f"Handler error for {agent_id} on "
                        f"{channel_name}: {e}"
                    )
            if agent_notified:
                notified += 1

        return notified

    def get_history(
        self,
        channel_name: str,
        count: int = 20,
        message_type: Optional[MessageType] = None,
    ) -> List[ChannelMessage]:
        """Get recent messages from a channel."""
        channel = self._channels.get(channel_name)
        if not channel:
            return []

        messages = channel.history
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        return messages[-count:]
