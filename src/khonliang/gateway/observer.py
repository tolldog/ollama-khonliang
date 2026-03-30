"""
Gateway observer — subscribe to agent events and forward them.

Base class for integrations that watch gateway events without
blocking the main pipeline. Fire-and-forget semantics.

Built-in observers:
- LogObserver: logs events to Python logging
- WebhookObserver: forwards to HTTP endpoints

Usage:
    observer = LogObserver(event_types=["vote", "signal"])
    gateway.add_observer(observer)
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Set

import requests

logger = logging.getLogger(__name__)


class BaseObserver(ABC):
    """
    Abstract observer for gateway events.

    Subclass and implement on_event() for custom event handling.
    Observers are fire-and-forget — failures are logged but don't
    block the gateway.
    """

    name: str = "base"

    def __init__(
        self,
        event_types: Optional[Set[str]] = None,
        channels: Optional[Set[str]] = None,
        agents: Optional[Set[str]] = None,
    ):
        self.event_types = event_types  # None = all
        self.channels = channels  # None = all
        self.agents = agents  # None = all

    def should_handle(self, event: Dict[str, Any]) -> bool:
        """Check if this observer cares about this event."""
        if self.event_types:
            event_type = event.get("type", event.get("intent", ""))
            if event_type not in self.event_types:
                return False
        if self.channels:
            channel = event.get("channel", "")
            if not channel or channel not in self.channels:
                return False
        if self.agents:
            sender = event.get("sender", event.get("from_agent_id", ""))
            if sender and sender not in self.agents:
                return False
        return True

    def notify(self, event: Dict[str, Any]) -> None:
        """Called by the gateway. Filters then dispatches."""
        if not self.should_handle(event):
            return
        try:
            self.on_event(event)
        except Exception as e:
            logger.debug(f"Observer {self.name} error: {e}")

    @abstractmethod
    def on_event(self, event: Dict[str, Any]) -> None:
        """Handle an event. Override in subclasses."""
        ...


class LogObserver(BaseObserver):
    """Logs events to Python logging."""

    name = "log"

    def __init__(
        self,
        level: int = logging.INFO,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.level = level

    def on_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type", event.get("intent", "?"))
        sender = event.get("sender", event.get("from_agent_id", "?"))
        channel = event.get("channel", "")
        content = event.get("content", event.get("text", ""))[:80]

        msg = f"[{event_type}] {sender}"
        if channel:
            msg += f" #{channel}"
        if content:
            msg += f": {content}"

        logger.log(self.level, msg)


class WebhookObserver(BaseObserver):
    """Forwards events to an HTTP webhook endpoint."""

    name = "webhook"

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def on_event(self, event: Dict[str, Any]) -> None:
        # Fire-and-forget: run the POST in a daemon thread so we don't
        # block the gateway's notify loop.
        thread = threading.Thread(
            target=self._send, args=(event,), daemon=True
        )
        thread.start()

    def _send(self, event: Dict[str, Any]) -> None:
        try:
            requests.post(
                self.url,
                json=event,
                headers=self.headers,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.debug(f"Webhook {self.url} failed: {e}")


class CallbackObserver(BaseObserver):
    """Calls a function for each matching event."""

    name = "callback"

    def __init__(
        self,
        callback: Callable[[Dict[str, Any]], None],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.callback = callback

    def on_event(self, event: Dict[str, Any]) -> None:
        self.callback(event)
