"""
Agent gateway — routes messages between agents via Redis streams.

Provides AgentGateway, a pub/sub message router with:
- Per-agent Redis streams for async communication
- Session tracking
- Activation-rule registration (query via is_agent_active)
- Fail-open semantics (Redis outage does not block agents)

Example:

    gw = AgentGateway(redis_url="redis://localhost:6379")
    await gw.start()
    await gw.send("analyst", AgentMessage(agent_id="analyst", content="hello"))
    messages = await gw.receive("analyst", count=10)
    await gw.stop()
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from khonliang.agents.activation import ActivationRule, ActivationTracker
from khonliang.gateway.messages import AgentMessage, GatewayMetrics

logger = logging.getLogger(__name__)


class AgentGateway:
    """
    Routes AgentMessages through Redis streams.

    Each agent gets its own stream at ``{stream_prefix}{agent_id}``.
    If Redis is unavailable, the gateway degrades gracefully (fail-open):
    sends are dropped with a warning, receives return empty lists.

    Args:
        redis_url:      Redis connection URL
        stream_prefix:  Prefix for all agent streams (default "khonliang:agent:")
        max_stream_len: Max entries per stream (older entries are trimmed)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_prefix: str = "khonliang:agent:",
        max_stream_len: int = 1000,
    ):
        self.redis_url = redis_url
        self.stream_prefix = stream_prefix
        self.max_stream_len = max_stream_len

        self._redis = None
        self._running = False
        self._agents: Dict[str, Any] = {}
        self._activation_tracker = ActivationTracker()
        self._metrics = GatewayMetrics()
        self._message_handlers: Dict[str, Callable] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Redis and start the gateway."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url, decode_responses=True
            )
            await self._redis.ping()
            logger.info(f"Gateway connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) — gateway running in degraded mode")
            self._redis = None
        self._running = True

    async def stop(self) -> None:
        """Disconnect from Redis and stop the gateway."""
        self._running = False
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
        logger.info("Gateway stopped")

    @property
    def is_running(self) -> bool:
        """Whether the gateway is currently running."""
        return self._running

    @property
    def metrics(self) -> GatewayMetrics:
        """Current gateway metrics (sent/received/failed counters)."""
        return self._metrics

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        agent: Any = None,
        activation_rule: Optional[ActivationRule] = None,
    ) -> None:
        """
        Register an agent with the gateway.

        Args:
            agent_id:        Unique agent identifier
            agent:           Agent instance (duck-typed — any object with agent_id)
            activation_rule: Optional activation rule for scheduling
        """
        self._agents[agent_id] = agent
        if activation_rule is not None:
            self._activation_tracker.register(agent_id, activation_rule)
        self._metrics.agents_connected = len(self._agents)
        logger.info(f"Registered agent '{agent_id}'")

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent. Returns True if it existed."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._activation_tracker.unregister(agent_id)
            self._metrics.agents_connected = len(self._agents)
            logger.info(f"Unregistered agent '{agent_id}'")
            return True
        return False

    def is_agent_active(self, agent_id: str) -> bool:
        """Check if an agent is currently active per its activation rule."""
        return self._activation_tracker.is_active(agent_id)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send(self, agent_id: str, message: AgentMessage) -> bool:
        """
        Send a message to an agent's stream.

        Returns True on success, False on failure (fail-open).
        """
        if self._redis is None:
            logger.warning(f"Redis unavailable — dropping message to '{agent_id}'")
            self._metrics.record_failure()
            return False

        stream_key = f"{self.stream_prefix}{agent_id}"
        try:
            payload = json.dumps(message.to_dict())
            await self._redis.xadd(
                stream_key,
                {"data": payload},
                maxlen=self.max_stream_len,
            )
            self._metrics.record_sent()
            return True
        except Exception as e:
            logger.warning(f"Failed to send to '{agent_id}': {e}")
            self._metrics.record_failure()
            return False

    async def receive(
        self,
        agent_id: str,
        count: int = 10,
        last_id: str = "0-0",
    ) -> tuple[List[AgentMessage], str]:
        """
        Read messages from an agent's stream after last_id.

        Uses exclusive lower bound so the same last_id entry is not returned
        again. Returns (messages, new_last_id) so callers can advance their
        cursor.

        Args:
            agent_id: Agent whose stream to read
            count:    Max messages to return
            last_id:  Read messages strictly after this stream ID

        Returns:
            Tuple of (messages, last_entry_id). On failure returns ([], last_id).
        """
        if self._redis is None:
            return [], last_id

        stream_key = f"{self.stream_prefix}{agent_id}"
        # Exclusive lower bound: "(" prefix tells Redis XRANGE to exclude
        min_id = f"({last_id}" if last_id != "0-0" else "-"
        try:
            entries = await self._redis.xrange(
                stream_key, min=min_id, count=count
            )
            messages = []
            new_last_id = last_id
            for entry_id, fields in entries:
                new_last_id = entry_id
                try:
                    data = json.loads(fields["data"])
                    messages.append(AgentMessage.from_dict(data))
                    self._metrics.record_received()
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Malformed message in '{stream_key}': {e}")
            return messages, new_last_id
        except Exception as e:
            logger.warning(f"Failed to read from '{agent_id}': {e}")
            return [], last_id

    async def broadcast(
        self, message: AgentMessage, agent_ids: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Send a message to multiple agents.

        Args:
            message:   The message to broadcast
            agent_ids: Target agents (default: all registered agents)

        Returns dict of {agent_id: success}.
        """
        targets = agent_ids or list(self._agents.keys())
        results = {}
        for aid in targets:
            results[aid] = await self.send(aid, message)
        return results

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def on_message(self, agent_id: str, handler: Callable) -> None:
        """Register a callback for messages arriving at an agent's stream."""
        self._message_handlers[agent_id] = handler

    async def poll(self, agent_id: str, interval: float = 1.0) -> None:
        """
        Poll an agent's stream and dispatch to the registered handler.

        Advances the cursor after each batch so messages are not re-processed.
        Runs until the gateway is stopped.
        """
        handler = self._message_handlers.get(agent_id)
        if handler is None:
            logger.warning(f"No handler registered for '{agent_id}'")
            return

        last_id = "0-0"
        while self._running:
            try:
                messages, last_id = await self.receive(
                    agent_id, count=10, last_id=last_id
                )
                for msg in messages:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(msg)
                        else:
                            handler(msg)
                    except Exception as e:
                        logger.error(f"Handler error for '{agent_id}': {e}")
            except Exception as e:
                logger.warning(f"Poll error for '{agent_id}': {e}")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a tracked session."""
        self._sessions[session_id] = {
            "created_at": time.time(),
            "metadata": metadata or {},
            "message_count": 0,
        }
        self._metrics.active_sessions = len(self._sessions)

    def end_session(self, session_id: str) -> bool:
        """End a session. Returns True if it existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._metrics.active_sessions = len(self._sessions)
            return True
        return False

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self._agents.keys())

    def get_status(self) -> Dict[str, Any]:
        """Return gateway status summary."""
        return {
            "running": self._running,
            "redis_connected": self._redis is not None,
            "agents": self.list_agents(),
            "sessions": len(self._sessions),
            "metrics": self._metrics.to_dict(),
            "activation_stats": self._activation_tracker.get_stats(),
        }
