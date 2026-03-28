"""
Session management helpers for the agent gateway.

Convenience functions that operate on an AgentGateway instance:
list sessions, inspect history, send messages, and challenge agents.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from khonliang.gateway.gateway import AgentGateway
from khonliang.gateway.messages import AgentMessage

logger = logging.getLogger(__name__)


def sessions_list(gateway: AgentGateway) -> List[Dict[str, Any]]:
    """
    List all active sessions with their metadata.

    Returns:
        List of dicts with session_id, created_at, metadata, message_count.
    """
    results = []
    for session_id in gateway.list_sessions():
        info = gateway.get_session(session_id)
        if info is not None:
            results.append({"session_id": session_id, **info})
    return results


async def sessions_history(
    gateway: AgentGateway,
    session_id: str,
    agent_id: str,
    count: int = 50,
) -> List[AgentMessage]:
    """
    Retrieve message history for a session from an agent's stream.

    Reads from the beginning of the stream and filters by session_id.

    Args:
        gateway:    The AgentGateway instance
        session_id: Session to filter for
        agent_id:   Agent whose stream to read
        count:      Max messages to retrieve

    Returns:
        List of AgentMessages belonging to the session.
    """
    all_messages, _ = await gateway.receive(agent_id, count=count)
    return [m for m in all_messages if m.session_id == session_id]


async def sessions_send(
    gateway: AgentGateway,
    session_id: str,
    agent_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send a message to an agent within a session context.

    Args:
        gateway:    The AgentGateway instance
        session_id: Session identifier
        agent_id:   Target agent
        content:    Message text
        metadata:   Optional metadata

    Returns:
        True if the message was sent successfully.
    """
    message = AgentMessage(
        agent_id=agent_id,
        content=content,
        session_id=session_id,
        metadata=metadata or {},
    )
    success = await gateway.send(agent_id, message)
    if success:
        session_info = gateway.get_session(session_id)
        if session_info is not None:
            session_info["message_count"] = (
                session_info.get("message_count", 0) + 1
            )
    return success


async def challenge_agent(
    gateway: AgentGateway,
    agent_id: str,
    challenge: str,
    session_id: str = "",
    timeout: float = 30.0,
) -> Optional[AgentMessage]:
    """
    Send a challenge message to an agent and wait for a response.

    Uses a cursor to avoid re-scanning old messages on each poll iteration.

    Args:
        gateway:    The AgentGateway instance
        agent_id:   Target agent
        challenge:  The challenge/question text
        session_id: Optional session context
        timeout:    Max seconds to wait for a response

    Returns:
        The agent's reply AgentMessage, or None on timeout.
    """
    msg = AgentMessage(
        agent_id=agent_id,
        content=challenge,
        session_id=session_id,
        metadata={"type": "challenge"},
    )

    sent = await gateway.send(agent_id, msg)
    if not sent:
        logger.warning(f"Failed to send challenge to '{agent_id}'")
        return None

    # Poll for a reply, advancing cursor each iteration
    deadline = time.time() + timeout
    last_id = "0-0"
    while time.time() < deadline:
        messages, last_id = await gateway.receive(
            agent_id, count=10, last_id=last_id
        )
        for m in messages:
            if m.reply_to == msg.message_id:
                return m
        await asyncio.sleep(0.5)

    logger.warning(f"Challenge to '{agent_id}' timed out after {timeout}s")
    return None
