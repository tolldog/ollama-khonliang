"""
Simple WebSocket chat server for khonliang agents.

Provides a lightweight WebSocket interface for interactive chat with
role-based agents. Clients connect, send JSON messages, and receive
JSON responses. The server handles routing, session management, and
optional knowledge indexing via the librarian.

Requires: pip install ollama-khonliang[all] websockets

Usage:
    from khonliang.integrations.websocket_chat import ChatServer

    server = ChatServer(
        roles={"researcher": my_researcher, "triage": my_triage},
        router=my_router,
        librarian=my_librarian,  # optional
    )
    await server.start(host="0.0.0.0", port=8765)

Client protocol (JSON over WebSocket):

    -> {"type": "message", "content": "Who were Timothy's grandparents?"}
    <- {"type": "response", "content": "...", "role": "researcher", "session_id": "..."}

    -> {"type": "message", "content": "check that", "session_id": "abc123"}
    <- {"type": "response", "content": "...", "role": "fact_checker", ...}

    -> {"type": "search", "query": "Toll"}
    <- {"type": "search_results", "results": [...]}

    -> {"type": "feedback", "message_id": "...", "rating": 4}
    <- {"type": "ack"}
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from khonliang.roles.base import BaseRole
from khonliang.roles.router import BaseRouter

logger = logging.getLogger(__name__)

try:
    from websockets.server import serve

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    serve = None  # type: ignore[assignment]


class ChatSession:
    """Tracks state for a connected client."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []
        self.message_count: int = 0

    def add_exchange(self, user_msg: str, agent_msg: str, role: str) -> str:
        """Record a message exchange. Returns a message ID."""
        msg_id = str(uuid.uuid4())[:8]
        self.history.append({
            "id": msg_id,
            "user": user_msg,
            "agent": agent_msg,
            "role": role,
        })
        self.message_count += 1
        return msg_id


class ChatServer:
    """
    WebSocket chat server backed by khonliang agents.

    Handles multiple concurrent clients, each with their own session.
    Routes messages through a BaseRouter to the appropriate role.

    Args:
        roles: Dict of role_name -> BaseRole instance
        router: BaseRouter for message routing
        librarian: Optional Librarian for knowledge indexing
        on_message: Optional callback(session_id, user_msg, agent_response, role)
    """

    def __init__(
        self,
        roles: Dict[str, BaseRole],
        router: BaseRouter,
        librarian: Optional[Any] = None,
        on_message: Optional[Callable] = None,
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package required. Install with: "
                "pip install websockets"
            )

        self.roles = roles
        self.router = router
        self.librarian = librarian
        self.on_message = on_message
        self._sessions: Dict[str, ChatSession] = {}
        self._server = None

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None:  # nosec B104
        """Start the WebSocket server."""
        self._server = await serve(self._handle_client, host, port)
        logger.info(f"Chat server listening on ws://{host}:{port}")
        await self._server.wait_closed()

    async def start_background(
        self, host: str = "0.0.0.0", port: int = 8765  # nosec B104
    ) -> None:
        """Start the server without blocking."""
        self._server = await serve(self._handle_client, host, port)
        logger.info(f"Chat server listening on ws://{host}:{port}")

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Chat server stopped")

    async def _handle_client(self, websocket) -> None:
        """Handle a single WebSocket client connection."""
        session_id = str(uuid.uuid4())[:12]
        session = ChatSession(session_id)
        self._sessions[session_id] = session

        logger.info(f"Client connected: {session_id}")

        # Send welcome
        await websocket.send(json.dumps({
            "type": "connected",
            "session_id": session_id,
            "roles": list(self.roles.keys()),
        }))

        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    response = await self._handle_message(msg, session)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": "Invalid JSON",
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": str(e),
                    }))
        finally:
            del self._sessions[session_id]
            logger.info(f"Client disconnected: {session_id}")

    async def _handle_message(
        self, msg: Dict[str, Any], session: ChatSession
    ) -> Dict[str, Any]:
        """Route and handle an incoming message."""
        msg_type = msg.get("type", "message")

        if msg_type == "message":
            return await self._handle_chat(msg, session)
        elif msg_type == "feedback":
            return self._handle_feedback(msg, session)
        elif msg_type == "history":
            return {
                "type": "history",
                "session_id": session.session_id,
                "messages": session.history,
            }
        else:
            return {"type": "error", "content": f"Unknown type: {msg_type}"}

    async def _handle_chat(
        self, msg: Dict[str, Any], session: ChatSession
    ) -> Dict[str, Any]:
        """Handle a chat message — route to role, get response."""
        content = msg.get("content", "").strip()
        if not content:
            return {"type": "error", "content": "Empty message"}

        # Route
        role_name, reason = self.router.route_with_reason(content)
        role = self.roles.get(role_name)
        if not role:
            role_name = next(iter(self.roles))
            role = self.roles[role_name]

        # Handle
        result = await role.handle(
            content, session_id=session.session_id
        )
        response_text = result.get("response", "")

        # Record exchange
        msg_id = session.add_exchange(content, response_text, role_name)

        # Index into knowledge if librarian is available
        if self.librarian and response_text:
            try:
                self.librarian.index_response(
                    content=response_text,
                    title=f"Response to: {content[:60]}",
                    agent_id=role_name,
                    query=content,
                )
            except Exception as e:
                logger.debug(f"Knowledge indexing failed: {e}")

        # Callback
        if self.on_message:
            try:
                self.on_message(
                    session.session_id, content, response_text, role_name
                )
            except Exception as e:  # nosec B110
                logger.debug(f"on_message callback error: {e}")

        return {
            "type": "response",
            "content": response_text,
            "role": role_name,
            "reason": reason,
            "message_id": msg_id,
            "session_id": session.session_id,
            "metadata": result.get("metadata", {}),
        }

    def _handle_feedback(
        self, msg: Dict[str, Any], session: ChatSession
    ) -> Dict[str, Any]:
        """Handle feedback on a previous response."""
        msg_id = msg.get("message_id")
        rating = msg.get("rating")

        if not msg_id or rating is None:
            return {"type": "error", "content": "Need message_id and rating"}

        # Find the exchange
        exchange = next(
            (h for h in session.history if h["id"] == msg_id), None
        )
        if not exchange:
            return {"type": "error", "content": f"Message {msg_id} not found"}

        # If librarian is available, update confidence based on feedback
        if self.librarian and rating is not None:
            # Rating 4-5 = good, 1-2 = bad
            confidence = min(1.0, max(0.0, rating / 5.0))
            logger.info(
                f"Feedback on {msg_id}: rating={rating}, "
                f"confidence={confidence:.0%}"
            )

        return {"type": "ack", "message_id": msg_id}

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_sessions": self.active_sessions,
            "roles": list(self.roles.keys()),
            "has_librarian": self.librarian is not None,
        }
