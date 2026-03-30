"""
Mattermost bot integration.

Connects to Mattermost as a bot user via WebSocket, dispatches @mentions
to registered handlers, and supports threaded replies and typing indicators.

Requires: pip install ollama-khonliang[mattermost]

Example:
    bot = MattermostBot(
        server_url="http://localhost:8065",
        bot_token="your-bot-token",
    )
    bot.on_mention("support", lambda msg: handle(msg))
    bot.on_direct_message(lambda msg: handle(msg))
    bot.connect()
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

logger = logging.getLogger(__name__)


@dataclass
class MattermostMessage:
    """Represents an incoming message from Mattermost."""

    id: str
    channel_id: str
    user_id: str
    username: str
    message: str
    root_id: str = ""
    post_id: str = ""
    channel_type: str = ""  # D=DM, O=public, P=private, G=group

    @property
    def is_thread_reply(self) -> bool:
        """True if this message is a reply within a thread."""
        return bool(self.root_id)

    @property
    def is_direct_message(self) -> bool:
        """True if this message was sent as a direct message."""
        return self.channel_type == "D"

    def get_mentions(self) -> List[str]:
        """Extract @mentioned usernames from the message text."""
        return re.findall(r"@(\w+)", self.message)


class MattermostBot:
    """
    Mattermost bot that connects via WebSocket and dispatches to handlers.

    Handlers registered with on_mention() are called when the bot's name
    (or any registered agent name) is @mentioned in a channel.

    Direct messages are handled by on_direct_message() or fall through to
    the first registered mention handler as a default.

    Example:
        bot = MattermostBot(server_url=..., bot_token=...)

        @bot.on_mention("assistant")
        def handle(msg):
            return f"You said: {msg.message}"

        bot.connect()
        bot.wait()
    """

    def __init__(
        self,
        server_url: str,
        bot_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.bot_token = bot_token or os.environ.get("MATTERMOST_BOT_TOKEN")
        self.username = username
        self.password = password

        self._session = requests.Session()
        self._ws: Optional["websocket.WebSocketApp"] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._connected = False
        self._reconnect = True

        self.user_id: Optional[str] = None
        self.bot_username: Optional[str] = None

        self._mention_handlers: Dict[str, Callable] = {}
        self._dm_handler: Optional[Callable] = None
        self._message_handlers: List[Callable] = []

        if not HAS_WEBSOCKET:
            logger.warning(
                "websocket-client not installed. "
                "Install with: pip install ollama-khonliang[mattermost]"
            )

    # --- Registration API ---

    def on_mention(self, agent_name: str, handler: Optional[Callable] = None):
        """
        Register a handler for @agent_name mentions.

        Can be used as a decorator or called directly:
            bot.on_mention("support", my_handler)

            @bot.on_mention("support")
            def my_handler(msg): ...
        """
        if handler is not None:
            self._mention_handlers[agent_name.lower()] = handler
            return handler

        def decorator(fn):
            self._mention_handlers[agent_name.lower()] = fn
            return fn
        return decorator

    def on_direct_message(self, handler: Callable):
        """Register a handler for direct messages (no @mention needed)."""
        self._dm_handler = handler
        return handler

    def on_message(self, handler: Callable):
        """Register a handler called for every message (mention or not)."""
        self._message_handlers.append(handler)
        return handler

    # --- Sending ---

    def post_message(
        self, channel_id: str, message: str, root_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Post a message to a channel, optionally as a thread reply."""
        try:
            data: Dict = {"channel_id": channel_id, "message": message}
            if root_id:
                data["root_id"] = root_id
            return self._api_post("/posts", data)
        except Exception as e:
            logger.error(f"Failed to post message: {e}")
            return None

    def reply_to(self, msg: MattermostMessage, response: str) -> Optional[Dict]:
        """Reply to a message in its thread."""
        root_id = msg.root_id or msg.post_id
        return self.post_message(msg.channel_id, response, root_id)

    def send_typing(self, channel_id: str, parent_id: Optional[str] = None) -> bool:
        """Send a typing indicator to a channel. Returns True on success."""
        if not self._ws or not self._connected:
            return False
        try:
            self._ws.send(json.dumps({
                "seq": int(time.time() * 1000) % 1_000_000,
                "action": "user_typing",
                "data": {
                    "channel_id": channel_id,
                    **({"parent_id": parent_id} if parent_id else {}),
                },
            }))
            return True
        except Exception as e:
            logger.warning(f"Failed to send typing indicator: {e}")
            return False

    def typing_context(
        self, channel_id: str, parent_id: Optional[str] = None, interval: float = 3.0
    ) -> "TypingContext":
        """Context manager that keeps the typing indicator alive during long operations."""
        return TypingContext(self, channel_id, parent_id, interval)

    def typing_context_for_message(
        self, msg: MattermostMessage, interval: float = 3.0
    ) -> "TypingContext":
        """Return a typing context manager scoped to the message's channel and thread."""
        return self.typing_context(msg.channel_id, msg.root_id or msg.post_id, interval)

    # --- Connection ---

    def authenticate(self) -> bool:
        """Authenticate with the Mattermost server. Returns True on success."""
        try:
            if self.bot_token:
                user = self._api_get("/users/me")
            else:
                resp = self._session.post(
                    f"{self.server_url}/api/v4/users/login",
                    json={"login_id": self.username, "password": self.password},
                )
                resp.raise_for_status()
                user = resp.json()
                self.bot_token = resp.headers.get("Token")

            self.user_id = user["id"]
            self.bot_username = user["username"]
            logger.info(f"Authenticated as {self.bot_username}")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def connect(self) -> bool:
        """Authenticate and open a WebSocket connection. Returns True on success."""
        if not self.bot_token and not (self.username and self.password):
            logger.error("No authentication credentials provided")
            return False
        if not self.authenticate():
            return False
        self._reconnect = True
        self._connect_websocket()
        time.sleep(1)
        return self._connected

    def disconnect(self):
        """Close the WebSocket connection and stop reconnection."""
        self._reconnect = False
        if self._ws:
            self._ws.close()
        self._connected = False

    def wait(self):
        """Block until disconnected (useful in main scripts)."""
        while self._reconnect:
            time.sleep(1)

    @property
    def is_connected(self) -> bool:
        """True if connected or actively attempting to reconnect."""
        return self._connected or self._reconnect

    # --- Internal ---

    def _api_get(self, endpoint: str) -> Dict:
        resp = self._session.get(
            f"{self.server_url}/api/v4{endpoint}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def _api_post(self, endpoint: str, data: Dict) -> Dict:
        resp = self._session.post(
            f"{self.server_url}/api/v4{endpoint}",
            headers=self._headers(),
            json=data,
        )
        resp.raise_for_status()
        return resp.json()

    def _headers(self) -> Dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.bot_token}",
        }

    def _connect_websocket(self):
        if not HAS_WEBSOCKET:
            logger.error("websocket-client not installed")
            return

        ws_url = (
            self.server_url.replace("http://", "ws://").replace("https://", "wss://")
            + "/api/v4/websocket"
        )
        logger.info(f"Connecting WebSocket: {ws_url}")

        self._ws = websocket.WebSocketApp(
            ws_url,
            header={"Authorization": f"Bearer {self.bot_token}"},
            on_message=self._on_ws_message,
            on_open=self._on_ws_open,
            on_close=self._on_ws_close,
            on_error=self._on_ws_error,
        )
        self._ws_thread = threading.Thread(
            target=lambda: self._ws.run_forever(ping_interval=60, ping_timeout=30),
            daemon=True,
            name="mattermost-ws",
        )
        self._ws_thread.start()

    def _on_ws_open(self, ws):
        logger.info("WebSocket connected")
        self._connected = True
        ws.send(json.dumps({
            "seq": 1, "action": "authentication_challenge",
            "data": {"token": self.bot_token},
        }))

    def _on_ws_close(self, ws, code, msg):
        logger.info(f"WebSocket closed: {code}")
        self._connected = False
        if self._reconnect:
            time.sleep(5)
            self._connect_websocket()

    def _on_ws_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_ws_message(self, ws, raw: str):
        try:
            data = json.loads(raw)
            if data.get("event") != "posted":
                return

            post = json.loads(data.get("data", {}).get("post", "{}"))
            if post.get("user_id") == self.user_id:
                return  # ignore our own messages

            msg = MattermostMessage(
                id=post.get("id", ""),
                channel_id=post.get("channel_id", ""),
                user_id=post.get("user_id", ""),
                username=data.get("data", {}).get("sender_name", "").lstrip("@"),
                message=post.get("message", ""),
                root_id=post.get("root_id", ""),
                post_id=post.get("id", ""),
                channel_type=data.get("data", {}).get("channel_type", ""),
            )
            self._dispatch(msg)
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def _dispatch(self, msg: MattermostMessage):
        handled = False

        for mention in msg.get_mentions():
            handler = self._mention_handlers.get(mention.lower())
            if handler:
                self.send_typing(msg.channel_id, msg.root_id or msg.post_id)
                try:
                    response = handler(msg)
                    if response:
                        self.reply_to(msg, response)
                    handled = True
                except Exception as e:
                    logger.error(f"Error in @{mention} handler: {e}")
                    self.reply_to(msg, f"Error: {e}")
                    handled = True

        if not handled and msg.is_direct_message:
            self.send_typing(msg.channel_id, msg.root_id or msg.post_id)
            handler = self._dm_handler or (
                next(iter(self._mention_handlers.values()), None)
                if self._mention_handlers else None
            )
            if handler:
                try:
                    response = handler(msg)
                    if response:
                        self.reply_to(msg, response)
                except Exception as e:
                    logger.error(f"Error in DM handler: {e}")
                    self.reply_to(msg, f"Error: {e}")

        for handler in self._message_handlers:
            try:
                response = handler(msg)
                if response:
                    self.reply_to(msg, response)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")


class TypingContext:
    """Context manager that keeps the typing indicator alive during long operations."""

    def __init__(
        self,
        bot: MattermostBot,
        channel_id: str,
        parent_id: Optional[str] = None,
        interval: float = 3.0,
    ):
        self.bot = bot
        self.channel_id = channel_id
        self.parent_id = parent_id
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.bot.send_typing(self.channel_id, self.parent_id)
        self._thread = threading.Thread(
            target=lambda: [
                self.bot.send_typing(self.channel_id, self.parent_id)
                for _ in iter(lambda: self._stop.wait(self.interval), True)
            ],
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return False
