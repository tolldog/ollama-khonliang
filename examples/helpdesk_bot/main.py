"""
Helpdesk bot — wires together roles, router, and Mattermost.

Usage:
    MATTERMOST_BOT_TOKEN=xxx python -m examples.helpdesk_bot.main

Environment variables:
    MATTERMOST_URL       Mattermost server URL (default: http://localhost:8065)
    MATTERMOST_BOT_TOKEN Bot access token
    OLLAMA_URL           Ollama server URL (default: http://localhost:11434)
"""

import asyncio
import logging
import os

from examples.helpdesk_bot.roles import EscalationRole, KnowledgeRole, TriageRole
from examples.helpdesk_bot.router import HelpdeskRouter
from khonliang import ModelPool
from khonliang.integrations.mattermost import MattermostBot, MattermostMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_handler(pool: ModelPool, router: HelpdeskRouter):
    """Build a synchronous message handler that routes to the right role."""

    roles = {
        "triage": TriageRole(pool),
        "knowledge": KnowledgeRole(pool),
        "escalation": EscalationRole(pool),
    }

    def handle(msg: MattermostMessage) -> str:
        role_name, reason = router.route_with_reason(msg.message)
        logger.info(f"Routing '{msg.message[:60]}' -> {role_name} ({reason})")

        role = roles[role_name]
        result = asyncio.run(role.handle(msg.message, session_id=msg.channel_id))
        return result["response"]

    return handle


def main():
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    mm_url = os.environ.get("MATTERMOST_URL", "http://localhost:8065")

    pool = ModelPool(
        {
            "triage": "llama3.2:3b",
            "knowledge": "qwen2.5:7b",
            "escalation": "llama3.1:8b",
        },
        base_url=ollama_url,
    )

    router = HelpdeskRouter()
    handler = build_handler(pool, router)

    bot = MattermostBot(server_url=mm_url)
    bot.on_mention("support", handler)
    bot.on_direct_message(handler)

    logger.info(f"Connecting to Mattermost at {mm_url}...")
    if not bot.connect():
        logger.error(
            "Failed to connect. Check MATTERMOST_BOT_TOKEN and MATTERMOST_URL."
        )
        return

    logger.info("Helpdesk bot is running. Mention @support in any channel.")
    bot.wait()


if __name__ == "__main__":
    main()
