"""
Home network agent — local network awareness with LLM intelligence.

A standalone service that scans your local network for devices and mDNS
services, then answers natural-language questions about what's connected.

Usage:
    python -m examples.home_network.main

    # Interactive mode
    python -m examples.home_network.main --interactive

    # One-shot query
    python -m examples.home_network.main --query "what devices are online?"

    # With Mattermost
    MATTERMOST_BOT_TOKEN=xxx python -m examples.home_network.main --mattermost

Environment variables:
    OLLAMA_URL           Ollama server URL (default: http://localhost:11434)
    MATTERMOST_URL       Mattermost server URL (default: http://localhost:8065)
    MATTERMOST_BOT_TOKEN Bot access token (required for --mattermost)
"""

import argparse
import asyncio
import logging
import os

from examples.home_network.roles import DeviceMonitorRole, NetworkInfoRole
from examples.home_network.router import HomeNetworkRouter
from examples.home_network.scanner import NetworkScanner
from khonliang import ModelPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_service(ollama_url: str):
    """Build the network agent service components."""
    scanner = NetworkScanner()

    pool = ModelPool(
        {
            "network_info": "llama3.2:3b",
            "device_monitor": "qwen2.5:7b",
        },
        base_url=ollama_url,
    )

    roles = {
        "network_info": NetworkInfoRole(pool, scanner=scanner),
        "device_monitor": DeviceMonitorRole(pool, scanner=scanner),
    }

    router = HomeNetworkRouter()

    # Take initial baseline for device monitoring
    count = roles["device_monitor"].snapshot_baseline()
    logger.info(f"Baseline: {count} devices on network")

    return roles, router, scanner


async def handle_query(
    message: str, roles: dict, router: HomeNetworkRouter
) -> str:
    """Route and handle a single query."""
    role_name, reason = router.route_with_reason(message)
    logger.info(f"Routing '{message[:60]}' -> {role_name} ({reason})")

    role = roles[role_name]
    result = await role.handle(message, session_id="cli")
    return result["response"]


def run_interactive(roles: dict, router: HomeNetworkRouter):
    """Interactive REPL mode."""
    print("Home Network Agent — ask about your local network")
    print("Type 'scan' for raw scan, 'baseline' to reset, 'exit' to quit.\n")

    scanner = roles["network_info"].scanner

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break

        if user_input.lower() == "scan":
            print(scanner.get_network_summary())
            continue

        if user_input.lower() == "baseline":
            count = roles["device_monitor"].snapshot_baseline()
            print(f"Baseline updated: {count} devices")
            continue

        if not user_input:
            continue

        response = asyncio.run(handle_query(user_input, roles, router))
        print(f"\nAgent: {response}\n")


def run_mattermost(roles: dict, router: HomeNetworkRouter):
    """Run as a Mattermost bot."""
    from khonliang.integrations.mattermost import MattermostBot, MattermostMessage

    mm_url = os.environ.get("MATTERMOST_URL", "http://localhost:8065")

    def handle(msg: MattermostMessage) -> str:
        return asyncio.run(handle_query(msg.message, roles, router))

    bot = MattermostBot(server_url=mm_url)
    bot.on_mention("network", handle)
    bot.on_direct_message(handle)

    logger.info(f"Connecting to Mattermost at {mm_url}...")
    if not bot.connect():
        logger.error("Failed to connect. Check credentials.")
        return

    logger.info("Network bot running. Mention @network in any channel.")
    bot.wait()


def main():
    parser = argparse.ArgumentParser(description="Home network agent")
    parser.add_argument("--interactive", action="store_true", help="REPL mode")
    parser.add_argument("--mattermost", action="store_true", help="Mattermost bot")
    parser.add_argument("--query", type=str, help="One-shot query")
    args = parser.parse_args()

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    roles, router, scanner = build_service(ollama_url)

    if args.query:
        response = asyncio.run(handle_query(args.query, roles, router))
        print(response)
    elif args.mattermost:
        run_mattermost(roles, router)
    else:
        run_interactive(roles, router)


if __name__ == "__main__":
    main()
