"""
khonliang CLI — test and evaluate roles from the terminal.

Commands:
    chat      Interactive chat with a role
    route     Show how the router classifies messages
    test      Run a test file of message -> expected_role pairs
    models    List available Ollama models
    health    Check Ollama server health

Usage:
    python -m khonliang.cli --help
    python -m khonliang.cli chat --role triage
    python -m khonliang.cli route "My server is down!"
    python -m khonliang.cli test examples/helpdesk_bot/tests.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional


def _get_ollama_url() -> str:
    import os
    return os.environ.get("OLLAMA_URL", "http://localhost:11434")


def cmd_models(args):
    """List available Ollama models."""
    from khonliang.client import OllamaClient

    async def run():
        client = OllamaClient(base_url=_get_ollama_url())
        try:
            models = await client.list_models()
            if models:
                print("Available models:")
                for m in models:
                    print(f"  {m}")
            else:
                print("No models found. Pull one with: ollama pull llama3.2:3b")
        finally:
            await client.close()

    asyncio.run(run())


def cmd_health(args):
    """Check Ollama server health."""
    from khonliang.client import OllamaClient

    url = _get_ollama_url()
    client = OllamaClient(base_url=url)
    if client.is_available():
        print(f"✓ Ollama is running at {url}")
    else:
        print(f"✗ Cannot reach Ollama at {url}")
        sys.exit(1)


def cmd_chat(args):
    """Interactive chat with a specific role."""
    from khonliang.client import OllamaClient
    from khonliang.pool import ModelPool

    model = args.model or "llama3.1:8b"
    role = args.role or "assistant"
    system = args.system or f"You are a helpful {role}."
    url = _get_ollama_url()

    print(f"Chatting with role '{role}' using model '{model}' at {url}")
    print("Type 'exit' or Ctrl-C to quit.\n")

    pool = ModelPool({role: model}, base_url=url)
    client = pool.get_client(role)

    async def run():
        session_id = "cli-session"
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye.")
                    break

                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue

                print("Assistant: ", end="", flush=True)
                try:
                    response = await client.generate(
                        prompt=user_input,
                        system=system,
                    )
                    print(response.strip())
                except Exception as e:
                    print(f"[Error: {e}]")
        finally:
            await pool.close_all()

    asyncio.run(run())


def cmd_route(args):
    """
    Show how a router classifies a message.

    Loads the router from the path provided (module:ClassName) or uses
    a simple demo router.
    """
    message = args.message
    router_spec = args.router

    if router_spec:
        # Load user-provided router: e.g. "examples.helpdesk_bot.router:HelpdeskRouter"
        module_path, class_name = router_spec.rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        router_cls = getattr(module, class_name)
        router = router_cls()
    else:
        # Demo router
        from khonliang.roles.router import BaseRouter
        router = BaseRouter(fallback_role="default")
        router.register_pattern(r"(?i)urgent|critical|down", "urgent")
        router.register_keywords(["how to", "what is", "explain"], "knowledge")
        print("[Using demo router. Pass --router module:Class to use your own.]\n")

    role, reason = router.route_with_reason(message)
    print(f"Message : {message!r}")
    print(f"Role    : {role}")
    print(f"Reason  : {reason}")


def cmd_test(args):
    """
    Run a test file and report accuracy.

    Test file format (JSONL — one JSON object per line):
        {"message": "My server is down!", "expected_role": "triage"}
        {"message": "How do I reset my password?", "expected_role": "knowledge"}

    Pass --router to use your own router class.
    """
    test_file = Path(args.file)
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        sys.exit(1)

    router_spec = args.router
    if router_spec:
        module_path, class_name = router_spec.rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        router = getattr(module, class_name)()
    else:
        from khonliang.roles.router import BaseRouter
        router = BaseRouter(fallback_role="default")
        print("[Using demo router. Pass --router module:Class to use your own.]\n")

    cases = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    passed = 0
    failed = []
    for case in cases:
        message = case["message"]
        expected = case["expected_role"]
        actual, reason = router.route_with_reason(message)
        if actual == expected:
            passed += 1
        else:
            failed.append((message, expected, actual, reason))

    total = len(cases)
    print(f"Results: {passed}/{total} passed ({100 * passed // total if total else 0}%)\n")

    if failed:
        print("Failures:")
        for msg, exp, act, reason in failed:
            print(f"  FAIL  expected={exp!r:15} got={act!r:15} reason={reason!r}")
            print(f"        message={msg!r}")

    if passed < total:
        sys.exit(1)


def cmd_generate(args):
    """
    Send a one-shot prompt to a model and print the response.

    Useful for quick testing without spinning up a full role.
    """
    from khonliang.client import OllamaClient

    async def run():
        client = OllamaClient(model=args.model, base_url=_get_ollama_url())
        try:
            response = await client.generate(
                prompt=args.prompt,
                system=args.system,
                temperature=args.temperature,
            )
            print(response.strip())
        finally:
            await client.close()

    asyncio.run(run())


def main():
    parser = argparse.ArgumentParser(
        prog="khonliang",
        description="khonliang CLI — test and evaluate LLM roles",
    )
    parser.add_argument(
        "--ollama-url", default=None,
        help="Ollama server URL (default: $OLLAMA_URL or http://localhost:11434)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # models
    sub.add_parser("models", help="List available Ollama models")

    # health
    sub.add_parser("health", help="Check Ollama server health")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat with a role")
    p_chat.add_argument("--role", default="assistant", help="Role name")
    p_chat.add_argument("--model", default=None, help="Model to use")
    p_chat.add_argument("--system", default=None, help="System prompt override")

    # route
    p_route = sub.add_parser("route", help="Show how router classifies a message")
    p_route.add_argument("message", help="Message to classify")
    p_route.add_argument(
        "--router", default=None,
        help="Router class (e.g. examples.helpdesk_bot.router:HelpdeskRouter)",
    )

    # test
    p_test = sub.add_parser("test", help="Run routing accuracy tests from a JSONL file")
    p_test.add_argument("file", help="Path to JSONL test file")
    p_test.add_argument(
        "--router", default=None,
        help="Router class (e.g. examples.helpdesk_bot.router:HelpdeskRouter)",
    )

    # generate
    p_gen = sub.add_parser("generate", help="One-shot prompt to a model")
    p_gen.add_argument("prompt", help="Prompt text")
    p_gen.add_argument("--model", default="llama3.1:8b", help="Model to use")
    p_gen.add_argument("--system", default=None, help="System prompt")
    p_gen.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()

    if args.ollama_url:
        import os
        os.environ["OLLAMA_URL"] = args.ollama_url

    commands = {
        "models": cmd_models,
        "health": cmd_health,
        "chat": cmd_chat,
        "route": cmd_route,
        "test": cmd_test,
        "generate": cmd_generate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
