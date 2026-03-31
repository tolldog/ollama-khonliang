"""
Run the khonliang MCP server.

Usage:
    python -m khonliang.mcp --transport stdio --db data/knowledge.db
    python -m khonliang.mcp --transport http --port 8080
"""

import argparse
import logging

from khonliang.gateway.blackboard import Blackboard
from khonliang.knowledge.store import KnowledgeStore
from khonliang.knowledge.triples import TripleStore
from khonliang.mcp.server import KhonliangMCPServer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="khonliang MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--db",
        default="data/knowledge.db",
        help="Path to knowledge database (default: data/knowledge.db)",
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    store = KnowledgeStore(args.db)
    triples = TripleStore(args.db)
    board = Blackboard()

    server = KhonliangMCPServer(
        knowledge_store=store,
        triple_store=triples,
        blackboard=board,
    )

    app = server.create_app()

    if args.transport == "stdio":
        logger.info("Starting khonliang MCP server (stdio)")
        app.run(transport="stdio")
    else:
        logger.info(f"Starting khonliang MCP server (http://{args.host}:{args.port})")
        app.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
