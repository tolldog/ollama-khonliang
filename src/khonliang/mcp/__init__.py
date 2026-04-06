"""MCP server for khonliang — expose knowledge, triples, blackboard, and roles."""

from khonliang.mcp.compact import (
    brief_or_full,
    compact_entry,
    compact_kv,
    compact_list,
    truncate,
)
from khonliang.mcp.server import KhonliangMCPServer

__all__ = [
    "KhonliangMCPServer",
    "compact_list",
    "compact_entry",
    "compact_kv",
    "truncate",
    "brief_or_full",
]
