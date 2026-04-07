"""MCP server for khonliang — expose knowledge, triples, blackboard, and roles."""

from khonliang.mcp.artifacts import CompactConcept, CompactFR, CompactSynthesis
from khonliang.mcp.budget import (
    BUDGET_BRIEF,
    BUDGET_COMPACT,
    BUDGET_FULL,
    ContextBudget,
    fit_to_budget,
)
from khonliang.mcp.compact import (
    brief_or_full,
    compact_entry,
    compact_kv,
    compact_list,
    compact_summary,
    format_response,
    truncate,
)
from khonliang.mcp.compress import compress_for_agent, compress_rule_based
from khonliang.mcp.server import KhonliangMCPServer

__all__ = [
    # Server
    "KhonliangMCPServer",
    # Response formatting
    "compact_list",
    "compact_entry",
    "compact_kv",
    "compact_summary",
    "format_response",
    "truncate",
    "brief_or_full",
    # Budget
    "ContextBudget",
    "fit_to_budget",
    "BUDGET_COMPACT",
    "BUDGET_BRIEF",
    "BUDGET_FULL",
    # Artifacts
    "CompactConcept",
    "CompactFR",
    "CompactSynthesis",
    # Compression
    "compress_for_agent",
    "compress_rule_based",
]
