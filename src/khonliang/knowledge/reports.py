"""
Report builder — presents accumulated knowledge and research to users.

Generic base that domain-specific report builders extend.
Summarizes knowledge store state, research findings, and gaps.

Usage:
    builder = ReportBuilder(knowledge_store)
    print(builder.knowledge_report())
    print(builder.session_report())
"""

import time
from typing import Optional

from khonliang.knowledge.store import KnowledgeStore, Tier


class ReportBuilder:
    """
    Builds reports from knowledge store data.

    Subclass to add domain-specific report types. The base class
    provides knowledge and session reports.

    Example:
        builder = ReportBuilder(store)
        print(builder.knowledge_report())
    """

    def __init__(self, knowledge_store: Optional[KnowledgeStore] = None):
        self.store = knowledge_store

    def knowledge_report(self) -> str:
        """Report on the state of the knowledge store."""
        if not self.store:
            return "No knowledge store configured."

        sections = []
        sections.append("# Knowledge Report")
        sections.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
        sections.append("")

        stats = self.store.get_stats()
        sections.append("## Summary")
        sections.append(f"- Total entries: {stats.get('total_entries', 0)}")

        by_tier = stats.get("by_tier", {})
        sections.append(f"- Axioms (Tier 1): {by_tier.get('axiom', 0)}")
        sections.append(f"- Imported (Tier 2): {by_tier.get('imported', 0)}")
        sections.append(f"- Derived (Tier 3): {by_tier.get('derived', 0)}")
        sections.append("")

        by_scope = stats.get("by_scope", {})
        if by_scope:
            sections.append("## By Scope")
            for scope, count in sorted(by_scope.items()):
                sections.append(f"- {scope}: {count}")
            sections.append("")

        # Recent derived — get_by_tier + in-memory sort is acceptable for
        # typical knowledge store sizes (hundreds to low-thousands of entries).
        derived = self.store.get_by_tier(Tier.DERIVED)
        if derived:
            derived.sort(key=lambda e: e.updated_at, reverse=True)
            sections.append(f"## Recent Research ({len(derived)} entries)")
            for entry in derived[:10]:
                sections.append(
                    f"- [{entry.confidence:.0%}] {entry.title} "
                    f"(source: {entry.source})"
                )
            sections.append("")

        # Promoted
        imported = self.store.get_by_tier(Tier.IMPORTED)
        promoted = [e for e in imported if e.source != "system"]
        if promoted:
            sections.append(f"## Validated Knowledge ({len(promoted)} entries)")
            for entry in promoted[:10]:
                sections.append(f"- {entry.title} ({entry.confidence:.0%})")
            sections.append("")

        # Axioms
        axioms = self.store.get_axioms()
        if axioms:
            sections.append(f"## Active Rules ({len(axioms)})")
            for a in axioms:
                sections.append(f"- {a.title}: {a.content[:80]}")
            sections.append("")

        return "\n".join(sections)

    def session_report(self, extra_context: str = "") -> str:
        """Brief summary suitable for showing on connect."""
        sections = []
        sections.append("# Session Summary")
        sections.append("")

        if self.store:
            stats = self.store.get_stats()
            total = stats.get("total_entries", 0)
            derived = stats.get("by_tier", {}).get("derived", 0)
            imported = stats.get("by_tier", {}).get("imported", 0)
            sections.append(
                f"Knowledge: {total} entries "
                f"({imported} verified, {derived} researched)"
            )

            recent = self.store.get_by_tier(Tier.DERIVED)
            recent.sort(key=lambda e: e.updated_at, reverse=True)
            if recent:
                sections.append("")
                sections.append("Recent research:")
                for entry in recent[:5]:
                    sections.append(f"  - {entry.title}")

        if extra_context:
            sections.append("")
            sections.append(extra_context)

        # TODO: Add tests for ReportBuilder (out of scope for this PR).

        return "\n".join(sections)

    def topic_report(self, query: str, scope: Optional[str] = None) -> str:
        """Report on everything known about a topic."""
        if not self.store:
            return "No knowledge store configured."

        entries = self.store.search(query, scope=scope, limit=20)
        if not entries:
            return f"No knowledge found for: {query}"

        sections = []
        sections.append(f"# Topic: {query}")
        sections.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
        sections.append(f"Found {len(entries)} entries")
        sections.append("")

        for entry in entries:
            tier_label = {
                Tier.AXIOM: "RULE",
                Tier.IMPORTED: "VERIFIED",
                Tier.DERIVED: "RESEARCHED",
            }.get(entry.tier, "?")
            sections.append(
                f"## [{tier_label}] {entry.title} "
                f"(confidence: {entry.confidence:.0%})"
            )
            sections.append(f"Source: {entry.source}")
            sections.append(entry.content[:500])
            sections.append("")

        return "\n".join(sections)
