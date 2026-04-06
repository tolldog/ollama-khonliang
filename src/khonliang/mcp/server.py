"""
khonliang MCP server — expose knowledge, triples, blackboard, and roles.

Provides MCP tools and resources so external LLMs (Claude, GPT, etc.)
can interact with the same knowledge stores and role system as local agents.

Example:
    from khonliang.mcp.server import KhonliangMCPServer
    from khonliang.knowledge.store import KnowledgeStore

    server = KhonliangMCPServer(knowledge_store=KnowledgeStore("data/knowledge.db"))
    server.create_app().run(transport="stdio")
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class KhonliangMCPServer:
    """MCP server exposing khonliang capabilities to external LLMs.

    Pass any combination of components — tools are only registered
    for components that are provided.

    Args:
        knowledge_store: Optional KnowledgeStore for search/ingest/context
        triple_store: Optional TripleStore for semantic triples
        blackboard: Optional Blackboard for agent coordination
        session: Optional SessionContext for conversation state
        roles: Optional dict of role_name -> BaseRole instances
        router: Optional BaseRouter for message routing
    """

    def __init__(
        self,
        knowledge_store: Optional[Any] = None,
        triple_store: Optional[Any] = None,
        blackboard: Optional[Any] = None,
        session: Optional[Any] = None,
        roles: Optional[Dict[str, Any]] = None,
        router: Optional[Any] = None,
    ):
        self.knowledge_store = knowledge_store
        self.triple_store = triple_store
        self.blackboard = blackboard
        self.session = session
        self.roles = roles or {}
        self.router = router

    def create_app(self):
        """Create a FastMCP app with tools and resources registered."""
        from mcp.server.fastmcp import FastMCP

        logger.info("Creating khonliang MCP server")
        mcp = FastMCP("khonliang")

        self._register_knowledge_tools(mcp)
        self._register_triple_tools(mcp)
        self._register_blackboard_tools(mcp)
        self._register_role_tools(mcp)
        self._register_session_tools(mcp)
        self._register_resources(mcp)

        return mcp

    # -- Knowledge tools --

    def _register_knowledge_tools(self, mcp: Any) -> None:
        if not self.knowledge_store:
            return

        store = self.knowledge_store

        @mcp.tool()
        def knowledge_search(
            query: str,
            scope: str = "global",
            max_results: int = 5,
            detail: str = "brief",
        ) -> str:
            """Search the knowledge store. Returns compact results by default.

            detail="brief": one line per result (id | title)
            detail="full": includes content preview, confidence, scope
            """
            from khonliang.mcp.compact import compact_list, truncate

            results = store.search(query, scope=scope, limit=max_results)
            if not results:
                return f"No entries for: {query}"

            if detail == "full":
                lines = [f"{len(results)} entries:"]
                for e in results:
                    tier = e.tier.value if hasattr(e.tier, "value") else e.tier
                    lines.append(
                        f"[{e.id}] {e.title} "
                        f"(T{tier}, {e.confidence:.0%}, {e.scope})\n"
                        f"  {truncate(e.content, 150)}"
                    )
                return "\n".join(lines)

            return compact_list(
                items=results,
                format_fn=lambda e: f"{e.id} | {e.title}",
                header=f"{len(results)} entries:",
                limit=max_results,
            )

        @mcp.tool()
        def knowledge_ingest(
            title: str, content: str, scope: str = "global"
        ) -> str:
            """Add content to the knowledge store as Tier 2 (imported)."""
            try:
                from khonliang.knowledge.store import KnowledgeEntry, Tier

                entry = KnowledgeEntry(
                    id=f"mcp-{uuid.uuid4().hex[:8]}",
                    tier=Tier.IMPORTED,
                    title=title,
                    content=content,
                    scope=scope,
                    source="mcp",
                    confidence=0.8,
                )
                store.add(entry)
                return f"Ingested: {title} (scope={scope})"
            except Exception as e:
                return f"Ingestion failed: {e}"

        @mcp.tool()
        def knowledge_context(
            query: str, scope: str = "global", max_chars: int = 1000
        ) -> str:
            """Build a context string from knowledge for LLM injection."""
            return store.build_context(
                query=query, scope=scope, max_chars=max_chars, include_axioms=True
            )

    # -- Triple tools --

    def _register_triple_tools(self, mcp: Any) -> None:
        if not self.triple_store:
            return

        triples = self.triple_store

        @mcp.tool()
        def triple_add(
            subject: str,
            predicate: str,
            object: str,
            confidence: float = 1.0,
            source: str = "mcp",
        ) -> str:
            """Add a semantic triple (subject-predicate-object)."""
            triples.add(subject, predicate, object, confidence=confidence, source=source)
            return f"Added: {subject} {predicate} {object} ({confidence:.0%})"

        @mcp.tool()
        def triple_query(
            subject: str = "",
            predicate: str = "",
            object: str = "",
            limit: int = 10,
        ) -> str:
            """Query triples. Compact by default: subject predicate object (confidence)."""
            results = triples.get(
                subject=subject or None,
                predicate=predicate or None,
                obj=object or None,
                limit=limit,
            )
            if not results:
                return "No triples found."
            lines = [f"{len(results)} triples:"]
            for t in results:
                lines.append(
                    f"{t.subject} {t.predicate} {t.object} ({t.confidence:.0%})"
                )
            return "\n".join(lines)

        @mcp.tool()
        def triple_context(
            subjects: str = "", max_triples: int = 20
        ) -> str:
            """Build compact context from triples. Pass comma-separated subjects."""
            subject_list = [s.strip() for s in subjects.split(",") if s.strip()] or None
            return triples.build_context(
                subjects=subject_list, max_triples=max_triples
            )

    # -- Blackboard tools --

    def _register_blackboard_tools(self, mcp: Any) -> None:
        if not self.blackboard:
            return

        board = self.blackboard

        @mcp.tool()
        def blackboard_post(
            agent_id: str, section: str, key: str, content: str, ttl: float = 0
        ) -> str:
            """Post an entry to the shared blackboard."""
            effective_ttl = ttl if ttl > 0 else None
            board.post(agent_id, section, key, content, ttl=effective_ttl)
            ttl_str = f", ttl={ttl}s" if effective_ttl else ""
            return f"Posted to {section}/{key}{ttl_str}"

        @mcp.tool()
        def blackboard_read(
            section: str, key: str = "", detail: str = "brief"
        ) -> str:
            """Read blackboard entries. brief=keys+preview, full=with content."""
            from khonliang.mcp.compact import truncate

            entries = board.read(section, key=key or None)
            if not entries:
                if key:
                    return f"Key '{key}' not found in {section}"
                return f"No entries in {section}"
            if detail == "full":
                lines = [f"{section} ({len(entries)}):"]
                for k, v in entries.items():
                    lines.append(f"  [{k}]: {v}")
                return "\n".join(lines)
            # Brief: keys + truncated preview
            lines = [f"{section} ({len(entries)}):"]
            for k, v in entries.items():
                lines.append(f"  {k}: {truncate(str(v), 60)}")
            return "\n".join(lines)

        @mcp.tool()
        def blackboard_context(
            sections: str = "", max_entries: int = 50
        ) -> str:
            """Build formatted context from blackboard entries."""
            section_list = [s.strip() for s in sections.split(",") if s.strip()] or None
            return board.build_context(sections=section_list, max_entries=max_entries)

    # -- Role tools --

    def _register_role_tools(self, mcp: Any) -> None:
        if not self.roles:
            return

        roles = self.roles
        router = self.router

        @mcp.tool()
        async def invoke_role(
            message: str, role: str = "", session_id: str = "mcp"
        ) -> str:
            """Route a message to a role and return the response.

            If role is empty, uses the router to pick the best role.
            """
            role_name = role
            if not role_name and router:
                role_name = router.route(message)
            if not role_name:
                role_name = next(iter(roles), "")
            if role_name not in roles:
                available = ", ".join(roles.keys())
                return f"Role '{role_name}' not found. Available: {available}"

            try:
                result = await roles[role_name].handle(message, session_id=session_id)
                response = result.get("response", "")
                metadata = result.get("metadata", {})
                role_used = metadata.get("role", role_name)
                gen_time = metadata.get("generation_time_ms", 0)
                return f"[{role_used}] ({gen_time}ms)\n{response}"
            except Exception as e:
                return f"Role '{role_name}' failed: {e}"

    # -- Session tools --

    def _register_session_tools(self, mcp: Any) -> None:
        if not self.session:
            return

        session = self.session

        @mcp.tool()
        def get_session_context(max_turns: int = 5) -> str:
            """Get the current session context (recent exchanges, entities, topic)."""
            return session.build_context(max_turns=max_turns)

    # -- Resources --

    def _register_resources(self, mcp: Any) -> None:
        if self.knowledge_store:
            store = self.knowledge_store

            @mcp.resource("knowledge://axioms")
            def knowledge_axioms() -> str:
                """All Tier 1 axioms (always-on system rules)."""
                axioms = store.get_axioms()
                if not axioms:
                    return "No axioms configured."
                lines = ["Axioms:"]
                for a in axioms:
                    lines.append(f"  [{a.id}] {a.content}")
                return "\n".join(lines)

            @mcp.resource("knowledge://stats")
            def knowledge_stats() -> str:
                """Knowledge store statistics."""
                return json.dumps(store.get_stats(), indent=2)

        if self.blackboard:
            board = self.blackboard

            @mcp.resource("blackboard://sections")
            def blackboard_sections() -> str:
                """List of active blackboard sections."""
                sections = board.sections
                if not sections:
                    return "No active sections."
                lines = ["Active sections:"]
                for s in sections:
                    entries = board.read(s)
                    lines.append(f"  {s}: {len(entries)} entries")
                return "\n".join(lines)
