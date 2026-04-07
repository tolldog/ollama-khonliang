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
        # Copy default guides so subclasses can extend without mutating base
        self.guide_tools: Dict[str, str] = dict(self._default_guides)

    # Default guide tools — immutable class-level template.
    _default_guides: Dict[str, str] = {
        "catalog": "lists all tools, start here",
    }

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
        self._register_catalog_tool(mcp)
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
            """Search the knowledge store.

            detail="compact": hits|ids|scope|top (for agent loops)
            detail="brief": one line per result (id | title)
            detail="full": includes content preview, confidence, scope
            """
            from khonliang.mcp.compact import (
                compact_list,
                compact_summary,
                format_response,
                truncate,
            )

            results = store.search(query, scope=scope, limit=max_results)
            if not results:
                return f"No entries for: {query}"

            return format_response(
                compact_fn=lambda: compact_summary({
                    "hits": len(results),
                    "ids": ",".join(e.id[:8] for e in results[:5]),
                    "scope": scope,
                    "top": results[0].title if results else "",
                }),
                brief_fn=lambda: compact_list(
                    items=results,
                    format_fn=lambda e: f"{e.id} | {e.title}",
                    header=f"{len(results)} entries:",
                    limit=max_results,
                ),
                full_fn=lambda: "\n".join(
                    [f"{len(results)} entries:"]
                    + [
                        f"[{e.id}] {e.title} "
                        f"(T{e.tier.value if hasattr(e.tier, 'value') else e.tier}"
                        f", {e.confidence:.0%}, {e.scope})\n"
                        f"  {truncate(e.content, 150)}"
                        for e in results
                    ]
                ),
                detail=detail,
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
            """Read blackboard entries.

            detail="compact": section|keys|names (for agent loops)
            detail="brief": keys + truncated preview
            detail="full": keys + full content
            """
            from khonliang.mcp.compact import (
                compact_summary,
                format_response,
                truncate,
            )

            entries = board.read(section, key=key or None)
            if not entries:
                if key:
                    return f"Key '{key}' not found in {section}"
                return f"No entries in {section}"

            return format_response(
                compact_fn=lambda: compact_summary({
                    "section": section,
                    "keys": len(entries),
                    "names": ",".join(list(entries.keys())[:5]),
                }),
                brief_fn=lambda: "\n".join(
                    [f"{section} ({len(entries)}):"]
                    + [f"  {k}: {truncate(str(v), 60)}" for k, v in entries.items()]
                ),
                full_fn=lambda: "\n".join(
                    [f"{section} ({len(entries)}):"]
                    + [f"  [{k}]: {v}" for k, v in entries.items()]
                ),
                detail=detail,
            )

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

    # -- Catalog --

    def _register_catalog_tool(self, mcp: Any) -> None:
        guide_tools = self.guide_tools
        server = self

        @mcp.tool()
        def catalog(detail: str = "brief") -> str:
            """List all available tools. Start here.

            Guide tools (marked with *) explain how to use subsystems —
            call them before diving into data tools.

            detail="compact": tools|guides|categories counts
            detail="brief": grouped by category with guides highlighted
            detail="full": includes parameters per tool
            """
            from khonliang.mcp.compact import compact_summary, format_response

            # Discover registered tools from the MCP app
            tool_map = server._discover_tools(mcp)

            return format_response(
                compact_fn=lambda: compact_summary({
                    "tools": len(tool_map),
                    "guides": ",".join(sorted(guide_tools.keys())),
                    "categories": ",".join(sorted(
                        {cat for cat, _, _ in tool_map.values()}
                    )),
                }),
                brief_fn=lambda: server._format_catalog_brief(tool_map, guide_tools),
                full_fn=lambda: server._format_catalog_full(tool_map, guide_tools),
                detail=detail,
            )

    def _discover_tools(self, mcp: Any) -> Dict[str, tuple]:
        """Introspect registered MCP tools.

        Returns {name: (category, description, params_str)}.
        """
        tools: Dict[str, tuple] = {}

        # FastMCP stores tools in _tool_manager._tools or _tools depending on version
        tool_list = None
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_list = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_list = mcp._tools
        elif hasattr(mcp, "tools"):
            tool_list = mcp.tools

        if tool_list is None:
            return tools

        # Handle both dict ({name: tool}) and list ([tool]) storage
        if isinstance(tool_list, dict):
            items = tool_list.items()
        elif isinstance(tool_list, (list, tuple)):
            items = [(getattr(t, "name", str(i)), t) for i, t in enumerate(tool_list)]
        else:
            items = []

        for name, tool in items:
            doc = ""
            if hasattr(tool, "description"):
                doc = tool.description or ""
            elif hasattr(tool, "fn") and tool.fn.__doc__:
                doc = tool.fn.__doc__.strip().split("\n")[0]

            # Infer category from tool name prefix
            category = self._infer_category(name)

            # Extract parameter info
            params = ""
            if hasattr(tool, "parameters"):
                param_names = list(tool.parameters.get("properties", {}).keys())
                params = ", ".join(param_names)

            tools[name] = (category, doc[:100], params)

        return tools

    @staticmethod
    def _infer_category(tool_name: str) -> str:
        """Infer category from tool name prefix."""
        prefixes = {
            "knowledge": "knowledge",
            "triple": "triples",
            "blackboard": "blackboard",
            "invoke_role": "roles",
            "get_session": "session",
            "catalog": "meta",
        }
        for prefix, category in prefixes.items():
            if tool_name.startswith(prefix):
                return category
        return "other"

    def _format_catalog_brief(
        self, tool_map: Dict[str, tuple], guide_tools: Dict[str, str]
    ) -> str:
        """Format catalog as grouped list with guides highlighted."""
        # Group by category
        categories: Dict[str, list] = {}
        for name, (cat, desc, _) in tool_map.items():
            categories.setdefault(cat, []).append((name, desc))

        lines = []

        # Guides section first
        guide_names = set(guide_tools.keys())
        if guide_names:
            lines.append("=== GUIDES (start here) ===")
            for name in sorted(guide_names):
                desc = guide_tools.get(name, "")
                if name in tool_map:
                    desc = desc or tool_map[name][1]
                lines.append(f"  * {name}: {desc}")
            lines.append("")

        # Other categories
        for cat in sorted(categories.keys()):
            if cat == "meta":
                continue  # already shown in guides
            tools = categories[cat]
            lines.append(f"=== {cat.upper()} ===")
            for name, desc in sorted(tools):
                marker = " *" if name in guide_names else ""
                lines.append(f"  {name}{marker}: {desc}")
            lines.append("")

        return "\n".join(lines).strip()

    def _format_catalog_full(
        self, tool_map: Dict[str, tuple], guide_tools: Dict[str, str]
    ) -> str:
        """Format catalog with parameter details."""
        categories: Dict[str, list] = {}
        for name, (cat, desc, params) in tool_map.items():
            categories.setdefault(cat, []).append((name, desc, params))

        lines = []
        guide_names = set(guide_tools.keys())

        if guide_names:
            lines.append("=== GUIDES (start here) ===")
            for name in sorted(guide_names):
                desc = guide_tools.get(name, "")
                if name in tool_map:
                    _, tool_desc, params = tool_map[name]
                    desc = desc or tool_desc
                    lines.append(f"  * {name}({params}): {desc}")
                else:
                    lines.append(f"  * {name}: {desc}")
            lines.append("")

        for cat in sorted(categories.keys()):
            if cat == "meta":
                continue
            tools = categories[cat]
            lines.append(f"=== {cat.upper()} ===")
            for name, desc, params in sorted(tools):
                marker = " *" if name in guide_names else ""
                lines.append(f"  {name}({params}){marker}: {desc}")
            lines.append("")

        return "\n".join(lines).strip()

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
