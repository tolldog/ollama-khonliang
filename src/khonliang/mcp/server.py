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
        guides: Optional dict of guide_name -> description to register
            at construction time. Merged with default guides (catalog).
            Can also add guides later via ``add_guide()``.
    """

    def __init__(
        self,
        knowledge_store: Optional[Any] = None,
        triple_store: Optional[Any] = None,
        blackboard: Optional[Any] = None,
        session: Optional[Any] = None,
        roles: Optional[Dict[str, Any]] = None,
        router: Optional[Any] = None,
        guides: Optional[Dict[str, str]] = None,
    ):
        self.knowledge_store = knowledge_store
        self.triple_store = triple_store
        self.blackboard = blackboard
        self.session = session
        self.roles = roles or {}
        self.router = router
        # Copy default guides so subclasses can extend without mutating base
        self.guide_tools: Dict[str, str] = dict(self._default_guides)
        if guides:
            self.guide_tools.update(guides)

    # Default guide tools — immutable class-level template.
    _default_guides: Dict[str, str] = {
        "catalog": "lists all tools, start here",
        "coding_guide": "development workflow and API reference",
        "response_modes": "how tool responses work: compact/brief/full detail parameter",
    }

    def add_guide(self, name: str, description: str) -> None:
        """Register a guide tool so it appears in the catalog.

        Can be called before or after ``create_app()`` — the catalog
        reads ``guide_tools`` at call time.
        """
        self.guide_tools[name] = description

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
        self._register_coding_guide_tool(mcp)
        self._register_response_modes_tool(mcp)
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

    # -- Coding guide --

    def _register_coding_guide_tool(self, mcp: Any) -> None:
        server = self

        @mcp.tool()
        def coding_guide(topic: str = "workflow") -> str:
            """Development workflow and khonliang API reference.

            topic="workflow": full dev lifecycle (FR → spec → code → PR → merge)
            topic="structure": project directory layout for specs and milestones
            topic="branches": worktree/branch conventions for milestones
            topic="reviews": code review and PR review loop
            topic="reviewer": guide for LLMs reviewing code and PRs
            topic="api": khonliang module map and quick-start
            topic="client": OllamaClient / OpenAIClient usage
            topic="roles": BaseRole, BaseRouter, ModelPool
            topic="knowledge": KnowledgeStore, TripleStore, tiers
            topic="consensus": AgentTeam, ConsensusEngine, voting
            topic="mcp": extending KhonliangMCPServer, adding guides
            """
            sections = server._coding_guide_sections()
            if topic in sections:
                return sections[topic]
            available = ", ".join(sorted(sections.keys()))
            return f"Unknown topic '{topic}'. Available: {available}"

    @staticmethod
    def _coding_guide_sections() -> Dict[str, str]:
        return {
            "workflow": (
                "# Development Workflow\n"
                "\n"
                "Every piece of work follows this lifecycle:\n"
                "\n"
                "  1. Feature Request (FR) — start from a tracked feature request.\n"
                "     See features/FR-###/request.md for the project's FR directory.\n"
                "  2. Spec — write specs/MS-##/spec.md before coding.\n"
                "     Review existing specs first. Get spec reviewed (specs/MS-##/review.md).\n"
                "  3. Milestone — create milestones/MS-##/milestone.md.\n"
                "     One milestone covers one or more FRs. Get reviewed.\n"
                "  4. Branch — create an isolated branch for the milestone.\n"
                "     All coding on branches, never on main/master.\n"
                "  5. Code — implement with a review loop at each step.\n"
                "     Write code, run tests, get review, iterate.\n"
                "  6. Code review — respond to all review comments.\n"
                "     Resolve every conversation before proceeding.\n"
                "  7. PR — open a pull request against main.\n"
                "     First commit is auto-reviewed by Copilot.\n"
                "  8. PR review — respond to all PR review feedback.\n"
                "     Request re-review: @copilot comment after pushing fixes.\n"
                "  9. Merge — merge to main.\n"
                "     Update FR status and any release tracking records.\n"
                "\n"
                "Key rules:\n"
                "  - Spec first, then milestone, then code. Don't skip steps.\n"
                "  - Review loop at every gate — don't proceed without approval.\n"
                "  - One milestone = one or more FRs, one branch.\n"
                "\n"
                "Call coding_guide(topic=...) for details on each phase."
            ),
            "structure": (
                "# Project Directory Structure\n"
                "\n"
                "Every project follows this layout for features, specs, and milestones:\n"
                "\n"
                "  $PROJECT/\n"
                "    features/\n"
                "      FR-001/\n"
                "        request.md        — the feature request\n"
                "        evaluation.md     — feasibility/impact assessment\n"
                "      FR-002/\n"
                "        request.md\n"
                "        evaluation.md\n"
                "    specs/\n"
                "      MS-001/\n"
                "        spec.md          — the specification\n"
                "        review.md        — spec review notes and feedback\n"
                "      MS-002/\n"
                "        spec.md\n"
                "        review.md\n"
                "    milestones/\n"
                "      MS-001/\n"
                "        milestone.md     — milestone plan, scope, and acceptance criteria\n"
                "        review.md        — milestone review notes\n"
                "        code_review.md   — code review feedback and resolutions\n"
                "      MS-002/\n"
                "        milestone.md\n"
                "        review.md\n"
                "        code_review.md\n"
                "\n"
                "## Features (features/FR-###/)\n"
                "  request.md     — what is needed and why.\n"
                "  evaluation.md  — feasibility, impact, and effort assessment.\n"
                "\n"
                "## Specs (specs/MS-##/)\n"
                "  spec.md    — what to build and why. Written before coding starts.\n"
                "  review.md  — feedback on the spec. Gaps, concerns, approvals.\n"
                "\n"
                "## Milestones (milestones/MS-##/)\n"
                "  milestone.md    — implementation plan. Maps FRs to tasks.\n"
                "  review.md       — review of the milestone plan.\n"
                "  code_review.md  — captures code review feedback during development.\n"
                "\n"
                "Each milestone covers one or more Feature Requests (FRs).\n"
                "The MS number ties specs, milestones, and branches together."
            ),
            "branches": (
                "# Branches & Worktrees\n"
                "\n"
                "All coding is done on isolated branches.\n"
                "Never commit directly to main/master.\n"
                "\n"
                "## Conventions\n"
                "  Follow the repository's branch naming conventions.\n"
                "  Common patterns: feature/description, fix/description, prefix/description.\n"
                "\n"
                "## Worktrees (optional)\n"
                "  git worktree add ../feature-description -b feature/description\n"
                "  cd ../feature-description\n"
                "  # Work in isolation, then clean up after merge:\n"
                "  git worktree remove ../feature-description\n"
                "\n"
                "## After merge\n"
                "  1. Update any issue, changelog, or release tracking records\n"
                "  2. Clean up the feature branch and worktree if used"
            ),
            "reviews": (
                "# Review Loop\n"
                "\n"
                "Every step has a review gate — don't skip ahead.\n"
                "\n"
                "## Code review (during development)\n"
                "  - Write code for one logical piece\n"
                "  - Run tests, trunk check, and type checker\n"
                "  - Present for review\n"
                "  - Address all feedback before moving to next piece\n"
                "\n"
                "## PR creation\n"
                "  git checkout -b feature/description  # (or use worktree)\n"
                "  gh pr create --base main --title 'Short descriptive title'\n"
                "\n"
                "## Copilot review\n"
                "  The first commit is auto-reviewed by Copilot.\n"
                "  After pushing fixes from that review, request re-review\n"
                "  by adding a PR comment:\n"
                "    @copilot please review this PR, I did the following:\n"
                "    - <summary of changes>\n"
                "\n"
                "## PR review\n"
                "  - Respond to every review comment\n"
                "  - Resolve every conversation\n"
                "  - Keep changes scoped — no domain bleed\n"
                "  - Run trunk check before final push\n"
                "  - Iterate until approved, then merge"
            ),
            "reviewer": (
                "# Reviewer Guide (for LLMs)\n"
                "\n"
                "When reviewing code or PRs, follow these principles:\n"
                "\n"
                "## IMPORTANT: Read-only for code\n"
                "  As a reviewer, you ONLY generate review files:\n"
                "    specs/MS-##/review.md\n"
                "    milestones/MS-##/review.md\n"
                "    milestones/MS-##/code_review.md\n"
                "  Everything else — source code, tests, configs — is READ ONLY.\n"
                "  Never modify code directly. Put all feedback in review files.\n"
                "\n"
                "## What to check\n"
                "  1. Scope — do the changes match the spec/milestone?\n"
                "     No domain bleed, no unrelated refactors, no feature creep.\n"
                "  2. Correctness — does the code do what it claims?\n"
                "     Check edge cases, error handling, off-by-one, null/empty.\n"
                "  3. Tests — are new behaviors covered? Are existing tests updated?\n"
                "  4. Consistency — does it follow the project's existing patterns?\n"
                "     Naming, structure, error handling style, logging conventions.\n"
                "  5. Safety — no secrets, no injection, no unvalidated external input.\n"
                "  6. Documentation — are docstrings/Args updated for API changes?\n"
                "\n"
                "## How to give feedback\n"
                "  - Be specific: reference file:line, quote the code, explain why.\n"
                "  - Distinguish blocking issues from suggestions/nits.\n"
                "  - Suggest concrete fixes, not just 'this is wrong'.\n"
                "  - One concern per comment — don't bundle unrelated issues.\n"
                "  - Acknowledge what's done well, not just problems.\n"
                "\n"
                "## What NOT to do\n"
                "  - Don't rewrite the PR — suggest, don't impose.\n"
                "  - Don't flag style issues the linter would catch (trunk handles that).\n"
                "  - Don't request changes outside the PR's scope.\n"
                "  - Don't block on hypothetical future problems.\n"
                "  - Don't repeat what another reviewer already said.\n"
                "\n"
                "## Review artifacts\n"
                "  Capture review feedback in milestones/MS-##/code_review.md\n"
                "  so decisions and resolutions are preserved for future reference."
            ),
            "api": (
                "# khonliang — module map\n"
                "\n"
                "Multi-agent LLM orchestration for Ollama (and OpenAI-compatible APIs).\n"
                "\n"
                "## Install\n"
                "  pip install ollama-khonliang\n"
                "\n"
                "## Modules\n"
                "  khonliang.client        — OllamaClient (async, typed errors, retry)\n"
                "  khonliang.openai_client — OpenAIClient (same interface, OpenAI backend)\n"
                "  khonliang.protocols     — LLMClient protocol (type against this)\n"
                "  khonliang.pool          — ModelPool (role → model mapping)\n"
                "  khonliang.health        — ModelHealthTracker (cooldown on failures)\n"
                "  khonliang.roles         — BaseRole, BaseRouter, SessionContext\n"
                "  khonliang.knowledge     — KnowledgeStore (3-tier), TripleStore\n"
                "  khonliang.consensus     — AgentTeam, ConsensusEngine, voting\n"
                "  khonliang.gateway       — Blackboard (agent coordination)\n"
                "  khonliang.mcp           — KhonliangMCPServer (MCP tool exposure)\n"
                "  khonliang.rag           — RAG pipeline\n"
                "  khonliang.training      — Feedback & training utilities\n"
                "\n"
                "## Minimal example\n"
                "  from khonliang import OllamaClient\n"
                "  client = OllamaClient(model='llama3.1:8b')\n"
                "  response = await client.generate('Hello!')\n"
                "\n"
                "Call coding_guide(topic=...) for details on each subsystem."
            ),
            "client": (
                "# OllamaClient & OpenAIClient\n"
                "\n"
                "  from khonliang import OllamaClient, GenerationResult\n"
                "\n"
                "  client = OllamaClient(\n"
                "      model='llama3.1:8b',\n"
                "      base_url='http://localhost:11434',  # default\n"
                "  )\n"
                "\n"
                "  # Simple generation\n"
                "  text = await client.generate('Summarize this', system='Be concise')\n"
                "\n"
                "  # With token metrics\n"
                "  result: GenerationResult = await client.generate_with_metrics('Hello')\n"
                "  # result.text, result.prompt_eval_count, result.eval_count\n"
                "\n"
                "  # Streaming\n"
                "  async for chunk in client.stream_generate('Tell me a story'):\n"
                "      print(chunk, end='')\n"
                "\n"
                "  # Structured JSON output\n"
                "  data = await client.generate_json('Extract entities', schema={...})\n"
                "\n"
                "  # OpenAI-compatible backend (same interface)\n"
                "  from khonliang import OpenAIClient\n"
                "  oai = OpenAIClient(model='gpt-4o', api_key='sk-...')\n"
                "\n"
                "  # Type-annotate with the protocol for backend-agnostic code\n"
                "  from khonliang import LLMClient\n"
                "  async def summarize(client: LLMClient, text: str) -> str: ...\n"
                "\n"
                "## Error handling\n"
                "  from khonliang import LLMTimeoutError,"
                " LLMUnavailableError, LLMModelNotFoundError\n"
                "  # All extend LLMError. Retry is built in (3 attempts, exponential backoff).\n"
                "  # Per-model timeouts: 3b=30s, 8b=60s, 32b=300s (configurable)."
            ),
            "roles": (
                "# Roles, routing, and model pools\n"
                "\n"
                "  from khonliang import BaseRole, BaseRouter, ModelPool\n"
                "\n"
                "## ModelPool — map roles to models\n"
                "  pool = ModelPool({\n"
                "      'triage': 'llama3.2:3b',\n"
                "      'researcher': 'qwen2.5:7b',\n"
                "      'analyst': 'deepseek-r1:32b',\n"
                "  })\n"
                "  model = pool.get('triage')  # 'llama3.2:3b'\n"
                "\n"
                "## BaseRole — subclass to define agent behavior\n"
                "  class TriageRole(BaseRole):\n"
                "      async def handle(self, message, session_id='default'):\n"
                "          response = await self.client.generate("
                "message, system=self.system_prompt)\n"
                "          return {'response': response, 'metadata': {'role': 'triage'}}\n"
                "\n"
                "## BaseRouter — route messages to roles\n"
                "  class MyRouter(BaseRouter):\n"
                "      def route(self, message: str) -> str:\n"
                "          if 'urgent' in message.lower(): return 'triage'\n"
                "          return 'researcher'\n"
                "\n"
                "## ModelHealthTracker — auto-cooldown on failures\n"
                "  from khonliang import ModelHealthTracker\n"
                "  tracker = ModelHealthTracker()\n"
                "  tracker.record_failure('deepseek-r1:32b')\n"
                "  if tracker.is_healthy('deepseek-r1:32b'): ..."
            ),
            "knowledge": (
                "# Knowledge & Triples\n"
                "\n"
                "## KnowledgeStore — three-tier storage\n"
                "  from khonliang.knowledge.store import KnowledgeStore, KnowledgeEntry, Tier\n"
                "\n"
                "  store = KnowledgeStore('data/knowledge.db')  # auto-creates schema\n"
                "\n"
                "  # Tiers:\n"
                "  #   Tier.AXIOM (1)    — immutable rules, always in context\n"
                "  #   Tier.IMPORTED (2) — user-provided docs, agent-managed\n"
                "  #   Tier.DERIVED (3)  — agent-built, tagged with provenance\n"
                "\n"
                "  store.add(KnowledgeEntry(\n"
                "      id='rule-001', tier=Tier.AXIOM,\n"
                "      title='Safety rule', content='Always verify before acting',\n"
                "      scope='global', source='human', confidence=1.0,\n"
                "  ))\n"
                "\n"
                "  results = store.search('safety', scope='global', limit=5)\n"
                "  context = store.build_context(query='safety', max_chars=1000)\n"
                "  axioms = store.get_axioms()  # always-on Tier 1 entries\n"
                "\n"
                "## TripleStore — semantic triples (subject-predicate-object)\n"
                "  from khonliang.knowledge.triples import TripleStore\n"
                "\n"
                "  triples = TripleStore('data/knowledge.db')\n"
                "  triples.add('transformer', 'enables', 'attention_mechanism', confidence=0.9)\n"
                "  results = triples.get(subject='transformer', limit=10)\n"
                "  context = triples.build_context(subjects=['transformer'], max_triples=20)"
            ),
            "consensus": (
                "# Consensus — multi-agent voting\n"
                "\n"
                "  from khonliang.consensus import AgentTeam, ConsensusEngine\n"
                "\n"
                "  team = AgentTeam(agents=[\n"
                "      {'name': 'reviewer', 'role': reviewer_role, 'weight': 1.0},\n"
                "      {'name': 'advocate', 'role': advocate_role, 'weight': 1.0},\n"
                "  ])\n"
                "  result = await team.evaluate('Is this document reliable?')\n"
                "  # result.decision, result.votes, result.confidence\n"
                "\n"
                "## Adaptive weights\n"
                "  from khonliang.consensus import AdaptiveWeightManager, OutcomeTracker\n"
                "  # Tracks agent accuracy over time and adjusts voting weights\n"
                "\n"
                "## Action vocabulary\n"
                "  from khonliang.consensus import ActionVocabulary\n"
                "  # Standardizes agent actions (APPROVE/REJECT/DEFER) for consistent voting"
            ),
            "mcp": (
                "# Extending KhonliangMCPServer\n"
                "\n"
                "  from khonliang.mcp import KhonliangMCPServer\n"
                "  from khonliang.knowledge.store import KnowledgeStore\n"
                "\n"
                "  # Basic server with components\n"
                "  server = KhonliangMCPServer(\n"
                "      knowledge_store=KnowledgeStore('data/knowledge.db'),\n"
                "      guides={'my_guide': 'explains my domain tools'},\n"
                "  )\n"
                "\n"
                "  # Or add guides after construction\n"
                "  server.add_guide('my_guide', 'explains my domain tools')\n"
                "\n"
                "  # Create the FastMCP app\n"
                "  mcp = server.create_app()\n"
                "\n"
                "  # Register your own tools on the app\n"
                "  @mcp.tool()\n"
                "  def my_custom_tool(query: str) -> str:\n"
                "      '''My domain-specific tool.'''\n"
                "      return 'result'\n"
                "\n"
                "  # Guides appear in catalog output (marked with *)\n"
                "  # They can be added before or after create_app()\n"
                "\n"
                "  # Run\n"
                "  mcp.run(transport='stdio')  # or 'streamable-http'\n"
                "\n"
                "  # CLI: python -m khonliang.mcp"
                " --transport stdio --db data/knowledge.db"
            ),
        }

    # -- Response modes guide --

    def _register_response_modes_tool(self, mcp: Any) -> None:

        @mcp.tool()
        def response_modes() -> str:
            """How tool responses work. Read this before calling other tools.

            Most tools accept a detail parameter: compact, brief, or full.
            This guide explains each mode and when to use it.
            """
            return (
                "# Response Modes\n"
                "\n"
                "Most khonliang MCP tools accept a `detail` parameter\n"
                "that controls response verbosity. Use it to save tokens.\n"
                "\n"
                "## Three modes\n"
                "  compact — key=value pairs, pipe-delimited, no prose.\n"
                "            For agent loops where every token costs.\n"
                "            Example: tools=12|guides=catalog,coding_guide"
                "|categories=knowledge,triples\n"
                "\n"
                "  brief   — one-line-per-item with small headers. (default)\n"
                "            Example:\n"
                "              === GUIDES (start here) ===\n"
                "                * catalog: lists all tools\n"
                "              === KNOWLEDGE ===\n"
                "                knowledge_search: Search the knowledge store\n"
                "\n"
                "  full    — rich detail with parameters and context.\n"
                "            For humans who need the complete picture.\n"
                "\n"
                "## Usage\n"
                "  catalog(detail='compact')                # minimal\n"
                "  knowledge_search(query, detail='brief')   # default\n"
                "  blackboard_read(section, detail='full')   # everything\n"
                "\n"
                "## When to use each\n"
                "  - Exploring autonomously: start with compact\n"
                "  - Presenting to user: use brief\n"
                "  - User asks for details: use full\n"
                "  - Iterative narrowing: compact → brief → full\n"
                "\n"
                "## Building tools with response modes\n"
                "  from khonliang.mcp.compact import "
                "format_response, compact_summary\n"
                "\n"
                "  @mcp.tool()\n"
                "  def my_status(detail: str = 'brief') -> str:\n"
                "      return format_response(\n"
                "          compact_fn=lambda: compact_summary({\n"
                "              'agents': 6, 'active': 3,\n"
                "          }),\n"
                "          brief_fn=lambda: '6 agents, 3 active',\n"
                "          full_fn=lambda: render_full(),\n"
                "          detail=detail,\n"
                "      )\n"
                "\n"
                "## Helpers (khonliang.mcp.compact)\n"
                "  compact_summary(dict)  — key=val|key=val\n"
                "  compact_list(items, format_fn)  — one line per item\n"
                "  compact_entry(id, title, status)  — single entry\n"
                "  compact_kv(dict)  — comma-separated key=value\n"
                "  truncate(text, max_chars)  — ellipsis truncation"
            )

    # -- Catalog --

    def _register_catalog_tool(self, mcp: Any) -> None:
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

            # Discover registered tools and read guides at call time
            tool_map = server._discover_tools(mcp)
            guides = server.guide_tools

            return format_response(
                compact_fn=lambda: compact_summary({
                    "tools": len(tool_map),
                    "guides": ",".join(sorted(guides.keys())),
                    "categories": ",".join(sorted(
                        {cat for cat, _, _ in tool_map.values()}
                    )),
                }),
                brief_fn=lambda: server._format_catalog_brief(tool_map, guides),
                full_fn=lambda: server._format_catalog_full(tool_map, guides),
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
            logger.warning(
                f"Unknown tool storage type: {type(tool_list).__name__}. "
                f"Catalog may be incomplete."
            )
            items = []

        from khonliang.mcp.compact import truncate

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

            tools[name] = (category, truncate(doc, 100), params)

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
            "coding_guide": "meta",
            "response_modes": "meta",
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
