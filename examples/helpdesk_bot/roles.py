"""
Helpdesk bot roles — domain-specific subclasses of BaseRole.

Demonstrates how to extend BaseRole for a support ticket use case:
- TriageRole:    fast classification (urgency, category, team)
- KnowledgeRole: answers from docs/KB using RAG-style context injection
- EscalationRole: generates human-readable escalation summaries
"""

from typing import Any, Dict, List, Optional

from khonliang.roles.base import BaseRole

# ---- Context providers (inject live data into prompts) ----


def get_open_ticket_count() -> int:
    """Stub — replace with real ticketing system API call."""
    return 42


def search_knowledge_base(query: str) -> List[str]:
    """Stub — replace with vector DB / KB search."""
    return [
        "To reset your password: go to Settings > Security > Reset Password.",
        "Two-factor authentication can be managed under Settings > Security > 2FA.",
        "Contact billing@example.com for invoice disputes.",
    ]


# ---- Roles ----


class TriageRole(BaseRole):
    """
    Fast ticket triage: classify urgency and route to the right team.

    Uses a smaller/faster model (e.g. llama3.2:3b) since this runs on
    every incoming message.
    """

    def __init__(self, model_pool, **kwargs):
        super().__init__(role="triage", model_pool=model_pool, **kwargs)
        self._system_prompt = (
            "You are a support ticket triage agent. "
            "Classify the urgency (critical/high/medium/low) and the team "
            "(billing/technical/account/general). "
            'Reply in JSON: {"urgency": "...", "team": "...", "summary": "..."}'
        )

    def build_context(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        count = get_open_ticket_count()
        return f"[Current queue depth: {count} open tickets]\n"

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self.build_context(message)
        prompt = f"{ctx}\nTicket: {message}\n\nClassify this ticket."

        try:
            result = await self.client.generate_json(
                prompt=prompt, system=self.system_prompt
            )
            return {
                "response": (
                    f"**{result.get('urgency', '?').upper()}** — "
                    f"Routing to {result.get('team', '?')} team.\n"
                    f"{result.get('summary', '')}"
                ),
                "metadata": {
                    "role": self.role,
                    "urgency": result.get("urgency"),
                    "team": result.get("team"),
                },
            }
        except Exception as e:
            return {
                "response": f"Could not triage ticket: {e}",
                "metadata": {"role": self.role},
            }


class KnowledgeRole(BaseRole):
    """
    Knowledge base Q&A: searches docs and answers with citations.

    Uses a mid-size model (e.g. qwen2.5:7b) with KB context injection.
    """

    def __init__(self, model_pool, **kwargs):
        super().__init__(role="knowledge", model_pool=model_pool, **kwargs)
        self._system_prompt = (
            "You are a helpful support agent with access to the knowledge base. "
            "Answer based only on the provided context. "
            "If you cannot answer from context, say so clearly."
        )

    def build_context(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        docs = search_knowledge_base(message)
        if not docs:
            return ""
        kb_text = "\n".join(f"- {doc}" for doc in docs)
        return f"Knowledge base results:\n{kb_text}\n"

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self.build_context(message)
        prompt = f"{ctx}\nQuestion: {message}\nAnswer:"

        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {
                "role": self.role,
                "generation_time_ms": elapsed_ms,
            },
        }


class EscalationRole(BaseRole):
    """
    Escalation summariser: generates a clear handoff note for a human.

    Uses a larger reasoning model (e.g. llama3.1:8b).
    """

    def __init__(self, model_pool, **kwargs):
        super().__init__(role="escalation", model_pool=model_pool, **kwargs)
        self._system_prompt = (
            "You are a senior support agent preparing a case handoff. "
            "Write a concise escalation note: what the customer needs, "
            "what was already tried, and the recommended next step."
        )

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history = (context or {}).get("history", "No prior context.")
        prompt = (
            f"Conversation history:\n{history}\n\n"
            f"Latest message: {message}\n\n"
            f"Write an escalation note."
        )
        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": f"**Escalation Note:**\n{response.strip()}",
            "metadata": {"role": self.role, "generation_time_ms": elapsed_ms},
        }
