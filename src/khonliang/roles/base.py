"""
Base class for domain-specific LLM roles.

Subclass BaseRole to define a role in your system. Each role:
- Is assigned a model via ModelPool
- Optionally injects context before generating (override build_context)
- Must implement handle() for its domain logic

Example — a customer support triage role:

    class TriageRole(BaseRole):
        async def handle(self, message, session_id, context=None):
            ctx = self.build_context(message)
            prompt = f"{ctx}\\nTicket: {message}\\nClassify urgency and route."
            response = await self.client.generate(prompt, system=self.system_prompt)
            return {"response": response, "metadata": {"role": self.role}}
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from khonliang.client import OllamaClient
from khonliang.pool import ModelPool

logger = logging.getLogger(__name__)


class BaseRole(ABC):
    """
    Abstract base for a named LLM role.

    Args:
        role: String name identifying this role (e.g. "triage", "researcher")
        model_pool: ModelPool instance mapping roles to Ollama models
        system_prompt: Optional system prompt for this role
        prompts_dir: Optional directory to load system prompts from files
    """

    def __init__(
        self,
        role: str,
        model_pool: ModelPool,
        system_prompt: Optional[str] = None,
        prompts_dir: Optional[Path] = None,
        max_context_tokens: Optional[int] = None,
    ):
        """
        Args:
            role: String name identifying this role
            model_pool: ModelPool instance mapping roles to Ollama models
            system_prompt: Optional system prompt for this role
            prompts_dir: Optional directory to load system prompts from files
            max_context_tokens: Optional token budget for context. When
                build_context() output exceeds this, older content is
                truncated. Uses chars/4 heuristic for token estimation.
                None means no budget enforcement.
        """
        self.role = role
        self._model_pool = model_pool
        self._prompts_dir = prompts_dir
        self._system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens

    @property
    def client(self) -> OllamaClient:
        return self._model_pool.get_client(self.role)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt or ""

    def load_prompt_file(self, filename: str, fallback: str = "") -> str:
        """Load a system prompt from a file in prompts_dir."""
        if self._prompts_dir:
            path = self._prompts_dir / filename
            if path.exists():
                try:
                    return path.read_text().strip()
                except Exception as e:
                    logger.warning(f"Could not load prompt {filename}: {e}")
        return fallback

    def build_context(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build context string to prepend to the prompt.

        Override this to inject live data (CRM records, DB state, API responses)
        relevant to your domain before the model call.

        Returns empty string by default.
        """
        return ""

    def enforce_budget(self, context: str) -> str:
        """
        Truncate context to fit within the token budget.

        Uses chars/4 heuristic for token estimation. Keeps the most
        recent content, truncates from the beginning.

        Returns context unchanged if no budget is set or context fits.
        """
        if self.max_context_tokens is None:
            return context

        # Heuristic: ~4 chars per token
        max_chars = self.max_context_tokens * 4

        if len(context) <= max_chars:
            return context

        # Truncate from the beginning, keep the most recent content
        truncated = context[-max_chars:]

        # Try to break at a newline to avoid mid-sentence truncation
        first_newline = truncated.find("\n")
        if first_newline > 0 and first_newline < len(truncated) // 4:
            truncated = truncated[first_newline + 1:]

        return f"[Context truncated to ~{self.max_context_tokens} tokens]\n{truncated}"

    @abstractmethod
    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a message within this role's domain.

        Args:
            message: User input text
            session_id: Session identifier for conversation continuity
            context: Optional dict of additional context

        Returns:
            Dict with at minimum {"response": str}. Add "metadata" for
            timing, token counts, role attribution, etc.
        """
        ...

    async def _timed_generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> tuple[str, int]:
        """Generate with wall-clock timing. Returns (text, elapsed_ms)."""
        start = time.time()
        response = await self.client.generate(prompt=prompt, system=system, **kwargs)
        elapsed_ms = int((time.time() - start) * 1000)
        return response, elapsed_ms
