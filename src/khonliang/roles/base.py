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

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from khonliang.pool import ModelPool

if TYPE_CHECKING:
    from khonliang.protocols import LLMClient

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
        board: Optional[Any] = None,
        model_router: Optional[Any] = None,
    ):
        """
        Args:
            role: String name identifying this role
            model_pool: ModelPool instance mapping roles to Ollama models
            system_prompt: Optional system prompt for this role
            prompts_dir: Optional directory to load system prompts from files
            max_context_tokens: Optional token budget for context. When
                build_context() output exceeds this, older content is
                truncated (not compressed). Uses chars/4 heuristic for
                token estimation. None or 0 means no budget enforcement.
                LLM-based compression is a future enhancement.
            board: Optional Blackboard instance for shared agent context.
                When set, build_context() appends board entries automatically.
            model_router: Optional ModelRouter for dynamic model selection
                within this role. When set, _select_model() picks the best
                model for each message based on the configured strategy.
        """
        self.role = role
        self._model_pool = model_pool
        self._prompts_dir = prompts_dir
        self._system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.board = board
        self._model_router = model_router

    @property
    def client(self) -> "LLMClient":
        """LLM client instance for this role's configured model.

        Returns an OllamaClient or OpenAIClient depending on the model
        specifier configured in the ModelPool.
        """
        return self._model_pool.get_client(self.role)

    @property
    def system_prompt(self) -> str:
        """System prompt for this role, or empty string if unset."""
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

        When a board (Blackboard) is attached, its entries are appended
        automatically. Subclasses that override this should call
        super().build_context() to preserve board integration.

        Returns empty string by default (plus board context if set).
        """
        parts = []
        if self.board is not None:
            board_ctx = self.board.build_context()
            if board_ctx:
                parts.append(board_ctx)
        return "\n".join(parts)

    def _get_context(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build context and enforce the token budget.

        Convenience wrapper that calls build_context() then enforce_budget().
        Subclasses should call this in handle() instead of calling
        build_context() and enforce_budget() separately.
        """
        raw = self.build_context(message, context)
        return self.enforce_budget(raw)

    def enforce_budget(
        self, context: str, strategy: str = "truncate"
    ) -> str:
        """
        Fit context within the token budget.

        Args:
            context: Raw context string
            strategy: Extraction strategy:
                "truncate" — keep most recent content, cut from start (default)
                "sections" — extract key document sections (abstract, intro,
                    methods, results, conclusion) and allocate budget across them

        Returns context unchanged if no budget is set or context fits.
        """
        if not self.max_context_tokens:
            return context

        max_chars = self.max_context_tokens * 4

        if len(context) <= max_chars:
            return context

        if strategy == "sections":
            return _extract_sections(context, max_chars)

        # Default: truncate from the beginning, keep most recent
        truncated = context[-max_chars:]
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

    async def _select_model(self, message: str) -> Optional[str]:
        """Select a model for this message using the model router.

        Returns the model name if a router is configured and selects
        a model, or None to use the default from ModelPool.
        """
        if self._model_router is None:
            return None
        try:
            selection = await self._model_router.select(self.role, message)
            if selection.model:
                return selection.model
        except Exception as e:
            logger.debug(f"Model router failed for {self.role}: {e}")
        return None

    async def _timed_generate(
        self, prompt: str, system: Optional[str] = None, model: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, int]:
        """Generate with wall-clock timing. Returns (text, elapsed_ms).

        Args:
            prompt: The prompt to generate from.
            system: Optional system prompt.
            model: Optional model override. When provided, passes it to
                the client's generate() method to override the default model.
        """
        start = time.time()
        if model:
            kwargs["model"] = model
        response = await self.client.generate(prompt=prompt, system=system, **kwargs)
        elapsed_ms = int((time.time() - start) * 1000)
        return response, elapsed_ms


# ------------------------------------------------------------------
# Section-aware context extraction (KH-12)
# ------------------------------------------------------------------

# Heading patterns for detecting document structure
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,3}\s+|"           # Markdown: ## Heading
    r"[A-Z][A-Z ]{2,}$|"   # ALL CAPS LINE
    r"\d+\.\s+[A-Z]"       # 1. Section Name
    r")",
    re.MULTILINE,
)

# Section names to look for (case-insensitive)
_SECTION_PRIORITIES: List[Tuple[str, float]] = [
    ("abstract", 0.25),
    ("introduction", 0.20),
    ("conclusion", 0.20),
    ("results", 0.15),
    ("method", 0.10),
    ("discussion", 0.10),
]


def _extract_sections(text: str, max_chars: int) -> str:
    """Extract key sections from a structured document.

    Detects headings (markdown, numbered, ALL CAPS), then allocates
    the character budget across high-priority sections.

    Falls back to first-chunk + last-chunk if no structure is detected.
    """
    sections = _split_into_sections(text)

    if len(sections) < 3:
        # No clear structure — take first and last chunks
        half = max_chars // 2
        result = text[:half] + "\n\n[...]\n\n" + text[-half:]
        return f"[Extracted first + last chunks]\n{result}"

    # Allocate budget to priority sections
    selected: List[Tuple[str, str]] = []
    remaining = max_chars

    for target_name, budget_frac in _SECTION_PRIORITIES:
        budget = int(max_chars * budget_frac)
        for sec_name, sec_text in sections:
            if target_name in sec_name.lower() and budget > 0:
                chunk = sec_text[:min(budget, remaining)]
                if chunk:
                    selected.append((sec_name, chunk))
                    remaining -= len(chunk)
                break

    # If budget remains, fill with other sections
    used_names = {name for name, _ in selected}
    for sec_name, sec_text in sections:
        if remaining <= 0:
            break
        if sec_name not in used_names:
            chunk = sec_text[:remaining]
            if chunk:
                selected.append((sec_name, chunk))
                remaining -= len(chunk)

    # Format
    parts = []
    for name, content in selected:
        parts.append(f"## {name}\n{content}")

    return f"[Extracted {len(selected)} sections]\n\n" + "\n\n".join(parts)


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split text into (heading, body) pairs based on detected headings."""
    lines = text.split("\n")
    sections: List[Tuple[str, str]] = []
    current_heading = "preamble"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if _HEADING_RE.match(stripped) and len(stripped) < 100:
            # Save current section
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, body))
            # Start new section
            current_heading = stripped.lstrip("#").strip().rstrip(".")
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    return sections
