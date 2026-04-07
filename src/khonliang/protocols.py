"""
LLMClient protocol — the common interface for LLM inference clients.

Both OllamaClient and OpenAIClient satisfy this protocol. Use it as a
type annotation whenever you need a client that could be either backend.

Example:
    async def summarize(client: LLMClient, text: str) -> str:
        return await client.generate(f"Summarize: {text}")
"""

from typing import Any, AsyncGenerator, Dict, Optional, Protocol, runtime_checkable

from khonliang.client import GenerationResult


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM inference clients.

    Satisfied by OllamaClient and OpenAIClient. Ollama-specific methods
    (pull_model, check_running_models, list_models, is_available sync)
    are intentionally excluded — use OllamaClient directly for those.
    """

    model: str

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        n_samples: int = 1,
    ) -> str: ...

    async def generate_with_metrics(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        n_samples: int = 1,
    ) -> GenerationResult: ...

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> AsyncGenerator[str, None]: ...

    async def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        constrained: bool = False,
        keep_alive: Optional[str] = None,
    ) -> Dict: ...

    async def is_available_async(self) -> bool: ...

    async def close(self) -> None: ...
