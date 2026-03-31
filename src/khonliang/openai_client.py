"""
OpenAI-compatible client for any LLM server that speaks /v1/chat/completions.

Works with: vLLM, SGLang, llama.cpp, LM Studio, LocalAI, Jan, llamafile,
koboldcpp, MLC LLM, TabbyAPI, and cloud providers (Groq, Together AI,
Fireworks AI, OpenRouter, Cerebras, SambaNova).

Example:
    # Local vLLM server
    client = OpenAIClient(model="Qwen/Qwen2.5-7B", base_url="http://localhost:8000/v1")

    # Groq cloud
    client = OpenAIClient(
        model="llama-3.2-3b-preview",
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_...",
    )

    response = await client.generate("Hello!")
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from khonliang._json_utils import parse_llm_json
from khonliang.client import GenerationResult
from khonliang.errors import (
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.0
RETRY_BACKOFF_MAX = 10.0


class OpenAIClient:
    """
    Async client for OpenAI-compatible LLM servers.

    Provides the same interface as OllamaClient (satisfies LLMClient protocol)
    but speaks the OpenAI /v1/chat/completions API format. Any server or cloud
    provider that implements this endpoint works as a drop-in backend.

    Features:
    - Per-model timeout overrides
    - Exponential backoff retry (skips on model-not-found)
    - Typed errors matching OllamaClient's error hierarchy
    - JSON generation with response_format + cleanup fallback
    - SSE streaming support
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        model_timeouts: Optional[Dict[str, int]] = None,
        default_keep_alive: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._model_timeouts: Dict[str, int] = model_timeouts or {}
        self._session: Optional[aiohttp.ClientSession] = None
        # keep_alive is accepted for LLMClient protocol compatibility but
        # has no effect on OpenAI-compatible backends (Ollama-specific feature).
        self.default_keep_alive: Optional[str] = default_keep_alive

    def _get_timeout(self, model_name: str) -> int:
        return self._model_timeouts.get(model_name, self.timeout)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self._headers(),
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    def _build_messages(
        self, prompt: str, system: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Convert prompt + system into OpenAI messages format."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> str:
        """Generate text from a prompt. Returns the response string."""
        result = await self.generate_with_metrics(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            extra_options=extra_options,
            keep_alive=keep_alive,
        )
        return result.text

    async def generate_with_metrics(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text and return a GenerationResult with token metrics."""
        model_name = model or self.model
        model_timeout = self._get_timeout(model_name)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": self._build_messages(prompt, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if extra_options:
            payload.update(extra_options)

        last_error: Optional[Exception] = None
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                return await self._do_generate(payload, model_name, model_timeout)
            except LLMModelNotFoundError:
                raise
            except (LLMTimeoutError, LLMUnavailableError) as e:
                last_error = e
                if attempt < RETRY_MAX_ATTEMPTS - 1:
                    backoff = min(RETRY_BACKOFF_BASE * (2**attempt), RETRY_BACKOFF_MAX)
                    logger.warning(
                        f"Attempt {attempt + 1}/{RETRY_MAX_ATTEMPTS} failed for "
                        f"{model_name}: {e}. Retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)

        raise last_error  # type: ignore[misc]

    async def _do_generate(
        self, payload: Dict[str, Any], model_name: str, timeout: int
    ) -> GenerationResult:
        per_request_timeout = aiohttp.ClientTimeout(total=timeout)
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=per_request_timeout,
            ) as response:
                if response.status == 404:
                    raise LLMModelNotFoundError(
                        f"Model '{model_name}' not found on {self.base_url}",
                        model=model_name,
                    )
                if response.status == 429:
                    raise LLMRateLimitError(
                        f"Rate limited by {self.base_url}", model=model_name
                    )
                response.raise_for_status()
                data = await response.json()

                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"No choices in response: {data}")

                text = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})

                return GenerationResult(
                    text=text,
                    model=data.get("model", model_name),
                    prompt_eval_count=usage.get("prompt_tokens", 0),
                    eval_count=usage.get("completion_tokens", 0),
                    total_duration_ns=0,
                    eval_duration_ns=0,
                )

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                f"Request exceeded {timeout}s timeout", model=model_name
            ) from e
        except aiohttp.ClientResponseError as e:
            raise LLMUnavailableError(
                f"HTTP {e.status} from {self.base_url}: {e.message}",
                model=model_name,
            ) from e
        except aiohttp.ClientError as e:
            raise LLMUnavailableError(
                f"Failed to connect to {self.base_url}: {e}", model=model_name
            ) from e

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
    ) -> Dict:
        """Generate structured JSON output.

        If constrained=True, uses response_format={"type": "json_object"}
        for servers that support it (OpenAI, vLLM, SGLang). Falls back to
        JSON cleanup parsing otherwise.
        """
        json_system = (
            f"{system}\n\nRespond with valid JSON only. No markdown, no explanations."
            if system
            else "Respond with valid JSON only. No markdown, no explanations."
        )
        if schema:
            json_system += f"\n\nFollow this JSON schema:\n{json.dumps(schema, indent=2)}"

        extra: Dict[str, Any] = {}
        if constrained:
            extra["response_format"] = {"type": "json_object"}

        response = await self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            extra_options=extra,
            keep_alive=keep_alive,
        )
        return parse_llm_json(response)

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text tokens via SSE from /v1/chat/completions."""
        model_name = model or self.model
        model_timeout = self._get_timeout(model_name)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": self._build_messages(prompt, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if extra_options:
            payload.update(extra_options)

        per_request_timeout = aiohttp.ClientTimeout(total=model_timeout)

        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=per_request_timeout,
            ) as response:
                if response.status == 404:
                    raise LLMModelNotFoundError(
                        f"Model '{model_name}' not found on {self.base_url}",
                        model=model_name,
                    )
                if response.status == 429:
                    raise LLMRateLimitError(
                        f"Rate limited by {self.base_url}", model=model_name
                    )
                response.raise_for_status()

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                f"Stream exceeded {model_timeout}s timeout", model=model_name
            ) from e
        except aiohttp.ClientResponseError as e:
            raise LLMUnavailableError(
                f"HTTP {e.status} from {self.base_url}: {e.message}",
                model=model_name,
            ) from e
        except aiohttp.ClientError as e:
            raise LLMUnavailableError(
                f"Failed to connect to {self.base_url}: {e}", model=model_name
            ) from e

    async def is_available_async(self) -> bool:
        """Async health check — tries /v1/models endpoint."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                return response.status == 200
        except Exception:
            return False
