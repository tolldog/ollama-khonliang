"""
Async Ollama client with typed errors, per-model timeouts, retry, and streaming.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from khonliang.errors import (
    LLMModelNotFoundError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

# Defaults — override per instance or via ModelPool
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS = 4000
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.0
RETRY_BACKOFF_MAX = 10.0

# Sensible per-model timeout defaults — override via OllamaClient(model_timeouts={...})
DEFAULT_MODEL_TIMEOUTS: Dict[str, int] = {
    "llama3.2:3b": 30,
    "llama3.1:8b": 60,
    "qwen2.5:7b": 60,
    "deepseek-r1:14b": 120,
    "deepseek-r1:32b": 300,
}


@dataclass
class GenerationResult:
    """Result from an LLM generation call, including token metrics."""

    text: str
    model: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ns: int = 0
    eval_duration_ns: int = 0

    @property
    def duration_s(self) -> float:
        """Total generation duration in seconds."""
        return self.total_duration_ns / 1e9 if self.total_duration_ns else 0.0


class OllamaClient:
    """
    Async client for Ollama local LLM inference.

    Features:
    - Per-model timeout overrides
    - Exponential backoff retry (skips on model-not-found)
    - Typed errors (LLMTimeoutError, LLMUnavailableError, LLMModelNotFoundError)
    - JSON generation with common LLM output fixing

    Example:
        client = OllamaClient(model="qwen2.5:7b")
        response = await client.generate("Summarise this ticket: ...")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        model_timeouts: Optional[Dict[str, int]] = None,
        default_keep_alive: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model_timeouts = model_timeouts or DEFAULT_MODEL_TIMEOUTS
        self._session: Optional[aiohttp.ClientSession] = None
        self.default_keep_alive: Optional[str] = default_keep_alive

    def _get_timeout(self, model_name: str) -> int:
        return self._model_timeouts.get(model_name, self.timeout)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
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

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> str:
        """Generate text from a prompt. Returns the response string.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            model: Override the client's default model.
            extra_options: Additional Ollama options merged into the request.
            keep_alive: Override keep_alive duration (e.g. "5m", "0", "-1").
                Falls back to default_keep_alive if not set.
        """
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
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text and return a GenerationResult with token metrics.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            model: Override the client's default model.
            extra_options: Additional Ollama options merged into the request.
            keep_alive: Override keep_alive duration (e.g. "5m", "0", "-1").
                Falls back to default_keep_alive if not set.
        """
        model_name = model or self.model
        model_timeout = self._get_timeout(model_name)

        options: Dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if extra_options:
            options.update(extra_options)

        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        effective_keep_alive = keep_alive or self.default_keep_alive
        if effective_keep_alive is not None:
            payload["keep_alive"] = effective_keep_alive

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
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=per_request_timeout,
            ) as response:
                if response.status == 404:
                    raise LLMModelNotFoundError(
                        f"Model '{model_name}' not found. "
                        f"Pull it with: ollama pull {model_name}",
                        model=model_name,
                    )
                response.raise_for_status()
                data = await response.json()

                if "response" not in data:
                    raise ValueError(f"Unexpected response format: {data}")

                return GenerationResult(
                    text=data["response"],
                    model=model_name,
                    prompt_eval_count=data.get("prompt_eval_count", 0),
                    eval_count=data.get("eval_count", 0),
                    total_duration_ns=data.get("total_duration", 0),
                    eval_duration_ns=data.get("eval_duration", 0),
                )

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                f"Request exceeded {timeout}s timeout", model=model_name
            ) from e
        except aiohttp.ClientError as e:
            raise LLMUnavailableError(
                f"Failed to connect to Ollama at {self.base_url}: {e}", model=model_name
            ) from e

    async def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        model: Optional[str] = None,
        constrained: bool = False,
        keep_alive: Optional[str] = None,
    ) -> Dict:
        """
        Generate structured JSON output.

        Args:
            constrained: If True, uses Ollama's native JSON mode
                (format="json") which constrains token generation to
                produce only valid JSON. Falls back to cleanup parsing
                if the result still needs fixing.
        """
        json_system = (
            f"{system}\n\nRespond with valid JSON only. No markdown, no explanations."
            if system
            else "Respond with valid JSON only. No markdown, no explanations."
        )
        if schema:
            json_system += f"\n\nFollow this JSON schema:\n{json.dumps(schema, indent=2)}"

        if constrained:
            # Use Ollama's native JSON mode
            model_name = model or self.model
            model_timeout = self._get_timeout(model_name)
            effective_keep_alive = keep_alive or self.default_keep_alive
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": prompt,
                "system": json_system,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if effective_keep_alive is not None:
                payload["keep_alive"] = effective_keep_alive
            try:
                result = await self._do_generate(
                    payload, model_name, model_timeout
                )
                return self._parse_json(result.text)
            except Exception as e:
                logger.debug(f"Constrained JSON mode failed, falling back: {e}")

        response = await self.generate(
            prompt=prompt, system=json_system, temperature=temperature,
            max_tokens=max_tokens, model=model, keep_alive=keep_alive,
        )
        return self._parse_json(response)

    def _parse_json(self, response: str) -> Dict:
        import re

        text = response.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Strip leading/trailing non-JSON content
            first = min(
                (text.find(c) for c in "{[" if text.find(c) >= 0), default=0
            )
            last = max(text.rfind("}"), text.rfind("]"))
            text = text[first : last + 1] if last >= 0 else text
            text = re.sub(r",\s*([}\]])", r"\1", text)
            text = re.sub(r"\bTrue\b", "true", text)
            text = re.sub(r"\bFalse\b", "false", text)
            text = re.sub(r"\bNone\b", "null", text)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e

    async def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        model: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text tokens as they are generated.

        Yields each text chunk as it arrives from Ollama, allowing the caller
        to display progressive output without waiting for the full response.

        Example:
            async for chunk in client.stream_generate("Explain RAG in one paragraph"):
                print(chunk, end="", flush=True)
        """
        model_name = model or self.model
        model_timeout = self._get_timeout(model_name)

        options: Dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if extra_options:
            options.update(extra_options)

        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        if system:
            payload["system"] = system

        effective_keep_alive = keep_alive or self.default_keep_alive
        if effective_keep_alive is not None:
            payload["keep_alive"] = effective_keep_alive

        per_request_timeout = aiohttp.ClientTimeout(total=model_timeout)

        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=per_request_timeout,
            ) as response:
                if response.status == 404:
                    raise LLMModelNotFoundError(
                        f"Model '{model_name}' not found. "
                        f"Pull it with: ollama pull {model_name}",
                        model=model_name,
                    )
                response.raise_for_status()

                async for raw_line in response.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = chunk.get("response", "")
                    if text:
                        yield text

                    if chunk.get("done", False):
                        break

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(
                f"Stream exceeded {model_timeout}s timeout", model=model_name
            ) from e
        except aiohttp.ClientError as e:
            raise LLMUnavailableError(
                f"Failed to connect to Ollama at {self.base_url}: {e}", model=model_name
            ) from e

    def is_available(self) -> bool:
        """Synchronous health check. Returns True if Ollama is reachable."""
        import requests
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """Return names of all models available on the Ollama server."""
        try:
            session = await self._ensure_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                response.raise_for_status()
                data = await response.json()
                return [m["name"] for m in data.get("models", [])]
        except aiohttp.ClientError as e:
            raise LLMUnavailableError(
                f"Failed to connect to Ollama at {self.base_url}: {e}", model=""
            ) from e

    async def check_running_models(self) -> List[Dict[str, Any]]:
        """
        Check which models are currently loaded in Ollama via /api/ps.

        Returns list of loaded models with VRAM usage, expiry, and details.
        Useful for GPU-constrained deployments to know what's in memory.
        """
        try:
            session = await self._ensure_session()
            async with session.get(f"{self.base_url}/api/ps") as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("models", [])
        except aiohttp.ClientError:
            return []

    async def is_available_async(self) -> bool:
        """Async health check — uses the existing aiohttp session."""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama registry.

        Blocks until the pull is complete. Useful for setup scripts.
        Returns True if successful.
        """
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=aiohttp.ClientTimeout(total=600),  # 10 min for large models
            ) as response:
                return response.status == 200
        except Exception:
            return False
