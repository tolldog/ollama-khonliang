"""
Model pool — maps roles to LLM client instances.

Lazy-loads clients on first access. Roles that share the same model
reuse a single client instance. Supports mixed backends via URI scheme:

    pool = ModelPool({
        "researcher": "llama3.2:3b",                  # Ollama (default)
        "narrator": "openai://llama3.1:70b",          # vLLM / OpenAI-compatible
        "classifier": "groq://llama-3.2-3b-preview",  # Groq cloud
    }, backends={
        "openai": {"base_url": "http://gpu-box:8000/v1"},
        "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_..."},
    })
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, Optional, Tuple

from khonliang.client import OllamaClient

if TYPE_CHECKING:
    from khonliang.protocols import LLMClient

logger = logging.getLogger(__name__)

# Default adaptive keep-alive configuration
DEFAULT_ADAPTIVE_CONFIG: Dict[str, Any] = {
    "window_seconds": 300,       # Lookback window for call frequency
    "hot_threshold": 10,         # Calls in window to be considered hot
    "warm_threshold": 3,         # Calls in window to be considered warm
    "hot_keep_alive": "10m",     # Keep-alive for hot models
    "warm_keep_alive": "5m",     # Keep-alive for warm models
    "cold_keep_alive": "0",      # Keep-alive for cold models (unload immediately)
}


class ModelPool:
    """
    Manages LLM client instances per role.

    Pass a plain dict mapping role names (str) or Enum values to model
    specifiers. Plain model names use Ollama. URI-prefixed names
    (e.g. "openai://model") use the corresponding backend.

    Examples:
        # Pure Ollama (backward compatible)
        pool = ModelPool({
            "triage":     "llama3.2:3b",
            "researcher": "qwen2.5:7b",
        })

        # Mixed backends
        pool = ModelPool({
            "researcher": "llama3.2:3b",
            "narrator": "openai://llama3.1:70b",
            "classifier": "groq://llama-3.2-3b-preview",
        }, backends={
            "openai": {"base_url": "http://gpu-box:8000/v1"},
            "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": "gsk_..."},
        })
    """

    def __init__(
        self,
        role_model_map: Dict,
        base_url: str = "http://localhost:11434",
        model_timeouts: Optional[Dict[str, int]] = None,
        keep_alive: Optional[Dict[str, str]] = None,
        adaptive_keep_alive: bool = False,
        adaptive_config: Optional[Dict[str, Any]] = None,
        on_call: Optional[Callable[[str, float], None]] = None,
        backends: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            role_model_map: Dict mapping role names to model specifiers.
                Plain names (e.g. "llama3.2:3b") use Ollama.
                Prefixed names (e.g. "openai://model") use the named backend.
            base_url: Ollama server URL (for plain model names)
            model_timeouts: Optional per-model timeout overrides
            keep_alive: Optional per-role keep_alive overrides
            adaptive_keep_alive: If True, automatically adjust keep_alive
                based on call frequency (hot/warm/cold classification)
            adaptive_config: Override adaptive keep-alive thresholds.
                See DEFAULT_ADAPTIVE_CONFIG for keys.
            on_call: Optional callback invoked after each generate() call
            backends: Optional dict of backend configs for non-Ollama models.
                Keys are scheme names (e.g. "openai", "groq").
                Values are dicts with at least "base_url" and optionally
                "api_key", "timeout", "model_timeouts".
        """
        self._map = {str(k): v for k, v in role_model_map.items()}
        self._clients: Dict[str, LLMClient] = {}
        self._base_url = base_url
        self._model_timeouts = model_timeouts
        self._backends = backends or {}
        self._keep_alive = keep_alive or {}
        self._adaptive_keep_alive = adaptive_keep_alive
        self._adaptive_config = {**DEFAULT_ADAPTIVE_CONFIG, **(adaptive_config or {})}
        self._on_call = on_call

        # Adaptive keep-alive: track call timestamps per model
        self._call_timestamps: Dict[str, Deque[float]] = {}

        # Per-role resolved keep_alive value (static or adaptive)
        self._resolved_keep_alive: Dict[str, Optional[str]] = {}

        # Usage metrics: per-role deque of (timestamp, duration_ms)
        self._usage: Dict[str, Deque[Tuple[float, float]]] = {}

    @staticmethod
    def _parse_model_spec(spec: str) -> tuple:
        """Parse a model specifier into (scheme, model_name).

        "llama3.2:3b" → (None, "llama3.2:3b")     — Ollama
        "openai://llama3.1:70b" → ("openai", "llama3.1:70b")
        "groq://llama-3.2-3b" → ("groq", "llama-3.2-3b")
        """
        if "://" in spec:
            scheme, model = spec.split("://", 1)
            return (scheme, model)
        return (None, spec)

    def get_client(self, role):
        """
        Get LLM client for the given role. Creates on first access.

        Returns an OllamaClient for plain model names, or an OpenAIClient
        for URI-prefixed model names (e.g. "openai://model").

        The per-role keep_alive value is resolved and stored separately
        so that shared clients (multiple roles using the same model) are
        not mutated. Use get_keep_alive(role) to retrieve the resolved
        value and pass it to generate(keep_alive=...).
        """
        role_str = str(role)
        spec = self._map.get(role_str)
        if spec is None:
            raise KeyError(f"No model configured for role '{role}'")

        scheme, model = self._parse_model_spec(spec)
        cache_key = f"{scheme}://{model}" if scheme else model

        if cache_key not in self._clients:
            if scheme and scheme in self._backends:
                from khonliang.openai_client import OpenAIClient

                cfg = self._backends[scheme]
                if "base_url" not in cfg:
                    raise ValueError(
                        f"Backend '{scheme}' missing required 'base_url'. "
                        f"Configure: backends={{'{scheme}': {{'base_url': '...'}}}}"
                    )
                logger.debug(f"Creating OpenAIClient for {role} -> {scheme}://{model}")
                client = OpenAIClient(
                    model=model,
                    base_url=cfg["base_url"],
                    api_key=cfg.get("api_key"),
                    timeout=cfg.get("timeout", 120),
                    model_timeouts=cfg.get("model_timeouts", self._model_timeouts),
                )
                self._clients[cache_key] = client
            elif scheme:
                raise KeyError(
                    f"Backend '{scheme}' not configured. "
                    f"Add it to ModelPool(backends={{'{scheme}': {{...}}}})"
                )
            else:
                logger.debug(f"Creating OllamaClient for {role} -> {model}")
                client = OllamaClient(
                    model=model,
                    base_url=self._base_url,
                    model_timeouts=self._model_timeouts,
                )
                self._clients[cache_key] = client

        # Always resolve per-role keep_alive (static config may change)
        if role_str in self._keep_alive:
            self._resolved_keep_alive[role_str] = self._keep_alive[role_str]

        # Track call timestamp for adaptive keep-alive
        now = time.time()
        if cache_key not in self._call_timestamps:
            self._call_timestamps[cache_key] = deque(maxlen=1000)
        self._call_timestamps[cache_key].append(now)

        # Resolve adaptive keep-alive per-role without mutating shared client
        if self._adaptive_keep_alive:
            temp = self.get_model_temperature(role_str)
            cfg = self._adaptive_config
            if temp == "hot":
                self._resolved_keep_alive[role_str] = cfg["hot_keep_alive"]
            elif temp == "warm":
                self._resolved_keep_alive[role_str] = cfg["warm_keep_alive"]
            else:
                self._resolved_keep_alive[role_str] = cfg["cold_keep_alive"]

        return self._clients[cache_key]

    def get_keep_alive(self, role) -> Optional[str]:
        """
        Return the resolved keep_alive value for a role.

        This accounts for both static per-role overrides and adaptive
        temperature-based adjustments. Pass the result to
        client.generate(keep_alive=...) instead of relying on
        client.default_keep_alive, which is shared across roles that
        use the same model.
        """
        return self._resolved_keep_alive.get(str(role))

    def get_model_name(self, role) -> Optional[str]:
        """Return the model name configured for a role, or None.

        Returns the clean model name without any scheme prefix.
        """
        spec = self._map.get(str(role))
        if spec is None:
            return None
        _, model = self._parse_model_spec(spec)
        return model

    def get_model_temperature(self, role) -> str:
        """
        Classify a model's usage temperature based on recent call frequency.

        Returns "hot", "warm", or "cold" based on calls within the
        configured lookback window.
        """
        spec = self._map.get(str(role))
        if spec is None:
            return "cold"

        scheme, model = self._parse_model_spec(spec)
        cache_key = f"{scheme}://{model}" if scheme else model
        timestamps = self._call_timestamps.get(cache_key, deque())
        cfg = self._adaptive_config
        window = cfg["window_seconds"]
        cutoff = time.time() - window
        recent = sum(1 for ts in timestamps if ts > cutoff)

        if recent >= cfg["hot_threshold"]:
            return "hot"
        elif recent >= cfg["warm_threshold"]:
            return "warm"
        return "cold"

    def record_call(self, role: str, duration_ms: float) -> None:
        """
        Record a completed call for usage metrics tracking.

        Consumers should call this after each generate() invocation to
        feed the on_call callback and usage statistics. It is not called
        automatically by get_client(); the caller is responsible for
        timing the generate() call and reporting the duration here.

        Args:
            role: The role that made the call
            duration_ms: How long the call took in milliseconds
        """
        role_str = str(role)
        now = time.time()
        if role_str not in self._usage:
            self._usage[role_str] = deque(maxlen=1000)
        self._usage[role_str].append((now, duration_ms))

        if self._on_call:
            try:
                self._on_call(role_str, duration_ms)
            except Exception as e:
                logger.warning(f"on_call callback failed for {role_str}: {e}")

    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Return per-role usage statistics.

        Returns a dict keyed by role name, each containing:
            - calls_5m: Number of calls in the last 5 minutes
            - avg_ms: Average call duration in ms (over last 5 minutes)
            - last_call: Timestamp of the most recent call (or None)
        """
        now = time.time()
        cutoff = now - 300  # 5 minutes
        stats: Dict[str, Dict[str, Any]] = {}

        for role_str, entries in self._usage.items():
            recent = [(ts, dur) for ts, dur in entries if ts > cutoff]
            calls_5m = len(recent)
            avg_ms = (
                sum(dur for _, dur in recent) / calls_5m
                if calls_5m > 0
                else 0.0
            )
            last_call = entries[-1][0] if entries else None
            stats[role_str] = {
                "calls_5m": calls_5m,
                "avg_ms": round(avg_ms, 1),
                "last_call": last_call,
            }

        return stats

    async def close_all(self) -> None:
        """Close all managed OllamaClient sessions."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
