"""
Model pool — maps roles to OllamaClient instances.

Lazy-loads clients on first access. Roles that share the same model
reuse a single OllamaClient instance.
"""

import logging
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from khonliang.client import OllamaClient

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
    Manages OllamaClient instances per role.

    Pass a plain dict mapping role names (str) or Enum values to model names.
    Clients that share the same model reuse a single connection.

    Example:
        pool = ModelPool({
            "triage":     "llama3.2:3b",
            "researcher": "qwen2.5:7b",
            "writer":     "llama3.1:8b",
        })
        client = pool.get_client("triage")
        response = await client.generate("Is this urgent?")
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
    ):
        """
        Args:
            role_model_map: Dict mapping role names to model names
            base_url: Ollama server URL
            model_timeouts: Optional per-model timeout overrides
                e.g. {"deepseek-r1:32b": 300, "llama3.2:3b": 30}
            keep_alive: Optional per-role keep_alive overrides
                e.g. {"triage": "5m", "researcher": "10m"}
            adaptive_keep_alive: If True, automatically adjust keep_alive
                based on call frequency (hot/warm/cold classification)
            adaptive_config: Override adaptive keep-alive thresholds.
                See DEFAULT_ADAPTIVE_CONFIG for keys.
            on_call: Optional callback invoked after each generate() call
                with (role: str, duration_ms: float). Useful for external
                monitoring. Consumers should call record_call() after their
                generate() call to trigger this callback.
        """
        self._map = {str(k): v for k, v in role_model_map.items()}
        self._clients: Dict[str, OllamaClient] = {}
        self._base_url = base_url
        self._model_timeouts = model_timeouts
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

    def get_client(self, role) -> OllamaClient:
        """
        Get OllamaClient for the given role. Creates on first access.

        The per-role keep_alive value is resolved and stored separately
        so that shared clients (multiple roles using the same model) are
        not mutated. Use get_keep_alive(role) to retrieve the resolved
        value and pass it to generate(keep_alive=...).
        """
        role_str = str(role)
        model = self._map.get(role_str)
        if model is None:
            raise KeyError(f"No model configured for role '{role}'")

        if model not in self._clients:
            logger.debug(f"Creating OllamaClient for {role} -> {model}")
            client = OllamaClient(
                model=model,
                base_url=self._base_url,
                model_timeouts=self._model_timeouts,
            )
            self._clients[model] = client

        # Always resolve per-role keep_alive (static config may change)
        if role_str in self._keep_alive:
            self._resolved_keep_alive[role_str] = self._keep_alive[role_str]

        # Track call timestamp for adaptive keep-alive
        now = time.time()
        if model not in self._call_timestamps:
            self._call_timestamps[model] = deque(maxlen=1000)
        self._call_timestamps[model].append(now)

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

        return self._clients[model]

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
        """Return the model name configured for a role, or None."""
        return self._map.get(str(role))

    def get_model_temperature(self, role) -> str:
        """
        Classify a model's usage temperature based on recent call frequency.

        Returns "hot", "warm", or "cold" based on calls within the
        configured lookback window.
        """
        model = self._map.get(str(role))
        if model is None:
            return "cold"

        timestamps = self._call_timestamps.get(model, deque())
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
