"""
Model health tracker with cooldown.

After N failures within a time window, a model enters cooldown and
requests are skipped for a configurable duration. Prevents hammering
a model that's consistently failing due to GPU contention or OOM.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class _ModelState:
    failure_timestamps: list = field(default_factory=list)
    cooldown_until: float = 0.0
    total_failures: int = 0
    total_successes: int = 0


class ModelHealthTracker:
    """
    Track model health and enforce cooldown periods.

    Example:
        tracker = ModelHealthTracker()
        if not tracker.is_cooled_down("llama3.1:8b"):
            try:
                response = await client.generate(...)
                tracker.record_success("llama3.1:8b")
            except LLMError:
                tracker.record_failure("llama3.1:8b")
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        failure_window: float = 300.0,
        cooldown_duration: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window
        self.cooldown_duration = cooldown_duration
        self._models: Dict[str, _ModelState] = {}

    def _get_state(self, model: str) -> _ModelState:
        if model not in self._models:
            self._models[model] = _ModelState()
        return self._models[model]

    def record_failure(self, model: str) -> None:
        state = self._get_state(model)
        now = time.monotonic()
        state.failure_timestamps.append(now)
        state.total_failures += 1

        cutoff = now - self.failure_window
        state.failure_timestamps = [t for t in state.failure_timestamps if t > cutoff]

        if len(state.failure_timestamps) >= self.failure_threshold:
            state.cooldown_until = now + self.cooldown_duration
            logger.warning(
                f"Model '{model}' entering cooldown for {self.cooldown_duration}s "
                f"after {len(state.failure_timestamps)} failures in "
                f"{self.failure_window}s window"
            )

    def record_success(self, model: str) -> None:
        state = self._get_state(model)
        state.failure_timestamps.clear()
        state.total_successes += 1

    def is_cooled_down(self, model: str) -> bool:
        state = self._get_state(model)
        if state.cooldown_until == 0.0:
            return False
        if time.monotonic() < state.cooldown_until:
            return True
        state.cooldown_until = 0.0
        return False

    def get_status(self, model: str) -> dict:
        state = self._get_state(model)
        now = time.monotonic()
        cutoff = now - self.failure_window
        recent_failures = len([t for t in state.failure_timestamps if t > cutoff])
        cooled_down = self.is_cooled_down(model)
        remaining = max(0, state.cooldown_until - now) if cooled_down else 0
        return {
            "model": model,
            "recent_failures": recent_failures,
            "total_failures": state.total_failures,
            "total_successes": state.total_successes,
            "cooled_down": cooled_down,
            "cooldown_remaining_s": round(remaining, 1),
        }

    def get_all_status(self) -> list:
        return [self.get_status(model) for model in self._models]
