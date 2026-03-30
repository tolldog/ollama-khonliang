"""
Adaptive weight management for agent consensus.

Adjusts agent weights based on recent performance and optional
regime-specific multipliers.
"""

import logging
from typing import Any, Dict, Optional

from khonliang.consensus.models import AgentPerformance

logger = logging.getLogger(__name__)


class AdaptiveWeightManager:
    """
    Adjust agent weights based on performance.

    Features:
    - Performance-based weight adjustment
    - Optional regime-specific multipliers
    - Min/max weight bounds to prevent dominance
    - Smooth weight transitions via EMA

    Example:
        >>> manager = AdaptiveWeightManager(
        ...     base_weights={"analyst": 0.3, "reviewer": 0.3, "skeptic": 0.2},
        ... )
        >>> weights = manager.calculate_weights(performances, regime="cautious")
    """

    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.40
    LEARNING_RATE = 0.1

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        regime_multipliers: Optional[Dict[str, Dict[str, float]]] = None,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
        learning_rate: float = LEARNING_RATE,
    ):
        """
        Initialize the weight manager.

        Args:
            base_weights: Starting weights per agent (default empty)
            regime_multipliers: Optional dict of regime -> {agent_id: multiplier}.
                Example: {"cautious": {"risk": 1.5, "quant": 0.8}}
            min_weight: Minimum weight for any agent
            max_weight: Maximum weight for any agent
            learning_rate: Rate of weight adjustment (EMA smoothing)
        """
        self.base_weights = base_weights or {}
        self.regime_multipliers = regime_multipliers or {}
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.learning_rate = learning_rate
        self._current_weights = self.base_weights.copy()

    @property
    def current_weights(self) -> Dict[str, float]:
        """Return a copy of the current agent weights."""
        return self._current_weights.copy()

    def calculate_weights(
        self,
        performances: Dict[str, AgentPerformance],
        regime: str = "normal",
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on recent performance.

        Args:
            performances: Dictionary of agent_id -> AgentPerformance
            regime: Current regime name (must match a key in regime_multipliers)

        Returns:
            Dictionary of agent weights (sum to 1.0)
        """
        weights = {}

        for agent_id, base_weight in self.base_weights.items():
            perf = performances.get(agent_id)

            if perf is None:
                weights[agent_id] = base_weight
                continue

            # Adjust based on recent accuracy
            # At 50% accuracy -> factor 1.0, at 100% -> 2.0, at 0% -> 0.5
            accuracy_factor = (
                perf.recent_accuracy / 0.5 if perf.recent_accuracy > 0 else 0.5
            )
            accuracy_factor = max(0.5, min(2.0, accuracy_factor))

            adjusted = base_weight * accuracy_factor

            # Apply regime multiplier
            regime_mult = self._get_regime_multiplier(agent_id, regime)
            adjusted *= regime_mult

            adjusted = max(self.min_weight, min(self.max_weight, adjusted))
            weights[agent_id] = adjusted

        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        self._smooth_update(weights)
        return self._current_weights.copy()

    def _get_regime_multiplier(self, agent_id: str, regime: str) -> float:
        regime_mults = self.regime_multipliers.get(regime, {})
        return regime_mults.get(agent_id, 1.0)

    def _smooth_update(self, new_weights: Dict[str, float]) -> None:
        """Smoothly update current weights using exponential moving average."""
        for agent_id, new_weight in new_weights.items():
            current = self._current_weights.get(agent_id, new_weight)
            smoothed = current + self.learning_rate * (new_weight - current)
            self._current_weights[agent_id] = smoothed

        total = sum(self._current_weights.values())
        if total > 0:
            self._current_weights = {
                k: v / total for k, v in self._current_weights.items()
            }

    def reset_weights(self) -> None:
        """Reset weights to base values."""
        self._current_weights = self.base_weights.copy()
        logger.info("Reset agent weights to base values")

    def get_weight_changes(self) -> Dict[str, float]:
        """Return the delta between current and base weights per agent."""
        return {
            agent_id: self._current_weights.get(agent_id, 0) - base
            for agent_id, base in self.base_weights.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize weight manager state to a plain dict."""
        return {
            "base_weights": self.base_weights,
            "current_weights": self._current_weights,
            "changes": self.get_weight_changes(),
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveWeightManager":
        """Restore a weight manager from a serialized dict."""
        manager = cls(
            base_weights=data.get("base_weights"),
            min_weight=data.get("min_weight", cls.MIN_WEIGHT),
            max_weight=data.get("max_weight", cls.MAX_WEIGHT),
        )
        if "current_weights" in data:
            manager._current_weights = data["current_weights"]
        return manager


class WeightScheduler:
    """
    Schedule weight recalculation at intervals.

    Example:
        >>> scheduler = WeightScheduler(manager, update_interval_hours=24)
        >>> if scheduler.should_update():
        ...     scheduler.update_weights(performances)
    """

    def __init__(
        self,
        weight_manager: AdaptiveWeightManager,
        update_interval_hours: int = 24,
    ):
        self.manager = weight_manager
        self.update_interval_hours = update_interval_hours
        self._last_update = None

    def should_update(self) -> bool:
        """True if enough time has elapsed since the last weight update."""
        from datetime import datetime, timedelta

        if self._last_update is None:
            return True
        elapsed = datetime.now() - self._last_update
        return elapsed >= timedelta(hours=self.update_interval_hours)

    def update_weights(
        self,
        performances: Dict[str, AgentPerformance],
        regime: str = "normal",
    ) -> Dict[str, float]:
        """Recalculate weights from performances and record the update time."""
        from datetime import datetime

        weights = self.manager.calculate_weights(performances, regime)
        self._last_update = datetime.now()
        logger.info(f"Updated weights: {weights}")
        return weights
