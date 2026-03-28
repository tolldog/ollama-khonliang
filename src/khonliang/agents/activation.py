"""
Agent activation rules — schedule-based and state-based agent control.

Define when agents should be active via ActivationRules, then use
ActivationTracker to check activation state at runtime.

Example:

    rule = ActivationRule(
        mode=ActivationMode.SCHEDULED,
        schedule_cron="*/5 * * * *",
    )
    tracker = ActivationTracker()
    tracker.register("my_agent", rule)

    if tracker.is_active("my_agent"):
        # run agent
        ...
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ActivationMode(str, Enum):
    """How an agent decides whether it should be active."""

    ALWAYS = "always"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    CONDITIONAL = "conditional"
    MANUAL = "manual"


@dataclass
class ActivationRule:
    """
    A rule that determines when an agent should be active.

    Attributes:
        mode:            Activation strategy
        schedule_cron:   Cron expression for SCHEDULED mode
        condition:       Callable returning bool for CONDITIONAL mode
        cooldown_seconds: Minimum time between activations
        max_activations: Max activations per window (0 = unlimited)
        window_seconds:  Rolling window for max_activations counting
        enabled:         Master switch — if False, agent never activates
        metadata:        Arbitrary extra config
    """

    mode: ActivationMode = ActivationMode.ALWAYS
    schedule_cron: Optional[str] = None
    condition: Optional[Callable[[], bool]] = None
    cooldown_seconds: float = 0.0
    max_activations: int = 0
    window_seconds: float = 3600.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActivationTracker:
    """
    Tracks agent activation state and enforces rules.

    Register agents with their ActivationRule, then query is_active()
    before dispatching work to them.
    """

    def __init__(self):
        self._rules: Dict[str, ActivationRule] = {}
        self._last_activation: Dict[str, float] = {}
        self._activation_history: Dict[str, List[float]] = {}

    def register(self, agent_id: str, rule: ActivationRule) -> None:
        """Register an activation rule for an agent."""
        self._rules[agent_id] = rule
        self._activation_history.setdefault(agent_id, [])
        logger.debug(f"Registered activation rule for '{agent_id}': mode={rule.mode.value}")

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent's activation rule. Returns True if it existed."""
        if agent_id in self._rules:
            del self._rules[agent_id]
            self._last_activation.pop(agent_id, None)
            self._activation_history.pop(agent_id, None)
            return True
        return False

    def is_active(self, agent_id: str) -> bool:
        """
        Check whether an agent should be active right now.

        Returns True (fail-open) if the agent has no registered rule.
        """
        rule = self._rules.get(agent_id)
        if rule is None:
            return True

        if not rule.enabled:
            return False

        now = time.time()

        # Check cooldown
        if rule.cooldown_seconds > 0:
            last = self._last_activation.get(agent_id, 0.0)
            if now - last < rule.cooldown_seconds:
                return False

        # Check rate limit
        if rule.max_activations > 0:
            history = self._activation_history.get(agent_id, [])
            cutoff = now - rule.window_seconds
            recent = [t for t in history if t > cutoff]
            self._activation_history[agent_id] = recent
            if len(recent) >= rule.max_activations:
                return False

        # Mode-specific checks
        if rule.mode == ActivationMode.ALWAYS:
            return True

        if rule.mode == ActivationMode.MANUAL:
            return False

        if rule.mode == ActivationMode.CONDITIONAL:
            if rule.condition is not None:
                try:
                    return rule.condition()
                except Exception as e:
                    logger.warning(f"Condition check failed for '{agent_id}': {e}")
                    return False
            return True

        if rule.mode == ActivationMode.SCHEDULED:
            return self._check_cron(rule.schedule_cron)

        if rule.mode == ActivationMode.EVENT_DRIVEN:
            # Event-driven agents are activated externally via record_activation
            return False

        return True

    def record_activation(self, agent_id: str) -> None:
        """Record that an agent was activated (for cooldown/rate tracking)."""
        now = time.time()
        self._last_activation[agent_id] = now
        self._activation_history.setdefault(agent_id, []).append(now)

    def get_rule(self, agent_id: str) -> Optional[ActivationRule]:
        """Get the activation rule for an agent."""
        return self._rules.get(agent_id)

    def get_all_rules(self) -> Dict[str, ActivationRule]:
        """Return a copy of all registered rules."""
        return dict(self._rules)

    def get_stats(self) -> Dict[str, Any]:
        """Return activation statistics."""
        now = time.time()
        stats = {}
        for agent_id, rule in self._rules.items():
            history = self._activation_history.get(agent_id, [])
            cutoff = now - rule.window_seconds
            recent = [t for t in history if t > cutoff]
            stats[agent_id] = {
                "mode": rule.mode.value,
                "enabled": rule.enabled,
                "activations_in_window": len(recent),
                "last_activation": self._last_activation.get(agent_id),
            }
        return stats

    @staticmethod
    def _check_cron(cron_expr: Optional[str]) -> bool:
        """
        Simple cron check. Returns True if the current minute matches.

        For production use, install croniter:
            pip install croniter
        """
        if cron_expr is None:
            return True
        try:
            import datetime

            from croniter import croniter

            cron = croniter(cron_expr, datetime.datetime.now())
            prev = cron.get_prev(datetime.datetime)
            return (datetime.datetime.now() - prev).total_seconds() < 60
        except ImportError:
            logger.warning("croniter not installed — treating SCHEDULED as always-active")
            return True
        except Exception as e:
            logger.warning(f"Cron parse error for '{cron_expr}': {e}")
            return False
