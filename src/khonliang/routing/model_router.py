"""
Model router — selects which model handles a request within a role.

Sits between BaseRouter (which picks the role) and ModelPool (which
provides the client). Optional — if not configured, roles use their
default model as before.

Example:
    from khonliang.routing.model_router import ModelRouter
    from khonliang.routing.strategies import ComplexityStrategy

    router = ModelRouter(
        role_models={
            "researcher": ["llama3.2:3b", "qwen2.5:7b", "llama3.1:70b"],
            "narrator": ["llama3.2:3b", "llama3.1:8b"],
        },
        strategy=ComplexityStrategy(classifier_client=fast_client),
    )

    selection = await router.select("researcher", "What year was X born?")
    # ModelSelection(model="llama3.2:3b", reason="complexity:simple", ...)
"""

import logging
from typing import Any, Dict, List, Optional

from khonliang.routing.strategies import ModelSelection, RoutingStrategy

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Selects which model to use for a given role and message.

    Delegates to a RoutingStrategy for the actual selection logic.
    Optionally filters out cooled-down models via ModelHealthTracker.

    Args:
        role_models: Dict mapping role names to ordered candidate model
            lists. The order represents tier preference (cheapest first).
        strategy: RoutingStrategy implementation for selection logic.
        health_tracker: Optional ModelHealthTracker to filter out
            unhealthy models before selection.
    """

    def __init__(
        self,
        role_models: Dict[str, List[str]],
        strategy: RoutingStrategy,
        health_tracker: Optional[Any] = None,
    ):
        self._role_models = role_models
        self._strategy = strategy
        self._health_tracker = health_tracker

    def get_candidates(self, role: str) -> List[str]:
        """Return the candidate model list for a role.

        Filters out cooled-down models if a health tracker is configured.
        """
        candidates = list(self._role_models.get(role, []))

        if self._health_tracker and candidates:
            healthy = [
                m for m in candidates if not self._health_tracker.is_cooled_down(m)
            ]
            if healthy:
                candidates = healthy
            else:
                logger.warning(
                    f"All candidates for {role} are cooled down, "
                    f"using full list: {candidates}"
                )

        return candidates

    async def select(self, role: str, message: str) -> ModelSelection:
        """Select a model for the given role and message.

        Returns a ModelSelection with the chosen model, reason, and
        preference list for the scheduler.

        If the role has no candidates configured, returns a fallback
        selection that signals no routing was done.
        """
        candidates = self.get_candidates(role)

        if not candidates:
            return ModelSelection(
                model="",
                reason="no_candidates",
                model_preferences=[],
            )

        return await self._strategy.select(role, message, candidates)
