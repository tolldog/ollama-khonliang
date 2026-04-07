"""
Task router — dynamic capability-driven routing of tasks to agents.

Wraps CapabilityRegistry with cost-aware selection, concurrency limits,
and fallback chains. Inspired by the Federation of Agents (FoA) framework's
semantic routing with cost-biased optimization.

Usage:
    router = TaskRouter(registry)
    match = router.route("analyze this data for anomalies")
    if match:
        agent = get_agent(match.agent_id)
        result = await agent.handle(task)
        router.release(match.agent_id)  # free concurrency slot
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from khonliang.agents.capabilities import CapabilityRegistry

logger = logging.getLogger(__name__)


@dataclass
class RouteMatch:
    """Result of a task routing decision.

    Attributes:
        agent_id:    Selected agent
        capability:  Matched capability name
        score:       Similarity score (0.0-1.0)
        cost:        Cost of using this agent
        description: Capability description
    """

    agent_id: str
    capability: str
    score: float
    cost: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capability": self.capability,
            "score": self.score,
            "cost": self.cost,
            "description": self.description,
        }


@dataclass
class TaskRouterConfig:
    """Configuration for task routing.

    Attributes:
        similarity_threshold: Minimum embedding similarity (0.0-1.0)
        cost_weight:          How much to penalize cost in scoring (0.0-1.0).
                              0.0 = ignore cost, 1.0 = heavily penalize expensive agents.
        max_results:          Max agents to consider per route
        prefer_available:     Skip agents at max concurrency (default True)
    """

    similarity_threshold: float = 0.5
    cost_weight: float = 0.3
    max_results: int = 5
    prefer_available: bool = True


class TaskRouter:
    """Routes tasks to the best available agent based on capabilities.

    Combines semantic similarity from CapabilityRegistry.find_capable()
    with cost-biased scoring and concurrency tracking.

    Scoring formula:
        effective_score = similarity * (1 - cost_weight) + (1 - normalized_cost) * cost_weight

    Example:
        registry = CapabilityRegistry()
        registry.register("analyst", [
            AgentCapability("analyst", "anomaly_detection", "Detect anomalies",
                            cost_per_call=0.1, max_concurrent=2),
        ])

        router = TaskRouter(registry)
        match = router.route("find anomalies in this dataset",
                             embed_fn=my_embed_fn)
        if match:
            # Use the agent
            result = await agents[match.agent_id].handle(task)
            router.release(match.agent_id)
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        config: Optional[TaskRouterConfig] = None,
    ):
        self.registry = registry
        self.config = config or TaskRouterConfig()
        self._active_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._route_history: List[Dict[str, Any]] = []

    def route(
        self,
        task: str,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        task_embedding: Optional[List[float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[RouteMatch]:
        """Route a task to the best available agent.

        Uses embedding similarity from the capability registry, then
        applies cost-biased scoring and concurrency filtering.

        Args:
            task: Task description
            embed_fn: Function to embed the task text
            task_embedding: Pre-computed task embedding
            context: Optional context (unused currently, reserved for
                     constraint-based routing extensions)

        Returns:
            RouteMatch for the best agent, or None if no match found
        """
        # Get similarity-ranked candidates from registry
        candidates = self.registry.find_capable(
            task=task,
            threshold=self.config.similarity_threshold,
            limit=self.config.max_results * 2,  # over-fetch for filtering
            embed_fn=embed_fn,
            task_embedding=task_embedding,
        )

        if not candidates:
            # Fallback: try exact capability name match
            exact = self.registry.find(task)
            if exact:
                cap = exact[0]
                match = RouteMatch(
                    agent_id=cap.agent_id,
                    capability=cap.capability,
                    score=1.0,
                    cost=cap.cost_per_call,
                    description=cap.description,
                )
                self._record_route(match, "exact")
                return match
            return None

        # Score candidates with cost bias
        scored = self._score_candidates(candidates)

        # Filter by availability
        if self.config.prefer_available:
            scored = [
                (agent_id, score) for agent_id, score in scored
                if self._is_available(agent_id)
            ]

        if not scored:
            return None

        # Pick the best
        best_agent_id, best_score = scored[0]

        # Find the matching capability for metadata
        caps = self.registry.get_agent_capabilities(best_agent_id)
        best_cap = caps[0] if caps else None

        match = RouteMatch(
            agent_id=best_agent_id,
            capability=best_cap.capability if best_cap else task,
            score=best_score,
            cost=best_cap.cost_per_call if best_cap else 0.0,
            description=best_cap.description if best_cap else "",
        )

        # Reserve concurrency slot
        self._acquire(best_agent_id)
        self._record_route(match, "semantic")

        logger.info(
            f"Routed task to {best_agent_id} "
            f"(score={best_score:.3f}, cost={match.cost:.2f})"
        )

        return match

    def route_multi(
        self,
        task: str,
        n: int = 3,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        task_embedding: Optional[List[float]] = None,
    ) -> List[RouteMatch]:
        """Route a task to the top N agents.

        Useful for fan-out patterns where multiple agents should process
        the same task (e.g., ALL_THEN_ORGANIZE conversations).

        Args:
            task: Task description
            n: Number of agents to return
            embed_fn: Embedding function
            task_embedding: Pre-computed embedding

        Returns:
            List of RouteMatch, sorted by score descending
        """
        candidates = self.registry.find_capable(
            task=task,
            threshold=self.config.similarity_threshold,
            limit=n * 2,
            embed_fn=embed_fn,
            task_embedding=task_embedding,
        )

        scored = self._score_candidates(candidates)

        if self.config.prefer_available:
            scored = [
                (aid, s) for aid, s in scored if self._is_available(aid)
            ]

        matches = []
        for agent_id, score in scored[:n]:
            caps = self.registry.get_agent_capabilities(agent_id)
            cap = caps[0] if caps else None
            matches.append(RouteMatch(
                agent_id=agent_id,
                capability=cap.capability if cap else task,
                score=score,
                cost=cap.cost_per_call if cap else 0.0,
                description=cap.description if cap else "",
            ))
            self._acquire(agent_id)

        return matches

    def release(self, agent_id: str) -> None:
        """Release a concurrency slot for an agent.

        Call this after the routed task completes.
        """
        with self._lock:
            count = self._active_counts.get(agent_id, 0)
            if count > 0:
                self._active_counts[agent_id] = count - 1

    def _score_candidates(
        self, candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Apply cost-biased scoring to similarity-ranked candidates.

        Formula: effective = similarity * (1 - w) + (1 - norm_cost) * w
        where w = cost_weight and norm_cost = cost / max_cost.
        """
        if not candidates:
            return []

        w = self.config.cost_weight

        # Get costs for normalization
        costs: Dict[str, float] = {}
        for agent_id, _ in candidates:
            caps = self.registry.get_agent_capabilities(agent_id)
            costs[agent_id] = min(
                (c.cost_per_call for c in caps), default=0.0
            )

        max_cost = max(costs.values()) if costs else 1.0
        if max_cost == 0:
            max_cost = 1.0  # avoid division by zero

        scored = []
        for agent_id, similarity in candidates:
            norm_cost = costs.get(agent_id, 0.0) / max_cost
            effective = similarity * (1 - w) + (1 - norm_cost) * w
            scored.append((agent_id, effective))

        scored.sort(key=lambda x: -x[1])
        return scored

    def _is_available(self, agent_id: str) -> bool:
        """Check if an agent has available concurrency slots."""
        caps = self.registry.get_agent_capabilities(agent_id)
        max_conc = min(
            (c.max_concurrent for c in caps if c.max_concurrent > 0),
            default=0,
        )
        if max_conc == 0:
            return True  # unlimited

        with self._lock:
            return self._active_counts.get(agent_id, 0) < max_conc

    def _acquire(self, agent_id: str) -> None:
        """Increment active count for an agent."""
        with self._lock:
            self._active_counts[agent_id] = (
                self._active_counts.get(agent_id, 0) + 1
            )

    def _record_route(self, match: RouteMatch, method: str) -> None:
        """Record a routing decision for observability."""
        self._route_history.append({
            "agent_id": match.agent_id,
            "capability": match.capability,
            "score": match.score,
            "cost": match.cost,
            "method": method,
        })

    def get_stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        return {
            "total_routes": len(self._route_history),
            "active_agents": {
                k: v for k, v in self._active_counts.items() if v > 0
            },
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "cost_weight": self.config.cost_weight,
                "prefer_available": self.config.prefer_available,
            },
        }
