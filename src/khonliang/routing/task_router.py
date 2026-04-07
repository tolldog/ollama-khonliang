"""
Task router — dynamic capability-driven routing of tasks to agents.

Wraps CapabilityRegistry with cost-aware selection, concurrency limits,
and fallback chains. Inspired by the Federation of Agents (FoA) framework's
semantic routing with cost-biased optimization.

Concurrency is tracked at the agent level (not per-capability). Call
release() after the routed task completes to free the slot.

Usage:
    router = TaskRouter(registry)
    match = router.route("analyze this data for anomalies", embed_fn=my_embed)
    if match:
        result = await agents[match.agent_id].handle(task)
        router.release(match.agent_id)  # free concurrency slot
"""

import collections
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
        score:       Effective score after cost bias (0.0-1.0)
        cost:        Cost of using this capability
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

    def __post_init__(self) -> None:
        self.similarity_threshold = max(0.0, min(1.0, self.similarity_threshold))
        self.cost_weight = max(0.0, min(1.0, self.cost_weight))
        self.max_results = max(1, self.max_results)


class TaskRouter:
    """Routes tasks to the best available agent based on capabilities.

    Combines semantic similarity from CapabilityRegistry.find_capable()
    with cost-biased scoring and concurrency tracking.

    Scoring formula:
        effective_score = similarity * (1 - cost_weight) + (1 - normalized_cost) * cost_weight

    Concurrency is tracked per-agent (not per-capability). An agent's
    max_concurrent is the minimum positive value across its capabilities.

    Example:
        router = TaskRouter(registry)
        match = router.route("find anomalies", embed_fn=my_embed)
        if match:
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
        self._route_history: collections.deque[Dict[str, Any]] = collections.deque(
            maxlen=1000
        )

    def route(
        self,
        task: str,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        task_embedding: Optional[List[float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[RouteMatch]:
        """Route a task to the best available agent.

        Args:
            task: Task description
            embed_fn: Function to embed the task text
            task_embedding: Pre-computed task embedding
            context: Reserved for constraint-based routing extensions

        Returns:
            RouteMatch for the best agent, or None if no match found
        """
        candidates = self.registry.find_capable(
            task=task,
            threshold=self.config.similarity_threshold,
            limit=self.config.max_results * 2,
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
                self._acquire(cap.agent_id)
                self._record_route(match, "exact")
                return match
            return None

        # Score with cost bias, using best-matching capability per agent
        scored = self._score_candidates(candidates, task_embedding, embed_fn, task)

        # Pick the best available agent (atomic check-and-acquire)
        match = None
        for agent_id, cap_name, score, cost in scored:
            if self.config.prefer_available and not self._try_acquire(agent_id):
                continue
            elif not self.config.prefer_available:
                self._acquire(agent_id)

            caps = self.registry.get_agent_capabilities(agent_id)
            desc = ""
            for c in caps:
                if c.capability == cap_name:
                    desc = c.description
                    break

            match = RouteMatch(
                agent_id=agent_id,
                capability=cap_name,
                score=score,
                cost=cost,
                description=desc,
            )
            break

        if match is None:
            return None

        self._record_route(match, "semantic")

        logger.info(
            f"Routed task to {match.agent_id} "
            f"(score={match.score:.3f}, cost={match.cost:.2f})"
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
        """
        candidates = self.registry.find_capable(
            task=task,
            threshold=self.config.similarity_threshold,
            limit=n * 2,
            embed_fn=embed_fn,
            task_embedding=task_embedding,
        )

        scored = self._score_candidates(candidates, task_embedding, embed_fn, task)

        matches = []
        for agent_id, cap_name, score, cost in scored:
            if len(matches) >= n:
                break
            if self.config.prefer_available and not self._try_acquire(agent_id):
                continue
            elif not self.config.prefer_available:
                self._acquire(agent_id)

            caps = self.registry.get_agent_capabilities(agent_id)
            desc = ""
            for c in caps:
                if c.capability == cap_name:
                    desc = c.description
                    break

            match = RouteMatch(
                agent_id=agent_id,
                capability=cap_name,
                score=score,
                cost=cost,
                description=desc,
            )
            matches.append(match)
            self._record_route(match, "semantic_multi")

        return matches

    def release(self, agent_id: str) -> None:
        """Release a concurrency slot for an agent."""
        with self._lock:
            count = self._active_counts.get(agent_id, 0)
            if count > 0:
                self._active_counts[agent_id] = count - 1

    def _score_candidates(
        self,
        candidates: List[Tuple[str, float]],
        task_embedding: Optional[List[float]],
        embed_fn: Optional[Callable],
        task: str,
    ) -> List[Tuple[str, str, float, float]]:
        """Score candidates with cost bias using the best-matching capability.

        Returns list of (agent_id, capability_name, effective_score, cost)
        sorted by effective_score descending.
        """
        if not candidates:
            return []

        w = self.config.cost_weight

        # Compute task embedding once (not per-candidate)
        emb = task_embedding
        if emb is None and embed_fn is not None:
            try:
                emb = embed_fn(task)
            except Exception:
                emb = None

        # For each agent, find the best-matching capability and its cost
        agent_info: Dict[str, Tuple[str, float, float]] = {}  # {id: (cap_name, sim, cost)}
        for agent_id, similarity in candidates:
            caps = self.registry.get_agent_capabilities(agent_id)
            if not caps:
                agent_info[agent_id] = ("", similarity, 0.0)
                continue

            # Find the best-matching capability by embedding similarity
            best_cap_name = caps[0].capability
            best_cost = caps[0].cost_per_call

            if emb is not None:
                from khonliang.agents.capabilities import _cosine_similarity

                best_sim = -1.0
                for c in caps:
                    if c.embedding is not None:
                        sim = _cosine_similarity(emb, c.embedding)
                        if sim > best_sim:
                            best_sim = sim
                            best_cap_name = c.capability
                            best_cost = c.cost_per_call

            agent_info[agent_id] = (best_cap_name, similarity, best_cost)

        # Normalize costs
        costs = [info[2] for info in agent_info.values()]
        max_cost = max(costs) if costs else 1.0
        if max_cost == 0:
            max_cost = 1.0

        scored = []
        for agent_id, (cap_name, similarity, cost) in agent_info.items():
            norm_cost = cost / max_cost
            effective = similarity * (1 - w) + (1 - norm_cost) * w
            scored.append((agent_id, cap_name, effective, cost))

        scored.sort(key=lambda x: -x[2])
        return scored

    def _try_acquire(self, agent_id: str) -> bool:
        """Atomically check availability and acquire a slot.

        Returns True if a slot was acquired, False if at capacity.
        """
        caps = self.registry.get_agent_capabilities(agent_id)
        max_conc = min(
            (c.max_concurrent for c in caps if c.max_concurrent > 0),
            default=0,
        )
        with self._lock:
            current = self._active_counts.get(agent_id, 0)
            if max_conc == 0 or current < max_conc:
                self._active_counts[agent_id] = current + 1
                return True
            return False

    def _is_available(self, agent_id: str) -> bool:
        """Check if an agent has available concurrency slots (read-only)."""
        caps = self.registry.get_agent_capabilities(agent_id)
        max_conc = min(
            (c.max_concurrent for c in caps if c.max_concurrent > 0),
            default=0,
        )
        if max_conc == 0:
            return True
        with self._lock:
            return self._active_counts.get(agent_id, 0) < max_conc

    def _acquire(self, agent_id: str) -> None:
        """Increment active count (use _try_acquire for safe concurrent access)."""
        with self._lock:
            self._active_counts[agent_id] = (
                self._active_counts.get(agent_id, 0) + 1
            )

    def _record_route(self, match: RouteMatch, method: str) -> None:
        """Record a routing decision for observability."""
        with self._lock:
            self._route_history.append({
                "agent_id": match.agent_id,
                "capability": match.capability,
                "score": match.score,
                "cost": match.cost,
                "method": method,
            })

    def get_stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        with self._lock:
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
