"""
Agent capability registry — registration and discovery.

Agents declare their capabilities at startup. Routers and orchestrators
use the registry to find the right agent(s) for a given task.

Supports both exact-match lookup and embedding-based similarity search.

Usage:
    registry = CapabilityRegistry()
    registry.register("analyst", [
        AgentCapability("analyst", "root_cause_analysis", "Deep investigation"),
    ])

    # Exact match
    agents = registry.find("root_cause_analysis")

    # Similarity search (requires embeddings)
    matches = registry.find_capable("investigate server crash", threshold=0.6)
    # Returns: [("analyst", 0.85), ...]
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """
    A capability that an agent provides.

    Attributes:
        agent_id: Owning agent identifier
        capability: Machine-readable capability name
        description: Human-readable description for LLM prompts
        input_schema: Optional JSON schema for structured input
        embedding: Optional vector for similarity-based task routing
    """

    agent_id: str
    capability: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapability":
        """Deserialize from a dict."""
        return cls(
            agent_id=data["agent_id"],
            capability=data["capability"],
            description=data["description"],
            input_schema=data.get("input_schema"),
            embedding=data.get("embedding"),
        )


class CapabilityRegistry:
    """
    In-memory agent capability registry with optional Redis persistence.

    Example:
        >>> registry = CapabilityRegistry()
        >>> registry.register("triage", [
        ...     AgentCapability("triage", "urgency_detection", "Classify urgency"),
        ... ])
        >>> agents = registry.find("urgency_detection")
        >>> print(agents[0].agent_id)  # "triage"
    """

    def __init__(self, redis_client: Optional[Any] = None):
        """
        Args:
            redis_client: Optional synchronous Redis client for persistence.
                          Must support hset/hget/hdel/hgetall (e.g. redis.Redis).
                          Do NOT pass a redis.asyncio client.
        """
        self._redis = redis_client
        self._redis_key = "khonliang:agent:capabilities"
        self._capabilities: Dict[str, List[AgentCapability]] = {}

    def register(
        self,
        agent_id: str,
        capabilities: List[AgentCapability],
    ) -> None:
        """Register capabilities for an agent. Replaces existing."""
        self._capabilities[agent_id] = capabilities
        logger.info(
            f"Registered {len(capabilities)} capabilities for {agent_id}: "
            f"{[c.capability for c in capabilities]}"
        )

        if self._redis is not None:
            try:
                data = json.dumps([c.to_dict() for c in capabilities])
                self._redis.hset(self._redis_key, agent_id, data)
            except Exception as e:
                logger.warning(f"Failed to persist registry to Redis: {e}")

    def unregister(self, agent_id: str) -> None:
        """Remove all capabilities for an agent."""
        self._capabilities.pop(agent_id, None)
        logger.info(f"Unregistered agent {agent_id}")

        if self._redis is not None:
            try:
                self._redis.hdel(self._redis_key, agent_id)
            except Exception as e:  # nosec B110
                logger.debug(f"Failed to remove {agent_id} from Redis: {e}")

    def find(self, capability: str) -> List[AgentCapability]:
        """Find all agents that provide a given capability."""
        results = []
        for caps in self._capabilities.values():
            for cap in caps:
                if cap.capability == capability:
                    results.append(cap)
        return results

    def find_agents_for(self, capability: str) -> List[str]:
        """Find agent IDs that provide a given capability."""
        return [cap.agent_id for cap in self.find(capability)]

    def get_agent_capabilities(self, agent_id: str) -> List[AgentCapability]:
        """Return all capabilities registered for a specific agent."""
        return self._capabilities.get(agent_id, [])

    def list_all(self) -> Dict[str, List[AgentCapability]]:
        """Return a copy of all agent-to-capabilities mappings."""
        return dict(self._capabilities)

    def list_capabilities(self) -> List[str]:
        """List all unique capability names across all agents."""
        capabilities = set()
        for caps in self._capabilities.values():
            for cap in caps:
                capabilities.add(cap.capability)
        return sorted(capabilities)

    def describe_for_llm(self) -> str:
        """Format the registry as a prompt snippet for LLM routing."""
        if not self._capabilities:
            return "No specialist agents are currently registered."

        lines = ["Available specialist agents:"]
        for agent_id in sorted(self._capabilities.keys()):
            caps = self._capabilities[agent_id]
            cap_descriptions = [
                f"  - {c.capability}: {c.description}" for c in caps
            ]
            lines.append(f"\n{agent_id}:")
            lines.extend(cap_descriptions)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire registry to a nested dict."""
        return {
            agent_id: [c.to_dict() for c in caps]
            for agent_id, caps in self._capabilities.items()
        }

    def load_from_redis(self) -> bool:
        """Load capabilities from Redis. Returns True on success."""
        if self._redis is None:
            return False
        try:
            data = self._redis.hgetall(self._redis_key)
            for agent_id, caps_json in data.items():
                caps = [
                    AgentCapability.from_dict(c) for c in json.loads(caps_json)
                ]
                self._capabilities[agent_id] = caps
            logger.info(f"Loaded {len(data)} agents from Redis registry")
            return True
        except Exception as e:
            logger.warning(f"Failed to load registry from Redis: {e}")
            return False

    def find_capable(
        self,
        task: str,
        threshold: float = 0.6,
        limit: int = 5,
        task_embedding: Optional[List[float]] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> List[Tuple[str, float]]:
        """Find agents capable of a task using embedding similarity.

        Compares the task embedding against each capability's embedding.
        Returns (agent_id, score) pairs sorted by best match.

        Args:
            task: Task description (used with embed_fn if task_embedding not given)
            threshold: Minimum cosine similarity (0-1)
            limit: Max results
            task_embedding: Pre-computed embedding for the task
            embed_fn: Callable that produces an embedding from text.
                Signature: (text: str) -> List[float]
                Used when task_embedding is not provided.

        Returns:
            List of (agent_id, similarity_score) sorted descending.
            Empty list if no capabilities have embeddings.
        """
        # Get task embedding
        emb = task_embedding
        if emb is None and embed_fn is not None:
            try:
                emb = embed_fn(task)
            except Exception as e:
                logger.warning("embed_fn failed for task routing: %s", e)
                return []

        if emb is None:
            logger.debug("No embedding for task routing — use find() for exact match")
            return []

        # Score each agent by best capability match
        agent_scores: Dict[str, float] = {}
        for agent_id, caps in self._capabilities.items():
            best = 0.0
            for cap in caps:
                if cap.embedding is None:
                    continue
                sim = _cosine_similarity(emb, cap.embedding)
                best = max(best, sim)
            if best >= threshold:
                agent_scores[agent_id] = best

        ranked = sorted(agent_scores.items(), key=lambda x: -x[1])
        return ranked[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the registry."""
        total_caps = sum(len(caps) for caps in self._capabilities.values())
        caps_with_emb = sum(
            1
            for caps in self._capabilities.values()
            for c in caps
            if c.embedding is not None
        )
        return {
            "registered_agents": len(self._capabilities),
            "total_capabilities": total_caps,
            "unique_capabilities": len(self.list_capabilities()),
            "capabilities_with_embeddings": caps_with_emb,
            "redis_backed": self._redis is not None,
        }


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        logger.debug(
            "Embedding dimension mismatch in cosine similarity: %d vs %d",
            len(a),
            len(b),
        )
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
