"""
Agent capability registry — registration and discovery.

Agents declare their capabilities at startup. Routers and orchestrators
use the registry to find the right agent(s) for a given task.

Usage:
    registry = CapabilityRegistry()
    registry.register("analyst", [
        AgentCapability("analyst", "root_cause_analysis", "Deep investigation"),
    ])
    agents = registry.find("root_cause_analysis")
    prompt_snippet = registry.describe_for_llm()
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

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
    """

    agent_id: str
    capability: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapability":
        return cls(
            agent_id=data["agent_id"],
            capability=data["capability"],
            description=data["description"],
            input_schema=data.get("input_schema"),
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
        return self._capabilities.get(agent_id, [])

    def list_all(self) -> Dict[str, List[AgentCapability]]:
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
        return {
            agent_id: [c.to_dict() for c in caps]
            for agent_id, caps in self._capabilities.items()
        }

    def load_from_redis(self) -> bool:
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

    def get_stats(self) -> Dict[str, Any]:
        total_caps = sum(len(caps) for caps in self._capabilities.values())
        return {
            "registered_agents": len(self._capabilities),
            "total_capabilities": total_caps,
            "unique_capabilities": len(self.list_capabilities()),
            "redis_backed": self._redis is not None,
        }
