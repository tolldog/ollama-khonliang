"""
Model pool — maps roles to OllamaClient instances.

Lazy-loads clients on first access. Roles that share the same model
reuse a single OllamaClient instance.
"""

import logging
from typing import Dict, Optional

from khonliang.client import OllamaClient

logger = logging.getLogger(__name__)


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
    ):
        """
        Args:
            role_model_map: Dict mapping role names to model names
            base_url: Ollama server URL
            model_timeouts: Optional per-model timeout overrides
                e.g. {"deepseek-r1:32b": 300, "llama3.2:3b": 30}
        """
        self._map = {str(k): v for k, v in role_model_map.items()}
        self._clients: Dict[str, OllamaClient] = {}
        self._base_url = base_url
        self._model_timeouts = model_timeouts

    def get_client(self, role) -> OllamaClient:
        """Get OllamaClient for the given role. Creates on first access."""
        model = self._map.get(str(role))
        if model is None:
            raise KeyError(f"No model configured for role '{role}'")

        if model not in self._clients:
            logger.debug(f"Creating OllamaClient for {role} -> {model}")
            self._clients[model] = OllamaClient(
                model=model,
                base_url=self._base_url,
                model_timeouts=self._model_timeouts,
            )

        return self._clients[model]

    def get_model_name(self, role) -> Optional[str]:
        return self._map.get(str(role))

    async def close_all(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
