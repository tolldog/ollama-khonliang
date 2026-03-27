"""
Generic agent configuration registry.

Provides a typed, file-backed registry for named agent configurations.
Load configs from JSON/YAML, enable/disable agents, and persist changes.

Usage:

    from dataclasses import dataclass
    from khonliang.agents.registry import ConfigRegistry

    @dataclass
    class BotConfig:
        id: str
        name: str
        model: str
        system_prompt: str = ""
        enabled: bool = True

        def to_dict(self):
            return self.__dict__.copy()

        @classmethod
        def from_dict(cls, data):
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    registry = ConfigRegistry(BotConfig, "config/bots.json")
    registry.load()

    bot = registry.get("triage")
    registry.enable("triage")
    registry.disable("escalation")
    registry.save()
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigRegistry(Generic[T]):
    """
    Generic registry for named agent/role configurations.

    Type parameter T must be a dataclass (or any class) that implements:
        - `id: str`              — unique identifier
        - `enabled: bool`        — whether the agent is active
        - `to_dict() -> dict`    — for JSON serialization
        - `from_dict(dict) -> T` — classmethod for deserialization

    Args:
        factory:       Callable that creates T from a dict (e.g. MyConfig.from_dict)
        config_path:   Path to JSON file for persistence
        defaults:      Optional dict of id → config-dict to use when no file exists
    """

    def __init__(
        self,
        factory: Callable[[Dict], T],
        config_path: str = "config/agents.json",
        defaults: Optional[Dict[str, Dict]] = None,
    ):
        self._factory = factory
        self._config_path = Path(config_path)
        self._defaults = defaults or {}
        self._configs: Dict[str, T] = {}

    def load(self) -> None:
        """
        Load configs from the JSON file, falling back to defaults if missing.
        """
        if self._config_path.exists():
            try:
                with open(self._config_path) as f:
                    data = json.load(f)

                if isinstance(data, dict) and "agents" in data:
                    items = data["agents"]
                elif isinstance(data, list):
                    items = data
                else:
                    items = list(data.values())

                for item in items:
                    cfg = self._factory(item)
                    self._configs[cfg.id] = cfg  # type: ignore[attr-defined]

                logger.info(f"Loaded {len(self._configs)} configs from {self._config_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {self._config_path}: {e} — using defaults")

        # Fall back to defaults
        for agent_id, cfg_dict in self._defaults.items():
            cfg_dict.setdefault("id", agent_id)
            cfg = self._factory(cfg_dict)
            self._configs[cfg.id] = cfg  # type: ignore[attr-defined]

        if self._configs:
            logger.info(f"Loaded {len(self._configs)} default configs")

    def save(self) -> None:
        """Persist current configs to the JSON file."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"agents": [cfg.to_dict() for cfg in self._configs.values()]}  # type: ignore[attr-defined]
        with open(self._config_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self._configs)} configs to {self._config_path}")

    def get(self, agent_id: str) -> Optional[T]:
        """Return the config for agent_id, or None if not found."""
        return self._configs.get(agent_id)

    def get_all(self, enabled_only: bool = False) -> List[T]:
        """
        Return all registered configs.

        Args:
            enabled_only: If True, only return configs where enabled=True
        """
        configs = list(self._configs.values())
        if enabled_only:
            configs = [c for c in configs if getattr(c, "enabled", True)]
        return configs

    def register(self, config: T) -> None:
        """Add or replace a config entry."""
        self._configs[config.id] = config  # type: ignore[attr-defined]

    def remove(self, agent_id: str) -> bool:
        """Remove an agent config. Returns True if it existed."""
        if agent_id in self._configs:
            del self._configs[agent_id]
            return True
        return False

    def enable(self, agent_id: str) -> bool:
        """Enable an agent. Returns True if the agent exists."""
        cfg = self._configs.get(agent_id)
        if cfg is None:
            return False
        cfg.enabled = True  # type: ignore[attr-defined]
        return True

    def disable(self, agent_id: str) -> bool:
        """Disable an agent. Returns True if the agent exists."""
        cfg = self._configs.get(agent_id)
        if cfg is None:
            return False
        cfg.enabled = False  # type: ignore[attr-defined]
        return True

    def find_by_alias(self, alias: str) -> Optional[T]:
        """
        Find a config by alias (if the config has an `aliases` list field).

        Returns the first config whose aliases list contains the given alias.
        """
        alias_lower = alias.lower()
        for cfg in self._configs.values():
            aliases = getattr(cfg, "aliases", [])
            if alias_lower in [a.lower() for a in aliases]:
                return cfg
        return None

    def __len__(self) -> int:
        return len(self._configs)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._configs
