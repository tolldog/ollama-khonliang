"""
Personality configuration system for multi-agent LLM applications.

Personalities define the perspective, focus, and voting weight of an agent
within a consensus team. They can be loaded from JSON config files or defined
programmatically, and are identified by @mention in chat interfaces.

Usage:

    registry = PersonalityRegistry()
    config = registry.get("analyst")
    prompt = build_prompt("analyst", "Why does the login flow fail?")

    # Load from file
    registry = PersonalityRegistry("config/personalities.json")

    # Add custom
    registry.add_custom(
        id="security",
        name="Security Reviewer",
        description="Flags authentication and authorization risks",
        voting_weight=0.20,
        focus=["auth", "permissions", "vulnerabilities"],
    )
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PersonalityConfig:
    """
    Configuration for a named agent personality.

    Personalities shape how an agent frames its analysis — focus areas,
    voting weight within a consensus team, and an optional custom system
    prompt that overrides the default role description.

    Args:
        id:            Unique identifier (used in @mention lookup)
        name:          Display name
        description:   One-line description of this personality's perspective
        voting_weight: Relative weight in consensus aggregation (0.0–1.0)
        focus:         Topic areas this personality prioritises
        system_prompt: Custom LLM system prompt (empty = use default template)
        enabled:       Whether this personality participates in consensus
        aliases:       Alternative names for @mention resolution
        metadata:      Free-form extra config (domain-specific extensions)
    """

    id: str
    name: str
    description: str
    voting_weight: float
    focus: List[str]
    system_prompt: str = ""
    enabled: bool = True
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "voting_weight": self.voting_weight,
            "focus": self.focus,
            "system_prompt": self.system_prompt,
            "enabled": self.enabled,
            "aliases": self.aliases,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityConfig":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            voting_weight=data.get("voting_weight", 0.10),
            focus=data.get("focus", []),
            system_prompt=data.get("system_prompt", ""),
            enabled=data.get("enabled", True),
            aliases=data.get("aliases", []),
            metadata=data.get("metadata", {}),
        )


# Generic default personalities — suitable as a starting point for any
# multi-agent application. Override via config file or add_custom().
DEFAULT_PERSONALITIES: Dict[str, Dict[str, Any]] = {
    "resolver": {
        "name": "Quick Resolver",
        "description": "Pragmatic, solution-first responder",
        "voting_weight": 0.30,
        "focus": ["resolution", "clarity", "speed", "actionability"],
        "system_prompt": """You are the Quick Resolver.
Your goal is fast, actionable answers. You cut through ambiguity and propose
concrete next steps. You prefer a working solution now over a perfect one later.
When evaluating options, favour the simplest approach that solves the problem.""",
        "aliases": ["resolve", "solution", "quick"],
    },
    "analyst": {
        "name": "Root Cause Analyst",
        "description": "Systematic investigation of underlying causes",
        "voting_weight": 0.25,
        "focus": ["root_cause", "patterns", "diagnostics", "evidence"],
        "system_prompt": """You are the Root Cause Analyst.
You dig beneath surface symptoms to find the true source of problems.
You ask 'why' repeatedly, gather evidence, and resist jumping to conclusions.
You prefer well-understood causes over guesses, and flag gaps in the data.""",
        "aliases": ["analysis", "diagnose", "investigate"],
    },
    "advocate": {
        "name": "User Advocate",
        "description": "Centres the user's experience and communication",
        "voting_weight": 0.25,
        "focus": ["empathy", "clarity", "user_impact", "communication"],
        "system_prompt": """You are the User Advocate.
You speak for the person affected by this issue. You evaluate responses for
clarity, tone, and empathy. You flag jargon, push for plain-language explanations,
and ensure the user feels heard and informed throughout the process.""",
        "aliases": ["user", "empathy", "communication"],
    },
    "skeptic": {
        "name": "Skeptical Reviewer",
        "description": "Challenges assumptions, surfaces risks and edge cases",
        "voting_weight": 0.20,
        "focus": ["risk", "edge_cases", "assumptions", "second_order_effects"],
        "system_prompt": """You are the Skeptical Reviewer.
Your role is to stress-test proposals. You ask 'what could go wrong?', surface
hidden assumptions, and identify edge cases that others overlook. You are not
obstructionist — you raise concerns so they can be addressed before they become
incidents. You support veto when risk is genuinely unacceptable.""",
        "aliases": ["skeptic", "risk", "review", "caution"],
    },
}


class PersonalityRegistry:
    """
    Registry for managing personality configurations.

    Loads built-in defaults on construction. Custom personalities can be
    added programmatically or loaded from a JSON file (see load_from_file).

    Args:
        config_path: Optional path to a JSON file with a "personalities" list.
                     Entries override defaults with the same id.

    Example:
        >>> registry = PersonalityRegistry("config/personalities.json")
        >>> config = registry.get("analyst")
        >>> print(config.name, config.voting_weight)
    """

    def __init__(self, config_path: Optional[str] = None):
        self._personalities: Dict[str, PersonalityConfig] = {}
        self._aliases: Dict[str, str] = {}

        self._load_defaults()

        if config_path:
            self.load_from_file(config_path)

    def _load_defaults(self) -> None:
        for pid, data in DEFAULT_PERSONALITIES.items():
            self.register(PersonalityConfig(
                id=pid,
                name=data["name"],
                description=data["description"],
                voting_weight=data["voting_weight"],
                focus=data["focus"],
                system_prompt=data.get("system_prompt", ""),
                aliases=data.get("aliases", []),
            ))

    def register(self, config: PersonalityConfig) -> None:
        """Register a personality and index its aliases."""
        self._personalities[config.id] = config
        for alias in config.aliases:
            self._aliases[alias.lower()] = config.id

    def get(self, personality_id: str) -> Optional[PersonalityConfig]:
        """Return a personality by ID or alias. Returns None if not found."""
        if personality_id in self._personalities:
            return self._personalities[personality_id]
        resolved = self.resolve_id(personality_id)
        if resolved:
            return self._personalities.get(resolved)
        return None

    def resolve_id(self, name: str) -> Optional[str]:
        """Resolve a name, alias, or @mention to a personality ID."""
        name = name.lstrip("@").lower().replace(" ", "_")

        if name in self._personalities:
            return name
        if name in self._aliases:
            return self._aliases[name]
        # Partial match
        for pid in self._personalities:
            if name in pid:
                return pid
        return None

    def list_all(self) -> List[PersonalityConfig]:
        return list(self._personalities.values())

    def list_enabled(self) -> List[PersonalityConfig]:
        return [p for p in self._personalities.values() if p.enabled]

    def load_from_file(self, path: str) -> int:
        """
        Load personalities from a JSON config file.

        The file should have the shape::

            {
              "personalities": [
                {"id": "security", "name": "Security Reviewer", ...}
              ]
            }

        Loaded entries override defaults with the same id.

        Returns:
            Number of personalities loaded
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"Personality config not found: {path}")
            return 0
        try:
            data = json.loads(p.read_text())
            count = 0
            for item in data.get("personalities", []):
                self.register(PersonalityConfig.from_dict(item))
                count += 1
            logger.info(f"Loaded {count} personalities from {path}")
            return count
        except Exception as e:
            logger.error(f"Failed to load personalities from {path}: {e}")
            return 0

    def save_to_file(self, path: str) -> None:
        """Save all registered personalities to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"personalities": [c.to_dict() for c in self._personalities.values()]},
            indent=2,
        ))
        logger.info(f"Saved {len(self._personalities)} personalities to {path}")

    def add_custom(
        self,
        id: str,
        name: str,
        description: str,
        voting_weight: float = 0.10,
        focus: Optional[List[str]] = None,
        system_prompt: str = "",
        **kwargs,
    ) -> PersonalityConfig:
        """
        Add a custom personality and return it.

        Args:
            id:            Unique identifier
            name:          Display name
            description:   One-line description
            voting_weight: Consensus weight (0.0–1.0)
            focus:         Focus areas
            system_prompt: Custom system prompt
            **kwargs:      Passed to PersonalityConfig (e.g. aliases, metadata)
        """
        config = PersonalityConfig(
            id=id,
            name=name,
            description=description,
            voting_weight=voting_weight,
            focus=focus or [],
            system_prompt=system_prompt,
            **kwargs,
        )
        self.register(config)
        return config


# Module-level singleton registry
_registry: Optional[PersonalityRegistry] = None


def get_registry() -> PersonalityRegistry:
    """Return the global personality registry, initialising it if needed."""
    global _registry
    if _registry is None:
        _registry = PersonalityRegistry()
    return _registry


def extract_mention(text: str) -> Optional[str]:
    """
    Extract a personality @mention from text and resolve it to a personality ID.

    Args:
        text: User message (e.g. "@analyst why is login slow?")

    Returns:
        Resolved personality ID, or None if no mention found
    """
    match = re.search(r"@(\w+)", text.lower())
    if match:
        return get_registry().resolve_id(match.group(1))
    return None


def format_response(
    personality_id: str,
    response_text: str,
    weight: Optional[float] = None,
) -> str:
    """
    Wrap a response with a personality header and attribution footer.

    Args:
        personality_id: The responding personality's ID
        response_text:  The LLM-generated response
        weight:         Optional weight to display in footer (uses config default)

    Returns:
        Formatted string with header/footer, or response_text unchanged if
        personality is not found.
    """
    config = get_registry().get(personality_id)
    if not config:
        return response_text

    w = weight if weight is not None else config.voting_weight
    header = f"**{config.name}**\n\n"
    footer = f"\n\n---\n*[{config.name} · {w * 100:.0f}% voting weight]*"
    return header + response_text + footer


def build_prompt(
    personality_id: str,
    question: str,
    context: str = "",
) -> str:
    """
    Build a system-prefixed prompt for a personality.

    Uses the personality's custom system_prompt if set; otherwise generates
    one from the name, description, and focus fields.

    Args:
        personality_id: Personality to speak as
        question:       The user's question or task
        context:        Optional retrieved context to inject

    Returns:
        Complete prompt string ready for LLM generation
    """
    config = get_registry().get(personality_id)
    if not config:
        return question

    if config.system_prompt:
        system_part = config.system_prompt
    else:
        system_part = (
            f"You are the {config.name}.\n"
            f"Role: {config.description}\n"
            f"Voting weight: {config.voting_weight * 100:.0f}%\n"
            f"Focus areas: {', '.join(config.focus)}"
        )

    parts = [system_part]
    if context:
        parts.append(f"\nContext:\n{context}")
    parts.append(f"\nQuestion: {question}")
    parts.append(f"\nRespond as the {config.name}:")

    return "\n".join(parts)
