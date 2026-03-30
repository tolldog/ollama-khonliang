"""
Blackboard — shared in-memory knowledge store for agent coordination.

Agents post structured entries to named sections with optional TTL.
Other agents (or roles) read sections to build shared context.

Example:

    board = Blackboard(default_ttl=120)
    board.post("analyst", "market", "trend", "Bearish divergence on RSI")
    board.post("researcher", "market", "news", "Fed minutes released")

    # Read all entries in a section
    entries = board.read("market")

    # Build context string for prompt injection
    ctx = board.build_context(["market", "signals"], max_entries=10)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BlackboardEntry:
    """A single entry on the blackboard."""

    agent_id: str
    section: str
    key: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Seconds until expiry; None = use board default

    @property
    def expires_at(self) -> Optional[float]:
        """Absolute expiry time, or None if no TTL."""
        if self.ttl is not None:
            return self.timestamp + self.ttl
        return None

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this entry has expired."""
        if self.ttl is None:
            return False
        return (now or time.time()) > self.timestamp + self.ttl


class Blackboard:
    """
    Shared in-memory blackboard for multi-agent coordination.

    Agents post key-value entries to named sections. Entries auto-expire
    based on TTL. The board provides a build_context() method to format
    current entries into a string suitable for LLM prompt injection.

    Args:
        default_ttl: Default time-to-live in seconds for entries.
            None means entries never expire.
    """

    def __init__(self, default_ttl: Optional[float] = None):
        self.default_ttl = default_ttl
        # {section: {key: BlackboardEntry}}
        self._sections: Dict[str, Dict[str, BlackboardEntry]] = {}

    def post(
        self,
        agent_id: str,
        section: str,
        key: str,
        content: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Post an entry to the blackboard.

        If a key already exists in the section, it is overwritten.

        Args:
            agent_id: ID of the posting agent
            section: Section name (e.g. "market", "signals", "alerts")
            key: Entry key within the section
            content: The content to store (string, dict, etc.)
            ttl: Time-to-live in seconds. None uses the board's default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        entry = BlackboardEntry(
            agent_id=agent_id,
            section=section,
            key=key,
            content=content,
            ttl=effective_ttl,
        )

        if section not in self._sections:
            self._sections[section] = {}
        self._sections[section][key] = entry

        logger.debug(
            f"Blackboard: {agent_id} posted to {section}/{key} "
            f"(ttl={effective_ttl})"
        )

    def read(
        self, section: str, key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Read entries from a section.

        Args:
            section: Section name to read from
            key: Optional specific key. If None, returns all live entries.

        Returns:
            Dict mapping keys to their content. Expired entries are
            excluded and cleaned up.
        """
        entries = self._sections.get(section, {})
        now = time.time()
        result: Dict[str, Any] = {}

        if key is not None:
            entry = entries.get(key)
            if entry and not entry.is_expired(now):
                result[key] = entry.content
            elif entry and entry.is_expired(now):
                del entries[key]
            return result

        # Return all live entries, clean expired ones
        expired_keys = []
        for k, entry in entries.items():
            if entry.is_expired(now):
                expired_keys.append(k)
            else:
                result[k] = entry.content

        for k in expired_keys:
            del entries[k]

        return result

    def build_context(
        self,
        sections: Optional[List[str]] = None,
        max_entries: int = 50,
    ) -> str:
        """
        Build a formatted context string from blackboard entries.

        Suitable for injecting into LLM prompts via BaseRole.build_context().

        Args:
            sections: List of section names to include. None = all sections.
            max_entries: Maximum total entries to include.

        Returns:
            Formatted string with section headers and entries.
            Empty string if no live entries exist.
        """
        target_sections = sections or list(self._sections.keys())
        now = time.time()
        lines: List[str] = []
        count = 0

        for section in target_sections:
            entries = self._sections.get(section, {})
            section_lines: List[str] = []
            expired_keys = []

            for key, entry in entries.items():
                if entry.is_expired(now):
                    expired_keys.append(key)
                    continue
                if count >= max_entries:
                    break
                section_lines.append(f"  [{key}] ({entry.agent_id}): {entry.content}")
                count += 1

            for k in expired_keys:
                del entries[k]

            if section_lines:
                lines.append(f"[{section}]")
                lines.extend(section_lines)

            if count >= max_entries:
                break

        return "\n".join(lines)

    def clear_section(self, section: str) -> None:
        """Remove all entries from a section."""
        self._sections.pop(section, None)

    def clear(self) -> None:
        """Remove all entries from all sections."""
        self._sections.clear()

    @property
    def sections(self) -> List[str]:
        """Return list of section names that have entries."""
        return list(self._sections.keys())
