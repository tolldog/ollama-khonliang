"""
Research triggers — detect when a message should queue research tasks.

Triggers parse user messages for patterns like:
    !lookup: Timothy Tolle
    !search: Maryland 1700 history
    !find: census records Adams County Ohio

And also detect implicit research needs from agent responses
(e.g. "I don't have enough information about...").

Usage:
    trigger = ResearchTrigger(pool)
    trigger.add_prefix("!lookup", "person_lookup")
    trigger.add_prefix("!search", "web_search")

    # Check a message — returns list of submitted task IDs
    task_ids = trigger.check_message("!lookup: Timothy Tolle", scope="toll")

    # Also check agent responses for implicit research needs
    task_ids = trigger.check_response(
        response="I don't have information about the 1850 census.",
        scope="toll",
    )
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from khonliang.research.models import ResearchTask
from khonliang.research.pool import ResearchPool

logger = logging.getLogger(__name__)


class ResearchTrigger:
    """
    Detects research-worthy patterns in messages and queues tasks.

    Supports:
    - Prefix triggers: "!lookup: query" -> queues a task
    - Pattern triggers: regex match -> queues a task
    - Implicit triggers: keywords in agent responses that suggest
      the agent needs more information
    """

    def __init__(self, pool: ResearchPool):
        self.pool = pool
        self._prefix_triggers: Dict[str, str] = {}  # prefix -> task_type
        self._pattern_triggers: List[Tuple[re.Pattern, str, str]] = []
        self._implicit_patterns: List[Tuple[re.Pattern, str]] = []

        # Default prefix triggers
        self.add_prefix("!lookup", "person_lookup")
        self.add_prefix("!search", "web_search")
        self.add_prefix("!find", "web_search")
        self.add_prefix("!history", "historical_context")

        # Default implicit triggers (agent needs more info)
        self.add_implicit(
            r"(?i)I don't have (enough )?information",
            "web_search",
        )
        self.add_implicit(
            r"(?i)no (data|records|results) (found|available)",
            "web_search",
        )
        self.add_implicit(
            r"(?i)could not (find|verify|confirm)",
            "web_search",
        )

    def add_prefix(self, prefix: str, task_type: str) -> None:
        """Add a command prefix trigger (e.g. "!lookup" -> "person_lookup")."""
        self._prefix_triggers[prefix.lower()] = task_type

    def add_pattern(
        self, pattern: str, task_type: str, query_group: str = "query"
    ) -> None:
        """Add a regex pattern trigger with a named group for the query."""
        self._pattern_triggers.append(
            (re.compile(pattern, re.IGNORECASE), task_type, query_group)
        )

    def add_implicit(self, pattern: str, task_type: str) -> None:
        """Add an implicit trigger that fires on agent responses."""
        self._implicit_patterns.append(
            (re.compile(pattern, re.IGNORECASE), task_type)
        )

    def check_message(
        self,
        message: str,
        scope: str = "global",
        source: str = "user",
    ) -> List[str]:
        """
        Check a user message for research triggers.

        Returns list of submitted task IDs (empty if no triggers matched).
        The original message content (after the prefix) is returned as
        remaining_text for the caller to process normally.
        """
        task_ids = []

        # Check prefix triggers: "!lookup: query text"
        msg_lower = message.lower().strip()
        for prefix, task_type in self._prefix_triggers.items():
            if msg_lower.startswith(prefix):
                # Extract query after prefix (handle "!lookup:" and "!lookup ")
                query = message[len(prefix):].lstrip(":").strip()
                if query:
                    try:
                        tid = self.pool.submit(ResearchTask(
                            task_type=task_type,
                            query=query,
                            scope=scope,
                            source=source,
                        ))
                        task_ids.append(tid)
                        logger.info(
                            f"Trigger '{prefix}' queued {task_type}: "
                            f"'{query[:40]}'"
                        )
                    except ValueError as e:
                        logger.warning(f"Failed to queue research: {e}")
                break

        # Check pattern triggers
        for pattern, task_type, query_group in self._pattern_triggers:
            match = pattern.search(message)
            if match:
                query = match.group(query_group) if query_group else message
                try:
                    tid = self.pool.submit(ResearchTask(
                        task_type=task_type,
                        query=query,
                        scope=scope,
                        source=source,
                    ))
                    task_ids.append(tid)
                except ValueError as e:
                    logger.warning(f"Failed to queue research: {e}")

        return task_ids

    def check_response(
        self,
        response: str,
        original_query: str = "",
        scope: str = "global",
    ) -> List[str]:
        """
        Check an agent response for implicit research needs.

        If the agent indicates it lacks information, automatically
        queue a research task to fill the gap.
        """
        task_ids = []

        for pattern, task_type in self._implicit_patterns:
            if pattern.search(response):
                query = original_query or response[:100]
                try:
                    tid = self.pool.submit(ResearchTask(
                        task_type=task_type,
                        query=query,
                        scope=scope,
                        source="agent_implicit",
                        priority=-1,  # lower priority than explicit requests
                    ))
                    task_ids.append(tid)
                    logger.info(
                        f"Implicit trigger queued {task_type}: "
                        f"'{query[:40]}'"
                    )
                except ValueError as e:
                    logger.debug(f"Implicit research queue failed: {e}")
                break  # Only one implicit trigger per response

        return task_ids

    def strip_prefix(self, message: str) -> Optional[str]:
        """
        If message starts with a trigger prefix, return the query part.
        Returns None if no prefix matched.
        """
        msg_lower = message.lower().strip()
        for prefix in self._prefix_triggers:
            if msg_lower.startswith(prefix):
                return message[len(prefix):].lstrip(":").strip()
        return None
