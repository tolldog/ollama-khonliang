"""
Base researcher — abstract class for pluggable research workers.

Subclass BaseResearcher to create a new type of researcher (web search,
database lookup, API call, etc.). Register it with a ResearchPool to
have it process tasks from the queue.

Example:

    class MyWebResearcher(BaseResearcher):
        name = "web_search"
        capabilities = ["person_lookup", "historical_context"]

        async def research(self, task: ResearchTask) -> ResearchResult:
            results = search_the_web(task.query)
            return ResearchResult(
                task_id=task.task_id,
                task_type=task.task_type,
                title=f"Web results for: {task.query}",
                content=format_results(results),
            )
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from khonliang.research.models import ResearchResult, ResearchTask

logger = logging.getLogger(__name__)


class BaseResearcher(ABC):
    """
    Abstract base for research workers.

    Subclasses must define:
        name: Unique researcher name (e.g. "web_search")
        capabilities: List of task_types this researcher handles

    And implement:
        research(): Process a task and return a result
    """

    name: str = "base"
    capabilities: List[str] = []

    def can_handle(self, task_type: str) -> bool:
        """Check if this researcher can handle a task type."""
        return task_type in self.capabilities

    @abstractmethod
    async def research(self, task: ResearchTask) -> ResearchResult:
        """
        Process a research task.

        Args:
            task: The research task to process

        Returns:
            ResearchResult with findings

        Raises:
            Exception on failure (pool will catch and mark task as failed)
        """
        ...
