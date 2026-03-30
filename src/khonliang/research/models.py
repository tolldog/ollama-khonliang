"""
Data models for the research system.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ResearchStatus(str, Enum):
    """Status of a research task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchTask:
    """
    A unit of research work to be processed.

    Attributes:
        task_type: What kind of research (e.g. "person_lookup", "web_search")
        query: The search query or lookup key
        scope: Knowledge scope to file results under
        priority: Higher = processed first (default 0)
        metadata: Arbitrary context for the researcher
        source: What triggered this task ("user", "agent", "auto")
        task_id: Auto-generated unique ID
    """

    task_type: str
    query: str
    scope: str = "global"
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "user"
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: ResearchStatus = ResearchStatus.PENDING
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict of the research task."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "query": self.query,
            "scope": self.scope,
            "priority": self.priority,
            "metadata": self.metadata,
            "source": self.source,
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class ResearchResult:
    """
    Result from a completed research task.

    Attributes:
        task_id: ID of the originating task
        task_type: Echoed from the task
        title: Short title for the result
        content: The research findings (text)
        confidence: How confident the researcher is (0.0-1.0)
        sources: URLs or references where the data came from
        metadata: Additional structured data
    """

    task_id: str
    task_type: str
    title: str
    content: str
    confidence: float = 0.7
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scope: str = "global"
    completed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dict of the research result."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "title": self.title,
            "content": self.content,
            "confidence": self.confidence,
            "sources": self.sources,
            "scope": self.scope,
            "completed_at": self.completed_at,
        }
