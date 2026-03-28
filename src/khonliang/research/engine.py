"""
Base engine — a single data source with its own thread pool.

Engines are the lowest-level data acquisition unit. Each engine
represents one external source (a search API, a database, a REST API)
with its own concurrency limits and rate controls.

Engines are composed into Researchers via CompositeResearcher.

Example:

    class GoogleEngine(BaseEngine):
        name = "google"
        max_threads = 4
        rate_limit = 1.0  # seconds between requests

        async def execute(self, query: str, **kwargs) -> List[EngineResult]:
            return do_google_search(query)

    class DDGEngine(BaseEngine):
        name = "ddg"
        max_threads = 3

        async def execute(self, query: str, **kwargs) -> List[EngineResult]:
            return do_ddg_search(query)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """A single result from an engine query."""

    title: str
    content: str
    url: str = ""
    source: str = ""  # engine name
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


class BaseEngine(ABC):
    """
    Abstract base for a data source engine.

    Each engine gets its own thread pool and rate limiter.
    Subclass and implement execute() for your data source.

    Class attributes:
        name: Unique engine identifier
        max_threads: Max concurrent requests for this engine
        rate_limit: Min seconds between requests (0 = no limit)
        timeout: Max seconds per request
    """

    name: str = "base"
    max_threads: int = 2
    rate_limit: float = 0.0
    timeout: float = 15.0

    def __init__(self):
        self._pool: Optional[ThreadPoolExecutor] = None
        self._last_request: float = 0.0
        self._request_count: int = 0
        self._error_count: int = 0

    def start(self) -> None:
        """Initialize the thread pool."""
        self._pool = ThreadPoolExecutor(
            max_workers=self.max_threads,
            thread_name_prefix=f"engine-{self.name}",
        )
        logger.debug(
            f"Engine '{self.name}' started: "
            f"{self.max_threads} threads, "
            f"rate_limit={self.rate_limit}s"
        )

    def stop(self) -> None:
        """Shutdown the thread pool."""
        if self._pool:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

    @abstractmethod
    async def execute(
        self, query: str, **kwargs: Any
    ) -> List[EngineResult]:
        """
        Execute a query against this data source.

        Args:
            query: The search/lookup query
            **kwargs: Engine-specific parameters

        Returns:
            List of results from this engine
        """
        ...

    async def query(
        self, query: str, **kwargs: Any
    ) -> List[EngineResult]:
        """
        Execute with rate limiting and error handling.

        This is the main entry point — wraps execute() with
        rate limiting, timeout, and error tracking.
        """
        # Rate limiting
        if self.rate_limit > 0:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)

        self._last_request = time.monotonic()
        self._request_count += 1

        try:
            if self.timeout > 0:
                results = await asyncio.wait_for(
                    self.execute(query, **kwargs),
                    timeout=self.timeout,
                )
            else:
                results = await self.execute(query, **kwargs)
            # Tag results with engine source
            for r in results:
                if not r.source:
                    r.source = self.name
            return results
        except asyncio.TimeoutError:
            logger.warning(f"Engine '{self.name}' timed out on: {query[:40]}")
            self._error_count += 1
            return []
        except Exception as e:
            logger.warning(f"Engine '{self.name}' error on '{query[:40]}': {e}")
            self._error_count += 1
            return []

    def run_sync(self, func, *args):
        """
        Run a synchronous function in this engine's thread pool.

        Use this inside execute() when your data source library
        is synchronous (most HTTP clients, database drivers, etc.)
        """
        if self._pool is None:
            self.start()
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self._pool, func, *args)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_threads": self.max_threads,
            "rate_limit": self.rate_limit,
            "requests": self._request_count,
            "errors": self._error_count,
        }
