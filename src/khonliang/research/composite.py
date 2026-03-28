"""
Composite researcher — manages multiple engines for a research domain.

A CompositeResearcher fans out queries to all its engines in parallel,
collects results, deduplicates, and optionally filters them.

Example:

    class WebResearcher(CompositeResearcher):
        name = "web"
        capabilities = ["web_search", "person_lookup"]

        def build_queries(self, task):
            return [task.query, f'"{task.query}" genealogy']

    researcher = WebResearcher()
    researcher.add_engine(GoogleEngine())
    researcher.add_engine(DDGEngine())
    researcher.add_engine(BingEngine())

    pool.register(researcher)
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from khonliang.research.base import BaseResearcher
from khonliang.research.engine import BaseEngine, EngineResult
from khonliang.research.models import ResearchResult, ResearchTask

logger = logging.getLogger(__name__)


class CompositeResearcher(BaseResearcher):
    """
    A researcher that delegates to multiple engines.

    Subclass to customize query building and result filtering
    for your domain. Or use directly with engines added at runtime.

    The max_concurrent for the pool is derived from the sum of
    engine max_threads (each engine manages its own concurrency).
    """

    name: str = "composite"
    capabilities: List[str] = []

    def __init__(self):
        self._engines: Dict[str, BaseEngine] = {}
        self._result_filter: Optional[
            Callable[[List[EngineResult], ResearchTask], List[EngineResult]]
        ] = None

    def add_engine(self, engine: BaseEngine) -> None:
        """Add an engine to this researcher."""
        self._engines[engine.name] = engine
        # Update max_concurrent to reflect total engine capacity
        self.max_concurrent = sum(e.max_threads for e in self._engines.values())
        logger.info(
            f"Researcher '{self.name}' added engine '{engine.name}' "
            f"({engine.max_threads} threads)"
        )

    def remove_engine(self, name: str) -> bool:
        """Remove an engine. Returns True if it existed."""
        if name in self._engines:
            self._engines[name].stop()
            del self._engines[name]
            self.max_concurrent = sum(
                e.max_threads for e in self._engines.values()
            )
            return True
        return False

    def set_filter(
        self,
        filter_fn: Callable[
            [List[EngineResult], ResearchTask], List[EngineResult]
        ],
    ) -> None:
        """Set a post-collection filter function."""
        self._result_filter = filter_fn

    def start_engines(self) -> None:
        """Start all engine thread pools."""
        for engine in self._engines.values():
            engine.start()

    def stop_engines(self) -> None:
        """Stop all engine thread pools."""
        for engine in self._engines.values():
            engine.stop()

    # ------------------------------------------------------------------
    # Query building (override in subclasses)
    # ------------------------------------------------------------------

    def build_queries(self, task: ResearchTask) -> List[str]:
        """
        Build query strings for the engines from a research task.

        Override to customize per domain. Default: just the task query.
        Returns a list — each query is sent to all engines.
        """
        return [task.query]

    # ------------------------------------------------------------------
    # Research execution
    # ------------------------------------------------------------------

    async def research(self, task: ResearchTask) -> ResearchResult:
        """
        Fan out queries to all engines, collect and filter results.
        """
        queries = self.build_queries(task)

        # Run all queries across all engines in parallel
        all_results = await self._fan_out(queries)

        # Deduplicate by URL
        seen = set()
        deduped = []
        for r in all_results:
            key = r.url or f"{r.title}:{r.content[:50]}"
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        # Apply filter if set
        if self._result_filter:
            try:
                deduped = self._result_filter(deduped, task)
            except Exception as e:
                logger.warning(f"Result filter error: {e}")

        # Build response
        content_parts = []
        sources = []
        for r in deduped:
            entry = f"[{r.source}] {r.title}"
            if r.content:
                entry += f"\n  {r.content[:200]}"
            content_parts.append(entry)
            if r.url:
                sources.append(r.url)

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Research: {task.query[:60]}",
            content="\n\n".join(content_parts) or "No results found.",
            confidence=0.7 if deduped else 0.2,
            sources=sources,
            scope=task.scope,
            metadata={
                "engines_used": list(self._engines.keys()),
                "total_raw": len(all_results),
                "after_dedup": len(deduped),
            },
        )

    async def _fan_out(self, queries: List[str]) -> List[EngineResult]:
        """Run all queries across all engines concurrently."""
        tasks = []
        for query in queries:
            for engine in self._engines.values():
                tasks.append(engine.query(query))

        if not tasks:
            return []

        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for item in gathered:
            if isinstance(item, BaseException):
                logger.debug(f"Engine query failed: {type(item).__name__}: {item}")
            elif isinstance(item, list):
                results.extend(item)
            else:
                logger.debug(f"Engine returned unexpected type: {type(item)}")

        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get stats for all engines."""
        return {
            name: engine.get_stats()
            for name, engine in self._engines.items()
        }

    def list_engines(self) -> List[str]:
        """List engine names."""
        return list(self._engines.keys())
