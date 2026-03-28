"""
Research pool — managed thread pool for async research workers.

Researchers register themselves, tasks are queued, and the pool
dispatches to the right researcher based on task_type. Results
can optionally flow into the knowledge store via the librarian.

Example:

    pool = ResearchPool()
    pool.register(MyWebResearcher())
    pool.register(MyDatabaseResearcher())

    # Optional: auto-index results into knowledge
    pool.set_librarian(librarian)

    pool.start(workers=3)
    pool.submit(ResearchTask(task_type="person_lookup", query="Timothy Tolle"))

    # Check status
    print(pool.get_status())

    pool.stop()
"""

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from khonliang.research.base import BaseResearcher
from khonliang.research.models import ResearchResult, ResearchStatus, ResearchTask

logger = logging.getLogger(__name__)


class ResearchPool:
    """
    Managed pool of research workers with task queue.

    Dispatches research tasks to registered researchers based on
    task_type matching against researcher capabilities.
    """

    def __init__(self, max_queue_size: int = 100):
        self._researchers: Dict[str, BaseResearcher] = {}
        self._capability_map: Dict[str, str] = {}  # task_type -> researcher name
        self._queue: List[ResearchTask] = []
        self._results: Dict[str, ResearchResult] = {}
        self._active_tasks: Dict[str, ResearchTask] = {}
        self._max_queue = max_queue_size
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._librarian: Optional[Any] = None
        self._on_result: Optional[Callable[[ResearchResult], None]] = None
        self._lock = threading.Lock()

        # Per-researcher semaphores (created on start)
        self._researcher_sems: Dict[str, asyncio.Semaphore] = {}

        # Stats
        self._completed_count = 0
        self._failed_count = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, researcher: BaseResearcher) -> None:
        """
        Register a researcher with the pool.

        The researcher's capabilities are indexed so tasks can be
        dispatched to the right worker.
        """
        self._researchers[researcher.name] = researcher
        for cap in researcher.capabilities:
            self._capability_map[cap] = researcher.name
        logger.info(
            f"Registered researcher '{researcher.name}': "
            f"{researcher.capabilities}"
        )

    def unregister(self, name: str) -> bool:
        """Remove a researcher. Returns True if it existed."""
        researcher = self._researchers.pop(name, None)
        if researcher:
            for cap in researcher.capabilities:
                if self._capability_map.get(cap) == name:
                    del self._capability_map[cap]
            return True
        return False

    def set_librarian(self, librarian: Any) -> None:
        """Set a librarian for auto-indexing results into knowledge."""
        self._librarian = librarian

    def on_result(self, callback: Callable[[ResearchResult], None]) -> None:
        """Set a callback for when results come in."""
        self._on_result = callback

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit(self, task: ResearchTask) -> str:
        """
        Submit a research task to the queue.

        Returns the task_id. Raises ValueError if queue is full or
        no researcher can handle the task_type.
        """
        if not self._capability_map.get(task.task_type):
            available = list(self._capability_map.keys())
            raise ValueError(
                f"No researcher for task_type '{task.task_type}'. "
                f"Available: {available}"
            )

        with self._lock:
            if len(self._queue) >= self._max_queue:
                raise ValueError("Research queue is full")
            self._queue.append(task)
            # Sort by priority (highest first)
            self._queue.sort(key=lambda t: -t.priority)

        logger.debug(
            f"Queued research task {task.task_id}: "
            f"{task.task_type} '{task.query[:40]}'"
        )
        return task.task_id

    def submit_lookup(
        self,
        query: str,
        task_type: str = "web_search",
        scope: str = "global",
        source: str = "user",
        priority: int = 0,
        **metadata,
    ) -> str:
        """Convenience method to submit a simple lookup task."""
        task = ResearchTask(
            task_type=task_type,
            query=query,
            scope=scope,
            source=source,
            priority=priority,
            metadata=metadata,
        )
        return self.submit(task)

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task. Returns True if it was found and cancelled."""
        with self._lock:
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    task.status = ResearchStatus.CANCELLED
                    self._queue.pop(i)
                    return True
        return False

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_result(self, task_id: str) -> Optional[ResearchResult]:
        """Get a completed result by task_id."""
        return self._results.get(task_id)

    def get_all_results(self, limit: int = 50) -> List[ResearchResult]:
        """Get recent results, newest first."""
        results = sorted(
            self._results.values(),
            key=lambda r: r.completed_at,
            reverse=True,
        )
        return results[:limit]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, workers: int = 2) -> None:
        """Start the worker thread pool."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(workers,),
            daemon=True,
            name="research-pool",
        )
        self._thread.start()
        logger.info(f"Research pool started with {workers} workers")

    def stop(self) -> None:
        """Stop the pool and wait for workers to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Research pool stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run_loop(self, workers: int) -> None:
        """Main worker loop — runs in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._process_loop(workers))
        except Exception as e:
            logger.error(f"Research pool error: {e}")
        finally:
            self._loop.close()

    async def _process_loop(self, workers: int) -> None:
        """Async task processing loop — per-researcher concurrency."""
        # Global semaphore caps total concurrent tasks
        global_sem = asyncio.Semaphore(workers)

        # Create per-researcher semaphores
        for name, researcher in self._researchers.items():
            self._researcher_sems[name] = asyncio.Semaphore(
                researcher.max_concurrent
            )

        async def _worker(task):
            researcher_name = self._capability_map.get(task.task_type, "")
            researcher_sem = self._researcher_sems.get(researcher_name)

            async with global_sem:
                if researcher_sem:
                    async with researcher_sem:
                        await self._process_task(task)
                else:
                    await self._process_task(task)

        while self._running:
            task = self._next_task()
            if task is None:
                await asyncio.sleep(0.5)
                continue

            asyncio.create_task(_worker(task))

    def _next_task(self) -> Optional[ResearchTask]:
        """Pop the highest-priority task from the queue."""
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
        return None

    async def _process_task(self, task: ResearchTask) -> None:
        """Process a single research task."""
        researcher_name = self._capability_map.get(task.task_type)
        if not researcher_name:
            logger.warning(f"No researcher for {task.task_type}")
            return

        researcher = self._researchers.get(researcher_name)
        if not researcher:
            logger.warning(f"Researcher '{researcher_name}' not found")
            return

        task.status = ResearchStatus.RUNNING
        self._active_tasks[task.task_id] = task

        try:
            result = await researcher.research(task)
            task.status = ResearchStatus.COMPLETED
            self._results[task.task_id] = result
            self._completed_count += 1

            logger.info(
                f"Research completed: {task.task_id} "
                f"'{task.query[:40]}' -> {result.title[:40]}"
            )

            # Auto-index into knowledge
            if self._librarian:
                try:
                    self._librarian.index_response(
                        content=result.content,
                        title=result.title,
                        agent_id=f"researcher:{researcher.name}",
                        query=task.query,
                        scope=result.scope or task.scope,
                        confidence=result.confidence,
                    )
                except Exception as e:
                    logger.debug(f"Knowledge indexing failed: {e}")

            # Callback
            if self._on_result:
                try:
                    self._on_result(result)
                except Exception as e:
                    logger.debug(f"on_result callback error: {e}")

        except Exception as e:
            task.status = ResearchStatus.FAILED
            self._failed_count += 1
            logger.error(
                f"Research failed: {task.task_id} "
                f"'{task.query[:40]}': {e}"
            )
        finally:
            self._active_tasks.pop(task.task_id, None)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get pool status summary."""
        with self._lock:
            queue_len = len(self._queue)

        return {
            "running": self._running,
            "researchers": {
                name: r.capabilities
                for name, r in self._researchers.items()
            },
            "queue_size": queue_len,
            "active_tasks": len(self._active_tasks),
            "completed": self._completed_count,
            "failed": self._failed_count,
            "results_cached": len(self._results),
            "capability_map": dict(self._capability_map),
        }

    def list_researchers(self) -> List[Dict[str, Any]]:
        """List registered researchers."""
        return [
            {
                "name": r.name,
                "capabilities": r.capabilities,
            }
            for r in self._researchers.values()
        ]
