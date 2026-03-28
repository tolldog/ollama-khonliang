"""Tests for the research pool system."""

import time

from khonliang.research.base import BaseResearcher
from khonliang.research.models import ResearchResult, ResearchStatus, ResearchTask
from khonliang.research.pool import ResearchPool
from khonliang.research.trigger import ResearchTrigger


class MockResearcher(BaseResearcher):
    name = "mock"
    capabilities = ["web_search", "person_lookup"]

    def __init__(self):
        self.calls = []

    async def research(self, task: ResearchTask) -> ResearchResult:
        self.calls.append(task)
        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Result for: {task.query}",
            content=f"Found info about {task.query}",
            confidence=0.8,
            sources=["https://example.com"],
            scope=task.scope,
        )


def test_task_creation():
    task = ResearchTask(task_type="web_search", query="Timothy Tolle")
    assert task.status == ResearchStatus.PENDING
    assert task.task_id
    assert task.task_type == "web_search"


def test_researcher_registration():
    pool = ResearchPool()
    researcher = MockResearcher()
    pool.register(researcher)

    status = pool.get_status()
    assert "mock" in status["researchers"]
    assert "web_search" in status["capability_map"]
    assert "person_lookup" in status["capability_map"]


def test_submit_task():
    pool = ResearchPool()
    pool.register(MockResearcher())

    task_id = pool.submit(ResearchTask(
        task_type="web_search",
        query="test query",
    ))
    assert task_id
    assert pool.get_status()["queue_size"] == 1


def test_submit_unknown_type():
    pool = ResearchPool()
    pool.register(MockResearcher())

    try:
        pool.submit(ResearchTask(task_type="unknown", query="test"))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_pool_processes_tasks():
    pool = ResearchPool()
    researcher = MockResearcher()
    pool.register(researcher)

    pool.submit(ResearchTask(task_type="web_search", query="test"))
    pool.start(workers=1)
    time.sleep(2)
    pool.stop()

    assert pool.get_status()["completed"] == 1
    assert len(researcher.calls) == 1


def test_priority_ordering():
    pool = ResearchPool()
    pool.register(MockResearcher())

    pool.submit(ResearchTask(task_type="web_search", query="low", priority=0))
    pool.submit(ResearchTask(task_type="web_search", query="high", priority=10))
    pool.submit(ResearchTask(task_type="web_search", query="med", priority=5))

    # Queue should be ordered by priority descending
    with pool._lock:
        queries = [t.query for t in pool._queue]
    assert queries == ["high", "med", "low"]


def test_trigger_prefix():
    pool = ResearchPool()
    pool.register(MockResearcher())

    trigger = ResearchTrigger(pool)
    task_ids = trigger.check_message("!lookup: Timothy Tolle", scope="toll")

    assert len(task_ids) == 1
    assert pool.get_status()["queue_size"] == 1


def test_trigger_search():
    pool = ResearchPool()
    pool.register(MockResearcher())

    trigger = ResearchTrigger(pool)
    task_ids = trigger.check_message("!search: Maryland 1700", scope="toll")

    assert len(task_ids) == 1


def test_trigger_no_match():
    pool = ResearchPool()
    pool.register(MockResearcher())

    trigger = ResearchTrigger(pool)
    task_ids = trigger.check_message(
        "Who were Timothy's parents?", scope="toll"
    )

    assert len(task_ids) == 0


def test_trigger_implicit():
    pool = ResearchPool()
    pool.register(MockResearcher())

    trigger = ResearchTrigger(pool)
    task_ids = trigger.check_response(
        response="I don't have enough information about this person.",
        original_query="Tell me about John Tolle",
        scope="toll",
    )

    assert len(task_ids) == 1


def test_trigger_strip_prefix():
    pool = ResearchPool()
    trigger = ResearchTrigger(pool)

    assert trigger.strip_prefix("!lookup: Tim Toll") == "Tim Toll"
    assert trigger.strip_prefix("!search: Maryland") == "Maryland"
    assert trigger.strip_prefix("regular question") is None


def test_cancel_task():
    pool = ResearchPool()
    pool.register(MockResearcher())

    task_id = pool.submit(ResearchTask(
        task_type="web_search", query="cancel me"
    ))
    assert pool.cancel(task_id)
    assert pool.get_status()["queue_size"] == 0
