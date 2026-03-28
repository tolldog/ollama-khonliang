"""Smoke tests for core modules that have no external dependencies."""

from khonliang.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from khonliang.health import ModelHealthTracker


def test_error_hierarchy():
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(LLMUnavailableError, LLMError)


def test_health_tracker_no_cooldown_initially():
    tracker = ModelHealthTracker()
    assert not tracker.is_cooled_down("llama3.1:8b")


def test_health_tracker_enters_cooldown():
    tracker = ModelHealthTracker(failure_threshold=2, cooldown_duration=60.0)
    tracker.record_failure("test-model")
    tracker.record_failure("test-model")
    assert tracker.is_cooled_down("test-model")


def test_health_tracker_success_clears_failures():
    tracker = ModelHealthTracker(failure_threshold=3)
    tracker.record_failure("test-model")
    tracker.record_failure("test-model")
    tracker.record_success("test-model")
    tracker.record_failure("test-model")
    assert not tracker.is_cooled_down("test-model")


def test_health_tracker_status():
    tracker = ModelHealthTracker()
    status = tracker.get_status("unknown-model")
    assert status["model"] == "unknown-model"
    assert status["total_failures"] == 0
    assert status["cooled_down"] is False
