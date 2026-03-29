"""Tests for the LLM manager and scheduler."""

import asyncio
import time

from khonliang.llm.protocol import (
    GPUSlot,
    InferenceRequest,
    ModelState,
    QueueType,
)
from khonliang.llm.scheduler import ModelScheduler


def test_enqueue_and_depth():
    scheduler = ModelScheduler()
    req = InferenceRequest(model="llama3.2:3b", prompt="hello")
    asyncio.run(scheduler.enqueue(req))
    assert asyncio.run(scheduler.queue_depth("llama3.2:3b")) == 1
    assert asyncio.run(scheduler.queue_depth()) == 1


def test_priority_ordering():
    scheduler = ModelScheduler()
    asyncio.run(scheduler.enqueue(
        InferenceRequest(model="m", prompt="low", priority=0)
    ))
    asyncio.run(scheduler.enqueue(
        InferenceRequest(model="m", prompt="high", priority=10)
    ))

    result = asyncio.run(scheduler.next_batch())
    assert result is not None
    _, _, batch = result
    assert batch[0].prompt == "high"
    assert batch[1].prompt == "low"


def test_scoring_prefers_longer_wait():
    scheduler = ModelScheduler()

    old_req = InferenceRequest(
        model="model_a", prompt="old",
        submitted_at=time.time() - 30,  # 30s ago
    )
    new_req = InferenceRequest(
        model="model_b", prompt="new",
        submitted_at=time.time(),
    )
    asyncio.run(scheduler.enqueue(old_req))
    asyncio.run(scheduler.enqueue(new_req))

    gpu = scheduler.gpus[0]
    score_a = scheduler.score_model("model_a", gpu)
    score_b = scheduler.score_model("model_b", gpu)
    assert score_a > score_b  # older request scores higher


def test_scoring_loaded_model_no_swap_cost():
    scheduler = ModelScheduler()

    req_a = InferenceRequest(model="model_a", prompt="a")
    req_b = InferenceRequest(model="model_b", prompt="b")
    asyncio.run(scheduler.enqueue(req_a))
    asyncio.run(scheduler.enqueue(req_b))

    # Simulate model_a already loaded
    gpu = scheduler.gpus[0]
    scheduler.update_gpu_state(0, "model_a", ModelState.LOADED)

    score_a = scheduler.score_model("model_a", gpu)
    score_b = scheduler.score_model("model_b", gpu)
    # model_a has no swap cost, so should score higher (all else equal)
    assert score_a > score_b


def test_scoring_queue_depth_matters():
    scheduler = ModelScheduler()

    # model_a has 5 requests, model_b has 1
    for _ in range(5):
        asyncio.run(scheduler.enqueue(
            InferenceRequest(model="model_a", prompt="a")
        ))
    asyncio.run(scheduler.enqueue(
        InferenceRequest(model="model_b", prompt="b")
    ))

    gpu = scheduler.gpus[0]
    score_a = scheduler.score_model("model_a", gpu)
    score_b = scheduler.score_model("model_b", gpu)
    assert score_a > score_b


def test_batch_extraction():
    scheduler = ModelScheduler(max_batch_size=3)

    for i in range(5):
        asyncio.run(scheduler.enqueue(
            InferenceRequest(model="m", prompt=f"req{i}")
        ))

    result = asyncio.run(scheduler.next_batch())
    assert result is not None
    model, gpu, batch = result
    assert model == "m"
    assert len(batch) == 3  # max_batch_size
    assert asyncio.run(scheduler.queue_depth("m")) == 2  # 2 remaining


def test_empty_queue_returns_none():
    scheduler = ModelScheduler()
    result = asyncio.run(scheduler.next_batch())
    assert result is None


def test_multi_gpu_scheduling():
    scheduler = ModelScheduler(
        gpus=[
            GPUSlot(gpu_id=0, vram_mb=8192),
            GPUSlot(gpu_id=1, vram_mb=24576),
        ],
        model_vram={"small": 4000, "large": 20000},
    )

    asyncio.run(scheduler.enqueue(
        InferenceRequest(model="small", prompt="s")
    ))
    asyncio.run(scheduler.enqueue(
        InferenceRequest(model="large", prompt="l")
    ))

    batches = asyncio.run(scheduler.next_batch_for_all_gpus())
    assert len(batches) == 2

    models_assigned = {b[0] for b in batches}
    assert "small" in models_assigned
    assert "large" in models_assigned


def test_vram_constraint():
    scheduler = ModelScheduler(
        gpus=[GPUSlot(gpu_id=0, vram_mb=8192)],
        model_vram={"too_big": 20000},
    )

    asyncio.run(scheduler.enqueue(
        InferenceRequest(model="too_big", prompt="test")
    ))

    gpu = scheduler.gpus[0]
    score = scheduler.score_model("too_big", gpu)
    assert score == -1.0  # can't fit


def test_queue_type_multiplier():
    scheduler = ModelScheduler()

    asyncio.run(scheduler.enqueue(InferenceRequest(
        model="interactive_m", prompt="fast",
        queue=QueueType.INTERACTIVE,
    )))
    asyncio.run(scheduler.enqueue(InferenceRequest(
        model="batch_m", prompt="slow",
        queue=QueueType.BATCH,
    )))

    gpu = scheduler.gpus[0]
    score_int = scheduler.score_model("interactive_m", gpu)
    score_batch = scheduler.score_model("batch_m", gpu)
    assert score_int > score_batch


def test_stats_tracking():
    scheduler = ModelScheduler()
    scheduler.record_inference("model_a", 500.0)
    scheduler.record_inference("model_a", 600.0)
    scheduler.record_load("model_a", 3000.0)

    stats = scheduler.get_stats("model_a")
    assert stats.avg_inference_ms == 550.0
    assert stats.avg_load_ms == 3000.0
    assert stats.total_requests == 2


def test_cancel_request():
    scheduler = ModelScheduler()
    req = InferenceRequest(model="m", prompt="cancel me")
    asyncio.run(scheduler.enqueue(req))
    assert asyncio.run(scheduler.cancel(req.request_id))
    assert asyncio.run(scheduler.queue_depth()) == 0


def test_scheduler_status():
    scheduler = ModelScheduler(
        gpus=[GPUSlot(gpu_id=0, vram_mb=8192)]
    )
    asyncio.run(scheduler.enqueue(
        InferenceRequest(model="m", prompt="test")
    ))
    scheduler.record_inference("m", 200.0)

    status = scheduler.get_status()
    assert status["queue_depths"]["m"] == 1
    assert status["completed"] == 1
    assert "m" in status["model_stats"]
