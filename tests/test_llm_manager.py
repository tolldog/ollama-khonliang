"""Tests for the LLM manager, scheduler, and profiles."""

import asyncio
import os
import tempfile
import time

from khonliang.llm.profiles import ModelProfile, ModelProfiles
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


def test_multi_gpu_no_duplicate_model_assignment():
    """Each model should be assigned to at most one GPU per scheduling round."""
    scheduler = ModelScheduler(
        gpus=[
            GPUSlot(gpu_id=0, vram_mb=24576),
            GPUSlot(gpu_id=1, vram_mb=24576),
        ],
    )

    # Enqueue several requests for model_a (should score highest on both GPUs)
    # and one for model_b, using a single event loop run for all enqueues.
    async def _setup():
        for _ in range(4):
            await scheduler.enqueue(
                InferenceRequest(model="model_a", prompt="a", submitted_at=time.time() - 60)
            )
        await scheduler.enqueue(
            InferenceRequest(model="model_b", prompt="b")
        )

    asyncio.run(_setup())

    batches = asyncio.run(scheduler.next_batch_for_all_gpus())
    # Both GPUs get work; model_a is only assigned to one GPU
    assigned_models = [b[0] for b in batches]
    assert assigned_models.count("model_a") <= 1
    assert len(batches) == 2  # both GPUs busy: one on model_a, one on model_b


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


# ------------------------------------------------------------------
# Profile tests
# ------------------------------------------------------------------


def test_profile_save_load():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)

    profiles = ModelProfiles(path)
    profiles.set(ModelProfile(
        model="llama3.2:3b",
        vram_mb=2048,
        avg_load_ms=1500,
        avg_inference_ms=500,
    ))
    profiles.set(ModelProfile(
        model="qwen2.5:7b",
        vram_mb=4700,
        avg_load_ms=3000,
        avg_inference_ms=1100,
    ))
    profiles.save()

    loaded = ModelProfiles(path)
    loaded.load()
    assert len(loaded.list_models()) == 2

    p = loaded.get("llama3.2:3b")
    assert p is not None
    assert p.vram_mb == 2048
    assert p.avg_load_ms == 1500

    os.unlink(path)


def test_profile_vram_map():
    profiles = ModelProfiles("/dev/null")
    profiles.set(ModelProfile(model="small", vram_mb=2000))
    profiles.set(ModelProfile(model="large", vram_mb=19000))
    profiles.set(ModelProfile(model="unknown", vram_mb=0))

    vmap = profiles.get_vram_map()
    assert vmap["small"] == 2000
    assert vmap["large"] == 19000
    assert "unknown" not in vmap  # 0 excluded


def test_profile_runtime_preferred():
    p = ModelProfile(
        model="test",
        avg_inference_ms=500,
        runtime_avg_inference_ms=300,
    )
    assert p.effective_inference_ms() == 300  # runtime preferred


def test_profile_update_from_stats():
    profiles = ModelProfiles("/dev/null")
    profiles.set(ModelProfile(model="m", avg_inference_ms=500))

    profiles.update_from_stats({
        "m": {"avg_inference_ms": 400, "avg_load_ms": 2000, "total_requests": 10},
    })

    p = profiles.get("m")
    assert p.runtime_avg_inference_ms == 400
    assert p.runtime_avg_load_ms == 2000


def test_profile_seed_scheduler():
    profiles = ModelProfiles("/dev/null")
    profiles.set(ModelProfile(
        model="m", avg_inference_ms=500, avg_load_ms=2000
    ))

    scheduler = ModelScheduler()
    profiles.seed_scheduler_stats(scheduler)

    stats = scheduler.get_stats("m")
    assert stats.avg_inference_ms == 500
    assert stats.avg_load_ms == 2000


def test_gpu_vram_budget():
    gpu = GPUSlot(gpu_id=0, vram_mb=8192, vram_reserve_mb=512, max_vram_pct=0.9)
    # 8192 * 0.9 = 7372, minus 512 reserve = 6860
    assert gpu.available_vram_mb == 6860
    assert gpu.can_fit(6000)
    assert not gpu.can_fit(7000)


def test_gpu_vram_unknown():
    gpu = GPUSlot(gpu_id=0, vram_mb=0)
    assert gpu.available_vram_mb == 0
    assert gpu.can_fit(99999)  # unknown = assume fits


# ------------------------------------------------------------------
# Model preferences and pinning tests
# ------------------------------------------------------------------


def test_model_preferences_uses_loaded():
    """If a preferred model is loaded, route there instead of primary."""
    scheduler = ModelScheduler()
    scheduler.update_gpu_state(0, "qwen2.5:7b", ModelState.LOADED)

    req = InferenceRequest(
        model="llama3.1:8b",
        prompt="test",
        model_preferences=["qwen2.5:7b", "mistral:latest"],
    )
    target = scheduler._resolve_model(req)
    assert target == "qwen2.5:7b"  # loaded, so use it


def test_model_preferences_falls_back_to_primary():
    """If no preferred model is loaded, use primary."""
    scheduler = ModelScheduler()
    # Nothing loaded

    req = InferenceRequest(
        model="llama3.1:8b",
        prompt="test",
        model_preferences=["qwen2.5:7b"],
    )
    target = scheduler._resolve_model(req)
    assert target == "llama3.1:8b"


def test_model_preferences_primary_is_loaded():
    """If primary model is loaded, use it even with preferences."""
    scheduler = ModelScheduler()
    scheduler.update_gpu_state(0, "llama3.1:8b", ModelState.LOADED)

    req = InferenceRequest(
        model="llama3.1:8b",
        prompt="test",
        model_preferences=["qwen2.5:7b"],
    )
    target = scheduler._resolve_model(req)
    assert target == "llama3.1:8b"


def test_model_preferences_no_prefs():
    """Without preferences, always use primary."""
    scheduler = ModelScheduler()
    scheduler.update_gpu_state(0, "qwen2.5:7b", ModelState.LOADED)

    req = InferenceRequest(model="llama3.1:8b", prompt="test")
    target = scheduler._resolve_model(req)
    assert target == "llama3.1:8b"


def test_pinned_models_in_status():
    scheduler = ModelScheduler(pinned_models=["llama3.2:3b"])
    status = scheduler.get_status()
    assert "llama3.2:3b" in status["pinned_models"]


def test_pinned_profile():
    profiles = ModelProfiles("/dev/null")
    profiles.set(ModelProfile(model="llama3.2:3b", pin=True, vram_mb=1920))
    profiles.set(ModelProfile(model="qwen2.5:7b", pin=False, vram_mb=4700))

    pinned = profiles.get_pinned_models()
    assert pinned == ["llama3.2:3b"]
