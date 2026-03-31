# LLM Manager

The LLM Manager provides score-based inference scheduling for multi-model workloads. On a single GPU host, loading a 19GB model while a 7B is running causes VRAM contention. The scheduler manages this by queuing requests, batching same-model work, and making score-based decisions about when to swap models.

This is **optional** — projects that don't need it continue using `OllamaClient` or `OpenAIClient` directly with zero overhead.

## Architecture

```text
LLMManager (facade)
  └── InternalBackend (asyncio queues + any LLMClient)
        └── ModelScheduler (score-based queue logic)
              ├── Per-model request queues
              ├── Per-GPU state tracking
              └── Model statistics (inference time, load time, errors)
```

Two backend modes, same Python interface:

| Mode         | Backend                    | Use Case                             |
| ------------ | -------------------------- | ------------------------------------ |
| `"internal"` | In-process asyncio queues  | Single host, Python apps             |
| `"grpc"`     | External Go/Rust scheduler | Multi-host, high throughput (future) |

## Basic Usage

```python
from khonliang.llm import LLMManager
from khonliang.llm.protocol import GPUSlot, InferenceRequest, QueueType

manager = LLMManager(
    backend="internal",
    ollama_url="http://localhost:11434",
    gpus=[GPUSlot(gpu_id=0, vram_mb=8192)],
    model_vram={"llama3.2:3b": 2048, "qwen2.5:7b": 4500, "llama3.1:8b": 5000},
)

await manager.start()

# Simple generation (blocks until complete)
response = await manager.generate(
    model="llama3.2:3b",
    prompt="Who were Timothy's grandparents?",
    system="You are a genealogy assistant.",
)

# Async submission (non-blocking)
request_id = await manager.submit(InferenceRequest(
    model="qwen2.5:7b",
    prompt="Check these dates for errors...",
    priority=5,
    queue=QueueType.INTERACTIVE,
))
result = await manager.get_result(request_id, timeout=60.0)

# Status
status = await manager.status()
print(f"Queued: {status.queued_requests}, Active: {status.active_requests}")

await manager.stop()
```

## Queue Types

| Queue         | Multiplier | Use Case                         |
| ------------- | ---------- | -------------------------------- |
| `SYSTEM`      | 100x       | Internal scheduler operations    |
| `INTERACTIVE` | 10x        | User-facing chat (low latency)   |
| `DEFAULT`     | 1x         | Normal operations                |
| `BATCH`       | 0.5x       | Background processing (can wait) |

```python
from khonliang.llm.protocol import QueueType

# Interactive chat — prioritized
await manager.generate(
    model="llama3.2:3b",
    prompt=user_question,
    queue=QueueType.INTERACTIVE,
)

# Background research — can wait
await manager.generate(
    model="qwen2.5:7b",
    prompt=research_query,
    queue=QueueType.BATCH,
)
```

## Scoring Algorithm

The scheduler decides which model to run next using a score:

```text
score(model) = avg_wait_time * queue_depth * max_priority * swap_factor

where:
  avg_wait_time  = average(now - submitted_at) for queued requests
  queue_depth    = number of pending requests for this model
  max_priority   = highest priority in the queue (adjusted by queue type multiplier)
  swap_factor    = 1.0 if model is already loaded, penalty if swap needed
```

The swap factor heavily penalizes evicting pinned models (0.01x) and moderately penalizes cold loads.

### Why This Works

Benchmarking showed that model loading is **9-32x more expensive** than inference. The scheduler is really a **model cache manager** — it batches same-model requests to minimize swaps:

```text
Without scheduler:  load A → infer → load B → infer → load A → infer
With scheduler:     load A → infer → infer → load B → infer → infer
```

## Multi-GPU Support

```python
manager = LLMManager(
    backend="internal",
    ollama_url="http://localhost:11434",
    gpus=[
        GPUSlot(gpu_id=0, vram_mb=8192),    # Small GPU
        GPUSlot(gpu_id=1, vram_mb=24576),   # Large GPU
    ],
    model_vram={
        "llama3.2:3b": 2048,
        "qwen2.5:7b": 4500,
        "deepseek-coder-v2:16b": 10000,
        "llama3.1:70b": 19000,
    },
)
```

Each GPU is an independent scheduling lane. Large models go to GPUs with enough VRAM. Small models can share a GPU if VRAM allows.

## Model Preferences

Requests can specify model preferences — alternative models to use if the preferred one isn't loaded:

```python
response = await manager.generate(
    model="qwen2.5:7b",
    prompt="Validate these dates...",
    model_preferences=["llama3.1:8b", "llama3.2:3b"],
)
# If qwen2.5:7b is loaded → use it
# If llama3.1:8b is loaded instead → use that (no swap needed)
# If neither is loaded → load qwen2.5:7b (primary choice)
```

## Pinned Models

Pin frequently-used models to prevent eviction. Pinned models are configured via model profiles:

```python
manager = LLMManager(
    backend="internal",
    ollama_url="http://localhost:11434",
    profiles_path="data/model_profiles.json",  # Profiles define which models are pinned
)
```

Pinned models are pre-loaded at startup. The scheduler heavily penalizes evicting them (0.01x swap factor). Configure pinning in your `model_profiles.json` by marking models with `"pinned": true`.

## Model Profiles

Seed the scheduler with known model performance data:

```python
manager = LLMManager(
    backend="internal",
    ollama_url="http://localhost:11434",
    profiles_path="data/model_profiles.json",
)
```

Profiles store avg inference time, avg load time, and VRAM requirements. The scheduler loads these at startup and updates them with live measurements. Profiles persist across restarts.

## Benchmarking

Validate model performance:

```python
from khonliang.llm.benchmark import ModelBenchmark

bench = ModelBenchmark(
    ollama_url="http://localhost:11434",
    models=["llama3.2:3b", "qwen2.5:7b"],
)
results = await bench.run(prompt="Hello", iterations=5)
for model, stats in results.items():
    print(f"{model}: {stats['avg_ms']:.0f}ms avg, {stats['vram_mb']}MB VRAM")
```

## When to Use the Scheduler

The scheduler adds value when:

- You run **multiple different models** that compete for VRAM
- Model load times are significant (large models, limited VRAM)
- You have **mixed workloads** (interactive chat + background research)

The scheduler adds overhead (queue management, scoring) for:

- Single-model workloads (Ollama already serializes same-model requests)
- Small models that fit in VRAM simultaneously

From our benchmarking: passthrough (no scheduler) wins for small models on a single GPU. The scheduler wins **3.4x** for workloads involving 32B+ model swaps.
