"""
LLM manager protocol — request/response models and backend interface.

These models are designed to map 1:1 to protobuf messages for a future
gRPC-based external scheduler. Keep them flat and serializable.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class QueueType(str, Enum):
    """Queue priority classes."""

    INTERACTIVE = "interactive"  # low latency, small models, preempts batch
    DEFAULT = "default"  # normal priority
    BATCH = "batch"  # throughput-oriented, can wait
    SYSTEM = "system"  # highest priority, internal operations


class ModelState(str, Enum):
    """State of a model on a GPU."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


@dataclass
class InferenceRequest:
    """
    A request for LLM inference.

    Model selection:
        model: Primary model to use.
        model_preferences: Optional list of acceptable models, in preference
            order. If any of these is already loaded, the scheduler will use
            it instead of loading the primary model — avoiding swap latency.
            The primary `model` is always implicitly first in the list.
    """

    model: str
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    priority: int = 0
    queue: QueueType = QueueType.DEFAULT
    timeout: float = 120.0
    extra_options: Optional[Dict[str, Any]] = None
    model_preferences: List[str] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    submitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "prompt": self.prompt[:100],
            "priority": self.priority,
            "queue": self.queue.value,
            "timeout": self.timeout,
            "submitted_at": self.submitted_at,
        }


@dataclass
class InferenceResult:
    """Result from an LLM inference request."""

    request_id: str
    text: str
    model: str
    duration_ms: int = 0
    queue_wait_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class GPUSlot:
    """Represents a GPU and its current state."""

    gpu_id: int = 0
    vram_mb: int = 0  # 0 = unknown/unlimited, total physical VRAM
    vram_reserve_mb: int = 512  # VRAM to reserve for system/display
    max_vram_pct: float = 0.9  # max % of VRAM to use (0.0-1.0)
    current_model: Optional[str] = None
    model_state: ModelState = ModelState.UNLOADED
    loaded_at: float = 0.0

    @property
    def available_vram_mb(self) -> int:
        """VRAM available for models after reserves."""
        if self.vram_mb == 0:
            return 0  # unknown
        budget = int(self.vram_mb * self.max_vram_pct)
        return max(0, budget - self.vram_reserve_mb)

    def can_fit(self, model_vram_mb: int) -> bool:
        """Check if this GPU can fit a model within budget."""
        if self.vram_mb == 0:
            return True  # unknown = assume it fits
        return model_vram_mb <= self.available_vram_mb


@dataclass
class ModelStats:
    """Runtime statistics for a model."""

    model: str
    avg_inference_ms: float = 0.0
    avg_load_ms: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    vram_mb: int = 0  # estimated VRAM usage
    _inference_times: List[float] = field(default_factory=list)
    _load_times: List[float] = field(default_factory=list)

    def record_inference(self, duration_ms: float) -> None:
        self._inference_times.append(duration_ms)
        # Keep last 100 for rolling average
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-100:]
        self.avg_inference_ms = (
            sum(self._inference_times) / len(self._inference_times)
        )
        self.total_requests += 1

    def record_load(self, duration_ms: float) -> None:
        self._load_times.append(duration_ms)
        if len(self._load_times) > 20:
            self._load_times = self._load_times[-20:]
        self.avg_load_ms = sum(self._load_times) / len(self._load_times)

    def record_error(self) -> None:
        self.total_errors += 1
        self.total_requests += 1


@dataclass
class SchedulerStatus:
    """Snapshot of scheduler state."""

    gpus: List[Dict[str, Any]]
    queued_requests: int
    active_requests: int
    completed_requests: int
    models: Dict[str, Dict[str, Any]]  # model -> stats
    queue_depths: Dict[str, int]  # model -> pending count


@runtime_checkable
class LLMBackend(Protocol):
    """
    Protocol for LLM inference backends.

    Implementations:
    - InternalBackend: in-process asyncio queues + OllamaClient
    - (future) GrpcBackend: talks to external Go/Rust scheduler
    """

    async def submit(self, request: InferenceRequest) -> str:
        """Submit a request. Returns request_id."""
        ...

    async def get_result(
        self, request_id: str, timeout: float = 30.0
    ) -> InferenceResult:
        """Wait for and return the result of a submitted request."""
        ...

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        priority: int = 0,
        queue: QueueType = QueueType.DEFAULT,
        **kwargs: Any,
    ) -> str:
        """Convenience: submit + wait + return text."""
        ...

    async def status(self) -> SchedulerStatus:
        """Get scheduler status snapshot."""
        ...

    async def start(self) -> None:
        """Start the backend."""
        ...

    async def stop(self) -> None:
        """Stop the backend."""
        ...
