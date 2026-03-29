"""
Model scheduler — score-based scheduling with batching.

Decides which model should run next based on:
- Queue depth per model
- Wait time (starvation prevention)
- Priority of queued requests
- Swap cost (model loading time)
- Current GPU state

Inspired by LSF (Load Sharing Facility) job scheduling.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from khonliang.llm.protocol import (
    GPUSlot,
    InferenceRequest,
    ModelState,
    ModelStats,
    QueueType,
)

logger = logging.getLogger(__name__)


class ModelScheduler:
    """
    Score-based model scheduler with GPU-aware batching.

    Maintains per-model request queues and calculates scheduling
    scores to decide which model should run next on which GPU.

    Score formula:
        score = (avg_wait * queue_depth * max_priority) - swap_cost

    Where swap_cost is 0 if the model is already loaded.

    Example:
        scheduler = ModelScheduler(
            gpus=[GPUSlot(0, vram_mb=8192)],
        )
        scheduler.enqueue(request)
        model, gpu, batch = scheduler.next_batch()
    """

    # Queue type priority multipliers
    QUEUE_MULTIPLIERS = {
        QueueType.SYSTEM: 100.0,
        QueueType.INTERACTIVE: 10.0,
        QueueType.DEFAULT: 1.0,
        QueueType.BATCH: 0.5,
    }

    def __init__(
        self,
        gpus: Optional[List[GPUSlot]] = None,
        max_batch_size: int = 10,
        model_vram: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            gpus: GPU slots available. Default: single unknown GPU.
            max_batch_size: Max requests per batch.
            model_vram: Optional map of model -> estimated VRAM in MB.
        """
        self.gpus = gpus or [GPUSlot(gpu_id=0)]
        self.max_batch_size = max_batch_size
        self.model_vram = model_vram or {}

        # Per-model request queues
        self._queues: Dict[str, List[InferenceRequest]] = defaultdict(list)
        self._queue_lock = asyncio.Lock()

        # Model stats (tracked from actual runs)
        self._stats: Dict[str, ModelStats] = {}

        # Completed count
        self._completed = 0

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    async def enqueue(self, request: InferenceRequest) -> None:
        """Add a request to the appropriate model queue."""
        async with self._queue_lock:
            self._queues[request.model].append(request)
            # Sort by priority descending, then by submit time ascending
            self._queues[request.model].sort(
                key=lambda r: (-r.priority, r.submitted_at)
            )

    async def cancel(self, request_id: str) -> bool:
        """Cancel a pending request. Returns True if found."""
        async with self._queue_lock:
            for model, model_queue in list(self._queues.items()):
                for i, req in enumerate(model_queue):
                    if req.request_id == request_id:
                        model_queue.pop(i)
                        if not model_queue:
                            del self._queues[model]
                        return True
        return False

    async def queue_depth(self, model: Optional[str] = None) -> int:
        """Get total queue depth, or for a specific model."""
        async with self._queue_lock:
            if model:
                return len(self._queues.get(model, []))
            return sum(len(q) for q in self._queues.values())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_model(self, model: str, gpu: GPUSlot) -> float:
        """
        Calculate scheduling score for a model on a GPU.

        Higher score = should run next.
        """
        queue = self._queues.get(model, [])
        if not queue:
            return 0.0

        now = time.time()

        # Queue depth
        depth = len(queue)

        # Average wait time (starvation prevention)
        wait_times = [now - r.submitted_at for r in queue]
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0

        # Max priority in queue (with queue type multiplier)
        max_priority = max(
            (r.priority + 1) * self.QUEUE_MULTIPLIERS.get(r.queue, 1.0)
            for r in queue
        )

        # Swap cost multiplier (1.0 if loaded, reduced if swap needed)
        if gpu.current_model == model and gpu.model_state == ModelState.LOADED:
            swap_factor = 1.0  # no swap needed
        else:
            stats = self._stats.get(model)
            swap_ms = stats.avg_load_ms if stats else 5000.0
            # Swap factor: 0.1 for very expensive swaps, up to 1.0
            swap_factor = max(0.1, 1.0 - (swap_ms / 30000.0))

        # VRAM check — can this GPU fit the model?
        model_vram = self.model_vram.get(model, 0)
        if model_vram > 0 and not gpu.can_fit(model_vram):
            return -1.0  # can't fit

        # Score formula: always positive when there's work
        # wait_time ensures starvation prevention (min 0.1 for fresh requests)
        score = max(0.1, avg_wait) * depth * max_priority * swap_factor

        return score

    def best_model_for_gpu(
        self, gpu: GPUSlot, exclude: Optional[set] = None
    ) -> Optional[str]:
        """Find the highest-scoring model for a specific GPU.

        Args:
            gpu: The GPU slot to score against.
            exclude: Optional set of model names to skip (already assigned).
        """
        best_model = None
        best_score = float("-inf")
        exclude = exclude or set()

        for model in self._queues:
            if model in exclude:
                continue
            if not self._queues[model]:
                continue
            score = self.score_model(model, gpu)
            if score == -1.0:
                continue  # VRAM can't fit
            if score > best_score:
                best_score = score
                best_model = model

        return best_model

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    async def next_batch(
        self, gpu: Optional[GPUSlot] = None
    ) -> Optional[Tuple[str, GPUSlot, List[InferenceRequest]]]:
        """
        Get the next batch to run.

        Returns (model, gpu_slot, batch_of_requests) or None if empty.
        """
        target_gpu = gpu or self.gpus[0]

        async with self._queue_lock:
            model = self.best_model_for_gpu(target_gpu)

            if model is None:
                return None

            queue = self._queues.get(model, [])
            if not queue:
                return None

            # Extract batch (up to max_batch_size)
            batch_size = min(self.max_batch_size, len(queue))
            batch = queue[:batch_size]
            self._queues[model] = queue[batch_size:]

            # Clean up empty queues
            if not self._queues[model]:
                del self._queues[model]

        return model, target_gpu, batch

    async def next_batch_for_all_gpus(
        self,
    ) -> List[Tuple[str, GPUSlot, List[InferenceRequest]]]:
        """Get next batch for each GPU that has work."""
        batches = []
        assigned_models: set = set()

        for gpu in self.gpus:
            # Prefer keeping current model if it has work
            if (
                gpu.current_model
                and gpu.current_model not in assigned_models
                and self._queues.get(gpu.current_model)
            ):
                batch = await self.next_batch(gpu)
                if batch:
                    batches.append(batch)
                    assigned_models.add(batch[0])
                    continue

            # Find best unassigned model for this GPU
            model = self.best_model_for_gpu(gpu, exclude=assigned_models)
            if model:
                batch = await self.next_batch(gpu)
                if batch:
                    batches.append(batch)
                    assigned_models.add(batch[0])

        return batches

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------

    def get_stats(self, model: str) -> ModelStats:
        """Get or create stats for a model."""
        if model not in self._stats:
            self._stats[model] = ModelStats(
                model=model,
                vram_mb=self.model_vram.get(model, 0),
            )
        return self._stats[model]

    def record_inference(self, model: str, duration_ms: float) -> None:
        """Record inference completion time."""
        self.get_stats(model).record_inference(duration_ms)
        self._completed += 1

    def record_load(self, model: str, duration_ms: float) -> None:
        """Record model load time."""
        self.get_stats(model).record_load(duration_ms)

    def record_error(self, model: str) -> None:
        """Record an inference error."""
        self.get_stats(model).record_error()

    def update_gpu_state(
        self,
        gpu_id: int,
        model: Optional[str],
        state: ModelState,
    ) -> None:
        """Update GPU model state."""
        for gpu in self.gpus:
            if gpu.gpu_id == gpu_id:
                gpu.current_model = model
                gpu.model_state = state
                if state == ModelState.LOADED:
                    gpu.loaded_at = time.time()
                break

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Get scheduler status snapshot."""
        return {
            "gpus": [
                {
                    "gpu_id": g.gpu_id,
                    "vram_mb": g.vram_mb,
                    "current_model": g.current_model,
                    "model_state": g.model_state.value,
                }
                for g in self.gpus
            ],
            "queue_depths": {
                model: len(queue) for model, queue in self._queues.items()
            },
            "completed": self._completed,
            "model_stats": {
                model: {
                    "avg_inference_ms": s.avg_inference_ms,
                    "avg_load_ms": s.avg_load_ms,
                    "total_requests": s.total_requests,
                    "total_errors": s.total_errors,
                }
                for model, s in self._stats.items()
            },
        }
