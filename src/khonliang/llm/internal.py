"""
Internal LLM backend — in-process scheduler wrapping an LLM client.

For single-host setups where Ollama (or any OpenAI-compatible server)
runs locally. Uses asyncio queues and the ModelScheduler for
score-based request ordering.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from khonliang.client import OllamaClient
from khonliang.llm.protocol import (
    GPUSlot,
    InferenceRequest,
    InferenceResult,
    ModelState,
    QueueType,
    SchedulerStatus,
)
from khonliang.llm.scheduler import ModelScheduler

logger = logging.getLogger(__name__)


class InternalBackend:
    """
    In-process LLM backend with score-based scheduling.

    Wraps OllamaClient with the ModelScheduler to queue, batch,
    and order inference requests. Runs a background scheduling
    loop that picks the best model batch to run next.

    Example:
        backend = InternalBackend(ollama_url="http://localhost:11434")
        await backend.start()
        text = await backend.generate("qwen2.5:7b", "Hello!")
        await backend.stop()
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        gpus: Optional[list] = None,
        max_batch_size: int = 10,
        model_vram: Optional[Dict[str, int]] = None,
        pinned_models: Optional[list] = None,
        client: Optional[Any] = None,
    ):
        self._ollama_url = ollama_url
        self._client = client or OllamaClient(base_url=ollama_url)
        self._scheduler = ModelScheduler(
            gpus=gpus,
            max_batch_size=max_batch_size,
            model_vram=model_vram or {},
            pinned_models=pinned_models,
        )

        # Result futures: request_id -> Future
        self._futures: Dict[str, asyncio.Future] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

        # Stats
        self._completed = 0

    async def start(self) -> None:
        """Start the scheduling loop and pre-load pinned models."""
        self._running = True

        # Pre-load pinned models so they are warm on startup
        for model in sorted(self._scheduler.pinned_models):
            gpu = self._find_gpu_for_pinned(model)
            if gpu is not None:
                logger.info(f"Pre-loading pinned model '{model}' on GPU {gpu.gpu_id}")
                await self._load_model(model, gpu)

        self._loop_task = asyncio.create_task(self._scheduling_loop())
        logger.info("Internal LLM backend started")

    def _find_gpu_for_pinned(self, model: str) -> Optional[GPUSlot]:
        """Find a suitable GPU for a pinned model, preferring unloaded GPUs."""
        # Prefer a GPU that has no model loaded yet
        for gpu in self._scheduler.gpus:
            if gpu.current_model is None:
                return gpu
        # Fall back to the first GPU that can fit the model
        for gpu in self._scheduler.gpus:
            model_vram = self._scheduler.model_vram.get(model, 0)
            if model_vram == 0 or gpu.can_fit(model_vram):
                return gpu
        return None

    async def stop(self) -> None:
        """Stop the scheduling loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        # Resolve all pending futures with an error before closing
        for request_id, future in list(self._futures.items()):
            if not future.done():
                future.set_result(
                    InferenceResult(
                        request_id=request_id,
                        text="",
                        model="",
                        error="Backend stopped",
                    )
                )
        self._futures.clear()
        await self._client.close()
        logger.info("Internal LLM backend stopped")

    async def submit(self, request: InferenceRequest) -> str:
        """Submit a request. Returns request_id."""
        future = asyncio.get_running_loop().create_future()
        self._futures[request.request_id] = future
        await self._scheduler.enqueue(request)
        return request.request_id

    async def get_result(
        self, request_id: str, timeout: float = 30.0
    ) -> InferenceResult:
        """Wait for and return the result."""
        future = self._futures.get(request_id)
        if future is None:
            return InferenceResult(
                request_id=request_id,
                text="",
                model="",
                error="Unknown request_id",
            )
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            future.cancel()
            self._futures.pop(request_id, None)
            return InferenceResult(
                request_id=request_id,
                text="",
                model="",
                error="Timeout waiting for result",
            )

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
        request = InferenceRequest(
            model=model,
            prompt=prompt,
            system=system,
            priority=priority,
            queue=queue,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4000),
            timeout=kwargs.get("timeout", 120.0),
            extra_options=kwargs.get("extra_options"),
            model_preferences=kwargs.get("model_preferences", []),
        )
        await self.submit(request)
        result = await self.get_result(request.request_id, timeout=request.timeout)
        if result.error:
            raise RuntimeError(f"LLM inference failed: {result.error}")
        return result.text

    async def status(self) -> SchedulerStatus:
        """Get scheduler status."""
        sched_status = self._scheduler.get_status()
        return SchedulerStatus(
            gpus=sched_status["gpus"],
            queued_requests=sum(sched_status["queue_depths"].values()),
            active_requests=len(
                [f for f in self._futures.values() if not f.done()]
            ),
            completed_requests=self._completed,
            models=sched_status.get("model_stats", {}),
            queue_depths=sched_status["queue_depths"],
        )

    # ------------------------------------------------------------------
    # Scheduling loop
    # ------------------------------------------------------------------

    async def _scheduling_loop(self) -> None:
        """
        Main scheduling loop — picks best batch and runs it.

        Runs continuously while the backend is active.
        """
        while self._running:
            try:
                batches = await self._scheduler.next_batch_for_all_gpus()

                if not batches:
                    await asyncio.sleep(0.1)
                    continue

                # Process all GPU batches concurrently
                tasks = [
                    self._process_batch(model, gpu, batch)
                    for model, gpu, batch in batches
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduling loop error: {e}")
                await asyncio.sleep(1.0)

    async def _process_batch(
        self,
        model: str,
        gpu: GPUSlot,
        batch: list,
    ) -> None:
        """Process a batch of requests for a model on a GPU."""
        # Check if model needs loading
        if gpu.current_model != model:
            if (
                gpu.current_model
                and gpu.current_model in self._scheduler.pinned_models
            ):
                logger.debug(
                    f"GPU {gpu.gpu_id} has pinned model "
                    f"'{gpu.current_model}', loading '{model}' alongside"
                )
            await self._load_model(model, gpu)

        # Process each request in the batch
        for request in batch:
            await self._process_request(request, model)

    async def _load_model(self, model: str, gpu: GPUSlot) -> None:
        """Load a model onto a GPU (via Ollama warmup request)."""
        self._scheduler.update_gpu_state(
            gpu.gpu_id, model, ModelState.LOADING
        )
        load_start = time.time()

        try:
            # Ollama loads models on first request. Send a tiny warmup.
            await self._client.generate(
                prompt="hi",
                model=model,
                max_tokens=1,
                temperature=0.0,
            )
            load_ms = (time.time() - load_start) * 1000
            self._scheduler.record_load(model, load_ms)
            self._scheduler.update_gpu_state(
                gpu.gpu_id, model, ModelState.LOADED
            )
            logger.info(
                f"Model '{model}' loaded on GPU {gpu.gpu_id} "
                f"({load_ms:.0f}ms)"
            )
        except Exception as e:
            logger.error(f"Failed to load model '{model}': {e}")
            self._scheduler.update_gpu_state(
                gpu.gpu_id, None, ModelState.UNLOADED
            )

    async def _process_request(
        self, request: InferenceRequest, model: str
    ) -> None:
        """Process a single inference request."""
        future = self._futures.get(request.request_id)
        if future is None or future.done():
            return

        queue_wait_ms = int((time.time() - request.submitted_at) * 1000)
        infer_start = time.time()

        try:
            result = await self._client.generate_with_metrics(
                prompt=request.prompt,
                system=request.system,
                model=model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                extra_options=request.extra_options,
            )

            duration_ms = int((time.time() - infer_start) * 1000)
            self._scheduler.record_inference(model, duration_ms)

            inference_result = InferenceResult(
                request_id=request.request_id,
                text=result.text,
                model=model,
                duration_ms=duration_ms,
                queue_wait_ms=queue_wait_ms,
                prompt_tokens=result.prompt_eval_count,
                completion_tokens=result.eval_count,
            )

        except Exception as e:
            self._scheduler.record_error(model)
            inference_result = InferenceResult(
                request_id=request.request_id,
                text="",
                model=model,
                error=str(e),
            )

        self._completed += 1

        if not future.done():
            future.set_result(inference_result)
        self._futures.pop(request.request_id, None)
