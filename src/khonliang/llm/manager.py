"""
LLM Manager — pluggable inference scheduler facade.

The entry point for managed LLM inference. Users choose a backend
and get the same interface regardless.

Example:

    # No manager (current behavior, zero overhead)
    client = OllamaClient(model="qwen2.5:7b")
    await client.generate(...)

    # With internal scheduler
    manager = LLMManager(backend="internal")
    await manager.start()
    text = await manager.generate("qwen2.5:7b", "Hello!")
    status = await manager.status()
    await manager.stop()

    # With external scheduler (future)
    manager = LLMManager(backend="grpc", endpoint="localhost:50051")
"""

import logging
from typing import Any, Dict, List, Optional

from khonliang.llm.protocol import (
    GPUSlot,
    InferenceRequest,
    InferenceResult,
    QueueType,
    SchedulerStatus,
)

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Facade for managed LLM inference.

    Wraps a backend (internal or external) and provides a clean API
    for submitting inference requests with priority, queuing, and
    score-based scheduling.

    Args:
        backend: "internal" (in-process) or "grpc" (external, future)
        ollama_url: Ollama server URL (for internal backend)
        gpus: GPU slots (for internal backend)
        model_vram: Map of model -> estimated VRAM in MB
        endpoint: gRPC endpoint (for grpc backend, future)
    """

    def __init__(
        self,
        backend: str = "internal",
        ollama_url: str = "http://localhost:11434",
        gpus: Optional[List[GPUSlot]] = None,
        model_vram: Optional[Dict[str, int]] = None,
        max_batch_size: int = 10,
        **kwargs: Any,
    ):
        self._backend_type = backend

        if backend == "internal":
            from khonliang.llm.internal import InternalBackend

            self._backend = InternalBackend(
                ollama_url=ollama_url,
                gpus=gpus,
                max_batch_size=max_batch_size,
                model_vram=model_vram,
            )
        elif backend == "grpc":
            raise NotImplementedError(
                "gRPC backend not yet implemented. "
                "Use backend='internal' or build the external scheduler."
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    async def start(self) -> None:
        """Start the manager and its backend."""
        await self._backend.start()
        logger.info(f"LLM Manager started (backend={self._backend_type})")

    async def stop(self) -> None:
        """Stop the manager."""
        await self._backend.stop()

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        priority: int = 0,
        queue: QueueType = QueueType.DEFAULT,
        **kwargs: Any,
    ) -> str:
        """
        Generate text — the main API.

        Submits to the scheduler, waits for result, returns text.
        Priority and queue type affect scheduling order.
        """
        return await self._backend.generate(
            model=model,
            prompt=prompt,
            system=system,
            priority=priority,
            queue=queue,
            **kwargs,
        )

    async def submit(self, request: InferenceRequest) -> str:
        """Submit a request for async processing. Returns request_id."""
        return await self._backend.submit(request)

    async def get_result(
        self, request_id: str, timeout: float = 30.0
    ) -> InferenceResult:
        """Wait for a submitted request's result."""
        return await self._backend.get_result(request_id, timeout=timeout)

    async def status(self) -> SchedulerStatus:
        """Get scheduler status."""
        return await self._backend.status()

    @property
    def backend_type(self) -> str:
        return self._backend_type
