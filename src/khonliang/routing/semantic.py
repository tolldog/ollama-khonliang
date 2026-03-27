"""
Semantic intent router using embedding similarity.

Routes messages to named roles based on cosine similarity against
example utterances per route. Runs on CPU via FastEmbed ONNX in <5ms.

Requires the `rag` optional dependency:
    pip install ollama-khonliang[rag]

Usage:

    router = SemanticIntentRouter(
        routes={
            "urgent": [
                "server is down",
                "critical issue",
                "nothing is working",
                "production outage",
            ],
            "billing": [
                "invoice",
                "charge",
                "refund",
                "payment failed",
            ],
            "general": [
                "hello",
                "how do I",
                "can you help",
            ],
        },
        fallback="general",
        score_threshold=0.70,
    )

    role = router.classify("my server crashed")   # -> "urgent"
    role, score = router.classify_with_score("refund my money")  # -> ("billing", 0.82)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SCORE_THRESHOLD = 0.70


class SemanticIntentRouter:
    """
    Embedding-based message router using FastEmbed ONNX encoder.

    Maps messages to named routes via cosine similarity against
    example utterances. No GPU required — <5ms per classification.

    Args:
        routes:          Dict of route_name → list of example utterances
        fallback:        Route name returned when no match exceeds threshold
        score_threshold: Minimum similarity score for a match (0.0-1.0)
        model:           FastEmbed model name (default: BAAI/bge-small-en-v1.5)
    """

    def __init__(
        self,
        routes: Dict[str, List[str]],
        fallback: str = "default",
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        model: str = "BAAI/bge-small-en-v1.5",
    ):
        self._fallback = fallback
        self._threshold = score_threshold
        self._router = None

        self._routes_config = routes
        self._model = model

    def _ensure_router(self) -> None:
        """Lazy-initialize the semantic router (loads ONNX model on first call)."""
        if self._router is not None:
            return

        try:
            from semantic_router import Route, SemanticRouter  # type: ignore[attr-defined]
            from semantic_router.encoders import FastEmbedEncoder  # type: ignore[import-unresolved]
        except ImportError as e:
            raise ImportError(
                "SemanticIntentRouter requires the 'rag' optional dependency. "
                "Install with: pip install ollama-khonliang[rag]"
            ) from e

        logger.info(f"Initializing semantic router encoder ({self._model})...")
        t0 = time.monotonic()
        encoder = FastEmbedEncoder(model_name=self._model)

        route_objects = [
            Route(
                name=name,
                utterances=utterances,
                score_threshold=self._threshold,
            )
            for name, utterances in self._routes_config.items()
        ]

        self._router = SemanticRouter(
            encoder=encoder, routes=route_objects, auto_sync="local"
        )

        elapsed = time.monotonic() - t0
        logger.info(f"Semantic router initialized in {elapsed:.1f}s ({len(route_objects)} routes)")

    def classify(self, message: str) -> str:
        """
        Classify a message to a route name.

        Args:
            message: Input text to classify

        Returns:
            Route name, or fallback if no match exceeds threshold
        """
        route, _ = self.classify_with_score(message)
        return route

    def classify_with_score(self, message: str) -> Tuple[str, float]:
        """
        Classify and return the route name and similarity score.

        Args:
            message: Input text to classify

        Returns:
            (route_name, score) — score is 0.0 when fallback is used
        """
        if not message or not message.strip():
            return self._fallback, 0.0

        self._ensure_router()

        t0 = time.monotonic()
        result = self._router(message)
        elapsed = time.monotonic() - t0

        if logger.isEnabledFor(logging.DEBUG):
            score = getattr(result, "similarity_score", 0.0) or 0.0
            logger.debug(
                f"Semantic route: '{message[:60]}' → {result.name or self._fallback} "
                f"(score={score:.3f}, {elapsed*1000:.1f}ms)"
            )

        if not result.name:
            return self._fallback, 0.0

        score = float(getattr(result, "similarity_score", 0.0) or 0.0)
        return result.name, score

    def add_route(self, name: str, utterances: List[str]) -> None:
        """
        Add or replace a route. Triggers router re-initialization.

        Args:
            name:        Route name
            utterances:  Example utterances for this route
        """
        self._routes_config[name] = utterances
        self._router = None  # Force re-init on next call

    @property
    def route_names(self) -> List[str]:
        """Names of all registered routes."""
        return list(self._routes_config.keys())
