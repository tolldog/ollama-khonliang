"""
Model routing strategies — select which model handles a request within a role.

Three strategies, all composable:
- StaticStrategy: always use the first candidate (no-op baseline)
- ComplexityStrategy: use a fast LLM to classify prompt complexity
- CascadeStrategy: try cheapest model first, escalate on low confidence

Example:
    from khonliang.routing.strategies import ComplexityStrategy, ModelSelection

    strategy = ComplexityStrategy(classifier_client=fast_client)
    selection = await strategy.select("researcher", "What year was X born?", candidates)
    # ModelSelection(model="llama3.2:3b", reason="complexity:simple", ...)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ModelSelection:
    """Result of a model routing decision."""

    model: str
    reason: str
    model_preferences: List[str] = field(default_factory=list)


@runtime_checkable
class RoutingStrategy(Protocol):
    """Protocol for model selection strategies."""

    async def select(
        self, role: str, message: str, candidates: List[str]
    ) -> ModelSelection: ...


class StaticStrategy:
    """Always use the first candidate. No-op baseline.

    Useful for testing or when you want explicit model assignment
    without any dynamic routing.
    """

    async def select(
        self, role: str, message: str, candidates: List[str]
    ) -> ModelSelection:
        return ModelSelection(
            model=candidates[0],
            reason="static",
            model_preferences=candidates,
        )


class ComplexityStrategy:
    """Classify prompt complexity and route to appropriately-sized model.

    Uses a fast classifier model (e.g. llama3.2:3b) to score prompt
    complexity as simple/medium/hard, then maps to the corresponding
    index in the candidates list.

    One extra ~100ms classifier call can save expensive model loads
    by routing simple queries to small models.

    Example:
        strategy = ComplexityStrategy(classifier_client=fast_client)
        # candidates=["llama3.2:3b", "qwen2.5:7b", "llama3.1:70b"]
        # simple → candidates[0], medium → candidates[1], hard → candidates[2]
    """

    CLASSIFIER_PROMPT = (
        "Classify the complexity of this user query. Reply with ONLY "
        "one word: simple, medium, or hard.\n\n"
        "- simple: factual lookup, yes/no, single-step reasoning\n"
        "- medium: multi-step reasoning, comparison, moderate analysis\n"
        "- hard: complex reasoning, synthesis, creative generation, "
        "multi-part questions\n\n"
        "Query: {message}\n\nComplexity:"
    )

    LEVELS = {"simple": 0, "medium": 1, "hard": 2}

    def __init__(
        self,
        classifier_client: Any,
        classifier_model: str = "llama3.2:3b",
        prompt_template: Optional[str] = None,
    ):
        self._client = classifier_client
        self._model = classifier_model
        self._prompt = prompt_template or self.CLASSIFIER_PROMPT

    async def select(
        self, role: str, message: str, candidates: List[str]
    ) -> ModelSelection:
        if len(candidates) == 1:
            return ModelSelection(
                model=candidates[0],
                reason="complexity:only_candidate",
                model_preferences=candidates,
            )

        try:
            prompt = self._prompt.format(message=message)
            response = await self._client.generate(
                prompt=prompt,
                model=self._model,
                temperature=0.1,
                max_tokens=10,
            )
            level = response.strip().lower().split()[0].rstrip(".,!:")
            idx = self.LEVELS.get(level, 0)
        except Exception as e:
            logger.debug(f"Complexity classification failed: {e}")
            idx = 0
            level = "fallback"

        # Clamp index to available candidates
        idx = min(idx, len(candidates) - 1)
        selected = candidates[idx]

        # Build preferences: selected first, then others in tier order
        preferences = [selected] + [c for c in candidates if c != selected]

        return ModelSelection(
            model=selected,
            reason=f"complexity:{level}",
            model_preferences=preferences,
        )


class CascadeStrategy:
    """Try cheapest model first, escalate if quality is low.

    Implements the FrugalGPT pattern: generate with the cheapest
    candidate, evaluate the response, and escalate to the next tier
    if confidence is below the threshold.

    Requires a client factory (to generate with different models) and
    optionally an evaluator (for confidence scoring). Without an
    evaluator, uses simple heuristics (response length, hedging markers).

    Example:
        strategy = CascadeStrategy(
            client_factory=lambda model: pool.get_client_for_model(model),
            confidence_threshold=0.7,
            evaluator=my_evaluator,
        )
    """

    HEDGING_MARKERS = [
        "i'm not sure",
        "i don't have",
        "i cannot",
        "i don't know",
        "uncertain",
        "it's unclear",
        "i'm unable",
        "no information",
    ]

    def __init__(
        self,
        client_factory: Callable[[str], Any],
        confidence_threshold: float = 0.7,
        evaluator: Optional[Any] = None,
        max_escalations: int = 2,
        system_prompt: str = "",
    ):
        self._client_factory = client_factory
        self._threshold = confidence_threshold
        self._evaluator = evaluator
        self._max_escalations = max_escalations
        self._system_prompt = system_prompt
        self.last_response: Optional[str] = None

    def _heuristic_confidence(self, response: str) -> float:
        """Simple heuristic confidence when no evaluator is configured."""
        text_lower = response.lower()

        if len(response.strip()) < 20:
            return 0.3

        hedging_count = sum(1 for m in self.HEDGING_MARKERS if m in text_lower)
        if hedging_count >= 2:
            return 0.4
        if hedging_count == 1:
            return 0.6

        return 0.85

    async def select(
        self, role: str, message: str, candidates: List[str]
    ) -> ModelSelection:
        if len(candidates) == 1:
            return ModelSelection(
                model=candidates[0],
                reason="cascade:only_candidate",
                model_preferences=candidates,
            )

        for i, model in enumerate(candidates[: self._max_escalations + 1]):
            try:
                client = self._client_factory(model)
                response = await client.generate(
                    prompt=message,
                    system=self._system_prompt,
                )
            except Exception as e:
                logger.warning(f"Cascade: {model} failed: {e}")
                continue

            # Evaluate confidence
            if self._evaluator:
                result = self._evaluator.evaluate(response, query=message)
                confidence = result.confidence
            else:
                confidence = self._heuristic_confidence(response)

            is_last = i >= len(candidates) - 1 or i >= self._max_escalations
            if confidence >= self._threshold or is_last:
                self.last_response = response
                remaining = [c for c in candidates[i:] if c != model]
                return ModelSelection(
                    model=model,
                    reason=f"cascade:tier{i}:confidence={confidence:.0%}",
                    model_preferences=[model] + remaining,
                )

            logger.info(
                f"Cascade: {model} confidence={confidence:.0%} < "
                f"{self._threshold:.0%}, escalating"
            )

        # Should not reach here, but fallback to last candidate
        return ModelSelection(
            model=candidates[-1],
            reason="cascade:exhausted",
            model_preferences=list(reversed(candidates)),
        )
