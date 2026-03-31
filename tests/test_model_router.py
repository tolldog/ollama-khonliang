"""Tests for ModelRouter and routing strategies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from khonliang.routing.model_router import ModelRouter
from khonliang.routing.strategies import (
    CascadeStrategy,
    ComplexityStrategy,
    StaticStrategy,
)


class TestStaticStrategy:
    @pytest.mark.asyncio
    async def test_returns_first_candidate(self):
        strategy = StaticStrategy()
        result = await strategy.select("researcher", "Hello", ["small", "medium", "large"])
        assert result.model == "small"
        assert result.reason == "static"
        assert result.model_preferences == ["small", "medium", "large"]

    @pytest.mark.asyncio
    async def test_single_candidate(self):
        strategy = StaticStrategy()
        result = await strategy.select("researcher", "Hello", ["only-model"])
        assert result.model == "only-model"


class TestComplexityStrategy:
    @pytest.mark.asyncio
    async def test_simple_query_routes_to_small(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="simple")

        strategy = ComplexityStrategy(classifier_client=mock_client)
        result = await strategy.select(
            "researcher", "What year was John born?", ["small", "medium", "large"]
        )
        assert result.model == "small"
        assert "simple" in result.reason

    @pytest.mark.asyncio
    async def test_medium_query_routes_to_medium(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="medium")

        strategy = ComplexityStrategy(classifier_client=mock_client)
        result = await strategy.select(
            "researcher", "Compare the migration patterns", ["small", "medium", "large"]
        )
        assert result.model == "medium"
        assert "medium" in result.reason

    @pytest.mark.asyncio
    async def test_hard_query_routes_to_large(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="hard")

        strategy = ComplexityStrategy(classifier_client=mock_client)
        result = await strategy.select(
            "researcher", "Synthesize the family history", ["small", "medium", "large"]
        )
        assert result.model == "large"
        assert "hard" in result.reason

    @pytest.mark.asyncio
    async def test_clamps_to_available_candidates(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="hard")

        strategy = ComplexityStrategy(classifier_client=mock_client)
        result = await strategy.select("researcher", "Complex query", ["small", "medium"])
        assert result.model == "medium"  # clamped to last

    @pytest.mark.asyncio
    async def test_fallback_on_classifier_error(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=Exception("timeout"))

        strategy = ComplexityStrategy(classifier_client=mock_client)
        result = await strategy.select(
            "researcher", "Hello", ["small", "medium", "large"]
        )
        assert result.model == "small"  # falls back to first
        assert "fallback" in result.reason

    @pytest.mark.asyncio
    async def test_single_candidate_skips_classification(self):
        mock_client = AsyncMock()

        strategy = ComplexityStrategy(classifier_client=mock_client)
        result = await strategy.select("researcher", "Hello", ["only-model"])
        assert result.model == "only-model"
        assert "only_candidate" in result.reason
        mock_client.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_preferences_include_all_candidates(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="medium")

        strategy = ComplexityStrategy(classifier_client=mock_client)
        result = await strategy.select(
            "researcher", "Query", ["small", "medium", "large"]
        )
        assert result.model_preferences[0] == "medium"
        assert set(result.model_preferences) == {"small", "medium", "large"}


class TestCascadeStrategy:
    @pytest.mark.asyncio
    async def test_returns_first_if_confident(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            return_value="John was born in 1842 in Ohio. His parents were..."
        )

        strategy = CascadeStrategy(
            client_factory=lambda m: mock_client,
            confidence_threshold=0.7,
        )
        result = await strategy.select(
            "researcher", "When was John born?", ["small", "medium", "large"]
        )
        assert result.model == "small"
        assert "tier0" in result.reason
        assert result.generated_text is not None

    @pytest.mark.asyncio
    async def test_escalates_on_low_confidence(self):
        call_count = 0

        async def generate_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "I'm not sure, I don't have enough information"
            return "John was born in 1842 in Ohio based on the GEDCOM records."

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=generate_side_effect)

        strategy = CascadeStrategy(
            client_factory=lambda m: mock_client,
            confidence_threshold=0.7,
        )
        result = await strategy.select(
            "researcher", "When was John born?", ["small", "medium", "large"]
        )
        assert result.model == "medium"
        assert "tier1" in result.reason

    @pytest.mark.asyncio
    async def test_escalates_with_evaluator(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="Some response")

        low_eval = MagicMock()
        low_eval.evaluate = MagicMock(
            return_value=MagicMock(confidence=0.3)
        )

        strategy = CascadeStrategy(
            client_factory=lambda m: mock_client,
            confidence_threshold=0.7,
            evaluator=low_eval,
            max_escalations=1,
        )
        result = await strategy.select(
            "researcher", "Query", ["small", "large"]
        )
        assert result.model == "large"

    @pytest.mark.asyncio
    async def test_stops_at_max_escalations(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            return_value="I'm not sure about this at all"
        )

        strategy = CascadeStrategy(
            client_factory=lambda m: mock_client,
            confidence_threshold=0.99,
            max_escalations=1,
        )
        result = await strategy.select(
            "researcher", "Query", ["small", "medium", "large"]
        )
        # Should stop at medium (index 1 = max_escalations)
        assert result.model == "medium"

    @pytest.mark.asyncio
    async def test_single_candidate(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="Response")

        strategy = CascadeStrategy(
            client_factory=lambda m: mock_client,
        )
        result = await strategy.select("researcher", "Hello", ["only-model"])
        assert result.model == "only-model"
        assert "only_candidate" in result.reason

    @pytest.mark.asyncio
    async def test_skips_failed_models(self):
        call_models = []

        def factory(model):
            call_models.append(model)
            mock = AsyncMock()
            if model == "small":
                mock.generate = AsyncMock(side_effect=Exception("connection refused"))
            else:
                mock.generate = AsyncMock(return_value="Good detailed response here.")
            return mock

        strategy = CascadeStrategy(
            client_factory=factory,
            confidence_threshold=0.7,
        )
        result = await strategy.select(
            "researcher", "Query", ["small", "medium"]
        )
        assert result.model == "medium"
        assert "small" in call_models


class TestModelRouter:
    @pytest.mark.asyncio
    async def test_delegates_to_strategy(self):
        strategy = StaticStrategy()
        router = ModelRouter(
            role_models={"researcher": ["small", "medium"]},
            strategy=strategy,
        )
        result = await router.select("researcher", "Hello")
        assert result.model == "small"

    @pytest.mark.asyncio
    async def test_unknown_role_returns_empty(self):
        strategy = StaticStrategy()
        router = ModelRouter(
            role_models={"researcher": ["small"]},
            strategy=strategy,
        )
        result = await router.select("unknown_role", "Hello")
        assert result.model == ""
        assert result.reason == "no_candidates"

    def test_get_candidates(self):
        router = ModelRouter(
            role_models={"researcher": ["small", "medium", "large"]},
            strategy=StaticStrategy(),
        )
        assert router.get_candidates("researcher") == ["small", "medium", "large"]
        assert router.get_candidates("unknown") == []

    def test_health_tracker_filters_candidates(self):
        mock_tracker = MagicMock()
        mock_tracker.is_cooled_down = MagicMock(
            side_effect=lambda m: m == "small"
        )

        router = ModelRouter(
            role_models={"researcher": ["small", "medium", "large"]},
            strategy=StaticStrategy(),
            health_tracker=mock_tracker,
        )
        candidates = router.get_candidates("researcher")
        assert "small" not in candidates
        assert candidates == ["medium", "large"]

    def test_health_tracker_keeps_all_if_all_cooled(self):
        mock_tracker = MagicMock()
        mock_tracker.is_cooled_down = MagicMock(return_value=True)

        router = ModelRouter(
            role_models={"researcher": ["small", "medium"]},
            strategy=StaticStrategy(),
            health_tracker=mock_tracker,
        )
        # Falls back to full list when all are cooled
        candidates = router.get_candidates("researcher")
        assert candidates == ["small", "medium"]
