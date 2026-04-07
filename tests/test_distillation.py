"""Tests for self-distillation inference mode (KH-14)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from khonliang.client import GenerationResult, OllamaClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_result(text: str, prompt_tokens: int = 10, eval_tokens: int = 20,
                duration_ns: int = 1_000_000_000) -> GenerationResult:
    return GenerationResult(
        text=text,
        model="test-model",
        prompt_eval_count=prompt_tokens,
        eval_count=eval_tokens,
        total_duration_ns=duration_ns,
        eval_duration_ns=duration_ns // 2,
    )


# ---------------------------------------------------------------------------
# GenerationResult distillation fields
# ---------------------------------------------------------------------------

class TestGenerationResultDistillation:
    def test_default_not_distilled(self):
        r = _gen_result("hello")
        assert r.distilled is False
        assert r.candidates_generated == 1
        assert r.selected_index == 0

    def test_distilled_flag(self):
        r = _gen_result("hello")
        r.candidates_generated = 3
        r.selected_index = 1
        assert r.distilled is True

    def test_total_token_fields(self):
        r = _gen_result("hello")
        r.total_prompt_tokens = 100
        r.total_eval_tokens = 200
        assert r.total_prompt_tokens == 100
        assert r.total_eval_tokens == 200


# ---------------------------------------------------------------------------
# Self-distillation (_distill)
# ---------------------------------------------------------------------------

class TestDistill:
    @pytest.mark.asyncio
    async def test_n_samples_1_no_distillation(self):
        """n_samples=1 should bypass distillation entirely."""
        client = OllamaClient(model="test")

        mock_result = _gen_result("single response")
        with patch.object(client, "_do_generate", new_callable=AsyncMock,
                          return_value=mock_result):
            result = await client.generate_with_metrics("test prompt", n_samples=1)

        assert result.text == "single response"
        assert result.distilled is False

    @pytest.mark.asyncio
    async def test_n_samples_zero_raises(self):
        """n_samples=0 should raise ValueError."""
        client = OllamaClient(model="test")
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            await client.generate_with_metrics("test prompt", n_samples=0)

    @pytest.mark.asyncio
    async def test_n_samples_negative_raises(self):
        """Negative n_samples should raise ValueError."""
        client = OllamaClient(model="test")
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            await client.generate_with_metrics("test prompt", n_samples=-1)


    async def test_distill_selects_best(self):
        """Distillation should run N samples and a selection call."""
        client = OllamaClient(model="test")

        candidates = [
            _gen_result("response A", prompt_tokens=10, eval_tokens=20),
            _gen_result("response B", prompt_tokens=10, eval_tokens=25),
            _gen_result("response C", prompt_tokens=10, eval_tokens=15),
        ]
        # Selection call returns "2" (picking response B)
        selection = _gen_result("2", prompt_tokens=50, eval_tokens=2)

        call_count = 0

        async def mock_generate_with_metrics(**kwargs):
            nonlocal call_count
            if kwargs.get("n_samples", 1) > 1:
                # This is the top-level call — delegate to real _distill
                return await real_distill(**kwargs)
            call_count += 1
            if call_count <= 3:
                return candidates[call_count - 1]
            return selection

        real_distill = client._distill

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_generate_with_metrics):
            result = await client._distill(
                prompt="test",
                system=None,
                temperature=0.7,
                max_tokens=4000,
                model=None,
                extra_options=None,
                keep_alive=None,
                n_samples=3,
            )

        assert result.text == "response B"
        assert result.selected_index == 1
        assert result.candidates_generated == 3
        assert result.distilled is True
        assert result.total_prompt_tokens > 0
        assert result.total_eval_tokens > 0

    @pytest.mark.asyncio
    async def test_distill_handles_partial_failures(self):
        """If some candidates fail, distillation should work with survivors."""
        client = OllamaClient(model="test")

        good_result = _gen_result("good response")
        selection = _gen_result("1")

        call_count = 0

        async def mock_gwm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return good_result
            if call_count == 2:
                raise RuntimeError("timeout")
            if call_count == 3:
                return _gen_result("another response")
            return selection  # selection call

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_gwm):
            result = await client._distill(
                prompt="test",
                system=None,
                temperature=0.7,
                max_tokens=4000,
                model=None,
                extra_options=None,
                keep_alive=None,
                n_samples=3,
            )

        # Should succeed with the surviving candidates
        assert result.candidates_generated == 3
        assert result.distilled is True

    @pytest.mark.asyncio
    async def test_distill_all_fail_raises(self):
        """If all candidates fail, should raise."""
        client = OllamaClient(model="test")

        async def mock_gwm(**kwargs):
            raise RuntimeError("all fail")

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_gwm):
            with pytest.raises(RuntimeError, match="all fail"):
                await client._distill(
                    prompt="test",
                    system=None,
                    temperature=0.7,
                    max_tokens=4000,
                    model=None,
                    extra_options=None,
                    keep_alive=None,
                    n_samples=3,
                )

    @pytest.mark.asyncio
    async def test_distill_single_survivor(self):
        """If only one candidate survives, return it without selection call."""
        client = OllamaClient(model="test")

        good_result = _gen_result("only survivor")
        call_count = 0

        async def mock_gwm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return good_result
            raise RuntimeError("fail")

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_gwm):
            result = await client._distill(
                prompt="test",
                system=None,
                temperature=0.7,
                max_tokens=4000,
                model=None,
                extra_options=None,
                keep_alive=None,
                n_samples=2,
            )

        assert result.text == "only survivor"
        assert result.candidates_generated == 2
        assert result.selected_index == 0

    @pytest.mark.asyncio
    async def test_distill_invalid_selection_defaults_to_first(self):
        """If selector returns garbage, default to first candidate."""
        client = OllamaClient(model="test")

        candidates = [
            _gen_result("first"),
            _gen_result("second"),
        ]
        selection = _gen_result("banana")  # invalid

        call_count = 0

        async def mock_gwm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return candidates[call_count - 1]
            return selection

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_gwm):
            result = await client._distill(
                prompt="test",
                system=None,
                temperature=0.7,
                max_tokens=4000,
                model=None,
                extra_options=None,
                keep_alive=None,
                n_samples=2,
            )

        assert result.text == "first"
        assert result.selected_index == 0

    @pytest.mark.asyncio
    async def test_distill_out_of_range_defaults_to_first(self):
        """If selector returns a number out of range, default to first."""
        client = OllamaClient(model="test")

        candidates = [_gen_result("first"), _gen_result("second")]
        selection = _gen_result("99")

        call_count = 0

        async def mock_gwm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return candidates[call_count - 1]
            return selection

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_gwm):
            result = await client._distill(
                prompt="test",
                system=None,
                temperature=0.7,
                max_tokens=4000,
                model=None,
                extra_options=None,
                keep_alive=None,
                n_samples=2,
            )

        assert result.text == "first"
        assert result.selected_index == 0

    @pytest.mark.asyncio
    async def test_elevated_temperature_for_diversity(self):
        """Sampling calls should use elevated temperature."""
        client = OllamaClient(model="test")

        captured_temps = []

        async def mock_gwm(prompt="", system=None, temperature=0.7,
                           max_tokens=4000, model=None, extra_options=None,
                           keep_alive=None, n_samples=1):
            captured_temps.append(temperature)
            return _gen_result(f"response {len(captured_temps)}")

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_gwm):
            await client._distill(
                prompt="test",
                system=None,
                temperature=0.5,
                max_tokens=4000,
                model=None,
                extra_options=None,
                keep_alive=None,
                n_samples=2,
            )

        # First 2 calls are samples (elevated temp), 3rd is selector (low temp)
        assert captured_temps[0] == 0.65  # 0.5 + 0.15
        assert captured_temps[1] == 0.65
        assert captured_temps[2] == 0.1   # selector

    @pytest.mark.asyncio
    async def test_aggregate_metrics(self):
        """Total tokens should sum across all candidates + selection."""
        client = OllamaClient(model="test")

        candidates = [
            _gen_result("a", prompt_tokens=10, eval_tokens=20),
            _gen_result("b", prompt_tokens=15, eval_tokens=25),
        ]
        selection = _gen_result("1", prompt_tokens=50, eval_tokens=2)

        call_count = 0

        async def mock_gwm(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return candidates[call_count - 1]
            return selection

        with patch.object(client, "generate_with_metrics",
                          side_effect=mock_gwm):
            result = await client._distill(
                prompt="test",
                system=None,
                temperature=0.7,
                max_tokens=4000,
                model=None,
                extra_options=None,
                keep_alive=None,
                n_samples=2,
            )

        assert result.total_prompt_tokens == 10 + 15 + 50
        assert result.total_eval_tokens == 20 + 25 + 2
