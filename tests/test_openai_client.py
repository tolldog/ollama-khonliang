"""Tests for OpenAIClient — OpenAI-compatible LLM client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from khonliang.client import GenerationResult
from khonliang.errors import (
    LLMModelNotFoundError,
    LLMRateLimitError,
)
from khonliang.openai_client import OpenAIClient


@pytest.fixture
def client():
    return OpenAIClient(model="test-model", base_url="http://localhost:8000/v1")


@pytest.fixture
def client_with_key():
    return OpenAIClient(
        model="test-model",
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_test123",
    )


class TestBuildMessages:
    def test_user_only(self, client):
        msgs = client._build_messages("Hello")
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_with_system(self, client):
        msgs = client._build_messages("Hello", system="Be helpful")
        assert msgs == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]


class TestHeaders:
    def test_no_api_key(self, client):
        headers = client._headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_with_api_key(self, client_with_key):
        headers = client_with_key._headers()
        assert headers["Authorization"] == "Bearer gsk_test123"


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self, client):
        mock_response = {
            "choices": [{"message": {"content": "Hello there!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            "model": "test-model",
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        result = await client.generate("Hello")
        assert result == "Hello there!"

    @pytest.mark.asyncio
    async def test_generate_with_metrics(self, client):
        mock_response = {
            "choices": [{"message": {"content": "Response text"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "test-model",
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        result = await client.generate_with_metrics("Hello")
        assert isinstance(result, GenerationResult)
        assert result.text == "Response text"
        assert result.model == "test-model"
        assert result.prompt_eval_count == 10
        assert result.eval_count == 5

    @pytest.mark.asyncio
    async def test_404_raises_model_not_found(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        with pytest.raises(LLMModelNotFoundError):
            await client.generate("Hello")

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit(self, client):
        mock_resp = AsyncMock()
        mock_resp.status = 429
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client._session = mock_session

        with pytest.raises(LLMRateLimitError):
            await client.generate("Hello")


class TestModelOverride:
    def test_default_model(self, client):
        assert client.model == "test-model"

    def test_timeout_override(self):
        client = OpenAIClient(
            model="test",
            model_timeouts={"big-model": 300},
        )
        assert client._get_timeout("big-model") == 300
        assert client._get_timeout("unknown") == 120  # default


class TestParseModelSpec:
    def test_plain_model(self):
        from khonliang.pool import ModelPool

        scheme, model = ModelPool._parse_model_spec("llama3.2:3b")
        assert scheme is None
        assert model == "llama3.2:3b"

    def test_openai_scheme(self):
        from khonliang.pool import ModelPool

        scheme, model = ModelPool._parse_model_spec("openai://llama3.1:70b")
        assert scheme == "openai"
        assert model == "llama3.1:70b"

    def test_groq_scheme(self):
        from khonliang.pool import ModelPool

        scheme, model = ModelPool._parse_model_spec("groq://llama-3.2-3b-preview")
        assert scheme == "groq"
        assert model == "llama-3.2-3b-preview"


class TestModelPoolMixedBackends:
    def test_plain_model_creates_ollama_client(self):
        from khonliang.pool import ModelPool

        pool = ModelPool({"triage": "llama3.2:3b"})
        client = pool.get_client("triage")
        assert type(client).__name__ == "OllamaClient"

    def test_openai_scheme_creates_openai_client(self):
        from khonliang.pool import ModelPool

        pool = ModelPool(
            {"narrator": "openai://llama3.1:70b"},
            backends={"openai": {"base_url": "http://gpu-box:8000/v1"}},
        )
        client = pool.get_client("narrator")
        assert type(client).__name__ == "OpenAIClient"
        assert client.model == "llama3.1:70b"

    def test_missing_backend_raises(self):
        from khonliang.pool import ModelPool

        pool = ModelPool({"narrator": "vllm://model"})
        with pytest.raises(KeyError, match="Backend 'vllm' not configured"):
            pool.get_client("narrator")

    def test_missing_base_url_raises_value_error(self):
        from khonliang.pool import ModelPool

        pool = ModelPool(
            {"narrator": "openai://llama3.1:70b"},
            backends={"openai": {"api_key": "sk-test"}},  # no base_url
        )
        with pytest.raises(ValueError, match="missing required 'base_url'"):
            pool.get_client("narrator")

    def test_get_model_name_strips_scheme(self):
        from khonliang.pool import ModelPool

        pool = ModelPool(
            {"narrator": "openai://llama3.1:70b"},
            backends={"openai": {"base_url": "http://localhost:8000/v1"}},
        )
        assert pool.get_model_name("narrator") == "llama3.1:70b"

    def test_mixed_pool(self):
        from khonliang.pool import ModelPool

        pool = ModelPool(
            {
                "researcher": "llama3.2:3b",
                "narrator": "openai://llama3.1:70b",
            },
            backends={"openai": {"base_url": "http://gpu-box:8000/v1"}},
        )
        ollama = pool.get_client("researcher")
        openai = pool.get_client("narrator")
        assert type(ollama).__name__ == "OllamaClient"
        assert type(openai).__name__ == "OpenAIClient"

    def test_api_key_passed_to_openai_client(self):
        from khonliang.pool import ModelPool

        pool = ModelPool(
            {"classifier": "groq://llama-3.2-3b-preview"},
            backends={
                "groq": {
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key": "gsk_test",
                }
            },
        )
        client = pool.get_client("classifier")
        assert client.api_key == "gsk_test"
