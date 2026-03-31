"""Tests for LLMClient protocol — verify both clients satisfy it."""

from khonliang.client import OllamaClient
from khonliang.openai_client import OpenAIClient
from khonliang.protocols import LLMClient


class TestLLMClientProtocol:
    def test_ollama_client_satisfies_protocol(self):
        assert isinstance(OllamaClient(), LLMClient)

    def test_openai_client_satisfies_protocol(self):
        assert isinstance(OpenAIClient(model="test"), LLMClient)

    def test_arbitrary_object_does_not_satisfy(self):
        assert not isinstance(object(), LLMClient)

    def test_both_have_model_attribute(self):
        ollama = OllamaClient(model="test")
        openai = OpenAIClient(model="test")
        assert hasattr(ollama, "model")
        assert hasattr(openai, "model")
        assert ollama.model == "test"
        assert openai.model == "test"

    def test_both_have_generate_method(self):
        ollama = OllamaClient()
        openai = OpenAIClient(model="test")
        assert hasattr(ollama, "generate")
        assert hasattr(openai, "generate")
        assert callable(ollama.generate)
        assert callable(openai.generate)

    def test_both_have_stream_generate_method(self):
        ollama = OllamaClient()
        openai = OpenAIClient(model="test")
        assert hasattr(ollama, "stream_generate")
        assert hasattr(openai, "stream_generate")

    def test_both_have_generate_json_method(self):
        ollama = OllamaClient()
        openai = OpenAIClient(model="test")
        assert hasattr(ollama, "generate_json")
        assert hasattr(openai, "generate_json")

    def test_both_have_close_method(self):
        ollama = OllamaClient()
        openai = OpenAIClient(model="test")
        assert hasattr(ollama, "close")
        assert hasattr(openai, "close")

    def test_both_have_is_available_async(self):
        ollama = OllamaClient()
        openai = OpenAIClient(model="test")
        assert hasattr(ollama, "is_available_async")
        assert hasattr(openai, "is_available_async")
