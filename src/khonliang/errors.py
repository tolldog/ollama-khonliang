"""
Typed error hierarchy for LLM operations.

Provides specific exception types with HTTP status codes for clean error
propagation from the Ollama client through to the application layer.
"""


class LLMError(Exception):
    """Base class for LLM operation errors."""

    status_code: int = 500

    def __init__(self, message: str, model: str = ""):
        self.model = model
        super().__init__(message)


class LLMTimeoutError(LLMError):
    """LLM request exceeded timeout."""

    status_code = 504


class LLMUnavailableError(LLMError):
    """LLM provider (Ollama) is unreachable."""

    status_code = 503


class LLMModelNotFoundError(LLMError):
    """Requested model is not loaded in Ollama."""

    status_code = 404


class LLMRateLimitError(LLMError):
    """Model is rate-limited or GPU is contended."""

    status_code = 429


class LLMCooldownError(LLMUnavailableError):
    """Model is in cooldown after repeated failures."""

    status_code = 503
