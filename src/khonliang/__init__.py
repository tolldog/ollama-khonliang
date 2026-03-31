"""
khonliang — A llama rancher: multi-agent LLM orchestration for Ollama.

Quick start:

    from khonliang import OllamaClient, ModelPool, BaseRole, BaseRouter

    # Connect to Ollama
    client = OllamaClient(model="llama3.1:8b")
    response = await client.generate("Hello!")

    # Pool models by role
    pool = ModelPool({
        "triage": "llama3.2:3b",
        "researcher": "qwen2.5:7b",
    })

    # Multi-agent consensus
    from khonliang.consensus import AgentTeam, ConsensusEngine

    team = AgentTeam(agents=[...])
    result = await team.evaluate("Is this urgent?")
"""

from khonliang.client import GenerationResult, OllamaClient
from khonliang.errors import (
    LLMCooldownError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from khonliang.gateway.blackboard import Blackboard
from khonliang.health import ModelHealthTracker
from khonliang.openai_client import OpenAIClient
from khonliang.personalities import PersonalityConfig, PersonalityRegistry
from khonliang.pool import ModelPool
from khonliang.protocols import LLMClient
from khonliang.roles import BaseRole, BaseRouter

__all__ = [
    # Connection layer
    "OllamaClient",
    "OpenAIClient",
    "LLMClient",
    "GenerationResult",
    "ModelPool",
    "ModelHealthTracker",
    # Role layer
    "BaseRole",
    "BaseRouter",
    # Gateway
    "Blackboard",
    # Personalities
    "PersonalityConfig",
    "PersonalityRegistry",
    # Errors
    "LLMError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "LLMModelNotFoundError",
    "LLMRateLimitError",
    "LLMCooldownError",
]

__version__ = "0.1.0"
