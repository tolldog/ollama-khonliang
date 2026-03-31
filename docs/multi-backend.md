# Multi-Backend Clients

khonliang supports any LLM backend that speaks the OpenAI `/v1/chat/completions` API, alongside native Ollama support. One `ModelPool` can mix local and cloud backends.

## LLMClient Protocol

Both `OllamaClient` and `OpenAIClient` satisfy the `LLMClient` protocol:

```python
from khonliang import LLMClient, OllamaClient, OpenAIClient

# Both satisfy the same protocol
ollama: LLMClient = OllamaClient(model="llama3.2:3b")
openai: LLMClient = OpenAIClient(model="llama3.1:70b", base_url="http://gpu:8000/v1")

# Same interface
response = await ollama.generate("Hello!")
response = await openai.generate("Hello!")
```

The protocol includes: `generate`, `generate_with_metrics`, `stream_generate`, `generate_json`, `is_available_async`, and `close`.

## OpenAIClient

Works with any server or cloud provider that implements `/v1/chat/completions`:

```python
from khonliang import OpenAIClient

# Local vLLM server
client = OpenAIClient(model="Qwen/Qwen2.5-7B", base_url="http://localhost:8000/v1")

# Groq cloud (custom silicon, extreme speed)
client = OpenAIClient(
    model="llama-3.2-3b-preview",
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
)

# Together AI
client = OpenAIClient(
    model="meta-llama/Llama-3.1-70B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    api_key="...",
)

response = await client.generate("Hello!", system="Be helpful.")
```

### Supported Backends

| Category                 | Backends                                                                   |
| ------------------------ | -------------------------------------------------------------------------- |
| **Local GPU**            | vLLM, SGLang, TGI                                                          |
| **Local consumer**       | llama.cpp, LM Studio, Jan, GPT4All, llamafile, koboldcpp, LocalAI, MLC LLM |
| **Cloud fast inference** | Groq, Cerebras, SambaNova                                                  |
| **Cloud broad catalog**  | Together AI, Fireworks AI, OpenRouter                                      |

### Features

- Same retry logic as OllamaClient (3 attempts, exponential backoff)
- Same typed errors (`LLMTimeoutError`, `LLMUnavailableError`, `LLMModelNotFoundError`, `LLMRateLimitError`)
- JSON generation with `response_format={"type": "json_object"}` + cleanup fallback
- SSE streaming via `stream_generate()`
- `keep_alive` parameter accepted for protocol compatibility (no-op on OpenAI backends)

## Mixed Backend ModelPool

`ModelPool` supports URI-prefixed model specifiers to route roles to different backends:

```python
from khonliang import ModelPool

pool = ModelPool(
    {
        "researcher": "llama3.2:3b",                    # Ollama (default)
        "narrator": "openai://llama3.1:70b",            # vLLM on GPU box
        "classifier": "groq://llama-3.2-3b-preview",   # Groq cloud
    },
    base_url="http://localhost:11434",  # Ollama URL for plain models
    backends={
        "openai": {"base_url": "http://gpu-box:8000/v1"},
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "gsk_...",
        },
    },
)

# Each role gets the right client type
ollama_client = pool.get_client("researcher")   # OllamaClient
openai_client = pool.get_client("narrator")     # OpenAIClient
groq_client = pool.get_client("classifier")     # OpenAIClient
```

### URI Scheme Format

- `"llama3.2:3b"` — plain name, uses Ollama (backward compatible)
- `"openai://model-name"` — uses the `openai` backend config
- `"groq://model-name"` — uses the `groq` backend config
- Any scheme name works — just add it to `backends`

### Backend Config

Each backend needs at least `base_url`. Optional keys:

```python
backends={
    "my_backend": {
        "base_url": "http://localhost:8000/v1",  # Required
        "api_key": "...",                         # Optional
        "timeout": 120,                           # Optional (default: 120s)
        "model_timeouts": {"big-model": 300},     # Optional per-model
    },
}
```

## Using with Roles

Roles work identically regardless of backend — `BaseRole.client` returns whatever `LLMClient` the pool provides:

```python
class ResearcherRole(BaseRole):
    async def handle(self, message, session_id, context=None):
        # self.client could be OllamaClient or OpenAIClient
        response, elapsed = await self._timed_generate(
            prompt=message, system=self.system_prompt
        )
        return {"response": response, "metadata": {"role": self.role}}
```

## Using with LLM Manager

`InternalBackend` accepts any `LLMClient`:

```python
from khonliang.llm import LLMManager
from khonliang import OpenAIClient

# Use vLLM as the inference backend instead of Ollama
vllm_client = OpenAIClient(model="default", base_url="http://gpu:8000/v1")

manager = LLMManager(
    backend="internal",
    ollama_url="http://gpu:8000/v1",  # Not actually Ollama, but works
)
```
