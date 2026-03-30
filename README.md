# ollama-khonliang

_A llama rancher_ — multi-agent LLM orchestration framework for [Ollama](https://ollama.com).

**khonliang** (คนเลี้ยง) means "caretaker" or "keeper" in Thai — a rancher wrangling your herd of local LLMs.

## Features

- **Async Ollama client** with typed errors, per-model timeouts, exponential backoff retry, and streaming
- **Role-based agents** with abstract base class and model pool management
- **Message routing** — priority-ordered: callable rules, regex, keywords, semantic embedding similarity, fallback
- **Multi-agent consensus** — weighted voting, VETO blocking, parallel agent orchestration with caching
- **Personality system** — named agent personas with voting weights, focus areas, and @mention resolution
- **Scoped RAG** — SQLite FTS5 retrieval with per-agent knowledge scopes (global, domain, conversational, expert)
- **Structured output parsing** — extracts typed JSON from LLM fenced code blocks with auto-cleanup
- **Flow classification** — LLM-based intent classification within conversation flows
- **Three-tier knowledge** — axioms (immutable rules), imported (user docs), derived (from interactions), with librarian agent
- **Research pool** — managed worker threads with BaseEngine/CompositeResearcher, multi-engine parallel search
- **LLM manager** — score-based inference scheduler with model preferences, pinning, VRAM budgeting, and benchmarking
- **Query parser** — schema-driven LLM extraction of structured parameters from natural language
- **Training pipeline** — interaction logging, feedback collection, JSONL export (alpaca/sharegpt/completion)
- **Mattermost integration** — WebSocket bot with @mention handlers and typing indicators
- **WebSocket chat server** — session tracking, role routing, feedback, knowledge indexing
- **mDNS service discovery** — advertise and discover services on the local network
- **Agent gateway** — Redis Streams message bus for distributed agents

## Installation

```bash
# Core only (aiohttp + requests)
pip install ollama-khonliang

# With optional dependencies
pip install ollama-khonliang[rag]          # + semantic routing & embeddings
pip install ollama-khonliang[mattermost]   # + Mattermost bot
pip install ollama-khonliang[gateway]      # + Redis message bus
pip install ollama-khonliang[discovery]    # + mDNS/zeroconf
pip install ollama-khonliang[all]          # Everything
pip install ollama-khonliang[dev]          # Development tools
```

Requires a running [Ollama](https://ollama.com) instance (default: `http://localhost:11434`).

## Quick Start

### Basic generation

```python
import asyncio
from khonliang import OllamaClient

async def main():
    async with OllamaClient(model="llama3.1:8b") as client:
        response = await client.generate("Explain async Python in one paragraph.")
        print(response)

asyncio.run(main())
```

### Role-based agents with routing

```python
from khonliang import BaseRole, BaseRouter, ModelPool

class TriageRole(BaseRole):
    async def handle(self, message, session_id, context=None):
        prompt = f"Classify urgency: {message}"
        response = await self.client.generate(prompt, system=self.system_prompt)
        return {"response": response, "metadata": {"role": self.role}}

pool = ModelPool({"triage": "llama3.2:3b", "knowledge": "qwen2.5:7b"})

router = BaseRouter(fallback_role="knowledge")
router.register_pattern(r"(?i)urgent|critical|down", "triage")
router.register_keywords(["how to", "explain", "what is"], "knowledge")

role = router.route("My server is down!")  # -> "triage"
```

### Multi-agent consensus

```python
from khonliang.consensus import AgentTeam, AgentVote

class UrgencyAgent:
    agent_id = "urgency"
    async def analyze(self, subject, context):
        return AgentVote(
            agent_id=self.agent_id,
            action="APPROVE",
            confidence=0.85,
            reasoning="High urgency detected",
        )

team = AgentTeam(agents=[UrgencyAgent(), ...])
result = await team.evaluate("Server is down!")
print(result.action, result.confidence)
```

### Semantic routing

```python
from khonliang.routing import SemanticIntentRouter
from khonliang import BaseRouter

router = BaseRouter(fallback_role="general")
router.register_pattern(r"(?i)urgent|critical", "triage")
router.set_semantic_router(SemanticIntentRouter(
    routes={
        "triage": ["server is down", "nothing works", "production broke"],
        "billing": ["invoice", "refund", "payment failed"],
        "knowledge": ["how do I", "tutorial", "explain"],
    },
    fallback="general",
))

role = router.route("I need a refund")  # -> "billing" (semantic match)
```

## CLI

```bash
khonliang models                    # List available Ollama models
khonliang health                    # Check Ollama server health
khonliang chat --role triage        # Interactive chat with a role
khonliang generate "Hello!" --model llama3.2:3b
khonliang route "My server is down!" --router myapp.router:MyRouter
khonliang test tests.jsonl --router myapp.router:MyRouter
```

## Architecture

```text
khonliang/
├── client.py          # Async Ollama client with retry & streaming
├── pool.py            # Role → model mapping with connection reuse
├── health.py          # Model health tracking & cooldown
├── errors.py          # Typed error hierarchy
├── personalities.py   # Agent personality configs & registry
├── roles/             # Base role & message router
├── consensus/         # Multi-agent voting, team orchestration, adaptive weights
├── routing/           # Flow classifier & semantic intent router
├── parsing/           # Structured JSON extraction + LLM query parser
├── rag/               # Document retrieval (FTS5) with scoped access
├── knowledge/         # Three-tier store, librarian agent, ingestion pipeline
├── research/          # Worker pool, engines, composite researcher, triggers
├── llm/               # Inference scheduler, model profiles, benchmarking
├── agents/            # Activation rules, capabilities, config registry
├── integrations/      # Mattermost bot, WebSocket chat server
├── training/          # Feedback collection & fine-tuning export
├── gateway/           # Redis Streams agent bus
├── discovery/         # mDNS service discovery
└── debate/            # Structured agent debate
```

## Module Overview

| Layer         | Module                                              | Description                                                     |
| ------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| Connection    | `client`, `pool`, `health`, `errors`                | Async Ollama client, model pool, health tracking                |
| Roles         | `roles.base`, `roles.router`                        | Abstract role base class, priority-ordered message router       |
| Personalities | `personalities`                                     | Named personas with voting weights and @mention resolution      |
| Consensus     | `consensus.models`, `engine`, `team`, `weights`     | Weighted voting, VETO, parallel orchestration, adaptive weights |
| Routing       | `routing.flow`, `routing.semantic`                  | LLM flow classification, embedding-based intent routing         |
| Parsing       | `parsing.structured`, `parsing.query_parser`        | Typed JSON extraction, schema-driven LLM query parsing          |
| RAG           | `rag.retriever`, `rag.scoped`                       | FTS5 search, agent-scoped knowledge retrieval                   |
| Knowledge     | `knowledge.store`, `librarian`, `ingestion`         | Three-tier RAG (axiom/imported/derived), librarian agent        |
| Research      | `research.pool`, `engine`, `composite`, `trigger`   | Managed worker threads, multi-engine parallel search            |
| LLM           | `llm.manager`, `scheduler`, `profiles`, `benchmark` | Score-based inference scheduling, model profiling               |
| Agents        | `agents.activation`, `capabilities`, `registry`     | Activation rules, capability discovery, config persistence      |
| Integrations  | `integrations.mattermost`, `websocket_chat`         | Mattermost bot, WebSocket chat with sessions                    |
| Training      | `training.feedback`, `training.exporter`            | Interaction logging, feedback, JSONL export                     |
| Gateway       | `gateway.gateway`, `messages`, `sessions`           | Redis Streams message bus, agent messaging                      |
| Discovery     | `discovery.mdns`                                    | mDNS service advertising via zeroconf                           |
| Debate        | `debate.orchestrator`                               | Structured agent disagreement resolution                        |

## License

MIT
