# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

`ollama-khonliang` (import as `khonliang`) is a Python library for building multi-role LLM applications on top of Ollama (local inference). It provides role-based dispatching, personality-based agent configuration, semantic/rule-based routing, multi-agent consensus voting, scoped RAG, structured output parsing, and Mattermost integration.

## Installation & Setup

```bash
pip install -e .                    # Core only (aiohttp, requests)
pip install -e ".[rag]"             # + fastembed + semantic-router
pip install -e ".[mattermost]"      # + websocket-client
pip install -e ".[gateway]"         # + redis
pip install -e ".[discovery]"       # + zeroconf
pip install -e ".[all]"             # Everything
pip install -e ".[dev]"             # + pytest, ruff, mypy
```

Requires a running Ollama instance (default: `http://localhost:11434`).

## Running the CLI

```bash
khonliang models
khonliang health
khonliang chat --role triage --model llama3.2:3b
khonliang generate "Explain async Python" --model llama3.2:3b
khonliang route "My server is down!" --router examples.helpdesk_bot.router:HelpdeskRouter
khonliang test examples/helpdesk_bot/tests.jsonl
```

## Architecture

The library is layered:

**Connection layer** (`client.py`, `pool.py`, `health.py`, `errors.py`):

- `OllamaClient` — async HTTP client for Ollama `/api/generate` with per-model timeouts, exponential backoff retry (3 attempts, 1s->2s->4s), JSON generation with auto-cleanup, typed errors, and token-by-token streaming via `stream_generate()` (async generator).
- `ModelPool` — maps role names to model strings and lazily creates/deduplicates `OllamaClient` instances.
- `ModelHealthTracker` — tracks failures per model and enforces cooldown (default: 3 failures in 300s -> 60s cooldown).

**Role layer** (`roles/`):

- `BaseRole` — abstract base; subclasses implement `handle()` and optionally override `build_context()` to inject live data (DB, API calls) into the prompt before generation.
- `BaseRouter` — hybrid router evaluated in priority order: callable predicates -> regex patterns -> keyword lists -> optional `SemanticIntentRouter` stage -> fallback role. Attach semantic stage via `set_semantic_router()` or the `semantic_router` constructor argument.

**Personality system** (`personalities.py`):

- `PersonalityConfig` — named agent personality with voting weight, focus areas, and an optional custom system prompt.
- `PersonalityRegistry` — load/save/register personalities; resolve `@mention` aliases. Built-in defaults: `resolver`, `analyst`, `advocate`, `skeptic`. Load custom personalities from JSON via `PersonalityRegistry("config/personalities.json")`.
- `get_registry()` — module-level singleton for the default registry.
- `build_prompt(personality_id, question, context)` — assembles a system-prefixed prompt for a personality.
- `extract_mention(text)` — parses `@name` from user messages and resolves to a personality ID.

**Advanced modules** (optional imports):

- `routing/flow.py` — `FlowClassifier` uses a fast LLM to classify conversation intents (SAVE/EXECUTE/UPDATE/EXPLAIN/OTHER).
- `routing/semantic.py` — `SemanticIntentRouter` maps messages to routes via FastEmbed cosine similarity; no GPU needed, <5ms per call. Plugs into `BaseRouter` as stage 4.
- `consensus/` — `AgentTeam` runs N agents in parallel (`asyncio.gather`), collects `AgentVote` objects, and `ConsensusEngine` aggregates via weighted scoring. VETO overrides all.
- `parsing/structured.py` — `StructuredBlockParser` extracts typed JSON blocks from LLM markdown responses (custom fence -> `json` fence -> raw JSON).
- `rag/retriever.py` — `DocumentRetriever` uses SQLite FTS5 + BM25 ranking; `get_relevant_context()` returns a formatted string for prompt injection.
- `rag/scoped.py` — `ScopedRetriever` extends retrieval with per-agent knowledge scopes: `RAGScope.GLOBAL` (shared), `DOMAIN` (agent-specific), `CONVERSATIONAL` (past interactions), `EXPERT` (curated per agent). Agents declare their access via `RAGConfig(scopes=[...], collections=[...])`. Falls back to BM25 order silently when optional cross-encoder reranker is unavailable.
- `integrations/mattermost.py` — `MattermostBot` connects via WebSocket, registers `on_mention` / `on_direct_message` handlers.
- `agents/registry.py` — `ConfigRegistry[T]` — generic typed JSON-backed config persistence; requires `T` to have `to_dict()` and `from_dict()`.

**Additional modules:**

- `gateway/` — Redis Streams agent message bus for distributed communication. Includes `Blackboard` (shared in-memory key-value store with TTL for multi-agent coordination), `session helper functions` management, and `Observer` pattern for event routing.
- `debate/` — `DebateOrchestrator` runs structured multi-agent debates with challenge/response rounds and scoring.
- `discovery/` — mDNS service advertising and discovery via zeroconf for distributed agent networks.
- `knowledge/` — Three-tier knowledge management: `KnowledgeStore` (SQLite-backed with confidence scoring), `Librarian` agent (axiom/imported/derived tiers), `TripleStore` (semantic subject-predicate-object triples), `ReportBuilder` for accumulated knowledge reports.
- `research/` — `ResearchPool` with `BaseEngine` protocol, `CompositeResearcher` for parallel multi-source search, `ResearchTrigger` for implicit research from chat, `HttpEngine` for external API adapters.
- `llm/` — `LLMManager` with pluggable backends (`InternalBackend` using asyncio queues, future gRPC external backend), `ModelScheduler` (score-based VRAM-aware scheduling), `ModelProfile` for per-model preferences, `ModelBenchmark` for performance validation.
- `training/` — `FeedbackStore` for RLHF-style data collection, `TrainingExporter` for fine-tuning dataset preparation, `HeuristicPool` for outcome-based pattern discovery.
- `parsing/query_parser.py` — `QueryParser` uses LLM-backed structured extraction to convert natural language queries into typed filter parameters.

**Public API** (from `khonliang/__init__.py`): `OllamaClient`, `GenerationResult`, `ModelPool`, `ModelHealthTracker`, `BaseRole`, `BaseRouter`, `PersonalityConfig`, `PersonalityRegistry`, and all error types. All other modules must be imported directly.

## Key Design Conventions

- `AgentTeam` accepts any object duck-typed with `agent_id: str` and `async analyze(...) -> AgentVote`.
- `BaseRole.build_context()` is the injection point for RAG, DB lookups, or external API data before LLM calls.
- `OllamaClient.generate_json()` auto-cleans Python-style booleans (`True`/`False`/`None`) and trailing commas before parsing.
- `SemanticIntentRouter` lazy-loads embeddings on first call; subsequent calls are fast.
- `AgentTeam` caches consensus results by key+input hash with configurable TTL (default 300s).
- `PersonalityRegistry` is a module-level singleton via `get_registry()`. Entries loaded from file override defaults with the same id.
- `ScopedRetriever` databases require two additional tables beyond `rag_documents`: `rag_scoped_documents` (with FTS5 virtual table `rag_scoped_fts`) and `agent_conversations` — see `rag/scoped.py` module docstring for DDL.

## Project Origin

This library was extracted from the [autostock](../autostock/) trading platform to provide a generic, domain-agnostic agent orchestration framework. All trading-specific logic remains in autostock; this library provides the reusable infrastructure.
