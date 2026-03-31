# Changelog

## [0.3.0] - 2026-03-31

### Added

- **Multi-backend LLM support** — `OpenAIClient` for any `/v1/chat/completions` backend (vLLM, SGLang, Groq, Together AI, Fireworks, OpenRouter, Cerebras, LM Studio, llama.cpp, and more) (#47)
- **LLMClient protocol** — runtime-checkable interface for backend abstraction; both `OllamaClient` and `OpenAIClient` satisfy it (#47)
- **Mixed-backend ModelPool** — URI scheme routing (`"openai://model"`, `"groq://model"`) with `backends` config dict (#47)
- **Model routing** — `ModelRouter` with pluggable strategies for intelligent model selection within roles (#48)
  - `StaticStrategy` — always use first candidate (no-op baseline)
  - `ComplexityStrategy` — fast LLM classifies prompt complexity, maps to model tier
  - `CascadeStrategy` — try cheapest model first, escalate on low confidence (FrugalGPT pattern)
- **MCP server** — expose knowledge, triples, blackboard, and roles to external LLMs via Model Context Protocol (#49)
  - 11 tools: knowledge_search, knowledge_ingest, knowledge_context, triple_add, triple_query, triple_context, blackboard_post, blackboard_read, blackboard_context, invoke_role, get_session_context
  - 3 resources: knowledge://axioms, knowledge://stats, blackboard://sections
  - Transports: stdio (Claude Code) and streamable HTTP
- Shared JSON cleanup utilities extracted to `_json_utils.py` (#47)
- Comprehensive documentation: 12 guides covering all features (#46, #50)

### Changed

- `ModelPool.get_client()` return type widened from `OllamaClient` to `LLMClient` (#47)
- `BaseRole.client` property returns `LLMClient` (was `OllamaClient`) (#47)
- `InternalBackend` accepts optional `client: LLMClient` parameter (#47)
- `BaseRole.__init__` accepts optional `model_router` parameter (#48)
- `_timed_generate` accepts optional `model` override (#48)

## [0.2.0] - 2026-03-29

### Features

- **Three-tier knowledge system** — `KnowledgeStore` with axiom/imported/derived tiers, `Librarian` agent, `IngestionPipeline` (#4)
- **Triple store** — semantic subject-predicate-object triples with confidence scoring and time decay (#35)
- **Research pool** — managed worker threads with `BaseEngine`, `CompositeResearcher`, `ResearchTrigger` (#5, #6)
- **HTTP engine** — `HttpEngine` for external REST API data sources (#21)
- **LLM manager** — score-based inference scheduler with `ModelScheduler`, `InternalBackend`, `LLMManager` facade (#7)
- **Model preferences** — route to already-loaded models to avoid swap latency (#9)
- **Pinned models** — pre-load and protect frequently-used models (#9)
- **Model benchmarking** — `ModelBenchmark` for validating model performance (#8)
- **Model profiles** — persist and seed scheduler with performance data (#9)
- **Query parser** — LLM-backed structured extraction from natural language (#10)
- **Self-evaluation** — `BaseEvaluator` with pluggable `EvalRule` system, `SpeculationRule`, `UncertaintyRule` (#21)
- **Session context** — `SessionContext` with exchange tracking and entity extraction (#21)
- **WebSocket chat server** — `ChatServer` with session management, role routing, feedback (#4)
- **Agent gateway** — Redis Streams message bus with `AgentGateway`, `Blackboard`, `Observer` pattern (#21)
- **Agent management** — activation rules, capability registry, typed channels, envelopes (#21)
- **Debate orchestrator** — structured multi-agent debate with challenge/response rounds (#34)
- **Adaptive consensus weights** — `AdaptiveWeightManager` with regime-based multipliers (#34)
- **Action vocabularies** — customizable vote action sets with blocking semantics (#21)
- **Knowledge reports** — `ReportBuilder` for knowledge state and topic reports (#21)
- **Heuristic extraction** — `HeuristicPool` for outcome-based pattern discovery (#35)
- **Context budget enforcement** — `BaseRole.enforce_budget()` with chars/4 heuristic (#35)
- **Blackboard** — shared in-memory key-value store with TTL for agent coordination (#45)
- **Keep-alive management** — static and adaptive per-role keep_alive in `ModelPool` (#45)
- **mDNS discovery** — `ServiceAdvertiser` for local network service advertising (#21)
- **Claude Code GitHub Action** for automated PR assistance (#11)
- Complete docstring coverage for all public API (#36)

## [0.1.0] - 2026-03-27

### Initial Release

- Initial release extracted from autostock trading platform
- Async Ollama client with typed errors, retry, streaming, and JSON generation
- Model pool with role-based mapping and connection reuse
- Model health tracker with cooldown enforcement
- Base role abstract class with context injection
- Message router with callable, regex, keyword, and semantic stages
- Personality system with registry, @mention resolution, and built-in defaults
- Multi-agent consensus engine with weighted voting and VETO blocking
- Agent team orchestrator with parallel execution and vote caching
- Semantic intent router using FastEmbed ONNX embeddings
- Flow classifier for mid-conversation intent detection
- Structured block parser for typed JSON extraction from LLM output
- Document retriever with SQLite FTS5 and BM25 ranking
- Scoped RAG retriever with global, domain, conversational, and expert scopes
- Generic config registry with JSON persistence
- Mattermost bot integration with WebSocket and typing indicators
- Interaction logging and training feedback store
- Training data exporter (alpaca, sharegpt, completion formats)
- CLI with chat, generate, route, test, models, and health commands
