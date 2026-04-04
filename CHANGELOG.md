# Changelog

## [0.5.0] - 2026-04-03

### Added

- **KH-10: Triple predicate normalization** — `TripleStore` auto-normalizes predicates on add/get/remove (lowercase, underscored, prefix stripping). New `predicate_aliases` constructor param for domain-specific synonyms. `normalize_predicate()` exported from `khonliang.knowledge`.

- **KH-11: KnowledgeStore status field** — `EntryStatus` class with workflow states (active, ingested, processing, distilled, failed, skipped, archived). New `status` field on `KnowledgeEntry`, `get_by_status()` and `set_status()` methods. Auto-migrates existing DBs via `ALTER TABLE`.

### Changed

- **KH-13: Constrained JSON default** — `generate_json()` now defaults to `constrained=True` (Ollama native JSON mode) on both `OllamaClient` and `OpenAIClient`. Falls back to cleanup parsing on failure. Significantly reduces JSON generation failures on small models.

## [0.4.0] - 2026-03-31

### Added

- **Reporting module** — agent report persistence and HTTP serving pipeline (#51)
  - `ReportManager` — SQLite CRUD with metadata, view tracking, TTL-based expiration
  - `ReportDetector` — pluggable heuristics (length, structure, keywords, custom criteria) for detecting report-worthy content
  - `ReportServer` — Flask app serving reports as styled HTML with JSON API endpoints
  - `ReportTheme` — CSS variable-based theming with logo, colors, fonts, footer, custom CSS; loadable from JSON config
  - HTML sanitization via `nh3` (Rust-backed) to prevent XSS in rendered reports
  - Static file serving for logos and assets
  - Chat context support for bi-directional linkback to chat integrations (e.g. Mattermost permalink)
- **Digest module** — activity accumulation and narrative synthesis (#51)
  - `DigestStore` — SQLite-backed transaction log with `audience` field for separate digest streams
  - `DigestSynthesizer` — LLM-backed narrative generation with configurable `DigestConfig` prompts and structured fallback
  - `DigestConfig` — per-application synthesis prompt, title template, grouping, and entry limit configuration
  - Middleware: `extract_from_response()` reads `digest`/`digest_audience` from response metadata
  - Middleware: `digest_blackboard()` hooks Blackboard posts to auto-record digest entries
  - Middleware: `digest_consensus()` hooks consensus results into the digest stream
- **`[reporting]` optional dependency** — `flask>=3.0`, `markdown>=3.5`, `nh3>=0.2`
- **SECURITY.md** — vulnerability reporting policy and security documentation
- **CONTRIBUTING.md** — development workflow, code style, testing, PR process
- **CODE_OF_CONDUCT.md** — Contributor Covenant v2.1
- **GitHub templates** — bug report, feature request, PR template, dependabot config

### Changed

- CI installs `[reporting]` extras for security test coverage

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
- Project logo and circle icon in `assets/`

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
