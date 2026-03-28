# Changelog

## [0.1.0] - 2026-03-27

### Added

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
