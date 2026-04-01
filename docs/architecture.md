# Architecture

khonliang is organized as a layered framework where each layer builds on the one below. All modules are optional — applications import only what they need.

## Layer Diagram

```mermaid
flowchart TB
    subgraph Foundation["Foundation"]
        Errors["errors"]
        Protocols["protocols\n(LLMClient)"]
        JsonUtils["_json_utils"]
    end

    subgraph Connection["Connection Layer"]
        Ollama["OllamaClient"]
        OpenAI["OpenAIClient"]
        Health["ModelHealthTracker"]
    end

    subgraph Management["Management Layer"]
        Pool["ModelPool"]
        FlowR["routing/flow"]
        Semantic["routing/semantic"]
        ModelR["routing/model_router"]
    end

    subgraph AgentLayer["Agent & Role Layer"]
        BaseRole["BaseRole"]
        BaseRouter["BaseRouter\n(5-stage)"]
        Evaluator["BaseEvaluator"]
        Session["SessionContext"]
        Personalities["PersonalityRegistry"]
    end

    subgraph Coordination["Coordination Layer"]
        Gateway["AgentGateway\n(Redis Streams)"]
        Blackboard["Blackboard\n(in-memory K-V)"]
        Consensus["ConsensusEngine\n+ AgentTeam"]
        Debate["DebateOrchestrator"]
    end

    subgraph Knowledge["Knowledge & Retrieval"]
        KStore["KnowledgeStore\n(3-tier)"]
        Triples["TripleStore"]
        Librarian["Librarian"]
        RAG["DocumentRetriever\n+ ScopedRetriever"]
    end

    subgraph Features["Feature Modules"]
        Reporting["reporting/\nReportManager\nReportServer"]
        Digest["digest/\nDigestStore\nDigestSynthesizer"]
        Research["research/\nResearchPool\nCompositeResearcher"]
        Training["training/\nFeedbackStore\nHeuristicPool"]
        Parsing["parsing/\nStructuredBlock\nQueryParser"]
        LLM["llm/\nLLMManager\nModelScheduler"]
    end

    subgraph Integration["Integration Layer"]
        MCP["MCP Server"]
        Mattermost["Mattermost Bot"]
        WSChat["WebSocket ChatServer"]
        Discovery["mDNS Discovery"]
    end

    Foundation --> Connection
    Connection --> Management
    Management --> AgentLayer
    AgentLayer --> Coordination
    Coordination --> Knowledge
    Knowledge --> Features
    Features --> Integration

    Blackboard -.->|context| BaseRole
    RAG -.->|context| BaseRole
    Digest -.->|middleware| Blackboard
    Digest -.->|middleware| Consensus
```

## Request Flow

```mermaid
flowchart TD
    Msg["User Message"]

    Msg --> Router["BaseRouter\n(5-stage routing)"]
    Router -->|"callable → regex → keyword\n→ semantic → fallback"| Role["BaseRole.handle()"]
    Role --> Context["build_context()\nRAG + Blackboard + Knowledge"]
    Context --> LLMCall["LLMClient.generate()"]
    LLMCall --> Response["Response"]

    Response -.->|"metadata.digest"| DigestS["DigestStore.record()"]
    Response -.->|"if report-worthy"| ReportS["ReportManager.create()"]
    Response -.->|"log interaction"| FeedbackS["FeedbackStore"]

    Msg -->|"high-stakes query"| Team["AgentTeam\n(parallel agents)"]
    Team --> CE["ConsensusEngine\n(weighted votes)"]
    CE -->|"disagreement"| DO["DebateOrchestrator"]
    CE --> Response
```

## Model Routing

```mermaid
flowchart LR
    subgraph Strategies["Model Selection Strategies"]
        direction TB
        Static["StaticStrategy\n(always first model)"]
        Complexity["ComplexityStrategy\n(LLM classifies difficulty)"]
        Cascade["CascadeStrategy\n(cheapest first, escalate)"]
    end

    Query["Query"] --> MR["ModelRouter"]
    MR --> Strategies
    Strategies --> Health["ModelHealthTracker\n(filter cooled-down)"]
    Health --> Model["Selected Model"]

    Pool["ModelPool"] -->|"mixed backends"| MR
    Pool -->|"ollama://"| OC["OllamaClient"]
    Pool -->|"openai://"| OAC["OpenAIClient"]
    Pool -->|"groq://"| OAC
```

## Knowledge Flow

```mermaid
flowchart LR
    subgraph Tiers["Three-Tier Knowledge"]
        direction TB
        T1["Tier 1: Axioms\n(immutable rules)"]
        T2["Tier 2: Imported\n(user docs, promoted)"]
        T3["Tier 3: Derived\n(from interactions)"]
    end

    Ingest["IngestionPipeline"] --> T3
    User["User / !ingest"] --> T2
    Config["Config / Code"] --> T1

    T3 -->|"auto-promote\n(high confidence)"| T2
    T3 -->|"prune\n(low confidence)"| Gone["Deleted"]

    Lib["Librarian"] --> Ingest
    Lib --> T3
    Lib --> T2

    Search["FTS5 Search"] --> Tiers
    Tiers -->|"build_context()"| Prompt["LLM Prompt"]
```

## Digest Pipeline

```mermaid
flowchart LR
    subgraph Producers["Producers (auto-record)"]
        Resp["Response metadata\n(digest key)"]
        BB["Blackboard posts\n(middleware hook)"]
        CV["Consensus results\n(middleware hook)"]
    end

    Producers --> Store["DigestStore\n(SQLite, by audience)"]

    Store -->|"get_unconsumed()\nor get_since(hours)"| Synth["DigestSynthesizer"]
    Synth -->|"LLM narrative\n(or structured fallback)"| Report["Publish via\nReportManager"]

    Config["DigestConfig\n(per-application\nsynthesis prompt)"] --> Synth
```

## Module Overview

### Foundation

The bottom layer provides error types, the `LLMClient` protocol, and shared utilities. These have no internal dependencies.

| Module        | Purpose                                                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `errors`      | Typed error hierarchy: `LLMError`, `LLMTimeoutError`, `LLMUnavailableError`, `LLMModelNotFoundError`, `LLMRateLimitError`, `LLMCooldownError` |
| `protocols`   | `LLMClient` — runtime-checkable protocol satisfied by both clients                                                                            |
| `_json_utils` | JSON cleanup for LLM outputs (Python-style booleans, trailing commas)                                                                         |

### Connection Layer

Async HTTP clients for LLM inference with retry, streaming, and typed errors.

| Module          | Purpose                                                                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `client`        | `OllamaClient` — async client for Ollama `/api/generate` with exponential backoff (3 attempts), JSON generation, and token-by-token streaming |
| `openai_client` | `OpenAIClient` — async client for any `/v1/chat/completions` endpoint (vLLM, SGLang, Groq, Together AI, Fireworks, OpenRouter, etc.)          |
| `health`        | `ModelHealthTracker` — tracks failures per model, enforces cooldown (default: 3 failures in 300s triggers 60s cooldown)                       |

Both clients implement the `LLMClient` protocol, so application code can swap backends without changes.

### Management Layer

Maps roles to models and routes messages to the right handler.

| Module                 | Purpose                                                                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pool`                 | `ModelPool` — role-to-model mapping with lazy client creation. Supports mixed backends via URI scheme: `"openai://model"`, `"groq://model"`                          |
| `routing/flow`         | `FlowClassifier` — LLM-based intent classification (SAVE/EXECUTE/UPDATE/EXPLAIN/OTHER)                                                                               |
| `routing/semantic`     | `SemanticIntentRouter` — FastEmbed cosine similarity for message-to-route mapping (<5ms per call)                                                                    |
| `routing/model_router` | `ModelRouter` — selects which model handles a request within a role. Three strategies: `StaticStrategy`, `ComplexityStrategy`, `CascadeStrategy` (FrugalGPT pattern) |

### Agent & Role Layer

The abstraction layer for domain-specific agents.

| Module            | Purpose                                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `roles/base`      | `BaseRole` — abstract base class. Subclasses implement `handle()` and optionally `build_context()` for RAG/DB injection before generation |
| `roles/router`    | `BaseRouter` — 5-stage message routing: callable predicates, regex patterns, keyword lists, semantic embeddings, fallback                 |
| `roles/evaluator` | `BaseEvaluator` — pluggable rule system for response quality evaluation                                                                   |
| `roles/session`   | `SessionContext` — per-conversation exchange tracking for multi-turn coherence                                                            |
| `personalities`   | `PersonalityConfig`, `PersonalityRegistry` — named agent personas with voting weights, focus areas, and `@mention` resolution             |

### Coordination Layer

Multi-agent communication, shared state, and decision-making.

| Module               | Purpose                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `gateway/`           | `AgentGateway` — Redis Streams message bus for distributed agents. Fail-open: degrades gracefully if Redis is unavailable            |
| `gateway/blackboard` | `Blackboard` — in-memory key-value store with TTL for agent coordination. Sections, keys, and `build_context()` for prompt injection |
| `consensus/`         | `AgentTeam` runs N agents in parallel, `ConsensusEngine` aggregates via weighted scoring. `VETO` overrides all votes                 |
| `debate/`            | `DebateOrchestrator` — structured multi-agent debates with challenge/response rounds when agents disagree                            |

### Knowledge & Retrieval

Persistent knowledge management and document retrieval.

| Module                | Purpose                                                                                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `knowledge/store`     | `KnowledgeStore` — three-tier SQLite store: axioms (immutable rules), imported (user docs), derived (from interactions). Confidence scoring and FTS5 search |
| `knowledge/triples`   | `TripleStore` — semantic subject-predicate-object triples with confidence and time decay                                                                    |
| `knowledge/librarian` | `Librarian` — agent that curates knowledge: ingests content, promotes high-confidence entries, prunes stale ones                                            |
| `rag/retriever`       | `DocumentRetriever` — SQLite FTS5 with BM25 ranking                                                                                                         |
| `rag/scoped`          | `ScopedRetriever` — per-agent knowledge scopes: GLOBAL, DOMAIN, CONVERSATIONAL, EXPERT                                                                      |

### Feature Modules

Higher-level features built on the layers above.

| Module       | Purpose                                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `reporting/` | Report persistence (SQLite), detection (pluggable heuristics), HTTP serving (Flask), and theming. HTML sanitized via `nh3`                                               |
| `digest/`    | Activity accumulation (SQLite transaction log with audience tagging), LLM-backed narrative synthesis, middleware hooks for Blackboard/consensus/response metadata        |
| `research/`  | `ResearchPool` with managed workers, `CompositeResearcher` for parallel multi-source search, `HttpEngine` for external APIs, `ResearchTrigger` for implicit research     |
| `training/`  | `FeedbackStore` for RLHF-style interaction logging, `TrainingExporter` for fine-tuning datasets (alpaca/sharegpt/completion), `HeuristicPool` for outcome-based patterns |
| `parsing/`   | `StructuredBlockParser` extracts typed JSON from LLM markdown. `QueryParser` uses LLM-backed structured extraction for natural language queries                          |
| `llm/`       | `LLMManager` with pluggable backends, `ModelScheduler` (score-based VRAM-aware scheduling), `ModelProfile` for per-model preferences, `ModelBenchmark` for validation    |

### Integration Layer

Bridges to external systems.

| Module                        | Purpose                                                                                                                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mcp/`                        | `KhonliangMCPServer` — exposes up to 11 tools and 3 resources to external LLMs via Model Context Protocol. Actual count depends on which components are provided. Transports: stdio and streamable HTTP |
| `integrations/mattermost`     | `MattermostBot` — WebSocket connection with `on_mention` / `on_direct_message` handlers                                                                                                                 |
| `integrations/websocket_chat` | `ChatServer` — WebSocket chat with session tracking, role routing, and knowledge indexing                                                                                                               |
| `discovery/`                  | `ServiceAdvertiser` — mDNS service advertising and discovery via zeroconf                                                                                                                               |

## Storage

All persistent stores use SQLite. `ReportManager` and `DigestStore` enable WAL mode explicitly; other stores use SQLite defaults:

| Store               | Database     | Purpose                                |
| ------------------- | ------------ | -------------------------------------- |
| `KnowledgeStore`    | configurable | Three-tier knowledge with FTS5 search  |
| `TripleStore`       | configurable | Semantic triples with confidence       |
| `DocumentRetriever` | configurable | FTS5 document retrieval                |
| `ReportManager`     | `reports.db` | Report persistence with TTL (WAL mode) |
| `DigestStore`       | `digest.db`  | Activity transaction log (WAL mode)    |
| `FeedbackStore`     | configurable | Interaction logging for training       |

## Optional Dependencies

The core library requires only `aiohttp` and `requests`. Everything else is optional:

| Extra          | Packages                   | Enables                                   |
| -------------- | -------------------------- | ----------------------------------------- |
| `[rag]`        | fastembed, semantic-router | Semantic intent routing, embeddings       |
| `[mattermost]` | websocket-client           | Mattermost bot integration                |
| `[gateway]`    | redis                      | Redis Streams agent gateway               |
| `[discovery]`  | zeroconf                   | mDNS service discovery                    |
| `[mcp]`        | mcp                        | Model Context Protocol server             |
| `[reporting]`  | flask, markdown, nh3       | Report HTTP serving and HTML sanitization |
| `[all]`        | all of the above           | Everything                                |

## Key Design Patterns

- **Protocol-based abstraction** — `LLMClient` and `RoutingStrategy` are runtime-checkable protocols, not base classes
- **Lazy initialization** — `ModelPool` creates clients on first use; `SemanticIntentRouter` loads embeddings on first call
- **Fail-open coordination** — `AgentGateway` degrades gracefully when Redis is unavailable; `ScopedRetriever` falls back to BM25 when cross-encoder is missing
- **Middleware hooks** — Digest module patches into Blackboard and consensus without modifying those modules
- **Pluggable detection** — `ReportDetector` and `DigestConfig` let applications customize behavior without subclassing
