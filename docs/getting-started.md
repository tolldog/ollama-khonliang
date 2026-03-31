# Getting Started

khonliang is a Python library for building multi-role LLM applications on top of Ollama (local inference). It provides role-based dispatching, routing, consensus voting, scoped RAG, knowledge management, research orchestration, and more.

## Installation

```bash
# Core only
pip install -e .

# With optional features
pip install -e ".[rag]"            # Semantic routing + embeddings
pip install -e ".[mattermost]"     # Mattermost bot integration
pip install -e ".[gateway]"        # Redis agent message bus
pip install -e ".[discovery]"      # mDNS service discovery
pip install -e ".[all]"            # Everything
pip install -e ".[dev]"            # Development tools (pytest, ruff, mypy)
```

Requires a running [Ollama](https://ollama.ai) instance (default: `http://localhost:11434`).

## Quick Example

The simplest possible agent: one role, one model, keyword routing.

```python
import asyncio
from khonliang import OllamaClient, ModelPool, BaseRole, BaseRouter


class GreeterRole(BaseRole):
    """A role that greets users."""

    def __init__(self, model_pool):
        super().__init__(role="greeter", model_pool=model_pool)
        self._system_prompt = "You are a friendly greeter. Say hello warmly."

    async def handle(self, message, session_id, context=None):
        response, elapsed_ms = await self._timed_generate(
            prompt=message, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {"role": self.role, "generation_time_ms": elapsed_ms},
        }


async def main():
    pool = ModelPool(
        {"greeter": "llama3.2:3b"},
        base_url="http://localhost:11434",
    )

    role = GreeterRole(pool)
    result = await role.handle("Hello!", session_id="demo")
    print(result["response"])


asyncio.run(main())
```

## Real-World Example: Genealogy Agent

The [khonliang-genealogy-example](https://github.com/tolldog/khonliang-genealogy-example) project demonstrates most khonliang features. It's an LLM-backed genealogy research tool that uses:

- **3 roles** — researcher, fact checker, narrator
- **Keyword routing** to dispatch queries to the right role
- **Intent classification** for natural language understanding
- **Research pool** with web search + WikiTree + Geni.com APIs
- **Three-tier knowledge management** with librarian agent
- **Self-evaluation** checking LLM responses against GEDCOM tree data
- **Session context** for multi-turn conversation coherence

### Project Structure

```text
genealogy_agent/
  server.py           # WebSocket chat server (extends ChatServer)
  roles.py            # ResearcherRole, FactCheckerRole, NarratorRole
  router.py           # GenealogyRouter (extends BaseRouter)
  researchers.py      # WebSearchResearcher, TreeResearcher
  self_eval.py        # DateCheckRule, RelationshipCheckRule
  chat_handler.py     # ! command handling
  intent.py           # LLM-based intent classification
  gedcom_parser.py    # GEDCOM 5.5 family tree parser
```

## Core Concepts

### Roles

A **role** is a specialized LLM agent. It has a system prompt, access to a model via `ModelPool`, and a `handle()` method that processes messages. Roles inject domain-specific context via `build_context()`.

See: [Roles & Routing](roles-and-routing.md)

### Routing

A **router** decides which role handles each message. Routes are evaluated in priority order: callable predicates, regex patterns, keyword lists, semantic similarity, then fallback.

See: [Roles & Routing](roles-and-routing.md)

### Knowledge

A three-tier knowledge system: axioms (always-on rules), imported (user-provided), and derived (auto-indexed from interactions). The librarian agent manages promotion, pruning, and context assembly.

See: [Knowledge System](knowledge.md)

### Research

A background research pool with pluggable data sources. Engines fetch data in parallel, results are deduplicated and auto-indexed into knowledge.

See: [Research Pool](research.md)

### Consensus

Multiple agents analyze the same question independently, then vote. A consensus engine aggregates votes with weighted scoring and optional veto.

See: [Consensus & Voting](consensus.md)

## Architecture Overview

```text
User Input
  → Router (predicate → regex → keyword → semantic → fallback)
  → Role.build_context() (inject tree data, knowledge, session history)
  → OllamaClient.generate() (local LLM inference)
  → Evaluator (check response quality)
  → Librarian (auto-index response as Tier 3 knowledge)
  → Research Pool (background research if uncertainty detected)
```

## Documentation Index

| Guide                                     | Description                                                |
| ----------------------------------------- | ---------------------------------------------------------- |
| [Roles & Routing](roles-and-routing.md)   | Defining roles, routing rules, session context, evaluation |
| [Knowledge System](knowledge.md)          | Three-tier store, librarian, triples, reports              |
| [Research Pool](research.md)              | Engines, researchers, triggers, parallel search            |
| [LLM Manager](llm-manager.md)             | Score-based scheduling, VRAM management, model profiles    |
| [Consensus & Voting](consensus.md)        | Multi-agent teams, weighted voting, debate                 |
| [Gateway & Agents](gateway-and-agents.md) | Message bus, blackboard, channels, activation              |
| [Training & Feedback](training.md)        | Interaction logging, feedback, heuristic extraction        |
| [Parsing](parsing.md)                     | Structured JSON extraction, query parsing                  |
