# MCP Server

khonliang includes an MCP (Model Context Protocol) server that exposes knowledge, triples, blackboard, and roles to external LLMs. This lets Claude, GPT, or any MCP-compatible client interact with the same data stores that local agents use — a shared context layer.

## Quick Start

```bash
# Install with MCP support
pip install ollama-khonliang[mcp]

# Run the MCP server (stdio transport for Claude Code)
python -m khonliang.mcp --transport stdio --db data/knowledge.db

# Or HTTP transport for remote access
python -m khonliang.mcp --transport http --port 8080
```

### Claude Code Integration

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "khonliang": {
      "command": "python",
      "args": [
        "-m",
        "khonliang.mcp",
        "--transport",
        "stdio",
        "--db",
        "data/knowledge.db"
      ]
    }
  }
}
```

## Server Setup

```python
from khonliang.mcp import KhonliangMCPServer
from khonliang.knowledge.store import KnowledgeStore
from khonliang.knowledge.triples import TripleStore
from khonliang.gateway.blackboard import Blackboard

server = KhonliangMCPServer(
    knowledge_store=KnowledgeStore("data/knowledge.db"),
    triple_store=TripleStore("data/knowledge.db"),
    blackboard=Blackboard(),
    # Optional: add roles for invoke_role tool
    # roles={"researcher": researcher_role},
    # router=my_router,
)

# Run with stdio (for Claude Code, IDE extensions)
server.create_app().run(transport="stdio")

# Or HTTP (for remote clients)
server.create_app().run(transport="streamable-http", host="127.0.0.1", port=8080)
```

All components are optional — tools are only registered for components you provide.

## Tools

### Knowledge Tools

| Tool                                          | Description                            |
| --------------------------------------------- | -------------------------------------- |
| `knowledge_search(query, scope, max_results)` | Search the knowledge store             |
| `knowledge_ingest(title, content, scope)`     | Add content as Tier 2 (imported)       |
| `knowledge_context(query, scope, max_chars)`  | Build context string for LLM injection |

### Triple Tools

| Tool                                                         | Description                            |
| ------------------------------------------------------------ | -------------------------------------- |
| `triple_add(subject, predicate, object, confidence, source)` | Add a semantic triple                  |
| `triple_query(subject, predicate, object)`                   | Query triples by any field combination |
| `triple_context(subjects, max_triples)`                      | Compact context from triples           |

### Blackboard Tools

| Tool                                                    | Description                  |
| ------------------------------------------------------- | ---------------------------- |
| `blackboard_post(agent_id, section, key, content, ttl)` | Post to shared blackboard    |
| `blackboard_read(section, key)`                         | Read entries from a section  |
| `blackboard_context(sections, max_entries)`             | Formatted context from board |

### Role Tools

| Tool                                     | Description                                       |
| ---------------------------------------- | ------------------------------------------------- |
| `invoke_role(message, role, session_id)` | Route a message to a role and return the response |
| `get_session_context(max_turns)`         | Get current session state                         |

`invoke_role` uses the configured `BaseRouter` to pick a role if none is specified.

## Resources

| URI                     | Description                                  |
| ----------------------- | -------------------------------------------- |
| `knowledge://axioms`    | All Tier 1 axioms (always-on rules)          |
| `knowledge://stats`     | Knowledge store statistics (JSON)            |
| `blackboard://sections` | Active blackboard sections with entry counts |

## Shared Context

The key value of the MCP server is **shared context**. External LLMs read and write the same stores as local agents:

```text
Local Ollama Roles (researcher, narrator, fact_checker)
  ↕ read/write
Shared Context Layer
  ├── KnowledgeStore (three-tier: axiom/imported/derived)
  ├── TripleStore (compact semantic facts)
  ├── Blackboard (live coordination)
  └── SessionContext (conversation state)
  ↕ read/write (via MCP)
External LLMs (Claude, GPT, etc.)
```

Example workflow:

1. Claude researches a topic using `knowledge_search` and `triple_query`
2. Claude adds findings via `knowledge_ingest` and `triple_add`
3. Local Ollama narrator reads the new knowledge via `build_context()`
4. Narrator generates a narrative enriched by Claude's research

## Extending with Domain Tools

For domain-specific tools, create a subclass or extend the server:

```python
from khonliang.mcp import KhonliangMCPServer

class GenealogyMCPServer(KhonliangMCPServer):
    def __init__(self, tree, **kwargs):
        super().__init__(**kwargs)
        self.tree = tree

    def create_app(self):
        app = super().create_app()

        @app.tool()
        def tree_search(query: str) -> str:
            """Search the family tree for persons matching a query."""
            results = self.tree.search_persons(query)
            return "\n".join(p.display for p in results)

        @app.tool()
        def tree_ancestors(name: str, generations: int = 4) -> str:
            """Get ancestor chain for a person."""
            person = self.tree.find_person(name)
            if not person:
                return f"Person '{name}' not found"
            ancestors = self.tree.get_ancestors(person.xref, generations)
            return "\n".join(a.display for a in ancestors)

        return app
```

## CLI Reference

```text
python -m khonliang.mcp [OPTIONS]

Options:
  --transport {stdio,http}  Transport mode (default: stdio)
  --db PATH                 Knowledge database path (default: data/knowledge.db)
  --host HOST               HTTP host (default: 127.0.0.1)
  --port PORT               HTTP port (default: 8080)
  -v, --verbose             Enable debug logging
```
