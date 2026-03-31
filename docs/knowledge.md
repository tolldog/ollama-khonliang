# Knowledge System

khonliang provides a three-tier knowledge management system. Knowledge entries have different trust levels, and a librarian agent manages promotion, pruning, and context assembly.

## Three Tiers

| Tier | Name         | Trust    | Source                            | Lifetime         |
| ---- | ------------ | -------- | --------------------------------- | ---------------- |
| 1    | **Axiom**    | Absolute | Developer-defined                 | Permanent        |
| 2    | **Imported** | High     | User-provided, promoted           | Long-lived       |
| 3    | **Derived**  | Variable | Auto-indexed from agent responses | Pruned over time |

- **Axioms** are always included in context. They define system rules and identity.
- **Imported** entries come from user ingestion or promotion from Tier 3.
- **Derived** entries are automatically created when agents respond. High-confidence entries can be promoted to Tier 2.

## KnowledgeStore

SQLite-backed store with FTS5 full-text search:

```python
from khonliang.knowledge.store import KnowledgeStore, Tier

store = KnowledgeStore("data/knowledge.db")

# Axioms (Tier 1) — always in context
store.set_axiom("identity", "You are a genealogy research assistant.")
store.set_axiom("no_fabrication", "Never fabricate names, dates, or places.")

# Search across Tier 2 and 3
results = store.search("Roger Tolle", scope="genealogy", limit=5)

# Build context string for LLM injection
ctx = store.build_context(
    query="Roger Tolle migration",
    scope="genealogy",
    max_chars=2000,
    include_axioms=True,
)

# Tier management
store.promote("entry_id_123")  # Tier 3 → Tier 2
store.demote("entry_id_456")   # Tier 2 → Tier 3

# Pruning — remove old, low-confidence derived entries
removed = store.prune(
    tier=Tier.DERIVED,
    max_age_days=30,
    min_confidence=0.3,
    min_access_count=0,
)

# Statistics
stats = store.get_stats()
# {"total": 142, "by_tier": {1: 3, 2: 28, 3: 111}, "by_scope": {"genealogy": 142}}
```

## Librarian

The librarian is the high-level interface for knowledge management. It wraps the store with ingestion pipelines, auto-promotion, conflict detection, and quality assessment.

```python
from khonliang.knowledge import Librarian
from khonliang.knowledge.store import KnowledgeStore

store = KnowledgeStore("data/knowledge.db")
librarian = Librarian(store)

# Set axioms
librarian.set_axiom(
    "cite_sources",
    "Always distinguish between facts from the family tree and your own interpretation."
)

# Ingest text (becomes Tier 2)
result = librarian.ingest_text(
    content="Roger Tolle emigrated from England to Connecticut around 1642.",
    title="Roger Tolle migration",
    scope="genealogy",
)
print(f"Added: {result.added}, Updated: {result.updated}")

# Ingest a file
result = librarian.ingest_file("research/tolle_history.txt", scope="genealogy")

# Ingest a directory of research notes
result = librarian.ingest_directory(
    "research/", scope="genealogy", patterns=["*.txt", "*.md"]
)

# Auto-index an agent response (becomes Tier 3)
result = librarian.index_response(
    content="Based on the GEDCOM data, Timothy Toll was born in 1842 in Ohio.",
    title="Response to: when was Timothy born?",
    agent_id="researcher",
    query="when was Timothy born?",
    scope="genealogy",
    confidence=0.8,
)

# Build context for a role
ctx = librarian.build_context(query="Roger Tolle", scope="genealogy")

# Auto-promote proven Tier 3 entries (high confidence + frequent access)
promoted = librarian.auto_promote()

# Detect conflicting entries
conflicts = librarian.find_conflicts(scope="genealogy")

# Prune low-quality derived entries
pruned = librarian.prune()

# Status overview
status = librarian.get_status()
```

### Wiring into the Chat Server

In the genealogy project, the librarian auto-indexes every agent response:

```python
# In ChatServer._handle_chat():
result = await role.handle(content, session_id=session.session_id)
response_text = result.get("response", "")

# Auto-index the response as Tier 3 knowledge
if librarian and response_text:
    librarian.index_response(
        content=response_text,
        title=f"Response to: {content[:60]}",
        agent_id=role_name,
        query=content,
    )
```

## Ingestion Pipeline

For bulk ingestion with chunking, summarization, and deduplication:

```python
from khonliang.knowledge.ingestion import IngestionPipeline
from khonliang.knowledge.store import Tier

pipeline = IngestionPipeline(
    store=store,
    chunk_size=500,              # Split large text into 500-char chunks
    summarize_threshold=1000,    # Summarize chunks over 1000 chars
    summarizer=my_summarize_fn,  # Optional LLM summarizer
    classifier=my_classify_fn,   # Optional scope classifier
)

# Ingest a large document
result = pipeline.ingest_file(
    "research/tolle_history.pdf",
    scope="genealogy",
    tier=Tier.IMPORTED,
    confidence=0.9,
)
print(f"Added {result.added} entries, skipped {result.skipped} duplicates")
```

## Triple Store

For compact, token-efficient knowledge representation. Each triple is a subject-predicate-object relationship:

```python
from khonliang.knowledge.triples import TripleStore

triples = TripleStore("data/knowledge.db")

# Add triples
triples.add("Roger Tolle", "born_in", "England", confidence=0.8, source="web_search")
triples.add("Roger Tolle", "emigrated_to", "Connecticut", confidence=0.9, source="wikitree")
triples.add("Roger Tolle", "born_year", "~1642", confidence=0.7, source="web_search")

# Duplicate triples reinforce confidence rather than creating duplicates
triples.add("Roger Tolle", "born_in", "England", confidence=0.9, source="geni")
# Now confidence is higher

# Query
results = triples.get(subject="Roger Tolle")
results = triples.get(predicate="born_in", obj="England")
results = triples.search("Tolle", limit=20)

# Build compact context for LLM (one line per triple)
ctx = triples.build_context(
    subjects=["Roger Tolle", "Timothy Toll"],
    max_triples=50,
    min_confidence=0.5,
)
# Output:
# Roger Tolle born_in England (90%)
# Roger Tolle emigrated_to Connecticut (90%)
# Roger Tolle born_year ~1642 (70%)
# Timothy Toll born_in Ohio (95%)

# Time-based confidence decay
decayed = triples.apply_decay(max_age_days=90)
```

### Token Efficiency

Triples are ~95% more token-efficient than full-text knowledge entries. A paragraph about Roger Tolle's migration might be 200 tokens as full text, but the same facts as triples use ~10 tokens.

## Reports

The `ReportBuilder` assembles formatted reports from knowledge:

```python
from khonliang.knowledge.reports import ReportBuilder

reports = ReportBuilder(knowledge_store=store)

# Full knowledge state
print(reports.knowledge_report())

# Topic deep-dive
print(reports.topic_report("Roger Tolle", scope="genealogy"))

# Session startup summary
print(reports.session_report(extra_context="User is researching Tolle family origins"))
```

In the genealogy project, reports are extended with domain-specific types:

```python
# Person dossier from tree + knowledge
!report Timothy Toll

# Gap analysis — what's missing from the tree
!report gaps Timothy Toll

# Session summary — what was discussed
!session
```

## Knowledge in Context Assembly

When a role's `build_context()` runs, it can pull from multiple knowledge sources:

```python
def build_context(self, message, context=None):
    # 1. Domain data (GEDCOM tree)
    tree_ctx = self.tree.build_context(person.xref, depth=2)

    # 2. Knowledge store (Tier 2+3 entries matching the query)
    knowledge_ctx = self.knowledge_store.build_context(
        query=message, max_chars=2000, include_axioms=False
    )

    # 3. Session history (multi-turn coherence)
    session_ctx = _session_context_var.get("")

    parts = [tree_ctx]
    if knowledge_ctx:
        parts.append(f"\n[KNOWLEDGE]\n{knowledge_ctx}")
    if session_ctx:
        parts.append(f"\n[SESSION CONTEXT]\n{session_ctx}")
    return "\n".join(parts)
```

Axioms are included automatically by the `ChatServer` via the librarian's `build_context()` method when `include_axioms=True`.
