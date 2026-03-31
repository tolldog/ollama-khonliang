# Research Pool

The research system provides background data acquisition from multiple sources in parallel. It's built around three abstractions: **engines** (single data sources), **researchers** (task processors that use engines), and the **pool** (queue manager that dispatches tasks to researchers).

## Architecture

```text
ResearchPool
  ├── WebSearchResearcher (capabilities: person_lookup, web_search, historical_context)
  │     ├── DDG engine
  │     ├── Google engine
  │     ├── Bing engine
  │     ├── WikiTree engine
  │     └── Geni engine
  ├── TreeResearcher (capabilities: tree_lookup, tree_ancestors, tree_migration)
  └── Librarian (auto-indexes completed results into Tier 3)
```

## Engines

A `BaseEngine` wraps a single data source (search API, database, REST service). Engines handle rate limiting, timeouts, and thread pool execution for blocking I/O.

```python
from khonliang.research.engine import BaseEngine, EngineResult

class WikiTreeEngine(BaseEngine):
    name = "wikitree"
    max_threads = 2
    rate_limit = 1.0    # Max 1 request/second
    timeout = 10.0      # 10s per request

    def __init__(self, app_id="my-app"):
        super().__init__()
        self.client = WikiTreeClient(app_id=app_id)

    async def execute(self, query, **kwargs):
        """Fetch results from WikiTree API."""
        # run_sync() runs blocking code in the thread pool
        profiles = await self.run_sync(self.client.search, query, limit=5)

        return [
            EngineResult(
                title=p.get("Name", ""),
                content=self._format_profile(p),
                url=f"https://www.wikitree.com/wiki/{p.get('Name', '')}",
                source="wikitree",
                score=0.8,
            )
            for p in profiles
        ]
```

### Engine Lifecycle

Engines manage a thread pool for blocking I/O:

```python
engine = WikiTreeEngine()
engine.start()   # Initialize thread pool

results = await engine.query("Roger Tolle")  # Rate-limited, with timeout

stats = engine.get_stats()
# {"requests": 12, "errors": 1, "avg_ms": 450}

engine.stop()    # Shutdown thread pool
```

### HTTP Engine

For external services that expose a REST API:

```python
from khonliang.research.http_engine import HttpEngine

engine = HttpEngine(
    name="ancestry_service",
    base_url="http://localhost:8080",
    endpoint="/analyze",
    health_endpoint="/health",
    max_threads=4,
    timeout=15.0,
)

# Engine POSTs to http://localhost:8080/analyze
# with body: {"query": "...", "options": {...}}
# expects: {"results": [{"title": ..., "content": ..., "url": ...}]}
```

## Researchers

A `BaseResearcher` processes research tasks. It declares its capabilities (task types it can handle) and implements a `research()` method.

### Simple Researcher

```python
from khonliang.research.base import BaseResearcher
from khonliang.research.models import ResearchResult, ResearchTask

class TreeResearcher(BaseResearcher):
    name = "tree_lookup"
    capabilities = ["tree_lookup", "tree_ancestors", "tree_migration"]
    max_concurrent = 5  # Local lookups are fast

    def __init__(self, tree):
        self.tree = tree

    async def research(self, task):
        if task.task_type == "tree_lookup":
            person = self.tree.find_person(task.query)
            if person:
                ctx = self.tree.build_context(person.xref, depth=2)
                return ResearchResult(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    title=f"Tree data: {person.full_name}",
                    content=ctx,
                    confidence=1.0,
                    sources=["gedcom"],
                )
        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title="Not found",
            content=f"No tree data found for: {task.query}",
            confidence=0.0,
            sources=[],
        )
```

### Composite Researcher

A `CompositeResearcher` fans out to multiple engines in parallel, deduplicates results, and combines them:

```python
from khonliang.research.composite import CompositeResearcher

class WebSearchResearcher(CompositeResearcher):
    name = "web_search"
    capabilities = ["person_lookup", "web_search", "historical_context"]
    max_concurrent = 3

    def __init__(self, tree=None):
        super().__init__()
        self.tree = tree

        # Register engines — all queried in parallel
        self.add_engine(DDGEngine())
        self.add_engine(GoogleEngine())
        self.add_engine(BingEngine())
        self.add_engine(WikiTreeEngine())
        self.add_engine(GeniEngine())

        # Optional post-collection filter
        self.set_filter(self._relevance_filter)

    def build_queries(self, task):
        """Customize query generation per task type."""
        if task.task_type == "person_lookup" and self.tree:
            person = self.tree.find_person(task.query)
            if person and person.birth_year:
                return [
                    f"{task.query} genealogy {person.birth_year}",
                    f"{task.query} family history {person.birth_place or ''}",
                ]
        return [f"{task.query} genealogy"]

    def _relevance_filter(self, results, task):
        """Filter out irrelevant results (e.g., serial killers matching a surname)."""
        return [r for r in results if self._is_relevant(r, task)]

    def start_engines(self):
        """Start all engine thread pools."""
        super().start_engines()

    def stop_engines(self):
        """Stop all engine thread pools."""
        super().stop_engines()
```

## Research Pool

The pool manages a queue of research tasks, dispatches them to capable researchers, and optionally auto-indexes results into the knowledge store.

```python
from khonliang.research import ResearchPool

pool = ResearchPool(max_queue_size=100)

# Register researchers
pool.register(WebSearchResearcher(tree=tree))
pool.register(TreeResearcher(tree=tree))

# Auto-index results into knowledge
pool.set_librarian(librarian)

# Start worker threads
pool.start(workers=2)

# Submit tasks
task_id = pool.submit(ResearchTask(
    task_type="person_lookup",
    query="Roger Tolle",
    scope="genealogy",
    source="user",
    priority=0,
))

# Or use the convenience wrapper
task_id = pool.submit_lookup(
    query="Roger Tolle",
    task_type="person_lookup",
    scope="genealogy",
    source="user",
)

# Check result (non-blocking)
result = pool.get_result(task_id)

# Get all recent results
results = pool.get_all_results(limit=20)

# Status
status = pool.get_status()
# {"queued": 3, "active": 1, "completed": 47, "failed": 2, "researchers": [...]}

# Cleanup
pool.stop()
```

### Task Priority

Higher priority tasks are processed first. Use negative priority for background/speculative research:

```python
# User-requested (high priority)
pool.submit(ResearchTask(query="Roger Tolle", priority=5, source="user"))

# Auto-triggered from evaluation (low priority)
pool.submit(ResearchTask(query="Roger Tolle", priority=-2, source="self_eval"))
```

### Result Callbacks

Register callbacks to act on completed research:

```python
def on_result(result):
    if result.confidence > 0.7:
        logger.info(f"High-confidence result: {result.title}")

pool.on_result(on_result)
```

## Research Triggers

`ResearchTrigger` maps user commands and agent responses to research tasks.

### Prefix Triggers

Map `!command` prefixes to task types:

```python
from khonliang.research import ResearchTrigger

trigger = ResearchTrigger(pool)
trigger.add_prefix("!lookup", "person_lookup")
trigger.add_prefix("!search", "web_search")
trigger.add_prefix("!find", "web_search")
trigger.add_prefix("!history", "historical_context")
trigger.add_prefix("!ancestors", "tree_ancestors")
trigger.add_prefix("!tree", "tree_lookup")

# Check a user message
task_ids = trigger.check_message("!lookup Roger Tolle", scope="genealogy")
# Automatically submits a person_lookup task and returns the task ID
```

### Implicit Triggers

Detect when an agent's response suggests missing information and auto-queue research:

```python
trigger.add_implicit(
    r"I don't have (?:enough )?information",
    "web_search",
)
trigger.add_implicit(
    r"(?:no|not) (?:available|found) (?:in|from) the (?:tree|data)",
    "web_search",
)

# After an agent responds:
implicit_tasks = trigger.check_response(
    response=agent_response,
    original_query=user_message,
    scope="genealogy",
)
if implicit_tasks:
    logger.info(f"Auto-queued {len(implicit_tasks)} research tasks")
```

## Chat Handler Integration

The genealogy project wraps the research pool in a chat handler that intercepts `!` commands:

```python
from khonliang.research import ResearchPool, ResearchTrigger

class ResearchChatHandler:
    COMMANDS = {
        "!lookup", "!search", "!find", "!history",
        "!ancestors", "!migration", "!tree",
        "!researchwho", "!gaps", "!dead-ends", "!anomalies",
        "!report", "!session",
        "!ingest", "!ingest-file",
        "!knowledge", "!prune", "!promote", "!demote", "!axiom",
    }

    def __init__(self, pool, trigger, librarian=None, tree=None):
        self.pool = pool
        self.trigger = trigger
        self.librarian = librarian
        self.tree = tree

    def is_command(self, message):
        return any(message.lower().startswith(cmd) for cmd in self.COMMANDS)

    async def handle(self, message, scope="global", source="user"):
        msg_lower = message.lower().strip()
        # Route to the right handler based on prefix
        if msg_lower.startswith("!lookup"):
            return await self._handle_research(message, "person_lookup")
        elif msg_lower.startswith("!search"):
            return await self._handle_research(message, "web_search")
        # ... etc
```

## Full Wiring Example

From the genealogy project's `server.py`:

```python
# Research pool
research_pool = ResearchPool()
research_pool.register(WebSearchResearcher(
    tree=tree,
    geni_api_key=os.environ.get("GENI_API_KEY", ""),
    geni_api_secret=os.environ.get("GENI_API_SECRET", ""),
))
research_pool.register(TreeResearcher(tree=tree))
research_pool.set_librarian(librarian)

# Triggers
trigger = ResearchTrigger(research_pool)
trigger.add_prefix("!lookup", "person_lookup")
trigger.add_prefix("!search", "web_search")
trigger.add_prefix("!ancestors", "tree_ancestors")
trigger.add_prefix("!tree", "tree_lookup")

# Start workers
research_pool.start(workers=2)

# Wire into chat handler
handler = ResearchChatHandler(research_pool, trigger, librarian=librarian, tree=tree)
```
