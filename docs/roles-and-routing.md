# Roles & Routing

Roles are the core abstraction in khonliang. Each role is a specialized LLM agent with its own system prompt, model, and domain logic. A router dispatches incoming messages to the right role.

## Defining a Role

Subclass `BaseRole` and implement `handle()`:

```python
from khonliang.roles.base import BaseRole

class ResearcherRole(BaseRole):
    def __init__(self, model_pool, tree, **kwargs):
        super().__init__(role="researcher", model_pool=model_pool, **kwargs)
        self.tree = tree
        self._system_prompt = (
            "You are a genealogy research assistant. Answer questions "
            "about family relationships based on the provided data. "
            "If the data doesn't contain the answer, say so."
        )

    def build_context(self, message, context=None):
        """Inject family tree data into the prompt."""
        person = self.tree.find_person(message)
        if person:
            return self.tree.build_context(person.xref, depth=2)
        return self.tree.get_summary()

    async def handle(self, message, session_id, context=None):
        ctx = self.build_context(message, context)
        prompt = f"Family tree data:\n{ctx}\n\nQuestion: {message}\n\nAnswer:"

        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {"role": self.role, "generation_time_ms": elapsed_ms},
        }
```

### Key Methods

| Method                                 | Purpose                                                                     |
| -------------------------------------- | --------------------------------------------------------------------------- |
| `build_context(message, context)`      | Inject live data (DB, API, tree) into the prompt. Called before generation. |
| `handle(message, session_id, context)` | Process a message and return a response dict.                               |
| `_timed_generate(prompt, system)`      | Generate via OllamaClient with timing. Returns `(text, elapsed_ms)`.        |
| `enforce_budget(context)`              | Truncate context to fit token limit (chars/4 heuristic).                    |
| `load_prompt_file(filename, fallback)` | Load system prompt from a file.                                             |

### System Prompts

Set `self._system_prompt` in `__init__()`, or load from a file:

```python
class NarratorRole(BaseRole):
    def __init__(self, model_pool, tree, knowledge_store=None):
        super().__init__(role="narrator", model_pool=model_pool)
        self.tree = tree
        self.knowledge_store = knowledge_store
        self._system_prompt = (
            "You are a genealogy narrator. Present family tree data "
            "in a readable, engaging way.\n\n"
            "STRICT RULES:\n"
            "1. ONLY state facts that appear in the provided data.\n"
            "2. Never fabricate names, dates, places, or occupations.\n"
            "3. If you don't have enough data, say so briefly.\n"
        )
```

### Context Injection

`build_context()` is the injection point for any data the LLM needs. Common patterns:

```python
def build_context(self, message, context=None):
    # Tree data
    tree_ctx = self._search_tree(message)

    # Knowledge store data (previously researched facts)
    knowledge_ctx = ""
    if self.knowledge_store:
        knowledge_ctx = self.knowledge_store.build_context(
            query=message, max_chars=2000, include_axioms=False
        )

    parts = [tree_ctx]
    if knowledge_ctx:
        parts.append(f"\n[KNOWLEDGE]\n{knowledge_ctx}")
    return "\n".join(parts)
```

### Blackboard Integration

If a `board` (Blackboard instance) is passed to the role, its entries are automatically appended to context:

```python
from khonliang.gateway.blackboard import Blackboard

board = Blackboard(default_ttl=120)
role = ResearcherRole(pool, tree=tree, board=board)

# Other agents post to the board
board.post("analyst", "findings", "key_dates", "Roger Tolle born ~1642 in England")

# ResearcherRole.build_context() will include board entries automatically
```

## ModelPool

`ModelPool` maps role names to Ollama model strings and manages client instances:

```python
from khonliang import ModelPool

pool = ModelPool(
    {
        "researcher": "llama3.2:3b",      # Fast for Q&A
        "fact_checker": "qwen2.5:7b",     # Medium for validation
        "narrator": "llama3.1:8b",        # Larger for narratives
    },
    base_url="http://localhost:11434",
    keep_alive={
        "researcher": "30m",   # Keep loaded (frequently used)
        "fact_checker": "5m",  # Unload after 5 min idle
        "narrator": "5m",
    },
)
```

Each role gets its own `OllamaClient` instance via `self.client`, configured for its model.

## Routing

`BaseRouter` evaluates rules in priority order and dispatches to the winning role.

### Keyword Routing

The simplest approach — match keywords in the message:

```python
from khonliang.roles.router import BaseRouter

class GenealogyRouter(BaseRouter):
    def __init__(self):
        super().__init__(fallback_role="researcher")

        self.register_keywords(
            ["check", "validate", "verify", "contradiction", "error"],
            "fact_checker",
        )

        self.register_keywords(
            ["story", "narrative", "tell me about", "biography", "describe"],
            "narrator",
        )

        # Everything else falls through to "researcher"
```

### Regex Routing

For pattern-based matching:

```python
router = BaseRouter(fallback_role="researcher")
router.register_pattern(r"^!check\b", "fact_checker")
router.register_pattern(r"^(tell|write|describe)\b", "narrator")
```

### Callable Routing

For complex logic:

```python
def is_fact_check(message):
    return "check" in message.lower() and "?" in message

router.register_rule(is_fact_check, "fact_checker")
```

### Semantic Routing

Uses FastEmbed cosine similarity for intent detection (requires `pip install ollama-khonliang[rag]`):

```python
from khonliang.routing.semantic import SemanticIntentRouter

semantic = SemanticIntentRouter()
semantic.add_route("researcher", [
    "Who were John's parents?",
    "When was Mary born?",
    "Where did the family come from?",
])
semantic.add_route("narrator", [
    "Tell me the story of the Smith family",
    "Write a biography of John",
])

router.set_semantic_router(semantic)
```

### Route Debugging

```python
role_name, reason = router.route_with_reason("check if Roger's dates are correct")
# role_name = "fact_checker"
# reason = "keyword: check"
```

### Evaluation Priority

Routes are checked in this order:

1. Callable predicates (highest priority)
2. Regex patterns
3. Keyword lists
4. Semantic router (optional)
5. Fallback role (lowest priority)

## Session Context

For multi-turn conversations, `SessionContext` tracks history and entities:

```python
from khonliang.roles.session import SessionContext

ctx = SessionContext(session_id="abc123")
ctx.add_exchange(
    user_message="Who were Timothy's parents?",
    agent_response="Timothy's parents were John and Mary Toll.",
    role="researcher",
)
ctx.add_exchange(
    user_message="Tell me more about his father",
    agent_response="John Toll was born in 1810 in Ohio...",
    role="researcher",
)

# Build context string for LLM injection
context_str = ctx.build_context(max_turns=5)
```

### Async-Safe Session Injection

Use `contextvars` to inject session context safely across concurrent WebSocket sessions:

```python
import contextvars
from khonliang.roles.session import SessionContext

_session_context_var = contextvars.ContextVar("_session_context_var", default="")

# In your chat server, before routing:
session_ctx = SessionContext(session_id=session.session_id)
_session_context_var.set(session_ctx.build_context(max_turns=5))

# In your role's build_context():
def build_context(self, message, context=None):
    tree_ctx = self._search_tree(message)
    session_ctx = _session_context_var.get("")
    if session_ctx:
        tree_ctx = f"{tree_ctx}\n\n[SESSION CONTEXT]\n{session_ctx}"
    return tree_ctx
```

## Self-Evaluation

The evaluator checks LLM responses for quality issues after generation.

### Built-in Rules

```python
from khonliang.roles.evaluator import (
    BaseEvaluator, SpeculationRule, UncertaintyRule
)

evaluator = BaseEvaluator(rules=[
    SpeculationRule(max_phrases=3),  # Flag excessive hedging
    UncertaintyRule(),                # Detect "I don't have info" responses
])

result = evaluator.evaluate(
    response_text, query=user_query, role="researcher"
)

if result.caveat:
    response_text += "\n\n" + result.caveat
print(f"Confidence: {result.confidence:.0%}, Issues: {len(result.issues)}")
```

### Custom Domain Rules

Subclass `EvalRule` for domain-specific checks:

```python
import re

from khonliang.roles.evaluator import EvalRule, EvalIssue

class DateCheckRule(EvalRule):
    """Check date claims against the family tree."""
    name = "date_check"

    def __init__(self, tree):
        self.tree = tree

    def check(self, response, query="", metadata=None):
        issues = []
        # Parse date claims from response text
        date_claims = re.findall(
            r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b[^.]*?"
            r"(?:born|died)\s+(?:in\s+)?(\d{4})",
            response,
        )
        for name, year_str in date_claims:
            person = self.tree.find_person(name)
            if person:
                tree_birth = self._extract_year(person.birth_date)
                if tree_birth and abs(tree_birth - int(year_str)) > 5:
                    issues.append(EvalIssue(
                        rule=self.name,
                        issue_type="date_mismatch",
                        detail=f"Response says {name} born {year_str}, tree says {tree_birth}",
                        severity="high",
                    ))
        return issues

# Combine built-in and custom rules
evaluator = BaseEvaluator(rules=[
    DateCheckRule(tree),
    RelationshipCheckRule(tree),
    SpeculationRule(max_phrases=3),
    UncertaintyRule(),
])
```

### Evaluation Feedback Loop

Wire evaluation into your chat server to auto-trigger research when the agent is uncertain:

```python
evaluation = evaluator.evaluate(response_text, query=content)

if evaluation.caveat:
    resp["content"] = response_text + "\n\n" + evaluation.caveat

# If the agent said "I don't have info", queue background research
for issue in evaluation.issues:
    if issue.issue_type == "uncertainty":
        research_pool.submit(ResearchTask(
            task_type="web_search", query=content, source="self_eval"
        ))
```

## Wiring It All Together

From the genealogy project's `server.py`:

```python
from khonliang import ModelPool
from khonliang.integrations.websocket_chat import ChatServer
from khonliang.knowledge import KnowledgeStore, Librarian

# Models
pool = ModelPool(
    config["ollama"]["models"],
    base_url=config["ollama"]["url"],
    keep_alive={"researcher": "30m", "fact_checker": "5m", "narrator": "5m"},
)

# Knowledge
store = KnowledgeStore(config["app"]["knowledge_db"])
librarian = Librarian(store)
librarian.set_axiom("identity", "You are a genealogy research assistant.")
librarian.set_axiom("no_fabrication", "Never fabricate names, dates, or places.")

# Roles
roles = {
    "researcher": ResearcherRole(pool, tree=tree),
    "fact_checker": FactCheckerRole(pool, tree=tree),
    "narrator": NarratorRole(pool, tree=tree, knowledge_store=store),
}

# Router
router = GenealogyRouter()

# Evaluator
evaluator = create_genealogy_evaluator(tree)

# Server
server = ChatServer(
    roles=roles,
    router=router,
    librarian=librarian,
)
await server.start(host="0.0.0.0", port=8765)
```
