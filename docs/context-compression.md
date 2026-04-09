# Context Compression Guide

## Design Principle

> **Local models are cheap. External context is expensive.**

khonliang applications (researcher, developer, genealogy) use local Ollama models to process, distill, and score data. The output flows to external coding agents (Claude, Codex) via MCP tools. Those agents pay per token — every verbose summary, narrative paragraph, or unfiltered list wastes their context window and budget.

The MCP response is the **compression checkpoint**. Before data crosses this boundary, it should be:

1. **Filtered** — only items above a relevance threshold
2. **Sorted** — highest value first
3. **Truncated** — within a declared budget
4. **Structured** — typed artifacts, not prose

## Budget Framework

### ContextBudget

Declare the output ceiling for a tool:

```python
from khonliang.mcp.budget import ContextBudget, fit_to_budget

budget = ContextBudget(
    max_tokens=500,        # rough ceiling (chars / 4)
    max_items=10,          # max list items
    max_preview_chars=80,  # per-item preview length
    priority_field="score" # sort by this field descending
)
```

### fit_to_budget

Apply the budget to a list of dicts:

```python
raw_items = [
    {"name": "concept_a", "score": 0.9, "description": "A long description..."},
    {"name": "concept_b", "score": 0.3, "description": "Another long one..."},
    # ... many more
]

trimmed = fit_to_budget(raw_items, budget)
# Returns top 10 by score, descriptions truncated to 80 chars
```

### Presets

Three presets cover common cases:

| Preset           | Items | Preview   | Use Case                    |
| ---------------- | ----- | --------- | --------------------------- |
| `BUDGET_COMPACT` | 5     | 40 chars  | Agent loops, compact mode   |
| `BUDGET_BRIEF`   | 10    | 80 chars  | Default brief mode          |
| `BUDGET_FULL`    | 25    | 200 chars | Human-requested full detail |

## Compressed Artifact Types

### CompactConcept

A concept scored for project relevance:

```python
from khonliang.mcp.artifacts import CompactConcept

concept = CompactConcept(
    name="reinforcement_learning",
    relevance=0.82,
    paper_count=5,
    top_paper="Policy Gradient Methods for RL",
    actionable=True,
)

concept.to_compact()
# "reinforcement_learning|0.82|5|Policy Gradient Methods for RL|yes"

concept.to_brief()
# "reinforcement_learning (82%) — 5 papers, top: Policy Gradient Methods for RL [actionable]"
```

### CompactFR

A feature request in compressed form:

```python
from khonliang.mcp.artifacts import CompactFR

fr = CompactFR(
    id="fr_khonliang_abc123",
    title="Add embedding cache",
    priority="high",
    target="khonliang",
    concept="embedding optimization",
    depends_on=["fr_khonliang_def456"],
)

fr.to_compact()
# "fr_khonliang_abc123|high|khonliang|Add embedding cache|deps=fr_khonliang_def456"

fr.to_brief()
# "fr_khonliang_abc123 [high] Add embedding cache -> khonliang [embedding optimization] (blocks: fr_khonliang_def456)"
```

### CompactSynthesis

A topic synthesis for agent consumption:

```python
from khonliang.mcp.artifacts import CompactSynthesis

synthesis = CompactSynthesis(
    topic="token optimization",
    paper_count=3,
    key_findings=["80% cost reduction via caching", "Compact prompts outperform verbose"],
    relevance={"developer": 0.4, "khonliang": 0.9},
    suggested_frs=["fr_khonliang_abc123"],
)

synthesis.to_compact()
# "token optimization|3|80% cost reduction via caching; Compact prompts outperform verbose|rel=developer:0.40,khonliang:0.90|frs=fr_khonliang_abc123"
```

All artifact types support `from_dict()` for construction from raw data (parsed JSON, database rows, etc.).

## Post-Compression

### compress_for_agent (async)

Uses a local model to compress raw text into a structured artifact:

```python
from khonliang.mcp.compress import compress_for_agent
from khonliang.mcp.artifacts import CompactSynthesis

artifact = await compress_for_agent(
    raw_text=long_synthesis_output,
    artifact_type=CompactSynthesis,
    model="llama3.2:3b",  # cheap, fast
)
return artifact.to_compact()
```

Strategy:

1. Try direct JSON parse (zero cost)
2. Invoke local 3b model to extract structured fields
3. Fall back to rule-based heuristics on error

### compress_rule_based (sync)

No model call — pure heuristic extraction:

```python
from khonliang.mcp.compress import compress_rule_based

artifact = compress_rule_based(raw_text, CompactSynthesis)
```

Use when you can't await or the model is unavailable.

## Integrating with MCP Tools

Typical pattern for a budget-aware MCP tool:

```python
from khonliang.mcp.compact import format_response, compact_list
from khonliang.mcp.budget import ContextBudget, fit_to_budget, BUDGET_COMPACT, BUDGET_BRIEF
from khonliang.mcp.artifacts import CompactConcept

@mcp.tool()
def concepts_for_project(project: str, detail: str = "compact") -> str:
    """Show concepts most relevant to a project."""
    raw = fetch_concepts(project)  # list of dicts

    budget = BUDGET_COMPACT if detail == "compact" else BUDGET_BRIEF
    trimmed = fit_to_budget(raw, budget)
    concepts = [CompactConcept.from_dict(c) for c in trimmed]

    return format_response(
        compact_fn=lambda: "\n".join(c.to_compact() for c in concepts),
        brief_fn=lambda: "\n".join(c.to_brief() for c in concepts),
        full_fn=lambda: render_full_concepts(concepts),
        detail=detail,
    )
```

## Migrating from brief_or_full()

`brief_or_full()` is deprecated. Migration is straightforward:

```python
# Before
return brief_or_full(
    brief_fn=lambda: brief_output,
    full_fn=lambda: full_output,
    detail=detail,
)

# After
return format_response(
    compact_fn=lambda: compact_output,  # NEW: add compact mode
    brief_fn=lambda: brief_output,
    full_fn=lambda: full_output,
    detail=detail,
)
```

The key addition is `compact_fn` — the densest format for agent control loops.
