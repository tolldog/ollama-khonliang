# KH-15: Local-Model Context Compression for External Coding Agents

**FR:** `fr_khonliang_6d0277b6`
**Priority:** high
**Class:** library + app
**Status:** draft

## Problem

khonliang-researcher uses local Ollama models to distill papers, extract triples, score relevance, and generate FRs. The output of that pipeline is consumed by external coding agents (Claude, Codex) via MCP tools. Today, the pipeline produces good artifacts, but the design doesn't treat **external token cost as a first-class constraint**. The result:

1. MCP tool outputs are often narrative-heavy — summaries written for human reading, not agent consumption.
2. Distillation produces structured data (triples, scores, FRs) but synthesis tools (`synthesize_topic`, `synthesize_project`, `synthesize_landscape`) re-expand it into prose.
3. No explicit budget mechanism — tools return whatever the LLM generated, with no post-compression pass.
4. The `brief_or_full()` pattern in researcher lacks `compact` mode entirely.

## Design Principle

> **Local models are cheap. External context is expensive.**
>
> Every token sent to an external coding agent must earn its place. If a local model can compress, score, filter, or structure data before it reaches the MCP boundary, it should. The MCP response is the compression checkpoint.

## Scope

### In scope (library — this milestone)

1. **Context budget framework** in `khonliang.mcp` — a utility that lets MCP tools declare a target token budget and auto-truncates/prioritizes output to fit.
2. **Compressed artifact types** — dataclass definitions for the structured outputs that local models should produce (concept summaries, scored FR candidates, relevance rankings). These become the canonical shapes returned by compact-mode tools.
3. **Post-compression helper** — a function that takes raw LLM output and produces a compact artifact using a local model (or rule-based fallback). This is the "last mile" compressor before MCP returns.
4. **Design documentation** — update `docs/mcp-server.md` and `CLAUDE.md` to state local-model compression as a core architectural goal.
5. **Migration path for `brief_or_full()`** — deprecation notice and adoption guide for moving researcher tools to 3-mode `format_response()`.

### Out of scope (researcher — separate agent)

- Migrating all 40+ researcher tools to `format_response()` (that's FR `fr_khonliang_35a53118`)
- Changing distillation prompts to produce more compact summaries
- FR generation pipeline changes (that's FR `fr_khonliang_6ec6d1e8`)

## Technical Design

### 1. Context Budget (`khonliang.mcp.budget`)

```python
@dataclass
class ContextBudget:
    max_tokens: int = 500          # target ceiling for compact mode
    max_items: int = 10            # max list items
    max_preview_chars: int = 80    # per-item preview length
    priority_field: str = "score"  # sort/trim by this field

def fit_to_budget(items: list[dict], budget: ContextBudget) -> list[dict]:
    """Sort by priority, truncate to budget, trim previews."""
```

Tools use this to declare their output ceiling. The `fit_to_budget` function handles the mechanical trimming. This keeps budget logic out of individual tool implementations.

### 2. Compressed Artifact Types (`khonliang.mcp.artifacts`)

```python
@dataclass
class CompactConcept:
    name: str
    relevance: float        # 0-1 project relevance score
    paper_count: int
    top_paper: str          # single best paper title
    actionable: bool        # is there a concrete implementation path?

@dataclass
class CompactFR:
    id: str
    title: str
    priority: str
    target: str
    concept: str
    depends_on: list[str]   # FR IDs

@dataclass
class CompactSynthesis:
    topic: str
    paper_count: int
    key_findings: list[str]   # max 5, one line each
    relevance: dict[str, float]  # project -> score
    suggested_frs: list[str]    # FR IDs or titles
```

Each artifact type has a `.to_compact() -> str` method that formats for pipe-delimited MCP output, and a `.to_brief() -> str` for structured one-liner output.

### 3. Post-Compression Helper (`khonliang.mcp.compress`)

```python
async def compress_for_agent(
    raw_text: str,
    artifact_type: type[T],
    budget: ContextBudget | None = None,
    model: str = "llama3.2:3b",
) -> T:
    """Compress raw LLM output into a structured artifact.

    Strategy:
    1. If raw_text is already structured (JSON/key-value), parse directly
    2. Otherwise, use a local model to extract fields into artifact_type
    3. Apply budget constraints
    4. Return the structured artifact instance

    Callers serialize the returned artifact with `.to_compact()` or
    `.to_brief()` based on the MCP response format they need.
    Falls back to rule-based extraction if model is unavailable.
    """
```

This is the key new capability: a local model doing the compression work so the external agent gets structured, budget-constrained output.

### 4. Documentation Updates

- `docs/mcp-server.md`: Add "Context Compression" section stating the design principle. Add examples of budget-constrained tools.
- `CLAUDE.md`: Add the design principle quote to the MCP Tool Response Convention section.
- `docs/context-compression.md`: New guide explaining the economic model, when to use budgets, and how to write compressed artifact types.

### 5. `brief_or_full()` Deprecation

- Add `DeprecationWarning` to `brief_or_full()` pointing to `format_response()`.
- Add migration guide in `docs/context-compression.md` showing before/after for a typical tool.

## File Changes

| File                             | Change                                            |
| -------------------------------- | ------------------------------------------------- |
| `src/khonliang/mcp/budget.py`    | New — ContextBudget, fit_to_budget                |
| `src/khonliang/mcp/artifacts.py` | New — CompactConcept, CompactFR, CompactSynthesis |
| `src/khonliang/mcp/compress.py`  | New — compress_for_agent                          |
| `src/khonliang/mcp/compact.py`   | Deprecation warning on brief_or_full()            |
| `src/khonliang/mcp/__init__.py`  | Export new modules                                |
| `docs/mcp-server.md`             | Context Compression section                       |
| `docs/context-compression.md`    | New — full guide                                  |
| `CLAUDE.md`                      | Design principle in MCP convention section        |
| `tests/test_budget.py`           | Unit tests for budget logic                       |
| `tests/test_artifacts.py`        | Unit tests for artifact formatting                |
| `tests/test_compress.py`         | Unit tests for compression (mocked LLM)           |

## Acceptance Criteria

1. `ContextBudget` + `fit_to_budget` work with arbitrary list-of-dict data
2. All three artifact types serialize to compact and brief formats
3. `compress_for_agent` produces structured output from raw text (with mocked local model in tests)
4. `brief_or_full()` emits deprecation warning
5. Design principle documented in CLAUDE.md and docs/
6. All new code passes ruff, mypy, and pytest

## Dependencies

None — this is foundational infrastructure for the other two FRs.

## Non-Goals

- Measuring actual token savings (deferred to researcher adoption)
- Changing any existing tool behavior (additive only)
- Building a token counter (use rough char/4 estimate for budgets)
