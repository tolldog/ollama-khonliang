# KH-15 Milestone: Local-Model Context Compression

**Spec:** `specs/KH-15/spec.md`
**FR:** `fr_khonliang_6d0277b6`
**Branch:** `kh-15/context-compression`

## Implementation Plan

### Phase 1: Budget Framework

1. Create `src/khonliang/mcp/budget.py`
   - `ContextBudget` dataclass (max_tokens, max_items, max_preview_chars, priority_field)
   - `fit_to_budget()` — sort by priority, truncate items, trim previews
2. Tests: `tests/mcp/test_budget.py`

### Phase 2: Compressed Artifact Types

3. Create `src/khonliang/mcp/artifacts.py`
   - `CompactConcept`, `CompactFR`, `CompactSynthesis` dataclasses
   - `.to_compact()` and `.to_brief()` methods on each
   - `from_dict()` class methods for construction from raw data
4. Tests: `tests/mcp/test_artifacts.py`

### Phase 3: Post-Compression Helper

5. Create `src/khonliang/mcp/compress.py`
   - `compress_for_agent()` — parse structured input or invoke local model
   - Rule-based fallback when model unavailable
6. Tests: `tests/mcp/test_compress.py` (mocked LLM calls)

### Phase 4: Deprecation + Exports

7. Add `DeprecationWarning` to `brief_or_full()` in `compact.py`
8. Update `src/khonliang/mcp/__init__.py` with new exports

### Phase 5: Documentation

9. Update `CLAUDE.md` — add design principle to MCP convention section
10. Update `docs/mcp-server.md` — add Context Compression section
11. Create `docs/context-compression.md` — full guide with examples and migration path

### Phase 6: Validation

12. Run full test suite, ruff, mypy
13. Verify no regressions in existing MCP tools

## Done When

- All 6 phases complete
- CI green (ruff + mypy + pytest)
- Spec acceptance criteria met
