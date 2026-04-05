---
applyTo: "**"
---

# Copilot Pull Request Review Instructions

## Review Only — Do Not Modify Code

When reviewing pull requests:

1. **Comment only** — provide feedback as review comments
2. **Do NOT push commits** — never modify code directly on the PR branch
3. **Do NOT create fixup commits** — the author will address your feedback
4. **Suggest changes** using GitHub's suggestion syntax (` ```suggestion `) so the author can apply them manually

## Review Focus

- Correctness: logic errors, edge cases, off-by-one
- Safety: thread safety, resource leaks, error handling
- API design: backward compatibility, naming consistency
- Testing: missing coverage, fragile tests
- Performance: unnecessary allocations, N+1 queries

## Domain Context

This is `ollama-khonliang`, a multi-agent LLM orchestration library. Key modules:

- `consensus/` — voting, debate, outcome tracking, credit assignment
- `knowledge/` — triple store, knowledge store, ingestion
- `roles/` — LLM role abstraction
- `gateway/` — blackboard, agent messaging

The library is domain-agnostic — avoid trading/finance-specific terminology in suggestions.
