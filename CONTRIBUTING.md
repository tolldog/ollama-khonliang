# Contributing to khonliang

Thank you for your interest in contributing to khonliang! This guide covers the development workflow, code standards, and how to submit changes.

## Getting Started

```bash
git clone https://github.com/tolldog/ollama-khonliang.git
cd ollama-khonliang
pip install -e ".[dev,reporting]"
```

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy-lang.org/) for type checking. Configuration is in `pyproject.toml`.

- **Line length:** 100 characters
- **Target Python:** 3.10+
- **Lint rules:** E, F, I, W (pycodestyle, pyflakes, isort, warnings)

Run the linter:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

We also use [trunk](https://trunk.io) for pre-commit checks. Run before pushing:

```bash
trunk check
```

## Testing

Tests use [pytest](https://docs.pytest.org/) with async support via `pytest-asyncio`.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=khonliang --cov-report=term-missing

# Run a specific test file
pytest tests/test_reporting.py -v
```

All PRs must pass the existing test suite. New features should include tests.

## Commit Messages

Write clear, concise commit messages:

- Lead with what changed, not why (save the why for the PR description)
- Use imperative mood ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issue numbers where applicable (`#42`)

## Pull Requests

- Keep PRs focused — one feature or fix per PR
- Include a description of what changed and why
- Add a test plan describing how to verify
- PRs are reviewed by Copilot and maintainers before merging

## Project Structure

```text
src/khonliang/
  client.py          # Ollama async client
  openai_client.py   # OpenAI-compatible client
  pool.py            # Model pool (role -> model mapping)
  health.py          # Model health tracking
  protocols.py       # LLMClient protocol
  roles/             # Role-based agents and routing
  consensus/         # Multi-agent voting
  knowledge/         # Three-tier knowledge store
  research/          # Research pool and engines
  reporting/         # Report persistence and HTTP serving
  digest/            # Activity accumulation and synthesis
  gateway/           # Redis Streams message bus
  ...
```

See [CLAUDE.md](CLAUDE.md) for a detailed architecture overview.

## Adding Dependencies

- Core dependencies (`aiohttp`, `requests`) are kept minimal
- Optional features use extras: `[rag]`, `[mattermost]`, `[gateway]`, `[reporting]`, etc.
- New optional dependencies should get their own extras group in `pyproject.toml`
- Avoid adding dependencies for functionality that can be implemented in a few lines

## Questions?

Open an issue or start a discussion on GitHub.
