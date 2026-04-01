# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in khonliang, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: **<tolldog@gmail.com>**

Include:

- A description of the vulnerability
- Steps to reproduce
- Impact assessment (what an attacker could achieve)
- Any suggested fix, if you have one

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days.

## Security Considerations

### HTML Report Serving

The `reporting` module serves agent-generated content as HTML pages via Flask. The following protections are in place:

- **HTML sanitization** — All rendered markdown is sanitized via [nh3](https://github.com/messense/nh3) (Rust-backed, used by GitHub). Script tags, event handlers, iframes, and `javascript:` URLs are stripped.
- **Output escaping** — All user-controlled fields (titles, types, creator names) are escaped via `html.escape()` before template interpolation.
- **URL validation** — Permalink and logo URLs are validated for safe schemes (`http`, `https`, relative paths only).
- **CSS injection prevention** — `custom_css` in `ReportTheme` is sanitized to prevent `</style>` breakout (case-insensitive).

**Note:** If `nh3` is not installed, sanitization is disabled with a logged warning. Always install the `[reporting]` extras for production use.

### SQLite

- Database connections use WAL mode for concurrent read performance.
- All queries use parameterized statements — no string interpolation in SQL.
- `check_same_thread=False` is used for Flask thread compatibility.

### LLM Content

Agent-generated content should be treated as untrusted. The sanitization pipeline handles this for HTML output, but consumers of raw markdown or JSON API responses should apply their own validation appropriate to their context.

### Dependencies

| Dependency | Purpose                  | Security Notes                                                   |
| ---------- | ------------------------ | ---------------------------------------------------------------- |
| `nh3`      | HTML sanitization        | Rust-backed, Ammonia allowlist                                   |
| `flask`    | Report HTTP server       | Bind to `0.0.0.0` by default — configure firewall for production |
| `aiohttp`  | Async HTTP client        | Used for LLM backend communication                               |
| `redis`    | Agent gateway (optional) | Requires authenticated Redis for production                      |

### Network

- The report server binds to `0.0.0.0:5050` by default. Restrict this behind a reverse proxy or firewall in production.
- LLM client connections are to localhost by default (`http://localhost:11434` for Ollama). Remote backends should use HTTPS.
- The MCP server supports stdio and streamable HTTP transports.
