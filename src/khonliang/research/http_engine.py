"""
HTTP transport engine — calls external agent containers via HTTP.

Enables agents written in any language (Go, Rust, Python) that expose
a simple HTTP API:
    POST /analyze  — accepts JSON, returns JSON result
    GET  /health   — health check

Usage:
    engine = HttpEngine(
        name="go_scanner",
        base_url="http://localhost:8080",
        max_threads=4,
    )
    results = await engine.query("scan the network")
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from khonliang.research.engine import BaseEngine, EngineResult

logger = logging.getLogger(__name__)


class HttpEngine(BaseEngine):
    """
    Engine that delegates to an external HTTP service.

    The service must implement:
        POST {base_url}/analyze
            Body: {"query": "...", "options": {...}}
            Response: {"results": [{"title": "...", "content": "...", ...}]}

        GET {base_url}/health
            Response: 200 OK

    Args:
        name: Engine name for identification
        base_url: Service base URL
        endpoint: Path for analysis requests (default: /analyze)
        health_endpoint: Path for health checks (default: /health)
        max_threads: Concurrent request limit
        rate_limit: Min seconds between requests
        timeout: Per-request timeout in seconds
        headers: Additional HTTP headers
    """

    def __init__(
        self,
        name: str = "http",
        base_url: str = "http://localhost:8080",
        endpoint: str = "/analyze",
        health_endpoint: str = "/health",
        max_threads: int = 4,
        rate_limit: float = 0.0,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.name = name
        self.max_threads = max_threads
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._health_endpoint = health_endpoint
        self._headers = headers or {"Content-Type": "application/json"}
        self._session = requests.Session()
        self._session.headers.update(self._headers)

    def stop(self) -> None:
        """Shutdown the thread pool and close the HTTP session."""
        super().stop()
        self._session.close()

    async def execute(
        self, query: str, **kwargs: Any
    ) -> List[EngineResult]:
        """Send query to the HTTP service and parse results."""
        payload = {"query": query}
        if kwargs:
            payload["options"] = kwargs

        response = await self.run_sync(self._post, payload)
        if response is None:
            return []

        # Parse results
        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raw_results = [response]

        results = []
        for item in raw_results:
            results.append(EngineResult(
                title=item.get("title", ""),
                content=item.get("content", item.get("text", "")),
                url=item.get("url", ""),
                score=item.get("score", 0.0),
                metadata=item.get("metadata", {}),
            ))

        return results

    # TODO: Add tests for HttpEngine (out of scope for this PR).

    def _post(self, payload: Dict) -> Optional[Dict]:
        """Synchronous POST to the service."""
        try:
            resp = self._session.post(
                f"{self._base_url}{self._endpoint}",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"HttpEngine {self.name} error: {e}")
            return None

    def is_healthy(self) -> bool:
        """Check if the service is reachable."""
        try:
            resp = self._session.get(
                f"{self._base_url}{self._health_endpoint}",
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
