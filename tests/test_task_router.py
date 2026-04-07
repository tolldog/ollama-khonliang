"""Tests for TaskRouter — dynamic capability-driven task routing (FR khonliang_2)."""


from khonliang.agents.capabilities import AgentCapability, CapabilityRegistry
from khonliang.routing.task_router import RouteMatch, TaskRouter, TaskRouterConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap(agent_id: str, capability: str, description: str = "",
         cost: float = 0.0, max_concurrent: int = 0,
         embedding: list[float] | None = None) -> AgentCapability:
    return AgentCapability(
        agent_id=agent_id,
        capability=capability,
        description=description or capability,
        cost_per_call=cost,
        max_concurrent=max_concurrent,
        embedding=embedding,
    )


def _simple_embed(text: str) -> list[float]:
    """Deterministic fake embedding based on text hash for testing."""
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(c, 16) / 15.0 for c in h[:8]]


def _registry_with_embeddings():
    """Build a registry with embedded capabilities."""
    registry = CapabilityRegistry()

    cap_a = _cap("analyst", "anomaly_detection", "Detect anomalies in data",
                  cost=0.1, embedding=_simple_embed("anomaly detection"))
    cap_b = _cap("researcher", "literature_review", "Search and summarize papers",
                  cost=0.5, embedding=_simple_embed("literature review"))
    cap_c = _cap("coder", "code_generation", "Write and fix code",
                  cost=0.2, embedding=_simple_embed("code generation"))

    registry.register("analyst", [cap_a])
    registry.register("researcher", [cap_b])
    registry.register("coder", [cap_c])

    return registry


# ---------------------------------------------------------------------------
# RouteMatch
# ---------------------------------------------------------------------------


class TestRouteMatch:
    def test_to_dict(self):
        m = RouteMatch(agent_id="a", capability="test", score=0.9, cost=0.1)
        d = m.to_dict()
        assert d["agent_id"] == "a"
        assert d["score"] == 0.9
        assert d["cost"] == 0.1


# ---------------------------------------------------------------------------
# TaskRouter — exact match fallback
# ---------------------------------------------------------------------------


class TestExactMatchFallback:
    def test_exact_capability_name(self):
        registry = CapabilityRegistry()
        registry.register("analyst", [_cap("analyst", "anomaly_detection")])

        router = TaskRouter(registry)
        match = router.route("anomaly_detection")

        assert match is not None
        assert match.agent_id == "analyst"
        assert match.score == 1.0

    def test_no_match_returns_none(self):
        registry = CapabilityRegistry()
        router = TaskRouter(registry)
        assert router.route("nonexistent") is None


# ---------------------------------------------------------------------------
# TaskRouter — embedding-based routing
# ---------------------------------------------------------------------------


class TestEmbeddingRouting:
    def test_routes_to_best_match(self):
        registry = _registry_with_embeddings()
        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,  # low threshold for test embeddings
        ))

        match = router.route(
            "find anomalies",
            task_embedding=_simple_embed("anomaly detection"),
        )

        assert match is not None
        assert match.agent_id == "analyst"

    def test_embed_fn_used_when_no_precomputed(self):
        registry = _registry_with_embeddings()
        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,
        ))

        match = router.route("find anomalies", embed_fn=_simple_embed)

        assert match is not None

    def test_route_multi(self):
        registry = _registry_with_embeddings()
        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,
        ))

        matches = router.route_multi(
            "analyze data",
            n=3,
            task_embedding=_simple_embed("analyze data"),
        )

        assert len(matches) <= 3
        assert all(isinstance(m, RouteMatch) for m in matches)


# ---------------------------------------------------------------------------
# Cost-biased scoring
# ---------------------------------------------------------------------------


class TestCostBias:
    def test_cheap_agent_preferred_at_similar_similarity(self):
        """When similarity is equal, cheaper agent should win."""
        registry = CapabilityRegistry()
        emb = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        registry.register("cheap", [_cap(
            "cheap", "task", "do task", cost=0.01, embedding=emb,
        )])
        registry.register("expensive", [_cap(
            "expensive", "task", "do task", cost=1.0, embedding=emb,
        )])

        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,
            cost_weight=0.5,
        ))

        match = router.route("do task", task_embedding=emb)

        assert match is not None
        assert match.agent_id == "cheap"

    def test_zero_cost_weight_ignores_cost(self):
        """When cost_weight=0, cost should not affect routing."""
        registry = CapabilityRegistry()
        emb = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        registry.register("cheap", [_cap(
            "cheap", "task", "do task", cost=0.01, embedding=emb,
        )])
        registry.register("expensive", [_cap(
            "expensive", "task", "do task", cost=100.0, embedding=emb,
        )])

        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,
            cost_weight=0.0,
        ))

        # Both should have same score — either could be picked
        match = router.route("do task", task_embedding=emb)
        assert match is not None


# ---------------------------------------------------------------------------
# Concurrency limits
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_skips_busy_agent(self):
        """When best agent is at capacity, next best is selected."""
        registry = CapabilityRegistry()
        # Give "busy" a perfect match, "free" a slightly weaker one
        busy_emb = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        free_emb = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        registry.register("busy", [_cap(
            "busy", "task", "do task", max_concurrent=1, embedding=busy_emb,
        )])
        registry.register("free", [_cap(
            "free", "task", "do task", max_concurrent=0, embedding=free_emb,
        )])

        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,
            prefer_available=True,
        ))

        # First route goes to "busy" (best match)
        match1 = router.route("do task", task_embedding=busy_emb)
        assert match1 is not None
        assert match1.agent_id == "busy"

        # busy is now at capacity — second route should go to free
        match2 = router.route("do task", task_embedding=busy_emb)
        assert match2 is not None
        assert match2.agent_id == "free"

    def test_release_frees_slot(self):
        registry = CapabilityRegistry()
        emb = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        registry.register("limited", [_cap(
            "limited", "task", "do task", max_concurrent=1, embedding=emb,
        )])

        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,
        ))

        match1 = router.route("do task", task_embedding=emb)
        assert match1 is not None
        assert match1.agent_id == "limited"

        # Agent is now busy — shouldn't be routed to
        assert not router._is_available("limited")

        # Release the slot
        router.release("limited")
        assert router._is_available("limited")

    def test_unlimited_always_available(self):
        registry = CapabilityRegistry()
        registry.register("unlimited", [_cap(
            "unlimited", "task", "do task", max_concurrent=0,
        )])

        router = TaskRouter(registry)
        assert router._is_available("unlimited")

        # Acquire many times — still available (unlimited)
        for _ in range(100):
            router._acquire("unlimited")
        assert router._is_available("unlimited")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_after_routes(self):
        registry = CapabilityRegistry()
        registry.register("a", [_cap("a", "task", embedding=[1.0] * 8)])

        router = TaskRouter(registry, config=TaskRouterConfig(
            similarity_threshold=0.0,
        ))
        router.route("test", task_embedding=[1.0] * 8)

        stats = router.get_stats()
        assert stats["total_routes"] == 1
        assert "a" in stats["active_agents"]


# ---------------------------------------------------------------------------
# AgentCapability new fields
# ---------------------------------------------------------------------------


class TestCapabilityFields:
    def test_cost_and_concurrency_defaults(self):
        cap = AgentCapability(agent_id="a", capability="test", description="test")
        assert cap.cost_per_call == 0.0
        assert cap.max_concurrent == 0

    def test_cost_and_concurrency_custom(self):
        cap = _cap("a", "test", cost=0.5, max_concurrent=3)
        assert cap.cost_per_call == 0.5
        assert cap.max_concurrent == 3

    def test_from_dict_with_new_fields(self):
        data = {
            "agent_id": "a",
            "capability": "test",
            "description": "test",
            "cost_per_call": 0.3,
            "max_concurrent": 5,
        }
        cap = AgentCapability.from_dict(data)
        assert cap.cost_per_call == 0.3
        assert cap.max_concurrent == 5

    def test_from_dict_without_new_fields(self):
        """Old data without new fields should use defaults."""
        data = {
            "agent_id": "a",
            "capability": "test",
            "description": "test",
        }
        cap = AgentCapability.from_dict(data)
        assert cap.cost_per_call == 0.0
        assert cap.max_concurrent == 0

    def test_to_dict_includes_new_fields(self):
        cap = _cap("a", "test", cost=0.5, max_concurrent=3)
        d = cap.to_dict()
        assert d["cost_per_call"] == 0.5
        assert d["max_concurrent"] == 3
