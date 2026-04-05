"""Tests for AgentCapability and CapabilityRegistry embedding-based routing."""

import pytest

from khonliang.agents.capabilities import (
    AgentCapability,
    CapabilityRegistry,
    _cosine_similarity,
)

# ---------------------------------------------------------------------------
# AgentCapability serialization
# ---------------------------------------------------------------------------


def test_to_dict_includes_embedding():
    cap = AgentCapability(
        "agent1", "sentiment", "Sentiment analysis", embedding=[0.1, 0.2, 0.3]
    )
    d = cap.to_dict()
    assert d["embedding"] == [0.1, 0.2, 0.3]


def test_from_dict_restores_embedding():
    cap = AgentCapability(
        "agent1", "sentiment", "Sentiment analysis", embedding=[0.1, 0.2, 0.3]
    )
    cap2 = AgentCapability.from_dict(cap.to_dict())
    assert cap2.embedding == [0.1, 0.2, 0.3]


def test_from_dict_embedding_defaults_to_none():
    data = {
        "agent_id": "agent1",
        "capability": "sentiment",
        "description": "Sentiment analysis",
        "input_schema": None,
        # no "embedding" key
    }
    cap = AgentCapability.from_dict(data)
    assert cap.embedding is None


def test_embedding_not_shown_in_repr():
    cap = AgentCapability("a", "cap", "desc", embedding=[1.0, 2.0])
    assert "embedding" not in repr(cap)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_empty_vectors():
    assert _cosine_similarity([], []) == 0.0


def test_cosine_mismatched_dimensions():
    assert _cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# CapabilityRegistry.find_capable
# ---------------------------------------------------------------------------


def _registry_with_agents():
    registry = CapabilityRegistry()
    registry.register(
        "quant",
        [
            AgentCapability(
                "quant",
                "price_patterns",
                "Detect price patterns in time series",
                embedding=[1.0, 0.0, 0.0],
            )
        ],
    )
    registry.register(
        "sentiment",
        [
            AgentCapability(
                "sentiment",
                "sentiment_analysis",
                "Analyse text sentiment",
                embedding=[0.0, 1.0, 0.0],
            )
        ],
    )
    return registry


def test_find_capable_returns_ranked_results():
    registry = _registry_with_agents()
    # Task embedding close to quant
    results = registry.find_capable("price patterns", task_embedding=[1.0, 0.0, 0.0])
    assert results
    assert results[0][0] == "quant"
    assert results[0][1] == pytest.approx(1.0)


def test_find_capable_quant_ranks_above_sentiment():
    registry = _registry_with_agents()
    # Use a task embedding between the two agents, closer to quant
    # and a low threshold to ensure both agents appear
    results = registry.find_capable(
        "price movement patterns",
        task_embedding=[0.9, 0.44, 0.0],  # ~24° from quant, ~64° from sentiment
        threshold=0.3,
    )
    agent_ids = [r[0] for r in results]
    assert "quant" in agent_ids
    assert "sentiment" in agent_ids
    assert agent_ids.index("quant") < agent_ids.index("sentiment")


def test_find_capable_low_threshold_returns_more_agents():
    registry = _registry_with_agents()
    # Task embedding at 45° between quant and sentiment (equal similarity ~0.71)
    task_emb = [0.707, 0.707, 0.0]
    high = registry.find_capable("task", task_embedding=task_emb, threshold=0.8)
    low = registry.find_capable("task", task_embedding=task_emb, threshold=0.5)
    assert len(low) >= len(high)


def test_find_capable_no_embeddings_returns_empty():
    registry = CapabilityRegistry()
    registry.register(
        "agent1",
        [AgentCapability("agent1", "cap", "No embedding")],
    )
    results = registry.find_capable("any task", task_embedding=[1.0, 0.0])
    assert results == []


def test_find_capable_no_embedding_no_embed_fn_returns_empty():
    registry = _registry_with_agents()
    results = registry.find_capable("price patterns")
    assert results == []


def test_find_capable_empty_registry_returns_empty():
    registry = CapabilityRegistry()
    results = registry.find_capable("task", task_embedding=[1.0, 0.0])
    assert results == []


def test_find_capable_embed_fn_callback():
    registry = _registry_with_agents()

    def embed_fn(text: str):
        # Returns quant-aligned embedding for any text
        return [1.0, 0.0, 0.0]

    results = registry.find_capable("anything", embed_fn=embed_fn)
    assert results
    assert results[0][0] == "quant"


def test_find_capable_embed_fn_failure_returns_empty():
    registry = _registry_with_agents()

    def bad_embed_fn(text: str):
        raise RuntimeError("embedding service unavailable")

    results = registry.find_capable("task", embed_fn=bad_embed_fn)
    assert results == []


def test_find_capable_limit():
    registry = CapabilityRegistry()
    for i in range(10):
        registry.register(
            f"agent{i}",
            [AgentCapability(f"agent{i}", f"cap{i}", "desc", embedding=[1.0, 0.0])],
        )
    results = registry.find_capable("task", task_embedding=[1.0, 0.0], limit=3)
    assert len(results) == 3


def test_find_capable_scores_sorted_descending():
    registry = CapabilityRegistry()
    registry.register(
        "a", [AgentCapability("a", "c1", "d", embedding=[1.0, 0.0, 0.0])]
    )
    registry.register(
        "b", [AgentCapability("b", "c2", "d", embedding=[0.9, 0.1, 0.0])]
    )
    registry.register(
        "c", [AgentCapability("c", "c3", "d", embedding=[0.6, 0.4, 0.0])]
    )
    results = registry.find_capable("task", task_embedding=[1.0, 0.0, 0.0])
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


def test_get_stats_includes_embedding_count():
    registry = _registry_with_agents()
    stats = registry.get_stats()
    assert stats["capabilities_with_embeddings"] == 2
    assert stats["registered_agents"] == 2
    assert stats["total_capabilities"] == 2


def test_get_stats_no_embeddings():
    registry = CapabilityRegistry()
    registry.register("a", [AgentCapability("a", "cap", "desc")])
    assert registry.get_stats()["capabilities_with_embeddings"] == 0


# ---------------------------------------------------------------------------
# Existing find() is unaffected
# ---------------------------------------------------------------------------


def test_find_exact_match_still_works():
    registry = _registry_with_agents()
    results = registry.find("price_patterns")
    assert len(results) == 1
    assert results[0].agent_id == "quant"
