"""Tests for the generic described-registry primitive (index + expand)."""

import pytest

from khonliang.registry import (
    DescribedRegistry,
    ExpandedItem,
    IndexEntry,
    InMemoryDescribedAdapter,
    ItemAdapter,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_index_entry_to_dict_omits_empty_meta():
    e = IndexEntry(id="a", description="alpha")
    assert e.to_dict() == {"id": "a", "description": "alpha"}


def test_index_entry_to_dict_includes_meta():
    e = IndexEntry(id="a", description="alpha", meta={"scope": "s"})
    assert e.to_dict() == {"id": "a", "description": "alpha", "meta": {"scope": "s"}}


def test_expanded_item_to_dict_shapes():
    x = ExpandedItem(id="a", detail="full text", connected=[{"id": "b"}])
    d = x.to_dict()
    assert d["id"] == "a"
    assert d["detail"] == "full text"
    assert d["connected"] == [{"id": "b"}]
    assert "meta" not in d


# ---------------------------------------------------------------------------
# Reference adapter conforms to the Protocol
# ---------------------------------------------------------------------------


def test_reference_adapter_is_item_adapter():
    adapter = InMemoryDescribedAdapter([])
    assert isinstance(adapter, ItemAdapter)


def _sample_items():
    return [
        {"id": "graph", "description": "graph memory", "detail": "GRAPH DETAIL " * 5,
         "neighbors": ["rag"], "centrality": 0.9, "scope": "memory"},
        {"id": "rag", "description": "retrieval aug", "detail": "RAG DETAIL " * 5,
         "neighbors": ["graph", "embed"], "centrality": 0.5, "scope": "memory"},
        {"id": "embed", "description": "embeddings", "detail": "EMBED DETAIL " * 5,
         "neighbors": ["rag"], "centrality": 0.3, "scope": "vector"},
    ]


# ---------------------------------------------------------------------------
# index()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_returns_catalog():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    entries = await reg.index()
    ids = [e.id for e in entries]
    assert set(ids) == {"graph", "rag", "embed"}
    # sorted by centrality descending
    assert ids[0] == "graph"


@pytest.mark.asyncio
async def test_index_scope_filter():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    entries = await reg.index(scope="vector")
    assert [e.id for e in entries] == ["embed"]


@pytest.mark.asyncio
async def test_index_hard_cap_enforced_over_adapter():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()), max_index=2)
    entries = await reg.index()  # limit=None -> uses max_index
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_index_limit_min_with_cap():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()), max_index=5)
    entries = await reg.index(limit=1)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# expand() — batching, dedup, depth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_batches_multiple_ids():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    result = await reg.expand(["graph", "embed"], depth=1)
    assert set(result.keys()) == {"graph", "embed"}
    assert result["graph"].detail.startswith("GRAPH DETAIL")


@pytest.mark.asyncio
async def test_expand_dedupes_ids():
    calls = {}

    class RecordingAdapter(InMemoryDescribedAdapter):
        async def expand(self, ids, depth=1):
            calls["ids"] = list(ids)
            return await super().expand(ids, depth=depth)

    reg = DescribedRegistry(RecordingAdapter(_sample_items()))
    await reg.expand(["graph", "graph", "rag"], depth=0)
    # adapter received one call with deduped, first-seen-order ids
    assert calls["ids"] == ["graph", "rag"]


@pytest.mark.asyncio
async def test_expand_empty_ids_short_circuits():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    assert await reg.expand([]) == {}


@pytest.mark.asyncio
async def test_expand_depth_zero_no_connected():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    result = await reg.expand(["graph"], depth=0)
    assert result["graph"].connected == []


@pytest.mark.asyncio
async def test_expand_depth_one_walks_neighbors():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    result = await reg.expand(["graph"], depth=1)
    connected_ids = {c["id"] for c in result["graph"].connected}
    assert connected_ids == {"rag"}
    assert all(c["depth"] == 1 for c in result["graph"].connected)


@pytest.mark.asyncio
async def test_expand_depth_two_walks_two_hops():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    result = await reg.expand(["graph"], depth=2)
    connected = {c["id"]: c["depth"] for c in result["graph"].connected}
    assert connected == {"rag": 1, "embed": 2}


@pytest.mark.asyncio
async def test_expand_depth_clamped_to_max():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()), max_depth=1)
    result = await reg.expand(["graph"], depth=99)
    # clamped to 1 hop -> only rag, not embed
    assert {c["id"] for c in result["graph"].connected} == {"rag"}


@pytest.mark.asyncio
async def test_expand_unknown_id_absent():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    result = await reg.expand(["nope", "graph"])
    assert set(result.keys()) == {"graph"}


# ---------------------------------------------------------------------------
# Registry construction guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_negative_limit_rejected():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    with pytest.raises(ValueError):
        await reg.index(limit=-1)


def test_registry_types_in_package_all():
    import khonliang

    for name in (
        "DescribedRegistry",
        "ItemAdapter",
        "IndexEntry",
        "ExpandedItem",
        "InMemoryDescribedAdapter",
    ):
        assert name in khonliang.__all__


def test_bad_max_index_rejected():
    with pytest.raises(ValueError):
        DescribedRegistry(InMemoryDescribedAdapter([]), max_index=0)


def test_bad_max_depth_rejected():
    with pytest.raises(ValueError):
        DescribedRegistry(InMemoryDescribedAdapter([]), max_depth=-1)


# ---------------------------------------------------------------------------
# Token-efficiency property: catalog << full detail dump
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_is_cheaper_than_full_detail_dump():
    reg = DescribedRegistry(InMemoryDescribedAdapter(_sample_items()))
    index = await reg.index()
    index_tokens = sum(len(e.description.split()) + len(e.id.split()) for e in index)

    full = await reg.expand([e.id for e in index], depth=0)
    full_tokens = sum(len(x.detail.split()) for x in full.values())

    assert index_tokens < full_tokens
