"""
Described registry — a generic "cheap index + nearest-match expand" primitive.

The ecosystem repeatedly implements one retrieval shape — *describe many items
cheaply, expand one/few on demand* — separately per domain (deferred-tool
autoload / ToolSearch; the agent ``welcome`` surface; concept-graph-over-RAG;
personality selection). This module factors that shape into a single two-call
contract so each domain becomes a thin adapter rather than bespoke code.

Two calls:

- ``index(scope?, limit?) -> [IndexEntry{id, description}]`` — a token-small
  catalog of available items, one line each, sourced from each item's own
  summary.
- ``expand([ids], depth=1) -> {id: ExpandedItem}`` — per-id detail plus items
  connected out to ``depth``, **batched** over one or more ids.

The LLM scans the index, picks the closest item(s), then issues ONE batched
``expand`` — instead of receiving a flat top-k chunk dump. Retrieval-as-tool-use:
catalog + on-demand detail is far cheaper on tokens than chunk stuffing, and the
API is shaped for an LLM consumer.

All item-type knowledge lives in an :class:`ItemAdapter`; the
:class:`DescribedRegistry` owns only the two-call contract, the dataclasses, and
orchestration (index-size cap, id de-duplication, batching). A new item-type is
a thin adapter with two async methods.
"""

from khonliang.registry.described import (
    DescribedRegistry,
    ExpandedItem,
    IndexEntry,
    InMemoryDescribedAdapter,
    ItemAdapter,
)

__all__ = [
    "DescribedRegistry",
    "ItemAdapter",
    "IndexEntry",
    "ExpandedItem",
    "InMemoryDescribedAdapter",
]
