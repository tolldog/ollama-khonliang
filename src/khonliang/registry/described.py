"""
Generic described-registry primitive: cheap ``index`` + nearest-match ``expand``.

See :mod:`khonliang.registry` for the design rationale. This module holds the
contract (dataclasses + adapter Protocol), the orchestrating
:class:`DescribedRegistry`, and an in-memory reference adapter used as the
canonical example and unit-test vehicle.

Migration path for the two existing in-house instances (documented, not yet
re-expressed here — see FR fr_khonliang_0f3c7542 phases):

- Deferred-tool autoload / ToolSearch: an adapter whose ``catalog`` returns
  ``IndexEntry(tool_name, one_line_tool_description)`` and whose ``expand``
  returns the full JSONSchema for each requested tool. ``depth`` could pull
  co-registered tools (same server / same capability cluster).
- Agent ``welcome`` surface: an adapter whose ``catalog`` returns
  ``IndexEntry(agent_id, agent_summary)`` and whose ``expand`` returns the
  agent's skill catalog; ``depth`` walks the fleet neighborhood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

__all__ = [
    "IndexEntry",
    "ExpandedItem",
    "ItemAdapter",
    "DescribedRegistry",
    "InMemoryDescribedAdapter",
]


@dataclass
class IndexEntry:
    """One line in the cheap catalog: an id and its own one-line description."""

    id: str
    description: str
    # Optional small hints the LLM can use to pick without expanding
    # (e.g. scope tag, centrality). Kept tiny on purpose.
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "description": self.description}
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass
class ExpandedItem:
    """Full detail for one requested id, plus items connected out to ``depth``."""

    id: str
    detail: str
    # Connected items, each a light dict — at minimum {id, description}; adapters
    # may add {relation, depth, ...}. Ordered by relevance/strength if the
    # adapter ranks them.
    connected: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "detail": self.detail}
        if self.connected:
            d["connected"] = self.connected
        if self.meta:
            d["meta"] = self.meta
        return d


@runtime_checkable
class ItemAdapter(Protocol):
    """
    Item-type-specific half of the contract. Owns ALL domain knowledge.

    Implementations are thin: two async methods. The generic
    :class:`DescribedRegistry` handles the LLM-facing two-call shape, index
    capping, id de-duplication, and batching — the adapter never has to.
    """

    async def catalog(
        self, scope: Optional[str] = None, limit: Optional[int] = None
    ) -> List[IndexEntry]:
        """
        Return the token-small catalog of available items.

        ``scope`` narrows the item set (adapter-defined: taxonomy root, source
        tag, ...). ``limit`` is a soft cap the adapter may use to return only the
        top-N most salient items (by centrality/recency/etc.); the registry also
        enforces a hard cap, so returning more is safe but wasteful.
        """
        ...

    async def expand(
        self, ids: Sequence[str], depth: int = 1
    ) -> Dict[str, ExpandedItem]:
        """
        Return full detail for each requested id, batched.

        Receives the **plural, de-duplicated** id list so a DB-backed adapter can
        satisfy the whole batch in one query. ``depth`` controls how far
        connected items are walked (0 = detail only). Unknown ids are simply
        absent from the returned mapping.
        """
        ...


class DescribedRegistry:
    """
    Orchestrates the generic two-call contract over a pluggable
    :class:`ItemAdapter`.

    Responsibilities kept OUT of adapters:

    - ``index`` hard-caps the catalog size (``max_index``) so a runaway adapter
      can't blow the token budget.
    - ``expand`` de-duplicates the id list (preserving first-seen order) and
      clamps ``depth`` to ``[0, max_depth]`` before handing the batch to the
      adapter in a single call.
    """

    def __init__(
        self,
        adapter: ItemAdapter,
        *,
        max_index: int = 200,
        max_depth: int = 3,
    ) -> None:
        if max_index < 1:
            raise ValueError("max_index must be >= 1")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        self.adapter = adapter
        self.max_index = max_index
        self.max_depth = max_depth

    async def index(
        self, scope: Optional[str] = None, limit: Optional[int] = None
    ) -> List[IndexEntry]:
        """Cheap catalog. ``limit`` (if given) is min'd with the hard cap."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        effective_limit = self.max_index if limit is None else min(limit, self.max_index)
        entries = await self.adapter.catalog(scope=scope, limit=effective_limit)
        return list(entries)[:effective_limit]

    async def expand(
        self, ids: Sequence[str], depth: int = 1
    ) -> Dict[str, ExpandedItem]:
        """
        Batched nearest-match expansion.

        De-dupes ``ids`` (first-seen order preserved), clamps ``depth`` into
        ``[0, max_depth]``, and issues a SINGLE adapter call for the whole batch.
        """
        deduped: List[str] = []
        seen = set()
        for _id in ids:
            if _id not in seen:
                seen.add(_id)
                deduped.append(_id)
        if not deduped:
            return {}
        clamped_depth = max(0, min(depth, self.max_depth))
        return await self.adapter.expand(deduped, depth=clamped_depth)


class InMemoryDescribedAdapter:
    """
    Reference :class:`ItemAdapter` over an in-memory item set with a neighbor
    graph. Canonical example + unit-test vehicle; also a drop-in for small
    static catalogs.

    Each item is ``{id, description, detail, neighbors: [ids]}``. ``expand``
    walks the neighbor graph breadth-first out to ``depth`` (0 = detail only),
    tagging each connected item with the hop distance at which it was reached.
    """

    def __init__(self, items: Sequence[Dict[str, Any]]) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        for it in items:
            self._items[it["id"]] = {
                "id": it["id"],
                "description": it.get("description", ""),
                "detail": it.get("detail", ""),
                "neighbors": list(it.get("neighbors", [])),
                "scope": it.get("scope"),
                "centrality": it.get("centrality", 0.0),
            }

    async def catalog(
        self, scope: Optional[str] = None, limit: Optional[int] = None
    ) -> List[IndexEntry]:
        items = list(self._items.values())
        if scope is not None:
            items = [it for it in items if it.get("scope") == scope]
        # Most-central first so a soft limit keeps the salient items.
        items.sort(key=lambda it: it.get("centrality", 0.0), reverse=True)
        if limit is not None:
            items = items[:limit]
        return [
            IndexEntry(id=it["id"], description=it["description"])
            for it in items
        ]

    async def expand(
        self, ids: Sequence[str], depth: int = 1
    ) -> Dict[str, ExpandedItem]:
        out: Dict[str, ExpandedItem] = {}
        for _id in ids:
            item = self._items.get(_id)
            if item is None:
                continue
            connected = self._walk(_id, depth) if depth > 0 else []
            out[_id] = ExpandedItem(
                id=_id,
                detail=item["detail"],
                connected=connected,
            )
        return out

    def _walk(self, start: str, depth: int) -> List[Dict[str, Any]]:
        """BFS the neighbor graph out to ``depth`` hops from ``start``."""
        connected: List[Dict[str, Any]] = []
        visited = {start}
        frontier = [start]
        for hop in range(1, depth + 1):
            next_frontier: List[str] = []
            for node in frontier:
                for nb in self._items.get(node, {}).get("neighbors", []):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    next_frontier.append(nb)
                    nb_item = self._items.get(nb)
                    if nb_item is not None:
                        connected.append(
                            {
                                "id": nb,
                                "description": nb_item["description"],
                                "depth": hop,
                            }
                        )
            frontier = next_frontier
            if not frontier:
                break
        return connected
