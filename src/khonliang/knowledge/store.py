"""
Three-tier knowledge store.

Tier 1 — Axioms: immutable rules, always in context, never retrieved via RAG.
Tier 2 — Imported: user-provided documents, agent-managed (summarized, pruned).
Tier 3 — Derived: agent-built through interaction, tagged with provenance.

Uses SQLite for persistence. Tier 2 and 3 entries are FTS5-searchable.

Schema is auto-created on first use.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    """Knowledge tier levels."""

    AXIOM = 1
    IMPORTED = 2
    DERIVED = 3


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge in the store."""

    id: str
    tier: Tier
    title: str
    content: str
    scope: str = "global"  # global, or a domain tag (e.g. "toll", "thomas")
    source: str = ""  # where this came from (filename, agent_id, user)
    confidence: float = 1.0  # 0.0-1.0, lower for derived
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "title": self.title,
            "content": self.content,
            "scope": self.scope,
            "source": self.source,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id          TEXT PRIMARY KEY,
    tier        INTEGER NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    scope       TEXT NOT NULL DEFAULT 'global',
    source      TEXT DEFAULT '',
    confidence  REAL DEFAULT 1.0,
    tags        TEXT DEFAULT '[]',
    metadata    TEXT DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    access_count INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    id, title, content, scope, tags,
    content='knowledge', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, id, title, content, scope, tags)
    VALUES (new.rowid, new.id, new.title, new.content, new.scope, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, id, title, content, scope, tags)
    VALUES ('delete', old.rowid, old.id, old.title, old.content, old.scope, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, id, title, content, scope, tags)
    VALUES ('delete', old.rowid, old.id, old.title, old.content, old.scope, old.tags);
    INSERT INTO knowledge_fts(rowid, id, title, content, scope, tags)
    VALUES (new.rowid, new.id, new.title, new.content, new.scope, new.tags);
END;
"""


class KnowledgeStore:
    """
    Persistent three-tier knowledge store backed by SQLite + FTS5.

    Example:
        store = KnowledgeStore("knowledge.db")

        # Tier 1 — axioms (set once, always available)
        store.set_axiom("identity", "You are a genealogy research assistant.")
        store.set_axiom("rule_cite", "Always cite the source of your data.")

        # Tier 2 — imported documents
        store.add(KnowledgeEntry(
            id="census_1850",
            tier=Tier.IMPORTED,
            title="1850 Census - Adams County, Ohio",
            content="...",
            scope="toll",
            source="census_records/1850_adams.txt",
        ))

        # Tier 3 — derived from agent interaction
        store.add(KnowledgeEntry(
            id="derived_001",
            tier=Tier.DERIVED,
            title="Tolle migration timeline",
            content="The Tolle family migrated from Maryland to Virginia...",
            source="researcher_agent",
            confidence=0.8,
        ))

        # Search across Tier 2+3
        results = store.search("Tolle migration", scope="toll")

        # Get all axioms (always loaded into system prompt)
        axioms = store.get_axioms()
    """

    def __init__(self, db_path: str = "data/knowledge.db"):
        self.db_path = db_path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Tier 1 — Axioms
    # ------------------------------------------------------------------

    def set_axiom(self, key: str, content: str) -> None:
        """Set or update an axiom (Tier 1). Key is used as the ID."""
        entry = KnowledgeEntry(
            id=f"axiom:{key}",
            tier=Tier.AXIOM,
            title=key,
            content=content,
            scope="global",
            source="system",
            confidence=1.0,
        )
        self.add(entry)

    def get_axioms(self) -> List[KnowledgeEntry]:
        """Get all axioms. These should always be loaded into context."""
        return self.get_by_tier(Tier.AXIOM)

    def get_axioms_text(self) -> str:
        """Get axioms as a single text block for system prompt injection."""
        axioms = self.get_axioms()
        if not axioms:
            return ""
        return "\n".join(f"- {a.content}" for a in axioms)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: KnowledgeEntry) -> None:
        """Add or replace a knowledge entry."""
        now = time.time()
        if entry.created_at == 0.0:
            entry.created_at = now
        entry.updated_at = now

        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge
                    (id, tier, title, content, scope, source, confidence,
                     tags, metadata, created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.tier.value,
                    entry.title,
                    entry.content,
                    entry.scope,
                    entry.source,
                    entry.confidence,
                    json.dumps(entry.tags),
                    json.dumps(entry.metadata),
                    entry.created_at,
                    entry.updated_at,
                    entry.access_count,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a specific entry by ID."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM knowledge WHERE id = ?", (entry_id,)
            ).fetchone()
            return self._row_to_entry(row) if row else None
        finally:
            conn.close()

    def remove(self, entry_id: str) -> bool:
        """Remove an entry. Returns True if it existed."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM knowledge WHERE id = ?", (entry_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_by_tier(self, tier: Tier) -> List[KnowledgeEntry]:
        """Get all entries in a tier."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM knowledge WHERE tier = ? ORDER BY created_at",
                (tier.value,),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]
        finally:
            conn.close()

    def get_by_scope(
        self, scope: str, tier: Optional[Tier] = None
    ) -> List[KnowledgeEntry]:
        """Get entries by scope, optionally filtered by tier."""
        conn = self._conn()
        try:
            if tier is not None:
                rows = conn.execute(
                    "SELECT * FROM knowledge WHERE scope = ? AND tier = ? "
                    "ORDER BY confidence DESC",
                    (scope, tier.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM knowledge WHERE scope = ? "
                    "ORDER BY tier, confidence DESC",
                    (scope,),
                ).fetchall()
            return [self._row_to_entry(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        scope: Optional[str] = None,
        tier: Optional[Tier] = None,
        limit: int = 10,
    ) -> List[KnowledgeEntry]:
        """
        Full-text search across knowledge entries.

        Args:
            query: Search query
            scope: Optional scope filter
            tier: Optional tier filter (default: Tier 2+3, not axioms)
            limit: Max results

        Returns:
            Entries sorted by relevance (BM25 score).
        """
        words = [
            w for w in query.split()
            if all(c.isalnum() or c in ("_", "-") for c in w) and len(w) > 1
        ]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)

        conn = self._conn()
        try:
            conditions = ["knowledge_fts MATCH ?"]
            params: list = [fts_query]

            if tier is not None:
                conditions.append("k.tier = ?")
                params.append(tier.value)
            else:
                # Default: exclude axioms from search (they're always loaded)
                conditions.append("k.tier > 1")

            if scope is not None:
                conditions.append("(k.scope = ? OR k.scope = 'global')")
                params.append(scope)

            where = " AND ".join(conditions)
            params.append(limit)

            rows = conn.execute(
                f"""
                SELECT k.*, bm25(knowledge_fts) AS score
                FROM knowledge_fts f
                JOIN knowledge k ON k.rowid = f.rowid
                WHERE {where}
                ORDER BY score
                LIMIT ?
                """,  # nosec B608 - parameterized query
                params,
            ).fetchall()

            entries = [self._row_to_entry(r) for r in rows]

            # Record access
            for entry in entries:
                conn.execute(
                    "UPDATE knowledge SET access_count = access_count + 1 "
                    "WHERE id = ?",
                    (entry.id,),
                )
            conn.commit()

            return entries
        except sqlite3.OperationalError as e:
            logger.debug(f"Knowledge search failed: {e}")
            return []
        finally:
            conn.close()

    def build_context(
        self,
        query: str,
        scope: Optional[str] = None,
        max_chars: int = 6000,
        include_axioms: bool = True,
    ) -> str:
        """
        Build a context string for LLM prompt injection.

        Assembles Tier 1 axioms + relevant Tier 2/3 results within
        a character budget.

        Args:
            query: Search query for Tier 2+3 retrieval
            scope: Optional scope filter
            max_chars: Maximum context size
            include_axioms: Whether to prepend axioms

        Returns:
            Formatted context string.
        """
        parts = []
        budget = max_chars

        if include_axioms:
            axiom_text = self.get_axioms_text()
            if axiom_text:
                section = f"[RULES]\n{axiom_text}\n"
                parts.append(section)
                budget -= len(section)

        results = self.search(query, scope=scope, limit=10)
        if results:
            parts.append("[KNOWLEDGE]")
            budget -= 12
            for entry in results:
                tier_label = {2: "IMPORTED", 3: "DERIVED"}.get(
                    entry.tier, "?"
                )
                header = (
                    f"[{tier_label}] {entry.title}"
                    f" (confidence: {entry.confidence:.0%},"
                    f" source: {entry.source})"
                )
                content = entry.content
                entry_len = len(header) + len(content) + 10

                if entry_len > budget:
                    remaining = budget - len(header) - 10
                    if remaining > 100:
                        content = content[:remaining] + "..."
                    else:
                        break

                parts.append(f"\n--- {header} ---\n{content}")
                budget -= len(header) + len(content) + 10

                if budget <= 0:
                    break

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def update_confidence(self, entry_id: str, confidence: float) -> bool:
        """Update confidence score for an entry."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE knowledge SET confidence = ?, updated_at = ? "
                "WHERE id = ?",
                (confidence, time.time(), entry_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def promote(self, entry_id: str) -> bool:
        """Promote a Tier 3 entry to Tier 2 (validated derived -> imported)."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE knowledge SET tier = ?, updated_at = ? "
                "WHERE id = ? AND tier = ?",
                (Tier.IMPORTED.value, time.time(), entry_id, Tier.DERIVED.value),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Promoted {entry_id} from Tier 3 to Tier 2")
                return True
            return False
        finally:
            conn.close()

    def demote(self, entry_id: str) -> bool:
        """Demote a Tier 2 entry to Tier 3."""
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE knowledge SET tier = ?, updated_at = ? "
                "WHERE id = ? AND tier = ?",
                (Tier.DERIVED.value, time.time(), entry_id, Tier.IMPORTED.value),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Demoted {entry_id} from Tier 2 to Tier 3")
                return True
            return False
        finally:
            conn.close()

    def prune(
        self,
        tier: Tier = Tier.DERIVED,
        max_age_days: float = 90,
        min_confidence: float = 0.3,
        min_access_count: int = 0,
    ) -> int:
        """
        Remove low-quality entries from a tier.

        Args:
            tier: Which tier to prune (default: Tier 3)
            max_age_days: Remove entries older than this
            min_confidence: Remove entries below this confidence
            min_access_count: Only prune entries accessed fewer times than this

        Returns:
            Number of entries removed.
        """
        cutoff = time.time() - (max_age_days * 86400)
        conn = self._conn()
        try:
            cursor = conn.execute(
                "DELETE FROM knowledge WHERE tier = ? AND ("
                "  (confidence < ? AND access_count <= ?) OR"
                "  (updated_at < ? AND access_count <= ?)"
                ")",
                (
                    tier.value,
                    min_confidence,
                    min_access_count,
                    cutoff,
                    min_access_count,
                ),
            )
            conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Pruned {count} entries from Tier {tier.value}")
            return count
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge store statistics."""
        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
            by_tier = {}
            for row in conn.execute(
                "SELECT tier, COUNT(*) FROM knowledge GROUP BY tier"
            ).fetchall():
                tier_name = {1: "axiom", 2: "imported", 3: "derived"}.get(
                    row[0], f"tier_{row[0]}"
                )
                by_tier[tier_name] = row[1]

            scopes = {}
            for row in conn.execute(
                "SELECT scope, COUNT(*) FROM knowledge GROUP BY scope"
            ).fetchall():
                scopes[row[0]] = row[1]

            return {
                "total_entries": total,
                "by_tier": by_tier,
                "by_scope": scopes,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=row["id"],
            tier=Tier(row["tier"]),
            title=row["title"],
            content=row["content"],
            scope=row["scope"],
            source=row["source"],
            confidence=row["confidence"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
        )
