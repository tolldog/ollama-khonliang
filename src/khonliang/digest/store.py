"""
DigestStore — SQLite-backed accumulator for digest entries.

Agents record short digest-worthy items as they work. The store
accumulates these entries over time, then a synthesizer pulls them
by time window to produce a narrative digest.

Usage:
    store = DigestStore("digest.db")

    # Record as agents produce results
    store.record("Found 1850 census for Patrick Smith", source="census_agent")
    store.record("Dead end resolved: Mary O'Brien parents", source="researcher",
                 tags=["resolved", "dead_end"])

    # Query for digest synthesis
    entries = store.get_since(hours=24)
    entries = store.get_unconsumed()

    # Mark as consumed after digest is published
    store.mark_consumed([e.id for e in entries])
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS digest_entries (
    id          TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    source      TEXT,
    audience    TEXT,           -- digest audience (e.g. "transactions", "changes", "ops")
    tags        TEXT,           -- JSON array
    metadata    TEXT,           -- JSON object
    created_at  REAL NOT NULL,
    consumed    INTEGER NOT NULL DEFAULT 0,
    consumed_at REAL,
    digest_id   TEXT            -- ID of the digest that consumed this entry
);

CREATE INDEX IF NOT EXISTS idx_digest_created ON digest_entries (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_digest_consumed ON digest_entries (consumed)
    WHERE consumed = 0;
CREATE INDEX IF NOT EXISTS idx_digest_source ON digest_entries (source);
CREATE INDEX IF NOT EXISTS idx_digest_audience ON digest_entries (audience);
"""


@dataclass
class DigestEntry:
    """A single accumulated digest item."""

    id: str
    summary: str
    source: Optional[str] = None
    audience: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    consumed: bool = False
    consumed_at: Optional[float] = None
    digest_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DigestEntry":
        return cls(
            id=row["id"],
            summary=row["summary"],
            source=row["source"],
            audience=row["audience"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            consumed=bool(row["consumed"]),
            consumed_at=row["consumed_at"],
            digest_id=row["digest_id"],
        )


class DigestStore:
    """
    SQLite-backed accumulator for digest entries.

    Args:
        db_path: Path to SQLite database. Use ":memory:" for testing.
    """

    def __init__(self, db_path: str = "digest.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def record(
        self,
        summary: str,
        source: Optional[str] = None,
        audience: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DigestEntry:
        """
        Record a digest-worthy item.

        Args:
            summary: Short one-liner describing what happened.
            source: Agent or role that produced this entry.
            audience: Target digest audience (e.g. "transactions", "changes",
                "ops"). Allows generating separate digests for different
                consumers from the same store.
            tags: Optional classification tags (e.g. ["discovery", "census"]).
            metadata: Arbitrary key-value data for context.

        Returns:
            The created DigestEntry.
        """
        entry = DigestEntry(
            id=uuid.uuid4().hex,
            summary=summary,
            source=source,
            audience=audience,
            tags=tags or [],
            metadata=metadata or {},
            created_at=time.time(),
        )

        self._conn.execute(
            """INSERT INTO digest_entries
               (id, summary, source, audience, tags, metadata, created_at, consumed)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                entry.id,
                entry.summary,
                entry.source,
                entry.audience,
                json.dumps(entry.tags) if entry.tags else None,
                json.dumps(entry.metadata) if entry.metadata else None,
                entry.created_at,
            ),
        )
        self._conn.commit()

        logger.debug(f"Digest entry recorded: [{source}] {summary[:60]}")
        return entry

    def get_since(
        self,
        hours: Optional[float] = None,
        since: Optional[float] = None,
        source: Optional[str] = None,
        audience: Optional[str] = None,
        tag: Optional[str] = None,
        include_consumed: bool = False,
    ) -> List[DigestEntry]:
        """
        Query entries from a time window.

        Args:
            hours: Get entries from the last N hours.
            since: Get entries after this Unix timestamp. Overrides hours.
            source: Filter by source agent/role.
            audience: Filter by target audience.
            tag: Filter by tag (entries containing this tag).
            include_consumed: Include already-consumed entries.

        Returns:
            List of DigestEntry, oldest first.
        """
        if since is None and hours is not None:
            since = time.time() - (hours * 3600)
        elif since is None:
            since = 0.0

        conditions = ["created_at > ?"]
        params: list = [since]

        if not include_consumed:
            conditions.append("consumed = 0")
        if source:
            conditions.append("source = ?")
            params.append(source)
        if audience:
            conditions.append("audience = ?")
            params.append(audience)

        where = " AND ".join(conditions)

        rows = self._conn.execute(
            f"SELECT * FROM digest_entries WHERE {where} ORDER BY created_at ASC",
            params,
        ).fetchall()

        entries = [DigestEntry.from_row(row) for row in rows]

        # Tag filter (post-query since tags are JSON)
        if tag:
            entries = [e for e in entries if tag in e.tags]

        return entries

    def get_unconsumed(
        self,
        source: Optional[str] = None,
        audience: Optional[str] = None,
        limit: int = 200,
    ) -> List[DigestEntry]:
        """
        Get all unconsumed entries, oldest first.

        Args:
            source: Filter by source.
            audience: Filter by target audience.
            limit: Maximum entries to return.
        """
        conditions = ["consumed = 0"]
        params: list = []

        if source:
            conditions.append("source = ?")
            params.append(source)
        if audience:
            conditions.append("audience = ?")
            params.append(audience)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM digest_entries WHERE {where} ORDER BY created_at ASC LIMIT ?",
            params,
        ).fetchall()

        return [DigestEntry.from_row(row) for row in rows]

    def mark_consumed(
        self,
        entry_ids: List[str],
        digest_id: Optional[str] = None,
    ) -> int:
        """
        Mark entries as consumed by a digest.

        Args:
            entry_ids: List of entry IDs to mark.
            digest_id: Optional digest report ID for traceability.

        Returns:
            Number of entries updated.
        """
        if not entry_ids:
            return 0

        now = time.time()
        placeholders = ",".join("?" for _ in entry_ids)
        params = [now, digest_id] + entry_ids

        cursor = self._conn.execute(
            f"""UPDATE digest_entries
                SET consumed = 1, consumed_at = ?, digest_id = ?
                WHERE id IN ({placeholders})""",
            params,
        )
        self._conn.commit()

        count = cursor.rowcount
        logger.debug(f"Marked {count} digest entries as consumed")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM digest_entries").fetchone()[0]
        unconsumed = self._conn.execute(
            "SELECT COUNT(*) FROM digest_entries WHERE consumed = 0"
        ).fetchone()[0]

        by_source = {}
        rows = self._conn.execute(
            "SELECT source, COUNT(*) as cnt FROM digest_entries WHERE consumed = 0 GROUP BY source"
        ).fetchall()
        for row in rows:
            by_source[row["source"] or "unknown"] = row["cnt"]

        return {
            "total_entries": total,
            "unconsumed": unconsumed,
            "by_source": by_source,
        }

    def purge_consumed(self, older_than_hours: float = 168) -> int:
        """
        Delete consumed entries older than a threshold (default 7 days).

        Returns count deleted.
        """
        cutoff = time.time() - (older_than_hours * 3600)
        cursor = self._conn.execute(
            "DELETE FROM digest_entries WHERE consumed = 1 AND consumed_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        count = cursor.rowcount
        if count:
            logger.info(f"Purged {count} consumed digest entries")
        return count

    def close(self) -> None:
        self._conn.close()
