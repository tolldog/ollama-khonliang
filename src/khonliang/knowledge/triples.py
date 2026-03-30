"""
Semantic triple store — subject-predicate-object knowledge representation.

Compact, queryable knowledge that uses 95% fewer tokens than full context.
Triples have confidence scores that decay over time and get reinforced
through repeated observation.

Based on Memori (arxiv:2603.19935).

Usage:
    store = TripleStore("knowledge.db")

    store.add("Roger Tolle", "born_in", "Wales", confidence=0.9, source="gedcom")
    store.add("Roger Tolle", "born_year", "1642", confidence=0.95, source="gedcom")
    store.add("Roger Tolle", "migrated_to", "Maryland", confidence=0.8, source="research")

    triples = store.get("Roger Tolle")
    context = store.build_context(subjects=["Roger Tolle"], max_triples=10)
    # "Roger Tolle born_in Wales (0.9). Roger Tolle born_year 1642 (0.95). ..."
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRIPLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS triples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    confidence      REAL DEFAULT 1.0,
    source          TEXT DEFAULT '',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    access_count    INTEGER DEFAULT 0,
    decay_rate      REAL DEFAULT 0.0,
    UNIQUE(subject, predicate, object)
);

CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
"""


@dataclass
class Triple:
    """A single subject-predicate-object fact."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
        }


class TripleStore:
    """
    SQLite-backed semantic triple store with confidence decay.

    Stores (subject, predicate, object) tuples. Duplicate triples
    reinforce confidence instead of creating new rows.

    Example:
        store = TripleStore("knowledge.db")
        store.add("TSLA", "correlates_with", "AMD", confidence=0.8)
        store.add("user", "prefers", "conservative positions", confidence=0.9)
        triples = store.get("TSLA")
        context = store.build_context(subjects=["TSLA"])

    .. todo:: Add unit tests for TripleStore — CRUD, confidence reinforcement,
       decay, build_context limits, and concurrent access.
    """

    def __init__(
        self,
        db_path: str = "data/knowledge.db",
        default_decay_rate: float = 0.01,
    ):
        self.db_path = db_path
        self.default_decay_rate = default_decay_rate
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(_TRIPLE_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """
        Add or reinforce a triple.

        If the triple already exists, confidence is updated to the
        max of current and new, and the timestamp is refreshed.
        """
        now = time.time()
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT id, confidence FROM triples "
                "WHERE subject = ? AND predicate = ? AND object = ?",
                (subject, predicate, obj),
            ).fetchone()

            if existing:
                new_confidence = max(existing["confidence"], confidence)
                conn.execute(
                    "UPDATE triples SET confidence = ?, updated_at = ?, "
                    "source = ?, access_count = access_count + 1 "
                    "WHERE id = ?",
                    (new_confidence, now, source, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO triples "
                    "(subject, predicate, object, confidence, source, "
                    "created_at, updated_at, decay_rate) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (subject, predicate, obj, confidence, source,
                     now, now, self.default_decay_rate),
                )
            conn.commit()
        finally:
            conn.close()

    def get(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[Triple]:
        """Query triples by any combination of fields.

        Args:
            subject: Filter by subject.
            predicate: Filter by predicate.
            obj: Filter by object.
            min_confidence: Minimum confidence threshold.
            limit: Maximum number of triples to return. None means no limit.
        """
        conditions = []
        params: list = []

        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if obj:
            conditions.append("object = ?")
            params.append(obj)
        if min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""

        conn = self._conn()
        try:
            rows = conn.execute(
                f"SELECT * FROM triples WHERE {where} "  # nosec B608
                f"ORDER BY confidence DESC{limit_clause}",
                params,
            ).fetchall()

            # Update access counts
            for row in rows:
                conn.execute(
                    "UPDATE triples SET access_count = access_count + 1 "
                    "WHERE id = ?",
                    (row["id"],),
                )
            conn.commit()

            return [self._row_to_triple(r) for r in rows]
        finally:
            conn.close()

    def search(self, query: str, limit: int = 20) -> List[Triple]:
        """Search triples by keyword across all fields."""
        pattern = f"%{query}%"
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM triples "
                "WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? "
                "ORDER BY confidence DESC LIMIT ?",
                (pattern, pattern, pattern, limit),
            ).fetchall()
            return [self._row_to_triple(r) for r in rows]
        finally:
            conn.close()

    def build_context(
        self,
        subjects: Optional[List[str]] = None,
        predicates: Optional[List[str]] = None,
        max_triples: int = 20,
        min_confidence: float = 0.3,
    ) -> str:
        """
        Build a compact context string for prompt injection.

        Much more token-efficient than full text — each triple is
        one line of "subject predicate object (confidence)".
        """
        all_triples: List[Triple] = []

        if subjects:
            for s in subjects:
                all_triples.extend(
                    self.get(subject=s, min_confidence=min_confidence,
                             limit=max_triples)
                )
        elif predicates:
            for p in predicates:
                all_triples.extend(
                    self.get(predicate=p, min_confidence=min_confidence,
                             limit=max_triples)
                )
        else:
            all_triples = self.get(min_confidence=min_confidence)

        # Deduplicate and sort by confidence
        seen = set()
        unique = []
        for t in all_triples:
            key = (t.subject, t.predicate, t.object)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        unique.sort(key=lambda t: -t.confidence)

        lines = []
        for t in unique[:max_triples]:
            lines.append(
                f"{t.subject} {t.predicate} {t.object} ({t.confidence:.0%})"
            )

        return "\n".join(lines)

    def apply_decay(self, max_age_days: float = 90) -> int:
        """
        Apply confidence decay to old triples.

        Triples not updated within ``max_age_days`` lose confidence
        (scaled by their per-row ``decay_rate``). Those that drop
        below 0.1 are removed.

        Note: decay is based on ``updated_at``, which is refreshed on
        add/reinforce. ``access_count`` tracks reads but does not
        currently prevent decay.

        Returns number of triples removed.
        """
        now = time.time()
        cutoff = now - (max_age_days * 86400)
        conn = self._conn()
        try:
            # Decay confidence based on age since last update
            conn.execute(
                "UPDATE triples SET confidence = confidence * (1 - decay_rate) "
                "WHERE updated_at < ? AND decay_rate > 0",
                (cutoff,),
            )
            # Remove very low confidence triples
            cursor = conn.execute(
                "DELETE FROM triples WHERE confidence < 0.1"
            )
            conn.commit()
            removed = cursor.rowcount
            if removed:
                logger.info(f"Decayed and removed {removed} low-confidence triples")
            return removed
        finally:
            conn.close()

    def remove(
        self,
        subject: str,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        """Remove triples matching criteria. Returns count removed."""
        conditions = ["subject = ?"]
        params: list = [subject]
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if obj:
            conditions.append("object = ?")
            params.append(obj)

        conn = self._conn()
        try:
            cursor = conn.execute(
                f"DELETE FROM triples WHERE {' AND '.join(conditions)}",  # nosec B608
                params,
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            subjects = conn.execute(
                "SELECT COUNT(DISTINCT subject) FROM triples"
            ).fetchone()[0]
            predicates = conn.execute(
                "SELECT COUNT(DISTINCT predicate) FROM triples"
            ).fetchone()[0]
            return {
                "total_triples": total,
                "unique_subjects": subjects,
                "unique_predicates": predicates,
            }
        finally:
            conn.close()

    @staticmethod
    def _row_to_triple(row: sqlite3.Row) -> Triple:
        return Triple(
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
        )
