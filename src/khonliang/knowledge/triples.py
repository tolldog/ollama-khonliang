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
import re
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
        """Serialize triple to a plain dict."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
        }


def normalize_predicate(predicate: str, aliases: Optional[Dict[str, str]] = None) -> str:
    """Normalize a predicate string to canonical form.

    Steps:
        1. Strip whitespace, lowercase
        2. Replace spaces/hyphens with underscores
        3. Strip leading "is_" / trailing "_of" variants
        4. Apply explicit aliases if provided

    Examples:
        "Is Applicable To"  → "applicable_to"
        "is_applicable_to"  → "applicable_to"
        "used for"          → "used_for"
        "Uses Method"       → "uses_method"
    """
    p = predicate.strip().lower()
    p = re.sub(r"[\s\-]+", "_", p)
    # Strip common prefixes that add no meaning
    p = re.sub(r"^is_", "", p)
    # Apply aliases
    if aliases and p in aliases:
        p = aliases[p]
    return p


class TripleStore:
    """
    SQLite-backed semantic triple store with confidence decay.

    Stores (subject, predicate, object) tuples. Duplicate triples
    reinforce confidence instead of creating new rows.

    Predicates are auto-normalized on add/query: lowercased, underscored,
    with common prefixes stripped. Use ``predicate_aliases`` to map
    domain-specific synonyms to canonical forms.

    Example:
        store = TripleStore("knowledge.db", predicate_aliases={
            "used_for": "uses_method",
            "achieves": "outperforms",
        })
        store.add("TSLA", "correlates_with", "AMD", confidence=0.8)
        store.add("X", "Is Applicable To", "Y")  # stored as "applicable_to"
        triples = store.get("TSLA")
        context = store.build_context(subjects=["TSLA"])
    """

    def __init__(
        self,
        db_path: str = "data/knowledge.db",
        default_decay_rate: float = 0.01,
        predicate_aliases: Optional[Dict[str, str]] = None,
    ):
        self.db_path = db_path
        self.default_decay_rate = default_decay_rate
        self.predicate_aliases = predicate_aliases or {}
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

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize a predicate using the store's aliases."""
        return normalize_predicate(predicate, self.predicate_aliases)

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

        The predicate is auto-normalized (lowercased, underscored,
        aliases applied) before storage.

        If the triple already exists, confidence is updated to the
        max of current and new, and the timestamp is refreshed.
        """
        predicate = self._normalize_predicate(predicate)
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
            params.append(self._normalize_predicate(predicate))
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

        Uses a single SQL query with ORDER BY confidence DESC LIMIT to
        avoid loading the full table into memory.
        """
        query = (
            "SELECT id, subject, predicate, object, confidence "
            "FROM triples WHERE confidence >= ?"
        )
        params: List[Any] = [min_confidence]

        if subjects:
            placeholders = ",".join("?" for _ in subjects)
            query += f" AND subject IN ({placeholders})"  # nosec B608 - only ? chars, values passed as params
            params.extend(subjects)
        elif predicates:
            placeholders = ",".join("?" for _ in predicates)
            query += f" AND predicate IN ({placeholders})"  # nosec B608 - only ? chars, values passed as params
            params.extend(predicates)

        query += " ORDER BY confidence DESC LIMIT ?"
        params.append(max_triples)

        conn = self._conn()
        try:
            rows = conn.execute(query, params).fetchall()

            # Increment access_count only for the rows actually returned
            triple_ids = [row["id"] for row in rows]
            if triple_ids:
                id_placeholders = ",".join("?" for _ in triple_ids)
                conn.execute(
                    "UPDATE triples SET access_count = access_count + 1 "
                    f"WHERE id IN ({id_placeholders})",  # nosec B608 - placeholders from integer DB ids, values passed as params
                    triple_ids,
                )
                conn.commit()

            lines = []
            for row in rows:
                lines.append(
                    f"{row['subject']} {row['predicate']} {row['object']}"
                    f" ({row['confidence']:.0%})"
                )
            return "\n".join(lines)
        finally:
            conn.close()

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
            params.append(self._normalize_predicate(predicate))
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
        """Return triple store statistics (counts of triples, subjects, predicates)."""
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
