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

-- Per-source provenance with its own confidence, so a fact asserted by several
-- contributors keeps each one's strength independently. The triples row carries
-- denormalized caches (``source`` = newline-joined tokens, ``confidence`` = max
-- over sources) for backward-compatible reads; this table is the source of
-- truth used to recompute them when a source is added or retracted
-- (bug_khonliang-researcher_a905176b).
CREATE TABLE IF NOT EXISTS triple_sources (
    triple_id   INTEGER NOT NULL,
    source      TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    PRIMARY KEY (triple_id, source)
);

CREATE INDEX IF NOT EXISTS idx_triple_sources_tid ON triple_sources(triple_id);
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

    @property
    def sources(self) -> List[str]:
        """Provenance tokens for this triple (``source`` may hold several).

        ``source`` stays a plain string for backward compatibility; use this
        when a triple can have more than one contributor (e.g. a fact asserted
        by both a paper and a blog) so callers don't have to know the
        on-disk encoding.
        """
        return split_sources(self.source)

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


# A triple's ``source`` column holds a *set* of contributor tokens (e.g. two
# different documents that each asserted the same fact), newline-joined. A
# single-source triple is stored as the bare token with no separator, so legacy
# rows and the common case are byte-for-byte unchanged.
_SOURCE_SEP = "\n"


def split_sources(raw: Optional[str]) -> List[str]:
    """Parse a stored ``source`` value into its ordered, de-duped token list."""
    if not raw:
        return []
    out: List[str] = []
    for part in raw.split(_SOURCE_SEP):
        tok = part.strip()
        if tok and tok not in out:
            out.append(tok)
    return out


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
            # Backfill provenance for legacy triples written before the
            # per-source table existed, so the "every triple has >=1 source row"
            # invariant holds: one ``triple_sources`` row per token in the
            # denormalized ``source`` string (or a single anonymous "" row when
            # the triple had no source), seeded with the triple's confidence.
            # Idempotent — only triples with no rows yet are touched, so it's a
            # no-op on every construction after the first.
            legacy = conn.execute(
                "SELECT id, source, confidence FROM triples "
                "WHERE id NOT IN (SELECT DISTINCT triple_id FROM triple_sources)"
            ).fetchall()
            for row in legacy:
                tokens = split_sources(row["source"]) or [""]
                for tok in tokens:
                    conn.execute(
                        "INSERT OR IGNORE INTO triple_sources "
                        "(triple_id, source, confidence) VALUES (?, ?, ?)",
                        (row["id"], tok, row["confidence"]),
                    )
            conn.commit()
        finally:
            conn.close()

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize a predicate using the store's aliases."""
        return normalize_predicate(predicate, self.predicate_aliases)

    @staticmethod
    def _resync_denormalized(conn, triple_id: int, *, touch_updated: bool) -> bool:
        """Refresh a triple's cached ``source``/``confidence`` from its sources.

        Returns False if the triple still has sources (row kept); True if it has
        none left and the triple row was deleted. ``touch_updated`` controls
        whether ``updated_at`` is bumped — adds reinforce (bump), source removals
        must not, or a retraction would reset the decay clock.
        """
        rows = conn.execute(
            "SELECT source, confidence FROM triple_sources "
            "WHERE triple_id = ? ORDER BY rowid",
            (triple_id,),
        ).fetchall()
        if not rows:
            conn.execute("DELETE FROM triples WHERE id = ?", (triple_id,))
            return True
        # Anonymous ("") rows count toward confidence but carry no display token.
        new_source = _SOURCE_SEP.join(r["source"] for r in rows if r["source"])
        new_conf = max(r["confidence"] for r in rows)
        if touch_updated:
            conn.execute(
                "UPDATE triples SET source = ?, confidence = ?, updated_at = ? "
                "WHERE id = ?",
                (new_source, new_conf, time.time(), triple_id),
            )
        else:
            conn.execute(
                "UPDATE triples SET source = ?, confidence = ? WHERE id = ?",
                (new_source, new_conf, triple_id),
            )
        return False

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

        If the triple already exists the ``source`` is *unioned* into its
        provenance (rather than overwriting it), each source keeping its own
        confidence; the triple's cached ``confidence`` is the max across
        sources and the timestamp is refreshed. A later :meth:`remove_source`
        can then drop one contributor — and recompute confidence from those
        that remain — without losing the fact (bug a905176b).
        """
        predicate = self._normalize_predicate(predicate)
        now = time.time()
        token = (source or "").strip()
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT id, confidence FROM triples "
                "WHERE subject = ? AND predicate = ? AND object = ?",
                (subject, predicate, obj),
            ).fetchone()

            if existing:
                triple_id = existing["id"]
                conn.execute(
                    "UPDATE triples SET access_count = access_count + 1 "
                    "WHERE id = ?",
                    (triple_id,),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO triples "
                    "(subject, predicate, object, confidence, source, "
                    "created_at, updated_at, decay_rate) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (subject, predicate, obj, confidence, token,
                     now, now, self.default_decay_rate),
                )
                triple_id = cur.lastrowid

            # Every triple carries its evidence in ``triple_sources`` — even an
            # anonymous (source="") assertion gets a row — so the invariant
            # "every triple has >=1 provenance row" holds. That lets decay act
            # on the authoritative per-source confidences and stops a later
            # sourced add from discarding earlier anonymous evidence. A source
            # re-asserting its own claim keeps the stronger value (max).
            self._upsert_source(conn, triple_id, token, confidence)
            self._resync_denormalized(conn, triple_id, touch_updated=True)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _upsert_source(conn, triple_id: int, source: str, confidence: float) -> None:
        """Insert or strengthen (max) a single source's confidence for a triple.

        ``source`` may be ``""`` — the provenance row for an anonymous
        assertion, so its confidence is tracked and decayed like any other.
        """
        conn.execute(
            "INSERT INTO triple_sources (triple_id, source, confidence) "
            "VALUES (?, ?, ?) ON CONFLICT(triple_id, source) DO UPDATE SET "
            "confidence = MAX(confidence, excluded.confidence)",
            (triple_id, source, confidence),
        )

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
            # Decay the authoritative per-source confidences (not just the
            # cached triples.confidence) so decay survives a later add/
            # remove_source recompute. Each source decays by its triple's
            # decay_rate. Then refresh the cached max from the decayed rows.
            stale = "(SELECT id FROM triples WHERE updated_at < ? AND decay_rate > 0)"
            conn.execute(
                "UPDATE triple_sources SET confidence = confidence * (1 - "
                "(SELECT decay_rate FROM triples WHERE triples.id = "
                f"triple_sources.triple_id)) WHERE triple_id IN {stale}",  # nosec B608
                (cutoff,),
            )
            conn.execute(
                "UPDATE triples SET confidence = (SELECT MAX(confidence) "
                "FROM triple_sources WHERE triple_id = triples.id) "
                f"WHERE id IN {stale}",  # nosec B608
                (cutoff,),
            )
            # Remove very low confidence triples
            cursor = conn.execute(
                "DELETE FROM triples WHERE confidence < 0.1"
            )
            self._prune_orphan_sources(conn)
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
            self._prune_orphan_sources(conn)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    @staticmethod
    def _prune_orphan_sources(conn) -> None:
        """Drop provenance rows whose triple was hard-deleted (no FK cascade)."""
        conn.execute(
            "DELETE FROM triple_sources WHERE triple_id NOT IN "
            "(SELECT id FROM triples)"
        )

    def remove_source(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str,
    ) -> bool:
        """Drop one provenance token from a triple, deleting it only if last.

        The counterpart to :meth:`add`'s source union: when a contributor is
        retracted (e.g. a paper is struck), remove just *its* token. The triple
        survives as long as any other source still asserts it, and is deleted
        only when ``source`` was its sole provenance.

        Returns ``True`` if the triple row was deleted (that was its last
        source), ``False`` if it was kept (other sources remain) or nothing
        matched (no such triple, or it didn't carry ``source``). Use
        :meth:`remove` for an unconditional delete regardless of provenance.
        """
        predicate = self._normalize_predicate(predicate)
        token = (source or "").strip()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id FROM triples "
                "WHERE subject = ? AND predicate = ? AND object = ?",
                (subject, predicate, obj),
            ).fetchone()
            if row is None:
                return False
            cur = conn.execute(
                "DELETE FROM triple_sources WHERE triple_id = ? AND source = ?",
                (row["id"], token),
            )
            if cur.rowcount == 0:
                return False  # this source never asserted the triple
            # Recompute confidence from the survivors (so retracting the
            # strongest source lowers the fact); don't touch updated_at — a
            # retraction must not reset the decay clock.
            deleted = self._resync_denormalized(conn, row["id"], touch_updated=False)
            conn.commit()
            return deleted
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
