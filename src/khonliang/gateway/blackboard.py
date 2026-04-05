"""
Blackboard — shared knowledge store for agent coordination.

Agents post structured entries to named sections with optional TTL.
Other agents (or roles) read sections to build shared context.

Supports two optional upgrades (both opt-in, backward compatible):
  - **Persistence**: SQLite backend for history and cold-start recovery
  - **Embeddings**: Vector similarity search across entries

Example (in-memory, default):

    board = Blackboard(default_ttl=120)
    board.post("analyst", "market", "trend", "Bearish divergence on RSI")
    entries = board.read("market")

Example (persistent + embeddings):

    board = Blackboard(persist_to="data/blackboard.db", default_ttl=120)
    board.post("analyst", "signals", "TSLA",
               "RSI oversold at 28", embedding=[0.12, -0.34, ...])
    similar = board.search_similar(embedding=[...], threshold=0.7)
    history = board.history("signals", limit=50)
"""

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BLACKBOARD_SCHEMA = """
CREATE TABLE IF NOT EXISTS blackboard_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    section     TEXT NOT NULL,
    key         TEXT NOT NULL,
    content     TEXT,
    embedding   TEXT,
    timestamp   REAL NOT NULL,
    ttl         REAL,
    expired     INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_bb_section ON blackboard_entries(section);
CREATE INDEX IF NOT EXISTS idx_bb_section_key ON blackboard_entries(section, key);
CREATE INDEX IF NOT EXISTS idx_bb_expired ON blackboard_entries(expired);
CREATE INDEX IF NOT EXISTS idx_bb_timestamp ON blackboard_entries(timestamp);
"""


@dataclass
class BlackboardEntry:
    """A single entry on the blackboard."""

    agent_id: str
    section: str
    key: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    embedding: Optional[List[float]] = None

    @property
    def expires_at(self) -> Optional[float]:
        """Absolute expiry time, or None if no TTL."""
        if self.ttl is not None:
            return self.timestamp + self.ttl
        return None

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this entry has expired."""
        if self.ttl is None:
            return False
        return (now or time.time()) > self.timestamp + self.ttl


class Blackboard:
    """
    Shared blackboard for multi-agent coordination.

    Agents post key-value entries to named sections. Entries auto-expire
    based on TTL.

    Args:
        default_ttl: Default time-to-live in seconds. None = never expire.
        persist_to: Optional SQLite path. Enables persistence + history.
    """

    def __init__(
        self,
        default_ttl: Optional[float] = None,
        persist_to: Optional[str] = None,
    ):
        self.default_ttl = default_ttl
        self._persist_to = persist_to
        # In-memory store: {section: {key: BlackboardEntry}}
        self._sections: Dict[str, Dict[str, BlackboardEntry]] = {}
        self._pending_expirations: List[tuple] = []

        if persist_to:
            self._ensure_schema()
            self._load_from_db()

    # ------------------------------------------------------------------
    # Core API (unchanged signatures)
    # ------------------------------------------------------------------

    def post(
        self,
        agent_id: str,
        section: str,
        key: str,
        content: Any,
        ttl: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Post an entry to the blackboard.

        If a key already exists in the section, it is overwritten.

        Args:
            agent_id: ID of the posting agent
            section: Section name (e.g. "market", "signals")
            key: Entry key within the section
            content: The content to store (string, dict, etc.)
            ttl: Time-to-live in seconds. None uses the board default.
            embedding: Optional vector for similarity search.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        entry = BlackboardEntry(
            agent_id=agent_id,
            section=section,
            key=key,
            content=content,
            ttl=effective_ttl,
            embedding=embedding,
        )

        if section not in self._sections:
            self._sections[section] = {}
        self._sections[section][key] = entry

        if self._persist_to:
            self._persist_entry(entry)

        logger.debug(
            "Blackboard: %s posted to %s/%s (ttl=%s, emb=%s)",
            agent_id, section, key, effective_ttl,
            "yes" if embedding else "no",
        )

    def read(
        self, section: str, key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Read entries from a section.

        Returns dict mapping keys to content. Expired entries are
        excluded and cleaned up.
        """
        entries = self._sections.get(section, {})
        now = time.time()
        result: Dict[str, Any] = {}

        if key is not None:
            entry = entries.get(key)
            if entry and not entry.is_expired(now):
                result[key] = entry.content
            elif entry and entry.is_expired(now):
                self._mark_expired(entry)
                del entries[key]
            return result

        expired_keys = []
        for k, entry in entries.items():
            if entry.is_expired(now):
                expired_keys.append(k)
            else:
                result[k] = entry.content

        for k in expired_keys:
            self._mark_expired(entries[k])
            del entries[k]

        self._flush_expirations()
        return result

    def build_context(
        self,
        sections: Optional[List[str]] = None,
        max_entries: int = 50,
    ) -> str:
        """Build a formatted context string from blackboard entries."""
        target_sections = sections or list(self._sections.keys())
        now = time.time()

        for section in target_sections:
            entries = self._sections.get(section, {})
            expired_keys = [k for k, e in entries.items() if e.is_expired(now)]
            for k in expired_keys:
                self._mark_expired(entries[k])
                del entries[k]

        self._flush_expirations()
        lines: List[str] = []
        count = 0

        for section in target_sections:
            entries = self._sections.get(section, {})
            section_lines: List[str] = []

            for key, entry in entries.items():
                if count >= max_entries:
                    break
                section_lines.append(
                    f"  [{key}] ({entry.agent_id}): {entry.content}"
                )
                count += 1

            if section_lines:
                lines.append(f"[{section}]")
                lines.extend(section_lines)

            if count >= max_entries:
                break

        return "\n".join(lines)

    def clear_section(self, section: str) -> None:
        """Remove all entries from a section."""
        self._sections.pop(section, None)

    def clear(self) -> None:
        """Remove all entries from all sections (memory + DB)."""
        self._sections.clear()
        self._pending_expirations.clear()
        if self._persist_to:
            try:
                conn = self._db_conn()
                try:
                    conn.execute(
                        "UPDATE blackboard_entries SET expired = 1 "
                        "WHERE expired = 0"
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                logger.warning("Blackboard DB clear failed: %s", e)

    @property
    def sections(self) -> List[str]:
        """Return list of section names that have entries."""
        return list(self._sections.keys())

    # ------------------------------------------------------------------
    # History (KH-5 — requires persistence)
    # ------------------------------------------------------------------

    def history(
        self,
        section: str,
        key: Optional[str] = None,
        limit: int = 50,
    ) -> List[BlackboardEntry]:
        """Return historical entries including expired ones.

        Requires persist_to. Returns entries ordered by timestamp desc.
        """
        if not self._persist_to:
            # Fallback: return current live entries from memory
            entries = self._sections.get(section, {})
            result = list(entries.values())
            if key:
                result = [e for e in result if e.key == key]
            return sorted(result, key=lambda e: e.timestamp, reverse=True)[:limit]

        conn = self._db_conn()
        try:
            if key:
                rows = conn.execute(
                    "SELECT * FROM blackboard_entries "
                    "WHERE section = ? AND key = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (section, key, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM blackboard_entries "
                    "WHERE section = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (section, limit),
                ).fetchall()
            return [self._row_to_entry(r) for r in rows]
        finally:
            conn.close()

    def query(
        self,
        predicate: Callable[[BlackboardEntry], bool],
        section: Optional[str] = None,
    ) -> List[BlackboardEntry]:
        """Filter entries by a callable predicate.

        Searches live in-memory entries. For historical queries,
        use history() instead.
        """
        results = []
        target = (
            {section: self._sections.get(section, {})}
            if section
            else self._sections
        )
        now = time.time()
        for sec_entries in target.values():
            for entry in sec_entries.values():
                if not entry.is_expired(now) and predicate(entry):
                    results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Embedding search (KH-7)
    # ------------------------------------------------------------------

    def search_similar(
        self,
        embedding: List[float],
        threshold: float = 0.7,
        limit: int = 5,
        section: Optional[str] = None,
    ) -> List[Tuple[BlackboardEntry, float]]:
        """Find entries similar to the given embedding.

        Args:
            embedding: Query vector
            threshold: Minimum cosine similarity (0-1)
            limit: Max results
            section: Optional section filter

        Returns:
            List of (BlackboardEntry, similarity_score) tuples,
            sorted by similarity descending.
        """
        results = []
        now = time.time()
        target = (
            {section: self._sections.get(section, {})}
            if section
            else self._sections
        )

        for sec_entries in target.values():
            for entry in sec_entries.values():
                if entry.is_expired(now) or entry.embedding is None:
                    continue
                sim = _cosine_similarity(embedding, entry.embedding)
                if sim >= threshold:
                    results.append((entry, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def similarity(self, key_a: str, key_b: str) -> Optional[float]:
        """Compute cosine similarity between two entries by key.

        Keys are in "section:key" format or just "key" (searches all sections).
        Returns None if either entry lacks an embedding.
        """
        entry_a = self._find_entry(key_a)
        entry_b = self._find_entry(key_b)

        if not entry_a or not entry_b:
            return None
        if entry_a.embedding is None or entry_b.embedding is None:
            return None

        return _cosine_similarity(entry_a.embedding, entry_b.embedding)

    # ------------------------------------------------------------------
    # Persistence internals
    # ------------------------------------------------------------------

    def _db_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._persist_to)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._db_conn()
        try:
            conn.executescript(_BLACKBOARD_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _load_from_db(self) -> None:
        """Load live (non-expired) entries from DB on cold start."""
        conn = self._db_conn()
        try:
            now = time.time()
            rows = conn.execute(
                "SELECT * FROM blackboard_entries WHERE expired = 0"
            ).fetchall()
            loaded = 0
            for row in rows:
                entry = self._row_to_entry(row)
                if entry.is_expired(now):
                    conn.execute(
                        "UPDATE blackboard_entries SET expired = 1 "
                        "WHERE section = ? AND key = ?",
                        (entry.section, entry.key),
                    )
                    continue
                if entry.section not in self._sections:
                    self._sections[entry.section] = {}
                self._sections[entry.section][entry.key] = entry
                loaded += 1
            conn.commit()
            if loaded:
                logger.info("Blackboard loaded %d entries from DB", loaded)
        finally:
            conn.close()

    def _persist_entry(self, entry: BlackboardEntry) -> None:
        """Write entry to DB. Expires previous entry for same section/key
        (preserving it for history) and inserts the new one."""
        try:
            content_json = _serialize_content(entry.content)
            embedding_json = (
                json.dumps(entry.embedding) if entry.embedding else None
            )
            conn = self._db_conn()
            try:
                # Expire previous entry (keeps it for history)
                conn.execute(
                    "UPDATE blackboard_entries SET expired = 1 "
                    "WHERE section = ? AND key = ? AND expired = 0",
                    (entry.section, entry.key),
                )
                # Insert new entry
                conn.execute(
                    "INSERT INTO blackboard_entries "
                    "(agent_id, section, key, content, embedding, "
                    "timestamp, ttl, expired) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        entry.agent_id,
                        entry.section,
                        entry.key,
                        content_json,
                        embedding_json,
                        entry.timestamp,
                        entry.ttl,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Blackboard persist failed: %s", e)

    def _mark_expired(self, entry: BlackboardEntry) -> None:
        """Mark an entry as expired in the DB (keeps it for history)."""
        if not self._persist_to:
            return
        self._pending_expirations.append((entry.section, entry.key))

    def _flush_expirations(self) -> None:
        """Batch-write pending expirations to DB."""
        if not self._persist_to or not self._pending_expirations:
            return
        try:
            conn = self._db_conn()
            try:
                for section, key in self._pending_expirations:
                    conn.execute(
                        "UPDATE blackboard_entries SET expired = 1 "
                        "WHERE section = ? AND key = ? AND expired = 0",
                        (section, key),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Blackboard expiration flush failed: %s", e)
        self._pending_expirations.clear()

    def _find_entry(self, key_spec: str) -> Optional[BlackboardEntry]:
        """Find an entry by 'section:key' or just 'key'."""
        if ":" in key_spec:
            section, key = key_spec.split(":", 1)
            entries = self._sections.get(section, {})
            return entries.get(key)
        # Search all sections
        for sec_entries in self._sections.values():
            if key_spec in sec_entries:
                return sec_entries[key_spec]
        return None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> BlackboardEntry:
        content = row["content"]
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except (json.JSONDecodeError, TypeError):
                pass

        return BlackboardEntry(
            agent_id=row["agent_id"],
            section=row["section"],
            key=row["key"],
            content=content,
            timestamp=row["timestamp"],
            ttl=row["ttl"],
            embedding=embedding,
        )


def _serialize_content(content: Any) -> str:
    """Serialize content to a JSON string for DB storage."""
    if isinstance(content, str):
        return json.dumps(content)  # Wrap in quotes for consistent round-trip
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return json.dumps(str(content))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
