"""
Heuristic pool — distilled rules from outcome data.

Records outcomes (success/failure with context), extracts patterns
via LLM reflection, and injects top heuristics into agent prompts.

Based on Experiential Reflective Learning (arxiv:2603.24639).

Usage:
    pool = HeuristicPool("knowledge.db")

    # Record outcomes
    pool.record_outcome(
        action="approve",
        context={"type": "urgent", "confidence": 0.9},
        result="success",
    )

    # Extract heuristics (run periodically)
    heuristics = pool.extract(min_samples=10)

    # Build prompt injection
    context = pool.build_prompt_context(max_rules=5)
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_HEURISTIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,
    context     TEXT,
    result      TEXT NOT NULL,
    details     TEXT,
    source      TEXT DEFAULT '',
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS heuristics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rule            TEXT NOT NULL UNIQUE,
    confidence      REAL DEFAULT 0.5,
    sample_count    INTEGER DEFAULT 0,
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    source          TEXT DEFAULT 'extracted',
    created_at      REAL NOT NULL,
    last_validated  REAL NOT NULL,
    decay_rate      REAL DEFAULT 0.01
);
"""


@dataclass
class Heuristic:
    """A distilled rule from outcome data."""

    rule: str
    confidence: float = 0.5
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    source: str = "extracted"
    created_at: float = 0.0
    last_validated: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "success_rate": self.success_rate,
        }


class HeuristicPool:
    """
    Records outcomes and extracts reusable heuristics.

    The extraction step can use an LLM to summarize patterns from
    raw outcome data (reflective learning). Or use the simpler
    statistical extraction for pattern counting.

    Args:
        db_path: SQLite database path
        extractor: Optional LLM-backed extraction function.
            Signature: (outcomes: List[Dict]) -> List[str]
            Returns list of rule strings.
    """

    def __init__(
        self,
        db_path: str = "data/knowledge.db",
        extractor: Optional[Callable] = None,
        default_decay_rate: float = 0.01,
    ):
        self.db_path = db_path
        self.extractor = extractor
        self.default_decay_rate = default_decay_rate
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(_HEURISTIC_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def record_outcome(
        self,
        action: str,
        result: str,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> int:
        """
        Record an outcome for future heuristic extraction.

        Args:
            action: What was done (e.g. "approve", "buy", "escalate")
            result: Outcome — must be "success" or "failure". Any other
                value is normalized: truthy/positive strings map to
                "success", everything else maps to "failure".
            context: Conditions when the action was taken
            details: Outcome details (e.g. metrics, scores)
            source: Which agent recorded this

        Returns:
            Outcome ID
        """
        # Normalize result to success/failure
        _success_values = {"success", "ok", "true", "yes", "1", "pass", "passed"}
        normalized = "success" if result.lower().strip() in _success_values else "failure"

        conn = self._conn()
        try:
            cur = conn.execute(
                "INSERT INTO outcomes (action, context, result, details, "
                "source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    action,
                    json.dumps(context) if context else None,
                    normalized,
                    json.dumps(details) if details else None,
                    source,
                    time.time(),
                ),
            )
            conn.commit()
            return cur.lastrowid or 0
        finally:
            conn.close()

    def add_heuristic(
        self,
        rule: str,
        confidence: float = 0.5,
        sample_count: int = 0,
        success_count: int = 0,
        failure_count: int = 0,
        source: str = "manual",
    ) -> None:
        """Add or update a heuristic rule."""
        now = time.time()
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT id FROM heuristics WHERE rule = ?", (rule,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE heuristics SET confidence = ?, sample_count = ?, "
                    "success_count = ?, failure_count = ?, last_validated = ? "
                    "WHERE id = ?",
                    (confidence, sample_count, success_count,
                     failure_count, now, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO heuristics (rule, confidence, sample_count, "
                    "success_count, failure_count, source, created_at, "
                    "last_validated, decay_rate) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rule, confidence, sample_count, success_count,
                     failure_count, source, now, now, self.default_decay_rate),
                )
            conn.commit()
        finally:
            conn.close()

    def extract(
        self,
        min_samples: int = 10,
        min_confidence: float = 0.6,
    ) -> List[Heuristic]:
        """
        Extract heuristics from recorded outcomes.

        If an extractor function is set, uses LLM reflection.
        Otherwise uses statistical pattern counting.

        Returns list of extracted/updated heuristics.
        """
        conn = self._conn()
        try:
            outcomes = conn.execute(
                "SELECT * FROM outcomes ORDER BY created_at DESC LIMIT 500"
            ).fetchall()
        finally:
            conn.close()

        if not outcomes:
            return []

        # Try LLM extraction
        if self.extractor:
            return self._extract_llm(outcomes, min_confidence)

        # Statistical extraction
        return self._extract_statistical(outcomes, min_samples, min_confidence)

    def _extract_statistical(
        self, outcomes, min_samples, min_confidence
    ) -> List[Heuristic]:
        """Extract heuristics by counting action success rates."""
        action_stats: Dict[str, Dict[str, int]] = {}

        for row in outcomes:
            action = row["action"]
            result = row["result"]
            if action not in action_stats:
                action_stats[action] = {"success": 0, "failure": 0, "total": 0}
            action_stats[action]["total"] += 1
            if result == "success":
                action_stats[action]["success"] += 1
            else:
                action_stats[action]["failure"] += 1

        heuristics = []
        for action, stats in action_stats.items():
            if stats["total"] < min_samples:
                continue
            rate = stats["success"] / stats["total"]
            if rate >= min_confidence:
                rule = f"{action}: {rate:.0%} success rate ({stats['total']} samples)"
                h = Heuristic(
                    rule=rule,
                    confidence=rate,
                    sample_count=stats["total"],
                    success_count=stats["success"],
                    failure_count=stats["failure"],
                )
                self.add_heuristic(
                    rule=rule,
                    confidence=rate,
                    sample_count=stats["total"],
                    success_count=stats["success"],
                    failure_count=stats["failure"],
                    source="statistical",
                )
                heuristics.append(h)

        return heuristics

    def _extract_llm(self, outcomes, min_confidence) -> List[Heuristic]:
        """Use LLM extractor to find patterns in outcomes."""
        outcome_dicts = []
        for row in outcomes:
            outcome_dicts.append({
                "action": row["action"],
                "result": row["result"],
                "context": json.loads(row["context"]) if row["context"] else {},
            })

        try:
            rules = self.extractor(outcome_dicts)

            # Validate that extractor returned an iterable of strings
            if not hasattr(rules, "__iter__"):
                logger.warning(
                    "LLM extractor returned non-iterable %s, expected list of strings",
                    type(rules).__name__,
                )
                return []

            heuristics = []
            for rule in rules:
                if isinstance(rule, str):
                    h = Heuristic(
                        rule=rule,
                        confidence=min_confidence,
                        source="llm_extracted",
                    )
                    self.add_heuristic(
                        rule=rule,
                        confidence=min_confidence,
                        source="llm_extracted",
                    )
                    heuristics.append(h)
            return heuristics
        except Exception as e:
            logger.warning(f"LLM heuristic extraction failed: {e}")
            return []

    def get_heuristics(
        self, min_confidence: float = 0.0, limit: int = 20
    ) -> List[Heuristic]:
        """Get stored heuristics sorted by confidence."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM heuristics WHERE confidence >= ? "
                "ORDER BY confidence DESC LIMIT ?",
                (min_confidence, limit),
            ).fetchall()
            return [self._row_to_heuristic(r) for r in rows]
        finally:
            conn.close()

    def build_prompt_context(
        self, max_rules: int = 5, min_confidence: float = 0.5
    ) -> str:
        """
        Build a prompt injection string from top heuristics.

        Returns compact rules suitable for system prompt inclusion.
        """
        heuristics = self.get_heuristics(
            min_confidence=min_confidence, limit=max_rules
        )
        if not heuristics:
            return ""

        lines = [f"Based on {self._total_outcomes()} past outcomes:"]
        for h in heuristics:
            lines.append(
                f"- {h.rule} (confidence: {h.confidence:.0%}, "
                f"{h.sample_count} samples)"
            )
        return "\n".join(lines)

    def apply_decay(self) -> int:
        """Apply confidence decay to old heuristics. Returns removed count."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE heuristics SET confidence = confidence * (1 - decay_rate) "
                "WHERE decay_rate > 0"
            )
            cursor = conn.execute("DELETE FROM heuristics WHERE confidence < 0.1")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def _total_outcomes(self) -> int:
        conn = self._conn()
        try:
            return conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._conn()
        try:
            outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            heuristics = conn.execute("SELECT COUNT(*) FROM heuristics").fetchone()[0]
            return {
                "total_outcomes": outcomes,
                "total_heuristics": heuristics,
            }
        finally:
            conn.close()

    @staticmethod
    def _row_to_heuristic(row: sqlite3.Row) -> Heuristic:
        return Heuristic(
            rule=row["rule"],
            confidence=row["confidence"],
            sample_count=row["sample_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            source=row["source"],
            created_at=row["created_at"],
            last_validated=row["last_validated"],
        )
