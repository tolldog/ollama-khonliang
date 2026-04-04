"""
Outcome tracking for consensus decisions.

Records the real-world outcome of each consensus decision, enabling:
  - Weight learning: which agents predict good outcomes?
  - Heuristic extraction: which patterns precede success?
  - Cooperation analysis: which agent agreements correlate with success?

Usage:
    tracker = OutcomeTracker("data/outcomes.db")

    # After consensus
    consensus_id = tracker.record_consensus(result)

    # Later, when outcome is known
    tracker.record_outcome(consensus_id, outcome=0.8, metadata={"task": "triage-42"})

    # Query
    history = tracker.get_history(agent_id="analyst", limit=50)
    stats = tracker.get_agent_stats("analyst")
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from khonliang.consensus.models import ConsensusResult

logger = logging.getLogger(__name__)

_OUTCOME_SCHEMA = """
CREATE TABLE IF NOT EXISTS consensus_outcomes (
    consensus_id    TEXT PRIMARY KEY,
    action          TEXT NOT NULL,
    confidence      REAL NOT NULL,
    votes_json      TEXT NOT NULL,
    scores_json     TEXT DEFAULT '{}',
    reason          TEXT DEFAULT '',
    subject         TEXT DEFAULT '',
    outcome         REAL,
    outcome_at      REAL,
    outcome_metadata TEXT DEFAULT '{}',
    created_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcomes_action ON consensus_outcomes(action);
CREATE INDEX IF NOT EXISTS idx_outcomes_subject ON consensus_outcomes(subject);
CREATE INDEX IF NOT EXISTS idx_outcomes_created ON consensus_outcomes(created_at);
"""


@dataclass
class OutcomeRecord:
    """A consensus decision paired with its real-world outcome."""

    consensus_id: str
    action: str
    confidence: float
    votes: List[Dict[str, Any]]
    scores: Dict[str, float]
    reason: str
    subject: str
    outcome: Optional[float]
    outcome_at: Optional[float]
    outcome_metadata: Dict[str, Any]
    created_at: float

    @property
    def has_outcome(self) -> bool:
        return self.outcome is not None

    def agent_voted(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent's vote from this consensus."""
        for v in self.votes:
            if v.get("agent_id") == agent_id:
                return v
        return None

    def agents_who_voted(self, action: str) -> List[str]:
        """Return agent_ids that voted for a specific action."""
        return [v["agent_id"] for v in self.votes if v.get("action") == action]


class OutcomeTracker:
    """
    SQLite-backed tracker for consensus outcomes.

    Records consensus decisions when they happen, then links them
    to real-world outcomes when those become available.
    """

    def __init__(self, db_path: str = "data/outcomes.db"):
        self.db_path = db_path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(_OUTCOME_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def record_consensus(
        self,
        result: ConsensusResult,
        subject: str = "",
        consensus_id: Optional[str] = None,
    ) -> str:
        """Record a consensus decision. Returns the consensus_id for later outcome linking.

        Args:
            result: The ConsensusResult from calculate_consensus()
            subject: What the decision was about (e.g., symbol, task)
            consensus_id: Optional custom ID. Auto-generated if omitted.
        """
        cid = consensus_id or str(uuid.uuid4())[:12]
        now = time.time()

        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO consensus_outcomes
                   (consensus_id, action, confidence, votes_json, scores_json,
                    reason, subject, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(consensus_id) DO UPDATE SET
                       action=excluded.action,
                       confidence=excluded.confidence,
                       votes_json=excluded.votes_json,
                       scores_json=excluded.scores_json,
                       reason=excluded.reason,
                       subject=excluded.subject""",
                (
                    cid,
                    result.action,
                    result.confidence,
                    json.dumps([v.to_dict() for v in result.votes]),
                    json.dumps(result.scores),
                    result.reason or "",
                    subject,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.debug(
            "Recorded consensus %s: %s (%.0f%%)",
            cid, result.action, result.confidence * 100,
        )
        return cid

    def record_outcome(
        self,
        consensus_id: str,
        outcome: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Link a real-world outcome to a previously recorded consensus.

        Args:
            consensus_id: The ID returned by record_consensus()
            outcome: Numeric outcome score (e.g., accuracy, success rate 0-1)
            metadata: Additional context (e.g., symbol, hold_days, exit_reason)

        Returns:
            True if the consensus was found and updated.
        """
        now = time.time()
        conn = self._conn()
        try:
            # Check existence first (rowcount can be 0 on no-op UPDATE)
            exists = conn.execute(
                "SELECT 1 FROM consensus_outcomes WHERE consensus_id = ?",
                (consensus_id,),
            ).fetchone()
            if not exists:
                logger.warning(
                    "Consensus %s not found for outcome recording",
                    consensus_id,
                )
                return False

            conn.execute(
                """UPDATE consensus_outcomes
                   SET outcome = ?, outcome_at = ?, outcome_metadata = ?
                   WHERE consensus_id = ?""",
                (outcome, now, json.dumps(metadata or {}), consensus_id),
            )
            conn.commit()
            logger.debug("Recorded outcome for %s: %.4f", consensus_id, outcome)
            return True
        finally:
            conn.close()

    def get_history(
        self,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        subject: Optional[str] = None,
        with_outcome_only: bool = False,
        limit: int = 100,
    ) -> List[OutcomeRecord]:
        """Query outcome history with optional filters.

        Args:
            agent_id: Filter to consensuses where this agent voted
            action: Filter by consensus action
            subject: Filter by subject
            with_outcome_only: Only return records that have an outcome
            limit: Max records to return
        """
        query = "SELECT * FROM consensus_outcomes WHERE 1=1"
        params: list = []

        if action:
            query += " AND action = ?"
            params.append(action)
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if with_outcome_only:
            query += " AND outcome IS NOT NULL"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = self._conn()
        try:
            rows = conn.execute(query, params).fetchall()
            records = [self._row_to_record(r) for r in rows]

            # Post-filter by agent_id (requires checking votes JSON)
            if agent_id:
                records = [r for r in records if r.agent_voted(agent_id)]

            return records
        finally:
            conn.close()

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Compute outcome statistics for a specific agent.

        Returns accuracy, mean outcome when agent's vote matched consensus,
        and mean outcome when it didn't.
        """
        records = self.get_history(agent_id=agent_id, with_outcome_only=True, limit=500)

        if not records:
            return {"agent_id": agent_id, "sample_count": 0}

        aligned_outcomes = []
        opposed_outcomes = []

        for r in records:
            vote = r.agent_voted(agent_id)
            if not vote:
                continue
            if vote["action"] == r.action:
                aligned_outcomes.append(r.outcome)
            else:
                opposed_outcomes.append(r.outcome)

        def _mean(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "agent_id": agent_id,
            "sample_count": len(records),
            "aligned_count": len(aligned_outcomes),
            "opposed_count": len(opposed_outcomes),
            "mean_outcome_aligned": _mean(aligned_outcomes),
            "mean_outcome_opposed": _mean(opposed_outcomes),
            "outcome_delta": _mean(aligned_outcomes) - _mean(opposed_outcomes),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Overall outcome tracker statistics."""
        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM consensus_outcomes").fetchone()[0]
            with_outcome = conn.execute(
                "SELECT COUNT(*) FROM consensus_outcomes WHERE outcome IS NOT NULL"
            ).fetchone()[0]
            return {
                "total_consensuses": total,
                "with_outcomes": with_outcome,
                "pending_outcomes": total - with_outcome,
            }
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> OutcomeRecord:
        return OutcomeRecord(
            consensus_id=row["consensus_id"],
            action=row["action"],
            confidence=row["confidence"],
            votes=json.loads(row["votes_json"]),
            scores=json.loads(row["scores_json"]),
            reason=row["reason"],
            subject=row["subject"],
            outcome=row["outcome"],
            outcome_at=row["outcome_at"],
            outcome_metadata=json.loads(row["outcome_metadata"]) if row["outcome_metadata"] else {},
            created_at=row["created_at"],
        )
