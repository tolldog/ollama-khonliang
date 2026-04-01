"""
ReportManager — SQLite-backed persistence for agent-generated reports.

Stores reports with metadata, view tracking, and optional TTL-based expiration.
Reports contain markdown content and are rendered to HTML on demand by the
ReportServer.

Usage:
    manager = ReportManager("reports.db")
    report = manager.create(
        title="Network Analysis",
        content_markdown="## Findings\\n...",
        report_type="analysis",
        created_by="monitor_agent",
    )
    print(report.id)  # UUID

    # Retrieve
    report = manager.get(report.id)

    # List recent
    reports = manager.list_reports(report_type="analysis", limit=10)

    # Cleanup expired
    manager.purge_expired()
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default TTL overrides by report type (seconds). None = never expires.
DEFAULT_TTL: Dict[str, Optional[int]] = {
    "quick_analysis": 30 * 86400,   # 30 days
    "session_summary": 7 * 86400,   # 7 days
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    report_type     TEXT NOT NULL,
    title           TEXT NOT NULL,
    content_markdown TEXT NOT NULL,
    created_at      REAL NOT NULL,
    created_by      TEXT,
    metadata        TEXT,         -- JSON blob for domain-specific data
    view_count      INTEGER NOT NULL DEFAULT 0,
    last_viewed_at  REAL,
    expires_at      REAL,        -- NULL = never expires
    chat_context    TEXT          -- JSON: integration-specific linkback data
);

CREATE INDEX IF NOT EXISTS idx_reports_type ON reports (report_type);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_expires ON reports (expires_at)
    WHERE expires_at IS NOT NULL;
"""


@dataclass
class Report:
    """A persisted agent report."""

    id: str
    report_type: str
    title: str
    content_markdown: str
    created_at: float
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    view_count: int = 0
    last_viewed_at: Optional[float] = None
    expires_at: Optional[float] = None
    chat_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Report":
        return cls(
            id=row["id"],
            report_type=row["report_type"],
            title=row["title"],
            content_markdown=row["content_markdown"],
            created_at=row["created_at"],
            created_by=row["created_by"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            view_count=row["view_count"],
            last_viewed_at=row["last_viewed_at"],
            expires_at=row["expires_at"],
            chat_context=json.loads(row["chat_context"]) if row["chat_context"] else {},
        )


class ReportManager:
    """
    SQLite-backed report persistence.

    Args:
        db_path: Path to SQLite database file. Use ":memory:" for testing.
        default_ttl_overrides: Dict mapping report_type -> TTL in seconds.
            Merged with built-in defaults. None values mean no expiration.
    """

    def __init__(
        self,
        db_path: str = "reports.db",
        default_ttl_overrides: Optional[Dict[str, Optional[int]]] = None,
    ):
        self.db_path = db_path
        self._ttl_map = {**DEFAULT_TTL}
        if default_ttl_overrides:
            self._ttl_map.update(default_ttl_overrides)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def create(
        self,
        title: str,
        content_markdown: str,
        report_type: str = "general",
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chat_context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = ...,  # type: ignore[assignment]
    ) -> Report:
        """
        Create and persist a new report.

        Args:
            title: Human-readable report title.
            content_markdown: Report body in markdown.
            report_type: Classification string (e.g. "analysis", "research").
            created_by: Agent or user ID that generated this report.
            metadata: Arbitrary key-value data (stored as JSON).
            chat_context: Integration-specific linkback data (e.g. Mattermost
                post_id, channel_id, thread_id, permalink).
            ttl: Time-to-live in seconds. Pass None for no expiration.
                Omit (default sentinel) to use the type-based default.

        Returns:
            The created Report with its generated UUID.
        """
        report_id = uuid.uuid4().hex
        now = time.time()

        # Resolve TTL: explicit > type default > no expiration
        if ttl is ...:
            ttl = self._ttl_map.get(report_type)

        expires_at = (now + ttl) if ttl is not None else None

        report = Report(
            id=report_id,
            report_type=report_type,
            title=title,
            content_markdown=content_markdown,
            created_at=now,
            created_by=created_by,
            metadata=metadata or {},
            expires_at=expires_at,
            chat_context=chat_context or {},
        )

        self._conn.execute(
            """INSERT INTO reports
               (id, report_type, title, content_markdown, created_at,
                created_by, metadata, view_count, last_viewed_at,
                expires_at, chat_context)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)""",
            (
                report.id,
                report.report_type,
                report.title,
                report.content_markdown,
                report.created_at,
                report.created_by,
                json.dumps(report.metadata) if report.metadata else None,
                report.expires_at,
                json.dumps(report.chat_context) if report.chat_context else None,
            ),
        )
        self._conn.commit()

        logger.info(f"Report created: {report_id} ({report_type}) '{title}'")
        return report

    def get(self, report_id: str, track_view: bool = True) -> Optional[Report]:
        """
        Retrieve a report by ID.

        Args:
            report_id: The report UUID.
            track_view: If True, increment view_count and update last_viewed_at.

        Returns:
            The Report, or None if not found or expired.
        """
        row = self._conn.execute(
            "SELECT * FROM reports WHERE id = ?", (report_id,)
        ).fetchone()

        if row is None:
            return None

        report = Report.from_row(row)

        if report.is_expired:
            self._conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
            self._conn.commit()
            return None

        if track_view:
            now = time.time()
            self._conn.execute(
                "UPDATE reports SET view_count = view_count + 1, last_viewed_at = ? WHERE id = ?",
                (now, report_id),
            )
            self._conn.commit()
            report.view_count += 1
            report.last_viewed_at = now

        return report

    def list_reports(
        self,
        report_type: Optional[str] = None,
        created_by: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Report]:
        """
        List reports, newest first.

        Args:
            report_type: Filter by type. None = all types.
            created_by: Filter by creator. None = all creators.
            limit: Max results to return.
            offset: Number of results to skip.
        """
        conditions = ["(expires_at IS NULL OR expires_at > ?)"]
        params: list = [time.time()]

        if report_type:
            conditions.append("report_type = ?")
            params.append(report_type)
        if created_by:
            conditions.append("created_by = ?")
            params.append(created_by)

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"SELECT * FROM reports WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()

        return [Report.from_row(row) for row in rows]

    def delete(self, report_id: str) -> bool:
        """Delete a report. Returns True if it existed."""
        cursor = self._conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def purge_expired(self) -> int:
        """Delete all expired reports. Returns count deleted."""
        cursor = self._conn.execute(
            "DELETE FROM reports WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )
        self._conn.commit()
        count = cursor.rowcount
        if count:
            logger.info(f"Purged {count} expired reports")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics for the report store."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM reports WHERE expires_at IS NULL OR expires_at > ?",
            (time.time(),),
        ).fetchone()[0]

        by_type = {}
        rows = self._conn.execute(
            """SELECT report_type, COUNT(*) as cnt FROM reports
               WHERE expires_at IS NULL OR expires_at > ?
               GROUP BY report_type""",
            (time.time(),),
        ).fetchall()
        for row in rows:
            by_type[row["report_type"]] = row["cnt"]

        total_views = self._conn.execute(
            "SELECT COALESCE(SUM(view_count), 0) FROM reports"
        ).fetchone()[0]

        return {
            "total_reports": total,
            "by_type": by_type,
            "total_views": total_views,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
