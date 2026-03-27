"""
Interaction logging and training feedback store.

Two complementary tables in the same SQLite database:

``agent_interactions``
    Every prompt/response pair routed through the helpdesk stack, with
    routing metadata (role, reason, generation time). Gives the operator
    full visibility into what the LLM is doing in production without any
    extra instrumentation in the role code.

``training_feedback``
    Rated subset of interactions. Each entry links back to an interaction
    and carries a 1-5 rating, free-text feedback, and an optional
    "expected" correction. High-rated entries (4-5) become positive RAG
    examples; low-rated ones (1-2) become negative examples that guide
    the LLM away from bad patterns.

Both tables are created automatically on first use.

Usage::

    store = FeedbackStore("knowledge.db")

    # Log every interaction (call from your role's handle())
    iid = store.log_interaction(
        message="My login is broken",
        role="triage",
        route_reason="keyword:broken",
        response="Have you tried resetting your password?",
        generation_ms=320,
        session_id="ch_abc123",
    )

    # Later, add a rating (from Mattermost feedback thread or CLI)
    store.add_feedback(
        interaction_id=iid,
        rating=4,
        feedback="Good but could also mention 2FA reset",
    )

    # Push good/bad examples into the RAG knowledge base
    store.index_into_rag()
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    message         TEXT    NOT NULL,
    role            TEXT,
    route_reason    TEXT,
    response        TEXT,
    generation_ms   INTEGER,
    metadata        TEXT,           -- JSON blob (extra role context)
    source          TEXT DEFAULT 'direct',  -- 'direct' | 'mattermost' | 'api'
    post_id         TEXT,           -- Mattermost post ID if source='mattermost'
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_feedback (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id      INTEGER REFERENCES agent_interactions(id),
    -- snapshot of prompt/response at time of feedback (denormalised for portability)
    prompt              TEXT    NOT NULL,
    response            TEXT,
    rating              INTEGER,    -- 1-5  (5 = excellent)
    feedback            TEXT,       -- free-text from reviewer
    expected            TEXT,       -- corrected / ideal response
    source              TEXT DEFAULT 'direct',
    channel_id          TEXT,
    user_id             TEXT,
    post_id             TEXT,       -- Mattermost feedback-comment post ID
    thread_id           TEXT,       -- Mattermost thread root ID
    indexed             INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (datetime('now')),
    indexed_at          TEXT
);
"""


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class AgentInteraction:
    """A logged agent interaction."""

    id: Optional[int]
    session_id: Optional[str]
    message: str
    role: Optional[str]
    route_reason: Optional[str]
    response: Optional[str]
    generation_ms: Optional[int]
    metadata: Dict[str, Any]
    source: str
    post_id: Optional[str]
    created_at: Optional[str]


@dataclass
class InteractionFeedback:
    """A training feedback entry linked to an interaction."""

    id: Optional[int]
    interaction_id: Optional[int]
    prompt: str
    response: Optional[str]
    rating: Optional[int]
    feedback: Optional[str]
    expected: Optional[str]
    source: str = "direct"
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    post_id: Optional[str] = None
    thread_id: Optional[str] = None
    indexed: bool = False
    created_at: Optional[str] = None
    indexed_at: Optional[str] = None


# ── Store ────────────────────────────────────────────────────────────────────

class FeedbackStore:
    """
    Log LLM interactions and collect training feedback.

    Args:
        db_path: Path to SQLite database file (created if missing)

    Example:
        >>> store = FeedbackStore("knowledge.db")
        >>> iid = store.log_interaction(
        ...     message="Password reset not working",
        ...     role="triage",
        ...     route_reason="keyword:error",
        ...     response="Let me help you reset your password...",
        ...     generation_ms=450,
        ... )
        >>> store.add_feedback(iid, rating=5, feedback="Concise and accurate")
        >>> store.index_into_rag()
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

    # ── Interaction logging ───────────────────────────────────────────────

    def log_interaction(
        self,
        message: str,
        role: Optional[str] = None,
        route_reason: Optional[str] = None,
        response: Optional[str] = None,
        generation_ms: Optional[int] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "direct",
        post_id: Optional[str] = None,
    ) -> int:
        """
        Log a single agent interaction.

        Call this from your role's ``handle()`` method (or a middleware wrapper)
        to capture every prompt/response pair with routing context.

        Args:
            message:       The user's original message
            role:          Which role handled it (e.g. 'triage', 'knowledge')
            route_reason:  Why it was routed here (e.g. 'regex:urgent', 'semantic:0.82')
            response:      The LLM's response text
            generation_ms: Time taken to generate the response
            session_id:    Conversation/channel ID for grouping
            metadata:      Arbitrary extra data (urgency, RAG docs used, etc.)
            source:        Where the message came from ('direct', 'mattermost', 'api')
            post_id:       Mattermost post ID if source='mattermost'

        Returns:
            The new interaction ID
        """
        conn = self._conn()
        try:
            cur = conn.execute(
                """
                INSERT INTO agent_interactions
                    (session_id, message, role, route_reason, response,
                     generation_ms, metadata, source, post_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message,
                    role,
                    route_reason,
                    response,
                    generation_ms,
                    json.dumps(metadata) if metadata else None,
                    source,
                    post_id,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_recent_interactions(
        self,
        limit: int = 50,
        role: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[AgentInteraction]:
        """
        Fetch recent interactions for operator review.

        Args:
            limit:  Max rows to return
            role:   Optional role filter (e.g. 'triage')
            source: Optional source filter ('direct', 'mattermost')
        """
        conn = self._conn()
        try:
            sql = "SELECT * FROM agent_interactions"
            params: list = []
            clauses = []
            if role:
                clauses.append("role = ?")
                params.append(role)
            if source:
                clauses.append("source = ?")
                params.append(source)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            return [self._row_to_interaction(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    # ── Feedback ──────────────────────────────────────────────────────────

    def add_feedback(
        self,
        interaction_id: Optional[int] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
        expected: Optional[str] = None,
        source: str = "direct",
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        post_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> int:
        """
        Record a rating and optional correction for an interaction.

        Either ``interaction_id`` (to link to a logged interaction) or
        ``prompt`` (to record standalone feedback) must be provided.

        Args:
            interaction_id: Link to an ``agent_interactions`` row
            prompt:         The original user message (required if no interaction_id)
            response:       The bot's response at the time of feedback
            rating:         1-5 quality rating (5 = excellent)
            feedback:       Free-text reviewer comment
            expected:       What the ideal response should have been
            source:         'direct', 'mattermost', 'api'
            channel_id:     Channel context
            user_id:        Who rated it
            post_id:        Mattermost post ID of the feedback comment (for dedup)
            thread_id:      Mattermost thread root ID

        Returns:
            The new feedback entry ID
        """
        # If interaction_id given and no prompt, fetch it from the interaction
        if interaction_id and not prompt:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT message, response FROM agent_interactions WHERE id = ?",
                    (interaction_id,),
                ).fetchone()
                if row:
                    prompt = prompt or row["message"]
                    response = response or row["response"]
            finally:
                conn.close()

        if not prompt:
            raise ValueError("Either interaction_id or prompt must be provided")

        conn = self._conn()
        try:
            cur = conn.execute(
                """
                INSERT INTO training_feedback
                    (interaction_id, prompt, response, rating, feedback, expected,
                     source, channel_id, user_id, post_id, thread_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    prompt,
                    response,
                    rating,
                    feedback,
                    expected,
                    source,
                    channel_id,
                    user_id,
                    post_id,
                    thread_id,
                ),
            )
            conn.commit()
            fid = cur.lastrowid
            logger.info(f"Feedback {fid} added: rating={rating}")
            return fid
        finally:
            conn.close()

    def get_feedback(self, limit: int = 100, min_rating: Optional[int] = None) -> List[InteractionFeedback]:
        """Fetch feedback entries, optionally filtered by minimum rating."""
        conn = self._conn()
        try:
            if min_rating is not None:
                rows = conn.execute(
                    "SELECT * FROM training_feedback WHERE rating >= ? ORDER BY created_at DESC LIMIT ?",
                    (min_rating, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM training_feedback ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_feedback(r) for r in rows]
        finally:
            conn.close()

    # ── RAG indexing ──────────────────────────────────────────────────────

    def index_into_rag(self, min_rating: Optional[int] = None) -> Dict[str, int]:
        """
        Index unindexed rated feedback into the RAG ``rag_documents`` table.

        High-rated feedback (4-5) becomes a positive example with guidance
        to emulate the response style. Low-rated feedback (1-2) becomes a
        negative example with guidance to avoid the pattern.

        Args:
            min_rating: Only index entries at or above this rating

        Returns:
            ``{"indexed": N, "skipped": N, "errors": N}``
        """
        stats = {"indexed": 0, "skipped": 0, "errors": 0}
        conn = self._conn()
        try:
            sql = "SELECT * FROM training_feedback WHERE indexed = 0 AND rating IS NOT NULL"
            if min_rating is not None:
                sql += f" AND rating >= {int(min_rating)}"
            rows = conn.execute(sql).fetchall()

            for row in rows:
                try:
                    fb = self._row_to_feedback(row)
                    doc_id, title, content = self._feedback_to_rag_doc(fb)

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO rag_documents
                            (id, source, path, title, content, doc_type)
                        VALUES (?, 'training_feedback', ?, ?, ?, 'feedback')
                        """,
                        (doc_id, f"feedback/{fb.id}", title, content),
                    )
                    conn.execute(
                        "UPDATE training_feedback SET indexed = 1, indexed_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), fb.id),
                    )
                    conn.commit()
                    stats["indexed"] += 1
                except Exception as e:
                    logger.error(f"Failed to index feedback {row['id']}: {e}")
                    stats["errors"] += 1

        finally:
            conn.close()

        logger.info(
            f"RAG index: {stats['indexed']} indexed, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the operator dashboard."""
        conn = self._conn()
        try:
            interactions = conn.execute(
                "SELECT COUNT(*) FROM agent_interactions"
            ).fetchone()[0]

            by_role = {}
            for r in conn.execute(
                "SELECT role, COUNT(*) FROM agent_interactions GROUP BY role"
            ).fetchall():
                by_role[r[0] or "unknown"] = r[1]

            feedback_total = conn.execute(
                "SELECT COUNT(*) FROM training_feedback"
            ).fetchone()[0]

            by_rating: Dict[int, int] = {}
            for r in conn.execute(
                "SELECT rating, COUNT(*) FROM training_feedback WHERE rating IS NOT NULL GROUP BY rating"
            ).fetchall():
                by_rating[r[0]] = r[1]

            indexed = conn.execute(
                "SELECT COUNT(*) FROM training_feedback WHERE indexed = 1"
            ).fetchone()[0]

            return {
                "interactions": interactions,
                "by_role": by_role,
                "feedback": {
                    "total": feedback_total,
                    "by_rating": by_rating,
                    "indexed": indexed,
                    "unindexed": feedback_total - indexed,
                },
            }
        finally:
            conn.close()

    # ── Mattermost sync ───────────────────────────────────────────────────

    def sync_from_mattermost(
        self,
        mattermost_url: str,
        token: str,
        channel_id: str,
        bot_user_id: str,
    ) -> Dict[str, int]:
        """
        Pull ``Training Feedback`` thread replies from a Mattermost channel
        and store them as feedback entries.

        Feedback is posted by reviewers as thread replies in the format::

            **Training Feedback:**
            ```json
            {"rating": 4, "feedback": "Good but verbose", "expected": "..."}
            ```

        Args:
            mattermost_url: Mattermost server URL
            token:          Bot or reviewer access token
            channel_id:     Channel to read from
            bot_user_id:    User ID of the helpdesk bot (to identify its responses)

        Returns:
            ``{"added": N, "skipped": N, "errors": N}``
        """
        import requests as _requests

        stats = {"added": 0, "skipped": 0, "errors": 0}
        headers = {"Authorization": f"Bearer {token}"}

        resp = _requests.get(
            f"{mattermost_url}/api/v4/channels/{channel_id}/posts",
            headers=headers,
            params={"per_page": 200},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(f"Mattermost fetch failed: {resp.status_code}")
            return stats

        posts = resp.json().get("posts", {})

        for post in posts.values():
            if not post.get("root_id"):
                continue
            if "Training Feedback" not in post.get("message", ""):
                continue

            try:
                message = post["message"]
                js = message.find("```json\n") + 8
                je = message.find("\n```", js)
                if js <= 7 or je <= js:
                    continue

                fb_data = json.loads(message[js:je])
                root_id = post["root_id"]

                # Reconstruct prompt and response from the thread
                thread_posts = sorted(
                    [p for p in posts.values()
                     if p.get("root_id") == root_id or p.get("id") == root_id],
                    key=lambda p: p.get("create_at", 0),
                )
                prompt = ""
                response = ""
                for tp in thread_posts:
                    if tp.get("user_id") == bot_user_id:
                        response = tp.get("message", "")
                    elif "Training Feedback" not in tp.get("message", ""):
                        prompt = tp.get("message", "")

                if not prompt:
                    root_post = posts.get(root_id, {})
                    prompt = root_post.get("message", "")

                self.add_feedback(
                    prompt=prompt,
                    response=response,
                    rating=fb_data.get("rating"),
                    feedback=fb_data.get("feedback"),
                    expected=fb_data.get("expected"),
                    source="mattermost",
                    channel_id=channel_id,
                    post_id=post["id"],
                    thread_id=root_id,
                )
                stats["added"] += 1

            except json.JSONDecodeError:
                stats["errors"] += 1
            except sqlite3.IntegrityError:
                stats["skipped"] += 1  # duplicate post_id
            except Exception as e:
                logger.error(f"Error processing feedback post: {e}")
                stats["errors"] += 1

        logger.info(
            f"Mattermost sync: added={stats['added']}, "
            f"skipped={stats['skipped']}, errors={stats['errors']}"
        )
        return stats

    # ── Helpers ───────────────────────────────────────────────────────────

    def _feedback_to_rag_doc(
        self, fb: "InteractionFeedback"
    ) -> tuple:
        """Convert feedback to (doc_id, title, content) for rag_documents."""
        rating = fb.rating or 0
        if rating >= 4:
            quality, guidance = "GOOD", (
                "This response style is preferred. Emulate the conciseness and "
                "approach for similar queries."
            )
        elif rating <= 2:
            quality, guidance = "NEEDS_IMPROVEMENT", (
                f"Avoid this response style. Issues: {fb.feedback or 'see feedback'}. "
                f"Preferred: {fb.expected or 'be more concise and accurate'}."
            )
        else:
            quality, guidance = "ACCEPTABLE", "Response was acceptable but could be improved."

        parts = [
            f"Training Example ({quality})",
            f"User: {fb.prompt}",
        ]
        if fb.response:
            parts.append(f"Bot: {fb.response[:500]}{'...' if len(fb.response or '') > 500 else ''}")
        parts.append(f"Rating: {rating}/5")
        if fb.feedback:
            parts.append(f"Reviewer note: {fb.feedback}")
        if fb.expected:
            parts.append(f"Ideal response: {fb.expected}")
        parts.append(f"Guidance: {guidance}")

        return (
            f"feedback:{fb.id}",
            f"Feedback: {fb.prompt[:60]}",
            "\n\n".join(parts),
        )

    @staticmethod
    def _row_to_interaction(row: sqlite3.Row) -> AgentInteraction:
        return AgentInteraction(
            id=row["id"],
            session_id=row["session_id"],
            message=row["message"],
            role=row["role"],
            route_reason=row["route_reason"],
            response=row["response"],
            generation_ms=row["generation_ms"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            source=row["source"],
            post_id=row["post_id"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_feedback(row: sqlite3.Row) -> InteractionFeedback:
        return InteractionFeedback(
            id=row["id"],
            interaction_id=row["interaction_id"],
            prompt=row["prompt"],
            response=row["response"],
            rating=row["rating"],
            feedback=row["feedback"],
            expected=row["expected"],
            source=row["source"],
            channel_id=row["channel_id"],
            user_id=row["user_id"],
            post_id=row["post_id"],
            thread_id=row["thread_id"],
            indexed=bool(row["indexed"]),
            created_at=row["created_at"],
            indexed_at=row["indexed_at"],
        )
