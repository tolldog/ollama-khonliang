"""
Scoped RAG retriever — agent-aware document retrieval across knowledge scopes.

Extends the flat DocumentRetriever with a scope model that lets each agent
access only the knowledge relevant to its role:

- GLOBAL      — shared knowledge available to all agents
- DOMAIN      — knowledge scoped to a specific agent or agent type
- CONVERSATIONAL — past conversation turns stored per agent
- EXPERT      — curated expert knowledge per agent

Each scope maps to a different SQLite table. Agents declare which scopes
they need via RAGConfig, keeping their context focused and reducing noise.

Database schema (create alongside your rag_documents table)::

    -- Scoped knowledge store
    CREATE TABLE rag_scoped_documents (
        id          TEXT PRIMARY KEY,
        scope       TEXT NOT NULL,       -- 'domain', 'expert', etc.
        agent_id    TEXT,                -- NULL = applies to all agents
        collection  TEXT NOT NULL,       -- logical grouping within scope
        title       TEXT NOT NULL,
        content     TEXT NOT NULL,
        metadata    TEXT                 -- JSON blob
    );
    CREATE VIRTUAL TABLE rag_scoped_fts USING fts5(
        id, scope, agent_id, collection, title, content,
        content='rag_scoped_documents', content_rowid='rowid'
    );

    -- Conversational memory (one row per agent interaction)
    CREATE TABLE agent_conversations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id    TEXT NOT NULL,
        session_id  TEXT,
        message     TEXT,
        response    TEXT NOT NULL,
        metadata    TEXT,               -- JSON blob (e.g. intent, confidence)
        created_at  TEXT DEFAULT (datetime('now'))
    );

Usage::

    from khonliang.rag.scoped import ScopedRetriever, RAGConfig, RAGScope

    retriever = ScopedRetriever("knowledge.db")
    config = RAGConfig(scopes=[RAGScope.GLOBAL, RAGScope.DOMAIN], max_results=6)
    docs = retriever.retrieve("password reset flow", config, agent_id="analyst")
    context = retriever.build_context(docs)
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGScope(str, Enum):
    """Knowledge scope available to an agent."""

    GLOBAL = "global"
    """Shared knowledge — available to every agent."""

    DOMAIN = "domain"
    """Agent-specific domain knowledge (filtered by agent_id)."""

    CONVERSATIONAL = "conversational"
    """Past interaction history stored per agent."""

    EXPERT = "expert"
    """Curated expert knowledge scoped to a specific agent."""


@dataclass
class RAGConfig:
    """
    Declares which knowledge scopes and collections an agent may access.

    Args:
        scopes:      Ordered list of scopes to retrieve from (earlier = higher priority).
                     Defaults to GLOBAL only.
        collections: Optional allow-list of collection names within scoped tables.
                     Empty list means all collections.
        max_results: Maximum documents returned after merging and (optionally) reranking.
    """

    scopes: List[RAGScope] = field(default_factory=lambda: [RAGScope.GLOBAL])
    collections: List[str] = field(default_factory=list)
    max_results: int = 5


@dataclass
class ScopedDocument:
    """A document retrieved from the scoped RAG pipeline."""

    id: str
    scope: str
    agent_id: Optional[str]
    collection: str
    title: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "agent_id": self.agent_id,
            "collection": self.collection,
            "title": self.title,
            "content": self.content,
            "score": self.score,
        }


class ScopedRetriever:
    """
    RAGConfig-aware retriever that combines knowledge from multiple scopes.

    Retrieves documents from the appropriate SQLite tables based on an
    agent's RAGConfig declaration, merges results by BM25 score, and
    optionally reranks with a cross-encoder.

    Args:
        db_path:      Path to the SQLite database file
        use_reranker: If True, attempt cross-encoder reranking (requires
                      a ``CrossEncoderReranker`` importable from
                      ``khonliang.rag.reranker``). Falls back to BM25
                      order silently when unavailable.

    Example:
        >>> retriever = ScopedRetriever("knowledge.db")
        >>> config = RAGConfig(
        ...     scopes=[RAGScope.GLOBAL, RAGScope.DOMAIN],
        ...     max_results=6,
        ... )
        >>> docs = retriever.retrieve("how to reset 2FA", config, agent_id="analyst")
        >>> context = retriever.build_context(docs)
    """

    def __init__(self, db_path: str = "data/knowledge.db", use_reranker: bool = False):
        self.db_path = db_path
        self._use_reranker = use_reranker
        self._reranker = None

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        rag_config: Optional[RAGConfig] = None,
        agent_id: Optional[str] = None,
    ) -> List[ScopedDocument]:
        """
        Retrieve documents matching the query within the allowed scopes.

        Args:
            query:      Natural language search query
            rag_config: Scope and collection constraints (defaults to GLOBAL)
            agent_id:   Agent identifier for scope filtering

        Returns:
            List of ScopedDocument sorted by relevance (descending score)
        """
        if rag_config is None:
            rag_config = RAGConfig()

        # Over-retrieve when reranking so the cross-encoder has candidates
        k = rag_config.max_results * 3 if self._use_reranker else rag_config.max_results

        results: List[ScopedDocument] = []
        for scope in rag_config.scopes:
            results.extend(
                self._retrieve_scope(
                    query=query,
                    scope=scope,
                    agent_id=agent_id,
                    collections=rag_config.collections,
                    max_results=k,
                )
            )

        results.sort(key=lambda d: d.score, reverse=True)

        if self._use_reranker and len(results) > 1:
            results = self._rerank(query, results, rag_config.max_results)
        else:
            results = results[: rag_config.max_results]

        return results

    def build_context(
        self,
        documents: List[ScopedDocument],
        max_chars: int = 6000,
        include_scope: bool = True,
    ) -> str:
        """
        Format retrieved documents into a context string for LLM injection.

        Args:
            documents:     Documents from retrieve()
            max_chars:     Maximum total characters in the returned string
            include_scope: Prefix each document header with its scope label

        Returns:
            Formatted context string, empty string if no documents
        """
        if not documents:
            return ""

        parts: List[str] = []
        total = 0

        for doc in documents:
            header = (
                f"[{doc.scope.upper()}] {doc.title}" if include_scope else doc.title
            )
            entry = f"--- {header} ---\n{doc.content}\n"

            if total + len(entry) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(f"--- {header} ---\n{doc.content[:remaining]}...\n")
                break

            parts.append(entry)
            total += len(entry)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Scope dispatch
    # ------------------------------------------------------------------

    def _retrieve_scope(
        self,
        query: str,
        scope: RAGScope,
        agent_id: Optional[str],
        collections: List[str],
        max_results: int,
    ) -> List[ScopedDocument]:
        if scope == RAGScope.GLOBAL:
            return self._search_global(query, max_results)
        elif scope in (RAGScope.DOMAIN, RAGScope.EXPERT):
            return self._search_scoped(query, scope.value, agent_id, collections, max_results)
        elif scope == RAGScope.CONVERSATIONAL:
            return self._search_conversations(query, agent_id, max_results)
        return []

    # ------------------------------------------------------------------
    # Table-level search methods
    # ------------------------------------------------------------------

    def _search_global(self, query: str, limit: int) -> List[ScopedDocument]:
        """Search the global rag_documents FTS5 table."""
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []
        try:
            conn = self._conn()
            rows = conn.execute(
                """
                SELECT d.id, d.source, d.title, d.content, d.doc_type,
                       bm25(rag_documents_fts) AS score
                FROM rag_documents_fts f
                JOIN rag_documents d ON d.rowid = f.rowid
                WHERE rag_documents_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            conn.close()

            return [
                ScopedDocument(
                    id=row["id"],
                    scope="global",
                    agent_id=None,
                    collection=row["doc_type"],
                    title=row["title"],
                    content=row["content"],
                    score=abs(row["score"]),
                )
                for row in rows
            ]
        except sqlite3.OperationalError as e:
            logger.debug(f"Global RAG search failed (table may not exist): {e}")
            return []

    def _search_scoped(
        self,
        query: str,
        scope: str,
        agent_id: Optional[str],
        collections: List[str],
        limit: int,
    ) -> List[ScopedDocument]:
        """Search the rag_scoped_documents FTS5 table."""
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []
        try:
            conn = self._conn()

            conditions = ["rag_scoped_fts MATCH ?", "d.scope = ?"]
            params: list = [fts_query, scope]

            if agent_id:
                conditions.append("(d.agent_id = ? OR d.agent_id IS NULL)")
                params.append(agent_id)

            if collections:
                placeholders = ",".join(["?"] * len(collections))
                conditions.append(f"d.collection IN ({placeholders})")
                params.extend(collections)

            where = " AND ".join(conditions)
            params.append(limit)

            rows = conn.execute(
                f"""
                SELECT d.id, d.scope, d.agent_id, d.collection,
                       d.title, d.content, d.metadata,
                       bm25(rag_scoped_fts) AS score
                FROM rag_scoped_fts f
                JOIN rag_scoped_documents d ON d.rowid = f.rowid
                WHERE {where}
                ORDER BY score
                LIMIT ?
                """,
                params,
            ).fetchall()
            conn.close()

            return [
                ScopedDocument(
                    id=row["id"],
                    scope=row["scope"],
                    agent_id=row["agent_id"],
                    collection=row["collection"],
                    title=row["title"],
                    content=row["content"],
                    score=abs(row["score"]),
                )
                for row in rows
            ]
        except sqlite3.OperationalError as e:
            logger.debug(f"Scoped RAG search ({scope}) failed (table may not exist): {e}")
            return []

    def _search_conversations(
        self, query: str, agent_id: Optional[str], limit: int
    ) -> List[ScopedDocument]:
        """Search past conversation responses stored in agent_conversations."""
        if not agent_id:
            return []
        try:
            conn = self._conn()
            rows = conn.execute(
                """
                SELECT id, agent_id, session_id, message, response,
                       metadata, created_at
                FROM agent_conversations
                WHERE agent_id = ?
                  AND (response LIKE ? OR message LIKE ?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (agent_id, f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            conn.close()

            return [
                ScopedDocument(
                    id=str(row["id"]),
                    scope="conversational",
                    agent_id=row["agent_id"],
                    collection="conversations",
                    title=f"Past response ({row['created_at'][:10]})",
                    content=row["response"],
                    score=0.5,
                    metadata={
                        "session_id": row["session_id"],
                        "message": row["message"],
                    },
                )
                for row in rows
            ]
        except sqlite3.OperationalError as e:
            logger.debug(f"Conversational search failed (table may not exist): {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Convert a natural language query to an FTS5 OR query."""
        words = [w for w in query.split() if w.isalnum() or w in ("-", "_")]
        if not words:
            return ""
        return " OR ".join(f'"{w}"' for w in words)

    def _rerank(
        self,
        query: str,
        documents: List[ScopedDocument],
        top_k: int,
    ) -> List[ScopedDocument]:
        """Rerank with a cross-encoder if available, else return BM25 order."""
        try:
            if self._reranker is None:
                from khonliang.rag.reranker import CrossEncoderReranker  # type: ignore[import]
                self._reranker = CrossEncoderReranker.instance()
            return self._reranker.rerank(query, documents, top_k=top_k)
        except (ImportError, Exception) as e:
            logger.debug(f"Reranker unavailable, using BM25 order: {e}")
            return documents[:top_k]
