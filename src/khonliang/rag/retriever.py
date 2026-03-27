"""
Document Retriever for RAG (Retrieval-Augmented Generation).

Retrieves relevant documents for context injection into LLM prompts
using SQLite FTS5 with BM25 ranking.

Schema expected in the SQLite database:

    CREATE VIRTUAL TABLE rag_documents_fts USING fts5(
        id, source, path, title, content, doc_type,
        content='rag_documents', content_rowid='rowid'
    );

    CREATE TABLE rag_documents (
        id TEXT PRIMARY KEY,
        source TEXT,
        path TEXT,
        title TEXT,
        content TEXT,
        doc_type TEXT,
        metadata TEXT  -- JSON
    );

Usage:

    retriever = DocumentRetriever("myapp.db")
    docs = retriever.search("how to reset password")
    context = retriever.build_context(docs, max_chars=4000)

    # Or one-shot:
    context = retriever.get_relevant_context("how to reset password")
"""

import logging
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A document retrieved for RAG context."""

    id: str
    source: str
    path: str
    title: str
    content: str
    doc_type: str
    score: float
    snippet: str = ""


class DocumentRetriever:
    """
    Retrieve relevant documents for RAG context injection.

    Uses SQLite FTS5 full-text search with BM25 ranking.
    Supports filtering by source and doc_type.

    Args:
        db_path: Path to SQLite database file

    Example:
        >>> retriever = DocumentRetriever("knowledge.db")
        >>> docs = retriever.search("password reset")
        >>> context = retriever.build_context(docs, max_chars=4000)
    """

    def __init__(self, db_path: str = "data/knowledge.db"):
        self.db_path = db_path

    def search(
        self,
        query: str,
        source: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 10,
        _fts_query: Optional[str] = None,
    ) -> List[RetrievedDocument]:
        """
        Search for relevant documents using FTS5 BM25 ranking.

        Args:
            query:     Natural language search query
            source:    Filter by source field (e.g. 'docs', 'api', 'code')
            doc_type:  Filter by doc_type field (e.g. 'markdown', 'endpoint')
            limit:     Maximum results to return
            _fts_query: Pre-built FTS5 query string (bypasses word splitting)

        Returns:
            List of RetrievedDocument sorted by relevance score (descending)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if _fts_query:
                clean_query = _fts_query
            else:
                words = [
                    word
                    for word in query.split()
                    if word.isalnum() or word in ["_", "-"]
                ]
                if not words:
                    return []
                # OR between double-quoted terms — prevents FTS5 column:term misparse
                clean_query = " OR ".join(f'"{w}"' for w in words)

            sql = """
                SELECT
                    d.id, d.source, d.path, d.title, d.content, d.doc_type,
                    bm25(rag_documents_fts) as score,
                    snippet(rag_documents_fts, 2, '<mark>', '</mark>', '...', 64) as snippet
                FROM rag_documents_fts f
                JOIN rag_documents d ON d.rowid = f.rowid
                WHERE rag_documents_fts MATCH ?
            """
            params: list = [clean_query]

            if source:
                sql += " AND d.source = ?"
                params.append(source)

            if doc_type:
                sql += " AND d.doc_type = ?"
                params.append(doc_type)

            sql += " ORDER BY score LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            results = []

            for row in cursor.fetchall():
                results.append(
                    RetrievedDocument(
                        id=row["id"],
                        source=row["source"],
                        path=row["path"],
                        title=row["title"],
                        content=row["content"],
                        doc_type=row["doc_type"],
                        score=abs(row["score"]),  # BM25 returns negative scores
                        snippet=row["snippet"],
                    )
                )

            return results

        except sqlite3.OperationalError as e:
            logger.warning(f"FTS search error (table may not exist): {e}")
            return []
        finally:
            conn.close()

    def search_multi(
        self,
        queries: List[str],
        limit_per_query: int = 5,
        dedupe: bool = True,
    ) -> List[RetrievedDocument]:
        """
        Search with multiple queries and merge results.

        Useful for decomposing complex questions into sub-queries.

        Args:
            queries:         List of search queries
            limit_per_query: Max results per query
            dedupe:          Remove duplicate documents by id

        Returns:
            Combined results sorted by score (descending)
        """
        all_results = []
        seen_ids: set = set()

        for query in queries:
            for doc in self.search(query, limit=limit_per_query):
                if dedupe and doc.id in seen_ids:
                    continue
                seen_ids.add(doc.id)
                all_results.append(doc)

        all_results.sort(key=lambda d: d.score, reverse=True)
        return all_results

    def search_by_keywords(
        self,
        question: str,
        limit: int = 10,
    ) -> List[RetrievedDocument]:
        """
        Extract keywords from a natural language question and search.

        Strips common stop words before building the FTS5 query.

        Args:
            question: Natural language question
            limit:    Maximum results

        Returns:
            Relevant documents sorted by score
        """
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "what", "where", "when", "why", "how", "which", "who", "whom",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "as", "until", "while",
            "please", "tell", "show", "give", "get", "let", "make",
        }

        words = question.lower().split()
        keywords = [
            w for w in words
            if w.isalnum() and w not in stop_words and len(w) > 2
        ]

        if not keywords:
            return self.search(question, limit=limit)

        fts_query = " OR ".join(f'"{w}"' for w in keywords)
        return self.search(question, limit=limit, _fts_query=fts_query)

    def build_context(
        self,
        documents: List[RetrievedDocument],
        max_chars: int = 8000,
        include_source: bool = True,
    ) -> str:
        """
        Build a context string from retrieved documents for LLM injection.

        Args:
            documents:      Documents to include
            max_chars:      Maximum characters in the returned context
            include_source: Prepend source/path attribution per document

        Returns:
            Formatted context string ready for prompt injection
        """
        if not documents:
            return ""

        context_parts = []
        total_chars = 0

        for doc in documents:
            header = (
                f"[Source: {doc.source}/{doc.path}]\n{doc.title}\n"
                if include_source
                else f"{doc.title}\n"
            )

            remaining = max_chars - total_chars - len(header) - 100
            if remaining <= 0:
                break

            content = doc.content[:remaining] if len(doc.content) > remaining else doc.content
            part = f"{header}{'-' * 40}\n{content}\n"
            context_parts.append(part)
            total_chars += len(part)

            if total_chars >= max_chars:
                break

        return "\n".join(context_parts)

    def get_relevant_context(
        self,
        question: str,
        max_chars: int = 8000,
        sources: Optional[List[str]] = None,
    ) -> str:
        """
        One-shot: search and return formatted context for a question.

        Args:
            question:  User's question
            max_chars: Maximum context size in characters
            sources:   Optional list of source filters to include
                       (e.g. ['docs', 'api']). None = all sources.

        Returns:
            Formatted context string, empty string if nothing found
        """
        if sources:
            docs = []
            seen: set = set()
            for src in sources:
                for doc in self.search_by_keywords(question, limit=5):
                    if doc.id not in seen and doc.source == src:
                        seen.add(doc.id)
                        docs.append(doc)
                # Also direct source-filtered search
                for doc in self.search(question, source=src, limit=3):
                    if doc.id not in seen:
                        seen.add(doc.id)
                        docs.append(doc)
        else:
            docs = self.search_by_keywords(question, limit=8)

        docs.sort(key=lambda d: d.score, reverse=True)
        return self.build_context(docs, max_chars=max_chars)
