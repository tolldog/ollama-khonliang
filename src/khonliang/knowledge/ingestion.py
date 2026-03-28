"""
Knowledge ingestion pipeline.

Processes raw content (text, files, agent responses) into Tier 2 or Tier 3
knowledge entries. The pipeline:
1. Reads raw content
2. Optionally chunks large documents
3. Classifies scope and tags via LLM or rules
4. Summarizes if content exceeds a size threshold
5. Checks for duplicates against existing entries
6. Stores in the KnowledgeStore

The pipeline can be used standalone or driven by the Librarian agent.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from khonliang.knowledge.store import KnowledgeEntry, KnowledgeStore, Tier

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting one or more items."""

    added: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    entries: List[str] = field(default_factory=list)  # IDs of added entries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "errors": self.errors,
            "entry_ids": self.entries,
        }


class IngestionPipeline:
    """
    Processes raw content into knowledge entries.

    Example:
        pipeline = IngestionPipeline(store)

        # Ingest a text file
        result = pipeline.ingest_file("research_notes.txt", scope="toll")

        # Ingest raw text (e.g. from an agent response)
        result = pipeline.ingest_text(
            content="The Tolle family migrated from Maryland...",
            title="Tolle migration analysis",
            source="researcher_agent",
            tier=Tier.DERIVED,
            scope="toll",
            confidence=0.8,
        )

        # Ingest a directory of files
        result = pipeline.ingest_directory("research/", scope="toll")

        # With a custom summarizer (LLM-backed)
        pipeline = IngestionPipeline(
            store,
            summarizer=my_llm_summarize_fn,
            classifier=my_llm_classify_fn,
        )
    """

    # Chunk size for splitting large documents
    DEFAULT_CHUNK_SIZE = 4000
    # Summarize if content exceeds this many chars
    SUMMARIZE_THRESHOLD = 8000

    def __init__(
        self,
        store: KnowledgeStore,
        summarizer: Optional[Callable[[str, str], str]] = None,
        classifier: Optional[Callable[[str], Dict[str, Any]]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        summarize_threshold: int = SUMMARIZE_THRESHOLD,
    ):
        """
        Args:
            store: KnowledgeStore to write to
            summarizer: Optional function(content, title) -> summarized text.
                        If not provided, long content is truncated.
            classifier: Optional function(content) -> {"scope": ..., "tags": [...]}.
                        If not provided, scope must be specified manually.
            chunk_size: Max chars per chunk when splitting documents
            summarize_threshold: Summarize content longer than this
        """
        self.store = store
        self.summarizer = summarizer
        self.classifier = classifier
        self.chunk_size = chunk_size
        self.summarize_threshold = summarize_threshold

    def ingest_text(
        self,
        content: str,
        title: str,
        source: str = "user",
        tier: Tier = Tier.IMPORTED,
        scope: Optional[str] = None,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest a single text block.

        If content exceeds summarize_threshold and a summarizer is set,
        the content is summarized before storing.
        """
        result = IngestionResult()

        # Classify if no scope provided
        if scope is None and self.classifier:
            try:
                classification = self.classifier(content)
                scope = classification.get("scope", "global")
                tags = tags or classification.get("tags", [])
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                scope = "global"
        elif scope is None:
            scope = "global"

        # Summarize if too long
        if len(content) > self.summarize_threshold and self.summarizer:
            try:
                content = self.summarizer(content, title)
            except Exception as e:
                logger.warning(f"Summarization failed, truncating: {e}")
                content = content[: self.summarize_threshold] + "..."

        # Check for duplicate
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        eid = entry_id or f"{tier.name.lower()}:{content_hash}"

        existing = self.store.get(eid)
        if existing and existing.content == content:
            result.skipped += 1
            return result

        entry = KnowledgeEntry(
            id=eid,
            tier=tier,
            title=title,
            content=content,
            scope=scope,
            source=source,
            confidence=confidence,
            tags=tags or [],
            metadata=metadata or {},
        )

        try:
            self.store.add(entry)
            if existing:
                result.updated += 1
            else:
                result.added += 1
            result.entries.append(eid)
        except Exception as e:
            logger.error(f"Failed to store entry: {e}")
            result.errors += 1

        return result

    def ingest_file(
        self,
        path: str,
        scope: Optional[str] = None,
        tier: Tier = Tier.IMPORTED,
        confidence: float = 1.0,
    ) -> IngestionResult:
        """
        Ingest a text file, optionally chunking large files.

        Supported: .txt, .md, .csv, .json, .ged (plain text formats).
        """
        result = IngestionResult()
        filepath = Path(path)

        if not filepath.exists():
            logger.error(f"File not found: {path}")
            result.errors += 1
            return result

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            result.errors += 1
            return result

        # If small enough, ingest as one entry
        if len(content) <= self.chunk_size:
            return self.ingest_text(
                content=content,
                title=filepath.name,
                source=str(filepath),
                tier=tier,
                scope=scope,
                confidence=confidence,
            )

        # Chunk the file
        chunks = self._chunk_text(content, self.chunk_size)
        for i, chunk in enumerate(chunks):
            chunk_title = f"{filepath.name} (part {i + 1}/{len(chunks)})"
            chunk_result = self.ingest_text(
                content=chunk,
                title=chunk_title,
                source=str(filepath),
                tier=tier,
                scope=scope,
                confidence=confidence,
                entry_id=f"imported:{filepath.stem}:{i}",
            )
            result.added += chunk_result.added
            result.updated += chunk_result.updated
            result.skipped += chunk_result.skipped
            result.errors += chunk_result.errors
            result.entries.extend(chunk_result.entries)

        return result

    def ingest_directory(
        self,
        directory: str,
        scope: Optional[str] = None,
        tier: Tier = Tier.IMPORTED,
        patterns: Optional[List[str]] = None,
    ) -> IngestionResult:
        """
        Ingest all matching files in a directory.

        Args:
            directory: Path to directory
            scope: Scope for all entries
            tier: Tier for all entries
            patterns: Glob patterns to match (default: *.txt, *.md)
        """
        result = IngestionResult()
        dirpath = Path(directory)

        if not dirpath.is_dir():
            logger.error(f"Not a directory: {directory}")
            result.errors += 1
            return result

        file_patterns = patterns or ["*.txt", "*.md", "*.csv"]
        files = []
        for pattern in file_patterns:
            files.extend(dirpath.glob(pattern))

        for filepath in sorted(files):
            file_result = self.ingest_file(
                str(filepath), scope=scope, tier=tier
            )
            result.added += file_result.added
            result.updated += file_result.updated
            result.skipped += file_result.skipped
            result.errors += file_result.errors
            result.entries.extend(file_result.entries)

        logger.info(
            f"Ingested {len(files)} files from {directory}: "
            f"{result.added} added, {result.skipped} skipped, "
            f"{result.errors} errors"
        )
        return result

    def ingest_agent_response(
        self,
        content: str,
        title: str,
        agent_id: str,
        scope: Optional[str] = None,
        confidence: float = 0.7,
        query: str = "",
    ) -> IngestionResult:
        """
        Ingest an agent's response as Tier 3 derived knowledge.

        Tags the entry with provenance metadata so the librarian
        knows where it came from and can validate it later.
        """
        return self.ingest_text(
            content=content,
            title=title,
            source=agent_id,
            tier=Tier.DERIVED,
            scope=scope,
            confidence=confidence,
            metadata={
                "agent_id": agent_id,
                "query": query,
                "derived_at": time.time(),
            },
        )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > chunk_size and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text[:chunk_size]]
