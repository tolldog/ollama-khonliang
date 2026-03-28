"""
Librarian agent — manages the knowledge store using only Tier 1 axioms.

The librarian:
- Only sees Tier 1 (axioms) — unbiased by content it manages
- Ingests raw content into Tier 2 via the IngestionPipeline
- Indexes agent responses into Tier 3
- Promotes validated Tier 3 entries to Tier 2
- Prunes low-quality entries
- Detects conflicts between tiers
- Assembles context packages for specialist agents
- Leverages specialist agents' capabilities for quality assessment

The librarian does NOT answer user questions directly — it curates
the knowledge that other agents use to answer questions.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from khonliang.knowledge.ingestion import IngestionPipeline, IngestionResult
from khonliang.knowledge.store import KnowledgeEntry, KnowledgeStore, Tier

logger = logging.getLogger(__name__)


class QualityAssessor(Protocol):
    """Protocol for agents that can assess information quality."""

    async def assess(
        self, content: str, context: str
    ) -> Dict[str, Any]:
        """
        Assess content quality.

        Returns dict with at minimum:
            {"valid": bool, "confidence": float, "reasoning": str}
        """
        ...


@dataclass
class LibrarianConfig:
    """Configuration for the librarian agent."""

    # Auto-index agent responses into Tier 3
    auto_index_responses: bool = True
    # Minimum confidence to keep Tier 3 entries
    min_derived_confidence: float = 0.3
    # Promote Tier 3 to Tier 2 after this many accesses
    promote_access_threshold: int = 5
    # Promote Tier 3 to Tier 2 above this confidence
    promote_confidence_threshold: float = 0.9
    # Max age in days for Tier 3 entries before pruning
    max_derived_age_days: float = 90
    # Max chars for context assembly
    max_context_chars: int = 6000
    # Default scope for unscoped content
    default_scope: str = "global"


class Librarian:
    """
    Knowledge curator that manages the three-tier store.

    Example:
        store = KnowledgeStore("knowledge.db")
        librarian = Librarian(store)

        # Set up axioms
        librarian.set_axiom("cite_sources", "Always cite data sources.")
        librarian.set_axiom("no_fabrication", "Never fabricate genealogy data.")

        # Ingest documents
        result = librarian.ingest_file("research/toll_notes.txt", scope="toll")

        # Index an agent's response
        librarian.index_response(
            content="The Tolle family migrated from Maryland...",
            title="Tolle migration",
            agent_id="researcher",
            query="migration patterns",
            scope="toll",
        )

        # Build context for a specialist agent
        context = librarian.build_context(
            query="When did the Tolles arrive in Indiana?",
            scope="toll",
        )

        # Maintenance
        librarian.auto_promote()
        librarian.prune()
    """

    def __init__(
        self,
        store: KnowledgeStore,
        config: Optional[LibrarianConfig] = None,
        summarizer: Optional[Callable[[str, str], str]] = None,
        classifier: Optional[Callable[[str], Dict[str, Any]]] = None,
        assessors: Optional[Dict[str, QualityAssessor]] = None,
    ):
        """
        Args:
            store: KnowledgeStore instance
            config: Librarian configuration
            summarizer: LLM-backed summarization function
            classifier: LLM-backed scope/tag classifier
            assessors: Named quality assessors (e.g. {"fact_checker": ...})
        """
        self.store = store
        self.config = config or LibrarianConfig()
        self._pipeline = IngestionPipeline(
            store=store,
            summarizer=summarizer,
            classifier=classifier,
        )
        self._assessors = assessors or {}

    # ------------------------------------------------------------------
    # Axiom management
    # ------------------------------------------------------------------

    def set_axiom(self, key: str, content: str) -> None:
        """Set a Tier 1 axiom."""
        self.store.set_axiom(key, content)

    def get_axioms(self) -> List[KnowledgeEntry]:
        """Get all axioms."""
        return self.store.get_axioms()

    # ------------------------------------------------------------------
    # Ingestion (Tier 2)
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        content: str,
        title: str,
        scope: Optional[str] = None,
        **kwargs,
    ) -> IngestionResult:
        """Ingest text content as Tier 2 imported knowledge."""
        return self._pipeline.ingest_text(
            content=content,
            title=title,
            scope=scope or self.config.default_scope,
            tier=Tier.IMPORTED,
            **kwargs,
        )

    def ingest_file(
        self, path: str, scope: Optional[str] = None
    ) -> IngestionResult:
        """Ingest a file as Tier 2 imported knowledge."""
        return self._pipeline.ingest_file(
            path=path,
            scope=scope or self.config.default_scope,
            tier=Tier.IMPORTED,
        )

    def ingest_directory(
        self,
        directory: str,
        scope: Optional[str] = None,
        patterns: Optional[List[str]] = None,
    ) -> IngestionResult:
        """Ingest a directory of files as Tier 2."""
        return self._pipeline.ingest_directory(
            directory=directory,
            scope=scope or self.config.default_scope,
            tier=Tier.IMPORTED,
            patterns=patterns,
        )

    # ------------------------------------------------------------------
    # Agent response indexing (Tier 3)
    # ------------------------------------------------------------------

    def index_response(
        self,
        content: str,
        title: str,
        agent_id: str,
        query: str = "",
        scope: Optional[str] = None,
        confidence: float = 0.7,
    ) -> IngestionResult:
        """
        Index an agent's response as Tier 3 derived knowledge.

        Called after a specialist agent answers a question. The response
        becomes searchable knowledge for future queries.
        """
        if not self.config.auto_index_responses:
            return IngestionResult(skipped=1)

        return self._pipeline.ingest_agent_response(
            content=content,
            title=title,
            agent_id=agent_id,
            scope=scope or self.config.default_scope,
            confidence=confidence,
            query=query,
        )

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_context(
        self,
        query: str,
        scope: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Build a context package for a specialist agent.

        Assembles Tier 1 axioms + relevant Tier 2/3 entries within
        the character budget.
        """
        return self.store.build_context(
            query=query,
            scope=scope,
            max_chars=max_chars or self.config.max_context_chars,
            include_axioms=True,
        )

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    async def assess_entry(
        self,
        entry_id: str,
        assessor_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Have a specialist agent assess a knowledge entry's quality.

        Args:
            entry_id: Entry to assess
            assessor_name: Which assessor to use (default: first available)

        Returns:
            Assessment result dict, or None if no assessors available.
        """
        entry = self.store.get(entry_id)
        if not entry:
            return None

        if not self._assessors:
            return None

        assessor_key = assessor_name or next(iter(self._assessors))
        assessor = self._assessors.get(assessor_key)
        if not assessor:
            return None

        axiom_context = self.store.get_axioms_text()

        try:
            result = await assessor.assess(entry.content, axiom_context)
            # Update confidence based on assessment
            if "confidence" in result:
                self.store.update_confidence(entry_id, result["confidence"])
            return result
        except Exception as e:
            logger.error(f"Assessment failed for {entry_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def auto_promote(self) -> int:
        """
        Promote Tier 3 entries that meet promotion criteria.

        Criteria: high confidence AND high access count.
        Returns number of entries promoted.
        """
        promoted = 0
        entries = self.store.get_by_tier(Tier.DERIVED)
        for entry in entries:
            if (
                entry.confidence >= self.config.promote_confidence_threshold
                and entry.access_count >= self.config.promote_access_threshold
            ):
                if self.store.promote(entry.id):
                    promoted += 1
                    logger.info(
                        f"Auto-promoted '{entry.title}' "
                        f"(confidence={entry.confidence:.0%}, "
                        f"accesses={entry.access_count})"
                    )
        return promoted

    def prune(self) -> int:
        """Prune low-quality Tier 3 entries."""
        return self.store.prune(
            tier=Tier.DERIVED,
            max_age_days=self.config.max_derived_age_days,
            min_confidence=self.config.min_derived_confidence,
        )

    def find_conflicts(
        self, scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find potential conflicts between knowledge entries.

        Looks for entries with overlapping titles/content but different
        facts. Returns pairs of conflicting entries for review.

        This is a simple heuristic — for deeper analysis, use an
        assessor agent.
        """
        entries = self.store.get_by_scope(scope) if scope else []
        if not entries:
            entries = (
                self.store.get_by_tier(Tier.IMPORTED)
                + self.store.get_by_tier(Tier.DERIVED)
            )

        conflicts = []
        seen = set()

        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                pair_key = tuple(sorted([a.id, b.id]))
                if pair_key in seen:
                    continue

                # Simple overlap detection: same title keywords
                a_words = set(a.title.lower().split())
                b_words = set(b.title.lower().split())
                overlap = a_words & b_words - {"the", "a", "an", "of", "in"}

                if len(overlap) >= 2 and a.tier != b.tier:
                    conflicts.append({
                        "entry_a": a.to_dict(),
                        "entry_b": b.to_dict(),
                        "overlap_words": list(overlap),
                        "tier_mismatch": a.tier != b.tier,
                    })
                    seen.add(pair_key)

        return conflicts

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get librarian and store status."""
        stats = self.store.get_stats()
        stats["config"] = {
            "auto_index": self.config.auto_index_responses,
            "promote_threshold": self.config.promote_confidence_threshold,
            "max_derived_age_days": self.config.max_derived_age_days,
        }
        stats["assessors"] = list(self._assessors.keys())
        return stats
