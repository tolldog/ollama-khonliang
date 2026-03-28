"""Tests for the three-tier knowledge store."""

import os
import tempfile

from khonliang.knowledge.ingestion import IngestionPipeline
from khonliang.knowledge.librarian import Librarian
from khonliang.knowledge.store import KnowledgeEntry, KnowledgeStore, Tier


def _temp_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return KnowledgeStore(path), path


def test_axioms():
    store, path = _temp_store()
    store.set_axiom("rule1", "Never fabricate data.")
    store.set_axiom("rule2", "Always cite sources.")

    axioms = store.get_axioms()
    assert len(axioms) == 2
    assert "Never fabricate" in store.get_axioms_text()
    os.unlink(path)


def test_add_and_search():
    store, path = _temp_store()
    store.add(KnowledgeEntry(
        id="test1",
        tier=Tier.IMPORTED,
        title="Toll family migration",
        content="The Toll family moved from Maryland to Virginia to Ohio.",
        scope="toll",
        source="test",
    ))
    store.add(KnowledgeEntry(
        id="test2",
        tier=Tier.DERIVED,
        title="Thomas family origins",
        content="The Thomas family came from Ohio.",
        scope="thomas",
        source="researcher",
        confidence=0.8,
    ))

    results = store.search("migration")
    assert len(results) >= 1
    assert any("Toll" in r.title for r in results)

    results = store.search("Ohio", scope="toll")
    assert len(results) >= 1
    os.unlink(path)


def test_promote_demote():
    store, path = _temp_store()
    store.add(KnowledgeEntry(
        id="d1",
        tier=Tier.DERIVED,
        title="Test entry",
        content="Some derived knowledge.",
        confidence=0.9,
    ))

    assert store.promote("d1")
    entry = store.get("d1")
    assert entry.tier == Tier.IMPORTED

    assert store.demote("d1")
    entry = store.get("d1")
    assert entry.tier == Tier.DERIVED
    os.unlink(path)


def test_prune():
    store, path = _temp_store()
    store.add(KnowledgeEntry(
        id="low_quality",
        tier=Tier.DERIVED,
        title="Bad info",
        content="Something unreliable.",
        confidence=0.1,
    ))
    store.add(KnowledgeEntry(
        id="high_quality",
        tier=Tier.DERIVED,
        title="Good info",
        content="Something reliable.",
        confidence=0.95,
    ))

    pruned = store.prune(min_confidence=0.3)
    assert pruned == 1
    assert store.get("low_quality") is None
    assert store.get("high_quality") is not None
    os.unlink(path)


def test_build_context():
    store, path = _temp_store()
    store.set_axiom("cite", "Always cite your sources.")
    store.add(KnowledgeEntry(
        id="doc1",
        tier=Tier.IMPORTED,
        title="Migration notes",
        content="The family moved west in the 1800s.",
        scope="toll",
    ))

    context = store.build_context("migration", scope="toll")
    assert "[RULES]" in context
    assert "cite" in context.lower()
    assert "[KNOWLEDGE]" in context
    assert "family moved west" in context
    os.unlink(path)


def test_ingestion_pipeline():
    store, path = _temp_store()
    pipeline = IngestionPipeline(store)

    result = pipeline.ingest_text(
        content="The Tolle family left Maryland around 1780.",
        title="Tolle departure",
        source="research_notes",
        scope="toll",
    )
    assert result.added == 1

    # Duplicate should be skipped
    result = pipeline.ingest_text(
        content="The Tolle family left Maryland around 1780.",
        title="Tolle departure",
        source="research_notes",
        scope="toll",
    )
    assert result.skipped == 1
    os.unlink(path)


def test_ingestion_file(tmp_path):
    store, db_path = _temp_store()
    pipeline = IngestionPipeline(store)

    test_file = tmp_path / "notes.txt"
    test_file.write_text("Roger Tolle was born in Wales in 1642.")

    result = pipeline.ingest_file(str(test_file), scope="toll")
    assert result.added == 1

    entries = store.search("Roger Wales")
    assert len(entries) >= 1
    os.unlink(db_path)


def test_librarian():
    store, path = _temp_store()
    librarian = Librarian(store)

    librarian.set_axiom("rule1", "Never guess dates.")

    result = librarian.ingest_text(
        content="Willis Tolle was born about 1826 in Ohio.",
        title="Willis Tolle birth",
        scope="toll",
    )
    assert result.added == 1

    librarian.index_response(
        content="Based on the census, Willis was a farmer.",
        title="Willis occupation",
        agent_id="researcher",
        query="what did Willis do?",
        scope="toll",
    )

    context = librarian.build_context("Willis Tolle", scope="toll")
    assert "Never guess" in context
    assert "Willis" in context

    stats = librarian.get_status()
    assert stats["total_entries"] >= 3  # 1 axiom + 1 imported + 1 derived
    os.unlink(path)


def test_stats():
    store, path = _temp_store()
    store.set_axiom("r1", "Rule one.")
    store.add(KnowledgeEntry(
        id="i1", tier=Tier.IMPORTED, title="Doc", content="Content."
    ))
    store.add(KnowledgeEntry(
        id="d1", tier=Tier.DERIVED, title="Derived", content="Content.",
        confidence=0.7,
    ))

    stats = store.get_stats()
    assert stats["total_entries"] == 3
    assert stats["by_tier"]["axiom"] == 1
    assert stats["by_tier"]["imported"] == 1
    assert stats["by_tier"]["derived"] == 1
    os.unlink(path)
