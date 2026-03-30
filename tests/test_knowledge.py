"""Tests for the three-tier knowledge store."""

import os
import tempfile
import time

from khonliang.knowledge.ingestion import IngestionPipeline
from khonliang.knowledge.librarian import Librarian
from khonliang.knowledge.reports import ReportBuilder
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


# ------------------------------------------------------------------
# ReportBuilder tests
# ------------------------------------------------------------------


def _temp_store():
    # re-define locally so tests below don't depend on module-level helper
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return KnowledgeStore(path), path


def _populated_store():
    store, path = _temp_store()

    store.set_axiom("rule1", "Always cite your sources.")

    store.add(KnowledgeEntry(
        id="i1",
        tier=Tier.IMPORTED,
        title="Migration notes",
        content="The family moved west in the 1800s.",
        scope="toll",
        source="manual",
        confidence=0.95,
    ))
    store.add(KnowledgeEntry(
        id="d1",
        tier=Tier.DERIVED,
        title="Roger Tolle birth",
        content="Roger Tolle was born around 1642 in Wales.",
        scope="toll",
        source="researcher",
        confidence=0.8,
    ))
    # Add a second derived entry with a slightly later updated_at so ordering
    # is deterministic in the "recent" assertions.
    time.sleep(0.01)
    store.add(KnowledgeEntry(
        id="d2",
        tier=Tier.DERIVED,
        title="Thomas origins",
        content="Thomas family came from Ohio.",
        scope="thomas",
        source="researcher",
        confidence=0.75,
    ))
    return store, path


def test_report_builder_no_store():
    builder = ReportBuilder()
    report = builder.knowledge_report()
    assert "No knowledge store" in report

    session = builder.session_report()
    assert "Session Summary" in session


def test_knowledge_report_sections():
    store, path = _populated_store()
    try:
        builder = ReportBuilder(store)
        report = builder.knowledge_report()

        assert "# Knowledge Report" in report
        assert "Total entries: 4" in report
        assert "Axioms (Tier 1): 1" in report
        assert "Imported (Tier 2): 1" in report
        assert "Derived (Tier 3): 2" in report
        # Recent research section should list derived entries
        assert "Recent Research" in report
        assert "Roger Tolle birth" in report
    finally:
        os.unlink(path)


def test_knowledge_report_recent_ordering():
    """Most-recently updated derived entry appears first."""
    store, path = _populated_store()
    try:
        builder = ReportBuilder(store)
        report = builder.knowledge_report()

        # d2 ("Thomas origins") was added last so should appear before d1
        idx_d2 = report.find("Thomas origins")
        idx_d1 = report.find("Roger Tolle birth")
        assert idx_d2 < idx_d1, "Most recent entry should appear first"
    finally:
        os.unlink(path)


def test_session_report_summary():
    store, path = _populated_store()
    try:
        builder = ReportBuilder(store)
        report = builder.session_report()

        assert "# Session Summary" in report
        assert "4 entries" in report
        # Should list recent derived titles
        assert "Roger Tolle birth" in report or "Thomas origins" in report
    finally:
        os.unlink(path)


def test_session_report_extra_context():
    store, path = _populated_store()
    try:
        builder = ReportBuilder(store)
        report = builder.session_report(extra_context="Tip: check parish records.")

        assert "Tip: check parish records." in report
    finally:
        os.unlink(path)


def test_topic_report_found():
    store, path = _populated_store()
    try:
        builder = ReportBuilder(store)
        report = builder.topic_report("Wales")

        assert "# Topic: Wales" in report
        assert "Roger Tolle birth" in report
        assert "RESEARCHED" in report
    finally:
        os.unlink(path)


def test_topic_report_no_results():
    store, path = _populated_store()
    try:
        builder = ReportBuilder(store)
        report = builder.topic_report("nonexistent_xyzzy")

        assert "No knowledge found" in report
    finally:
        os.unlink(path)


def test_topic_report_scoped():
    store, path = _populated_store()
    try:
        builder = ReportBuilder(store)
        # "Ohio" appears in the thomas-scoped entry only
        report = builder.topic_report("Ohio", scope="thomas")

        assert "Thomas origins" in report
        # toll-scoped entry should not appear
        assert "Migration notes" not in report
    finally:
        os.unlink(path)
