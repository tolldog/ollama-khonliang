"""Tests for the three-tier knowledge store and semantic triple store."""

import os
import tempfile
import time

import pytest

from khonliang.knowledge.ingestion import IngestionPipeline
from khonliang.knowledge.librarian import Librarian
from khonliang.knowledge.reports import ReportBuilder
from khonliang.knowledge.store import KnowledgeEntry, KnowledgeStore, Tier
from khonliang.knowledge.triples import (
    Triple,
    TripleStore,
    split_sources,
)


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

    # scope=None searches across all scopes
    results = store.search("Ohio")
    assert {r.scope for r in results} == {"toll", "thomas"}
    os.unlink(path)


def test_search_tokenizes_slash_and_dot_queries():
    store, path = _temp_store()
    store.add(KnowledgeEntry(
        id="repo1",
        tier=Tier.IMPORTED,
        title="Repo notes",
        content="The gadgetlib repo pins release v42 of the toolchain.",
        scope="global",
        source="test",
    ))

    # 'owner/repo' splits into subtokens instead of being dropped whole
    results = store.search("acme/gadgetlib")
    assert any(r.id == "repo1" for r in results)

    # dotted versions split too; len<=1 fragments are discarded as noise
    results = store.search("v42.0.1")
    assert any(r.id == "repo1" for r in results)

    # a query with no usable tokens still returns [] without error
    assert store.search("/ .. @ !") == []
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


# ------------------------------------------------------------------
# TripleStore tests
# ------------------------------------------------------------------


def _temp_triple_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return TripleStore(path), path


def test_triple_add_and_get():
    store, path = _temp_triple_store()
    try:
        store.add("Roger Tolle", "born_in", "Wales", confidence=0.9, source="gedcom")
        store.add("Roger Tolle", "born_year", "1642", confidence=0.95, source="gedcom")

        triples = store.get(subject="Roger Tolle")
        assert len(triples) == 2
        subjects = {t.subject for t in triples}
        assert subjects == {"Roger Tolle"}
        preds = {t.predicate for t in triples}
        assert "born_in" in preds
        assert "born_year" in preds
    finally:
        os.unlink(path)


def test_triple_dedup_reinforcement():
    store, path = _temp_triple_store()
    try:
        store.add("TSLA", "correlates_with", "AMD", confidence=0.6)
        store.add("TSLA", "correlates_with", "AMD", confidence=0.8)

        triples = store.get(subject="TSLA", predicate="correlates_with")
        assert len(triples) == 1
        # confidence should be max of 0.6 and 0.8
        assert triples[0].confidence == 0.8
    finally:
        os.unlink(path)


def test_triple_dedup_keeps_lower_when_existing_higher():
    store, path = _temp_triple_store()
    try:
        store.add("TSLA", "correlates_with", "AMD", confidence=0.9)
        store.add("TSLA", "correlates_with", "AMD", confidence=0.5)

        triples = store.get(subject="TSLA", predicate="correlates_with")
        assert len(triples) == 1
        # confidence stays at max (0.9)
        assert triples[0].confidence == 0.9
    finally:
        os.unlink(path)


# Multi-source provenance (bug_khonliang-researcher_a905176b)
# ------------------------------------------------------------------


def test_split_sources():
    assert split_sources("") == []
    assert split_sources(None) == []
    assert split_sources("paper:a") == ["paper:a"]
    assert split_sources("paper:a\nidea:b\npaper:a") == ["paper:a", "idea:b"]


def test_single_source_is_stored_bare_and_backward_compatible():
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.7, source="paper:x")
        t = store.get(subject="S")[0]
        assert t.source == "paper:x"          # no separator for the common case
        assert t.sources == ["paper:x"]
    finally:
        os.unlink(path)


def test_add_unions_sources_and_keeps_max_confidence():
    # A blog re-asserting a paper's triple must NOT clobber the paper's source
    # or downgrade its confidence — it adds a second provenance token.
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.8, source="paper:x")
        store.add("S", "p", "O", confidence=0.4, source="idea:y")

        triples = store.get(subject="S")
        assert len(triples) == 1                       # still one row
        assert triples[0].confidence == 0.8            # paper's higher conf wins
        assert triples[0].sources == ["paper:x", "idea:y"]
    finally:
        os.unlink(path)


def test_remove_source_keeps_triple_while_other_sources_remain():
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", source="paper:x")
        store.add("S", "p", "O", source="idea:y")

        deleted = store.remove_source("S", "p", "O", "paper:x")
        assert deleted is False                        # idea:y still supports it
        t = store.get(subject="S")
        assert len(t) == 1 and t[0].sources == ["idea:y"]
    finally:
        os.unlink(path)


def test_remove_strongest_source_recomputes_confidence_to_remaining_max():
    # Per-source confidence: striking the 0.9 paper must drop the surviving
    # fact to the 0.4 blog's strength, not leave it overstated at 0.9.
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.9, source="paper:x")
        store.add("S", "p", "O", confidence=0.4, source="idea:y")
        assert store.get(subject="S")[0].confidence == 0.9

        deleted = store.remove_source("S", "p", "O", "paper:x")
        assert deleted is False
        t = store.get(subject="S")[0]
        assert t.sources == ["idea:y"]
        assert t.confidence == 0.4          # recomputed, not stale 0.9
    finally:
        os.unlink(path)


def test_remove_source_preserves_decay_timestamp():
    # Retracting a contributor must not reset the aging clock on the survivor.
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.9, source="paper:x")
        store.add("S", "p", "O", confidence=0.4, source="idea:y")
        before = store.get(subject="S")[0].updated_at
        time.sleep(0.01)
        store.remove_source("S", "p", "O", "paper:x")
        after = store.get(subject="S")[0].updated_at
        assert after == before              # updated_at untouched by retraction
    finally:
        os.unlink(path)


def test_per_source_confidence_is_independent():
    # Re-asserting one source at higher confidence lifts the max; the other
    # source's strength is untouched and resurfaces if the stronger is removed.
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.5, source="paper:x")
        store.add("S", "p", "O", confidence=0.3, source="idea:y")
        store.add("S", "p", "O", confidence=0.8, source="idea:y")   # idea strengthens
        assert store.get(subject="S")[0].confidence == 0.8
        store.remove_source("S", "p", "O", "idea:y")
        assert store.get(subject="S")[0].confidence == 0.5          # paper's own value
    finally:
        os.unlink(path)


def test_get_exposes_per_source_confidence(tmp_path):
    # FR cecbd365: a co-sourced triple must surface EACH source's own
    # confidence, not just the aggregate max, so the concept matrix can credit a
    # weaker paper at its real strength.
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.9, source="paper:x")
        store.add("S", "p", "O", confidence=0.4, source="idea:y")
        t = store.get(subject="S")[0]
        assert t.source_confidences == {"paper:x": 0.9, "idea:y": 0.4}
        assert t.confidence_for("paper:x") == 0.9
        assert t.confidence_for("idea:y") == 0.4
        assert t.confidence == 0.9  # aggregate cache unchanged (max)
    finally:
        os.unlink(path)


def test_confidence_for_unknown_source_falls_back_to_aggregate(tmp_path):
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.7, source="paper:x")
        t = store.get(subject="S")[0]
        assert t.confidence_for("paper:never") == 0.7  # falls back to aggregate
    finally:
        os.unlink(path)


def test_directly_constructed_triple_confidence_for_uses_aggregate():
    # No store read path populated source_confidences → confidence_for must not
    # KeyError; it degrades to the aggregate confidence.
    t = Triple(subject="S", predicate="p", object="O", confidence=0.6)
    assert t.source_confidences == {}
    assert t.confidence_for("anything") == 0.6


def test_anonymous_source_excluded_from_per_source_map(tmp_path):
    # An anonymous ("") contributor carries confidence but no display token, so
    # it must not appear in source_confidences (mirrors Triple.sources).
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.5)               # anonymous ""
        store.add("S", "p", "O", confidence=0.8, source="paper:x")
        t = store.get(subject="S")[0]
        assert t.source_confidences == {"paper:x": 0.8}
        assert "" not in t.source_confidences
    finally:
        os.unlink(path)


def test_search_also_populates_per_source_confidence(tmp_path):
    store, path = _temp_triple_store()
    try:
        store.add("Subject", "p", "Object", confidence=0.9, source="paper:x")
        store.add("Subject", "p", "Object", confidence=0.4, source="idea:y")
        t = store.search("Subject")[0]
        assert t.confidence_for("idea:y") == 0.4
    finally:
        os.unlink(path)


def test_remove_source_deletes_triple_when_last_source():
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", source="idea:y")
        deleted = store.remove_source("S", "p", "O", "idea:y")
        assert deleted is True
        assert store.get(subject="S") == []
    finally:
        os.unlink(path)


def test_decay_persists_for_sourced_triples_after_resync():
    # Decay must hit the per-source rows, not just the cache — otherwise a
    # later add/remove_source recompute would restore the pre-decay value.
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.9, source="paper:x")
        store.add("S", "p", "O", confidence=0.6, source="idea:y")
        # Force decay by backdating updated_at well past the window.
        import sqlite3
        conn = sqlite3.connect(path)
        conn.execute("UPDATE triples SET updated_at = 0, decay_rate = 0.5")
        conn.commit()
        conn.close()

        store.apply_decay(max_age_days=1)
        decayed = store.get(subject="S")[0].confidence
        assert decayed < 0.9                      # actually decayed

        # A later remove_source recompute must NOT resurrect pre-decay 0.9.
        store.remove_source("S", "p", "O", "idea:y")
        assert store.get(subject="S")[0].confidence == pytest.approx(decayed)
    finally:
        os.unlink(path)


def test_anonymous_evidence_survives_a_later_sourced_add():
    # An anonymous assertion must not be silently dropped when a sourced
    # duplicate arrives: its confidence is kept and it still supports the fact.
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.7)              # anonymous
        store.add("S", "p", "O", confidence=0.5, source="paper:x")
        t = store.get(subject="S")[0]
        assert t.confidence == 0.7                            # anonymous kept
        assert t.sources == ["paper:x"]                       # display tokens only

        # Removing the named source leaves the fact alive on anonymous support.
        deleted = store.remove_source("S", "p", "O", "paper:x")
        assert deleted is False
        assert store.get(subject="S")[0].confidence == 0.7
    finally:
        os.unlink(path)


def test_legacy_triples_are_backfilled_into_source_table():
    # A triple written before triple_sources existed (single source, no rows)
    # must be backfilled on the next open so remove_source treats it correctly.
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store = TripleStore(path)
        store.add("S", "p", "O", confidence=0.7, source="paper:x")
        # Simulate a legacy DB: wipe the provenance table, keep the triples row.
        import sqlite3
        conn = sqlite3.connect(path)
        conn.execute("DELETE FROM triple_sources")
        conn.commit()
        conn.close()

        store2 = TripleStore(path)            # _ensure_schema backfills
        assert store2.remove_source("S", "p", "O", "paper:x") is True
        assert store2.get(subject="S") == []
    finally:
        os.unlink(path)


def test_hard_remove_prunes_orphan_source_rows():
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", confidence=0.7, source="paper:x")
        store.remove("S", "p", "O")
        import sqlite3
        conn = sqlite3.connect(path)
        n = conn.execute("SELECT COUNT(*) FROM triple_sources").fetchone()[0]
        conn.close()
        assert n == 0                          # no orphaned provenance rows
    finally:
        os.unlink(path)


def test_remove_source_noop_for_unknown_source_or_triple():
    store, path = _temp_triple_store()
    try:
        store.add("S", "p", "O", source="paper:x")
        assert store.remove_source("S", "p", "O", "idea:missing") is False
        assert store.remove_source("nope", "p", "O", "paper:x") is False
        # original triple untouched
        assert store.get(subject="S")[0].sources == ["paper:x"]
    finally:
        os.unlink(path)


def test_remove_source_normalizes_predicate():
    # remove_source must match the same normalized predicate add() stored.
    store, path = _temp_triple_store()
    try:
        store.add("S", "Is Applicable To", "O", source="paper:x")
        # query/removal with a differently-cased/spaced predicate still matches
        assert store.remove_source("S", "applicable_to", "O", "paper:x") is True
        assert store.get(subject="S") == []
    finally:
        os.unlink(path)


def test_triple_get_by_predicate():
    store, path = _temp_triple_store()
    try:
        store.add("Alice", "knows", "Bob")
        store.add("Carol", "knows", "Dave")
        store.add("Alice", "likes", "cats")

        triples = store.get(predicate="knows")
        assert len(triples) == 2

        triples = store.get(predicate="likes")
        assert len(triples) == 1
        assert triples[0].subject == "Alice"
    finally:
        os.unlink(path)


def test_triple_get_min_confidence_filter():
    store, path = _temp_triple_store()
    try:
        store.add("A", "rel", "B", confidence=0.9)
        store.add("A", "rel", "C", confidence=0.4)
        store.add("A", "rel", "D", confidence=0.2)

        triples = store.get(subject="A", min_confidence=0.5)
        assert len(triples) == 1
        assert triples[0].object == "B"
    finally:
        os.unlink(path)


def test_triple_get_limit():
    store, path = _temp_triple_store()
    try:
        for i in range(10):
            store.add("subject", f"pred{i}", "object", confidence=float(i) / 10)

        triples = store.get(subject="subject", limit=3)
        assert len(triples) == 3
        # Should return the highest-confidence triples
        confidences = [t.confidence for t in triples]
        assert confidences == sorted(confidences, reverse=True)
    finally:
        os.unlink(path)


def test_triple_search():
    store, path = _temp_triple_store()
    try:
        store.add("Roger Tolle", "born_in", "Wales")
        store.add("Roger Tolle", "migrated_to", "Maryland")
        store.add("Other Person", "born_in", "England")

        results = store.search("Wales")
        assert len(results) == 1
        assert results[0].object == "Wales"

        results = store.search("Roger")
        assert len(results) == 2
    finally:
        os.unlink(path)


def test_triple_build_context_all():
    store, path = _temp_triple_store()
    try:
        store.add("Alice", "likes", "cats", confidence=0.9)
        store.add("Bob", "dislikes", "dogs", confidence=0.7)

        ctx = store.build_context()
        assert "Alice likes cats" in ctx
        assert "Bob dislikes dogs" in ctx
        # Confidence formatted as percentage
        assert "90%" in ctx
    finally:
        os.unlink(path)


def test_triple_build_context_subjects_filter():
    store, path = _temp_triple_store()
    try:
        store.add("Alice", "likes", "cats", confidence=0.9)
        store.add("Bob", "likes", "dogs", confidence=0.8)
        store.add("Carol", "likes", "fish", confidence=0.7)

        ctx = store.build_context(subjects=["Alice", "Bob"])
        assert "Alice" in ctx
        assert "Bob" in ctx
        assert "Carol" not in ctx
    finally:
        os.unlink(path)


def test_triple_build_context_predicates_filter():
    store, path = _temp_triple_store()
    try:
        store.add("Alice", "likes", "cats", confidence=0.9)
        store.add("Bob", "dislikes", "cats", confidence=0.8)
        store.add("Carol", "owns", "cats", confidence=0.7)

        ctx = store.build_context(predicates=["likes", "owns"])
        assert "likes" in ctx
        assert "owns" in ctx
        assert "dislikes" not in ctx
    finally:
        os.unlink(path)


def test_triple_build_context_max_triples():
    store, path = _temp_triple_store()
    try:
        for i in range(10):
            store.add("subject", f"pred{i}", "object", confidence=float(i + 1) / 10)

        ctx = store.build_context(max_triples=3)
        lines = [ln for ln in ctx.splitlines() if ln.strip()]
        assert len(lines) == 3
    finally:
        os.unlink(path)


def test_triple_build_context_min_confidence():
    store, path = _temp_triple_store()
    try:
        store.add("A", "rel", "B", confidence=0.9)
        store.add("A", "rel", "C", confidence=0.2)

        # min_confidence=0.5 should exclude the 0.2 triple
        ctx = store.build_context(min_confidence=0.5)
        assert "B" in ctx
        assert "C" not in ctx
    finally:
        os.unlink(path)


def test_triple_build_context_ordered_by_confidence():
    store, path = _temp_triple_store()
    try:
        store.add("A", "rel", "low", confidence=0.3)
        store.add("A", "rel", "high", confidence=0.9)
        store.add("A", "rel", "mid", confidence=0.6)

        ctx = store.build_context(subjects=["A"])
        lines = ctx.splitlines()
        # Highest-confidence triple should come first
        assert "high" in lines[0]
    finally:
        os.unlink(path)


def test_triple_apply_decay():
    store, path = _temp_triple_store()
    try:
        store.add("A", "rel", "B", confidence=0.5)

        # Force updated_at to be old by directly manipulating the DB
        import sqlite3
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        old_time = time.time() - (200 * 86400)  # 200 days ago
        conn.execute("UPDATE triples SET updated_at = ?", (old_time,))
        conn.commit()
        conn.close()

        removed = store.apply_decay(max_age_days=90)
        # Confidence was 0.5, after one decay step: 0.5 * (1 - 0.01) = 0.495
        # Not below 0.1 threshold, so not removed yet
        assert removed == 0

        # A triple starting at 0.05 should be removed
        store.add("X", "rel", "Y", confidence=0.05)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            "UPDATE triples SET updated_at = ? WHERE subject = 'X'", (old_time,)
        )
        conn.commit()
        conn.close()

        removed = store.apply_decay(max_age_days=90)
        assert removed == 1
        remaining = store.get(subject="X")
        assert remaining == []
    finally:
        os.unlink(path)


def test_triple_remove():
    store, path = _temp_triple_store()
    try:
        store.add("A", "rel", "B")
        store.add("A", "rel", "C")
        store.add("A", "other", "D")

        removed = store.remove("A", predicate="rel", obj="B")
        assert removed == 1
        remaining = store.get(subject="A")
        assert len(remaining) == 2

        removed = store.remove("A", predicate="rel")
        assert removed == 1
        remaining = store.get(subject="A")
        assert len(remaining) == 1

        removed = store.remove("A")
        assert removed == 1
        assert store.get(subject="A") == []
    finally:
        os.unlink(path)


def test_triple_get_stats():
    store, path = _temp_triple_store()
    try:
        store.add("Alice", "likes", "cats")
        store.add("Alice", "owns", "dog")
        store.add("Bob", "likes", "birds")

        stats = store.get_stats()
        assert stats["total_triples"] == 3
        assert stats["unique_subjects"] == 2
        assert stats["unique_predicates"] == 2
    finally:
        os.unlink(path)
