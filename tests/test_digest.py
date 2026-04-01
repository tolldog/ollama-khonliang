"""Tests for the digest module — store, synthesizer, and middleware."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from khonliang.digest.middleware import (
    digest_blackboard,
    digest_consensus,
    extract_from_response,
)
from khonliang.digest.store import DigestEntry, DigestStore
from khonliang.digest.synthesizer import (
    DigestConfig,
    DigestSynthesizer,
    format_entries,
)

# ---------------------------------------------------------------------------
# DigestStore
# ---------------------------------------------------------------------------

class TestDigestStore:
    def setup_method(self):
        self.store = DigestStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_record_and_get(self):
        entry = self.store.record("Found census record", source="researcher")
        assert entry.id
        assert entry.summary == "Found census record"
        assert entry.source == "researcher"
        assert entry.consumed is False

    def test_record_with_audience(self):
        entry = self.store.record(
            "Trade executed", source="trader", audience="transactions"
        )
        assert entry.audience == "transactions"

    def test_record_with_tags(self):
        entry = self.store.record(
            "New discovery", tags=["discovery", "census"]
        )
        assert entry.tags == ["discovery", "census"]

    def test_record_with_metadata(self):
        entry = self.store.record(
            "Found record", metadata={"person": "Patrick Smith"}
        )
        assert entry.metadata["person"] == "Patrick Smith"

    def test_get_unconsumed(self):
        self.store.record("A", source="a")
        self.store.record("B", source="b")
        entries = self.store.get_unconsumed()
        assert len(entries) == 2
        assert entries[0].summary == "A"  # oldest first

    def test_get_unconsumed_by_source(self):
        self.store.record("A", source="a")
        self.store.record("B", source="b")
        entries = self.store.get_unconsumed(source="a")
        assert len(entries) == 1

    def test_get_unconsumed_by_audience(self):
        self.store.record("A", audience="research")
        self.store.record("B", audience="transactions")
        self.store.record("C", audience="research")

        entries = self.store.get_unconsumed(audience="research")
        assert len(entries) == 2

    def test_get_since_hours(self):
        self.store.record("Recent")
        entries = self.store.get_since(hours=1)
        assert len(entries) == 1

    def test_get_since_audience(self):
        self.store.record("A", audience="ops")
        self.store.record("B", audience="dev")
        entries = self.store.get_since(hours=1, audience="ops")
        assert len(entries) == 1
        assert entries[0].summary == "A"

    def test_get_since_tag_filter(self):
        self.store.record("A", tags=["census"])
        self.store.record("B", tags=["birth"])
        entries = self.store.get_since(hours=1, tag="census")
        assert len(entries) == 1

    def test_get_since_excludes_consumed(self):
        e1 = self.store.record("A")
        self.store.record("B")
        self.store.mark_consumed([e1.id])

        entries = self.store.get_since(hours=1)
        assert len(entries) == 1
        assert entries[0].summary == "B"

    def test_get_since_include_consumed(self):
        e1 = self.store.record("A")
        self.store.record("B")
        self.store.mark_consumed([e1.id])

        entries = self.store.get_since(hours=1, include_consumed=True)
        assert len(entries) == 2

    def test_mark_consumed(self):
        e1 = self.store.record("A")
        e2 = self.store.record("B")
        count = self.store.mark_consumed([e1.id, e2.id], digest_id="d1")
        assert count == 2

        entries = self.store.get_unconsumed()
        assert len(entries) == 0

    def test_mark_consumed_empty(self):
        assert self.store.mark_consumed([]) == 0

    def test_get_stats(self):
        self.store.record("A", source="agent1")
        self.store.record("B", source="agent2")
        self.store.record("C", source="agent1")
        e = self.store.record("D", source="agent1")
        self.store.mark_consumed([e.id])

        stats = self.store.get_stats()
        assert stats["total_entries"] == 4
        assert stats["unconsumed"] == 3
        assert stats["by_source"]["agent1"] == 2  # 3 total, 1 consumed
        assert stats["by_source"]["agent2"] == 1

    def test_purge_consumed(self):
        e1 = self.store.record("Old")
        self.store.mark_consumed([e1.id])
        # Hack consumed_at to be old
        self.store._conn.execute(
            "UPDATE digest_entries SET consumed_at = ? WHERE id = ?",
            (time.time() - 999999, e1.id),
        )
        self.store._conn.commit()

        purged = self.store.purge_consumed(older_than_hours=1)
        assert purged == 1

    def test_get_unconsumed_limit(self):
        for i in range(10):
            self.store.record(f"Entry {i}")
        entries = self.store.get_unconsumed(limit=3)
        assert len(entries) == 3


# ---------------------------------------------------------------------------
# DigestEntry
# ---------------------------------------------------------------------------

class TestDigestEntry:
    def test_to_dict(self):
        entry = DigestEntry(
            id="x", summary="test", source="a",
            created_at=time.time(), audience="ops",
        )
        d = entry.to_dict()
        assert d["summary"] == "test"
        assert d["audience"] == "ops"


# ---------------------------------------------------------------------------
# DigestSynthesizer
# ---------------------------------------------------------------------------

class TestDigestSynthesizer:
    def setup_method(self):
        self.store = DigestStore(":memory:")

    def teardown_method(self):
        self.store.close()

    @pytest.mark.asyncio
    async def test_generate_empty(self):
        synth = DigestSynthesizer(self.store)
        result = await synth.generate()
        assert result.entry_count == 0
        assert "No new activity" in result.markdown

    @pytest.mark.asyncio
    async def test_generate_fallback_no_llm(self):
        self.store.record("Found census record", source="researcher")
        self.store.record("Resolved dead end", source="analyst")

        synth = DigestSynthesizer(self.store)
        result = await synth.generate()

        assert result.entry_count == 2
        assert result.synthesized is False
        assert "Found census record" in result.markdown
        assert "Resolved dead end" in result.markdown

    @pytest.mark.asyncio
    async def test_generate_with_llm(self):
        self.store.record("Found census", source="researcher")

        mock_client = AsyncMock()
        mock_client.generate.return_value = "Summary: one census record found."

        synth = DigestSynthesizer(self.store, client=mock_client)
        result = await synth.generate()

        assert result.synthesized is True
        assert "one census record found" in result.markdown
        mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_llm_failure_falls_back(self):
        self.store.record("Entry", source="agent")

        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("LLM down")

        synth = DigestSynthesizer(self.store, client=mock_client)
        result = await synth.generate()

        assert result.synthesized is False
        assert "Entry" in result.markdown

    @pytest.mark.asyncio
    async def test_generate_marks_consumed(self):
        self.store.record("A")
        self.store.record("B")

        synth = DigestSynthesizer(self.store)
        result = await synth.generate()

        assert result.entry_count == 2
        remaining = self.store.get_unconsumed()
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_generate_no_mark_consumed(self):
        self.store.record("A")

        synth = DigestSynthesizer(self.store)
        await synth.generate(mark_consumed=False)

        remaining = self.store.get_unconsumed()
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_generate_by_audience(self):
        self.store.record("Trade X", audience="transactions")
        self.store.record("Config changed", audience="changes")
        self.store.record("Trade Y", audience="transactions")

        synth = DigestSynthesizer(self.store)
        result = await synth.generate(audience="transactions")

        assert result.entry_count == 2
        assert "Trade X" in result.markdown
        assert "Config changed" not in result.markdown

    @pytest.mark.asyncio
    async def test_generate_by_hours(self):
        self.store.record("Recent")

        synth = DigestSynthesizer(self.store)
        result = await synth.generate(hours=1)
        assert result.entry_count == 1

    @pytest.mark.asyncio
    async def test_custom_config(self):
        self.store.record("Test", source="agent")

        mock_client = AsyncMock()
        mock_client.generate.return_value = "Custom digest."

        config = DigestConfig(
            system_prompt="You are a genealogy assistant.",
            synthesis_prompt="Focus on discoveries.",
            title_template="Research Digest — {date}",
        )
        synth = DigestSynthesizer(self.store, config, client=mock_client)
        result = await synth.generate()

        assert "Research Digest" in result.title
        call_kwargs = mock_client.generate.call_args
        assert "genealogy" in call_kwargs.kwargs["system"]


# ---------------------------------------------------------------------------
# format_entries
# ---------------------------------------------------------------------------

class TestFormatEntries:
    def test_empty(self):
        assert format_entries([]) == "No activity recorded."

    def test_grouped(self):
        entries = [
            DigestEntry(id="1", summary="A", source="agent1", created_at=time.time()),
            DigestEntry(id="2", summary="B", source="agent2", created_at=time.time()),
        ]
        text = format_entries(entries, group_by_source=True)
        assert "[agent1]" in text
        assert "[agent2]" in text

    def test_ungrouped(self):
        entries = [
            DigestEntry(id="1", summary="A", source="agent1", created_at=time.time()),
        ]
        text = format_entries(entries, group_by_source=False)
        assert "[agent1]" in text
        assert "A" in text

    def test_with_tags(self):
        entries = [
            DigestEntry(
                id="1", summary="A", source="x",
                tags=["census"], created_at=time.time(),
            ),
        ]
        text = format_entries(entries)
        assert "census" in text


# ---------------------------------------------------------------------------
# Middleware: extract_from_response
# ---------------------------------------------------------------------------

class TestExtractFromResponse:
    def setup_method(self):
        self.store = DigestStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_extracts_digest(self):
        response = {
            "content": "Full response text",
            "role": "researcher",
            "metadata": {"digest": "Found census record"},
        }
        assert extract_from_response(response, self.store) is True

        entries = self.store.get_unconsumed()
        assert len(entries) == 1
        assert entries[0].summary == "Found census record"
        assert entries[0].source == "researcher"

    def test_extracts_audience(self):
        response = {
            "role": "trader",
            "metadata": {
                "digest": "Trade executed",
                "digest_audience": "transactions",
            },
        }
        extract_from_response(response, self.store)

        entries = self.store.get_unconsumed()
        assert entries[0].audience == "transactions"

    def test_extracts_tags(self):
        response = {
            "metadata": {
                "digest": "Thing happened",
                "digest_tags": ["important", "census"],
            },
        }
        extract_from_response(response, self.store)
        entries = self.store.get_unconsumed()
        assert entries[0].tags == ["important", "census"]

    def test_no_digest_key(self):
        response = {"content": "No digest", "metadata": {"other": "stuff"}}
        assert extract_from_response(response, self.store) is False

    def test_no_metadata(self):
        response = {"content": "No metadata"}
        assert extract_from_response(response, self.store) is False

    def test_source_override(self):
        response = {
            "role": "researcher",
            "metadata": {"digest": "Found something"},
        }
        extract_from_response(response, self.store, source="custom_agent")

        entries = self.store.get_unconsumed()
        assert entries[0].source == "custom_agent"

    def test_default_audience(self):
        response = {"metadata": {"digest": "Test"}}
        extract_from_response(response, self.store, default_audience="ops")

        entries = self.store.get_unconsumed()
        assert entries[0].audience == "ops"

    def test_extra_metadata_passed_through(self):
        response = {
            "metadata": {
                "digest": "Found record",
                "person": "Patrick Smith",
                "confidence": 0.85,
            },
        }
        extract_from_response(response, self.store)
        entries = self.store.get_unconsumed()
        assert entries[0].metadata["person"] == "Patrick Smith"


# ---------------------------------------------------------------------------
# Middleware: digest_blackboard
# ---------------------------------------------------------------------------

class TestDigestBlackboard:
    def setup_method(self):
        self.store = DigestStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_hooks_all_sections(self):
        from khonliang.gateway.blackboard import Blackboard

        board = Blackboard()
        digest_blackboard(board, self.store)

        board.post("agent1", "findings", "key1", "Found a record")

        entries = self.store.get_unconsumed()
        assert len(entries) == 1
        assert entries[0].source == "agent1"
        assert "Found a record" in entries[0].summary

    def test_hooks_specific_sections(self):
        from khonliang.gateway.blackboard import Blackboard

        board = Blackboard()
        digest_blackboard(board, self.store, sections=["alerts"])

        board.post("agent1", "alerts", "k1", "Alert!")
        board.post("agent2", "other", "k2", "Not tracked")

        entries = self.store.get_unconsumed()
        assert len(entries) == 1
        assert entries[0].summary == "Alert!"

    def test_blackboard_still_works(self):
        from khonliang.gateway.blackboard import Blackboard

        board = Blackboard()
        digest_blackboard(board, self.store)

        board.post("a", "s", "k", "content")
        data = board.read("s")
        assert data["k"] == "content"

    def test_audience_set(self):
        from khonliang.gateway.blackboard import Blackboard

        board = Blackboard()
        digest_blackboard(board, self.store, audience="board_updates")

        board.post("a", "s", "k", "content")

        entries = self.store.get_unconsumed()
        assert entries[0].audience == "board_updates"


# ---------------------------------------------------------------------------
# Middleware: digest_consensus
# ---------------------------------------------------------------------------

class TestDigestConsensus:
    def setup_method(self):
        self.store = DigestStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_records_consensus(self):
        callback = digest_consensus(self.store)

        result = MagicMock()
        result.action = "approve"
        result.confidence = 0.85
        result.votes = [1, 2, 3]

        callback(result, subject="Escalate issue?")

        entries = self.store.get_unconsumed()
        assert len(entries) == 1
        assert "approve" in entries[0].summary
        assert "Escalate issue?" in entries[0].summary
        assert entries[0].metadata["confidence"] == 0.85

    def test_audience_set(self):
        callback = digest_consensus(self.store, audience="decisions")

        result = MagicMock()
        result.action = "reject"
        result.confidence = 0.6
        result.votes = []

        callback(result)

        entries = self.store.get_unconsumed()
        assert entries[0].audience == "decisions"
