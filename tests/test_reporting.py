"""Tests for the reporting module — manager, detector, server, and templates."""

import time

import pytest

from khonliang.reporting.detector import ReportDetector
from khonliang.reporting.manager import Report, ReportManager
from khonliang.reporting.templates import ReportTheme

# ---------------------------------------------------------------------------
# ReportManager
# ---------------------------------------------------------------------------

class TestReportManager:
    def setup_method(self):
        self.mgr = ReportManager(":memory:")

    def teardown_method(self):
        self.mgr.close()

    def test_create_and_get(self):
        report = self.mgr.create(
            title="Test Report",
            content_markdown="## Findings\nAll good.",
            report_type="analysis",
            created_by="agent_1",
        )
        assert report.id
        assert report.title == "Test Report"
        assert report.report_type == "analysis"
        assert report.view_count == 0

        fetched = self.mgr.get(report.id)
        assert fetched is not None
        assert fetched.title == "Test Report"
        assert fetched.view_count == 1  # tracked

    def test_get_nonexistent_returns_none(self):
        assert self.mgr.get("nonexistent") is None

    def test_get_without_view_tracking(self):
        report = self.mgr.create(title="T", content_markdown="C", report_type="x")
        fetched = self.mgr.get(report.id, track_view=False)
        assert fetched is not None
        assert fetched.view_count == 0

    def test_list_reports(self):
        self.mgr.create(title="A", content_markdown="a", report_type="analysis")
        self.mgr.create(title="B", content_markdown="b", report_type="research")
        self.mgr.create(title="C", content_markdown="c", report_type="analysis")

        all_reports = self.mgr.list_reports()
        assert len(all_reports) == 3

        analysis = self.mgr.list_reports(report_type="analysis")
        assert len(analysis) == 2

    def test_list_by_creator(self):
        self.mgr.create(title="A", content_markdown="a", created_by="bot1")
        self.mgr.create(title="B", content_markdown="b", created_by="bot2")

        bot1_reports = self.mgr.list_reports(created_by="bot1")
        assert len(bot1_reports) == 1
        assert bot1_reports[0].title == "A"

    def test_list_ordered_newest_first(self):
        self.mgr.create(title="First", content_markdown="a")
        self.mgr.create(title="Second", content_markdown="b")

        reports = self.mgr.list_reports()
        assert reports[0].title == "Second"
        assert reports[1].title == "First"

    def test_delete(self):
        report = self.mgr.create(title="T", content_markdown="C")
        assert self.mgr.delete(report.id) is True
        assert self.mgr.get(report.id) is None
        assert self.mgr.delete(report.id) is False

    def test_ttl_expiration(self):
        report = self.mgr.create(
            title="Ephemeral",
            content_markdown="Gone soon",
            report_type="general",
            ttl=0,  # expires immediately
        )
        # Expired report should not be returned
        assert self.mgr.get(report.id) is None

    def test_type_default_ttl(self):
        mgr = ReportManager(":memory:", default_ttl_overrides={"test_type": 86400})
        report = mgr.create(title="T", content_markdown="C", report_type="test_type")
        assert report.expires_at is not None
        assert report.expires_at > time.time()
        mgr.close()

    def test_no_ttl(self):
        report = self.mgr.create(
            title="Permanent", content_markdown="C", report_type="general", ttl=None
        )
        assert report.expires_at is None

    def test_purge_expired(self):
        self.mgr.create(title="A", content_markdown="a", ttl=0)
        self.mgr.create(title="B", content_markdown="b", ttl=None)

        purged = self.mgr.purge_expired()
        assert purged == 1

        remaining = self.mgr.list_reports()
        assert len(remaining) == 1
        assert remaining[0].title == "B"

    def test_metadata(self):
        report = self.mgr.create(
            title="T", content_markdown="C",
            metadata={"symbol": "TSLA", "timeframe": "1D"},
        )
        fetched = self.mgr.get(report.id)
        assert fetched.metadata["symbol"] == "TSLA"

    def test_chat_context(self):
        report = self.mgr.create(
            title="T", content_markdown="C",
            chat_context={
                "post_id": "abc123",
                "channel_id": "ch1",
                "permalink": "https://chat.example.com/team/pl/abc123",
            },
        )
        fetched = self.mgr.get(report.id)
        assert fetched.chat_context["permalink"] == "https://chat.example.com/team/pl/abc123"

    def test_get_stats(self):
        self.mgr.create(title="A", content_markdown="a", report_type="analysis")
        self.mgr.create(title="B", content_markdown="b", report_type="research")

        stats = self.mgr.get_stats()
        assert stats["total_reports"] == 2
        assert stats["by_type"]["analysis"] == 1
        assert stats["by_type"]["research"] == 1

    def test_to_dict(self):
        report = self.mgr.create(title="T", content_markdown="C")
        d = report.to_dict()
        assert d["title"] == "T"
        assert "id" in d
        assert "created_at" in d


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

class TestReport:
    def test_is_expired_false(self):
        r = Report(
            id="x", report_type="t", title="T",
            content_markdown="C", created_at=time.time(),
            expires_at=None,
        )
        assert r.is_expired is False

    def test_is_expired_true(self):
        r = Report(
            id="x", report_type="t", title="T",
            content_markdown="C", created_at=time.time() - 100,
            expires_at=time.time() - 1,
        )
        assert r.is_expired is True


# ---------------------------------------------------------------------------
# ReportDetector
# ---------------------------------------------------------------------------

class TestReportDetector:
    def setup_method(self):
        self.detector = ReportDetector()

    def test_empty_content(self):
        assert self.detector.is_report_worthy("") is False
        assert self.detector.is_report_worthy("   ") is False

    def test_short_content(self):
        assert self.detector.is_report_worthy("Short text.") is False

    def test_headers_trigger(self):
        content = "# Title\n\nSome text here.\n\n## Section\n\nMore text here that is long enough."
        content += " " * 300
        assert self.detector.is_report_worthy(content) is True

    def test_list_items_trigger(self):
        items = "\n".join(f"- Item {i}" for i in range(6))
        content = "A" * 300 + "\n" + items
        assert self.detector.is_report_worthy(content) is True

    def test_table_trigger(self):
        content = "A" * 300 + "\n| Col1 | Col2 |\n|------|------|\n| a | b |"
        assert self.detector.is_report_worthy(content) is True

    def test_keyword_trigger(self):
        content = (
            "This analysis provides a summary of findings. "
            "The evaluation shows good results."
        )
        content += " " * 300
        assert self.detector.is_report_worthy(content) is True

    def test_below_keyword_threshold(self):
        content = "This analysis is here." + " " * 300
        assert self.detector.is_report_worthy(content) is False

    def test_custom_criterion(self):
        self.detector.add_criterion(lambda t: "URGENT" in t)
        assert self.detector.is_report_worthy("URGENT: do something") is True

    def test_detect_type_analysis(self):
        content = "This analysis and assessment shows evaluation results."
        assert self.detector.detect_type(content) == "analysis"

    def test_detect_type_research(self):
        content = "Our research and investigation found new evidence and discovery."
        assert self.detector.detect_type(content) == "research"

    def test_detect_type_fallback(self):
        content = "Hello world, nothing special here."
        assert self.detector.detect_type(content) == "general"

    def test_custom_keywords(self):
        detector = ReportDetector(
            analysis_keywords=["genealogy", "lineage"],
            min_keywords=1,
        )
        content = "This genealogy trace found several branches." + " " * 300
        assert detector.is_report_worthy(content) is True

    def test_custom_type_rules(self):
        detector = ReportDetector(
            report_type_rules={
                "family_tree": ["ancestor", "descendant", "lineage"],
                "census": ["census", "household", "enumeration"],
            }
        )
        content = "Found an ancestor and descendant in this lineage."
        assert detector.detect_type(content) == "family_tree"


# ---------------------------------------------------------------------------
# ReportServer (Flask app)
# ---------------------------------------------------------------------------

class TestReportServer:
    def setup_method(self):
        self.mgr = ReportManager(":memory:")

        try:
            from khonliang.reporting.server import create_app
            self.app = create_app(self.mgr, base_url="")
            self.client = self.app.test_client()
            self.flask_available = True
        except ImportError:
            self.flask_available = False

    def teardown_method(self):
        self.mgr.close()

    @pytest.fixture(autouse=True)
    def skip_without_flask(self):
        if not self.flask_available:
            pytest.skip("Flask not installed")

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_report_list_empty(self):
        resp = self.client.get("/reports")
        assert resp.status_code == 200
        assert b"No reports yet" in resp.data

    def test_report_detail(self):
        report = self.mgr.create(
            title="Test", content_markdown="## Hello\nWorld", report_type="analysis"
        )
        resp = self.client.get(f"/reports/{report.id}")
        assert resp.status_code == 200
        assert b"Test" in resp.data
        assert b"analysis" in resp.data

    def test_report_not_found(self):
        resp = self.client.get("/reports/nonexistent")
        assert resp.status_code == 404

    def test_report_view_count_increments(self):
        report = self.mgr.create(title="T", content_markdown="C")
        self.client.get(f"/reports/{report.id}")
        self.client.get(f"/reports/{report.id}")
        fetched = self.mgr.get(report.id, track_view=False)
        assert fetched.view_count == 2  # server hits only, not the get above

    def test_api_list(self):
        self.mgr.create(title="A", content_markdown="a", report_type="x")
        resp = self.client.get("/api/reports")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1

    def test_api_detail(self):
        report = self.mgr.create(title="T", content_markdown="C")
        resp = self.client.get(f"/api/reports/{report.id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["title"] == "T"

    def test_api_delete(self):
        report = self.mgr.create(title="T", content_markdown="C")
        resp = self.client.delete(f"/api/reports/{report.id}")
        assert resp.status_code == 200
        assert resp.get_json()["deleted"] == report.id

        resp = self.client.get(f"/api/reports/{report.id}")
        assert resp.status_code == 404

    def test_report_list_json_format(self):
        self.mgr.create(title="A", content_markdown="a")
        resp = self.client.get("/reports?format=json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_chat_context_permalink(self):
        report = self.mgr.create(
            title="T", content_markdown="C",
            chat_context={"permalink": "https://chat.example.com/team/pl/abc"},
        )
        resp = self.client.get(f"/reports/{report.id}")
        assert b"View Original Conversation" in resp.data
        assert b"https://chat.example.com/team/pl/abc" in resp.data

    def test_custom_theme_applied(self):
        from khonliang.reporting.server import create_app

        theme = ReportTheme(
            name="Genealogy Tracker",
            primary_color="#16a34a",
            footer_text="Powered by khonliang",
        )
        app = create_app(self.mgr, base_url="", theme=theme)
        client = app.test_client()

        report = self.mgr.create(title="Family Tree", content_markdown="## Smiths\nFound 3.")
        resp = client.get(f"/reports/{report.id}")
        assert b"Genealogy Tracker" in resp.data
        assert b"#16a34a" in resp.data
        assert b"Powered by khonliang" in resp.data

    def test_custom_theme_on_list(self):
        from khonliang.reporting.server import create_app

        theme = ReportTheme(name="My Agents", logo_url="/static/logo.png")
        app = create_app(self.mgr, base_url="", theme=theme)
        client = app.test_client()

        resp = client.get("/reports")
        assert b"My Agents" in resp.data
        assert b"/static/logo.png" in resp.data


# ---------------------------------------------------------------------------
# ReportTheme
# ---------------------------------------------------------------------------

class TestReportTheme:
    def test_defaults(self):
        theme = ReportTheme()
        assert theme.primary_color == "#2563eb"
        assert theme.name == "Agent Reports"

    def test_from_dict(self):
        theme = ReportTheme.from_dict({
            "name": "Test",
            "primary_color": "#ff0000",
            "unknown_field": "ignored",
        })
        assert theme.name == "Test"
        assert theme.primary_color == "#ff0000"

    def test_to_dict_roundtrip(self):
        theme = ReportTheme(name="RT", footer_text="Footer")
        d = theme.to_dict()
        theme2 = ReportTheme.from_dict(d)
        assert theme2.name == "RT"
        assert theme2.footer_text == "Footer"

    def test_render_css_contains_variables(self):
        theme = ReportTheme(primary_color="#abcdef")
        css = theme.render_css()
        assert "#abcdef" in css
        assert "--primary:" in css

    def test_custom_css_injected(self):
        theme = ReportTheme(custom_css=".special { color: red; }")
        css = theme.render_css()
        assert ".special { color: red; }" in css

    def test_render_report_with_logo(self):
        from khonliang.reporting.templates import render_report

        theme = ReportTheme(name="Logo Test", logo_url="/img/logo.svg")
        report = Report(
            id="abc", report_type="test", title="T",
            content_markdown="Hello", created_at=time.time(),
        )
        html = render_report(report, theme)
        assert "/img/logo.svg" in html
        assert "Logo Test" in html


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

class TestSecurity:
    def test_script_stripped_from_markdown(self):
        from khonliang.reporting.templates import render_report

        theme = ReportTheme()
        report = Report(
            id="x", report_type="t", title="T",
            content_markdown='<script>alert(1)</script><p>safe</p>',
            created_at=time.time(),
        )
        html = render_report(report, theme)
        assert "<script>" not in html
        assert "alert(1)" not in html
        assert "safe" in html

    def test_event_handler_stripped(self):
        from khonliang.reporting.templates import render_report

        theme = ReportTheme()
        report = Report(
            id="x", report_type="t", title="T",
            content_markdown='<img src="x" onerror="alert(1)">',
            created_at=time.time(),
        )
        html = render_report(report, theme)
        assert "onerror" not in html

    def test_title_xss_escaped(self):
        from khonliang.reporting.templates import render_report

        theme = ReportTheme()
        report = Report(
            id="x", report_type="t", title='<script>alert("xss")</script>',
            content_markdown="safe", created_at=time.time(),
        )
        html = render_report(report, theme)
        assert "<script>" not in html.split('<div class="content">')[0]

    def test_custom_css_style_breakout_blocked(self):
        theme = ReportTheme(custom_css='</STYLE><script>alert(1)</script>')
        css = theme.render_css()
        assert "</STYLE>" not in css
        assert "</style>" not in css.lower().replace("</style>", "")  # only the closing tag

    def test_javascript_permalink_blocked(self):
        from khonliang.reporting.templates import render_report

        theme = ReportTheme()
        report = Report(
            id="x", report_type="t", title="T",
            content_markdown="C", created_at=time.time(),
            chat_context={"permalink": "javascript:alert(1)"},
        )
        html = render_report(report, theme)
        assert "javascript:" not in html
