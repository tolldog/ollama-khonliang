"""
reporting — Persist, serve, and link agent-generated reports.

Provides a pipeline: detect report-worthy content → persist to SQLite →
serve as styled HTML via Flask → post links to chat integrations.

Quick start:

    from khonliang.reporting import ReportManager, ReportDetector

    manager = ReportManager("reports.db")
    detector = ReportDetector()

    if detector.is_report_worthy(content):
        report = manager.create(
            title="Analysis Results",
            content_markdown=content,
            report_type=detector.detect_type(content),
            created_by="researcher",
        )
        # report.url available when ReportServer is running
"""

from khonliang.reporting.detector import ReportDetector
from khonliang.reporting.manager import Report, ReportManager
from khonliang.reporting.templates import ReportTheme

__all__ = [
    "Report",
    "ReportManager",
    "ReportDetector",
    "ReportTheme",
]
