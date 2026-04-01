"""
ReportServer — Flask application serving persisted reports as styled HTML.

Provides endpoints for viewing individual reports, listing recent reports,
and a JSON API for programmatic access. Supports customizable themes via
ReportTheme for branding (logo, colors, fonts, footer).

Routes:
    GET  /reports              — HTML list of recent reports
    GET  /reports/<id>         — Rendered HTML report page
    GET  /api/reports          — JSON list of reports
    GET  /api/reports/<id>     — JSON report detail
    GET  /health               — Server health check

Usage:
    from khonliang.reporting import ReportManager
    from khonliang.reporting.server import create_app
    from khonliang.reporting.templates import ReportTheme

    manager = ReportManager("reports.db")
    theme = ReportTheme(name="My Project", logo_url="/static/logo.png")
    app = create_app(manager, base_url="http://localhost:5050", theme=theme)
    app.run(port=5050)

Or with the built-in runner:
    server = ReportServer(manager, port=5050, theme=theme)
    server.run()
"""

import logging
from typing import Optional

from khonliang.reporting.manager import ReportManager
from khonliang.reporting.templates import ReportTheme, render_report, render_report_list

logger = logging.getLogger(__name__)


def create_app(
    manager: ReportManager,
    base_url: str = "",
    theme: Optional[ReportTheme] = None,
    static_dir: Optional[str] = None,
) -> "Flask":  # noqa: F821
    """
    Create a Flask application for serving reports.

    Args:
        manager: The ReportManager instance to read reports from.
        base_url: Base URL prefix for generated links (e.g. "http://localhost:5050").
        theme: ReportTheme for customizing HTML output. Uses defaults if omitted.
        static_dir: Directory path for serving static files (logos, CSS, images)
            at /static/. Theme logo_url can reference these as "/static/logo.png".

    Returns:
        A Flask application instance.
    """
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError:
        raise ImportError(
            "Flask is required for ReportServer. "
            'Install with: pip install ollama-khonliang[reporting]'
        )

    if theme is None:
        theme = ReportTheme()

    app = Flask(__name__)
    if static_dir:
        app.config["REPORT_STATIC_DIR"] = static_dir

    def _parse_limit(default: int = 50, maximum: int = 500) -> int:
        """Parse and clamp the 'limit' query parameter."""
        raw = request.args.get("limit")
        if raw is None or raw == "":
            return default
        try:
            return min(max(int(raw), 1), maximum)
        except ValueError:
            return default

    @app.route("/static/<path:filename>")
    def static_file(filename):
        """Serve static files (logos, CSS, etc.) from the configured directory."""
        from flask import send_from_directory

        static_dir = app.config.get("REPORT_STATIC_DIR")
        if static_dir:
            return send_from_directory(static_dir, filename)
        return Response("Not found", status=404)

    @app.route("/health")
    def health():
        stats = manager.get_stats()
        return jsonify({"status": "ok", "reports": stats})

    @app.route("/reports")
    def report_list():
        report_type = request.args.get("type")
        created_by = request.args.get("by")
        limit = _parse_limit()

        reports = manager.list_reports(
            report_type=report_type, created_by=created_by, limit=limit
        )

        if request.args.get("format") == "json":
            return jsonify([r.to_dict() for r in reports])

        html = render_report_list(reports, theme, base_url)
        return Response(html, content_type="text/html")

    @app.route("/reports/<report_id>")
    def report_detail(report_id):
        report = manager.get(report_id, track_view=True)
        if report is None:
            return Response("<h1>Report not found</h1>", status=404, content_type="text/html")

        if request.args.get("format") == "json":
            return jsonify(report.to_dict())

        html = render_report(report, theme, base_url)
        return Response(html, content_type="text/html")

    @app.route("/api/reports")
    def api_report_list():
        report_type = request.args.get("type")
        created_by = request.args.get("by")
        limit = _parse_limit()

        reports = manager.list_reports(
            report_type=report_type, created_by=created_by, limit=limit
        )
        return jsonify([r.to_dict() for r in reports])

    @app.route("/api/reports/<report_id>")
    def api_report_detail(report_id):
        report = manager.get(report_id, track_view=False)
        if report is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(report.to_dict())

    @app.route("/api/reports/<report_id>", methods=["DELETE"])
    def api_report_delete(report_id):
        deleted = manager.delete(report_id)
        if not deleted:
            return jsonify({"error": "not found"}), 404
        return jsonify({"deleted": report_id})

    return app


class ReportServer:
    """
    Convenience wrapper to run the report Flask app.

    Args:
        manager: ReportManager instance.
        host: Bind address.
        port: Bind port.
        base_url: External URL for links. Auto-generated from host/port if omitted.
        theme: ReportTheme for branding. Uses defaults if omitted.
    """

    def __init__(
        self,
        manager: ReportManager,
        host: str = "0.0.0.0",
        port: int = 5050,
        base_url: Optional[str] = None,
        theme: Optional[ReportTheme] = None,
        static_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.base_url = base_url or f"http://localhost:{port}"
        self.theme = theme or ReportTheme()
        self.app = create_app(manager, self.base_url, self.theme, static_dir)
        self.manager = manager

    def report_url(self, report_id: str) -> str:
        """Get the full URL for a report."""
        return f"{self.base_url}/reports/{report_id}"

    def run(self, debug: bool = False) -> None:
        """Start the Flask server."""
        logger.info(f"Report server starting on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
