"""
Report templates — customizable HTML/CSS theming for report output.

Provides a ReportTheme dataclass for branding (logo, colors, fonts, footer)
and renders reports using the theme. Users configure a theme once and pass
it to create_app() or ReportServer.

Usage:
    theme = ReportTheme(
        name="My Project",
        logo_url="/static/logo.png",
        primary_color="#16a34a",
        footer_text="Powered by My Agent System",
    )

    # Or load from a dict/JSON config
    theme = ReportTheme.from_dict({
        "name": "My Project",
        "logo_url": "https://example.com/logo.png",
        "primary_color": "#16a34a",
        "custom_css": "body { font-family: 'Fira Code', monospace; }",
    })

    app = create_app(manager, theme=theme)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from khonliang.reporting.manager import Report


@dataclass
class ReportTheme:
    """
    Customizable theme for rendered report HTML.

    All fields are optional — defaults produce a clean, professional look.

    Args:
        name: Project/organization name shown in headers.
        logo_url: URL or path to a logo image. Shown in report headers.
        logo_height: CSS height for the logo (e.g. "40px", "2rem").
        primary_color: Primary accent color (hex). Used for headers, links, badges.
        secondary_color: Secondary color for hover states and borders.
        background_color: Page background color.
        card_background: Card/container background color.
        text_color: Primary text color.
        font_family: CSS font-family string.
        max_width: Max content width (e.g. "900px", "60rem").
        footer_text: Custom footer text shown at bottom of every report.
        custom_css: Additional CSS injected after the base styles. Use this
            for any overrides or additions not covered by the fields above.
    """

    name: str = "Agent Reports"
    logo_url: Optional[str] = None
    logo_height: str = "40px"
    primary_color: str = "#2563eb"
    secondary_color: str = "#1d4ed8"
    background_color: str = "#f5f5f5"
    card_background: str = "#ffffff"
    text_color: str = "#333333"
    font_family: str = (
        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", Arial, sans-serif'
    )
    max_width: str = "900px"
    footer_text: str = ""
    custom_css: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "logo_url": self.logo_url,
            "logo_height": self.logo_height,
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "background_color": self.background_color,
            "card_background": self.card_background,
            "text_color": self.text_color,
            "font_family": self.font_family,
            "max_width": self.max_width,
            "footer_text": self.footer_text,
            "custom_css": self.custom_css,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportTheme":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "ReportTheme":
        """Load theme from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def render_css(self) -> str:
        """Generate the complete CSS block for this theme."""
        return f"""<style>
  :root {{
    --primary: {self.primary_color};
    --primary-hover: {self.secondary_color};
    --bg: {self.background_color};
    --card-bg: {self.card_background};
    --text: {self.text_color};
    --max-width: {self.max_width};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: {self.font_family};
    line-height: 1.6; color: var(--text); background: var(--bg);
    padding: 2rem;
  }}
  .container {{ max-width: var(--max-width); margin: 0 auto; }}
  .card {{
    background: var(--card-bg); border-radius: 8px; padding: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1.5rem;
  }}
  .header {{ border-bottom: 2px solid var(--primary); padding-bottom: 1rem; margin-bottom: 1.5rem; }}
  .header h1 {{ color: var(--primary); font-size: 1.5rem; }}
  .header-brand {{
    display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;
  }}
  .header-brand img {{ height: {self.logo_height}; }}
  .header-brand .brand-name {{
    font-size: 0.85rem; color: #666; font-weight: 500;
  }}
  .meta {{ color: #666; font-size: 0.85rem; margin-top: 0.5rem; }}
  .meta span {{ margin-right: 1.5rem; }}
  .badge {{
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    background: color-mix(in srgb, var(--primary) 15%, white);
    color: var(--primary);
  }}
  .content {{ margin-top: 1rem; }}
  .content h1, .content h2, .content h3 {{ color: var(--primary); margin: 1.2rem 0 0.5rem; }}
  .content h1 {{ font-size: 1.4rem; }}
  .content h2 {{ font-size: 1.2rem; }}
  .content h3 {{ font-size: 1.05rem; }}
  .content p {{ margin: 0.5rem 0; }}
  .content ul, .content ol {{ margin: 0.5rem 0 0.5rem 1.5rem; }}
  .content table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  .content th, .content td {{
    border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left;
  }}
  .content th {{ background: color-mix(in srgb, var(--primary) 8%, white); font-weight: 600; }}
  .content code {{
    background: #f0f0f0; padding: 0.15rem 0.4rem; border-radius: 3px;
    font-size: 0.9em;
  }}
  .content pre {{ background: #f8f8f8; padding: 1rem; border-radius: 6px;
    overflow-x: auto; margin: 0.75rem 0; }}
  .content pre code {{ background: none; padding: 0; }}
  .chat-link {{
    display: inline-block; margin-top: 1rem; padding: 0.5rem 1rem;
    background: var(--primary); color: #fff; text-decoration: none;
    border-radius: 6px; font-size: 0.85rem;
  }}
  .chat-link:hover {{ background: var(--primary-hover); }}
  .report-list a {{ text-decoration: none; color: inherit; display: block; }}
  .report-list .card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
  .footer {{
    text-align: center; color: #999; font-size: 0.75rem; margin-top: 2rem;
  }}
  .nav {{ margin-bottom: 1.5rem; }}
  .nav a {{ color: var(--primary); text-decoration: none; font-size: 0.9rem; }}
  {self.custom_css}
</style>"""


def _format_time(ts: Optional[float]) -> str:
    if ts is None:
        return "\u2014"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _render_markdown(md: str) -> str:
    """
    Convert markdown to HTML.

    Uses the `markdown` library if available (with tables and fenced_code),
    otherwise a simple regex fallback.
    """
    try:
        import markdown

        return markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        pass

    import re

    html = md
    html = re.sub(
        r"```(\w*)\n(.*?)```", r"<pre><code>\2</code></pre>",
        html, flags=re.DOTALL,
    )
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    html = re.sub(r"^[\s]*[-*+]\s+(.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
    lines = html.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("<"):
            result.append(f"<p>{stripped}</p>")
        else:
            result.append(line)
    return "\n".join(result)


def render_report(report: Report, theme: ReportTheme, base_url: str = "") -> str:
    """Render a single report as a full HTML page."""
    content_html = _render_markdown(report.content_markdown)

    chat_link = ""
    permalink = report.chat_context.get("permalink")
    if permalink:
        chat_link = (
            f'<a class="chat-link" href="{permalink}">'
            "View Original Conversation</a>"
        )

    logo_html = ""
    if theme.logo_url:
        logo_html = f'<img src="{theme.logo_url}" alt="{theme.name}">'

    brand_html = ""
    if theme.logo_url or theme.name != "Agent Reports":
        brand_html = f"""
    <div class="header-brand">
      {logo_html}
      <span class="brand-name">{theme.name}</span>
    </div>"""

    footer_extra = f"<br>{theme.footer_text}" if theme.footer_text else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{report.title} — {theme.name}</title>
  {theme.render_css()}
</head>
<body>
  <div class="container">
    <div class="nav"><a href="{base_url}/reports">&larr; All Reports</a></div>
    <div class="card">
      <div class="header">
        {brand_html}
        <h1>{report.title}</h1>
        <div class="meta">
          <span class="badge">{report.report_type}</span>
          <span>Created: {_format_time(report.created_at)}</span>
          <span>By: {report.created_by or "unknown"}</span>
          <span>Views: {report.view_count}</span>
        </div>
        {chat_link}
      </div>
      <div class="content">{content_html}</div>
    </div>
    <div class="footer">
      Report ID: {report.id}
      {' | Last viewed: ' + _format_time(report.last_viewed_at) if report.last_viewed_at else ''}
      {footer_extra}
    </div>
  </div>
</body>
</html>"""


def render_report_list(reports, theme: ReportTheme, base_url: str = "") -> str:
    """Render a list of reports as an HTML page."""
    items = []
    for r in reports:
        items.append(f"""
    <a href="{base_url}/reports/{r.id}">
      <div class="card">
        <span class="badge">{r.report_type}</span>
        <strong>{r.title}</strong>
        <div class="meta">
          <span>{_format_time(r.created_at)}</span>
          <span>By: {r.created_by or "unknown"}</span>
          <span>Views: {r.view_count}</span>
        </div>
      </div>
    </a>""")

    report_items = "\n".join(items) if items else "<p>No reports yet.</p>"

    logo_html = ""
    if theme.logo_url:
        logo_html = f'<img src="{theme.logo_url}" alt="{theme.name}" style="height: {theme.logo_height}; margin-right: 0.75rem;">'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{theme.name}</title>
  {theme.render_css()}
</head>
<body>
  <div class="container">
    <div class="card header">
      <div style="display: flex; align-items: center;">
        {logo_html}
        <h1>{theme.name}</h1>
      </div>
    </div>
    <div class="report-list">
      {report_items}
    </div>
  </div>
</body>
</html>"""
