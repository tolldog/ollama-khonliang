"""
Genealogy report server example.

Demonstrates the reporting module with a genealogy-themed setup:
- Custom ReportTheme with the khonliang genealogy logo
- ReportDetector tuned for genealogy keywords
- Sample reports showing what agent-generated genealogy research looks like
- Flask server serving styled HTML reports

Run:
    python examples/genealogy_reports/main.py

Then open http://localhost:5050/reports in your browser.
"""

import os
from pathlib import Path

from khonliang.reporting import ReportDetector, ReportManager, ReportTheme
from khonliang.reporting.server import ReportServer

# Resolve paths relative to this example
EXAMPLE_DIR = Path(__file__).parent
PROJECT_ROOT = EXAMPLE_DIR.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def create_theme() -> ReportTheme:
    """Build the genealogy-branded theme."""
    return ReportTheme(
        name="Khonliang Genealogy",
        logo_url="/static/khonliang_genealogy.png",
        logo_height="48px",
        primary_color="#6d28d9",
        secondary_color="#5b21b6",
        footer_text="Khonliang Genealogy Research",
    )


def create_detector() -> ReportDetector:
    """Build a genealogy-tuned report detector."""
    detector = ReportDetector(
        analysis_keywords=[
            "ancestor", "descendant", "lineage", "genealogy",
            "census", "birth", "death", "marriage", "immigration",
            "family", "generation", "pedigree", "surname", "record",
        ],
        min_keywords=2,
        report_type_rules={
            "family_tree": ["ancestor", "descendant", "lineage", "pedigree", "generation"],
            "census_record": ["census", "household", "enumeration", "dwelling", "occupation"],
            "vital_record": ["birth", "death", "marriage", "baptism", "burial"],
            "immigration": ["immigration", "naturalization", "passenger", "ship", "port"],
            "research_summary": ["research", "findings", "summary", "evidence", "conclusion"],
        },
    )
    return detector


def seed_sample_reports(manager: ReportManager, detector: ReportDetector) -> None:
    """Create sample reports to demonstrate the system."""

    # --- Family Tree Report ---
    family_tree_md = """\
# Smith Family Lineage — 4 Generations

## Overview

Research into the Smith family of County Cork, Ireland has uncovered
four generations of descendants spanning 1845-1940. The lineage traces
from Patrick Smith (b. ~1820) through his grandson William Smith who
immigrated to Boston in 1892.

## Generation 1: Patrick Smith (c. 1820-1878)

- **Born:** ~1820, County Cork, Ireland
- **Married:** Mary O'Brien, 1843
- **Died:** 1878, County Cork
- **Children:** 3 (Thomas, Bridget, James)
- **Source:** Parish records, St. Finbarr's Cathedral

## Generation 2: Thomas Smith (1845-1912)

- **Born:** 1845, County Cork
- **Married:** Catherine Walsh, 1870
- **Died:** 1912, County Cork
- **Children:** 5 (William, Patrick Jr., Ellen, Margaret, John)
- **Occupation:** Farmer (per 1901 Census)

## Generation 3: William Smith (1872-1940)

- **Born:** 1872, County Cork
- **Immigrated:** 1892, arrived Boston via SS Nevada
- **Married:** Anna Kowalski, 1898, Boston MA
- **Died:** 1940, Boston MA
- **Children:** 4 (Thomas Jr., Mary, Francis, Helen)
- **Occupation:** Dockworker (1900 Census), Foreman (1920 Census)

## Generation 4: Thomas Smith Jr. (1900-1968)

- **Born:** 1900, Boston MA
- **Married:** Rose DiMaggio, 1925
- **Children:** 3 (William Jr., Patricia, Robert)

## Pending Research

- Bridget Smith (Gen 2) — marriage record not found
- James Smith (Gen 2) — emigrated? No records after 1880
- Anna Kowalski's family — Polish origins, need to check
  Galicia parish records
- Patrick Jr. (Gen 3) — believed to have stayed in Ireland
"""

    report_type = detector.detect_type(family_tree_md)
    manager.create(
        title="Smith Family Lineage — 4 Generations",
        content_markdown=family_tree_md,
        report_type=report_type,
        created_by="genealogy_agent",
        metadata={"surname": "Smith", "origin": "County Cork, Ireland"},
    )

    # --- Census Record ---
    census_md = """\
# 1900 US Census — Smith Household

## Household Details

| Field | Value |
|-------|-------|
| **Enumeration District** | 1547 |
| **Sheet** | 12A |
| **Location** | 47 Fleet St, Boston, Suffolk Co., MA |
| **Date** | June 5, 1900 |

## Household Members

| Name | Relation | Age | Birthplace | Occupation | Immigration |
|------|----------|-----|------------|------------|-------------|
| William Smith | Head | 28 | Ireland | Dockworker | 1892 |
| Anna Smith | Wife | 24 | Poland | — | 1895 |
| Thomas Smith | Son | 1 | Massachusetts | — | — |

## Notes

- William listed as "naturalized" — need to find naturalization papers
- Anna's birthplace listed as "Poland" (likely Galicia/Austria-Hungary)
- Dwelling is a rented tenement, monthly rent $8
- Both William and Anna listed as literate
"""

    report_type = detector.detect_type(census_md)
    manager.create(
        title="1900 US Census — Smith Household, Boston",
        content_markdown=census_md,
        report_type=report_type,
        created_by="census_agent",
        metadata={"surname": "Smith", "year": 1900, "location": "Boston, MA"},
    )

    # --- Research Summary ---
    research_md = """\
# Research Summary — Smith-Kowalski Connection

## Findings

Investigation into the Smith-Kowalski marriage (1898, Boston) has yielded
several new leads connecting the families to broader immigrant communities.

### Evidence Gathered

1. **Marriage Certificate** (Boston City Hall, Book 412, Page 89)
   - Confirms William Smith (age 26, Ireland) married Anna Kowalski (age 22, Austria)
   - Witnesses: Jan Kowalski (brother) and Patrick Murphy

2. **SS Nevada Passenger Manifest** (1892)
   - William Smith, age 20, laborer, from Queenstown
   - Destination: Boston, contact "cousin M. Murphy, Fleet St"
   - Confidence: 85% match (common name, but age and details align)

3. **1895 Immigration Index**
   - Anna Kowalski, age 19, from Bremen, Germany (transit port)
   - Final destination: Boston
   - Confidence: 70% (need to cross-reference with Polish parish records)

### Gaps and Next Steps

- Locate Jan Kowalski in Boston directories (1895-1910)
- Check St. Adalbert's Parish (Polish church) for baptism records
- Search for Patrick Murphy connection — likely the "cousin" from
  William's passenger manifest
- Cork parish records for Smith baptism (~1872)

### Conclusion

The evidence strongly supports the Smith-Kowalski family connection
as documented. The immigration timeline is consistent, and the witness
records provide additional research avenues into both families'
social networks in Boston's immigrant community.
"""

    report_type = detector.detect_type(research_md)
    manager.create(
        title="Research Summary — Smith-Kowalski Connection",
        content_markdown=research_md,
        report_type=report_type,
        created_by="research_agent",
        metadata={"surnames": ["Smith", "Kowalski"], "topic": "immigration"},
    )

    print(f"Seeded {len(manager.list_reports())} sample reports.")


def main():
    db_path = os.environ.get("REPORT_DB", str(EXAMPLE_DIR / "reports.db"))
    port = int(os.environ.get("REPORT_PORT", "5050"))

    manager = ReportManager(db_path)
    detector = create_detector()
    theme = create_theme()

    # Seed sample data if database is empty
    if not manager.list_reports():
        seed_sample_reports(manager, detector)

    server = ReportServer(
        manager,
        port=port,
        theme=theme,
        static_dir=str(ASSETS_DIR),
    )

    print(f"Genealogy Report Server running at http://localhost:{port}/reports")
    server.run()


if __name__ == "__main__":
    main()
