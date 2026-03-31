"""
ReportDetector — determine if agent output warrants a persisted report.

Applies configurable heuristics to decide whether content is substantial
enough to save. Criteria are pluggable so each domain can tune detection
without subclassing.

Usage:
    detector = ReportDetector()

    # Check with defaults
    if detector.is_report_worthy(content):
        report_type = detector.detect_type(content)

    # Custom criteria
    detector = ReportDetector(
        min_length=500,
        min_headers=3,
        analysis_keywords=["forecast", "recommendation"],
        report_type_rules={
            "forecast": ["forecast", "prediction", "outlook"],
            "review": ["retrospective", "review", "summary"],
        },
    )

    # Register additional detection functions
    detector.add_criterion(lambda text: "URGENT" in text)
"""

import logging
import re
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default keywords that suggest analytical/report-worthy content
_DEFAULT_ANALYSIS_KEYWORDS = [
    "analysis",
    "summary",
    "findings",
    "conclusion",
    "recommendation",
    "result",
    "report",
    "overview",
    "assessment",
    "evaluation",
]

# Default type detection rules: type -> keywords
_DEFAULT_TYPE_RULES: Dict[str, List[str]] = {
    "analysis": ["analysis", "assessment", "evaluation", "findings"],
    "research": ["research", "investigation", "discovery", "evidence"],
    "summary": ["summary", "overview", "briefing", "digest"],
    "recommendation": ["recommendation", "suggestion", "proposal", "advise"],
}


class ReportDetector:
    """
    Detect whether content should be persisted as a report.

    Applies a set of heuristics in priority order:
    1. Custom criterion functions (if any returns True, content is report-worthy)
    2. Structural checks (headers, lists, tables)
    3. Length + keyword density

    Args:
        min_length: Minimum character count for content to be considered.
        min_keywords: Minimum number of analysis keywords that must appear.
        min_headers: If content has >= this many markdown headers, it qualifies.
        min_list_items: If content has >= this many list items, it qualifies.
        analysis_keywords: Keywords to look for. Defaults provided.
        report_type_rules: Dict mapping type names to keyword lists for
            detect_type(). Defaults provided.
    """

    def __init__(
        self,
        min_length: int = 300,
        min_keywords: int = 2,
        min_headers: int = 2,
        min_list_items: int = 5,
        analysis_keywords: Optional[List[str]] = None,
        report_type_rules: Optional[Dict[str, List[str]]] = None,
    ):
        self.min_length = min_length
        self.min_keywords = min_keywords
        self.min_headers = min_headers
        self.min_list_items = min_list_items
        self.analysis_keywords = analysis_keywords or list(_DEFAULT_ANALYSIS_KEYWORDS)
        self.type_rules = report_type_rules or dict(_DEFAULT_TYPE_RULES)
        self._custom_criteria: List[Callable[[str], bool]] = []

    def add_criterion(self, fn: Callable[[str], bool]) -> None:
        """
        Register a custom detection function.

        If any custom criterion returns True, the content is immediately
        considered report-worthy (short-circuits other checks).

        Args:
            fn: Callable that takes content string and returns bool.
        """
        self._custom_criteria.append(fn)

    def is_report_worthy(self, content: str) -> bool:
        """
        Determine whether content should be saved as a report.

        Args:
            content: The markdown content to evaluate.

        Returns:
            True if the content meets report criteria.
        """
        if not content or not content.strip():
            return False

        # Custom criteria (short-circuit)
        for criterion in self._custom_criteria:
            if criterion(content):
                return True

        # Too short — skip further checks
        if len(content) < self.min_length:
            return False

        # Structural: enough headers?
        header_count = len(re.findall(r"^#{1,6}\s+", content, re.MULTILINE))
        if header_count >= self.min_headers:
            return True

        # Structural: enough list items?
        list_count = len(re.findall(r"^[\s]*[-*+]\s+", content, re.MULTILINE))
        numbered_count = len(re.findall(r"^[\s]*\d+[.)]\s+", content, re.MULTILINE))
        if (list_count + numbered_count) >= self.min_list_items:
            return True

        # Structural: has a table?
        if re.search(r"^\|.+\|$", content, re.MULTILINE):
            return True

        # Keyword density
        content_lower = content.lower()
        keyword_hits = sum(1 for kw in self.analysis_keywords if kw in content_lower)
        if keyword_hits >= self.min_keywords:
            return True

        return False

    def detect_type(self, content: str) -> str:
        """
        Classify the report type based on keyword matching.

        Returns the type with the most keyword matches, or "general"
        if no rules match.

        Args:
            content: The markdown content to classify.

        Returns:
            Report type string.
        """
        content_lower = content.lower()

        best_type = "general"
        best_score = 0

        for report_type, keywords in self.type_rules.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_type = report_type

        return best_type
