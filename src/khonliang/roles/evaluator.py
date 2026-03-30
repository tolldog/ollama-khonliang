"""
Base evaluator — validates agent responses using role-declared rules.

Each role can declare evaluation rules that the evaluator applies.
This drives the autonomous improvement cycle: evaluate → research → improve.

Usage:
    # Define rules
    class DateRule(EvalRule):
        name = "date_check"
        def check(self, response, metadata):
            # validate dates in response...
            return []  # list of issues

    # Evaluator applies rules
    evaluator = BaseEvaluator(rules=[DateRule(), SpeculationRule()])
    result = evaluator.evaluate(response_text, metadata=resp_metadata)
    if result.caveat:
        response_text += result.caveat
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalIssue:
    """A single issue found during evaluation."""

    rule: str  # which rule found this
    issue_type: str  # date_mismatch, wrong_relationship, speculation, etc.
    detail: str
    severity: str = "medium"  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a response."""

    passed: bool
    confidence: float  # 0.0-1.0
    issues: List[EvalIssue] = field(default_factory=list)
    caveat: Optional[str] = None

    @property
    def high_issues(self) -> List[EvalIssue]:
        """Issues with severity 'high'."""
        return [i for i in self.issues if i.severity == "high"]

    @property
    def medium_issues(self) -> List[EvalIssue]:
        """Issues with severity 'medium'."""
        return [i for i in self.issues if i.severity == "medium"]


class EvalRule(ABC):
    """
    Abstract evaluation rule.

    Subclass to create domain-specific checks. Each rule examines
    the response and returns a list of issues found.
    """

    name: str = "base"

    @abstractmethod
    def check(
        self,
        response: str,
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvalIssue]:
        """
        Check the response for issues.

        Args:
            response: The LLM-generated text
            query: The original user query
            metadata: Response metadata (may include structured facts)

        Returns:
            List of EvalIssue objects (empty if no issues)
        """
        ...


class SpeculationRule(EvalRule):
    """Detects excessive speculation in responses."""

    name = "speculation"

    def __init__(self, max_phrases: int = 3):
        self.max_phrases = max_phrases
        self.speculation_phrases = [
            "it is possible",
            "it's possible",
            "speculation",
            "may have",
            "might have",
            "could have been",
            "likely",
            "perhaps",
            "probably",
        ]

    def check(self, response, query="", metadata=None):
        """Flag responses with too many speculative phrases."""
        resp_lower = response.lower()
        count = sum(1 for p in self.speculation_phrases if p in resp_lower)
        if count >= self.max_phrases:
            return [EvalIssue(
                rule=self.name,
                issue_type="excessive_speculation",
                detail=f"Response contains {count} speculative phrases",
                severity="medium",
            )]
        return []


class UncertaintyRule(EvalRule):
    """Detects when the agent expresses uncertainty."""

    name = "uncertainty"

    def __init__(self):
        self.phrases = [
            "i'm not sure",
            "i don't have",
            "no information",
            "no data",
            "cannot find",
            "could not find",
            "no records",
        ]

    def check(self, response, query="", metadata=None):
        """Flag responses where the agent expresses uncertainty."""
        resp_lower = response.lower()
        for phrase in self.phrases:
            if phrase in resp_lower:
                detail = f"Agent expressed uncertainty: '{phrase}'"
                if query:
                    detail += f" (query: {query[:40]})"
                return [EvalIssue(
                    rule=self.name,
                    issue_type="uncertainty",
                    detail=detail,
                    severity="low",
                )]
        return []


class BaseEvaluator:
    """
    Evaluates agent responses using configurable rules.

    Rules are checked in order. Results aggregated into a single
    EvalResult with confidence score and optional caveat text.

    Args:
        rules: List of EvalRule instances to apply
        skip_roles: Roles to skip evaluation for (e.g. "research", "system")
    """

    def __init__(
        self,
        rules: Optional[List[EvalRule]] = None,
        skip_roles: Optional[set] = None,
    ):
        self.rules = rules or [SpeculationRule(), UncertaintyRule()]
        self.skip_roles = skip_roles or {"research", "system", "analyst", "librarian"}

    def evaluate(
        self,
        response: str,
        query: str = "",
        role: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Evaluate a response against all rules."""
        if role in self.skip_roles:
            return EvalResult(passed=True, confidence=0.95)

        all_issues: List[EvalIssue] = []
        for rule in self.rules:
            try:
                issues = rule.check(response, query=query, metadata=metadata)
                all_issues.extend(issues)
            except Exception as e:
                logger.debug(f"Rule {rule.name} failed: {e}")

        # Score
        high = [i for i in all_issues if i.severity == "high"]
        med = [i for i in all_issues if i.severity == "medium"]

        if high:
            confidence = 0.3
        elif med:
            confidence = 0.6
        elif all_issues:
            confidence = 0.8
        else:
            confidence = 0.95

        # Build caveat
        caveat = None
        if high:
            details = "; ".join(i.detail for i in high[:3])
            caveat = f"Verification issues: {details}"
        elif med:
            details = "; ".join(i.detail for i in med[:2])
            caveat = f"Note: {details}"

        return EvalResult(
            passed=len(high) == 0,
            confidence=confidence,
            issues=all_issues,
            caveat=caveat,
        )
