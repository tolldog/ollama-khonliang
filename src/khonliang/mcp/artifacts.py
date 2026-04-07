"""Compressed artifact types for MCP tool outputs.

These dataclasses define the canonical shapes that local-model distillation
should produce. Each has ``to_compact()`` (pipe-delimited) and ``to_brief()``
(structured one-liner) for direct use in MCP responses.

Usage:

    concept = CompactConcept.from_dict(raw)
    return concept.to_compact()   # "rl|0.82|5|Policy Gradient Methods|yes"
    return concept.to_brief()     # "rl (82%) — 5 papers, top: Policy Gradient Methods [actionable]"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompactConcept:
    """A concept scored for project relevance."""

    name: str
    relevance: float  # 0-1
    paper_count: int
    top_paper: str
    actionable: bool = False

    def to_compact(self) -> str:
        act = "yes" if self.actionable else "no"
        return (
            f"{self.name}|{self.relevance:.2f}|{self.paper_count}"
            f"|{self.top_paper}|{act}"
        )

    def to_brief(self) -> str:
        tag = " [actionable]" if self.actionable else ""
        return (
            f"{self.name} ({self.relevance:.0%})"
            f" — {self.paper_count} papers, top: {self.top_paper}{tag}"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactConcept:
        return cls(
            name=str(data.get("name", "")),
            relevance=float(data.get("relevance", 0.0)),
            paper_count=int(data.get("paper_count", 0)),
            top_paper=str(data.get("top_paper", "")),
            actionable=bool(data.get("actionable", False)),
        )


@dataclass
class CompactFR:
    """A feature request in compressed form."""

    id: str
    title: str
    priority: str
    target: str
    concept: str = ""
    depends_on: list[str] = field(default_factory=list)

    def to_compact(self) -> str:
        deps = ",".join(self.depends_on) if self.depends_on else "none"
        return f"{self.id}|{self.priority}|{self.target}|{self.title}|deps={deps}"

    def to_brief(self) -> str:
        dep_str = f" (blocks: {', '.join(self.depends_on)})" if self.depends_on else ""
        concept_str = f" [{self.concept}]" if self.concept else ""
        return f"{self.id} [{self.priority}] {self.title} -> {self.target}{concept_str}{dep_str}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactFR:
        deps = data.get("depends_on", [])
        if isinstance(deps, str):
            deps = [d.strip() for d in deps.split(",") if d.strip()]
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            priority=str(data.get("priority", "medium")),
            target=str(data.get("target", "")),
            concept=str(data.get("concept", "")),
            depends_on=list(deps),
        )


@dataclass
class CompactSynthesis:
    """A topic synthesis compressed for agent consumption."""

    topic: str
    paper_count: int
    key_findings: list[str] = field(default_factory=list)  # max 5
    relevance: dict[str, float] = field(default_factory=dict)  # project -> score
    suggested_frs: list[str] = field(default_factory=list)

    def to_compact(self) -> str:
        findings = "; ".join(self.key_findings[:5])
        rel = ",".join(f"{k}:{v:.2f}" for k, v in self.relevance.items())
        frs = ",".join(self.suggested_frs) if self.suggested_frs else "none"
        return f"{self.topic}|{self.paper_count}|{findings}|rel={rel}|frs={frs}"

    def to_brief(self) -> str:
        lines = [f"{self.topic} ({self.paper_count} papers)"]
        for finding in self.key_findings[:5]:
            lines.append(f"  - {finding}")
        if self.relevance:
            rel_parts = [f"{k}: {v:.0%}" for k, v in self.relevance.items()]
            lines.append(f"  Relevance: {', '.join(rel_parts)}")
        if self.suggested_frs:
            lines.append(f"  FRs: {', '.join(self.suggested_frs)}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactSynthesis:
        return cls(
            topic=str(data.get("topic", "")),
            paper_count=int(data.get("paper_count", 0)),
            key_findings=list(data.get("key_findings", []))[:5],
            relevance=dict(data.get("relevance", {})),
            suggested_frs=list(data.get("suggested_frs", [])),
        )
