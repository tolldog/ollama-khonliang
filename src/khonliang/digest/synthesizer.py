"""
DigestSynthesizer — generate narrative digests from accumulated entries.

Pulls entries from DigestStore, groups and formats them, then uses an
LLM (via LLMClient) to synthesize a readable narrative. Falls back to
a structured summary when no LLM is available.

The synthesis prompt is configurable per application via DigestConfig.

Usage:
    config = DigestConfig(
        system_prompt="You are a genealogy research assistant.",
        synthesis_prompt="Summarize these activities into a brief digest.",
    )
    synthesizer = DigestSynthesizer(store, config, client=ollama_client)

    # Generate digest from unconsumed entries
    result = await synthesizer.generate()
    print(result.markdown)

    # Generate from a time window
    result = await synthesizer.generate(hours=24)
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from khonliang.digest.store import DigestEntry, DigestStore

logger = logging.getLogger(__name__)


@dataclass
class DigestConfig:
    """
    Configuration for digest synthesis.

    Args:
        system_prompt: System prompt for the LLM providing context about
            the application domain.
        synthesis_prompt: Instruction prompt telling the LLM how to
            synthesize entries into a digest. The formatted entry list
            is appended after this prompt.
        title_template: Template for the digest title. Supports {date},
            {count}, {hours} placeholders.
        group_by_source: Whether to group entries by source in the
            formatted input.
        max_entries: Maximum entries to include in a single digest.
        include_metadata: Whether to include entry metadata in the
            LLM context.
    """

    system_prompt: str = "You are an assistant summarizing agent activity."
    synthesis_prompt: str = (
        "Summarize the following agent activities into a concise digest. "
        "Highlight key findings, completed work, and pending items. "
        "Be specific about what was found or changed."
    )
    title_template: str = "Activity Digest — {date}"
    group_by_source: bool = True
    max_entries: int = 100
    include_metadata: bool = False


@dataclass
class DigestResult:
    """Output of a digest synthesis."""

    id: str
    markdown: str
    title: str
    entry_count: int
    entry_ids: List[str]
    created_at: float
    synthesized: bool  # True if LLM was used, False if fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "markdown": self.markdown,
            "title": self.title,
            "entry_count": self.entry_count,
            "created_at": self.created_at,
            "synthesized": self.synthesized,
        }


def format_entries(
    entries: List[DigestEntry],
    group_by_source: bool = True,
    include_metadata: bool = False,
) -> str:
    """
    Format digest entries into a text block for LLM consumption.

    Args:
        entries: The entries to format.
        group_by_source: Group entries under source headings.
        include_metadata: Include metadata key-values.

    Returns:
        Formatted text string.
    """
    if not entries:
        return "No activity recorded."

    if group_by_source:
        groups: Dict[str, List[DigestEntry]] = {}
        for e in entries:
            key = e.source or "system"
            groups.setdefault(key, []).append(e)

        lines = []
        for source, group in sorted(groups.items()):
            lines.append(f"[{source}]")
            for e in group:
                ts = time.strftime("%H:%M", time.localtime(e.created_at))
                tag_str = f" ({', '.join(e.tags)})" if e.tags else ""
                lines.append(f"  {ts} — {e.summary}{tag_str}")
                if include_metadata and e.metadata:
                    for k, v in e.metadata.items():
                        lines.append(f"    {k}: {v}")
            lines.append("")
        return "\n".join(lines)

    lines = []
    for e in entries:
        ts = time.strftime("%H:%M", time.localtime(e.created_at))
        source = f"[{e.source}] " if e.source else ""
        tag_str = f" ({', '.join(e.tags)})" if e.tags else ""
        lines.append(f"{ts} — {source}{e.summary}{tag_str}")
    return "\n".join(lines)


def _structured_fallback(entries: List[DigestEntry], title: str) -> str:
    """
    Generate a structured digest without LLM synthesis.

    Used when no LLM client is available or synthesis fails.
    """
    sections = [f"# {title}", ""]

    # Stats
    sources = {}
    tags_count: Dict[str, int] = {}
    for e in entries:
        src = e.source or "system"
        sources[src] = sources.get(src, 0) + 1
        for t in e.tags:
            tags_count[t] = tags_count.get(t, 0) + 1

    sections.append(f"**{len(entries)} activities** from {len(sources)} sources")
    sections.append("")

    # By source
    for source, group_entries in sorted(
        _group_entries(entries).items(), key=lambda x: -len(x[1])
    ):
        sections.append(f"## {source} ({len(group_entries)})")
        for e in group_entries:
            ts = time.strftime("%H:%M", time.localtime(e.created_at))
            sections.append(f"- {ts} — {e.summary}")
        sections.append("")

    # Tag summary
    if tags_count:
        sections.append("## Tags")
        for tag, count in sorted(tags_count.items(), key=lambda x: -x[1]):
            sections.append(f"- {tag}: {count}")
        sections.append("")

    return "\n".join(sections)


def _group_entries(entries: List[DigestEntry]) -> Dict[str, List[DigestEntry]]:
    groups: Dict[str, List[DigestEntry]] = {}
    for e in entries:
        key = e.source or "system"
        groups.setdefault(key, []).append(e)
    return groups


class DigestSynthesizer:
    """
    Generate narrative digests from accumulated entries.

    Uses an LLMClient for synthesis when available, falls back to
    structured formatting otherwise.

    Args:
        store: DigestStore to read entries from.
        config: DigestConfig controlling prompts and formatting.
        client: Optional LLMClient for narrative synthesis.
        model: Optional model override for generation.
    """

    def __init__(
        self,
        store: DigestStore,
        config: Optional[DigestConfig] = None,
        client: Optional[Any] = None,
        model: Optional[str] = None,
    ):
        self.store = store
        self.config = config or DigestConfig()
        self.client = client
        self.model = model

    async def generate(
        self,
        hours: Optional[float] = None,
        since: Optional[float] = None,
        source: Optional[str] = None,
        audience: Optional[str] = None,
        tag: Optional[str] = None,
        mark_consumed: bool = True,
    ) -> DigestResult:
        """
        Generate a digest from accumulated entries.

        Args:
            hours: Include entries from the last N hours.
            since: Include entries after this timestamp.
            source: Filter by source agent/role.
            audience: Filter by target audience (e.g. "transactions").
            tag: Filter by tag.
            mark_consumed: Mark entries as consumed after digest.

        Returns:
            DigestResult with the synthesized markdown.
        """
        # Pull entries
        if hours is not None or since is not None:
            entries = self.store.get_since(
                hours=hours, since=since, source=source,
                audience=audience, tag=tag,
            )
        else:
            entries = self.store.get_unconsumed(source=source, audience=audience)
            if tag:
                entries = [e for e in entries if tag in e.tags]

        entries = entries[: self.config.max_entries]

        digest_id = uuid.uuid4().hex
        title = self.config.title_template.format(
            date=time.strftime("%Y-%m-%d %H:%M"),
            count=len(entries),
            hours=hours or 0,
        )

        if not entries:
            return DigestResult(
                id=digest_id,
                markdown=f"# {title}\n\nNo new activity to report.",
                title=title,
                entry_count=0,
                entry_ids=[],
                created_at=time.time(),
                synthesized=False,
            )

        # Try LLM synthesis
        markdown = None
        synthesized = False

        if self.client is not None:
            try:
                formatted = format_entries(
                    entries,
                    group_by_source=self.config.group_by_source,
                    include_metadata=self.config.include_metadata,
                )
                prompt = f"{self.config.synthesis_prompt}\n\n{formatted}"

                kwargs: Dict[str, Any] = {"prompt": prompt, "system": self.config.system_prompt}
                if self.model:
                    kwargs["model"] = self.model

                response = await self.client.generate(**kwargs)
                markdown = f"# {title}\n\n{response.strip()}"
                synthesized = True
            except Exception:
                logger.warning("Digest LLM synthesis failed, using fallback", exc_info=True)

        if markdown is None:
            markdown = _structured_fallback(entries, title)

        entry_ids = [e.id for e in entries]

        # Mark consumed
        if mark_consumed and entry_ids:
            self.store.mark_consumed(entry_ids, digest_id=digest_id)

        result = DigestResult(
            id=digest_id,
            markdown=markdown,
            title=title,
            entry_count=len(entries),
            entry_ids=entry_ids,
            created_at=time.time(),
            synthesized=synthesized,
        )

        logger.info(
            f"Digest generated: {len(entries)} entries, "
            f"synthesized={synthesized}, id={digest_id}"
        )
        return result
