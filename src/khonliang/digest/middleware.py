"""
Digest middleware — auto-record digest entries from existing modules.

Provides hooks that wire into Blackboard posts, consensus results, and
response metadata so digest entries accumulate automatically without
manual instrumentation in every agent.

Usage:
    store = DigestStore("digest.db")

    # Hook into Blackboard — posts to specific sections get recorded
    board = Blackboard()
    digest_blackboard(board, store, sections=["findings", "alerts"])

    # Extract digest entries from response metadata
    response = {"content": "...", "metadata": {"digest": "Found census record"}}
    extract_from_response(response, store, source="researcher")

    # Extract with audience
    response = {"metadata": {"digest": "Trade executed", "digest_audience": "transactions"}}
    extract_from_response(response, store, source="trader")
"""

import logging
from typing import Any, Dict, List, Optional

from khonliang.digest.store import DigestStore

logger = logging.getLogger(__name__)


def extract_from_response(
    response: Dict[str, Any],
    store: DigestStore,
    source: Optional[str] = None,
    default_audience: Optional[str] = None,
) -> bool:
    """
    Extract a digest entry from a response's metadata.

    Looks for a "digest" key in response metadata. If found, records it
    as a digest entry. Also checks for "digest_audience" and "digest_tags".

    Args:
        response: Response dict with optional metadata.digest field.
        store: DigestStore to record to.
        source: Source agent/role. Falls back to response's "role" field.
        default_audience: Default audience if not specified in metadata.

    Returns:
        True if a digest entry was recorded.
    """
    metadata = response.get("metadata", {})
    if not metadata:
        return False

    digest_text = metadata.get("digest")
    if not digest_text:
        return False

    entry_source = source or response.get("role", "agent")
    audience = metadata.get("digest_audience", default_audience)
    tags = metadata.get("digest_tags", [])

    # Pass through any extra metadata (excluding digest control keys)
    extra = {
        k: v for k, v in metadata.items()
        if not k.startswith("digest") and k not in ("role", "reason")
    }

    store.record(
        summary=digest_text,
        source=entry_source,
        audience=audience,
        tags=tags if isinstance(tags, list) else [tags],
        metadata=extra or None,
    )
    return True


def digest_blackboard(
    blackboard: Any,
    store: DigestStore,
    sections: Optional[List[str]] = None,
    audience: Optional[str] = None,
) -> None:
    """
    Monkey-patch a Blackboard to auto-record digest entries on post().

    Every post to the specified sections (or all sections if None) will
    be recorded as a digest entry.

    Args:
        blackboard: Blackboard instance to hook into.
        store: DigestStore to record to.
        sections: List of section names to track. None = all sections.
        audience: Default audience for entries from this blackboard.
    """
    original_post = blackboard.post

    def patched_post(agent_id, section, key, content, ttl=None):
        original_post(agent_id, section, key, content, ttl=ttl)

        if sections is not None and section not in sections:
            return

        # Record to digest
        summary = str(content)[:200] if not isinstance(content, str) else content[:200]
        store.record(
            summary=summary,
            source=agent_id,
            audience=audience,
            tags=[section, key],
        )

    blackboard.post = patched_post
    logger.debug(
        f"Blackboard digest hook installed for sections: "
        f"{sections or 'all'}"
    )


def digest_consensus(
    store: DigestStore,
    audience: Optional[str] = None,
):
    """
    Create a callback for consensus results that records digest entries.

    Returns a callable that accepts a ConsensusResult and records it.

    Usage:
        on_consensus = digest_consensus(store, audience="decisions")
        # Call after consensus completes
        on_consensus(result, subject="Should we escalate?")

    Args:
        store: DigestStore to record to.
        audience: Default audience for consensus entries.

    Returns:
        Callback function(result, subject=None).
    """
    def callback(result, subject: Optional[str] = None):
        action = getattr(result, "action", "unknown")
        confidence = getattr(result, "confidence", 0)
        vote_count = len(getattr(result, "votes", []))

        summary = (
            f"Consensus: {action} (confidence={confidence:.0%}, "
            f"{vote_count} votes)"
        )
        if subject:
            summary = f"{subject} — {summary}"

        store.record(
            summary=summary,
            source="consensus",
            audience=audience,
            tags=["consensus", action],
            metadata={
                "action": action,
                "confidence": confidence,
                "vote_count": vote_count,
            },
        )

    return callback
