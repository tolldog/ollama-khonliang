"""
digest — Accumulate, synthesize, and publish activity digests.

Agents tag responses with digest metadata as they work. Entries
accumulate in a DigestStore, then a DigestSynthesizer pulls them
by time window or audience to produce narrative digests via LLM.

Quick start:

    from khonliang.digest import DigestStore, DigestSynthesizer, DigestConfig

    store = DigestStore("digest.db")
    store.record("Found census record", source="researcher", audience="research")

    config = DigestConfig(synthesis_prompt="Summarize research progress.")
    synth = DigestSynthesizer(store, config, client=llm_client)
    result = await synth.generate(audience="research")
    print(result.markdown)
"""

from khonliang.digest.middleware import (
    digest_blackboard,
    digest_consensus,
    extract_from_response,
)
from khonliang.digest.store import DigestEntry, DigestStore
from khonliang.digest.synthesizer import DigestConfig, DigestResult, DigestSynthesizer

__all__ = [
    "DigestEntry",
    "DigestStore",
    "DigestConfig",
    "DigestResult",
    "DigestSynthesizer",
    "extract_from_response",
    "digest_blackboard",
    "digest_consensus",
]
