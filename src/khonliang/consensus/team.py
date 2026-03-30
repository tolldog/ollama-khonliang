"""
AgentTeam — orchestrates multiple voting agents in parallel.

Agents are any class implementing the VotingAgent protocol:

    class MyAgent:
        agent_id: str
        async def analyze(self, subject: str, context: dict) -> AgentVote: ...

The team fans out analyze() calls in parallel, collects votes,
and runs them through ConsensusEngine.

Example:

    class UrgencyAgent:
        agent_id = "urgency"
        async def analyze(self, subject, context):
            # subject = ticket text, context = metadata
            score = score_urgency(subject)
            return AgentVote(
                agent_id=self.agent_id,
                action="APPROVE" if score > 0.7 else "DEFER",
                confidence=score,
                reasoning=f"Urgency score: {score:.2f}",
            )

    team = AgentTeam(agents=[UrgencyAgent(), SentimentAgent()])
    result = await team.evaluate("Server is down!", context={})
    print(result.action, result.confidence)
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from khonliang.consensus.engine import ConsensusEngine
from khonliang.consensus.models import AgentVote, ConsensusResult

logger = logging.getLogger(__name__)


class AgentTeam:
    """
    Runs N agents in parallel and aggregates their votes.

    Args:
        agents:          List of voting agents (must have agent_id + analyze())
        consensus_engine: Custom ConsensusEngine (or default)
        agent_timeout:   Max seconds to wait for each agent
        enable_caching:  Cache votes per subject for cache_ttl_seconds
        cache_ttl_seconds: How long to cache votes (default 300s / 5 min)
    """

    def __init__(
        self,
        agents: List[Any],
        consensus_engine: Optional[ConsensusEngine] = None,
        agent_timeout: float = 30.0,
        enable_caching: bool = True,
        cache_ttl_seconds: float = 300.0,
        summarizer_fn: Optional[Callable] = None,
        rounds: int = 1,
    ):
        self.agents = agents
        self.consensus_engine = consensus_engine or ConsensusEngine()
        self.agent_timeout = agent_timeout
        self.enable_caching = enable_caching
        self._cache_ttl = cache_ttl_seconds
        self._vote_cache: Dict[str, tuple] = {}  # {key: (votes, timestamp)}
        self.summarizer_fn = summarizer_fn
        self.rounds = max(1, rounds)

        logger.info(f"AgentTeam initialized with {len(agents)} agents")

    async def evaluate(
        self,
        subject: str,
        context: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
    ) -> ConsensusResult:
        """
        Evaluate a subject with all agents and return consensus.

        When rounds > 1 and summarizer_fn is set, runs multiple deliberation
        rounds. Each round's summary is prepended to the context for the
        next round, allowing agents to refine their votes based on prior
        consensus.

        Args:
            subject:   The thing being evaluated (ticket, document, code, etc.)
            context:   Optional metadata dict passed to each agent
            cache_key: Key for vote caching (defaults to subject)
            use_cache: Use cached votes if available

        Returns:
            ConsensusResult with action, confidence, all votes, and
            optional summary from summarizer_fn
        """
        ctx = context or {}
        key = cache_key or subject

        if use_cache and self.enable_caching:
            cached = self._get_cached(key)
            if cached is not None:
                logger.debug(f"Using cached votes for '{key[:40]}'")
                return self.consensus_engine.calculate_consensus(cached)

        summary = None
        votes: List[AgentVote] = []

        for round_num in range(self.rounds):
            round_ctx = dict(ctx)

            # Inject prior round's summary into context for rounds > 0
            if summary is not None:
                round_ctx["prior_summary"] = summary

            votes = await self._collect_votes(subject, round_ctx)

            # Generate summary after collecting votes
            if self.summarizer_fn is not None:
                try:
                    summary = await self._call_summarizer(votes, subject, round_ctx)
                except Exception as e:
                    logger.error(f"Summarizer failed in round {round_num + 1}: {e}")
                    summary = None

            # Only continue to next round if we have a summarizer
            if self.summarizer_fn is None:
                break

            if round_num < self.rounds - 1:
                logger.debug(
                    f"Round {round_num + 1}/{self.rounds} complete, "
                    f"continuing deliberation"
                )

        if self.enable_caching:
            self._cache_votes(key, votes)

        result = self.consensus_engine.calculate_consensus(votes)
        result.summary = summary
        result.debate_rounds = self.rounds if self.rounds > 1 else result.debate_rounds

        logger.info(
            f"Consensus for '{subject[:40]}': {result.action} "
            f"(confidence={result.confidence:.2f}, {len(votes)} votes, "
            f"rounds={self.rounds})"
        )
        return result

    async def _call_summarizer(
        self,
        votes: List[AgentVote],
        subject: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Call the summarizer function, handling both sync and async."""
        result = self.summarizer_fn(votes, subject, context)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def evaluate_sync(
        self,
        subject: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ConsensusResult:
        """Synchronous wrapper for evaluate()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.evaluate(subject, context, **kwargs),
                    )
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(self.evaluate(subject, context, **kwargs))
        except Exception as e:
            logger.error(f"Sync evaluate failed: {e}")
            return ConsensusResult(action="DEFER", confidence=0.0, reason=str(e))

    async def _collect_votes(
        self, subject: str, context: Dict[str, Any]
    ) -> List[AgentVote]:
        tasks = [
            (agent.agent_id, asyncio.create_task(agent.analyze(subject, context)))
            for agent in self.agents
        ]
        votes = []
        for agent_id, task in tasks:
            try:
                vote = await asyncio.wait_for(task, timeout=self.agent_timeout)
                if vote:
                    votes.append(vote)
            except asyncio.TimeoutError:
                logger.warning(f"Agent '{agent_id}' timed out")
            except Exception as e:
                logger.error(f"Agent '{agent_id}' failed: {e}")
        return votes

    def _get_cached(self, key: str) -> Optional[List[AgentVote]]:
        if key in self._vote_cache:
            votes, ts = self._vote_cache[key]
            if time.time() - ts < self._cache_ttl:
                return votes
        return None

    def _cache_votes(self, key: str, votes: List[AgentVote]) -> None:
        self._vote_cache[key] = (votes, time.time())
        if len(self._vote_cache) > 500:
            now = time.time()
            self._vote_cache = {
                k: v for k, v in self._vote_cache.items()
                if now - v[1] < self._cache_ttl
            }

    def clear_cache(self) -> None:
        """Remove all cached vote results."""
        self._vote_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return team configuration and cache statistics."""
        return {
            "agent_count": len(self.agents),
            "agents": [a.agent_id for a in self.agents],
            "caching_enabled": self.enable_caching,
            "cache_size": len(self._vote_cache),
            "cache_ttl_seconds": self._cache_ttl,
            "agent_timeout_seconds": self.agent_timeout,
        }
