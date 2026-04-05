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


def _select_best_vote(candidates: List[AgentVote]) -> AgentVote:
    """Select the best vote from N candidates.

    Heuristic: pick the candidate whose action matches the plurality
    of all candidates, with highest confidence as tiebreaker.

    Example with 3 candidates: [BUY(0.7), BUY(0.9), SELL(0.6)]
      → Plurality action = BUY (2 votes)
      → Best = BUY(0.9) (highest confidence among BUY votes)
    """
    if len(candidates) == 1:
        return candidates[0]

    # Count actions
    action_counts: Dict[str, int] = {}
    for v in candidates:
        action_counts[v.action] = action_counts.get(v.action, 0) + 1

    # Find plurality action
    plurality_action = max(action_counts, key=lambda a: action_counts[a])

    # Among candidates with the plurality action, pick highest confidence
    matching = [v for v in candidates if v.action == plurality_action]
    return max(matching, key=lambda v: v.confidence)


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
        self._result_cache: Dict[str, tuple] = {}  # {key: (ConsensusResult, timestamp)}
        self.summarizer_fn = summarizer_fn
        self.rounds = max(1, rounds)

        logger.info(f"AgentTeam initialized with {len(agents)} agents")

    async def evaluate(
        self,
        subject: str,
        context: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        sample_count: int = 1,
    ) -> ConsensusResult:
        """
        Evaluate a subject with all agents and return consensus.

        When rounds > 1 and summarizer_fn is set, runs multiple deliberation
        rounds. Each round's summary is prepended to the context for the
        next round, allowing agents to refine their votes based on prior
        consensus.

        Args:
            subject:     The thing being evaluated (ticket, document, code, etc.)
            context:     Optional metadata dict passed to each agent
            cache_key:   Key for vote caching (defaults to subject)
            use_cache:   Use cached votes if available
            sample_count: Number of candidate votes per agent (default 1).
                When > 1, each agent generates N votes and the best one
                (plurality action + highest confidence) is submitted.

        Returns:
            ConsensusResult with action, confidence, all votes, and
            optional summary from summarizer_fn
        """
        ctx = context or {}
        key = cache_key or subject

        if use_cache and self.enable_caching:
            cached_result = self._get_cached(key)
            if cached_result is not None:
                logger.debug(f"Using cached result for '{key[:40]}'")
                return cached_result

        summary = None
        votes: List[AgentVote] = []

        for round_num in range(self.rounds):
            round_ctx = dict(ctx)

            # Inject prior round's summary into context for rounds > 0
            if summary is not None:
                round_ctx["prior_summary"] = summary

            votes = await self._collect_votes(subject, round_ctx, sample_count)

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

        result = self.consensus_engine.calculate_consensus(votes)
        result.summary = summary
        # Track actual rounds completed (loop runs 0..rounds-1)
        actual_rounds = min(round_num + 1, self.rounds) if self.rounds > 1 else 0
        result.debate_rounds = actual_rounds

        if self.enable_caching:
            self._cache_result(key, result)

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
        """Call the summarizer function, handling sync, async, and awaitable results."""
        import inspect

        result = self.summarizer_fn(votes, subject, context)
        if inspect.isawaitable(result):
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
        self,
        subject: str,
        context: Dict[str, Any],
        sample_count: int = 1,
    ) -> List[AgentVote]:
        if sample_count <= 1:
            return await self._collect_single_votes(subject, context)

        # Group sampling: each agent generates N candidates in parallel
        agent_tasks = [
            self._sample_agent(agent, subject, context, sample_count)
            for agent in self.agents
        ]
        per_agent_candidates = await asyncio.gather(*agent_tasks)
        best_votes = []
        for candidates in per_agent_candidates:
            if candidates:
                best_votes.append(_select_best_vote(candidates))
        return best_votes

    async def _collect_single_votes(
        self, subject: str, context: Dict[str, Any]
    ) -> List[AgentVote]:
        """Standard single-vote collection (original behavior)."""
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

    async def _sample_agent(
        self,
        agent: Any,
        subject: str,
        context: Dict[str, Any],
        n: int,
    ) -> List[AgentVote]:
        """Collect N candidate votes from a single agent in parallel."""
        tasks = [
            asyncio.create_task(agent.analyze(subject, context))
            for _ in range(n)
        ]
        candidates = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, asyncio.TimeoutError):
                logger.warning(f"Agent '{agent.agent_id}' sample timed out")
            elif isinstance(res, Exception):
                logger.debug(f"Agent '{agent.agent_id}' sample failed: {res}")
            elif res:
                candidates.append(res)
        return candidates

    def _get_cached(self, key: str) -> Optional[ConsensusResult]:
        if key in self._result_cache:
            result, ts = self._result_cache[key]
            if time.time() - ts < self._cache_ttl:
                return result
        return None

    def _cache_result(self, key: str, result: ConsensusResult) -> None:
        self._result_cache[key] = (result, time.time())
        if len(self._result_cache) > 500:
            now = time.time()
            self._result_cache = {
                k: v for k, v in self._result_cache.items()
                if now - v[1] < self._cache_ttl
            }

    def clear_cache(self) -> None:
        """Remove all cached results."""
        self._result_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return team configuration and cache statistics."""
        return {
            "agent_count": len(self.agents),
            "agents": [a.agent_id for a in self.agents],
            "caching_enabled": self.enable_caching,
            "cache_size": len(self._result_cache),
            "cache_ttl_seconds": self._cache_ttl,
            "agent_timeout_seconds": self.agent_timeout,
        }
