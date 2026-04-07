"""
Conversation manager — orchestrates multi-agent conversations.

Supports multiple turn policies for different collaboration patterns.
Agents must implement the ConversableAgent protocol (respond method).

The ALL_THEN_ORGANIZE policy is designed for unbiased input collection:
round 1 is parallel fan-out where each agent responds independently
(no anchoring bias), then subsequent rounds provide full visibility
of all contributions for structured follow-up.

Usage:
    manager = ConversationManager(agents, config)
    result = await manager.run("Should we buy TSLA?")
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from khonliang.conversation.models import (
    ConversationConfig,
    ConversationHistory,
    ConversationMessage,
    Stance,
    TurnPolicy,
)

logger = logging.getLogger(__name__)


class ConversableAgent(Protocol):
    """Protocol for agents participating in conversations.

    Any object with agent_id and an async respond() method works.
    The respond method receives the topic, conversation history,
    and optional context.
    """

    @property
    def agent_id(self) -> str: ...

    async def respond(
        self,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]] = None,
    ) -> str: ...


class ConversationResult:
    """Result of a completed conversation.

    Attributes:
        history:        Full conversation history
        rounds:         Number of rounds completed
        terminated_by:  Why the conversation ended ("max_rounds", "consensus",
                        "moderator", "no_responses")
        summary:        Optional summary (populated if a summarizer is provided)
    """

    def __init__(
        self,
        history: ConversationHistory,
        terminated_by: str = "max_rounds",
        summary: Optional[str] = None,
    ):
        self.history = history
        self.rounds = history.current_round
        self.terminated_by = terminated_by
        self.summary = summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rounds": self.rounds,
            "terminated_by": self.terminated_by,
            "participants": self.history.participants,
            "message_count": len(self.history.messages),
            "summary": self.summary,
            "history": self.history.to_dict(),
        }


class ConversationManager:
    """Orchestrates multi-agent conversations with configurable turn policies.

    Turn policies:
        ROUND_ROBIN:       Agents speak in order each round.
        ALL_THEN_ORGANIZE: Round 1 is parallel (unbiased input from all agents
                           without seeing each other's responses). Subsequent
                           rounds are sequential with full history visibility.
        DIRECTED:          Current speaker nominates the next via reply_to metadata.
        MODERATOR:         A designated agent picks who speaks next.
        ANY:               All agents are asked; first responder wins the turn.

    Example:
        manager = ConversationManager(
            agents=[analyst, researcher, critic],
            config=ConversationConfig(
                turn_policy=TurnPolicy.ALL_THEN_ORGANIZE,
                max_rounds=3,
            ),
        )
        result = await manager.run("Evaluate TSLA for swing trade entry")
    """

    def __init__(
        self,
        agents: List[Any],
        config: Optional[ConversationConfig] = None,
        summarizer: Optional[Callable[..., Awaitable[str]]] = None,
    ):
        """
        Args:
            agents: List of ConversableAgent-protocol objects
            config: Conversation configuration
            summarizer: Optional async fn(history) -> summary string,
                        called after conversation ends
        """
        self.agents = agents
        self.config = config or ConversationConfig()
        self.summarizer = summarizer
        self._agent_map = {a.agent_id: a for a in agents}

    async def run(
        self,
        topic: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConversationResult:
        """Run a conversation to completion.

        Args:
            topic: The subject to discuss
            context: Optional domain context passed to all agents

        Returns:
            ConversationResult with full history and metadata
        """
        history = ConversationHistory(topic=topic)
        terminated_by = "max_rounds"

        for round_num in range(1, self.config.max_rounds + 1):
            messages = await self._run_round(
                round_num, topic, history, context
            )

            if not messages:
                terminated_by = "no_responses"
                break

            for msg in messages:
                history.add(msg)

            # Check for consensus termination
            if self.config.terminate_on_consensus and self._check_consensus(messages):
                terminated_by = "consensus"
                logger.info(
                    f"Conversation terminated by consensus in round {round_num}"
                )
                break

        # Run summarizer if configured
        summary = None
        if self.summarizer:
            try:
                summary = await self.summarizer(history)
            except Exception as e:
                logger.warning(f"Summarizer failed: {e}")

        return ConversationResult(
            history=history,
            terminated_by=terminated_by,
            summary=summary,
        )

    async def _run_round(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """Run a single conversation round based on the turn policy."""
        policy = self.config.turn_policy

        if policy == TurnPolicy.ALL_THEN_ORGANIZE:
            if round_num == 1:
                # Parallel fan-out — unbiased input, no one sees others
                return await self._round_parallel(
                    round_num, topic, history, context
                )
            else:
                # Sequential with full history visibility
                return await self._round_sequential(
                    round_num, topic, history, context, self.agents
                )

        elif policy == TurnPolicy.ROUND_ROBIN:
            return await self._round_sequential(
                round_num, topic, history, context, self.agents
            )

        elif policy == TurnPolicy.ANY:
            return await self._round_any(
                round_num, topic, history, context
            )

        elif policy == TurnPolicy.MODERATOR:
            return await self._round_moderated(
                round_num, topic, history, context
            )

        elif policy == TurnPolicy.DIRECTED:
            return await self._round_directed(
                round_num, topic, history, context
            )

        return []

    async def _round_parallel(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """All agents respond simultaneously — no one sees others' responses.

        This prevents anchoring bias: each agent forms an independent
        opinion before being exposed to the group.
        """
        tasks = [
            self._get_response(agent, round_num, topic, history, context)
            for agent in self.agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        messages = []
        for result in results:
            if isinstance(result, ConversationMessage):
                messages.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Agent failed in parallel round: {result}")

        return messages

    async def _round_sequential(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
        agent_order: List[Any],
    ) -> List[ConversationMessage]:
        """Agents respond one at a time, each seeing prior messages in this round."""
        messages = []
        # Build a working history that includes this round's messages as they arrive
        for agent in agent_order:
            msg = await self._get_response(
                agent, round_num, topic, history, context
            )
            if msg is not None:
                messages.append(msg)
                # Add to history so next agent can see it
                history.add(msg)

        # Remove from history — caller will re-add all at once
        for msg in messages:
            if msg in history.messages:
                history.messages.remove(msg)

        return messages

    async def _round_any(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """All agents race — first valid response wins."""
        tasks = {
            agent.agent_id: asyncio.create_task(
                self._get_response(agent, round_num, topic, history, context)
            )
            for agent in self.agents
        }

        done, pending = await asyncio.wait(
            tasks.values(),
            timeout=self.config.agent_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel losers
        for task in pending:
            task.cancel()

        for task in done:
            try:
                result = task.result()
                if isinstance(result, ConversationMessage):
                    return [result]
            except Exception:
                continue

        return []

    async def _round_moderated(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """Moderator picks who speaks, one at a time."""
        moderator_id = self.config.moderator_id
        if not moderator_id or moderator_id not in self._agent_map:
            logger.warning("No moderator configured, falling back to round_robin")
            return await self._round_sequential(
                round_num, topic, history, context, self.agents
            )

        moderator = self._agent_map[moderator_id]
        non_moderators = [a for a in self.agents if a.agent_id != moderator_id]

        # Ask moderator who should speak
        pick_history = ConversationHistory(topic=topic, messages=list(history.messages))
        moderator_msg = await self._get_response(
            moderator, round_num, topic, pick_history, context
        )

        messages = []
        if moderator_msg:
            messages.append(moderator_msg)

        # Let each non-moderator respond sequentially
        for agent in non_moderators:
            msg = await self._get_response(
                agent, round_num, topic, history, context
            )
            if msg:
                messages.append(msg)

        return messages

    async def _round_directed(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """Previous speaker nominates the next via reply_to metadata."""
        # Start with first agent, then follow reply_to chain
        if not self.agents:
            return []

        # Determine starting agent
        if history.messages:
            last = history.messages[-1]
            next_id = last.metadata.get("next_speaker")
            current = self._agent_map.get(next_id, self.agents[0])
        else:
            current = self.agents[0]

        messages = []
        visited = set()

        for _ in range(len(self.agents)):
            if current.agent_id in visited:
                break
            visited.add(current.agent_id)

            msg = await self._get_response(
                current, round_num, topic, history, context
            )
            if msg is None:
                break

            messages.append(msg)
            history.add(msg)

            # Check for next_speaker nomination
            next_id = msg.metadata.get("next_speaker")
            if next_id and next_id in self._agent_map:
                current = self._agent_map[next_id]
            else:
                break

        # Remove from history — caller re-adds
        for msg in messages:
            if msg in history.messages:
                history.messages.remove(msg)

        return messages

    async def _get_response(
        self,
        agent: Any,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ConversationMessage]:
        """Get a single response from an agent with timeout."""
        try:
            content = await asyncio.wait_for(
                agent.respond(topic, history, context),
                timeout=self.config.agent_timeout,
            )
            meta = getattr(agent, "_last_metadata", {})
            stance = meta.pop("stance", Stance.NEUTRAL)
            conviction = meta.pop("conviction", 0.5)
            return ConversationMessage(
                agent_id=agent.agent_id,
                content=content,
                round_num=round_num,
                stance=stance if isinstance(stance, str) else stance.value,
                conviction=float(conviction),
                metadata=meta,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Agent {agent.agent_id} timed out in round {round_num}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Agent {agent.agent_id} failed in round {round_num}: {e}"
            )
            return None

    def _check_consensus(self, messages: List[ConversationMessage]) -> bool:
        """Check if agents have converged in this round.

        Considers:
        - Explicit action agreement (metadata "action" field)
        - All agents conceding (stance = conceding)
        - High average conviction with same action
        - Identical responses
        """
        if len(messages) < 2:
            return False

        # All agents conceding = consensus reached
        stances = [m.stance for m in messages]
        if all(s == Stance.CONCEDING for s in stances):
            return True

        # Check metadata "action" field if present
        actions = [m.metadata.get("action") for m in messages if m.metadata.get("action")]
        if actions and len(actions) == len(messages) and len(set(actions)) == 1:
            return True

        # High conviction + same action = strong consensus
        if actions and len(set(actions)) == 1:
            avg_conviction = sum(m.conviction for m in messages) / len(messages)
            if avg_conviction >= 0.7:
                return True

        # Identical text responses
        texts = [m.content.strip().lower() for m in messages]
        if len(set(texts)) == 1:
            return True

        return False
