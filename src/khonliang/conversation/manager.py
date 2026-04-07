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
    result = await manager.run("How should we prioritize the next sprint?")
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
        DIRECTED:          Current speaker nominates the next via ``next_speaker``
                           key in their response metadata.
        MODERATOR:         A designated moderator agent speaks first each round
                           to set direction, then other agents respond sequentially.
        ANY:               All agents are asked; first valid responder wins the turn.

    Example:
        manager = ConversationManager(
            agents=[agent_a, agent_b, agent_c],
            config=ConversationConfig(
                turn_policy=TurnPolicy.ALL_THEN_ORGANIZE,
                max_rounds=3,
            ),
        )
        result = await manager.run("How should we approach this problem?")
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

        Raises:
            ValueError: If agent_id values are not unique
        """
        self.agents = agents
        self.config = config or ConversationConfig()
        self.summarizer = summarizer

        # Validate unique agent IDs
        seen: set[str] = set()
        for a in agents:
            if a.agent_id in seen:
                raise ValueError(f"Duplicate agent_id: '{a.agent_id}'")
            seen.add(a.agent_id)

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
                return await self._round_parallel(
                    round_num, topic, history, context
                )
            else:
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
        """Agents respond one at a time, each seeing prior messages in this round.

        Uses a working copy of history so the caller's history is not mutated.
        """
        messages: List[ConversationMessage] = []
        # Working copy: agents see base history + this round's messages so far
        working = ConversationHistory(
            topic=history.topic,
            messages=list(history.messages),
        )

        for agent in agent_order:
            msg = await self._get_response(
                agent, round_num, topic, working, context
            )
            if msg is not None:
                messages.append(msg)
                working.add(msg)

        return messages

    async def _round_any(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """All agents race — first valid (non-None) response wins."""
        tasks = [
            asyncio.create_task(
                self._get_response(agent, round_num, topic, history, context)
            )
            for agent in self.agents
        ]

        remaining = set(tasks)
        result_msg: Optional[ConversationMessage] = None

        while remaining and result_msg is None:
            done, remaining = await asyncio.wait(
                remaining,
                timeout=self.config.agent_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break

            for task in done:
                try:
                    result = task.result()
                    if isinstance(result, ConversationMessage):
                        result_msg = result
                        break
                except Exception:
                    continue

        # Cancel remaining tasks
        for task in remaining:
            task.cancel()

        return [result_msg] if result_msg else []

    async def _round_moderated(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """Moderator speaks first to set direction, then others respond sequentially.

        The moderator's message is visible to all subsequent speakers in the round.
        """
        moderator_id = self.config.moderator_id
        if not moderator_id or moderator_id not in self._agent_map:
            logger.warning("No moderator configured, falling back to round_robin")
            return await self._round_sequential(
                round_num, topic, history, context, self.agents
            )

        moderator = self._agent_map[moderator_id]
        non_moderators = [a for a in self.agents if a.agent_id != moderator_id]

        # Working copy for this round
        working = ConversationHistory(
            topic=history.topic,
            messages=list(history.messages),
        )

        messages: List[ConversationMessage] = []

        # Moderator speaks first
        mod_msg = await self._get_response(
            moderator, round_num, topic, working, context
        )
        if mod_msg:
            messages.append(mod_msg)
            working.add(mod_msg)

        # Others respond sequentially, seeing moderator's message
        for agent in non_moderators:
            msg = await self._get_response(
                agent, round_num, topic, working, context
            )
            if msg:
                messages.append(msg)
                working.add(msg)

        return messages

    async def _round_directed(
        self,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> List[ConversationMessage]:
        """Current speaker nominates the next via ``next_speaker`` metadata key.

        Each speaker sets metadata["next_speaker"] to an agent_id to pass
        the conversation to that agent. The chain stops when no valid
        next_speaker is nominated or all agents have spoken.
        """
        if not self.agents:
            return []

        # Working copy for this round
        working = ConversationHistory(
            topic=history.topic,
            messages=list(history.messages),
        )

        # Determine starting agent from last message's next_speaker
        if history.messages:
            last = history.messages[-1]
            next_id = last.metadata.get("next_speaker")
            current = self._agent_map.get(next_id, self.agents[0])
        else:
            current = self.agents[0]

        messages: List[ConversationMessage] = []
        visited: set[str] = set()

        for _ in range(len(self.agents)):
            if current.agent_id in visited:
                break
            visited.add(current.agent_id)

            msg = await self._get_response(
                current, round_num, topic, working, context
            )
            if msg is None:
                break

            messages.append(msg)
            working.add(msg)

            # Follow next_speaker nomination
            next_id = msg.metadata.get("next_speaker")
            if next_id and next_id in self._agent_map:
                current = self._agent_map[next_id]
            else:
                break

        return messages

    async def _get_response(
        self,
        agent: Any,
        round_num: int,
        topic: str,
        history: ConversationHistory,
        context: Optional[Dict[str, Any]],
    ) -> Optional[ConversationMessage]:
        """Get a single response from an agent with timeout.

        Copies agent metadata to avoid mutating agent state.
        """
        try:
            content = await asyncio.wait_for(
                agent.respond(topic, history, context),
                timeout=self.config.agent_timeout,
            )
            # Copy metadata to avoid mutating agent state
            meta = dict(getattr(agent, "_last_metadata", {}))
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
        - All agents conceding (stance = conceding)
        - All agents agree on explicit action + high average conviction
        - Identical text responses
        """
        if len(messages) < 2:
            return False

        # All agents conceding = consensus reached
        stances = [m.stance for m in messages]
        if all(s == Stance.CONCEDING for s in stances):
            return True

        # Check metadata "action" field — require all agents to have one
        actions = [m.metadata.get("action") for m in messages]
        if all(a is not None for a in actions) and len(set(actions)) == 1:
            # Same action from everyone — check conviction threshold
            avg_conviction = sum(m.conviction for m in messages) / len(messages)
            if avg_conviction >= 0.7:
                return True

        # Identical text responses
        texts = [m.content.strip().lower() for m in messages]
        if len(set(texts)) == 1:
            return True

        return False
