"""Tests for multi-agent conversation framework (FR khonliang_0)."""

import pytest

from khonliang.conversation.manager import ConversationManager, ConversationResult
from khonliang.conversation.models import (
    ConversationConfig,
    ConversationHistory,
    ConversationMessage,
    Stance,
    TurnPolicy,
)

# ---------------------------------------------------------------------------
# Stub agents
# ---------------------------------------------------------------------------


class StubAgent:
    """Minimal agent for conversation testing."""

    def __init__(self, agent_id: str, responses: list[str] | None = None,
                 stance: str = "neutral", conviction: float = 0.5):
        self.agent_id = agent_id
        self._responses = responses or [f"{agent_id} response"]
        self._call_count = 0
        self._last_metadata = {"stance": stance, "conviction": conviction}

    async def respond(self, topic, history, context=None):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class ActionAgent(StubAgent):
    """Agent that includes an action in metadata."""

    def __init__(self, agent_id: str, action: str, response: str = "",
                 conviction: float = 0.8):
        super().__init__(agent_id, [response or f"{action} because reasons"])
        self._last_metadata = {
            "action": action,
            "stance": "firm",
            "conviction": conviction,
        }


class ConcedingAgent(StubAgent):
    """Agent that concedes based on conversation round number."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["initial position", "I concede"])

    async def respond(self, topic, history, context=None):
        current_round = history.current_round + 1  # next round being built
        if current_round > 1:
            self._last_metadata = {
                "stance": Stance.CONCEDING,
                "conviction": 0.3,
            }
            return "I concede"
        else:
            self._last_metadata = {
                "stance": Stance.FIRM,
                "conviction": 0.8,
            }
            return "initial position"


class TimeoutAgent:
    """Agent that always times out."""

    agent_id = "slow"
    _last_metadata = {}

    async def respond(self, topic, history, context=None):
        import asyncio
        await asyncio.sleep(100)
        return "never reached"


# ---------------------------------------------------------------------------
# ConversationMessage tests
# ---------------------------------------------------------------------------


class TestConversationMessage:
    def test_creation(self):
        msg = ConversationMessage(agent_id="a", content="hello", round_num=1)
        assert msg.agent_id == "a"
        assert msg.stance == "neutral"
        assert msg.conviction == 0.5

    def test_to_dict(self):
        msg = ConversationMessage(
            agent_id="a", content="hi", stance="firm", conviction=0.9
        )
        d = msg.to_dict()
        assert d["stance"] == "firm"
        assert d["conviction"] == 0.9


class TestConversationHistory:
    def test_add_and_query(self):
        h = ConversationHistory(topic="test")
        h.add(ConversationMessage(agent_id="a", content="hello", round_num=1))
        h.add(ConversationMessage(agent_id="b", content="world", round_num=1))

        assert h.current_round == 1
        assert len(h.participants) == 2
        assert len(h.get_round(1)) == 2

    def test_build_context(self):
        h = ConversationHistory(topic="test")
        h.add(ConversationMessage(agent_id="a", content="I think yes", round_num=1))
        h.add(ConversationMessage(agent_id="b", content="I disagree", round_num=1))

        ctx = h.build_context(for_agent="a")
        assert "You: I think yes" in ctx
        assert "b: I disagree" in ctx

    def test_to_dict(self):
        h = ConversationHistory(topic="test")
        h.add(ConversationMessage(agent_id="a", content="hi", round_num=1))
        d = h.to_dict()
        assert d["topic"] == "test"
        assert d["rounds"] == 1


class TestStance:
    def test_values(self):
        assert Stance.FIRM == "firm"
        assert Stance.FLEXIBLE == "flexible"
        assert Stance.CONCEDING == "conceding"
        assert Stance.NEUTRAL == "neutral"


# ---------------------------------------------------------------------------
# ConversationManager — ROUND_ROBIN
# ---------------------------------------------------------------------------


class TestRoundRobin:
    @pytest.mark.asyncio
    async def test_basic_round_robin(self):
        agents = [StubAgent("a"), StubAgent("b"), StubAgent("c")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ROUND_ROBIN, max_rounds=2
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("test topic")

        assert result.rounds == 2
        assert result.terminated_by == "max_rounds"
        assert len(result.history.messages) == 6  # 3 agents x 2 rounds

    @pytest.mark.asyncio
    async def test_sequential_visibility(self):
        """In round_robin, later agents should see earlier messages."""
        seen_counts = {}

        class SpyAgent:
            def __init__(self, aid):
                self.agent_id = aid
                self._last_metadata = {}

            async def respond(self, topic, history, context=None):
                seen_counts[self.agent_id] = len(history.messages)
                return f"{self.agent_id} responds"

        agents = [SpyAgent("a"), SpyAgent("b"), SpyAgent("c")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ROUND_ROBIN, max_rounds=1
        )
        manager = ConversationManager(agents, config)
        await manager.run("test")

        # a sees 0 messages, b sees 1 (a's), c sees 2 (a's + b's)
        assert seen_counts["a"] == 0
        assert seen_counts["b"] == 1
        assert seen_counts["c"] == 2


# ---------------------------------------------------------------------------
# ConversationManager — ALL_THEN_ORGANIZE
# ---------------------------------------------------------------------------


class TestAllThenOrganize:
    @pytest.mark.asyncio
    async def test_round_1_parallel(self):
        """Round 1 should be parallel — all agents respond without seeing others."""
        seen_counts = {}

        class SpyAgent:
            def __init__(self, aid):
                self.agent_id = aid
                self._last_metadata = {}

            async def respond(self, topic, history, context=None):
                seen_counts[self.agent_id] = len(history.messages)
                return f"{self.agent_id} responds"

        agents = [SpyAgent("a"), SpyAgent("b"), SpyAgent("c")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ALL_THEN_ORGANIZE, max_rounds=1
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("test")

        # All agents should see 0 messages in round 1 (unbiased)
        assert seen_counts["a"] == 0
        assert seen_counts["b"] == 0
        assert seen_counts["c"] == 0
        assert len(result.history.messages) == 3

    @pytest.mark.asyncio
    async def test_round_2_sequential_with_visibility(self):
        """Round 2+ should be sequential with full history."""
        round_2_seen = {}

        class SpyAgent:
            def __init__(self, aid):
                self.agent_id = aid
                self._last_metadata = {}
                self._call_count = 0

            async def respond(self, topic, history, context=None):
                self._call_count += 1
                if self._call_count > 1:  # Round 2
                    round_2_seen[self.agent_id] = len(history.messages)
                return f"{self.agent_id} round {self._call_count}"

        agents = [SpyAgent("a"), SpyAgent("b"), SpyAgent("c")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ALL_THEN_ORGANIZE, max_rounds=2
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("test")

        # Round 2 agents should see round 1 messages (3) + preceding round 2 msgs
        assert round_2_seen["a"] == 3  # sees all 3 from round 1
        assert round_2_seen["b"] == 4  # sees 3 from round 1 + a's round 2
        assert round_2_seen["c"] == 5  # sees 3 + a + b
        assert result.rounds == 2


# ---------------------------------------------------------------------------
# ConversationManager — ANY
# ---------------------------------------------------------------------------


class TestAnyPolicy:
    @pytest.mark.asyncio
    async def test_first_responder_wins(self):
        import asyncio

        class FastAgent:
            agent_id = "fast"
            _last_metadata = {}

            async def respond(self, topic, history, context=None):
                return "fast response"

        class SlowAgent:
            agent_id = "slow"
            _last_metadata = {}

            async def respond(self, topic, history, context=None):
                await asyncio.sleep(10)
                return "slow response"

        agents = [SlowAgent(), FastAgent()]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ANY, max_rounds=1, agent_timeout=5.0
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("test")

        assert len(result.history.messages) == 1
        assert result.history.messages[0].agent_id == "fast"


# ---------------------------------------------------------------------------
# Consensus detection
# ---------------------------------------------------------------------------


class TestConsensus:
    @pytest.mark.asyncio
    async def test_action_consensus_terminates(self):
        agents = [
            ActionAgent("a", "APPROVE"),
            ActionAgent("b", "APPROVE"),
            ActionAgent("c", "APPROVE"),
        ]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ALL_THEN_ORGANIZE,
            max_rounds=3,
            terminate_on_consensus=True,
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("evaluate proposal")

        assert result.terminated_by == "consensus"
        assert result.rounds == 1

    @pytest.mark.asyncio
    async def test_conceding_agents_reach_consensus(self):
        agents = [ConcedingAgent("a"), ConcedingAgent("b")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ROUND_ROBIN,
            max_rounds=3,
            terminate_on_consensus=True,
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("debate topic")

        assert result.terminated_by == "consensus"
        assert result.rounds == 2  # Round 2 both concede

    @pytest.mark.asyncio
    async def test_no_consensus_runs_all_rounds(self):
        agents = [
            ActionAgent("a", "APPROVE"),
            ActionAgent("b", "REJECT"),
        ]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ALL_THEN_ORGANIZE,
            max_rounds=2,
            terminate_on_consensus=True,
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("evaluate proposal")

        assert result.terminated_by == "max_rounds"
        assert result.rounds == 2


# ---------------------------------------------------------------------------
# Stance and conviction
# ---------------------------------------------------------------------------


class TestStanceConviction:
    @pytest.mark.asyncio
    async def test_stance_captured(self):
        agent = StubAgent("a", stance="firm", conviction=0.9)
        config = ConversationConfig(max_rounds=1)
        manager = ConversationManager([agent], config)
        result = await manager.run("test")

        msg = result.history.messages[0]
        assert msg.stance == "firm"
        assert msg.conviction == 0.9

    @pytest.mark.asyncio
    async def test_conviction_in_history(self):
        agents = [
            StubAgent("a", conviction=0.9, stance="firm"),
            StubAgent("b", conviction=0.3, stance="flexible"),
        ]
        config = ConversationConfig(
            turn_policy=TurnPolicy.ALL_THEN_ORGANIZE, max_rounds=1
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("test")

        msgs = result.history.messages
        a_msg = [m for m in msgs if m.agent_id == "a"][0]
        b_msg = [m for m in msgs if m.agent_id == "b"][0]
        assert a_msg.conviction == 0.9
        assert b_msg.conviction == 0.3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_timeout_agent(self):
        agents = [StubAgent("a"), TimeoutAgent()]
        config = ConversationConfig(max_rounds=1, agent_timeout=0.1)
        manager = ConversationManager(agents, config)
        result = await manager.run("test")

        # Only the non-timeout agent should have a message
        assert len(result.history.messages) == 1
        assert result.history.messages[0].agent_id == "a"

    @pytest.mark.asyncio
    async def test_empty_agents(self):
        config = ConversationConfig(max_rounds=1)
        manager = ConversationManager([], config)
        result = await manager.run("test")

        assert result.terminated_by == "no_responses"
        assert result.rounds == 0

    @pytest.mark.asyncio
    async def test_summarizer(self):
        agents = [StubAgent("a")]
        config = ConversationConfig(max_rounds=1)

        async def summarize(history):
            return f"Summary: {len(history.messages)} messages"

        manager = ConversationManager(agents, config, summarizer=summarize)
        result = await manager.run("test")

        assert result.summary == "Summary: 1 messages"


class TestConversationResult:
    def test_to_dict(self):
        history = ConversationHistory(topic="test")
        history.add(ConversationMessage(agent_id="a", content="hi"))
        result = ConversationResult(history=history, terminated_by="max_rounds")

        d = result.to_dict()
        assert d["terminated_by"] == "max_rounds"
        assert d["message_count"] == 1
        assert d["participants"] == ["a"]


# ---------------------------------------------------------------------------
# ConversationManager — DIRECTED
# ---------------------------------------------------------------------------


class TestDirectedPolicy:
    @pytest.mark.asyncio
    async def test_directed_chain(self):
        """Each speaker nominates the next via next_speaker metadata."""

        class DirectingAgent:
            def __init__(self, aid: str, next_id: str | None = None):
                self.agent_id = aid
                self._next_id = next_id
                self._last_metadata: dict = {}

            async def respond(self, topic, history, context=None):
                if self._next_id:
                    self._last_metadata = {"next_speaker": self._next_id}
                else:
                    self._last_metadata = {}
                return f"{self.agent_id} speaks"

        agent_a = DirectingAgent("a", next_id="b")
        agent_b = DirectingAgent("b", next_id="c")
        agent_c = DirectingAgent("c")  # chain ends here

        agents = [agent_a, agent_b, agent_c]
        config = ConversationConfig(turn_policy=TurnPolicy.DIRECTED, max_rounds=1)
        manager = ConversationManager(agents, config)
        result = await manager.run("test topic")

        speaker_ids = [m.agent_id for m in result.history.messages]
        assert speaker_ids == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_directed_stops_without_nomination(self):
        """Chain stops when the current speaker does not nominate a next speaker."""

        class NonDirectingAgent:
            def __init__(self, aid):
                self.agent_id = aid
                self._last_metadata: dict = {}

            async def respond(self, topic, history, context=None):
                return f"{self.agent_id} speaks"

        agents = [NonDirectingAgent("a"), NonDirectingAgent("b")]
        config = ConversationConfig(turn_policy=TurnPolicy.DIRECTED, max_rounds=1)
        manager = ConversationManager(agents, config)
        result = await manager.run("test topic")

        # Only the first agent speaks — no nomination to continue
        assert len(result.history.messages) == 1
        assert result.history.messages[0].agent_id == "a"


# ---------------------------------------------------------------------------
# ConversationManager — MODERATOR fallback
# ---------------------------------------------------------------------------


class TestModeratorPolicy:
    @pytest.mark.asyncio
    async def test_moderator_speaks_first(self):
        """Moderator's message appears before other agents' messages."""
        order: list[str] = []

        class OrderAgent:
            def __init__(self, aid):
                self.agent_id = aid
                self._last_metadata: dict = {}

            async def respond(self, topic, history, context=None):
                order.append(self.agent_id)
                return f"{self.agent_id} speaks"

        agents = [OrderAgent("a"), OrderAgent("mod"), OrderAgent("b")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.MODERATOR,
            max_rounds=1,
            moderator_id="mod",
        )
        manager = ConversationManager(agents, config)
        await manager.run("test topic")

        assert order[0] == "mod"

    @pytest.mark.asyncio
    async def test_moderator_fallback_when_not_configured(self):
        """Falls back to round_robin when no moderator is configured."""
        agents = [StubAgent("a"), StubAgent("b")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.MODERATOR,
            max_rounds=1,
            moderator_id=None,
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("test topic")

        # Fallback to round_robin means both agents speak
        assert len(result.history.messages) == 2

    @pytest.mark.asyncio
    async def test_moderator_fallback_when_id_not_in_agents(self):
        """Falls back to round_robin when moderator_id doesn't match any agent."""
        agents = [StubAgent("a"), StubAgent("b")]
        config = ConversationConfig(
            turn_policy=TurnPolicy.MODERATOR,
            max_rounds=1,
            moderator_id="ghost",
        )
        manager = ConversationManager(agents, config)
        result = await manager.run("test topic")

        assert len(result.history.messages) == 2


# ---------------------------------------------------------------------------
# ConversationHistory.get_agent_messages
# ---------------------------------------------------------------------------


class TestGetAgentMessages:
    def test_returns_only_that_agents_messages(self):
        h = ConversationHistory(topic="test")
        h.add(ConversationMessage(agent_id="a", content="msg1", round_num=1))
        h.add(ConversationMessage(agent_id="b", content="msg2", round_num=1))
        h.add(ConversationMessage(agent_id="a", content="msg3", round_num=2))

        msgs = h.get_agent_messages("a")
        assert len(msgs) == 2
        assert all(m.agent_id == "a" for m in msgs)

    def test_returns_empty_for_unknown_agent(self):
        h = ConversationHistory(topic="test")
        h.add(ConversationMessage(agent_id="a", content="hi", round_num=1))

        assert h.get_agent_messages("unknown") == []


# ---------------------------------------------------------------------------
# Multi-round stance persistence
# ---------------------------------------------------------------------------


class TestMultiRoundStance:
    @pytest.mark.asyncio
    async def test_stance_preserved_across_rounds(self):
        """Stance set in round 1 should still be read correctly in round 2.

        Verifies that _get_response copies metadata rather than popping from
        the agent's dict in-place (which would reset stance to NEUTRAL in round 2).
        """
        agent = StubAgent("a", stance="firm", conviction=0.9)
        config = ConversationConfig(
            turn_policy=TurnPolicy.ROUND_ROBIN, max_rounds=2
        )
        manager = ConversationManager([agent], config)
        result = await manager.run("test topic")

        msgs = result.history.messages
        assert len(msgs) == 2
        assert msgs[0].stance == "firm"
        assert msgs[0].conviction == 0.9
        assert msgs[1].stance == "firm"   # NOT reset to "neutral"
        assert msgs[1].conviction == 0.9  # NOT reset to 0.5


# ---------------------------------------------------------------------------
# Duplicate agent_id validation
# ---------------------------------------------------------------------------


class TestDuplicateAgentId:
    def test_raises_on_duplicate_ids(self):
        with pytest.raises(ValueError, match="Duplicate agent_id"):
            ConversationManager([StubAgent("a"), StubAgent("a")])
