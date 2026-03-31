# Gateway & Agents

The gateway provides inter-agent communication via Redis Streams, while the agent management system handles activation rules, capabilities, typed channels, and a shared blackboard for coordination.

## Blackboard

The simplest coordination mechanism — a shared in-memory key-value store where agents post findings and other agents read them. Entries auto-expire based on TTL.

```python
from khonliang.gateway.blackboard import Blackboard

board = Blackboard(default_ttl=120)  # Entries expire after 2 minutes

# Agents post to named sections
board.post("researcher", "findings", "roger_birth", "Roger Tolle born ~1642 in England")
board.post("researcher", "findings", "roger_migration", "Emigrated to CT ~1660s")
board.post("fact_checker", "issues", "date_gap", "20-year gap between birth and first record")

# Read a section
findings = board.read("findings")
# {"roger_birth": "Roger Tolle born ~1642...", "roger_migration": "Emigrated to CT..."}

# Read a specific key
birth = board.read("findings", key="roger_birth")
# {"roger_birth": "Roger Tolle born ~1642 in England"}

# Build context string for LLM injection
ctx = board.build_context(
    sections=["findings", "issues"],
    max_entries=10,
)
# [findings]
#   [roger_birth] (researcher): Roger Tolle born ~1642 in England
#   [roger_migration] (researcher): Emigrated to CT ~1660s
# [issues]
#   [date_gap] (fact_checker): 20-year gap between birth and first record

# Cleanup
board.clear_section("findings")
board.clear()
```

### With Roles

Pass a blackboard to a role and its entries are automatically appended to context:

```python
board = Blackboard(default_ttl=300)

researcher = ResearcherRole(pool, tree=tree, board=board)
narrator = NarratorRole(pool, tree=tree, board=board)

# Research findings posted by one role are visible to all roles
board.post("web_search", "context", "roger_history",
           "WikiTree shows Roger Tolle arrived in New Haven Colony")

# When narrator builds context, board entries are included automatically
```

## Agent Gateway

For distributed communication via Redis Streams:

```python
from khonliang.gateway.gateway import AgentGateway
from khonliang.gateway.messages import AgentMessage

gateway = AgentGateway(
    redis_url="redis://localhost:6379",
    stream_prefix="khonliang:",
    max_stream_len=1000,
)

await gateway.start()

# Register agents
gateway.register_agent("researcher", researcher_instance)
gateway.register_agent("fact_checker", fact_checker_instance)

# Send a message to an agent
msg = AgentMessage(
    agent_id="researcher",
    content="Look up Roger Tolle's parents",
    session_id="session_123",
)
await gateway.send("researcher", msg)

# Receive messages
messages, last_id = await gateway.receive("researcher", count=10)

# Metrics
metrics = gateway.metrics
print(f"Sent: {metrics.messages_sent}, Failed: {metrics.messages_failed}")

await gateway.stop()
```

### Graceful Degradation

The gateway is designed to fail-open. If Redis is unavailable, `receive()` returns empty results and `send()` returns `False`. Agents continue operating without inter-agent communication.

## Activation Rules

Control when agents are active:

```python
from khonliang.agents.activation import ActivationRule, ActivationMode, ActivationTracker

tracker = ActivationTracker()

# Always active
tracker.register("researcher", ActivationRule(mode=ActivationMode.ALWAYS))

# Scheduled (cron expression)
tracker.register("batch_processor", ActivationRule(
    mode=ActivationMode.SCHEDULED,
    schedule_cron="0 */6 * * *",  # Every 6 hours
))

# Rate-limited
tracker.register("web_searcher", ActivationRule(
    mode=ActivationMode.ALWAYS,
    cooldown_seconds=5,           # Min 5s between activations
    max_activations=100,          # Max 100 per window
    window_seconds=3600,          # Per hour
))

# Check and record
if tracker.is_active("web_searcher"):
    tracker.record_activation("web_searcher")
    # ... do work
```

## Capabilities

Declare what each agent can do for dynamic routing:

```python
from khonliang.agents.capabilities import AgentCapability, CapabilityRegistry

registry = CapabilityRegistry()

registry.register("researcher", [
    AgentCapability(
        agent_id="researcher",
        capability="person_lookup",
        description="Look up a person in the family tree",
        input_schema={"name": "str"},
    ),
    AgentCapability(
        agent_id="researcher",
        capability="date_search",
        description="Search for events by date range",
    ),
])

registry.register("narrator", [
    AgentCapability(
        agent_id="narrator",
        capability="write_narrative",
        description="Write a family narrative from tree data",
    ),
])

# Find agents that can do something
agents = registry.find_agents_for("person_lookup")
# ["researcher"]

# Generate LLM-friendly description of all capabilities
prompt_ctx = registry.describe_for_llm()
```

## Typed Channels

For structured pub/sub communication between agents:

```python
from khonliang.agents.channels import ChannelManager, ChannelMessage, MessageType

channels = ChannelManager()

# Create a channel
channels.create_channel(
    "research_findings",
    description="Completed research results",
    allowed_types=[MessageType.ANALYSIS, MessageType.SUMMARY],
)

# Subscribe agents
def on_finding(msg):
    print(f"New finding from {msg.sender}: {msg.content}")

channels.subscribe("research_findings", "narrator", on_finding)
channels.subscribe("research_findings", "fact_checker", on_finding)

# Publish
msg = ChannelMessage(
    channel="research_findings",
    message_type=MessageType.ANALYSIS,
    sender="researcher",
    content="Roger Tolle emigrated to Connecticut around 1642",
    data={"confidence": 0.85, "sources": ["wikitree", "ddg"]},
)
count = channels.publish("research_findings", msg)
# Notifies 2 subscribers

# History
recent = channels.get_history("research_findings", count=10, message_type=MessageType.ANALYSIS)
```

### Message Types

| Type        | Purpose                          |
| ----------- | -------------------------------- |
| `ANALYSIS`  | Research findings, data analysis |
| `VOTE`      | Consensus votes                  |
| `QUESTION`  | Agent asking for information     |
| `CHALLENGE` | Debate challenges                |
| `RESPONSE`  | Replies to questions/challenges  |
| `SIGNAL`    | Status signals, alerts           |
| `SUMMARY`   | Aggregated summaries             |
| `ALERT`     | Urgent notifications             |

## Agent Envelope

For structured inter-agent messages with model metadata:

```python
from khonliang.agents.envelope import AgentEnvelope, ModelMeta

envelope = AgentEnvelope.create(
    from_role="researcher",
    from_agent_id="researcher_1",
    intent="person_lookup",
    payload={"name": "Roger Tolle", "scope": "genealogy"},
    model_meta=ModelMeta(
        model_name="llama3.2:3b",
        inference_ms=450,
        prompt_tokens=120,
        completion_tokens=85,
    ),
)

# Create a reply linked to the original
reply = AgentEnvelope.reply(
    original=envelope,
    from_role="fact_checker",
    from_agent_id="fact_checker_1",
    intent="verification_result",
    payload={"verified": True, "confidence": 0.9},
)
print(reply.correlation_id == envelope.envelope_id)  # True
```

## Session Helpers

Convenience functions for session-scoped communication:

```python
from khonliang.gateway.sessions import (
    sessions_list, sessions_history, sessions_send, challenge_agent
)

# List active sessions
active = sessions_list(gateway)

# Get message history for a session
history = await sessions_history(gateway, session_id="abc123", agent_id="researcher", count=20)

# Send in session context
await sessions_send(gateway, session_id="abc123", agent_id="researcher",
                    content="Look up migration patterns")

# Challenge an agent and wait for reply
reply = await challenge_agent(
    gateway, agent_id="fact_checker",
    challenge="Verify: Roger Tolle born 1642",
    session_id="abc123",
    timeout=30.0,
)
```

## Observers

React to gateway events without modifying agent logic:

```python
from khonliang.gateway.observer import LogObserver, WebhookObserver, CallbackObserver

# Log all events
gateway.add_observer(LogObserver(name="logger", level="INFO"))

# POST events to a webhook
gateway.add_observer(WebhookObserver(
    name="slack_notifier",
    url="https://hooks.slack.com/...",
    event_types=["alert", "error"],
))

# Custom callback
def on_event(event):
    if event.get("type") == "research_complete":
        print(f"Research done: {event['data']}")

gateway.add_observer(CallbackObserver(name="my_handler", callback=on_event))
```
