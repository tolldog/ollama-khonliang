# Consensus & Voting

The consensus system runs multiple agents in parallel on the same question, collects their votes, and aggregates them into a decision. This is useful when you want multiple perspectives or need confidence that an answer is correct.

## Core Concepts

- **AgentTeam** — runs N agents in parallel via `asyncio.gather`
- **AgentVote** — each agent's decision (APPROVE/REJECT/DEFER/VETO) with confidence and reasoning
- **ConsensusEngine** — aggregates votes using weighted scoring
- **DebateOrchestrator** — optional structured debate when agents disagree

## Agent Votes

```python
from khonliang.consensus.models import AgentVote, AgentAction

vote = AgentVote(
    agent_id="analyst",
    action=AgentAction.APPROVE,
    confidence=0.85,
    reasoning="Dates are consistent with historical records",
    factors={"date_check": True, "source_quality": "high"},
    weight=1.0,
)

# Weighted score = confidence * weight
print(vote.weighted_score())  # 0.85
```

### Actions

| Action    | Meaning                          |
| --------- | -------------------------------- |
| `APPROVE` | Agrees with the proposal         |
| `REJECT`  | Disagrees with the proposal      |
| `DEFER`   | Not enough information to decide |
| `VETO`    | Blocks the decision entirely     |

## Agent Team

Any object with `agent_id: str` and `async analyze(...) -> AgentVote` can participate:

```python
from khonliang.consensus.team import AgentTeam
from khonliang.consensus.engine import ConsensusEngine

class FactCheckAgent:
    agent_id = "fact_checker"

    async def analyze(self, subject, context=None):
        # Run your LLM, check facts, return a vote
        return AgentVote(
            agent_id=self.agent_id,
            action=AgentAction.APPROVE,
            confidence=0.9,
            reasoning="All dates verified against tree data",
        )

class SkepticAgent:
    agent_id = "skeptic"

    async def analyze(self, subject, context=None):
        return AgentVote(
            agent_id=self.agent_id,
            action=AgentAction.REJECT,
            confidence=0.6,
            reasoning="Birth date seems too early for this region",
        )

# Create team
engine = ConsensusEngine(
    agent_weights={"fact_checker": 1.5, "skeptic": 1.0},
    veto_blocks=True,
    min_confidence=0.3,
)

team = AgentTeam(
    agents=[FactCheckAgent(), SkepticAgent()],
    consensus_engine=engine,
    agent_timeout=30.0,
)

# Run evaluation
result = await team.evaluate(
    subject="Roger Tolle was born in 1642 in England",
    context={"source": "web_search"},
)

print(f"Action: {result.action}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Reason: {result.reason}")
```

## Consensus Engine

The engine aggregates votes:

```python
engine = ConsensusEngine(
    agent_weights={
        "fact_checker": 1.5,   # More trusted
        "skeptic": 1.0,
        "advocate": 0.8,
    },
    veto_blocks=True,       # Any VETO vote blocks the decision
    min_confidence=0.3,     # Ignore votes below this threshold
)

result = engine.calculate_consensus(votes)
```

### Scoring

Each vote contributes `confidence * weight` to its action's score. Scores are normalized by total weight. The highest-scoring action wins.

### Veto

If `veto_blocks=True`, any VETO vote immediately returns `action="VETO"` regardless of other votes. This is useful for safety-critical decisions.

### Judge Function

Override the consensus with a custom judge:

```python
def my_judge(votes):
    """Custom judge that requires unanimous approval."""
    if all(v.action == AgentAction.APPROVE for v in votes):
        return None  # Accept consensus
    # Override with a custom vote
    return AgentVote(
        agent_id="judge",
        action=AgentAction.DEFER,
        confidence=0.5,
        reasoning="Not unanimous — deferring for review",
    )

engine = ConsensusEngine(judge_fn=my_judge)
```

## Multi-Round Deliberation

Run multiple rounds where each round's summary is injected as context for the next:

```python
def summarize(votes, round_num):
    """Summarize the current round for the next."""
    lines = [f"Round {round_num} summary:"]
    for v in votes:
        lines.append(f"  {v.agent_id}: {v.action.value} ({v.confidence:.0%}) — {v.reasoning}")
    return "\n".join(lines)

team = AgentTeam(
    agents=[FactCheckAgent(), SkepticAgent(), AdvocateAgent()],
    consensus_engine=engine,
    rounds=3,                    # Up to 3 deliberation rounds
    summarizer_fn=summarize,     # Required for multi-round
)

result = await team.evaluate("Roger Tolle born 1642")
print(f"Debate rounds: {result.debate_rounds}")
```

## Caching

Results are cached by key + input hash with configurable TTL:

```python
team = AgentTeam(
    agents=[...],
    consensus_engine=engine,
    enable_caching=True,
    cache_ttl_seconds=300,  # 5 minute cache
)

# First call runs all agents
result1 = await team.evaluate("Roger Tolle", cache_key="tolle_check")

# Second call (within 5 min) returns cached result
result2 = await team.evaluate("Roger Tolle", cache_key="tolle_check")
```

## Adaptive Weighting

Automatically adjust agent weights based on recent performance:

```python
from khonliang.consensus.weights import AdaptiveWeightManager

weight_mgr = AdaptiveWeightManager()

# After tracking performance over time:
new_weights = weight_mgr.calculate_weights(
    performances={
        "fact_checker": AgentPerformance(accuracy=0.92, ...),
        "skeptic": AgentPerformance(accuracy=0.78, ...),
    },
    regime="cautious",  # Boost risk-related agents
)
# {"fact_checker": 0.35, "skeptic": 0.25, ...}
```

Weight bounds enforce fairness: no agent drops below 5% or exceeds 40%.

## Debate Orchestrator

When agents strongly disagree, trigger a structured debate:

```python
from khonliang.debate.orchestrator import DebateOrchestrator, DebateConfig

debate = DebateOrchestrator(
    agents={"fact_checker": fact_agent, "skeptic": skeptic_agent},
    config=DebateConfig(
        disagreement_threshold=0.6,  # Trigger on high-confidence disagreement
        max_rounds=3,
        challenge_timeout=30.0,
    ),
)

# Detect if debate is needed
pair = debate.detect_disagreement(votes)
if pair:
    challenger, target = pair
    # Run structured challenge/response
    refined_votes = await debate.run_debate(votes, subject="Roger Tolle born 1642")
```

### Assigned Stances

Force agents to argue specific positions (useful for exploring edge cases):

```python
config = DebateConfig(
    assigned_stances={"fact_checker": "support", "skeptic": "oppose"},
)
```

## Custom Vocabularies

Define custom action sets for domain-specific decisions:

```python
from khonliang.consensus.vocabulary import ActionVocabulary

genealogy_vocab = ActionVocabulary(
    actions=["VERIFIED", "UNVERIFIED", "CONTRADICTED", "INSUFFICIENT_DATA"],
    blocking=["CONTRADICTED"],
    default_weights={"VERIFIED": 1.0, "UNVERIFIED": 0.5, "CONTRADICTED": 1.0, "INSUFFICIENT_DATA": 0.3},
)
```
