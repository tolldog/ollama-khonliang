# Training & Feedback

The training module captures agent interactions, collects user feedback, and extracts heuristics from outcomes. This data can be used for RLHF-style fine-tuning, RAG improvement, or agent weight calibration.

## Feedback Store

SQLite-backed storage for interactions and ratings:

```python
from khonliang.training.feedback import FeedbackStore

feedback = FeedbackStore("data/knowledge.db")

# Log an agent interaction
interaction_id = feedback.log_interaction(
    message="Who were Timothy's grandparents?",
    role="researcher",
    route_reason="keyword: grandparents",
    response="Based on the tree data, Timothy's grandparents were...",
    generation_ms=1200,
    session_id="abc123",
)

# User rates the response
feedback.add_feedback(
    interaction_id=interaction_id,
    rating=4,           # 1-5 scale
    feedback="Good answer but missed the maternal side",
    expected="Should have included both paternal and maternal grandparents",
)
```

### Rating Scale

| Rating | Meaning   | Training Use                     |
| ------ | --------- | -------------------------------- |
| 5      | Excellent | Strong positive example          |
| 4      | Good      | Positive example                 |
| 3      | Okay      | Neutral (excluded from training) |
| 2      | Poor      | Negative example                 |
| 1      | Wrong     | Strong negative example          |

### Feedback in the Chat Server

The genealogy project collects feedback via WebSocket:

```python
# Client sends:
# {"type": "feedback", "message_id": "abc123", "rating": 4}

# Server handles in ChatServer._handle_feedback():
if rating >= 4:
    # High rating → boost confidence on related knowledge entries
    for entry_id in exchange.get("knowledge_entry_ids", []):
        librarian.store.update_confidence(entry_id, rating / 5.0)
```

The `/rate` command in the CLI provides the same:

```
> Tell me about Timothy Toll
[researcher] Timothy Toll was born in 1842 in Ohio...
> /rate 5
Feedback recorded. Thanks!
```

## Training Data Export

Export interaction data for fine-tuning:

```python
from khonliang.training.exporter import TrainingExporter

# Point the exporter at the same feedback database
exporter = TrainingExporter(db_path="data/knowledge.db")

# Collect supervised fine-tuning examples (e.g., rating >= 4)
examples = exporter.collect(min_rating=4, limit=1000)
# [TrainingExample(prompt="...", completion="...", system="...", rating=5), ...]

# Export as JSONL for fine-tuning (supports alpaca, sharegpt, completion formats)
exporter.export("training_data.jsonl", fmt="alpaca", min_rating=4)

# Check export stats
print(exporter.stats())
```

## Indexing Feedback into RAG

High-quality feedback can be auto-indexed into the knowledge store:

```python
# Index all high-rated, unindexed feedback into knowledge
stats = feedback.index_into_rag(min_rating=4)
# {"indexed": 12, "skipped": 3, "errors": 0}

# Check overall feedback statistics
feedback_stats = feedback.get_stats()
# {"interactions": 142, "by_role": {...}, "feedback": {"total": 89, "indexed": 45, "unindexed": 44}}
```

## Heuristic Discovery

`HeuristicPool` discovers patterns from recorded outcomes — what actions lead to success or failure:

```python
from khonliang.training.heuristics import HeuristicPool

pool = HeuristicPool(db_path="data/knowledge.db")

# Record outcomes as they happen
pool.record_outcome(
    action="cite_tree_data",
    result="success",
    context={"role": "researcher", "query_type": "person_lookup"},
    source="feedback",
)

# Extract heuristics from accumulated outcomes
heuristics = pool.extract(min_samples=10, min_confidence=0.6)
# [Heuristic(rule="cite_tree_data → success", confidence=0.89, sample_count=45), ...]

# Get existing heuristics
rules = pool.get_heuristics(min_confidence=0.5, limit=20)

# Build context for LLM prompt injection
prompt_ctx = pool.build_prompt_context(max_rules=5, min_confidence=0.5)
# "Based on past interactions:\n- cite_tree_data → success (89%)\n- ..."

# Decay old heuristic confidence over time
decayed = pool.apply_decay()
```

## Feedback Loop Architecture

The complete feedback loop in the genealogy project:

```text
User Query
  → Router → Role → LLM Response
  → Auto-index response as Tier 3 knowledge
  → Self-evaluator checks against tree data
    → If uncertain: queue background research
    → If date mismatch: flag and research
  → User sees response (with caveat if issues found)
  → User rates response (/rate 1-5)
    → High rating: boost knowledge confidence
    → Low rating: flag for review
  → Training exporter collects rated examples
  → HeuristicPool discovers success/failure patterns
  → Patterns inform system prompt refinement
```

Each piece feeds the next:

1. **Knowledge store** improves context for future queries
2. **Evaluation** catches errors before the user sees them
3. **Research pool** fills gaps the agent couldn't answer
4. **Feedback** calibrates confidence scores
5. **Heuristics** reveal what makes good vs bad responses
