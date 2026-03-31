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

exporter = TrainingExporter(feedback_store=feedback)

# Export positive examples (rating >= 4)
examples = exporter.export_positive(min_rating=4, limit=1000)
# [{"prompt": "...", "completion": "...", "system": "...", "rating": 5}, ...]

# Export as JSONL for fine-tuning
exporter.export_jsonl("training_data.jsonl", min_rating=4)

# Export negative examples for DPO/RLHF
pairs = exporter.export_preference_pairs(min_gap=2)
# [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]
```

## Unindexed Feedback

Feedback entries that haven't been indexed into RAG yet:

```python
# Get unprocessed feedback
unindexed = feedback.list_unindexed(limit=50)

for fb in unindexed:
    # Index into knowledge store
    if fb.rating >= 4:
        librarian.index_response(
            content=fb.response,
            title=f"Rated response: {fb.prompt[:60]}",
            agent_id="training",
            query=fb.prompt,
            confidence=fb.rating / 5.0,
        )

    # Mark as processed
    feedback.mark_indexed(fb.id)
```

## Heuristic Extraction

Discover patterns from interaction outcomes:

```python
from khonliang.training.heuristics import HeuristicExtractor

extractor = HeuristicExtractor(feedback_store=feedback)

# Find patterns in high-rated responses
patterns = extractor.extract_positive_patterns(min_rating=4, min_count=5)
# [
#   {"pattern": "cites_tree_data", "frequency": 0.89, "avg_rating": 4.6},
#   {"pattern": "mentions_uncertainty", "frequency": 0.72, "avg_rating": 4.3},
# ]

# Find patterns in low-rated responses
anti_patterns = extractor.extract_negative_patterns(max_rating=2, min_count=3)
# [
#   {"pattern": "fabricated_details", "frequency": 0.65, "avg_rating": 1.8},
#   {"pattern": "missing_sources", "frequency": 0.58, "avg_rating": 2.0},
# ]

# Route-level analysis — which roles perform best
route_stats = extractor.analyze_by_route()
# {"researcher": {"avg_rating": 4.1, "count": 89}, "narrator": {"avg_rating": 3.8, "count": 34}}
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
  → Heuristic extractor finds success/failure patterns
  → Patterns inform system prompt refinement
```

Each piece feeds the next:

1. **Knowledge store** improves context for future queries
2. **Evaluation** catches errors before the user sees them
3. **Research pool** fills gaps the agent couldn't answer
4. **Feedback** calibrates confidence scores
5. **Heuristics** reveal what makes good vs bad responses
