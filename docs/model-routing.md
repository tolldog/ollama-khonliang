# Model Routing

The model router selects which model handles a request **within** a role. While `BaseRouter` picks the role (researcher, narrator, etc.), `ModelRouter` picks the model size (3b, 7b, 70b) based on query complexity.

## Architecture

```text
User message
  → BaseRouter.route()     → role name (e.g., "researcher")
  → ModelRouter.select()   → model name (e.g., "llama3.2:3b")
  → role.handle()          → response
```

## ModelRouter

```python
from khonliang.routing import ModelRouter, ComplexityStrategy

router = ModelRouter(
    role_models={
        "researcher": ["llama3.2:3b", "qwen2.5:7b", "llama3.1:70b"],
        "narrator": ["llama3.2:3b", "llama3.1:8b"],
    },
    strategy=ComplexityStrategy(classifier_client=fast_client),
)

selection = await router.select("researcher", "What year was John born?")
# ModelSelection(model="llama3.2:3b", reason="complexity:simple", ...)
```

### With Health Tracking

Cooled-down models are automatically filtered:

```python
from khonliang import ModelHealthTracker

tracker = ModelHealthTracker()
router = ModelRouter(
    role_models={"researcher": ["small", "medium", "large"]},
    strategy=ComplexityStrategy(classifier_client=fast_client),
    health_tracker=tracker,
)

# If "small" is in cooldown, candidates become ["medium", "large"]
```

## Strategies

### StaticStrategy

Always uses the first candidate. No-op baseline for testing or explicit assignment:

```python
from khonliang.routing import StaticStrategy

strategy = StaticStrategy()
# Always returns candidates[0]
```

### ComplexityStrategy

Uses a fast classifier model to score prompt complexity as simple/medium/hard, then maps to the corresponding model tier:

```python
from khonliang.routing import ComplexityStrategy

strategy = ComplexityStrategy(
    classifier_client=fast_client,      # e.g., a 3b model
    classifier_model="llama3.2:3b",
)

# candidates=["llama3.2:3b", "qwen2.5:7b", "llama3.1:70b"]
# "What year was X born?"           → simple  → candidates[0] (3b)
# "Compare migration patterns"       → medium  → candidates[1] (7b)
# "Synthesize the complete history"   → hard    → candidates[2] (70b)
```

One extra ~100ms classifier call saves expensive model loads on simple queries. If classification fails, falls back to the cheapest model.

### CascadeStrategy

Try the cheapest model first. If confidence is low, escalate to the next tier. This is the FrugalGPT pattern:

```python
from khonliang.routing import CascadeStrategy

strategy = CascadeStrategy(
    client_factory=lambda model: pool.get_client_for_model(model),
    confidence_threshold=0.7,
    evaluator=my_evaluator,       # Optional BaseEvaluator
    max_escalations=2,
)
```

The cascade flow:

1. Generate with cheapest candidate
2. Evaluate confidence (via evaluator or heuristics)
3. If confidence >= threshold: return response
4. If not: escalate to next candidate and repeat
5. Stop at `max_escalations` regardless

**Key feature:** `CascadeStrategy` returns the generated text in `ModelSelection.generated_text`, so callers can reuse it without a redundant second generation:

```python
selection = await strategy.select("researcher", message, candidates)
if selection.generated_text:
    # Already have the response — no need to generate again
    return selection.generated_text
```

Without an evaluator, the strategy uses heuristics:

- Short responses (<20 chars) → low confidence (0.3)
- Hedging markers ("I'm not sure", "I don't have") → low confidence (0.4-0.6)
- Otherwise → high confidence (0.85)

## ModelSelection

All strategies return a `ModelSelection`:

```python
@dataclass
class ModelSelection:
    model: str                              # Selected model
    reason: str                             # Why (e.g., "complexity:simple")
    model_preferences: List[str] = []       # Fallback order for scheduler
    generated_text: Optional[str] = None    # Response if already generated (cascade)
```

The `model_preferences` list integrates with `LLMManager`'s scheduler — if the selected model isn't loaded, the scheduler can use an already-loaded alternative from the preferences list to avoid a model swap.

## Integration with BaseRole

Pass a `ModelRouter` to a role for automatic model selection:

```python
from khonliang.routing import ModelRouter, ComplexityStrategy

model_router = ModelRouter(
    role_models={"researcher": ["llama3.2:3b", "qwen2.5:7b"]},
    strategy=ComplexityStrategy(classifier_client=fast_client),
)

researcher = ResearcherRole(pool, tree=tree, model_router=model_router)

# In your role's handle():
async def handle(self, message, session_id, context=None):
    model = await self._select_model(message)  # Uses model_router if configured
    response, elapsed = await self._timed_generate(
        prompt=prompt, system=self.system_prompt, model=model
    )
```

If no `model_router` is configured, `_select_model()` returns `None` and the default model from `ModelPool` is used — zero overhead.
