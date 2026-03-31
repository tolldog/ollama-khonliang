# Parsing

Two parsing modules for extracting structured data from LLM responses and user queries.

## StructuredBlockParser

Extracts typed JSON blocks from LLM markdown responses. LLMs often wrap structured output in code fences — this parser handles the common formats.

````python
from khonliang.parsing.structured import StructuredBlockParser

# Define what you expect
parser = StructuredBlockParser(
    fence_name="analysis",          # Look for ```analysis blocks
    items_key="findings",           # Extract the "findings" array
    item_factory=dict,              # Convert items to dicts
    valid_actions=["verify", "research", "ignore"],
)

# Parse an LLM response
response = '''
Based on the tree data, here are my findings:

```analysis
{
  "findings": [
    {"name": "Roger Tolle", "action": "verify", "detail": "Birth date uncertain"},
    {"name": "Timothy Toll", "action": "research", "detail": "Missing mother's maiden name"}
  ],
  "confidence": 0.85
}
````

'''

result = parser.parse(response)
if result:
for item in result.items:
print(f"{item['name']}: {item['action']} — {item['detail']}")
print(f"Metadata: {result.metadata}")

````

### Search Order

The parser tries to find JSON in this order:
1. Custom fence (e.g., ` ```analysis `)
2. Generic ` ```json ` fence
3. Raw JSON (first `{...}` block in the response)

### LLM JSON Cleanup

LLMs frequently produce invalid JSON. The parser auto-fixes:
- Python booleans: `True`/`False`/`None` → `true`/`false`/`null`
- Trailing commas: `[1, 2, 3,]` → `[1, 2, 3]`
- Single-line comments: `// this is a comment` → removed
- Unquoted keys (common with smaller models)

## QueryParser

LLM-backed natural language to structured query parameters. Uses a fast model to extract filters from conversational queries.

```python
from khonliang.parsing.query_parser import QueryParser

parser = QueryParser(
    client=ollama_client,           # OllamaClient instance
    model="llama3.2:3b",           # Fast model for extraction
    domain="genealogy",
    schema={
        "name": {"type": "str", "description": "Person name to search for"},
        "birth_year_min": {"type": "int", "description": "Earliest birth year"},
        "birth_year_max": {"type": "int", "description": "Latest birth year"},
        "birth_place": {"type": "str", "description": "Birth location"},
        "gender": {"type": "str", "description": "Gender (male/female)"},
    },
    examples=[
        ("find men from Ohio", '{"gender": "male", "birth_place": "Ohio"}'),
        ("Tolls born before 1900", '{"name": "Toll", "birth_year_max": 1900}'),
    ],
)

# Parse natural language
params = await parser.parse("find all men born in Ohio before 1920")
# {"gender": "male", "birth_place": "Ohio", "birth_year_max": 1920}

params = await parser.parse("women named Toll from the 1800s")
# {"name": "Toll", "gender": "female", "birth_year_min": 1800, "birth_year_max": 1899}
````

### Fallback

If the LLM is unavailable, the parser falls back to regex extraction or a custom fallback function:

```python
def regex_fallback(message):
    """Extract what we can without an LLM."""
    params = {}
    # Year extraction
    year_match = re.search(r"before (\d{4})", message)
    if year_match:
        params["birth_year_max"] = int(year_match.group(1))
    # Gender
    if "men" in message.lower() or "male" in message.lower():
        params["gender"] = "male"
    return params

parser = QueryParser(
    client=None,  # No LLM
    model="",
    domain="genealogy",
    schema={...},
    fallback=regex_fallback,
)
```

### System Prompt

The parser auto-generates a system prompt from your schema and examples:

```python
print(parser.system_prompt)
# You are a query parameter extractor for genealogy.
# Extract structured parameters from natural language queries.
#
# Schema:
#   name (str): Person name to search for
#   birth_year_min (int): Earliest birth year
#   ...
#
# Examples:
#   "find men from Ohio" → {"gender": "male", "birth_place": "Ohio"}
#   ...
#
# Return ONLY valid JSON. Include only fields mentioned in the query.
```

### Genealogy Usage

The genealogy project uses `QueryParser` for the `!researchwho` command:

```python
# User types: !researchwho males born in ohio before 1920
# QueryParser extracts: {"gender": "male", "birth_place": "Ohio", "birth_year_max": 1920}
# Tree is filtered by these criteria
# Matching persons are batch-researched via the research pool
```
