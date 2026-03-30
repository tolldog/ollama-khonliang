"""
Generic LLM query parser — extracts structured parameters from natural language.

Given a schema describing available fields and example queries, uses a
fast LLM to parse user input into structured parameters. Falls back to
regex extraction when the LLM is unavailable.

This is the generic base — domain-specific parsers extend it by providing
their own schema and examples.

Example:
    # Generic usage
    parser = QueryParser(
        client=ollama_client,
        schema={
            "name": {"type": "string", "description": "Person name"},
            "date": {"type": "string", "description": "Date or year"},
        },
        domain="genealogy research",
        examples=[
            ('find Roger Tolle', '{"name": "Roger Tolle"}'),
            ('records from 1850', '{"date": "1850"}'),
        ],
    )
    result = await parser.parse("look up Roger Tolle from 1642")
    # {"name": "Roger Tolle", "date": "1642"}
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryParser:
    """
    LLM-backed structured query extraction.

    Configurable for any domain via schema + examples. The schema
    describes available output fields, and examples teach the model
    the expected mapping from natural language to JSON.

    Args:
        client: OllamaClient instance (or any object with async generate())
        model: Model to use for parsing (fast model recommended)
        schema: Dict of field_name -> {type, description, enum?}
        domain: Short description of the domain (for system prompt)
        examples: List of (input, output_json_str) tuples
        fallback: Optional regex fallback function(message) -> dict
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "llama3.2:3b",
        schema: Optional[Dict[str, Dict[str, Any]]] = None,
        domain: str = "",
        examples: Optional[List[Tuple[str, str]]] = None,
        fallback: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self.client = client
        self.model = model
        self.schema = schema or {}
        self.domain = domain
        self.examples = examples or []
        self.fallback = fallback
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt from schema and examples."""
        lines = [
            f"You are a query parser for a {self.domain} service."
            if self.domain
            else "You are a query parser.",
            "Given a user's natural language request, extract structured parameters.",
            "",
            "Available fields:",
        ]

        for name, spec in self.schema.items():
            desc = spec.get("description", "")
            ftype = spec.get("type", "string")
            enum = spec.get("enum")
            samples = spec.get("sample_values")
            line = f"- {name} ({ftype}): {desc}"
            if enum:
                line += f" — values: {', '.join(str(e) for e in enum)}"
            elif samples:
                line += f" — e.g. {', '.join(str(s) for s in samples[:4])}"
            lines.append(line)

        lines.append("")
        lines.append(
            "Respond with ONLY a JSON object containing the extracted fields."
        )
        lines.append("Only include fields clearly stated or implied.")
        lines.append("Omit fields not mentioned.")

        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for query, output in self.examples:
                lines.append(f'"{query}"')
                lines.append(f"→ {output}")
                lines.append("")

        return "\n".join(lines)

    async def parse(self, message: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured parameters.

        Tries LLM first, falls back to regex/custom fallback.
        """
        if self.client:
            try:
                result = await self._parse_llm(message)
                if result is not None:
                    return result
            except (TimeoutError, ConnectionError, ValueError) as e:
                logger.debug(f"LLM parse failed: {e}")

        if self.fallback:
            return self.fallback(message)

        return {}

    async def _parse_llm(self, message: str) -> Optional[Dict[str, Any]]:
        """Use the fast model to parse.

        Prefers generate_json() when available on the client for native
        JSON-mode support; otherwise falls back to generate() + manual
        extraction.
        """
        prompt = f'User query: "{message}"\n\nExtract parameters:'

        if hasattr(self.client, "generate_json"):
            try:
                data = await self.client.generate_json(
                    prompt=prompt,
                    system=self._system_prompt,
                    model=self.model,
                    temperature=0.1,
                    max_tokens=200,
                )
                if isinstance(data, dict):
                    result = {k: v for k, v in data.items() if v is not None}
                    if self.schema:
                        result = {k: v for k, v in result.items() if k in self.schema}
                    return result
            except Exception:
                logger.debug("generate_json unavailable or failed, falling back to generate")

        response = await self.client.generate(
            prompt=prompt,
            system=self._system_prompt,
            model=self.model,
            temperature=0.1,
            max_tokens=200,
        )

        return self._extract_json(response)

    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from LLM response text.

        Strips null values and filters to schema keys when a schema is set.
        """
        text = response.strip()

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start: i + 1])
                        result = {k: v for k, v in data.items() if v is not None}
                        if self.schema:
                            result = {k: v for k, v in result.items() if k in self.schema}
                        return result
                    except json.JSONDecodeError:
                        return None
        return None

    @property
    def system_prompt(self) -> str:
        """The generated system prompt (useful for debugging)."""
        return self._system_prompt
