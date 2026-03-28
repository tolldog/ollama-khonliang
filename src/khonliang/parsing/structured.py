"""
Structured block parser — extracts typed JSON from LLM responses.

LLMs often embed structured data inside fenced code blocks alongside
prose. This module finds those blocks, cleans common LLM JSON issues
(comments, comma-formatted numbers, trailing commas), and returns
typed dataclass instances.

Usage — define your schema, tell the parser what fence name to look for:

    parser = StructuredBlockParser(
        fence_name="helpdesk-action",
        item_class=TicketAction,
        items_key="actions",
        valid_actions=["escalate", "resolve", "reassign"],
    )
    result = parser.parse(llm_response)
    if result:
        for action in result.items:
            print(action.action, action.target)

The LLM would emit:
    Here is my recommended action:
    ```helpdesk-action
    {
      "actions": [
        {"action": "escalate", "target": "tier2", "reason": "needs DB access"}
      ]
    }
    ```

For custom domains, configure fence_name, items_key, and valid_actions to match
your schema. E.g. fence_name="review-action", valid_actions=["approve", "reject"].
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Common LLM JSON cleanup patterns
_CLEAN_PATTERNS = [
    (re.compile(r"//[^\n]*"), ""),                             # Strip // comments
    (re.compile(r"(?<=\d),(?=\d{3}(?:[.\s,}\]]|$))"), ""),   # 1,008 → 1008
    (re.compile(r",\s*([}\]])"), r"\1"),                       # trailing commas
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
]


def _clean_json(text: str) -> str:
    for pattern, replacement in _CLEAN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _parse_json_safe(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(_clean_json(text))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed after cleanup: {e}")
            return None


@dataclass
class ParsedBlock:
    """Result of parsing a structured block from an LLM response."""

    items: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""  # Original response with block stripped

    def to_dict(self) -> dict:
        return {
            "items": [asdict(i) if hasattr(i, "__dataclass_fields__") else i for i in self.items],
            "metadata": self.metadata,
        }


class StructuredBlockParser:
    """
    Extracts and validates a typed JSON block from LLM output.

    Searches in order:
    1. Fenced block with the configured fence name (e.g. ```my-fence ... ```)
    2. Generic ```json block containing the items_key
    3. Raw JSON object containing the items_key

    Args:
        fence_name:     Name of the custom fence to look for
        items_key:      JSON key holding the list of items (e.g. "trades", "actions")
        item_factory:   Callable that builds an item from a dict. Defaults to dict passthrough.
        valid_actions:  If set, items with action not in this list are dropped
        action_key:     Key name for the action field (default "action")
        id_key:         Key name for the identifier field (default "id" or first key)
        strip_block:    Remove the matched block from raw_response
    """

    def __init__(
        self,
        fence_name: str,
        items_key: str = "items",
        item_factory: Optional[Callable[[Dict], Any]] = None,
        valid_actions: Optional[List[str]] = None,
        action_key: str = "action",
        strip_block: bool = True,
    ):
        self.fence_name = fence_name
        self.items_key = items_key
        self.item_factory = item_factory or (lambda d: d)
        self.valid_actions = {a.lower() for a in valid_actions} if valid_actions else None
        self.action_key = action_key
        self.strip_block = strip_block

        self._fence_re = re.compile(
            rf"```{re.escape(fence_name)}\s*\n(.*?)\n\s*```",
            re.DOTALL,
        )
        self._json_fence_re = re.compile(
            rf"```json\s*\n(\{{.*?{re.escape(items_key)}.*?\}})\s*\n```",
            re.DOTALL,
        )
        self._raw_re = re.compile(
            rf'(\{{[^{{}}]*"{re.escape(items_key)}"\s*:\s*\[.*?\]\s*[^{{}}]*\}})',
            re.DOTALL,
        )

    def parse(self, response: str) -> Optional[ParsedBlock]:
        """
        Extract and parse a structured block from an LLM response.

        Returns ParsedBlock if a valid block is found, None otherwise.
        """
        json_text, matched_pattern = self._find_block(response)
        if json_text is None:
            return None

        data = _parse_json_safe(json_text)
        if data is None:
            return None

        raw_items = data.get(self.items_key, [])
        if not isinstance(raw_items, list):
            logger.warning(f"'{self.items_key}' is not a list")
            return None

        items = []
        for raw in raw_items:
            item = self._parse_item(raw)
            if item is not None:
                items.append(item)

        if not items:
            logger.warning(f"No valid items found in {self.fence_name} block")
            return None

        metadata = {k: v for k, v in data.items() if k != self.items_key}
        raw_response = self._strip(response) if self.strip_block else response

        return ParsedBlock(items=items, metadata=metadata, raw_response=raw_response)

    def strip(self, response: str) -> str:
        """Remove all matching blocks from a response string."""
        return self._strip(response)

    def _find_block(self, response: str):
        m = self._fence_re.search(response)
        if m:
            return m.group(1).strip(), "fence"

        m = self._json_fence_re.search(response)
        if m:
            logger.debug("Using ```json block fallback")
            return m.group(1).strip(), "json_fence"

        m = self._raw_re.search(response)
        if m:
            logger.debug("Using raw JSON fallback")
            return m.group(1).strip(), "raw"

        return None, None

    def _parse_item(self, data: dict) -> Optional[Any]:
        if not isinstance(data, dict):
            return None

        if self.valid_actions is not None:
            action = str(data.get(self.action_key, "")).lower().strip()
            if action not in self.valid_actions:
                logger.debug(f"Skipping item with invalid action: {action!r}")
                return None

        try:
            return self.item_factory(data)
        except Exception as e:
            logger.warning(f"Item factory failed: {e}")
            return None

    def _strip(self, response: str) -> str:
        cleaned = self._fence_re.sub("", response)
        cleaned = self._json_fence_re.sub("", cleaned)
        cleaned = self._raw_re.sub("", cleaned)
        return cleaned.strip()
