"""
Flow intent classifier — LLM-based classification within conversation flows.

When a user is mid-conversation (e.g. they just received a suggestion or plan),
this classifier decides what they want to do next: save it, execute it, update
it, get more detail, or something else.

Runs on a fast CPU-only model (llama3.2:3b recommended) to avoid GPU contention.

Usage:

    # Define your flow types and their prompts
    classifier = FlowClassifier(
        ollama_url="http://localhost:11434",
        flow_prompts={
            "report": \"\"\"The user just received a generated report.
    Classify: SAVE (keep it), REVISE (change something), EXPLAIN (more detail), OTHER.
    Reply JSON only: {{"action": "...", "confidence": 0.0-1.0, "reasoning": "..."}}\n
    User message: "{message}"\"\"\",
        },
        valid_actions=["save", "revise", "explain", "other"],
    )
    result = classifier.classify("looks good, save it", flow_type="report")
    print(result.action, result.confidence)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FlowAction(str, Enum):
    """Default flow actions. Override by passing valid_actions to FlowClassifier."""
    SAVE = "save"
    EXECUTE = "execute"
    UPDATE = "update"
    EXPLAIN = "explain"
    OTHER = "other"


@dataclass
class FlowClassification:
    """Result of classifying a user message within a flow context."""
    action: str
    confidence: float
    reasoning: str


# Built-in prompt templates for common flow types.
# Pass your own via flow_prompts= to override or extend.
DEFAULT_FLOW_PROMPTS: Dict[str, str] = {
    "suggestion": """You are a conversation flow classifier.
The user received a list of suggestions from the assistant.

Classify their response:
SAVE    — store/keep the suggestion for later
EXECUTE — act on it now (confirm, go ahead, do it)
UPDATE  — modify before deciding (change something, more/less aggressive)
EXPLAIN — wants more detail or reasoning
OTHER   — unrelated or topic change

Reply with ONLY JSON:
{{"action": "save|execute|update|explain|other", "confidence": 0.0-1.0, "reasoning": "..."}}

User message: "{message}"
""",
    "plan": """You are a conversation flow classifier.
The user received a generated plan from the assistant.

Classify their response:
SAVE    — keep the plan for reference
UPDATE  — change or refine the plan
EXPLAIN — wants more detail about the plan
OTHER   — unrelated or topic change

Reply with ONLY JSON:
{{"action": "save|update|explain|other", "confidence": 0.0-1.0, "reasoning": "..."}}

User message: "{message}"
""",
}


class FlowClassifier:
    """
    Classifies user intent within an active conversation flow.

    Args:
        ollama_url:    Ollama server URL
        model:         Model to use (fast 3b model recommended)
        flow_prompts:  Dict of flow_type → prompt template (use {message} placeholder)
        valid_actions: Set of valid action strings (lowercased)
        fallback:      Action returned when classification fails
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        flow_prompts: Optional[Dict[str, str]] = None,
        valid_actions: Optional[list] = None,
        fallback: str = "other",
    ):
        self.ollama_url = ollama_url
        self.model = model
        self._prompts = {**DEFAULT_FLOW_PROMPTS, **(flow_prompts or {})}
        self._valid_actions = {a.lower() for a in valid_actions} if valid_actions else None
        self.fallback = fallback

    def classify(self, message: str, flow_type: str) -> FlowClassification:
        """
        Synchronously classify a message within a flow type.

        Args:
            message:   The user's message
            flow_type: Which flow context (must match a key in flow_prompts)

        Returns:
            FlowClassification with action, confidence, reasoning
        """
        prompt_template = self._prompts.get(flow_type)
        if not prompt_template:
            logger.warning(f"Unknown flow type: {flow_type!r}, returning fallback")
            return FlowClassification(action=self.fallback, confidence=1.0,
                                       reasoning=f"Unknown flow type: {flow_type}")

        prompt = prompt_template.format(message=message)

        async def _run():
            from khonliang.client import OllamaClient
            client = OllamaClient(model=self.model, base_url=self.ollama_url, timeout=15)
            try:
                return await client.generate(
                    prompt=prompt,
                    system="You are a precise classifier. Respond with ONLY valid JSON.",
                    temperature=0.1,
                    max_tokens=150,
                    extra_options={"num_gpu": 0},  # CPU-only: fast, no GPU contention
                )
            finally:
                await client.close()

        try:
            raw = asyncio.run(_run())
            return self._parse(raw)
        except Exception as e:
            logger.error(f"Flow classification failed: {e}")
            return FlowClassification(action=self.fallback, confidence=0.0, reasoning=str(e))

    def _parse(self, raw: str) -> FlowClassification:
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            text = m.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse classifier response: {raw[:200]}")
            return FlowClassification(action=self.fallback, confidence=0.5,
                                       reasoning="Unparseable response")

        action = data.get("action", self.fallback).lower().strip()
        if self._valid_actions and action not in self._valid_actions:
            action = self.fallback

        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        reasoning = data.get("reasoning", "")

        return FlowClassification(action=action, confidence=confidence, reasoning=reasoning)
