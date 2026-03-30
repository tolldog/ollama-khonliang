"""
Agent envelope — standard message wrapper with model provenance.

Every agent response can be wrapped in an envelope that tracks:
- Who generated it (role, agent_id)
- What model was used (name, version, inference time, tokens)
- The intent and payload
- Correlation for request/response pairing

Usage:
    envelope = AgentEnvelope.create(
        from_role="researcher",
        from_agent_id="web_search",
        intent="analysis",
        payload={"text": "Roger Tolle was born in 1642..."},
        model_meta=ModelMeta(model_name="llama3.2:3b", inference_ms=230),
    )
"""

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelMeta:
    """Tracks which model generated a response and how."""

    model_name: str = ""
    inference_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    temperature: float = 0.0
    model_version: str = ""  # e.g. LoRA checkpoint name
    container_id: str = ""  # if running in a container

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, omitting falsy/empty values."""
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class AgentEnvelope:
    """
    Standard wrapper for agent messages.

    Carries routing info, intent, payload, and model provenance.
    Can wrap any agent output — responses, votes, research results.
    """

    envelope_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    from_role: str = ""
    from_agent_id: str = ""
    intent: str = ""  # analysis, vote, question, challenge, response, signal, summary
    payload: Dict[str, Any] = field(default_factory=dict)
    model_meta: Optional[ModelMeta] = None
    correlation_id: str = ""  # links request/response pairs
    reply_to: str = ""  # envelope_id this is replying to
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        from_role: str,
        from_agent_id: str = "",
        intent: str = "",
        payload: Optional[Dict[str, Any]] = None,
        model_meta: Optional[ModelMeta] = None,
        correlation_id: str = "",
    ) -> "AgentEnvelope":
        """Create a new envelope, auto-generating a correlation ID if none is provided."""
        return cls(
            from_role=from_role,
            from_agent_id=from_agent_id,
            intent=intent,
            payload=payload or {},
            model_meta=model_meta,
            correlation_id=correlation_id or str(uuid.uuid4())[:12],
        )

    @classmethod
    def reply(
        cls,
        original: "AgentEnvelope",
        from_role: str,
        from_agent_id: str = "",
        intent: str = "response",
        payload: Optional[Dict[str, Any]] = None,
        model_meta: Optional[ModelMeta] = None,
    ) -> "AgentEnvelope":
        """Create a reply envelope linked to the original."""
        return cls(
            from_role=from_role,
            from_agent_id=from_agent_id,
            intent=intent,
            payload=payload or {},
            model_meta=model_meta,
            correlation_id=original.correlation_id,
            reply_to=original.envelope_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, including model_meta if present."""
        data = {
            "envelope_id": self.envelope_id,
            "from_role": self.from_role,
            "from_agent_id": self.from_agent_id,
            "intent": self.intent,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "timestamp": self.timestamp,
        }
        if self.model_meta:
            data["model_meta"] = self.model_meta.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentEnvelope":
        """Deserialize from a dict, reconstructing ModelMeta if present."""
        meta = data.get("model_meta")
        return cls(
            envelope_id=data.get("envelope_id", ""),
            from_role=data.get("from_role", ""),
            from_agent_id=data.get("from_agent_id", ""),
            intent=data.get("intent", ""),
            payload=data.get("payload", {}),
            model_meta=ModelMeta(**meta) if meta else None,
            correlation_id=data.get("correlation_id", ""),
            reply_to=data.get("reply_to", ""),
            timestamp=data.get("timestamp", 0),
        )
