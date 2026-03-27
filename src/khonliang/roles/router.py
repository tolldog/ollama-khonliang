"""
Message router — maps incoming messages to roles.

Extend BaseRouter to define your intent space and routing rules.
The dispatch pipeline is evaluated in priority order:
  1. Callable rules (highest — full access to message)
  2. Regex patterns
  3. Keyword lists
  4. Semantic router (optional — embedding similarity, <5ms, CPU-only)
  5. Fallback role

Attaching a SemanticIntentRouter as stage 4 replaces a long list of keyword
rules with learned similarity, while keeping deterministic regex fast-paths
for high-priority patterns (e.g. urgent alerts, structured commands).

Example:

    from khonliang.routing import SemanticIntentRouter

    class SupportRouter(BaseRouter):
        def __init__(self):
            super().__init__(fallback_role="general")
            self.register_pattern(r"(?i)urgent|critical|down|outage", "triage")
            self.register_keywords(["billing", "invoice", "charge"], "billing")

            # Semantic stage covers everything else
            self.set_semantic_router(SemanticIntentRouter(
                routes={
                    "triage":    ["server is down", "nothing is working", "production broke"],
                    "billing":   ["invoice", "refund request", "payment failed"],
                    "knowledge": ["how do I", "tutorial", "docs", "explain"],
                },
                fallback="general",
            ))

    router = SupportRouter()
    role = router.route("My server is down!")   # -> "triage"  (regex fast-path)
    role = router.route("I need a refund")      # -> "billing" (semantic match)
    role = router.route("Hi there")             # -> "general" (fallback)
"""

import logging
import re
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from khonliang.routing.semantic import SemanticIntentRouter

logger = logging.getLogger(__name__)


class BaseRouter:
    """
    Configurable message router with regex, keyword, callable, and semantic rules.

    Rules are evaluated in priority order:
      1. Callable rules (highest — full access to message)
      2. Regex patterns
      3. Keyword lists
      4. SemanticIntentRouter (optional — embedding similarity, lazy-loaded)
      5. Fallback role

    Args:
        fallback_role:   Role name returned when no rule matches
        semantic_router: Optional SemanticIntentRouter for the semantic stage.
                         Can also be set later via set_semantic_router().
    """

    def __init__(
        self,
        fallback_role: str = "default",
        semantic_router: Optional["SemanticIntentRouter"] = None,
    ):
        self.fallback_role = fallback_role
        self._callable_rules: List[Tuple[Callable[[str], bool], str]] = []
        self._regex_rules: List[Tuple[re.Pattern, str]] = []
        self._keyword_rules: List[Tuple[List[str], str]] = []
        self._semantic_router: Optional["SemanticIntentRouter"] = semantic_router

    def set_semantic_router(self, router: "SemanticIntentRouter") -> None:
        """
        Attach a SemanticIntentRouter as the final classification stage.

        When set, messages that don't match any regex or keyword rule are
        passed to the semantic router before falling back to fallback_role.
        The semantic router's own fallback is ignored — use this router's
        fallback_role instead.

        Args:
            router: A configured SemanticIntentRouter instance
        """
        self._semantic_router = router

    def register_rule(self, predicate: Callable[[str], bool], role: str) -> None:
        """Register a callable predicate. Called with the raw message, returns bool."""
        self._callable_rules.append((predicate, role))

    def register_pattern(self, pattern: str, role: str, flags: int = re.IGNORECASE) -> None:
        """Register a regex pattern that routes to role when matched."""
        self._regex_rules.append((re.compile(pattern, flags), role))

    def register_keywords(self, keywords: List[str], role: str) -> None:
        """Route to role when any keyword appears in the message (case-insensitive)."""
        self._keyword_rules.append((keywords, role))

    def route(self, message: str) -> str:
        """
        Route a message to a role name.

        Evaluates rules in priority order:
        callable → regex → keyword → semantic → fallback.
        """
        if not message or not message.strip():
            return self.fallback_role

        msg_lower = message.lower()

        for predicate, role in self._callable_rules:
            try:
                if predicate(message):
                    logger.debug(f"Callable rule matched -> {role}")
                    return role
            except Exception as e:
                logger.warning(f"Callable rule error: {e}")

        for pattern, role in self._regex_rules:
            if pattern.search(message):
                logger.debug(f"Regex pattern matched -> {role}")
                return role

        for keywords, role in self._keyword_rules:
            if any(kw.lower() in msg_lower for kw in keywords):
                logger.debug(f"Keyword matched -> {role}")
                return role

        if self._semantic_router is not None:
            try:
                result = self._semantic_router.classify(message)
                if result != self._semantic_router._fallback:
                    logger.debug(f"Semantic router matched -> {result}")
                    return result
            except Exception as e:
                logger.warning(f"Semantic router error: {e}")

        return self.fallback_role

    def route_with_reason(self, message: str) -> Tuple[str, str]:
        """Route and return (role, reason) for debugging."""
        msg_lower = message.lower()

        for predicate, role in self._callable_rules:
            try:
                if predicate(message):
                    return role, "callable_rule"
            except Exception:
                pass

        for pattern, role in self._regex_rules:
            if pattern.search(message):
                return role, f"regex:{pattern.pattern[:40]}"

        for keywords, role in self._keyword_rules:
            matched = next((kw for kw in keywords if kw.lower() in msg_lower), None)
            if matched:
                return role, f"keyword:{matched}"

        if self._semantic_router is not None:
            try:
                result, score = self._semantic_router.classify_with_score(message)
                if result != self._semantic_router._fallback:
                    return result, f"semantic:{score:.3f}"
            except Exception:
                pass

        return self.fallback_role, "fallback"
