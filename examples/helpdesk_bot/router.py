"""
Helpdesk router — maps messages to roles using regex and keyword rules.
"""

from khonliang.roles.router import BaseRouter


class HelpdeskRouter(BaseRouter):
    """
    Routes support messages to the appropriate role.

    Priority (first match wins):
      1. Escalation: explicit request or repeated failures
      2. Knowledge: how-to / documentation questions
      3. Triage: new ticket keywords
      4. Fallback: triage (safe default for unknowns)
    """

    def __init__(self):
        super().__init__(fallback_role="triage")

        # Explicit escalation requests
        self.register_pattern(
            r"(?i)escalate|speak to (a human|someone|manager)"
            r"|not (happy|working|resolved)",
            "escalation",
        )

        # Knowledge base lookups
        self.register_keywords(
            [
                "how do i",
                "how to",
                "can i",
                "is it possible",
                "documentation",
                "tutorial",
                "guide",
                "where",
                "what is",
                "explain",
            ],
            "knowledge",
        )

        # General triage (new issues)
        self.register_keywords(
            [
                "broken",
                "error",
                "not working",
                "can't",
                "cannot",
                "issue",
                "problem",
                "bug",
                "fail",
                "down",
                "crash",
            ],
            "triage",
        )
