"""
GRA Pipeline — Generator-Reviewer-Adjudicator orchestration.

Implements the GRA collaborative framework from "A Strategic Coordination
Framework of Small LLMs Matches Large LLMs in Data Synthesis" as a
reusable base class.

Roles (all optional beyond the generator):
    - Generator:   Produces initial assessment (required; async callable)
    - Reviewer:    Critiques the assessment (optional; async callable)
    - Adjudicator: Rule-based tiebreaker (optional; BaseAdjudicator subclass)

Resolution paths (reflected in GRAResult.resolved_by):
    - "generator"   — no reviewer configured; generator result returned directly
    - "consensus"   — reviewer agrees and verdicts match
    - "reviewer"    — reviewer disagrees but no adjudicator configured
    - "adjudicator" — reviewer disagrees and adjudicator resolves

Usage:
    pipeline = GRAPipeline(
        generate_fn=my_generator,
        review_fn=my_reviewer,       # optional
        adjudicator=my_adjudicator,  # optional
    )
    result = await pipeline.evaluate(subject="TSLA", context=market_data)
"""

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from khonliang.consensus.models import AgentVote
from khonliang.debate.adjudicator import AdjudicationResult, BaseAdjudicator

logger = logging.getLogger(__name__)


@dataclass
class GRAResult:
    """Combined result from the full GRA pipeline.

    Attributes:
        verdict:      Final decision (domain-specific string)
        confidence:   Confidence in the verdict 0.0-1.0
        assessment:   The Generator's output
        review:       The Reviewer's output (None if reviewer skipped)
        adjudication: The Adjudicator's ruling (None if consensus reached)
        resolved_by:  How the verdict was determined:
                      "generator" (no reviewer), "consensus", or "adjudicator"
    """

    verdict: str
    confidence: float
    assessment: Dict[str, Any]
    review: Optional[Dict[str, Any]]
    adjudication: Optional[AdjudicationResult]
    resolved_by: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "resolved_by": self.resolved_by,
            "assessment": self.assessment,
            "review": self.review,
            "adjudication": (
                self.adjudication.to_dict() if self.adjudication else None
            ),
        }


class GRAPipeline:
    """Orchestrates the Generator-Reviewer-Adjudicator flow.

    Flow:
        1. Generator produces an assessment
        2. Reviewer critiques the assessment (optional; skip if review_fn is None)
        3. If they agree on verdict -> return consensus result
        4. If they disagree -> Adjudicator applies domain criteria (optional;
           return reviewer verdict if adjudicator is None)

    The verdict_key parameter tells the pipeline which key in the
    assessment and review dicts holds the verdict string for
    agreement comparison.

    Example (full pipeline):
        async def generate(subject, context=None):
            return {"verdict": "match", "confidence": 0.9, ...}

        async def review(subject, assessment, context=None):
            return {"agrees": True, "verdict": "match", ...}

        pipeline = GRAPipeline(
            generate_fn=generate,
            review_fn=review,
            adjudicator=my_adjudicator,
        )
        result = await pipeline.evaluate("compare persons A and B")

    Example (generator-only):
        pipeline = GRAPipeline(generate_fn=generate)
        result = await pipeline.evaluate("assess this item")
        # result.resolved_by == "generator"
    """

    def __init__(
        self,
        generate_fn: Callable[..., Awaitable[Dict[str, Any]]],
        review_fn: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
        adjudicator: Optional[BaseAdjudicator] = None,
        verdict_key: str = "verdict",
        agrees_key: str = "agrees",
        confidence_key: str = "confidence",
    ):
        """
        Args:
            generate_fn: Async function that produces an assessment dict
            review_fn: Async function that critiques the assessment (optional;
                       if None, generator result is returned directly)
            adjudicator: BaseAdjudicator subclass for conflict resolution
                         (optional; if None, disagreements return reviewer verdict)
            verdict_key: Key in assessment/review dicts holding the verdict
            agrees_key: Key in review dict indicating agreement (bool)
            confidence_key: Key in dicts holding confidence float
        """
        self.generate_fn = generate_fn
        self.review_fn = review_fn
        self.adjudicator = adjudicator
        self.verdict_key = verdict_key
        self.agrees_key = agrees_key
        self.confidence_key = confidence_key

    async def evaluate(
        self,
        subject: str,
        context: Optional[Any] = None,
    ) -> GRAResult:
        """Run the full GRA pipeline.

        Args:
            subject: What is being evaluated (symbol, person pair, PR, etc.)
            context: Domain-specific context passed to all three stages

        Returns:
            GRAResult with verdict and resolution path
        """
        # Step 1: Generator
        assessment = await self.generate_fn(subject, context)
        gen_verdict = assessment.get(self.verdict_key, "")
        gen_confidence = assessment.get(self.confidence_key, 0.5)
        logger.info(
            f"Generator: {gen_verdict} ({gen_confidence:.2f}) "
            f"for '{subject[:40]}'"
        )

        # Generator-only path: no reviewer configured
        if self.review_fn is None:
            return GRAResult(
                verdict=gen_verdict,
                confidence=gen_confidence,
                assessment=assessment,
                review=None,
                adjudication=None,
                resolved_by="generator",
            )

        # Step 2: Reviewer
        review = await self.review_fn(subject, assessment, context)
        rev_agrees = review.get(self.agrees_key, False)
        rev_verdict = review.get(self.verdict_key, "")
        rev_confidence = review.get(self.confidence_key, 0.5)
        logger.info(
            f"Reviewer: {'agrees' if rev_agrees else 'disagrees'} "
            f"({rev_verdict}, {rev_confidence:.2f})"
        )

        # Step 3: Check consensus
        if rev_agrees and rev_verdict == gen_verdict:
            avg_confidence = (gen_confidence + rev_confidence) / 2
            return GRAResult(
                verdict=gen_verdict,
                confidence=avg_confidence,
                assessment=assessment,
                review=review,
                adjudication=None,
                resolved_by="consensus",
            )

        # Step 4: Disagreement -> Adjudicator (if configured)
        if self.adjudicator is None:
            # No adjudicator: return reviewer's verdict
            return GRAResult(
                verdict=rev_verdict,
                confidence=rev_confidence,
                assessment=assessment,
                review=review,
                adjudication=None,
                resolved_by="reviewer",
            )

        logger.info(
            f"Disagreement: Generator={gen_verdict}, "
            f"Reviewer={rev_verdict}. Invoking adjudicator."
        )

        votes = [
            AgentVote(
                agent_id="generator",
                action=gen_verdict,
                confidence=gen_confidence,
                reasoning=str(assessment.get("reasoning", "")),
            ),
            AgentVote(
                agent_id="reviewer",
                action=rev_verdict,
                confidence=rev_confidence,
                reasoning=str(
                    review.get("critique", review.get("reasoning", ""))
                ),
            ),
        ]

        adjudication = self.adjudicator.adjudicate(votes, subject, context)

        return GRAResult(
            verdict=adjudication.action,
            confidence=adjudication.confidence,
            assessment=assessment,
            review=review,
            adjudication=adjudication,
            resolved_by="adjudicator",
        )
