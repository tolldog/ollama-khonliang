"""
Training data exporter — builds fine-tuning datasets from captured interactions.

Pulls from two sources:
- ``training_feedback`` — rated interactions (use high-rated as positive examples)
- ``agent_interactions`` — all interactions (use as general corpus)

Exports to JSONL in three formats:
- ``alpaca``    — ``{instruction, input, output}``
- ``sharegpt``  — ``{conversations: [{from, value}, ...]}``
- ``completion``— ``{prompt, completion}``

Usage::

    exporter = TrainingExporter("knowledge.db")
    path = exporter.export("training.jsonl", fmt="alpaca")
    print(exporter.stats())
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example."""

    instruction: str
    input_text: str
    output: str
    source: str = "feedback"
    context: Optional[str] = None


class TrainingExporter:
    """
    Build JSONL training datasets from stored interactions and feedback.

    Args:
        db_path:    Path to SQLite database (same file used by FeedbackStore)
        output_dir: Directory for exported files (created if missing)
        agent_name: Name used in instruction templates (e.g. 'helpdesk agent')

    Example:
        >>> exporter = TrainingExporter("knowledge.db")
        >>> path = exporter.export("training.jsonl", fmt="alpaca", min_rating=4)
        >>> print(f"Exported to {path}")
    """

    def __init__(
        self,
        db_path: str = "data/knowledge.db",
        output_dir: str = "data/training",
        agent_name: str = "helpdesk agent",
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_name = agent_name

    # ── Public API ────────────────────────────────────────────────────────

    def collect(
        self,
        min_rating: int = 3,
        include_unrated: bool = False,
        limit: int = 500,
    ) -> List[TrainingExample]:
        """
        Collect training examples from the database.

        Args:
            min_rating:       Minimum feedback rating to include (default 3)
            include_unrated:  Also include all interactions without feedback
            limit:            Max examples to collect per source

        Returns:
            List of TrainingExample
        """
        examples: List[TrainingExample] = []

        # Rated feedback examples
        examples.extend(self._from_feedback(min_rating=min_rating, limit=limit))

        # Raw interactions (no feedback required)
        if include_unrated:
            examples.extend(self._from_interactions(limit=limit))

        logger.info(f"Collected {len(examples)} training examples")
        return examples

    def export(
        self,
        filename: str = "training.jsonl",
        fmt: str = "alpaca",
        min_rating: int = 3,
        include_unrated: bool = False,
    ) -> Path:
        """
        Export training examples to a JSONL file.

        Args:
            filename:        Output filename (placed in output_dir)
            fmt:             Format: 'alpaca', 'sharegpt', or 'completion'
            min_rating:      Minimum rating to include from feedback table
            include_unrated: Also include unrated interactions

        Returns:
            Path to the exported file
        """
        examples = self.collect(min_rating=min_rating, include_unrated=include_unrated)
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(self._format(ex, fmt)) + "\n")

        logger.info(f"Exported {len(examples)} examples to {output_path} ({fmt})")
        return output_path

    def stats(self) -> Dict:
        """Return statistics about available training data."""
        conn = sqlite3.connect(self.db_path)
        try:
            interactions = conn.execute(
                "SELECT COUNT(*) FROM agent_interactions"
            ).fetchone()[0]

            rated = conn.execute(
                "SELECT COUNT(*) FROM training_feedback WHERE rating IS NOT NULL"
            ).fetchone()[0]

            by_rating: Dict[int, int] = {}
            for row in conn.execute(
                "SELECT rating, COUNT(*) FROM training_feedback "
                "WHERE rating IS NOT NULL GROUP BY rating"
            ).fetchall():
                by_rating[row[0]] = row[1]

            good = sum(v for k, v in by_rating.items() if k >= 4)
            return {
                "total_interactions": interactions,
                "rated_feedback": rated,
                "by_rating": by_rating,
                "good_examples": good,
                "ready_for_export": good >= 10,
            }
        except sqlite3.OperationalError:
            return {
                "total_interactions": 0,
                "rated_feedback": 0,
                "by_rating": {},
                "good_examples": 0,
                "ready_for_export": False,
            }
        finally:
            conn.close()

    # ── Internal collectors ───────────────────────────────────────────────

    def _from_feedback(self, min_rating: int, limit: int) -> List[TrainingExample]:
        examples = []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT prompt, response, rating, feedback, expected
                FROM training_feedback
                WHERE rating >= ?
                ORDER BY rating DESC, created_at DESC
                LIMIT ?
                """,
                (min_rating, limit),
            ).fetchall()

            for row in rows:
                # Use the corrected "expected" response if available, else bot response
                output = row["expected"] or row["response"] or ""
                if not output:
                    continue

                examples.append(TrainingExample(
                    instruction=(
                        f"You are a {self.agent_name}. "
                        "Answer the user's question helpfully and concisely."
                    ),
                    input_text=row["prompt"],
                    output=output,
                    source="feedback",
                    context=(
                        f"Rating: {row['rating']}/5"
                        + (f". Note: {row['feedback']}" if row["feedback"] else "")
                    ),
                ))
        except sqlite3.OperationalError as e:
            logger.debug(f"Feedback table not found: {e}")
        finally:
            conn.close()

        return examples

    def _from_interactions(self, limit: int) -> List[TrainingExample]:
        examples = []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT i.message, i.response, i.role
                FROM agent_interactions i
                LEFT JOIN training_feedback f ON f.interaction_id = i.id
                WHERE f.id IS NULL      -- not already covered by feedback
                  AND i.response IS NOT NULL
                  AND LENGTH(i.response) > 30
                ORDER BY i.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            for row in rows:
                examples.append(TrainingExample(
                    instruction=(
                        f"You are a {self.agent_name} "
                        f"({row['role'] or 'general'}). "
                        "Answer the user's question."
                    ),
                    input_text=row["message"],
                    output=row["response"],
                    source="interaction",
                ))
        except sqlite3.OperationalError as e:
            logger.debug(f"Interactions table not found: {e}")
        finally:
            conn.close()

        return examples

    # ── Formatters ────────────────────────────────────────────────────────

    def _format(self, ex: TrainingExample, fmt: str) -> Dict:
        if fmt == "alpaca":
            data = {
                "instruction": ex.instruction,
                "input": ex.input_text,
                "output": ex.output,
            }
            if ex.context:
                data["context"] = ex.context
            return data

        elif fmt == "sharegpt":
            messages = [
                {"from": "system", "value": ex.instruction},
            ]
            if ex.context:
                messages.append({"from": "system", "value": f"Context: {ex.context}"})
            messages.append({"from": "human", "value": ex.input_text})
            messages.append({"from": "gpt", "value": ex.output})
            return {"conversations": messages}

        elif fmt == "completion":
            prompt = ex.instruction
            if ex.context:
                prompt += f"\n\nContext: {ex.context}"
            prompt += f"\n\nUser: {ex.input_text}\n\nAssistant:"
            return {"prompt": prompt, "completion": ex.output}

        else:
            raise ValueError(f"Unknown format: {fmt!r}. Use 'alpaca', 'sharegpt', or 'completion'.")
