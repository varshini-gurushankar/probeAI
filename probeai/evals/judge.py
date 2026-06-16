"""The isolated LLM-as-judge.

Deliberately constructed SEPARATE from the moderator: its own LLMClient, a
DIFFERENT model (DEFAULT_JUDGE_MODEL), and temperature 0.0 for stable verdicts. It
sees only the snippet it's judging — never the moderator's reasoning or system
prompt — so it can't simply rubber-stamp the moderator's own logic.

Scope is intentionally narrow: it scores follow-up RELEVANCE, and offers an
optional second opinion on leading questions to back up the deterministic
heuristics. It NEVER computes the headline scenario-match accuracy — that is
measured against hand labels (see scenarios.py / metrics.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .. import prompts
from ..llm import DEFAULT_JUDGE_MODEL, LLMClient


@dataclass
class RelevanceVerdict:
    relevant: bool
    reason: str


@dataclass
class LeadingVerdict:
    leading: bool
    reason: str


class Judge:
    def __init__(self, llm: Optional[LLMClient] = None):
        # Isolation: distinct client, judge model, deterministic temperature.
        self.llm = llm or LLMClient(model=DEFAULT_JUDGE_MODEL, temperature=0.0)

    def followup_relevance(
        self, objective_label: str, participant_answer: str, followup: str
    ) -> RelevanceVerdict:
        """Did this follow-up dig into a real gap in the preceding answer?"""
        data = self.llm.complete_json(
            prompts.JUDGE_FOLLOWUP_RELEVANCE.format(
                objective_label=objective_label,
                participant_answer=participant_answer,
                followup=followup,
            )
        )
        return RelevanceVerdict(
            relevant=bool(data.get("relevant", False)),
            reason=str(data.get("reason", "")).strip(),
        )

    def leading_question(self, question: str) -> LeadingVerdict:
        """Second opinion on leading questions (backs up the heuristics)."""
        data = self.llm.complete_json(
            prompts.JUDGE_LEADING_QUESTION.format(question=question)
        )
        return LeadingVerdict(
            leading=bool(data.get("leading", False)),
            reason=str(data.get("reason", "")).strip(),
        )
