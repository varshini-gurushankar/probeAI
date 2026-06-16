"""Auto-synthesis — Great Question's "analysis built-in".

After an interview, produce a short findings summary plus a few objective-linked
highlights (verbatim participant quotes). The LLM does the writing; this module
validates and grounds its output: every highlight must map to a real objective and
quote text that actually appears in a participant turn, so we never ship a
hallucinated quote into a research deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import prompts
from .llm import LLMClient
from .moderator import _format_objective_menu  # reuse the same menu formatting
from .study_config import Study
from .transcript import Transcript


@dataclass
class Highlight:
    objective_id: str
    quote: str
    insight: str
    # turn id of the participant turn the quote was matched to (evidence link).
    turn_id: Optional[int] = None


@dataclass
class Synthesis:
    summary: str
    highlights: List[Highlight] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "highlights": [h.__dict__ for h in self.highlights],
        }


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _match_quote_to_turn(quote: str, transcript: Transcript) -> Optional[int]:
    """Find the participant turn a quote came from (substring match, normalized)."""
    nq = _norm(quote)
    if not nq:
        return None
    for turn in transcript.participant_turns():
        if nq in _norm(turn.text):
            return turn.turn_id
    return None


def synthesize(
    study: Study,
    transcript: Transcript,
    llm: Optional[LLMClient] = None,
) -> Synthesis:
    """Generate a grounded findings summary + verbatim highlights."""
    llm = llm or LLMClient(temperature=0.3)
    valid_ids = {o.id for o in study.objectives}

    data = llm.complete_json(
        prompts.SYNTHESIS_PROMPT.format(
            research_goal=study.research_goal,
            objective_menu=_format_objective_menu(study),
            transcript=transcript.render(),
        )
    )

    highlights: List[Highlight] = []
    for h in data.get("highlights", []) or []:
        oid = h.get("objective_id")
        quote = str(h.get("quote", "")).strip()
        if oid not in valid_ids or not quote:
            continue
        turn_id = _match_quote_to_turn(quote, transcript)
        if turn_id is None:
            # Drop ungrounded quotes — a research artifact must not contain a
            # quote the participant never said.
            continue
        highlights.append(
            Highlight(
                objective_id=oid,
                quote=quote,
                insight=str(h.get("insight", "")).strip(),
                turn_id=turn_id,
            )
        )

    return Synthesis(summary=str(data.get("summary", "")).strip(), highlights=highlights)
