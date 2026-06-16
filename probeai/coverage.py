"""Per-objective coverage tracking — the explicit interview state.

Taxonomy (a DESIGN DECISION, see README):
    uncovered -> the objective has not been substantively addressed
    partial   -> on-topic but vague/generic; missing specifics -> worth a probe
    covered   -> the participant gave a SUBSTANTIVE, SPECIFIC answer to the
                 objective's intent; we store the supporting turn(s) as evidence

The moderator READS this to decide what to ask next and UPDATES it after each
answer. Everything here is deterministic given a ``Verdict`` — no LLM — which is
exactly what the unit tests exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .study_config import Priority, Study


class CoverageStatus(str, Enum):
    uncovered = "uncovered"
    partial = "partial"
    covered = "covered"


class AnswerVerdict(str, Enum):
    """How the latest answer addressed the objective it was assessed against."""

    covered = "covered"
    partial = "partial"
    off_topic = "off_topic"


# Status ranking so we never *downgrade* an objective once it's been covered
# (e.g. a later vague aside shouldn't undo solid evidence).
_RANK = {CoverageStatus.uncovered: 0, CoverageStatus.partial: 1, CoverageStatus.covered: 2}
_VERDICT_TO_STATUS = {
    AnswerVerdict.covered: CoverageStatus.covered,
    AnswerVerdict.partial: CoverageStatus.partial,
}


@dataclass
class ObjectiveAssessment:
    """An LLM judgment that one answer addressed a given objective."""

    objective_id: str
    verdict: AnswerVerdict


@dataclass
class Verdict:
    """The moderator's assessment of a single participant answer (LLM output).

    Kept as a plain dataclass so tests can construct it directly and feed it to
    the deterministic coverage/decision logic without any model call.
    """

    current_objective_id: str
    status: AnswerVerdict  # how the answer addressed the CURRENT objective
    is_specific: bool  # substantive/specific? (specificity dominates probing)
    gap: str = ""  # what's missing — used to craft a targeted probe
    # Other objectives this same answer happened to address (handles the
    # participant who volunteers info for a later objective).
    also_addressed: List[ObjectiveAssessment] = field(default_factory=list)


@dataclass
class ObjectiveCoverage:
    objective_id: str
    priority: Priority
    status: CoverageStatus = CoverageStatus.uncovered
    evidence_turn_ids: List[int] = field(default_factory=list)
    probes_used: int = 0  # follow-ups asked specifically for this objective
    times_asked: int = 0  # how often we've put a question for it to the participant


class CoverageState:
    """The full set of objective statuses for one interview."""

    def __init__(self, study: Study) -> None:
        self._study = study
        self.objectives: dict[str, ObjectiveCoverage] = {
            o.id: ObjectiveCoverage(objective_id=o.id, priority=o.priority)
            for o in study.objectives
        }

    # --- reads ---------------------------------------------------------------
    def get(self, objective_id: str) -> ObjectiveCoverage:
        return self.objectives[objective_id]

    @property
    def total(self) -> int:
        return len(self.objectives)

    @property
    def covered_count(self) -> int:
        return sum(
            1 for c in self.objectives.values() if c.status is CoverageStatus.covered
        )

    def coverage_pct(self) -> float:
        """% of objectives that reached 'covered'."""
        return 100.0 * self.covered_count / self.total if self.total else 0.0

    def is_fully_covered(self) -> bool:
        return self.covered_count == self.total

    # --- writes (deterministic given a Verdict) ------------------------------
    def mark_asked(self, objective_id: str) -> None:
        self.objectives[objective_id].times_asked += 1

    def record_probe(self, objective_id: str) -> None:
        self.objectives[objective_id].probes_used += 1

    def _upgrade(self, objective_id: str, status: CoverageStatus, turn_id: int) -> None:
        """Raise an objective's status (never lower it) and log evidence."""
        cov = self.objectives[objective_id]
        if _RANK[status] > _RANK[cov.status]:
            cov.status = status
        # Record the supporting turn for any on-topic (partial/covered) verdict.
        if status in (CoverageStatus.partial, CoverageStatus.covered):
            if turn_id not in cov.evidence_turn_ids:
                cov.evidence_turn_ids.append(turn_id)

    def apply_verdict(self, verdict: Verdict, turn_id: int) -> None:
        """Fold one answer's assessment into coverage state.

        Updates the current objective and any *also_addressed* objectives (the
        participant who volunteers info for a later topic). ``off_topic`` answers
        change nothing — they neither cover nor add evidence.
        """
        if verdict.status in _VERDICT_TO_STATUS:
            self._upgrade(
                verdict.current_objective_id,
                _VERDICT_TO_STATUS[verdict.status],
                turn_id,
            )
        for extra in verdict.also_addressed:
            if extra.verdict in _VERDICT_TO_STATUS and extra.objective_id in self.objectives:
                self._upgrade(
                    extra.objective_id, _VERDICT_TO_STATUS[extra.verdict], turn_id
                )

    # --- serialization for the UI / transcript -------------------------------
    def snapshot(self) -> list[dict]:
        """A JSON-friendly view for the coverage panel and saved transcript."""
        out = []
        for o in self._study.objectives:  # preserve study order
            c = self.objectives[o.id]
            out.append(
                {
                    "objective_id": o.id,
                    "label": o.label,
                    "priority": o.priority.value,
                    "status": c.status.value,
                    "evidence_turn_ids": list(c.evidence_turn_ids),
                    "probes_used": c.probes_used,
                    "times_asked": c.times_asked,
                }
            )
        return out
