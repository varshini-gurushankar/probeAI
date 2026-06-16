"""The moderator engine — the agentic core.

Each turn is three steps, deliberately separated so the logic is testable:

  1. assess()             -> LLM reads the answer, returns a Verdict (JSON)
  2. decide_next_action() -> PURE function: Verdict + coverage + policy -> Decision
  3. generate()           -> LLM turns the Decision into one on-policy utterance

Only steps 1 and 3 touch the model. Step 2 — the actual "should I probe, move on,
steer back, or wrap up?" decision — is a deterministic function unit-tested with
injected verdicts. ``decide_next_action`` and ``pick_next_objective`` below are the
heart of the moderation policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from . import prompts
from .coverage import (
    AnswerVerdict,
    CoverageState,
    CoverageStatus,
    ObjectiveAssessment,
    Verdict,
)
from .llm import LLMClient
from .study_config import Objective, Policy, Priority, Study
from .transcript import Transcript


class Action(str, Enum):
    OPEN = "open"  # first line: intro + first question
    ASK_NEXT = "ask_next"  # move to a new objective's question
    PROBE = "probe"  # one targeted follow-up on the current objective's gap
    STEER_BACK = "steer_back"  # redirect an off-topic answer to the current objective
    CLOSE = "close"  # wrap up


@dataclass
class Decision:
    """A fully-determined next move, plus the rationale (for logs/demo)."""

    action: Action
    target_objective_id: Optional[str]
    rationale: str
    gap: str = ""


# Priority ordering for "what to ask next" — high first.
_PRIORITY_ORDER = {Priority.high: 0, Priority.medium: 1, Priority.low: 2}


# =============================================================================
# PURE DECISION LOGIC (no LLM — this is what the unit tests target)
# =============================================================================


def pick_next_objective(
    coverage: CoverageState, study: Study, policy: Policy, remaining_budget: int
) -> Optional[str]:
    """Choose the next *fresh* objective to introduce, by priority.

    Only objectives that are not yet covered AND have never been asked are
    candidates — so we never loop back onto an objective the participant already
    declined (this is what makes termination graceful). Low-priority objectives
    are skipped when the remaining turn budget is tight.
    """
    skip_low = remaining_budget <= policy.low_priority_skip_threshold
    candidates: list[Objective] = []
    for o in study.objectives:
        c = coverage.get(o.id)
        if c.status is CoverageStatus.covered:
            continue
        if c.times_asked > 0:  # already introduced once — don't re-ask / loop
            continue
        if skip_low and o.priority is Priority.low:
            continue
        candidates.append(o)
    if not candidates:
        return None
    # Stable sort by priority preserves the researcher's authored order within a tier.
    candidates.sort(key=lambda o: _PRIORITY_ORDER[o.priority])
    return candidates[0].id


def decide_next_action(
    *,
    current_objective_id: str,
    last_verdict_status: AnswerVerdict,
    coverage: CoverageState,
    policy: Policy,
    study: Study,
    turns_used: int,
) -> Decision:
    """Decide the moderator's next move. Pure + deterministic.

    Assumes ``coverage`` has ALREADY had the latest verdict applied. Encodes the
    policy: probe partial answers (up to the priority-based cap), steer back
    off-topic ones (also under the cap), otherwise advance to the next objective,
    and close when the budget is spent or no fresh objective remains.
    """
    remaining = policy.turn_budget(study.objectives.__len__()) - turns_used
    if remaining <= 0:
        return Decision(Action.CLOSE, None, "Turn budget reached — wrapping up.")

    current = coverage.get(current_objective_id)
    cap = policy.probe_cap(current.priority)
    objective_label = study.objective(current_objective_id).label

    if last_verdict_status is AnswerVerdict.off_topic:
        # Steer-back attempts count against the same depth cap as probes so we
        # don't chase a rambler forever.
        if current.probes_used < cap:
            return Decision(
                Action.STEER_BACK,
                current_objective_id,
                f"Answer was off-topic; steering back to '{objective_label}' "
                f"(redirect {current.probes_used + 1}/{cap}).",
            )
        # Out of redirect budget for this objective — move on.

    elif last_verdict_status is AnswerVerdict.partial:
        if current.probes_used < cap:
            return Decision(
                Action.PROBE,
                current_objective_id,
                f"Answer was partial/vague on '{objective_label}'; probing the gap "
                f"(follow-up {current.probes_used + 1}/{cap}).",
            )
        # Cap reached — accept partial coverage and move on.

    # covered, or we've exhausted probes/redirects: advance or close.
    next_id = pick_next_objective(coverage, study, policy, remaining)
    if next_id is None:
        reason = (
            "All objectives covered."
            if coverage.is_fully_covered()
            else "No fresh objectives remain to ask — closing gracefully."
        )
        return Decision(Action.CLOSE, None, reason)
    next_label = study.objective(next_id).label
    return Decision(
        Action.ASK_NEXT,
        next_id,
        f"'{objective_label}' is handled; moving to next objective '{next_label}'.",
    )


# =============================================================================
# LLM-BACKED STEPS (assess + generate) and the session orchestrator
# =============================================================================


def _format_objective_menu(study: Study) -> str:
    return "\n".join(
        f"- {o.id}: {o.label} — {o.description.strip()}" for o in study.objectives
    )


def _format_dialogue(history: List[dict], limit: int = 6) -> str:
    recent = history[-limit:]
    if not recent:
        return "(interview has not started yet)"
    speaker = {"moderator": "Moderator", "participant": "Participant"}
    return "\n".join(f"{speaker[h['speaker']]}: {h['text']}" for h in recent)


class Moderator:
    """Wraps the two LLM-backed steps (assess + generate) around the pure logic."""

    def __init__(
        self,
        study: Study,
        policy: Policy,
        llm: Optional[LLMClient] = None,
        lean: bool = False,
    ):
        self.study = study
        self.policy = policy
        # Low temperature: we want consistent, on-policy interviewing.
        self.llm = llm or LLMClient(temperature=0.4)
        # lean=True (used by the eval harness on tight quota): only PROBE calls the
        # LLM; the rote open/next/steer/close lines are templated from the
        # researcher's own seed questions. Cuts LLM calls roughly in half. The
        # assess step still runs every turn — it is the behavior under test.
        self.lean = lean

    def assess(
        self, answer: str, current_objective_id: str, history: List[dict]
    ) -> Verdict:
        """LLM step 1: judge the latest answer -> Verdict (drives state)."""
        current = self.study.objective(current_objective_id)
        user = prompts.MODERATOR_ASSESS.format(
            research_goal=self.study.research_goal,
            current_id=current.id,
            current_label=current.label,
            current_description=current.description.strip(),
            objective_menu=_format_objective_menu(self.study),
            recent_dialogue=_format_dialogue(history),
            answer=answer,
        )
        data = self.llm.complete_json(user)
        return self._parse_verdict(data, current_objective_id)

    def _parse_verdict(self, data: dict, current_objective_id: str) -> Verdict:
        cur = data.get("current", {}) or {}
        status = _coerce_verdict(cur.get("status"), default=AnswerVerdict.partial)
        valid_ids = set(self.study.objective(o.id).id for o in self.study.objectives)
        also: list[ObjectiveAssessment] = []
        for entry in data.get("also_addressed", []) or []:
            oid = entry.get("objective_id")
            if oid in valid_ids and oid != current_objective_id:
                v = _coerce_verdict(entry.get("status"), default=None)
                if v in (AnswerVerdict.covered, AnswerVerdict.partial):
                    also.append(ObjectiveAssessment(objective_id=oid, verdict=v))
        return Verdict(
            current_objective_id=current_objective_id,
            status=status,
            is_specific=bool(cur.get("is_specific", False)),
            gap=str(cur.get("gap", "")).strip(),
            also_addressed=also,
        )

    def generate(self, decision: Decision, history: List[dict]) -> str:
        """LLM step 3: turn a Decision into one on-policy utterance.

        In lean mode, everything except a PROBE is templated (no LLM call).
        """
        if self.lean and decision.action is not Action.PROBE:
            return self._template(decision)
        directive = self._build_directive(decision)
        user = prompts.MODERATOR_GENERATE.format(
            directive=directive, recent_dialogue=_format_dialogue(history)
        )
        return self.llm.complete(user, system=self._system())

    def _template(self, decision: Decision) -> str:
        """Deterministic, LLM-free utterances for lean/harness mode."""
        obj = (
            self.study.objective(decision.target_objective_id)
            if decision.target_objective_id
            else None
        )
        if decision.action is Action.OPEN:
            return f"{self.study.intro.strip()} {obj.seed_questions[0]}".strip()
        if decision.action is Action.ASK_NEXT:
            return obj.seed_questions[0]
        if decision.action is Action.STEER_BACK:
            return (
                f"Thanks for sharing that. Coming back to {obj.label.lower()} — "
                f"{obj.seed_questions[0]}"
            )
        return self.study.outro.strip()

    def _system(self) -> str:
        return prompts.MODERATOR_SYSTEM.format(research_goal=self.study.research_goal)

    def _build_directive(self, decision: Decision) -> str:
        obj = (
            self.study.objective(decision.target_objective_id)
            if decision.target_objective_id
            else None
        )
        if decision.action is Action.OPEN:
            return prompts.DIRECTIVE_OPEN.format(label=obj.label, seed=obj.seed_questions[0])
        if decision.action is Action.ASK_NEXT:
            return prompts.DIRECTIVE_ASK_NEXT.format(
                label=obj.label, seed=obj.seed_questions[0]
            )
        if decision.action is Action.PROBE:
            return prompts.DIRECTIVE_PROBE.format(label=obj.label, gap=decision.gap)
        if decision.action is Action.STEER_BACK:
            return prompts.DIRECTIVE_STEER_BACK.format(label=obj.label)
        return prompts.DIRECTIVE_CLOSE.format(outro=self.study.outro)


def _coerce_verdict(value, default):
    try:
        return AnswerVerdict(value)
    except (ValueError, KeyError, TypeError):
        return default


class InterviewSession:
    """Holds interview state and runs one turn at a time.

    Owns the coverage state, the turn counter, the current objective, and an
    in-memory dialogue history. Slice 2 adds transcript persistence + a simulated
    participant on top of this.
    """

    def __init__(self, study: Study, policy: Policy, moderator: Optional[Moderator] = None):
        self.study = study
        self.policy = policy
        self.moderator = moderator or Moderator(study, policy)
        self.coverage = CoverageState(study)
        self.transcript = Transcript(study.study_id)
        self.participant_turns = 0
        self.current_objective_id: Optional[str] = None
        self.finished = False
        self.last_decision: Optional[Decision] = None
        # Per-turn record of decisions/verdicts (used by the eval harness).
        self.turn_log: List[dict] = []

    @property
    def history(self) -> List[dict]:
        """Dialogue in the shape the moderator engine consumes."""
        return self.transcript.entries

    @property
    def turn_budget(self) -> int:
        return self.policy.turn_budget(len(self.study.objectives))

    def _say(self, text: str, decision: Decision) -> None:
        self.transcript.append(
            "moderator", text, action=decision.action.value, rationale=decision.rationale
        )

    def start(self) -> str:
        """Produce the opening line (intro + first objective question)."""
        first_id = pick_next_objective(self.coverage, self.study, self.policy, self.turn_budget)
        decision = Decision(Action.OPEN, first_id, "Opening the interview.")
        self.current_objective_id = first_id
        self.coverage.mark_asked(first_id)
        self.last_decision = decision
        line = self.moderator.generate(decision, self.history)
        self._say(line, decision)
        return line

    def step(self, answer: str) -> dict:
        """Process one participant answer; return the moderator's next line + state."""
        if self.finished:
            raise RuntimeError("interview already finished")
        turn_id = self.transcript.append("participant", answer).turn_id
        self.participant_turns += 1

        verdict = self.moderator.assess(answer, self.current_objective_id, self.history)
        self.coverage.apply_verdict(verdict, turn_id)

        decision = decide_next_action(
            current_objective_id=self.current_objective_id,
            last_verdict_status=verdict.status,
            coverage=self.coverage,
            policy=self.policy,
            study=self.study,
            turns_used=self.participant_turns,
        )
        self.last_decision = decision

        # Deterministic bookkeeping before we speak.
        if decision.action in (Action.PROBE, Action.STEER_BACK):
            self.coverage.record_probe(decision.target_objective_id)
        elif decision.action is Action.ASK_NEXT:
            self.current_objective_id = decision.target_objective_id
            self.coverage.mark_asked(decision.target_objective_id)
        elif decision.action is Action.CLOSE:
            self.finished = True

        line = self.moderator.generate(decision, self.history)
        self._say(line, decision)

        result = {
            "moderator_line": line,
            "verdict": verdict,
            "decision": decision,
            "coverage": self.coverage.snapshot(),
            "finished": self.finished,
            "turns_used": self.participant_turns,
            "turn_budget": self.turn_budget,
        }
        self.turn_log.append(
            {
                "answer": answer,
                "answer_turn_id": turn_id,
                "verdict_status": verdict.status.value,
                "is_specific": verdict.is_specific,
                "action": decision.action.value,
                "target_objective_id": decision.target_objective_id,
                "rationale": decision.rationale,
                "moderator_line": line,  # the utterance this decision produced
            }
        )
        return result

    def save(self, sessions_dir=None):
        """Persist the speaker-attributed transcript to disk; returns the path."""
        return self.transcript.save(sessions_dir)
