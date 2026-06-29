"""LangGraph moderator runtime tests — fully offline (a fake moderator, no LLM).

These mirror the deterministic-only style of the other tests: the LLM steps
(assess / generate / repair) are replaced by a programmable fake so routing, the
coverage/decision policy, and the validation/repair loop are all exercised without a
single network call.
"""

from __future__ import annotations

import pytest

from probeai.agent_graph import (
    GENERIC_FALLBACK,
    GraphInterviewSession,
    build_interview_graph,
    validate_question,
)
from probeai.coverage import AnswerVerdict, CoverageStatus, Verdict


class _FakeLLM:
    """Stands in for moderator.llm — records calls, returns a canned repair reply."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, user: str, system: str | None = None, **kw) -> str:
        self.calls += 1
        return self.reply


class _FakeModerator:
    """A moderator with no LLM: assess() returns a fixed verdict, generate() returns a
    grounded (objective-referencing, so it passes the relevance guard) question unless
    a leading/stacked override is supplied to trigger the repair loop.
    """

    def __init__(
        self,
        study,
        policy,
        *,
        status: AnswerVerdict,
        is_specific: bool = False,
        gap: str = "specifics",
        question_override: str | None = None,
        repair_reply: str = "Could you walk me through objective a_high in more depth?",
    ):
        self.study = study
        self.policy = policy
        self.status = status
        self.is_specific = is_specific
        self.gap = gap
        self.question_override = question_override
        self.llm = _FakeLLM(repair_reply)
        self.assess_calls = 0
        self.generate_calls = 0

    def assess(self, answer, current_objective_id, history) -> Verdict:
        self.assess_calls += 1
        return Verdict(
            current_objective_id=current_objective_id,
            status=self.status,
            is_specific=self.is_specific,
            gap=self.gap,
        )

    def generate(self, decision, history) -> str:
        self.generate_calls += 1
        if self.question_override is not None:
            return self.question_override
        oid = decision.target_objective_id
        label = self.study.objective(oid).label if oid else "this topic"
        # References the objective so the deterministic relevance guard passes.
        return f"Could you describe {label.lower()} in a bit more depth?"


def _graph_session(study, policy, mod, *, max_repairs: int = 2) -> GraphInterviewSession:
    session = GraphInterviewSession(study, policy, mod, max_repairs=max_repairs)
    session.start()  # opening line (OPEN) — sets current objective to a_high
    return session


# --- graph construction ------------------------------------------------------
def test_graph_builds_with_all_nodes():
    compiled = build_interview_graph()
    nodes = set(compiled.get_graph().nodes)
    for expected in (
        "assess_answer",
        "update_coverage",
        "decide_action",
        "generate_question",
        "validate_question",
        "repair_question",
        "synthesize",
    ):
        assert expected in nodes


# --- routing through one normal turn ----------------------------------------
def test_normal_turn_routes_assess_coverage_decide_generate_validate(study, policy):
    mod = _FakeModerator(study, policy, status=AnswerVerdict.partial)
    session = _graph_session(study, policy, mod)

    result = session.step("It was kind of annoying I guess.")

    nodes = [e["node"] for e in result["trace"]]
    assert nodes == [
        "assess_answer",
        "update_coverage",
        "decide_action",
        "generate_question",
        "validate_question",
    ]
    assert result["validation"]["valid"] is True


def test_partial_answer_routes_to_probe(study, policy):
    mod = _FakeModerator(study, policy, status=AnswerVerdict.partial)
    session = _graph_session(study, policy, mod)
    result = session.step("Eh, it was slow.")
    assert result["decision"].action.value == "probe"
    assert result["decision"].target_objective_id == "a_high"


def test_offtopic_answer_routes_to_steer_back(study, policy):
    mod = _FakeModerator(study, policy, status=AnswerVerdict.off_topic)
    session = _graph_session(study, policy, mod)
    result = session.step("Anyway, the weather has been wild lately.")
    assert result["decision"].action.value == "steer_back"


def test_covered_answer_routes_to_move_on(study, policy):
    mod = _FakeModerator(study, policy, status=AnswerVerdict.covered, is_specific=True)
    session = _graph_session(study, policy, mod)
    result = session.step("I abandoned at the payment screen because Apple Pay failed twice.")
    assert result["decision"].action.value == "ask_next"
    assert result["decision"].target_objective_id == "b_med"
    assert session.current_objective_id == "b_med"


def test_wrap_condition_routes_to_synthesize_and_ends(study, policy):
    mod = _FakeModerator(study, policy, status=AnswerVerdict.covered, is_specific=True)
    session = _graph_session(study, policy, mod)
    # Leave no fresh objective: pre-cover the other two so a covered a_high -> CLOSE.
    session.coverage.objectives["b_med"].status = CoverageStatus.covered
    session.coverage.objectives["c_low"].status = CoverageStatus.covered

    result = session.step("That fully answers it — the saved-cart flow was the fix.")

    assert result["decision"].action.value == "close"
    assert result["finished"] is True
    nodes = [e["node"] for e in result["trace"]]
    assert "synthesize" in nodes
    assert "generate_question" not in nodes  # CLOSE skips the question path


# --- validation / repair loop ------------------------------------------------
def test_leading_question_triggers_repair_then_emits_clean(study, policy):
    # generate() emits a leading question; the (faked) repair returns a clean one.
    mod = _FakeModerator(
        study,
        policy,
        status=AnswerVerdict.partial,
        question_override="Don't you think that checkout was frustrating?",
        repair_reply="Could you walk me through objective a_high in more depth?",
    )
    session = _graph_session(study, policy, mod)

    result = session.step("It was a bit much.")

    nodes = [e["node"] for e in result["trace"]]
    assert "repair_question" in nodes
    assert mod.llm.calls == 1  # exactly one repair attempt
    assert result["moderator_line"] == "Could you walk me through objective a_high in more depth?"
    assert result["validation"]["valid"] is True


def test_repair_loop_stops_after_max_attempts_and_falls_back(study, policy):
    # Both generate() and every repair stay leading -> exhaust repairs -> safe fallback.
    mod = _FakeModerator(
        study,
        policy,
        status=AnswerVerdict.partial,
        question_override="Don't you think that was awful?",
        repair_reply="Wasn't it obviously the worst part?",
    )
    session = _graph_session(study, policy, mod, max_repairs=2)

    result = session.step("Sort of.")

    assert mod.llm.calls == 2  # two repair attempts, then give up
    assert result["moderator_line"] == GENERIC_FALLBACK
    last = session.turn_log[-1]
    assert last["fallback_used"] is True
    assert last["repair_attempts"] == 2


# --- the validate_question helper (pure) ------------------------------------
def test_validate_flags_leading_and_stacked():
    assert validate_question("Don't you think it was slow?", objective_label="speed")
    assert validate_question(
        "What broke and how did you feel?", answer="checkout broke", objective_label="checkout"
    )


def test_validate_passes_clean_grounded_question():
    reasons = validate_question(
        "What happened right before you stopped?",
        answer="I stopped at the payment screen",
        objective_label="The moment they stopped",
    )
    assert reasons == []


# --- full scripted interview parity smoke ------------------------------------
def test_graph_session_runs_full_scripted_interview(study, policy):
    from probeai.participant import ScriptedParticipant
    from probeai.runner import run_interview

    mod = _FakeModerator(study, policy, status=AnswerVerdict.covered, is_specific=True)
    session = GraphInterviewSession(study, policy, mod)
    participant = ScriptedParticipant(["a", "b", "c", "d", "e"])

    run_interview(session, participant)

    assert session.finished is True
    assert session.turn_log  # produced per-turn records
    assert all("trace" in entry for entry in session.turn_log)


# --- classic vs. graph behavioral parity (the eval-trust guarantee, offline) -
class _ParityModerator:
    """Deterministic moderator (no LLM): the verdict depends only on the answer text, so
    the classic engine and the graph receive identical assessments and must therefore
    make identical decisions. ``generate`` references the objective so the graph's
    validation passes cleanly (no repair) — isolating the runtime, not the wording.
    """

    def __init__(self, study, policy):
        self.study = study
        self.policy = policy
        self.llm = _FakeLLM("Could you walk me through objective a_high in more depth?")

    def assess(self, answer, current_objective_id, history) -> Verdict:
        specific = len(answer.split()) >= 6  # longer answers read as substantive
        return Verdict(
            current_objective_id=current_objective_id,
            status=AnswerVerdict.covered if specific else AnswerVerdict.partial,
            is_specific=specific,
            gap="specifics",
        )

    def generate(self, decision, history) -> str:
        oid = decision.target_objective_id
        label = self.study.objective(oid).label if oid else "this topic"
        return f"Could you describe {label.lower()} in a bit more depth?"


def _behavior(session) -> dict:
    """The signals the eval scenarios actually score — wording deliberately excluded."""
    return {
        "actions": [e["action"] for e in session.turn_log],
        "verdicts": [e["verdict_status"] for e in session.turn_log],
        "finished": session.finished,
        "participant_turns": session.participant_turns,
        "coverage": {c["objective_id"]: c["status"] for c in session.coverage.snapshot()},
        "probes": {c["objective_id"]: c["probes_used"] for c in session.coverage.snapshot()},
        "covered_count": session.coverage.covered_count,
    }


def test_classic_and_graph_are_behaviorally_identical(study, policy):
    """The eval-trust guarantee, proven without the LLM: the same scripted interview run
    through the classic engine and through the graph produces the *same behavior* —
    identical action sequence, coverage, probe counts, and termination. The graph only
    adds a question-wording gate on top of an unchanged decision path.
    """
    from probeai.moderator import InterviewSession
    from probeai.participant import ScriptedParticipant
    from probeai.runner import run_interview

    # A mix of thin and substantive answers to exercise PROBE, ASK_NEXT and CLOSE.
    script = [
        "yeah",
        "it crashed at the payment screen after Apple Pay failed twice in a row",
        "nope",
        "the saved-cart flow is what finally let me finish the order properly",
        "meh",
        "honestly the whole thing felt smooth once the cart persisted between sessions",
    ]

    classic = InterviewSession(study, policy, _ParityModerator(study, policy))
    run_interview(classic, ScriptedParticipant(list(script)))

    graph = GraphInterviewSession(study, policy, _ParityModerator(study, policy))
    run_interview(graph, ScriptedParticipant(list(script)))

    assert _behavior(graph) == _behavior(classic)
