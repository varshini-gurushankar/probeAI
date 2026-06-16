"""Decision-logic tests — the pure agentic core, no LLM."""

from __future__ import annotations

from probeai.coverage import AnswerVerdict, CoverageState, Verdict
from probeai.moderator import Action, decide_next_action, pick_next_objective


def _decide(study, policy, coverage, current, status, turns_used):
    return decide_next_action(
        current_objective_id=current,
        last_verdict_status=status,
        coverage=coverage,
        policy=policy,
        study=study,
        turns_used=turns_used,
    )


# --- turn-budget + priority math --------------------------------------------


def test_turn_budget_scales_with_objectives_and_respects_floor(policy):
    # 3 objectives * 3.0 = 9, above the floor of 4.
    assert policy.turn_budget(3) == 9
    # 1 objective * 3.0 = 3, below floor -> floor wins.
    assert policy.turn_budget(1) == 4


def test_probe_caps_by_priority(policy):
    from probeai.study_config import Priority

    assert policy.probe_cap(Priority.high) == 2
    assert policy.probe_cap(Priority.medium) == 1
    assert policy.probe_cap(Priority.low) == 1


# --- pick_next_objective -----------------------------------------------------


def test_pick_next_prefers_high_priority_first(study, policy):
    cov = CoverageState(study)
    assert pick_next_objective(cov, study, policy, remaining_budget=9) == "a_high"


def test_pick_next_skips_already_asked_to_avoid_loops(study, policy):
    cov = CoverageState(study)
    cov.mark_asked("a_high")  # already introduced, still uncovered
    # Should move past it rather than loop back.
    assert pick_next_objective(cov, study, policy, remaining_budget=9) == "b_med"


def test_pick_next_skips_low_priority_when_budget_tight(study, policy):
    cov = CoverageState(study)
    cov.mark_asked("a_high")
    cov.mark_asked("b_med")
    # Only c_low remains, but budget is at/under the skip threshold -> skip it.
    assert pick_next_objective(cov, study, policy, remaining_budget=3) is None
    # With budget room, c_low is fair game.
    assert pick_next_objective(cov, study, policy, remaining_budget=9) == "c_low"


def test_pick_next_returns_none_when_all_covered(study, policy):
    cov = CoverageState(study)
    for oid in ("a_high", "b_med", "c_low"):
        cov.apply_verdict(Verdict(oid, AnswerVerdict.covered, is_specific=True), turn_id=0)
    assert pick_next_objective(cov, study, policy, remaining_budget=9) is None


# --- decide_next_action ------------------------------------------------------


def test_partial_answer_triggers_probe_within_cap(study, policy):
    cov = CoverageState(study)
    cov.mark_asked("a_high")
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.partial, is_specific=False), turn_id=1)
    d = _decide(study, policy, cov, "a_high", AnswerVerdict.partial, turns_used=1)
    assert d.action is Action.PROBE
    assert d.target_objective_id == "a_high"


def test_partial_answer_moves_on_when_probe_cap_reached(study, policy):
    cov = CoverageState(study)
    cov.mark_asked("a_high")
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.partial, is_specific=False), turn_id=1)
    # high cap is 2 -> exhaust it
    cov.record_probe("a_high")
    cov.record_probe("a_high")
    d = _decide(study, policy, cov, "a_high", AnswerVerdict.partial, turns_used=3)
    assert d.action is Action.ASK_NEXT
    assert d.target_objective_id == "b_med"


def test_off_topic_steers_back_then_moves_on_after_cap(study, policy):
    cov = CoverageState(study)
    cov.mark_asked("a_high")  # medium-priority style; use a_high (cap 2)
    d1 = _decide(study, policy, cov, "a_high", AnswerVerdict.off_topic, turns_used=1)
    assert d1.action is Action.STEER_BACK
    # Exhaust the redirect cap (2 for high) then it must move on, not loop.
    cov.record_probe("a_high")
    cov.record_probe("a_high")
    d2 = _decide(study, policy, cov, "a_high", AnswerVerdict.off_topic, turns_used=3)
    assert d2.action is Action.ASK_NEXT


def test_covered_answer_advances_to_next_objective(study, policy):
    cov = CoverageState(study)
    cov.mark_asked("a_high")
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.covered, is_specific=True), turn_id=1)
    d = _decide(study, policy, cov, "a_high", AnswerVerdict.covered, turns_used=1)
    assert d.action is Action.ASK_NEXT
    assert d.target_objective_id == "b_med"


def test_budget_exhausted_forces_close(study, policy):
    cov = CoverageState(study)
    cov.mark_asked("a_high")
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.partial, is_specific=False), turn_id=1)
    # turns_used == budget (9) -> must close even though a_high is only partial.
    d = _decide(study, policy, cov, "a_high", AnswerVerdict.partial, turns_used=9)
    assert d.action is Action.CLOSE


def test_uncooperative_path_closes_gracefully_without_looping(study, policy):
    """All objectives asked once but left uncovered -> close, never re-ask."""
    cov = CoverageState(study)
    for oid in ("a_high", "b_med", "c_low"):
        cov.mark_asked(oid)
    d = _decide(study, policy, cov, "c_low", AnswerVerdict.off_topic, turns_used=6)
    # c_low off_topic with cap room would steer back; but if cap is spent it closes.
    cov.record_probe("c_low")  # low cap = 1, now exhausted
    d = _decide(study, policy, cov, "c_low", AnswerVerdict.off_topic, turns_used=6)
    assert d.action is Action.CLOSE
