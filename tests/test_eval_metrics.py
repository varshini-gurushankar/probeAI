"""Deterministic eval-metric + scenario-check tests (no LLM)."""

from __future__ import annotations

from probeai.evals import metrics
from probeai.evals.scenarios import (
    RunData,
    _check_ideal,
    _check_offtopic,
    _check_uncooperative,
    _check_vague,
    _check_volunteers,
)


def _cov(objective_id, status, priority="high", times_asked=1, probes_used=0):
    return {
        "objective_id": objective_id,
        "label": objective_id,
        "priority": priority,
        "status": status,
        "evidence_turn_ids": [1] if status != "uncovered" else [],
        "probes_used": probes_used,
        "times_asked": times_asked,
    }


# --- metrics -----------------------------------------------------------------


def test_objective_coverage_pct():
    snap = [_cov("a", "covered"), _cov("b", "partial"), _cov("c", "uncovered")]
    cov = metrics.objective_coverage(snap)
    assert cov["covered"] == 1
    assert cov["total"] == 3
    assert round(cov["pct"]) == 33


def test_one_question_compliance_counts_stacked():
    qs = ["What happened?", "What did you do and why did you stop?"]
    out = metrics.one_question_compliance(qs)
    assert out["stacked_count"] == 1
    assert out["total_questions"] == 2


def test_leading_report_flags_via_heuristics_without_judge():
    qs = ["What was it like?", "Don't you think it was slow?"]
    rep = metrics.leading_question_report(qs, judge=None)
    assert rep["count"] == 1
    assert rep["violations"][0]["source"] == "heuristic"


def test_extract_followups_pairs_probe_with_answer(study):
    turn_log = [
        {"action": "probe", "target_objective_id": "a_high", "answer": "It was annoying.",
         "moderator_line": "What made it annoying?"},
        {"action": "ask_next", "target_objective_id": "b_med", "answer": "ok",
         "moderator_line": "Tell me about signup?"},
    ]
    fus = metrics.extract_followups(turn_log, study)
    assert len(fus) == 1
    assert fus[0]["answer"] == "It was annoying."
    assert fus[0]["followup"] == "What made it annoying?"


# --- scenario checks (the headline ground-truth logic) -----------------------


def _run(**kw) -> RunData:
    base = dict(
        persona_id="p",
        turn_log=[],
        coverage=[],
        finished=True,
        participant_turns=3,
        turn_budget=9,
        transcript_turns=[],
    )
    base.update(kw)
    return RunData(**base)


def test_vague_passes_when_a_probe_happened():
    run = _run(turn_log=[{"action": "probe"}, {"action": "ask_next"}])
    assert _check_vague(run, None)[0] is True
    run2 = _run(turn_log=[{"action": "ask_next"}])
    assert _check_vague(run2, None)[0] is False


def test_volunteers_passes_when_objective_covered_without_being_asked():
    run = _run(coverage=[_cov("a", "covered", times_asked=1), _cov("b", "covered", times_asked=0)])
    assert _check_volunteers(run, None)[0] is True
    run2 = _run(coverage=[_cov("a", "covered", times_asked=1)])
    assert _check_volunteers(run2, None)[0] is False


def test_offtopic_passes_when_steer_back_happened():
    run = _run(turn_log=[{"action": "steer_back"}])
    assert _check_offtopic(run, None)[0] is True


def test_ideal_requires_few_probes_and_high_coverage():
    good = _run(
        turn_log=[{"action": "ask_next"}],
        coverage=[_cov("a", "covered"), _cov("b", "covered")],
    )
    assert _check_ideal(good, None)[0] is True
    too_many = _run(
        turn_log=[{"action": "probe"}, {"action": "probe"}],
        coverage=[_cov("a", "covered"), _cov("b", "covered")],
    )
    assert _check_ideal(too_many, None)[0] is False


def test_uncooperative_requires_graceful_termination(policy):
    ok = _run(
        finished=True,
        participant_turns=6,
        turn_budget=9,
        coverage=[_cov("a", "uncovered", priority="high", probes_used=1)],
    )
    assert _check_uncooperative(ok, policy)[0] is True
    looped = _run(
        finished=False,
        participant_turns=20,
        turn_budget=9,
        coverage=[_cov("a", "uncovered", priority="high", probes_used=1)],
    )
    assert _check_uncooperative(looped, policy)[0] is False
