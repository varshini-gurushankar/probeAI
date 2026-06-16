"""Coverage state-machine tests — deterministic given a Verdict."""

from __future__ import annotations

from probeai.coverage import (
    AnswerVerdict,
    CoverageState,
    CoverageStatus,
    ObjectiveAssessment,
    Verdict,
)


def test_covered_verdict_sets_status_and_records_evidence(study):
    cov = CoverageState(study)
    cov.apply_verdict(
        Verdict("a_high", AnswerVerdict.covered, is_specific=True), turn_id=3
    )
    a = cov.get("a_high")
    assert a.status is CoverageStatus.covered
    assert a.evidence_turn_ids == [3]


def test_partial_then_covered_upgrades_and_keeps_all_evidence(study):
    cov = CoverageState(study)
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.partial, is_specific=False), turn_id=1)
    assert cov.get("a_high").status is CoverageStatus.partial
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.covered, is_specific=True), turn_id=2)
    a = cov.get("a_high")
    assert a.status is CoverageStatus.covered
    assert a.evidence_turn_ids == [1, 2]


def test_status_never_downgrades(study):
    cov = CoverageState(study)
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.covered, is_specific=True), turn_id=1)
    # A later vague aside must not undo solid coverage.
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.partial, is_specific=False), turn_id=2)
    assert cov.get("a_high").status is CoverageStatus.covered


def test_off_topic_changes_nothing(study):
    cov = CoverageState(study)
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.off_topic, is_specific=False), turn_id=1)
    a = cov.get("a_high")
    assert a.status is CoverageStatus.uncovered
    assert a.evidence_turn_ids == []


def test_volunteered_info_covers_a_later_objective(study):
    """Participant answering 'a_high' also volunteers specifics about 'b_med'."""
    cov = CoverageState(study)
    cov.apply_verdict(
        Verdict(
            "a_high",
            AnswerVerdict.covered,
            is_specific=True,
            also_addressed=[ObjectiveAssessment("b_med", AnswerVerdict.covered)],
        ),
        turn_id=4,
    )
    assert cov.get("a_high").status is CoverageStatus.covered
    assert cov.get("b_med").status is CoverageStatus.covered
    assert cov.get("b_med").evidence_turn_ids == [4]


def test_also_addressed_ignores_unknown_objective_ids(study):
    cov = CoverageState(study)
    cov.apply_verdict(
        Verdict(
            "a_high",
            AnswerVerdict.partial,
            is_specific=False,
            also_addressed=[ObjectiveAssessment("does_not_exist", AnswerVerdict.covered)],
        ),
        turn_id=1,
    )
    assert set(cov.objectives.keys()) == {"a_high", "b_med", "c_low"}


def test_coverage_pct_and_counts(study):
    cov = CoverageState(study)
    assert cov.total == 3
    assert cov.covered_count == 0
    cov.apply_verdict(Verdict("a_high", AnswerVerdict.covered, is_specific=True), turn_id=1)
    assert cov.covered_count == 1
    assert cov.coverage_pct() == 100.0 / 3
    assert not cov.is_fully_covered()
