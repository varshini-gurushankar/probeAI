"""Config loading + validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from probeai.study_config import (
    Objective,
    Priority,
    Study,
    load_policy,
    load_study,
)


def test_sample_study_loads_and_validates():
    study = load_study()  # the shipped sample
    assert study.objectives
    assert study.research_goal
    # All priorities parse into the enum.
    assert all(isinstance(o.priority, Priority) for o in study.objectives)


def test_sample_policy_loads():
    policy = load_policy()
    assert policy.probe_cap(Priority.high) >= policy.probe_cap(Priority.low)
    assert policy.turn_budget(5) >= policy.turn_budget_floor


def test_duplicate_objective_ids_rejected():
    with pytest.raises(ValidationError):
        Study(
            study_id="x",
            title="x",
            research_goal="x",
            objectives=[
                Objective(id="dup", priority=Priority.high, label="a", description="a", seed_questions=["q"]),
                Objective(id="dup", priority=Priority.low, label="b", description="b", seed_questions=["q"]),
            ],
        )


def test_objective_requires_a_seed_question():
    with pytest.raises(ValidationError):
        Objective(id="a", priority=Priority.high, label="a", description="a", seed_questions=[])


def test_objective_lookup_by_id():
    study = load_study()
    first = study.objectives[0]
    assert study.objective(first.id) is first
    with pytest.raises(KeyError):
        study.objective("nope")
