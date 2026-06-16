"""Shared fixtures: small hand-built studies/policies so tests stay deterministic.

None of these tests call an LLM — they exercise the pure coverage + decision
logic only (the moderator's assess/generate steps are the eval-harness layer).
"""

from __future__ import annotations

import pytest

from probeai.study_config import Objective, Policy, Priority, ProbeTrigger, Study


def make_objective(oid: str, priority: Priority) -> Objective:
    return Objective(
        id=oid,
        priority=priority,
        label=f"Objective {oid}",
        description=f"Learn about {oid}.",
        seed_questions=[f"Tell me about {oid}?"],
    )


@pytest.fixture
def study() -> Study:
    """Three objectives: high, medium, low (in that authored order)."""
    return Study(
        study_id="t",
        title="Test study",
        research_goal="Understand testing.",
        objectives=[
            make_objective("a_high", Priority.high),
            make_objective("b_med", Priority.medium),
            make_objective("c_low", Priority.low),
        ],
    )


@pytest.fixture
def policy() -> Policy:
    return Policy(
        turn_budget_per_objective=3.0,
        turn_budget_floor=4,
        probe_caps={Priority.high: 2, Priority.medium: 1, Priority.low: 1},
        low_priority_skip_threshold=4,
        probe_trigger=ProbeTrigger(specificity_is_primary=True, short_answer_word_count=6),
        require_all_objectives_to_close=False,
    )
