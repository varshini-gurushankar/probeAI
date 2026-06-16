"""Leading-question + stacked-question heuristic tests — pure string logic."""

from __future__ import annotations

import pytest

from probeai.evals import heuristics


@pytest.mark.parametrize(
    "question",
    [
        "Don't you think the checkout was confusing?",
        "Wouldn't you agree the form was too long?",
        "The payment step was frustrating, right?",
        "What frustrated you about creating an account?",  # presupposition
        "How confusing was the payment screen?",  # adjective presumes verdict
    ],
)
def test_leading_questions_are_flagged(question):
    assert heuristics.find_leading_violations(question)


@pytest.mark.parametrize(
    "question",
    [
        "What was the checkout like for you?",
        "Walk me through the moment you decided to stop.",
        "Did the app ask you to create an account?",  # neutral yes/no is allowed
        "How did that feel at that point?",
    ],
)
def test_neutral_questions_are_clean(question):
    assert heuristics.find_leading_violations(question) == []


def test_stacked_detected_by_multiple_question_marks():
    assert heuristics.is_stacked("What happened? And how did you feel?")


def test_stacked_detected_by_joined_interrogatives():
    assert heuristics.is_stacked("What did you do and why did you stop?")


def test_single_question_is_not_stacked():
    assert not heuristics.is_stacked("What was the payment step like?")


def test_analyze_question_combines_both():
    flags = heuristics.analyze_question("Don't you think it was slow and clunky?")
    assert flags.is_leading
    assert not flags.is_clean
