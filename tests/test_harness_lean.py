"""Tests for the quota-lean harness pieces: scripted participant + lean generation."""

from __future__ import annotations

from probeai.moderator import Action, Decision, Moderator
from probeai.participant import ScriptedParticipant


def test_scripted_participant_replays_then_holds_last():
    p = ScriptedParticipant(["one", "two"])
    assert p.respond("q1") == "one"
    assert p.respond("q2") == "two"
    assert p.respond("q3") == "two"  # holds the final line when exhausted


class _RecordingLLM:
    """Stand-in LLM that records whether it was called."""

    def __init__(self):
        self.calls = 0

    def complete(self, *_a, **_k):
        self.calls += 1
        return "GENERATED PROBE"


def _moderator(study, policy, lean):
    return Moderator(study, policy, llm=_RecordingLLM(), lean=lean)


def test_lean_templates_non_probe_lines_without_llm(study, policy):
    mod = _moderator(study, policy, lean=True)
    # ASK_NEXT -> the objective's seed question, no LLM call.
    line = mod.generate(Decision(Action.ASK_NEXT, "a_high", "r"), [])
    assert line == study.objective("a_high").seed_questions[0]
    assert mod.llm.calls == 0
    # CLOSE -> the study outro, still no LLM call.
    mod.generate(Decision(Action.CLOSE, None, "done"), [])
    assert mod.llm.calls == 0


def test_lean_still_uses_llm_for_probes(study, policy):
    mod = _moderator(study, policy, lean=True)
    line = mod.generate(Decision(Action.PROBE, "a_high", "r", gap="what specifically"), [])
    assert line == "GENERATED PROBE"
    assert mod.llm.calls == 1


def test_rich_mode_uses_llm_for_every_line(study, policy):
    mod = _moderator(study, policy, lean=False)
    mod.generate(Decision(Action.ASK_NEXT, "a_high", "r"), [])
    assert mod.llm.calls == 1
