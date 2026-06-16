"""Synthesis grounding tests — no real LLM (a fake returns canned JSON).

We test the part that must never fail in a research deliverable: every shipped
highlight maps to a real objective and a quote that actually appears in a
participant turn. Ungrounded/invented quotes are dropped.
"""

from __future__ import annotations

from probeai.synthesis import synthesize
from probeai.transcript import Transcript


class FakeLLM:
    def __init__(self, payload: dict):
        self._payload = payload

    def complete_json(self, *_args, **_kwargs) -> dict:
        return self._payload


def _transcript() -> Transcript:
    t = Transcript("t", session_id="s")
    t.append("moderator", "Tell me about checkout?")
    t.append("participant", "I gave up at the payment screen because it kept rejecting my card.")
    t.append("moderator", "What about creating an account?")
    t.append("participant", "Being forced to sign up before buying really annoyed me.")
    return t


def test_keeps_grounded_highlights_and_links_turn_ids(study):
    payload = {
        "summary": "Participants abandon at payment and dislike forced signup.",
        "highlights": [
            {
                "objective_id": "a_high",
                "quote": "I gave up at the payment screen because it kept rejecting my card.",
                "insight": "Payment failures drive abandonment.",
            }
        ],
    }
    syn = synthesize(study, _transcript(), llm=FakeLLM(payload))
    assert len(syn.highlights) == 1
    assert syn.highlights[0].turn_id == 1  # the participant turn it came from


def test_drops_ungrounded_quote(study):
    payload = {
        "summary": "x",
        "highlights": [
            {"objective_id": "a_high", "quote": "A quote nobody ever said.", "insight": "x"}
        ],
    }
    syn = synthesize(study, _transcript(), llm=FakeLLM(payload))
    assert syn.highlights == []


def test_drops_highlight_with_unknown_objective(study):
    payload = {
        "summary": "x",
        "highlights": [
            {
                "objective_id": "not_a_real_objective",
                "quote": "Being forced to sign up before buying really annoyed me.",
                "insight": "x",
            }
        ],
    }
    syn = synthesize(study, _transcript(), llm=FakeLLM(payload))
    assert syn.highlights == []
