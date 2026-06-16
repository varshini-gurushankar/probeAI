"""Transcript persistence tests — no LLM."""

from __future__ import annotations

from probeai.transcript import Transcript


def _sample() -> Transcript:
    t = Transcript("study-x", session_id="sess-1")
    t.append("moderator", "Hi, tell me about checkout?", action="open", rationale="opening")
    t.append("participant", "It was annoying.")
    t.append("moderator", "What made it annoying?", action="probe", rationale="vague")
    t.append("participant", "The payment screen kept failing.")
    return t


def test_turn_ids_are_sequential_and_speaker_attributed():
    t = _sample()
    assert [turn.turn_id for turn in t.turns] == [0, 1, 2, 3]
    assert t.turns[1].speaker == "participant"
    assert len(t.participant_turns()) == 2


def test_entries_shape_matches_engine_expectation():
    t = _sample()
    e = t.entries[0]
    assert set(e.keys()) == {"turn_id", "speaker", "text"}


def test_render_is_speaker_labeled():
    rendered = _sample().render()
    assert "Moderator:" in rendered and "Participant:" in rendered


def test_save_and_load_roundtrip(tmp_path):
    t = _sample()
    path = t.save(tmp_path)
    assert path.exists()
    loaded = Transcript.load(path)
    assert loaded.study_id == "study-x"
    assert loaded.session_id == "sess-1"
    assert [x.text for x in loaded.turns] == [x.text for x in t.turns]
    assert loaded.turns[2].action == "probe"
