"""Simulated participant — an LLM playing a research subject.

Two jobs: (1) let you run a full interview solo for a live demo, and (2) drive the
offline eval harness's adversarial personas. The personas live in
config/personas.yaml so the behaviors are editable and inspectable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

from . import prompts
from .llm import LLMClient
from .study_config import CONFIG_DIR, Study

DEFAULT_PERSONAS_PATH = CONFIG_DIR / "personas.yaml"


def load_personas(path: Path | str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_PERSONAS_PATH
    return yaml.safe_load(p.read_text(encoding="utf-8"))


class SimulatedParticipant:
    """Answers the moderator in character for a given persona id."""

    def __init__(
        self,
        study: Study,
        persona_id: str,
        llm: Optional[LLMClient] = None,
        personas_path: Path | str | None = None,
    ):
        personas = load_personas(personas_path)
        if persona_id not in personas:
            raise KeyError(
                f"unknown persona '{persona_id}'. Available: {sorted(personas)}"
            )
        self.study = study
        self.persona_id = persona_id
        self.persona = personas[persona_id]
        # Slightly higher temperature so personas feel human, not canned.
        self.llm = llm or LLMClient(temperature=0.8)
        self._history: List[str] = []

    @property
    def system_prompt(self) -> str:
        return prompts.PARTICIPANT_SYSTEM.format(
            persona_label=self.persona["label"],
            persona_behavior=self.persona["behavior"].strip(),
            research_goal=self.study.research_goal,
        )

    def respond(self, moderator_line: str) -> str:
        """Produce the participant's reply to the moderator's latest line."""
        user = prompts.PARTICIPANT_TURN.format(moderator_line=moderator_line)
        reply = self.llm.complete(user, system=self.system_prompt)
        self._history.append(moderator_line)
        self._history.append(reply)
        return reply


class ScriptedParticipant:
    """A deterministic, zero-LLM participant that replays a fixed answer script.

    Used by the offline eval harness: it makes runs reproducible (the only moving
    part is the moderator) and costs no API calls for the participant side — which
    matters a lot on a tight free-tier quota. When the script is exhausted it holds
    on the final line (so a moderator that over-asks still gets a stable answer).
    """

    def __init__(self, script: list[str]):
        if not script:
            raise ValueError("scripted participant needs at least one line")
        self.script = script
        self._i = 0

    def respond(self, _moderator_line: str = "") -> str:
        line = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return line
