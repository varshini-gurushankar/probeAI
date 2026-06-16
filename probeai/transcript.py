"""Append-only, speaker-attributed transcript with disk persistence.

A transcript is the raw material for everything downstream: the coverage panel
cites turn ids, synthesis quotes participant turns verbatim, and the eval harness
replays moderator turns. Persisted as JSONL (one turn per line) so it is
append-friendly and trivially diffable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Where sessions are written by default (gitignored).
DEFAULT_SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"


@dataclass
class Turn:
    turn_id: int
    speaker: str  # "moderator" | "participant"
    text: str
    # Optional moderator-side annotations (action taken, rationale) for the demo.
    action: Optional[str] = None
    rationale: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Transcript:
    """An ordered list of attributed turns for one interview."""

    def __init__(self, study_id: str, session_id: Optional[str] = None):
        self.study_id = study_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.turns: List[Turn] = []

    def append(
        self,
        speaker: str,
        text: str,
        *,
        action: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> Turn:
        turn = Turn(
            turn_id=len(self.turns),
            speaker=speaker,
            text=text,
            action=action,
            rationale=rationale,
        )
        self.turns.append(turn)
        return turn

    # --- views ---------------------------------------------------------------
    @property
    def entries(self) -> List[dict]:
        """Dialogue as the moderator engine expects it (speaker/text dicts)."""
        return [{"turn_id": t.turn_id, "speaker": t.speaker, "text": t.text} for t in self.turns]

    def participant_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.speaker == "participant"]

    def render(self) -> str:
        label = {"moderator": "Moderator", "participant": "Participant"}
        return "\n".join(f"{label.get(t.speaker, t.speaker)}: {t.text}" for t in self.turns)

    def get(self, turn_id: int) -> Turn:
        return self.turns[turn_id]

    # --- persistence ---------------------------------------------------------
    def save(self, sessions_dir: Path | str | None = None) -> Path:
        """Write the transcript to ``<sessions_dir>/<session_id>.jsonl``."""
        directory = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.session_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            # First line is a header record; the rest are turns.
            header = {"type": "header", "study_id": self.study_id, "session_id": self.session_id}
            fh.write(json.dumps(header) + "\n")
            for t in self.turns:
                fh.write(json.dumps({"type": "turn", **asdict(t)}) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Transcript":
        path = Path(path)
        transcript: Optional["Transcript"] = None
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("type") == "header":
                    transcript = cls(rec["study_id"], rec["session_id"])
                elif rec.get("type") == "turn":
                    assert transcript is not None, "transcript file missing header"
                    rec.pop("type")
                    transcript.turns.append(Turn(**rec))
        if transcript is None:
            raise ValueError(f"no header record in {path}")
        return transcript
