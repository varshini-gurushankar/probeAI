"""Mock of Great Question's integration surface.

I don't have access to Great Question's real API/MCP, so this module stands in for
it — and marks EXACTLY where ProbeAI would plug into their platform. Each function
below maps to a Great Question concept; in production these would be calls to their
MCP tools / REST API instead of local files.

    get_study(study_id)            -> GQ "Studies" (the discussion guide)
    save_transcript(study_id, ...) -> GQ "Interviews / Transcripts"
    save_highlights(study_id, ...) -> GQ "Highlights" (objective-linked excerpts)
    save_synthesis(study_id, ...)  -> GQ "Insights / Analysis"

Keeping this seam explicit is the point: swapping these bodies for real MCP calls
is the entire "productionize against Great Question" step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .study_config import Study, load_study
from .synthesis import Synthesis
from .transcript import Transcript

# Local stand-in for GQ's backend storage.
_STORE = Path(__file__).parent.parent / "data" / "gq_store"


def _study_path(study_id: str) -> Path:
    return _STORE / study_id


def get_study(study_id: str, study_path: Optional[str] = None) -> Study:
    """Fetch a study (discussion guide).

    PRODUCTION: GET the study from Great Question's API / `get_study` MCP tool.
    Here we load the local YAML discussion guide.
    """
    return load_study(study_path)


def save_transcript(study_id: str, transcript: Transcript) -> Path:
    """Persist a completed interview transcript.

    PRODUCTION: POST to Great Question's interviews/transcripts endpoint (or
    `save_transcript` MCP tool), attaching the recording + speaker-attributed
    turns to the study. Here we write JSONL locally.
    """
    out_dir = _study_path(study_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    return transcript.save(out_dir)


def save_highlights(study_id: str, synthesis: Synthesis) -> Path:
    """Persist objective-linked highlights (quoted excerpts).

    PRODUCTION: create Highlights in Great Question via their `save_highlights`
    MCP tool — each highlight references a transcript turn + the objective/tag it
    supports, exactly like the GQ Highlights feature.
    """
    out_dir = _study_path(study_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "highlights.json"
    payload = [h.__dict__ for h in synthesis.highlights]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_synthesis(study_id: str, synthesis: Synthesis) -> Path:
    """Persist the findings summary.

    PRODUCTION: write to Great Question's Insights/Analysis surface so the summary
    shows up alongside the study like their built-in AI analysis.
    """
    out_dir = _study_path(study_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthesis.json"
    path.write_text(json.dumps(synthesis.to_dict(), indent=2), encoding="utf-8")
    return path
