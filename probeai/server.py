"""FastAPI backend for the ProbeAI web app.

Endpoints (single local session — multi-tenant is an explicit non-goal):
    GET  /api/study        -> study metadata + objectives + empty coverage
    POST /api/start        -> begin an interview, returns the opening line
    POST /api/turn         -> submit a participant answer, get the next moderator line
    POST /api/participant  -> generate a simulated participant answer (sim toggle)
    POST /api/synthesize   -> findings summary + highlights (saved via gq_mock)
    POST /api/eval         -> single-interview quality report (live eval)
    POST /api/reset        -> discard the session

The front end (web/) handles mic STT + TTS playback; the brain lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import gq_mock
from .agent_graph import make_session
from .evals import metrics
from .evals.judge import Judge
from .llm import LLMError
from .moderator import InterviewSession
from .participant import SimulatedParticipant
from .study_config import load_policy, load_study
from .synthesis import synthesize

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="ProbeAI", version="0.1.0")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Local demo: never let the browser cache the front end, so edits to
    app.js / styles.css / index.html show up on a normal reload instead of a
    stale copy. (Harmless for the JSON API; only the static assets matter.)"""
    response = await call_next(request)
    if request.url.path in ("/", "/app.js", "/styles.css", "/index.html"):
        response.headers["Cache-Control"] = "no-store"
    return response


class _State:
    """Holds the single active interview session for this local demo."""

    def __init__(self) -> None:
        self.study = load_study()
        self.policy = load_policy()
        self.session: Optional[InterviewSession] = None
        self.participants: dict[str, SimulatedParticipant] = {}

    def reset(self) -> None:
        self.session = None
        self.participants = {}

    def participant(self, persona_id: str) -> SimulatedParticipant:
        if persona_id not in self.participants:
            self.participants[persona_id] = SimulatedParticipant(self.study, persona_id)
        return self.participants[persona_id]


state = _State()


# --- request models ----------------------------------------------------------
class StartIn(BaseModel):
    graph: bool = True  # LangGraph runtime is the default; pass {"graph": false} for classic


class TurnIn(BaseModel):
    answer: str


class ParticipantIn(BaseModel):
    persona_id: str = "realistic"
    moderator_line: str


class EvalIn(BaseModel):
    use_judge: bool = False  # default off to conserve free-tier quota


def _require_session() -> InterviewSession:
    if state.session is None:
        raise HTTPException(status_code=409, detail="No active interview. POST /api/start.")
    return state.session


def _llm_guard(exc: LLMError) -> None:
    msg = str(exc).lower()
    if "resource_exhausted" in msg or "quota" in msg:
        detail = "Gemini free-tier quota exhausted — resets ~midnight Pacific. Try again later."
    else:
        detail = f"LLM unavailable: {exc}"
    raise HTTPException(status_code=503, detail=detail)


# --- endpoints ---------------------------------------------------------------
@app.get("/api/study")
def get_study():
    s = state.study
    empty = state.session.coverage.snapshot() if state.session else _empty_coverage()
    return {
        "study_id": s.study_id,
        "title": s.title,
        "research_goal": s.research_goal,
        "objectives": [
            {"id": o.id, "label": o.label, "priority": o.priority.value,
             "description": o.description}
            for o in s.objectives
        ],
        "coverage": empty,
        "turn_budget": state.policy.turn_budget(len(s.objectives)),
        "in_progress": state.session is not None,
    }


def _empty_coverage():
    return [
        {"objective_id": o.id, "label": o.label, "priority": o.priority.value,
         "status": "uncovered", "evidence_turn_ids": [], "probes_used": 0, "times_asked": 0}
        for o in state.study.objectives
    ]


@app.post("/api/start")
def start(body: StartIn | None = None):
    # Graph is the default runtime; an explicit {"graph": false} opts into classic.
    use_graph = body.graph if body else True
    state.reset()
    try:
        state.session = make_session(state.study, state.policy, graph=use_graph)
        opening = state.session.start()
    except LLMError as exc:
        state.session = None
        _llm_guard(exc)
    return {
        "moderator_line": opening,
        "coverage": state.session.coverage.snapshot(),
        "turns_used": 0,
        "turn_budget": state.session.turn_budget,
        "finished": False,
        "moderator_runtime": "graph" if use_graph else "classic",
    }


@app.post("/api/turn")
def turn(body: TurnIn):
    session = _require_session()
    if session.finished:
        raise HTTPException(status_code=409, detail="Interview already finished.")
    try:
        result = session.step(body.answer)
    except LLMError as exc:
        _llm_guard(exc)
    return {
        "moderator_line": result["moderator_line"],
        "decision": {
            "action": result["decision"].action.value,
            "rationale": result["decision"].rationale,
        },
        "verdict": {
            "status": result["verdict"].status.value,
            "is_specific": result["verdict"].is_specific,
        },
        "coverage": result["coverage"],
        "turns_used": result["turns_used"],
        "turn_budget": result["turn_budget"],
        "finished": result["finished"],
        # Graph runtime exposes the per-node decision trace + validation result so a
        # reviewer can inspect assess -> coverage -> decide -> validate -> repair.
        "trace": result.get("trace"),
        "validation": result.get("validation"),
    }


@app.post("/api/participant")
def participant(body: ParticipantIn):
    """Generate a simulated participant answer (for the sim-vs-human toggle)."""
    try:
        answer = state.participant(body.persona_id).respond(body.moderator_line)
    except (LLMError, KeyError) as exc:
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=400, detail=f"Unknown persona: {exc}")
        _llm_guard(exc)
    return {"answer": answer}


@app.post("/api/synthesize")
def synthesize_endpoint():
    session = _require_session()
    try:
        syn = synthesize(state.study, session.transcript)
    except LLMError as exc:
        _llm_guard(exc)
    # Persist through the Great Question integration seam.
    session.save()
    gq_mock.save_transcript(state.study.study_id, session.transcript)
    gq_mock.save_highlights(state.study.study_id, syn)
    gq_mock.save_synthesis(state.study.study_id, syn)
    return syn.to_dict()


@app.post("/api/eval")
def eval_endpoint(body: EvalIn):
    """Single-interview quality report over the current transcript."""
    session = _require_session()
    judge = None
    if body.use_judge:
        try:
            judge = Judge()
        except LLMError as exc:
            _llm_guard(exc)

    questions = metrics.moderator_questions(session.transcript)
    followups = metrics.extract_followups(session.turn_log, state.study)
    report = {
        "coverage": metrics.objective_coverage(session.coverage.snapshot()),
        "leading": _safe(lambda: metrics.leading_question_report(questions, judge=judge),
                         fallback=lambda: metrics.leading_question_report(questions, judge=None)),
        "one_question": metrics.one_question_compliance(questions),
        "followup_relevance": None,
    }
    if judge is not None and followups:
        report["followup_relevance"] = _safe(
            lambda: metrics.followup_relevance(followups, judge), fallback=lambda: None
        )
    report["followups_captured"] = len(followups)
    return report


def _safe(fn, fallback):
    try:
        return fn()
    except LLMError:
        return fallback()


@app.post("/api/reset")
def reset():
    state.reset()
    return {"ok": True}


# --- static front end --------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR)), name="web")
