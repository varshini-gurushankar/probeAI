"""LangGraph moderator runtime — the interview turn as an explicit state machine.

The classic engine (``moderator.InterviewSession.step``) runs one turn as a hidden
imperative sequence. This module models that exact same turn as a LangGraph graph so
the control flow is visible and inspectable, and adds a question validation/repair
loop that the classic path doesn't have:

    assess_answer -> update_coverage -> decide_action
        -> (CLOSE) synthesize ----------------------------> END
        -> generate_question -> validate_question
               -> valid ---------------------------------> END (emit)
               -> invalid & repairs left -> repair_question -> validate_question ...
               -> invalid & exhausted ----> fallback ------> END

Design principle: *agentic behavior with deterministic policy control*. Every node
reuses existing logic — the LLM only does language understanding (assess) and
generation (generate/repair); deterministic code (``decide_next_action``, the coverage
state machine, the leading/stacked heuristics) owns all policy. Nothing here
re-implements business logic; it orchestrates the modules that already exist.

The graph is the unit of ONE turn. The outer "repeat until wrap" loop stays in
``runner.run_interview`` — it just calls a ``GraphInterviewSession`` whose ``step`` runs
the graph instead of the classic sequence.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import prompts
from .coverage import CoverageState, Verdict
from .evals.heuristics import analyze_question
from .moderator import (
    Action,
    Decision,
    InterviewSession,
    Moderator,
    decide_next_action,
)
from .study_config import Policy, Study
from .transcript import Transcript

# Safe, neutral question used only when repair can't produce a clean one.
GENERIC_FALLBACK = "Can you tell me a little more about that?"
DEFAULT_MAX_REPAIRS = 2


# =============================================================================
# STATE
# =============================================================================
class InterviewGraphState(TypedDict, total=False):
    """Everything one turn needs, plus a record of how it got there.

    Two blocks. The first are live references shared across the whole interview — the
    graph mutates ``coverage``/``transcript`` in place, exactly as the classic engine
    does, so identity is stable turn to turn. The second is per-turn working data; each
    field's comment names the node that *writes* it, so you can read the state object and
    know exactly where every value comes from.

    ``trace`` is special: it uses an add-reducer (``operator.add``), so every node can
    append its own one-line record without overwriting earlier nodes — that's what makes
    the whole turn inspectable end to end.
    """

    # --- shared references (seeded once per turn from the live session) -------------
    study: Study                      # the discussion guide + objectives
    policy: Policy                    # follow-up caps, turn budget, stop rules
    transcript: Transcript            # speaker-attributed log (mutated in place)
    moderator: Moderator              # owns assess() / generate() / the LLM handle
    coverage: CoverageState           # per-objective state machine (mutated in place)
    judge: Optional[Any]              # optional evals.judge.Judge for a richer validation pass
    use_judge: bool                   # opt-in to the LLM judge in validation (default off)

    # --- per-turn working data (← the node that writes it) --------------------------
    answer: str                       # ← seeded: the participant's latest utterance
    answer_turn_id: int               # ← seeded: transcript id of that answer (for evidence)
    current_objective_id: Optional[str]  # ← seeded; rewritten by decide_action (on ASK_NEXT)
    verdict: Verdict                  # ← assess_answer_node  (status + is_specific + gap)
    decision: Decision                # ← decide_action_node  (PROBE/ASK_NEXT/STEER_BACK/CLOSE)
    question: str                     # ← generate/repair/fallback/synthesize (the line to emit)
    validation: dict                  # ← validate_question_node  {"valid": bool, "reasons": [...]}
    repair_attempts: int              # ← repair_question_node  (incremented per rewrite)
    max_repairs: int                  # ← seeded: hard cap on repair rewrites (default 2)
    fallback_used: bool               # ← fallback_node  (True iff repair was exhausted)
    turns_used: int                   # ← seeded: participant turns so far (drives the budget)
    turn_budget: int                  # ← seeded: max participant turns for this study
    should_wrap: bool                 # ← decide_action_node (CLOSE) / synthesize_node
    final_summary: Optional[str]      # ← synthesize_node  (the closing line)
    trace: Annotated[List[dict], operator.add]  # ← every node appends one record (add-reducer)


# =============================================================================
# TRACE + VALIDATION HELPERS
# =============================================================================
def _summ(text: Optional[str], n: int = 90) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def _trace(node: str, inp: str, out: str, reason: str = "") -> List[dict]:
    """Return a one-element trace list (merged into state via the add-reducer)."""
    return [{"node": node, "input": _summ(inp), "output": _summ(out), "reason": _summ(reason, 140)}]


_STOPWORDS = {
    "the", "and", "for", "was", "were", "you", "your", "that", "this", "with", "what",
    "how", "why", "when", "did", "does", "are", "can", "could", "would", "will", "have",
    "had", "about", "tell", "more", "just", "they", "them", "their", "there", "from",
    "into", "out", "but", "not", "all", "any", "some", "one", "ago", "got", "get",
}


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z']+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _looks_relevant(question: str, answer: str, objective_label: str, gap: str) -> bool:
    """Conservative relevance guard: does the question share ANY content word with the
    participant's answer / current objective / probe gap? Only a complete miss is
    flagged, so a clean grounded question never triggers a needless repair LLM call.
    """
    qwords = _content_words(question)
    if not qwords:
        return False
    context = _content_words(" ".join([answer, objective_label, gap]))
    if not context:
        return True  # nothing to compare against — don't flag
    return bool(qwords & context)


def validate_question(
    question: str,
    *,
    answer: str = "",
    objective_label: str = "",
    gap: str = "",
    judge: Optional[Any] = None,
    use_judge: bool = False,
) -> List[str]:
    """Deterministic question hygiene check; returns reasons (empty == clean).

    Reuses the already-tested heuristics in ``evals.heuristics`` for leading/loaded/
    presuppositional/double-barreled phrasing, plus a cheap relevance guard. An LLM
    judge second opinion is opt-in (and only runs when the heuristics found nothing,
    to keep free-tier cost down).
    """
    if not question or not question.strip():
        return ["empty question"]
    flags = analyze_question(question)
    reasons: List[str] = list(flags.leading_reasons)
    if flags.stacked:
        reasons.append("double-barreled / more than one question at once")
    if not _looks_relevant(question, answer, objective_label, gap):
        reasons.append("appears unrelated to the participant's answer/objective")
    if use_judge and judge is not None and not reasons:
        try:
            verdict = judge.leading_question(question)
            if verdict.leading:
                reasons.append(f"judge: {verdict.reason or 'leading'}")
        except Exception:  # judge is best-effort; never break a turn on it
            pass
    return reasons


def _intent_for(decision: Optional[Decision], study: Study) -> str:
    obj = (
        study.objective(decision.target_objective_id)
        if decision and decision.target_objective_id
        else None
    )
    label = obj.label if obj else "the current topic"
    if decision and decision.action is Action.PROBE:
        return f"dig into the gap \"{decision.gap}\" on the objective '{label}'"
    if decision and decision.action is Action.STEER_BACK:
        return f"steer the participant back to the objective '{label}'"
    return f"explore the objective '{label}'"


def _objective_label(state: InterviewGraphState) -> str:
    decision = state.get("decision")
    if decision and decision.target_objective_id:
        return state["study"].objective(decision.target_objective_id).label
    return ""


# =============================================================================
# NODES
# =============================================================================
def assess_answer_node(state: InterviewGraphState) -> dict:
    """LLM: judge the latest answer -> Verdict (drives coverage)."""
    mod = state["moderator"]
    answer = state["answer"]
    verdict = mod.assess(answer, state["current_objective_id"], state["transcript"].entries)
    out = f"status={verdict.status.value} specific={verdict.is_specific}"
    return {"verdict": verdict, "trace": _trace("assess_answer", answer, out, verdict.gap)}


def update_coverage_node(state: InterviewGraphState) -> dict:
    """Deterministic: fold the verdict into the coverage state machine (in place)."""
    cov = state["coverage"]
    verdict = state["verdict"]
    cov.apply_verdict(verdict, state["answer_turn_id"])
    out = f"{cov.covered_count}/{cov.total} objectives covered"
    return {"trace": _trace("update_coverage", verdict.status.value, out)}


def decide_action_node(state: InterviewGraphState) -> dict:
    """Deterministic policy: pick the next move + do the same bookkeeping as step()."""
    decision = decide_next_action(
        current_objective_id=state["current_objective_id"],
        last_verdict_status=state["verdict"].status,
        coverage=state["coverage"],
        policy=state["policy"],
        study=state["study"],
        turns_used=state["turns_used"],
    )
    updates: dict = {"decision": decision}
    cov = state["coverage"]
    if decision.action in (Action.PROBE, Action.STEER_BACK):
        cov.record_probe(decision.target_objective_id)
    elif decision.action is Action.ASK_NEXT:
        updates["current_objective_id"] = decision.target_objective_id
        cov.mark_asked(decision.target_objective_id)
    elif decision.action is Action.CLOSE:
        updates["should_wrap"] = True
    updates["trace"] = _trace(
        "decide_action", state["verdict"].status.value, decision.action.value, decision.rationale
    )
    return updates


def generate_question_node(state: InterviewGraphState) -> dict:
    """LLM (or template, in lean mode): turn the Decision into one utterance.

    Covers PROBE / ASK_NEXT (move-on) / STEER_BACK (redirect) — the existing
    ``Moderator.generate`` already branches on the action, so one node serves all three.
    """
    mod = state["moderator"]
    decision = state["decision"]
    question = mod.generate(decision, state["transcript"].entries)
    return {"question": question, "trace": _trace("generate_question", decision.action.value, question)}


def validate_question_node(state: InterviewGraphState) -> dict:
    """Deterministic quality gate before any moderator question is emitted."""
    question = state["question"]
    decision = state.get("decision")
    reasons = validate_question(
        question,
        answer=state.get("answer", ""),
        objective_label=_objective_label(state),
        gap=decision.gap if decision else "",
        judge=state.get("judge"),
        use_judge=bool(state.get("use_judge")),
    )
    valid = not reasons
    return {
        "validation": {"valid": valid, "reasons": reasons},
        "trace": _trace(
            "validate_question", question, "valid" if valid else "invalid", "; ".join(reasons)
        ),
    }


def repair_question_node(state: InterviewGraphState) -> dict:
    """LLM: rewrite a flagged question to be open, neutral, single, and grounded."""
    mod = state["moderator"]
    study = state["study"]
    reasons = state["validation"]["reasons"]
    attempt = state["repair_attempts"] + 1
    system = prompts.MODERATOR_SYSTEM.format(research_goal=study.research_goal)
    user = prompts.MODERATOR_REPAIR.format(
        question=state["question"],
        problems="; ".join(reasons) or "leading/stacked phrasing",
        answer=state.get("answer", ""),
        intent=_intent_for(state.get("decision"), study),
    )
    new_q = mod.llm.complete(user, system=system).strip()
    return {
        "question": new_q,
        "repair_attempts": attempt,
        "trace": _trace(
            "repair_question", "; ".join(reasons), new_q, f"attempt {attempt}/{state['max_repairs']}"
        ),
    }


def fallback_node(state: InterviewGraphState) -> dict:
    """Repair exhausted: emit a safe, neutral generic question rather than a bad one."""
    reasons = state["validation"]["reasons"]
    return {
        "question": GENERIC_FALLBACK,
        "fallback_used": True,
        "trace": _trace(
            "fallback", "; ".join(reasons), GENERIC_FALLBACK, "repair exhausted — safe generic question"
        ),
    }


def synthesize_node(state: InterviewGraphState) -> dict:
    """WRAP terminal: emit the closing line. (Full findings synthesis stays a separate
    post-interview call — see synthesis.synthesize — to avoid an LLM cost every wrap.)
    """
    mod = state["moderator"]
    decision = state["decision"]
    line = mod.generate(decision, state["transcript"].entries)
    return {
        "question": line,
        "should_wrap": True,
        "final_summary": line,
        "trace": _trace("synthesize", decision.action.value, line, "wrap-up"),
    }


# =============================================================================
# ROUTING
# =============================================================================
def _route_after_decide(state: InterviewGraphState) -> str:
    """CLOSE wraps the interview (no question needed); everything else generates one."""
    return "synthesize" if state["decision"].action is Action.CLOSE else "generate_question"


def _route_after_validate(state: InterviewGraphState) -> str:
    """The repair loop's exit logic — and the reason it always terminates.

    Three outcomes, checked in order:
      1. clean question                  -> emit (END)
      2. flagged, repair budget remains  -> repair_question (rewrite, then re-validate)
      3. flagged, repair budget spent    -> fallback (safe generic question, then END)

    The loop is bounded by ``max_repairs`` (default 2): repair_question increments
    ``repair_attempts`` every pass, so after at most two rewrites this returns "fallback"
    instead of "repair_question". A model that keeps producing leading questions can
    therefore never loop forever and never ships a bad line — worst case it emits the
    neutral GENERIC_FALLBACK. Bounding it also caps the LLM cost of a turn at 2 extra
    calls.
    """
    if state["validation"]["valid"]:
        return "emit"
    if state["repair_attempts"] >= state["max_repairs"]:
        return "fallback"
    return "repair_question"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def build_interview_graph():
    """Build and compile the per-turn interview graph.

    Read the edges below top-to-bottom and you have the whole turn: assess the answer,
    fold it into coverage, decide the move, then either wrap up or generate→validate
    (→repair→fallback) a question before emitting it.
    """
    g = StateGraph(InterviewGraphState)
    g.add_node("assess_answer", assess_answer_node)
    g.add_node("update_coverage", update_coverage_node)
    g.add_node("decide_action", decide_action_node)
    g.add_node("generate_question", generate_question_node)
    g.add_node("validate_question", validate_question_node)
    g.add_node("repair_question", repair_question_node)
    g.add_node("fallback", fallback_node)
    g.add_node("synthesize", synthesize_node)

    # Straight-line spine: read the answer → update coverage → decide the next move.
    g.set_entry_point("assess_answer")
    g.add_edge("assess_answer", "update_coverage")
    g.add_edge("update_coverage", "decide_action")

    # Fork 1 — wrap vs. continue. CLOSE means every objective is covered or the budget
    # is spent, so we skip question generation entirely and go straight to the closing
    # line; any other action needs a question.
    g.add_conditional_edges(
        "decide_action",
        _route_after_decide,
        {"generate_question": "generate_question", "synthesize": "synthesize"},
    )

    # Every generated (or repaired) question passes through the same quality gate.
    g.add_edge("generate_question", "validate_question")

    # Fork 2 — the question quality gate (the loop). valid → emit (END); flagged but
    # repairs remain → rewrite and re-validate; flagged and repairs exhausted → swap in a
    # safe generic question. This is the only cycle in the graph, and it is bounded
    # (see _route_after_validate) so it can never spin forever.
    g.add_conditional_edges(
        "validate_question",
        _route_after_validate,
        {"emit": END, "repair_question": "repair_question", "fallback": "fallback"},
    )
    g.add_edge("repair_question", "validate_question")  # re-validate every rewrite

    # Terminal edges — both fallback (bad question replaced) and synthesize (wrap-up) end.
    g.add_edge("fallback", END)
    g.add_edge("synthesize", END)
    return g.compile()


_COMPILED_GRAPH = None


def get_interview_graph():
    """Compiled graph, built once and reused (construction is non-trivial)."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_interview_graph()
    return _COMPILED_GRAPH


# =============================================================================
# TURN DRIVER + SESSION
# =============================================================================
def run_graph_turn(
    session: InterviewSession,
    answer: str,
    *,
    use_judge: bool = False,
    judge: Optional[Any] = None,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
) -> dict:
    """Run ONE turn through the graph against a live session.

    Mirrors ``InterviewSession.step`` (same transcript/coverage mutations, same return
    shape) and adds a ``trace`` key plus richer ``turn_log`` fields. Designed so the
    eval harness, CLI, runner, and server can swap runtimes with no other changes.
    """
    if session.finished:
        raise RuntimeError("interview already finished")

    turn_id = session.transcript.append("participant", answer).turn_id
    session.participant_turns += 1

    init: InterviewGraphState = {
        "study": session.study,
        "policy": session.policy,
        "transcript": session.transcript,
        "moderator": session.moderator,
        "coverage": session.coverage,
        "judge": judge,
        "use_judge": use_judge,
        "answer": answer,
        "answer_turn_id": turn_id,
        "current_objective_id": session.current_objective_id,
        "repair_attempts": 0,
        "max_repairs": max_repairs,
        "fallback_used": False,
        "turns_used": session.participant_turns,
        "turn_budget": session.turn_budget,
        "should_wrap": False,
        "trace": [],
    }

    final = get_interview_graph().invoke(init)

    verdict = final["verdict"]
    decision = final["decision"]
    line = final["question"]
    validation = final.get("validation", {"valid": True, "reasons": []})

    # Write back the turn's deterministic state, exactly like step() does.
    session.current_objective_id = final.get("current_objective_id", session.current_objective_id)
    session.last_decision = decision
    if final.get("should_wrap"):
        session.finished = True
    session._say(line, decision)

    session.turn_log.append(
        {
            "answer": answer,
            "answer_turn_id": turn_id,
            "verdict_status": verdict.status.value,
            "is_specific": verdict.is_specific,
            "action": decision.action.value,
            "target_objective_id": decision.target_objective_id,
            "rationale": decision.rationale,
            "moderator_line": line,
            # graph-only extras (ignored by classic metrics/checks):
            "validation": validation,
            "repair_attempts": final.get("repair_attempts", 0),
            "fallback_used": final.get("fallback_used", False),
            "trace": final["trace"],
        }
    )

    return {
        "moderator_line": line,
        "verdict": verdict,
        "decision": decision,
        "coverage": session.coverage.snapshot(),
        "finished": session.finished,
        "turns_used": session.participant_turns,
        "turn_budget": session.turn_budget,
        "validation": validation,
        "trace": final["trace"],
    }


class GraphInterviewSession(InterviewSession):
    """An InterviewSession whose ``step`` runs the LangGraph workflow.

    Inherits ``start``, ``save``, ``history``, ``turn_budget``, coverage/transcript, and
    every public attribute from the classic session — only ``step`` is overridden — so
    it is a drop-in wherever an ``InterviewSession`` is used.
    """

    def __init__(
        self,
        study: Study,
        policy: Policy,
        moderator: Optional[Moderator] = None,
        *,
        use_judge: bool = False,
        judge: Optional[Any] = None,
        max_repairs: int = DEFAULT_MAX_REPAIRS,
    ):
        super().__init__(study, policy, moderator)
        self.use_judge = use_judge
        self.judge = judge
        self.max_repairs = max_repairs

    def step(self, answer: str) -> dict:
        return run_graph_turn(
            self,
            answer,
            use_judge=self.use_judge,
            judge=self.judge,
            max_repairs=self.max_repairs,
        )


def make_session(
    study: Study,
    policy: Policy,
    *,
    graph: bool = False,
    lean: bool = False,
    use_judge: bool = False,
    judge: Optional[Any] = None,
) -> InterviewSession:
    """Factory: classic or graph-backed interview session sharing one interface."""
    moderator = Moderator(study, policy, lean=lean)
    if graph:
        return GraphInterviewSession(study, policy, moderator, use_judge=use_judge, judge=judge)
    return InterviewSession(study, policy, moderator)
