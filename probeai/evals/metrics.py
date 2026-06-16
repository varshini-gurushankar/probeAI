"""Per-interview quality metrics.

These functions take plain primitives (a coverage snapshot, the moderator's
questions, extracted follow-ups) rather than a live session, so they are pure and
unit-testable. The LLM judge is passed in only where genuinely needed
(follow-up relevance, optional leading-question second opinion); the COUNTING and
the headline comparisons stay deterministic.
"""

from __future__ import annotations

from typing import List, Optional

from ..study_config import Study
from ..transcript import Transcript
from . import heuristics
from .judge import Judge


# --- extraction helpers ------------------------------------------------------


def moderator_questions(transcript: Transcript) -> List[str]:
    """Moderator utterances that actually ask something (contain a '?')."""
    return [t.text for t in transcript.turns if t.speaker == "moderator" and "?" in t.text]


def extract_followups(turn_log: List[dict], study: Study) -> List[dict]:
    """Pair each probe with the answer it responded to + its objective label."""
    out: List[dict] = []
    for entry in turn_log:
        if entry.get("action") != "probe":
            continue
        oid = entry.get("target_objective_id")
        try:
            label = study.objective(oid).label
        except KeyError:
            label = oid or "(unknown)"
        out.append(
            {
                "objective_label": label,
                "answer": entry.get("answer", ""),
                "followup": entry.get("moderator_line", ""),
            }
        )
    return out


# --- metrics -----------------------------------------------------------------


def objective_coverage(coverage_snapshot: List[dict]) -> dict:
    """% of objectives that reached 'covered', with evidence turn ids."""
    total = len(coverage_snapshot)
    covered = [c for c in coverage_snapshot if c["status"] == "covered"]
    return {
        "covered": len(covered),
        "total": total,
        "pct": (100.0 * len(covered) / total) if total else 0.0,
        "by_objective": [
            {
                "objective_id": c["objective_id"],
                "label": c["label"],
                "status": c["status"],
                "evidence_turn_ids": c["evidence_turn_ids"],
            }
            for c in coverage_snapshot
        ],
    }


def leading_question_report(questions: List[str], judge: Optional[Judge] = None) -> dict:
    """Flag leading questions via heuristics, optionally confirmed by the judge.

    A question is a violation if the heuristics flag it OR (when a judge is
    provided) the isolated judge flags it — defense in depth. Target: zero.
    """
    violations: List[dict] = []
    for q in questions:
        reasons = heuristics.find_leading_violations(q)
        source = "heuristic" if reasons else None
        if judge is not None and not reasons:
            verdict = judge.leading_question(q)
            if verdict.leading:
                reasons = [verdict.reason or "judge: leading"]
                source = "judge"
        if reasons:
            violations.append({"question": q, "reasons": reasons, "source": source})
    return {
        "violations": violations,
        "count": len(violations),
        "total_questions": len(questions),
    }


def one_question_compliance(questions: List[str]) -> dict:
    """Flag stacked / double-barreled questions. Target: zero stacked."""
    stacked = [q for q in questions if heuristics.is_stacked(q)]
    total = len(questions)
    return {
        "stacked": stacked,
        "stacked_count": len(stacked),
        "total_questions": total,
        "compliance_pct": (100.0 * (total - len(stacked)) / total) if total else 100.0,
    }


def followup_relevance(followups: List[dict], judge: Judge) -> dict:
    """Relevant-follow-up rate via the isolated judge (the secondary headline)."""
    details: List[dict] = []
    relevant = 0
    for f in followups:
        verdict = judge.followup_relevance(f["objective_label"], f["answer"], f["followup"])
        relevant += int(verdict.relevant)
        details.append(
            {
                "objective_label": f["objective_label"],
                "followup": f["followup"],
                "relevant": verdict.relevant,
                "reason": verdict.reason,
            }
        )
    total = len(followups)
    return {
        "relevant": relevant,
        "total": total,
        "rate_pct": (100.0 * relevant / total) if total else None,
        "details": details,
    }
