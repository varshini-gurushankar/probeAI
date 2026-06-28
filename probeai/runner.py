"""Drive a full interview start->finish with a simulated participant.

Used for solo demos and as the substrate for the offline eval harness:

    python -m probeai.runner --persona realistic        # full sim interview, saved
    python -m probeai.runner --persona vague_oneword -v  # watch it probe

Returns the completed ``InterviewSession`` (coverage, transcript, per-turn log).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .agent_graph import make_session
from .moderator import InterviewSession
from .participant import SimulatedParticipant
from .study_config import Policy, Study, load_policy, load_study


def run_interview(
    session: InterviewSession,
    participant: SimulatedParticipant,
    *,
    verbose: bool = False,
    safety_margin: int = 4,
) -> InterviewSession:
    """Run moderator <-> simulated participant until the moderator wraps up.

    ``safety_margin`` is a hard stop above the turn budget so a misbehaving model
    can never loop forever.
    """
    max_turns = session.turn_budget + safety_margin

    line = session.start()
    if verbose:
        print(f"Moderator: {line}\n")

    while not session.finished and session.participant_turns < max_turns:
        answer = participant.respond(line)
        if verbose:
            print(f"Participant: {answer}")
        result = session.step(answer)
        line = result["moderator_line"]
        if verbose:
            d = result["decision"]
            print(f"  → {d.action.value}: {d.rationale}")
            for entry in result.get("trace", []):  # graph mode only
                reason = f" — {entry['reason']}" if entry["reason"] else ""
                print(f"      · {entry['node']}: {entry['output']}{reason}")
            print(f"Moderator: {line}\n")

    return session


def run_simulated(
    persona_id: str,
    study: Optional[Study] = None,
    policy: Optional[Policy] = None,
    *,
    verbose: bool = False,
    graph: bool = False,
) -> InterviewSession:
    study = study or load_study()
    policy = policy or load_policy()
    session = make_session(study, policy, graph=graph)
    participant = SimulatedParticipant(study, persona_id)
    return run_interview(session, participant, verbose=verbose)


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and choke on the arrows/glyphs we print."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> None:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Run a full simulated interview")
    parser.add_argument("--persona", default="realistic")
    parser.add_argument("--study", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--no-save", action="store_true", help="don't write the transcript")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="use the LangGraph moderator runtime (validation/repair loop + trace)",
    )
    args = parser.parse_args()

    study = load_study(args.study)
    policy = load_policy(args.policy)
    session = run_simulated(
        args.persona, study, policy, verbose=args.verbose, graph=args.graph
    )

    print("\n=== interview complete ===")
    print(
        f"Coverage: {session.coverage.covered_count}/{session.coverage.total} "
        f"covered ({session.coverage.coverage_pct():.0f}%), "
        f"{session.participant_turns} participant turns"
    )
    if not args.no_save:
        path = session.save()
        print(f"Transcript saved -> {path}")


if __name__ == "__main__":
    main()
