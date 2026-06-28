"""Interactive text CLI — the slice-1 demo and a quick way to feel the moderator.

    python -m probeai.cli                 # type your own answers
    python -m probeai.cli --persona ideal_participant   # let an LLM play the participant

Prints the moderator's line, its decision rationale (why it probed vs. moved on),
and the live objective-coverage panel each turn.
"""

from __future__ import annotations

import argparse
import sys

from .agent_graph import make_session
from .study_config import load_policy, load_study

# Small status glyphs for the coverage panel.
_GLYPH = {"uncovered": "·", "partial": "◐", "covered": "●"}


def _print_coverage(snapshot: list[dict]) -> None:
    print("\n  Objective coverage:")
    for c in snapshot:
        glyph = _GLYPH.get(c["status"], "?")
        probes = f"  ({c['probes_used']} probe)" if c["probes_used"] else ""
        print(
            f"    {glyph} [{c['priority']:<6}] {c['label']}: "
            f"{c['status']}{probes}"
        )
    print()


def main() -> None:
    # Windows consoles default to cp1252 and choke on the coverage glyphs.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="ProbeAI text interview")
    parser.add_argument("--study", default=None, help="path to a study YAML")
    parser.add_argument("--policy", default=None, help="path to a policy YAML")
    parser.add_argument(
        "--persona",
        default=None,
        help="simulate the participant with this persona id (else you type)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="use the LangGraph moderator runtime (validation/repair loop)",
    )
    args = parser.parse_args()

    study = load_study(args.study)
    policy = load_policy(args.policy)
    session = make_session(study, policy, graph=args.graph)
    if args.graph:
        print("[graph mode] LangGraph moderator runtime\n")

    # Optional simulated participant (slice 2 module; imported lazily so slice-1
    # typed mode works without it).
    participant = None
    if args.persona:
        from .participant import SimulatedParticipant

        participant = SimulatedParticipant(study, args.persona)

    print(f"\n=== {study.title} ===")
    print(f"Research goal: {study.research_goal}\n")
    print(f"Turn budget: {session.turn_budget} participant turns\n")

    opening = session.start()
    print(f"Moderator: {opening}\n")

    while not session.finished:
        if participant is not None:
            answer = participant.respond(opening if session.participant_turns == 0 else last_line)
            print(f"Participant: {answer}")
        else:
            try:
                answer = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[ended early]")
                break
        if not answer:
            continue

        result = session.step(answer)
        last_line = result["moderator_line"]
        print(f"\n  → decision: {result['decision'].action.value} — {result['decision'].rationale}")
        _print_coverage(result["coverage"])
        print(f"Moderator: {last_line}\n")

    print("=== interview complete ===")
    print(
        f"Coverage: {session.coverage.covered_count}/{session.coverage.total} "
        f"objectives covered ({session.coverage.coverage_pct():.0f}%)\n"
    )


if __name__ == "__main__":
    main()
