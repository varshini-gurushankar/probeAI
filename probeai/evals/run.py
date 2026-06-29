"""Offline eval harness — one command, one clean report.

    python -m probeai.evals.run                     # all 5 scenarios, graph runtime (default)
    python -m probeai.evals.run --moderator classic # same scenarios, classic engine
    python -m probeai.evals.run --judge             # add the isolated LLM judge (extra calls)
    python -m probeai.evals.run --use-cached        # re-score saved runs, no LLM calls
    python -m probeai.evals.run --scenario vague_oneword

Headline (the number to say on camera):
    "Behaves as expected on N/5 scenarios (incl. graceful termination),
     Y% relevant follow-ups, 0 leading questions — coverage X%."

The N/5 is measured against HAND LABELS (scenarios.py), never the LLM. The LLM
judge only scores follow-up relevance + offers a second opinion on leading
questions. See README "Evaluation limitations".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..agent_graph import make_session
from ..llm import LLMError
from ..participant import ScriptedParticipant, SimulatedParticipant
from ..runner import run_interview
from ..study_config import CONFIG_DIR, Policy, Study, load_policy, load_study
from . import metrics
from .judge import Judge
from .scenarios import SCENARIOS, RunData, Scenario, evaluate

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "eval_runs"
# Compact study keeps a full harness run cheap on tight free-tier quota.
DEFAULT_EVAL_STUDY = CONFIG_DIR / "study_eval.yaml"


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _run_scenario_live(
    scenario: Scenario,
    study: Study,
    policy: Policy,
    *,
    scripted: bool,
    lean: bool,
    moderator: str = "classic",
) -> RunData:
    session = make_session(study, policy, graph=moderator == "graph", lean=lean)
    participant = (
        ScriptedParticipant(scenario.script)
        if scripted
        else SimulatedParticipant(study, scenario.persona_id)
    )
    run_interview(session, participant)
    return RunData.capture(scenario.persona_id, session)


def _cache_path(scenario_id: str, moderator: str = "classic") -> Path:
    # Graph runs are cached separately so they never overwrite the trusted classic cache.
    suffix = "" if moderator == "classic" else f".{moderator}"
    return CACHE_DIR / f"{scenario_id}{suffix}.json"


def _load_cached(scenario_id: str, moderator: str = "classic") -> Optional[RunData]:
    p = _cache_path(scenario_id, moderator)
    if not p.exists():
        return None
    return RunData.from_dict(json.loads(p.read_text(encoding="utf-8")))


def _save_cache(scenario_id: str, run: RunData, moderator: str = "classic") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(scenario_id, moderator).write_text(
        json.dumps(run.to_dict(), indent=2), encoding="utf-8"
    )


def _hr(width: int = 78) -> str:
    return "-" * width


def main() -> None:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description="ProbeAI offline eval harness")
    parser.add_argument("--study", default=None, help="study YAML (default: compact eval study)")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--judge", action="store_true",
                        help="enable the isolated LLM judge for follow-up relevance (extra calls)")
    parser.add_argument("--persona", action="store_true",
                        help="use LLM participant personas instead of fixed scripts (more calls)")
    parser.add_argument("--rich", action="store_true",
                        help="full LLM generation for every moderator line (more calls)")
    parser.add_argument("--use-cached", action="store_true", help="re-score saved runs (no LLM)")
    parser.add_argument("--scenario", default=None, help="run a single scenario id")
    parser.add_argument(
        "--moderator",
        choices=["classic", "graph"],
        default="graph",
        help="moderator runtime: 'graph' (default, the LangGraph workflow with the "
        "validation/repair loop) or 'classic' (the imperative engine). Both are scored "
        "against the same hand labels; runs cache separately so neither clobbers the other.",
    )
    args = parser.parse_args()

    # Default to the compact eval study; scripted + lean keep the run cheap.
    study = load_study(args.study or DEFAULT_EVAL_STUDY)
    policy = load_policy(args.policy)
    scripted = not args.persona
    lean = not args.rich
    scenarios = [s for s in SCENARIOS if not args.scenario or s.id == args.scenario]
    judge = Judge() if args.judge else None

    mode = "cached" if args.use_cached else (
        f"live · {args.moderator} · {'scripted' if scripted else 'persona'} · "
        f"{'lean' if lean else 'rich'} · {'judge' if judge else 'no-judge'}"
    )
    print(f"\nProbeAI offline eval — study: {study.title}")
    print(f"Mode: {mode}\n")

    # --- run/load each scenario, evaluate hand-labeled pass criteria ----------
    results: List[dict] = []
    runs: dict[str, RunData] = {}
    errors: List[str] = []
    for s in scenarios:
        try:
            run = (
                _load_cached(s.id, args.moderator)
                if args.use_cached
                else _run_scenario_live(
                    s, study, policy, scripted=scripted, lean=lean, moderator=args.moderator
                )
            )
        except LLMError as exc:
            # Most likely free-tier quota — skip this scenario, keep partial results,
            # and surface a clear message rather than crashing the whole report.
            errors.append(s.id)
            print(f"  ! '{s.id}' could not run (LLM error): {str(exc)[:120]}...")
            continue
        if run is None:
            print(f"  ! no cached run for '{s.id}' — run live once first; skipping")
            continue
        if not args.use_cached:
            _save_cache(s.id, run, args.moderator)
        runs[s.id] = run
        results.append(evaluate(s, run, policy))

    if not results:
        print(
            "\nNo scenarios completed. If this was a quota error, the free tier may be "
            "exhausted for today — retry later, use a higher-quota GEMINI_API_KEY, or "
            "re-score saved runs with `--use-cached`.\n"
        )
        return

    # --- scenario match-accuracy table (the headline) ------------------------
    print("Scenario match-accuracy (moderator behavior vs. hand-labeled expectation):")
    print(_hr())
    print(f"  {'scenario':<26}{'expected':<42}{'result'}")
    print(_hr())
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['label']:<26}{r['expected']:<42}{mark}")
        print(f"  {'':<26}{'  └ ' + r['detail']}")
    passed = sum(1 for r in results if r["passed"])
    print(_hr())
    print(f"  Scenario match-accuracy: {passed}/{len(results)} "
          f"({100.0 * passed / len(results):.0f}%)\n" if results else "  (no scenarios)\n")

    # --- cross-scenario quality metrics --------------------------------------
    all_questions: List[str] = []
    all_followups: List[dict] = []
    coverage_pcts: List[float] = []
    for s in scenarios:
        if s.id not in runs:
            continue
        run = runs[s.id]
        all_questions += metrics.moderator_questions(run.transcript())
        all_followups += metrics.extract_followups(run.turn_log, study)
        coverage_pcts.append(run.coverage_pct)

    try:
        lead = metrics.leading_question_report(all_questions, judge=judge)
    except LLMError:
        # Judge ran out of quota — fall back to the deterministic heuristics only.
        lead = metrics.leading_question_report(all_questions, judge=None)
        print("  (judge unavailable for leading-Q — using heuristics only)")
    stacked = metrics.one_question_compliance(all_questions)

    print("Interview-quality metrics (aggregated across scenarios):")
    print(_hr())
    avg_cov = sum(coverage_pcts) / len(coverage_pcts) if coverage_pcts else 0.0
    print(f"  Objective coverage (avg):     {avg_cov:.0f}%")
    print(f"  Moderator questions checked:  {lead['total_questions']}")
    print(f"  Leading questions:            {lead['count']}  (target: 0)")
    for v in lead["violations"]:
        print(f"      ✗ [{v['source']}] {v['question']!r} — {'; '.join(v['reasons'])}")
    print(f"  Stacked (double-barreled):    {stacked['stacked_count']}  (target: 0)")
    for q in stacked["stacked"]:
        print(f"      ✗ {q!r}")

    rel_rate: Optional[float] = None
    if judge is not None and all_followups:
        try:
            rel = metrics.followup_relevance(all_followups, judge)
            rel_rate = rel["rate_pct"]
            print(f"  Follow-ups judged:            {rel['total']}")
            print(f"  Relevant follow-up rate:      {rel_rate:.0f}%")
            for d in rel["details"]:
                mark = "✓" if d["relevant"] else "✗"
                print(f"      {mark} {d['followup']!r} — {d['reason']}")
        except LLMError:
            print(f"  Relevant follow-up rate:      (judge ran out of quota; "
                  f"{len(all_followups)} follow-ups captured)")
    elif judge is None:
        print(f"  Relevant follow-up rate:      (skipped — judge disabled; "
              f"{len(all_followups)} follow-ups captured)")
    print(_hr())

    # --- headline ------------------------------------------------------------
    rel_str = f"{rel_rate:.0f}% relevant follow-ups" if rel_rate is not None else \
        "relevant follow-ups n/a (judge off)"
    print("\nHEADLINE:")
    print(f'  "Behaves as expected on {passed}/{len(results)} scenarios '
          f"(incl. graceful termination), {rel_str}, "
          f'{lead["count"]} leading questions — coverage {avg_cov:.0f}%."\n')


if __name__ == "__main__":
    main()
