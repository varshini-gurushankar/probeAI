"""The offline harness's hand-labeled adversarial scenarios.

Each scenario pairs an adversarial participant persona with a HAND-LABELED
expectation of how a good moderator should behave. The ``check`` function compares
the moderator's ACTUAL recorded behavior to that label — deterministically, with
no LLM in the loop. The fraction of scenarios that match is the headline
"scenario match-accuracy".

Pass criteria (the chosen "behavioral, per-scenario" definition):
  vague_oneword     -> moderator probed at least once
  volunteers_ahead  -> an objective got covered via volunteered info (asked 0 times)
  offtopic_rambler  -> moderator steered back at least once
  ideal_participant -> <= 1 total probe AND >= 80% objective coverage
  uncooperative     -> terminated within budget, no objective over its probe cap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from ..study_config import Policy, Priority, Study


@dataclass
class RunData:
    """A serialized interview run — everything the checks + metrics need."""

    persona_id: str
    turn_log: List[dict]
    coverage: List[dict]  # coverage snapshot (includes times_asked, probes_used)
    finished: bool
    participant_turns: int
    turn_budget: int
    transcript_turns: List[dict] = field(default_factory=list)

    @classmethod
    def capture(cls, persona_id: str, session) -> "RunData":
        from dataclasses import asdict

        return cls(
            persona_id=persona_id,
            turn_log=session.turn_log,
            coverage=session.coverage.snapshot(),
            finished=session.finished,
            participant_turns=session.participant_turns,
            turn_budget=session.turn_budget,
            transcript_turns=[asdict(t) for t in session.transcript.turns],
        )

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "RunData":
        return cls(**d)

    def transcript(self):
        """Rebuild a Transcript for the metric extractors."""
        from ..transcript import Transcript, Turn

        t = Transcript("eval", session_id=self.persona_id)
        t.turns = [Turn(**turn) for turn in self.transcript_turns]
        return t

    # convenience views for the checks
    @property
    def coverage_pct(self) -> float:
        total = len(self.coverage)
        covered = sum(1 for c in self.coverage if c["status"] == "covered")
        return 100.0 * covered / total if total else 0.0

    def action_count(self, action: str) -> int:
        return sum(1 for e in self.turn_log if e.get("action") == action)


# A check returns (passed, human-readable detail).
CheckFn = Callable[[RunData, Policy], "tuple[bool, str]"]


def _check_vague(run: RunData, policy: Policy):
    probes = run.action_count("probe")
    return probes >= 1, f"{probes} probe(s) issued on vague answers"


def _check_volunteers(run: RunData, policy: Policy):
    volunteered = [
        c for c in run.coverage if c["status"] == "covered" and c["times_asked"] == 0
    ]
    if volunteered:
        ids = ", ".join(c["objective_id"] for c in volunteered)
        return True, f"covered via volunteered info without re-asking: {ids}"
    return False, "no objective was covered from volunteered (un-asked) info"


def _check_offtopic(run: RunData, policy: Policy):
    steers = run.action_count("steer_back")
    return steers >= 1, f"{steers} steer-back(s) to refocus the participant"


def _check_ideal(run: RunData, policy: Policy):
    probes = run.action_count("probe")
    ok = probes <= 1 and run.coverage_pct >= 80.0
    return ok, f"{probes} probe(s), {run.coverage_pct:.0f}% coverage (want <=1 probe, >=80%)"


def _check_uncooperative(run: RunData, policy: Policy):
    within_budget = run.finished and run.participant_turns <= run.turn_budget
    no_over_probe = all(
        c["probes_used"] <= policy.probe_cap(Priority(c["priority"])) for c in run.coverage
    )
    ok = within_budget and no_over_probe
    detail = (
        f"finished={run.finished}, turns={run.participant_turns}/{run.turn_budget}, "
        f"no objective over probe cap={no_over_probe}"
    )
    return ok, detail


@dataclass
class Scenario:
    id: str
    persona_id: str
    label: str
    expected_behavior: str
    check: CheckFn
    # Deterministic answer script for the (default) scripted harness mode. The
    # persona_id is used instead when running with --persona (LLM participant).
    script: List[str] = field(default_factory=list)


# Fixed answer scripts — these make harness runs reproducible and free of
# participant-side LLM calls. They embody the same behaviors as the personas in
# config/personas.yaml.
_SCRIPTS = {
    "vague_oneword": [
        "It was annoying.",
        "I dunno.",
        "It was just slow.",
        "Fine, I guess.",
        "Don't really remember.",
    ],
    "volunteers_ahead": [
        "I bailed at checkout because I didn't trust putting my card into a site I'd "
        "never heard of — the payment page looked sketchy and asked for way too much info.",
        "Yeah, it was right at the payment step when it asked for my card.",
        "I just closed the app and gave up on it.",
    ],
    "offtopic_rambler": [
        "Honestly the weather's been amazing lately, I've been hiking every weekend.",
        "Oh, and there's this recipe app I'm totally addicted to right now.",
        "Sorry — checkout? It kept making me log in before I could pay.",
        "Right, the payment screen — it timed out on me twice.",
    ],
    "ideal_participant": [
        "I was buying a sweater on a fashion app. I added it to cart, hit checkout, and it "
        "forced me to a login screen. I tried 'continue as guest' but it still demanded my "
        "email, a password, and a bunch of personal details. At the payment step the form "
        "rejected my card twice with no clear error, and by then I didn't trust entering my "
        "details, so I abandoned it right at the payment screen.",
        "That's really everything — the forced signup and the failing payment were the issues.",
    ],
    "uncooperative": [
        "I don't know.",
        "I don't remember.",
        "No.",
        "Not really.",
        "I don't know.",
    ],
}


SCENARIOS: List[Scenario] = [
    Scenario(
        id="vague_oneword",
        persona_id="vague_oneword",
        label="Vague one-word answerer",
        expected_behavior="Probe for specifics (>=1 follow-up)",
        check=_check_vague,
        script=_SCRIPTS["vague_oneword"],
    ),
    Scenario(
        id="volunteers_ahead",
        persona_id="volunteers_ahead",
        label="Volunteers later-objective info",
        expected_behavior="Mark volunteered objective covered; don't re-ask it",
        check=_check_volunteers,
        script=_SCRIPTS["volunteers_ahead"],
    ),
    Scenario(
        id="offtopic_rambler",
        persona_id="offtopic_rambler",
        label="Off-topic rambler",
        expected_behavior="Steer back to the objective (>=1 redirect)",
        check=_check_offtopic,
        script=_SCRIPTS["offtopic_rambler"],
    ),
    Scenario(
        id="ideal_participant",
        persona_id="ideal_participant",
        label="Ideal participant (control)",
        expected_behavior="Few/no probes (<=1) and >=80% coverage",
        check=_check_ideal,
        script=_SCRIPTS["ideal_participant"],
    ),
    Scenario(
        id="uncooperative",
        persona_id="uncooperative",
        label="Uncooperative 'I don't know'",
        expected_behavior="Terminate gracefully within budget; no probe-loop",
        check=_check_uncooperative,
        script=_SCRIPTS["uncooperative"],
    ),
]


def evaluate(scenario: Scenario, run: RunData, policy: Policy) -> dict:
    passed, detail = scenario.check(run, policy)
    return {
        "id": scenario.id,
        "label": scenario.label,
        "expected": scenario.expected_behavior,
        "passed": passed,
        "detail": detail,
    }
