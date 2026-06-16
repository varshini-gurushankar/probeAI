# ProbeAI — an AI-moderated research interviewer that grades its own work

ProbeAI is a small working demo I built for Great Question's AI engineering
internship. It's a miniature of the AI Moderated Interviews feature that is included in the role's responsibilites: a moderator that runs a user-research interview from a discussion guide, asks follow-ups when an answer is thin, tracks how well it covered each objective, writes up the findings, and then scores how good the interview actually was.

I picked this because it lines up with two of the challenges in the role description which is a
realtime agentic moderator (TTS/STT), evals/quality measures, and because Great
Question's own pitch for the feature is a problem I find genuinely interesting: surveys
scale but stay shallow, interviews go deep but don't scale, and the bet is that an AI
moderator can get you some of both. The hard part isn't making it talk. It's making it
interview well and knowing whether it did. That second half is where I spent most of
my effort.

## What one turn looks like

```
Participant: "It was annoying."
   ├─ assess  (LLM)   → verdict: partial, is_specific: false, gap: "what specifically was annoying"
   ├─ decide  (pure)  → PROBE  (high-priority objective, follow-up 1/2)
   └─ generate (LLM)  → "You said it was annoying — what specifically made it annoying?"
Coverage: [The moment they abandon checkout]  uncovered → partial   (evidence: turn 3)
```

If the answer is specific and substantive, the objective gets marked covered and the
moderator moves on. If the participant volunteers something that belongs to a later
objective, that one gets credited too and never re-asked. If they wander off, it steers
back. Once everything's covered or the turn budget runs out, it wraps up.

## How it's put together

I kept the modules separate so each piece is testable on its own.

| Module | What it does |
| --- | --- |
| `study_config.py` | Loads and validates the discussion guide (`config/study_checkout.yaml`) and the policy (`config/policy.yaml`). |
| `moderator.py` | The engine: `assess` (LLM) → `decide_next_action` (pure, deterministic) → `generate` (LLM). I pulled the decision logic out of the LLM call so it can be unit-tested. |
| `coverage.py` | Per-objective state machine (uncovered / partial / covered) plus the evidence turns. Deterministic once it has a verdict. |
| `transcript.py` | Speaker-attributed log, appended as it goes, saved to JSONL. |
| `synthesis.py` | Findings summary and objective-linked highlights. Every quote is checked verbatim against a real participant turn so the synthesis can't invent one. |
| `participant.py` | Simulated-participant personas, for solo demos and the eval harness. |
| `evals/` | heuristics, judge, metrics, scenarios, and the runner. |
| `gq_mock.py` | The stand-in for Great Question's API (see below). |
| `server.py` + `web/` | FastAPI backend and a plain-JS front end: mic, voice, live transcript, coverage panel, eval report. |

Prompts all live in `prompts.py` so they're easy to find and tweak, and the moderator logs
why it made each call (probe vs. move on) every turn, which made debugging a lot easier.

## The moderation rules

These are baked into both the moderator prompt and the decision code:

- Treat the discussion guide as a framework, not a script, cover the objectives but follow
  the participant.
- When an answer leaves a gap, ask one targeted follow up into that specific gap.
- Don't over-probe. Caps are priority-weighted (below), and once something's covered, move on.
- Never ask leading questions (no presupposition, loaded phrasing, or two questions at once).
  The eval layer measures this separately and the target is zero.
- One question at a time, and open/close the conversation cleanly.

Everything tunable lives in `config/policy.yaml`.

## Decisions I made on purpose

A few of these I went back and forth on, so I'll explain the reasoning:

**What counts as "covered."** Three states, and "covered" means the participant gave a
specific, substantive answer to what the objective was actually after, with the turn that
earned it stored as evidence. "Partial" means they were on-topic but vague. I didn't want a
throwaway mention to count as having learned something, and storing the evidence means any
"covered" call can be checked.

**When an answer is vague.** The decision to probe relies primarily on the model's assessment of whether a response is specific; response length serves only as a minor secondary signal and never triggers a probe on its own. A short answer can still be substantive. For example, "I quit at the payment screen" so length alone is an unreliable indicator.

**Catching leading questions.** Detection is hybrid. Deterministic regex and heuristics flag the explicit cases (presupposition, loaded phrasing, and double-barreled questions), while a separately isolated LLM judge identifies the subtler presupposition that rule-based checks cannot reliably capture.

**The headline metric.** The primary measure is scenario match-accuracy: the extent to which the moderator's actual behavior aligns with the behavior I defined as correct for each test scenario, reported alongside the follow-up relevance rate and the leading-question count. I selected this metric so that the headline reflects the quality of the interview rather than its mere completion, and so that it remains grounded in human-defined labels rather than a model evaluating its own output.

**The headline number.** Scenario match-accuracy. Does it behave the way I labeled it should
across the test scenarios, with the follow-up relevance rate and the leading-question count
alongside. I wanted the headline to measure interviewing well, not just finishing, and I
wanted it grounded in my own labels rather than a model grading itself.

## The eval layer

This is the part I care most about, and I split it deliberately so a model is never grading
its own output.

**1. Behavioral accuracy (the headline, no LLM involved).** `evals/scenarios.py` has five
adversarial participants, each with a behavior I hand-labeled as correct. The harness compares
what the moderator actually did against my label with a plain deterministic check:

| Scenario | What it should do |
| --- | --- |
| Vague one-word answerer | Probe for specifics (at least 1 follow-up) |
| Volunteers later-objective info | Credit that objective, don't re-ask it |
| Off-topic rambler | Steer back (at least 1 redirect) |
| Ideal participant (control) | Few or no probes, and 80%+ coverage |
| Uncooperative "I don't know" | Wrap up gracefully, don't loop |

The ideal participant is a control on purpose — without it, the harness would just reward a
moderator that probes everyone, including people who already gave good answers. The
uncooperative one is there because the scariest failure for an agent is not knowing when to
stop.

**2. LLM-as-judge (narrow and isolated).** `evals/judge.py` only scores follow-up relevance
("did this dig into a real gap?") and gives a second opinion on leading questions. It runs in
its own context on a different model from the moderator (`gemini-2.5-flash` vs the moderator's
`flash-lite`), at temperature 0, and it only ever sees the snippet it's judging. It never
touches the headline number.

### Running it

```bash
python -m probeai.evals.run            # all 5 scenarios (cheap defaults)
python -m probeai.evals.run --judge    # also score follow-up relevance (more LLM calls)
python -m probeai.evals.run --persona  # LLM personas instead of scripted answers
python -m probeai.evals.run --rich     # full LLM generation for every line
python -m probeai.evals.run --use-cached    # re-score saved runs, no LLM calls
python -m probeai.evals.run --scenario vague_oneword
```

Gemini's free tier is stingy, and the project I was on was capped at a very low shared daily request budget. So the harness runs lean by default, a compact 3-objective study, scripted answers (no participant LLM calls), and only the probes actually hit the model; the routine open/next/close lines are templated from the seed questions. 

## Where it would plug into Great Question

I made this for Great Question because rather than building something generic, I wanted to take two of the challenges from the role: the agentic moderator and evals and show I could build toward them and reason about how they'd fit your product. 

## Setup

```bash
# 1. Install
python -m venv .venv && .venv/Scripts/activate     # Windows; use source on macOS/Linux
pip install -r requirements.txt

# 2. Free Gemini key (no card): https://aistudio.google.com/apikey
cp .env.example .env        # then set GEMINI_API_KEY=

# 3a. Web app (voice + UI)
uvicorn probeai.server:app --reload     # http://localhost:8000

# 3b. Terminal demo
python -m probeai.cli                                  # type answers yourself
python -m probeai.runner --persona vague_oneword -v    # watch it probe a simulated participant

# 4. Evals
python -m probeai.evals.run
```

I made this with Claude Code as I was allowed to use AI to create a demo. If you have `ANTHROPIC_API_KEY` set, Claude Code will bill the paid API instead, so I kept it unset.
ProbeAI itself never calls Anthropic, at runtime it only uses Gemini's free tier, so it costs
nothing to run.
