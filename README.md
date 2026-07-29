# ProbeAI

**An AI-moderated research interviewer that grades its own work.**

ProbeAI is a small working demo of an AI moderator that runs a user-research interview from a discussion guide: it asks follow-ups when an answer is thin, tracks how well it covers each objective, writes up the findings, and then scores how good the interview actually was.

I built it for Great Question's AI engineering internship as a miniature of the **AI Moderated Interviews** feature. It targets two of the challenges in the role — a realtime agentic moderator (TTS/STT) and evals/quality measures — because the underlying problem is a genuinely interesting one:

> Fixed surveys scale to many people but can't follow up to dig deeper. One-on-one interviews get you deep answers, but a researcher runs each by hand, so you can only do a handful.

ProbeAI's bet is that an AI moderator can give you **interview depth at survey scale** — and, just as importantly, can tell you whether any given interview was actually any good. The hard part isn't making it talk; it's making it interview *well* and knowing whether it did. That second half is where most of the effort went.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && .venv/Scripts/activate     # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# 2. Free Gemini key (no card): https://aistudio.google.com/apikey
cp .env.example .env        # then set GEMINI_API_KEY=
```

```bash
# Web app (voice + UI)
uvicorn probeai.server:app --reload     # http://localhost:8000

# Terminal demo — LangGraph runtime by default
python -m probeai.cli                                          # type answers yourself
python -m probeai.runner --persona vague_oneword -v            # watch it probe, with the per-node trace
python -m probeai.runner --persona vague_oneword --classic -v  # same, via the classic engine (no trace)

# Evals — LangGraph runtime by default
python -m probeai.evals.run                       # the trusted 5-scenario headline
python -m probeai.evals.run --moderator classic   # same scenarios via the classic engine
```

ProbeAI runs only on Gemini's free tier, so it costs nothing to run.

---

## What one turn looks like

```
Participant: "It was annoying."
   ├─ assess  (LLM)   → verdict: partial, is_specific: false, gap: "what specifically was annoying"
   ├─ decide  (pure)  → PROBE  (high-priority objective, follow-up 1/2)
   └─ generate (LLM)  → "You said it was annoying — what specifically made it annoying?"

Coverage: [The moment they abandon checkout]  uncovered → partial   (evidence: turn 3)
```

If the answer is specific and substantive, the objective is marked **covered** and the moderator moves on. If the participant volunteers something belonging to a later objective, that one is credited too and never re-asked. If they wander off, it steers back. Once everything's covered or the turn budget runs out, it wraps up and writes the findings.

---

## Architecture

The spine of the system is one design choice: **the LLM handles language; deterministic code owns the policy.** Each turn is `assess → decide → generate`, where the decision in the middle — whether to probe, what counts as covered, when to stop — is pulled out of the LLM into a pure, unit-testable function. Policy that lives in a prompt drifts turn to turn and can't be tested; policy in code is deterministic and stable.

Modules are kept separate so each is testable on its own:

| Module | What it does |
|---|---|
| `study_config.py` | Loads and validates the discussion guide (`config/study_checkout.yaml`) and the policy (`config/policy.yaml`). |
| `moderator.py` | The classic engine: `assess` (LLM) → `decide_next_action` (pure) → `generate` (LLM). |
| `agent_graph.py` | The **LangGraph** runtime — the same turn as named, traceable nodes, plus a validate/repair loop. The default everywhere. |
| `coverage.py` | Per-objective state machine (uncovered / partial / covered) with the evidence turns. Deterministic once it has a verdict. |
| `transcript.py` | Speaker-attributed log, appended as it goes, saved to JSONL. |
| `synthesis.py` | Findings summary and objective-linked highlights. Every quote is checked verbatim against a real participant turn, so it can't invent one. |
| `participant.py` | Simulated-participant personas, for solo demos and the eval harness. |
| `evals/` | Heuristics, judge, metrics, scenarios, and the runner. |
| `gq_mock.py` | Stand-in for Great Question's API — the seam where the real integration would slot in. |
| `server.py` + `web/` | FastAPI backend and a plain-JS front end: mic, voice, live transcript, coverage panel, eval report. |

Prompts all live in `prompts.py`, and the moderator logs why it made each call (probe vs. move on) every turn, which makes debugging much easier.

> For a full walk-through of the turn — every node, what flows through the graph state, and what LangGraph does and doesn't buy this design — see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### Moderation rules

These are baked into both the moderator prompt and the decision code:

- Treat the discussion guide as a framework, not a script — cover the objectives, but follow the participant.
- When an answer leaves a gap, ask **one** targeted follow-up into that specific gap.
- Don't over-probe. Caps are priority-weighted, and once something's covered, move on.
- Never ask leading questions — no presupposition, loaded phrasing, or two questions at once. The eval layer measures this separately, and the target is zero.
- One question at a time; open and close the conversation cleanly.

Everything tunable lives in `config/policy.yaml`.

---

## The LangGraph runtime

The moderator turn is modeled as a LangGraph state machine, so the interview loop is explicit and inspectable instead of buried inside one method. The idea is **agentic behavior with deterministic policy control**: the LLM handles language judgment and question generation, while deterministic code keeps owning coverage, follow-up caps, the turn budget, and when to stop.

```
START
 → assess_answer      (LLM)   read the answer → verdict
 → update_coverage    (pure)  fold the verdict into the coverage state machine
 → decide_action      (pure)  decide_next_action → PROBE / MOVE_ON / REDIRECT / WRAP
     ├─ WRAP → synthesize ───────────────────────────────────────────→ END
     └─ generate_question (LLM/template)   probe / move on / redirect
         → validate_question  (deterministic: leading? double-barreled? off-topic?)
             ├─ valid ─────────────────────────────────────────────────→ emit → END
             ├─ invalid, repairs left → repair_question (LLM) → validate_question …
             └─ invalid, exhausted    → safe generic fallback ──────────→ END
```

Every node reuses what already exists — `assess`, the pure `decide_next_action`, the coverage state machine, the leading/double-barreled heuristics — so no logic is duplicated or pushed into the LLM. The one thing the graph **adds** over the classic path is the **validate → repair → fallback loop**: before any question ships, it's checked for leading or loaded phrasing, double-barreled questions, and basic relevance. If it fails, `repair_question` rewrites it to be open, neutral, and single — up to two tries, then a safe generic fallback rather than shipping a bad question. Each node logs a small trace (node, input/output, reason) that surfaces in `runner -v` and the web backend's `/api/turn` response.

The graph is the default runtime everywhere — CLI, runner, web, and the eval harness. The five hand-labeled scenarios behave **identically** under both runtimes (verified before/after the switch), so the headline number is unaffected; pass `--moderator classic` to run the same scenarios through the classic engine for comparison.

---

## The eval layer

This is the part I care most about, and it's split deliberately so **a model is never grading its own output.**

### 1. Behavioral accuracy — the headline, no LLM involved

`evals/scenarios.py` has five adversarial participants, each with a behavior I hand-labeled as correct. The harness compares what the moderator actually did against my label with a plain deterministic check:

| Scenario | What it should do |
|---|---|
| Vague one-word answerer | Probe for specifics (at least 1 follow-up) |
| Volunteers later-objective info | Credit that objective, don't re-ask it |
| Off-topic rambler | Steer back (at least 1 redirect) |
| Ideal participant (control) | Few or no probes, and 80%+ coverage |
| Uncooperative "I don't know" | Wrap up gracefully, don't loop |

The **ideal participant is a control on purpose** — without it, the harness would just reward a moderator that probes everyone, including people who already gave good answers. The **uncooperative** one is there because the scariest failure for an agent is not knowing when to stop.

The headline metric is **scenario match-accuracy** — does the moderator behave the way I labeled it should — reported alongside the follow-up relevance rate and the leading-question count. I chose this so the headline measures interviewing *well*, not just finishing, and stays grounded in my own labels rather than a model grading itself.

### 2. LLM-as-judge — narrow and isolated

`evals/judge.py` only scores follow-up relevance ("did this dig into a real gap?") and gives a second opinion on leading questions. It runs in its own context, on a **different model** from the moderator (`gemini-2.5-flash` vs. the moderator's `flash-lite`), at **temperature 0**, and it only ever sees the snippet it's judging. **It never touches the headline number.**

### Running the evals

```bash
python -m probeai.evals.run                  # all 5 scenarios (cheap defaults)
python -m probeai.evals.run --judge          # also score follow-up relevance (more LLM calls)
python -m probeai.evals.run --persona        # LLM personas instead of scripted answers
python -m probeai.evals.run --rich           # full LLM generation for every line
python -m probeai.evals.run --use-cached     # re-score saved runs, no LLM calls
python -m probeai.evals.run --scenario vague_oneword
```

Gemini's free tier is stingy and the project ran under a low shared daily request budget, so the harness runs lean by default: a compact 3-objective study, scripted answers (no participant LLM calls), and only the probes actually hit the model — the routine open/next/close lines are templated from the seed questions.

---

## Key decisions

A few I went back and forth on:

- **What counts as "covered."** Three states. "Covered" means a specific, substantive answer to what the objective was actually after, with the turn that earned it stored as evidence. "Partial" means on-topic but vague. A throwaway mention shouldn't count as having learned something, and storing the evidence means any "covered" call can be checked.
- **When an answer is vague.** The probe decision relies primarily on the model's assessment of whether a response is *specific*; length is only a weak secondary signal and never triggers a probe on its own. "I quit at the payment screen" is short but fully substantive.
- **Catching leading questions.** Detection is hybrid: deterministic regex and heuristics flag the explicit cases (presupposition, loaded phrasing, double-barreled), while an isolated LLM judge catches the subtler presupposition rules can't reliably detect.
- **Two runtimes.** The classic engine is the simple, trusted reference; the graph is the default and adds traceability and the repair loop. Keeping both means the headline can be verified for parity across runtimes rather than assumed.

---

## Where it would plug into Great Question

I built this for Great Question specifically — rather than something generic — to take two of the role's challenges (the agentic moderator and the evals) and show I could build toward them and reason about how they'd fit the product. `gq_mock.py` is the stand-in for the real API: the moderator loop is the AI Moderated Interviews feature, and the eval harness is the more transferable asset — a way to regression-test moderator quality as prompts and models change.

---
