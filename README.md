# ProbeAI — an AI-moderated user-research interviewer with a self-eval layer

> Models Great Question's **AI Moderated Interviews**: *"the reach of a survey with the
> depth of an interview."* A discussion-guide–driven moderator that **adapts and doesn't
> stick to a rigid script** — it probes on response gaps, tracks objective coverage,
> synthesizes findings, and then **grades its own interview quality**.

ProbeAI is a small, real, runnable demo built for Great Question's AI-native SWE internship.
It targets two of the role's stated challenges head-on:

1. **Realtime agentic AI moderator (TTS + STT)** — a turn-based voice interviewer.
2. **Evals & quality measures** — a layer that scores whether the interview was *good*,
   not just whether it finished.

---

## The problem, in Great Question's terms

Surveys scale but stay shallow; interviews go deep but don't scale. GQ's AI Moderated
Interviews close that gap: a researcher writes a **discussion guide** (a research goal +
prioritized **objectives**), and an AI **moderator** runs the interview — using the guide
**as a framework, not a script**, **probing when answers need more detail**, and respecting
topic priorities (*spend more time here; skip this if time runs short*). The thing human
moderators still win is *reading between the lines*; the thing AI moderation wins is
**adaptive probing based on response gaps**. ProbeAI is a faithful miniature of exactly that.

---

## What it does (a single turn)

```
Participant: "It was annoying."
   ├─ assess  (LLM)   → verdict: partial, is_specific: false, gap: "what specifically was annoying"
   ├─ decide  (pure)  → PROBE  (high-priority objective, follow-up 1/2)
   └─ generate (LLM)  → "You said it was annoying — what specifically made it annoying?"
Coverage: [The moment they abandon checkout]  uncovered → partial   (evidence: turn 3)
```

When an answer is **specific and substantive**, the objective is marked **covered** and the
moderator moves on. When the participant **volunteers info for a later objective**, that
objective is covered too (and never re-asked). When they **wander**, the moderator steers
back. When objectives are covered or the **turn budget** is hit, it **wraps up gracefully**.

---

## Architecture

Clean, independently testable modules:

| Module | Role |
| --- | --- |
| `study_config.py` | Loads + validates the discussion guide (`config/study_checkout.yaml`) and the moderation policy (`config/policy.yaml`). |
| `moderator.py` | **The agentic engine.** `assess` (LLM) → `decide_next_action` (**pure, deterministic**) → `generate` (LLM). The decision logic is separated from the LLM so it is unit-testable. |
| `coverage.py` | Per-objective state machine: `uncovered / partial / covered` + evidence turns. Deterministic given a verdict. |
| `transcript.py` | Append-only, speaker-attributed log; persists to JSONL. |
| `synthesis.py` | Findings summary + objective-linked **highlights** — every quote is verified verbatim against a real participant turn (no hallucinated quotes). |
| `participant.py` | Simulated-participant LLM personas (solo demo + eval harness). |
| `evals/` | `heuristics` + `judge` + `metrics` + `scenarios` + `run` (see below). |
| `gq_mock.py` | The Great Question integration seam (see below). |
| `server.py` + `web/` | FastAPI backend + a vanilla SPA: mic STT, TTS, live transcript, coverage panel, eval report. |

Prompts live in `prompts.py` so they're easy to read, diff, and tune. The moderator logs its
**decision rationale every turn** (why it probed vs. moved on).

---

## The moderation policy (the product-sense core)

Encoded as hard constraints in the moderator prompt **and** the decision code:

- **Framework, not a script** — cover objectives, but adapt order/depth to the participant.
- **Probe on response gaps** — one targeted follow-up into the *specific* gap.
- **Don't over-probe** — priority-weighted caps (see below); once covered, move on.
- **Never leading** — no presupposition / loaded / double-barreled questions (separately
  measured by the eval layer; target zero).
- **One question at a time.**
- **Graceful open and close.**

All thresholds are editable in [`config/policy.yaml`](probeai/config/policy.yaml).

---

## Design decisions I own (and can defend)

These were deliberate choices, not defaults:

| Decision | Choice | Why |
| --- | --- | --- |
| **"Covered" definition** | 3 states; *covered* = **substantive + specific** answer to the objective's intent, with the participant turn stored as **evidence**; *partial* = on-topic but vague. | A vague mention shouldn't count as learning something. Evidence makes coverage auditable. |
| **Probe thresholds** | **Priority-weighted**: high = 2 follow-ups, medium = 1, low = 0–1 and skippable when budget is low. Turn budget ≈ 3 × #objectives. | Mirrors GQ's *"spend more time here."* |
| **"Vague" trigger** | The LLM **specificity judgment is primary**; word count is only a **weak secondary nudge**, never a standalone reason to probe. | A short answer can be substantive ("I quit at the payment screen"). |
| **Leading-question detection** | **Hybrid**: deterministic regex/heuristics (presupposition + loaded + double-barreled) **and** an isolated LLM judge. | Defense in depth; the cheap rules are testable, the judge catches subtler presupposition. |
| **Headline metric** | **Scenario match-accuracy** (behaves as expected on N/5 hand-labeled scenarios), with relevant-follow-up-rate and "0 leading questions" alongside. | It measures *interviewing well*, not just task completion — and the headline number is grounded in **human labels**, not an LLM. |

---

## The eval layer

Two **deliberately separated** kinds of measurement, so a model never grades itself:

### 1. Ground-truth behavioral accuracy — the headline (no LLM in the loop)
`evals/scenarios.py` defines five adversarial personas, each with a **hand-labeled** expected
behavior. The **scenario match-accuracy** compares the moderator's *actual* recorded behavior
to the human label by **deterministic** check:

| Scenario | Hand-labeled expectation |
| --- | --- |
| Vague one-word answerer | Probe for specifics (≥ 1 follow-up) |
| Volunteers later-objective info | Cover the volunteered objective; **don't re-ask** it |
| Off-topic rambler | Steer back (≥ 1 redirect) |
| Ideal participant (control) | Few/no probes (≤ 1) **and** ≥ 80% coverage |
| Uncooperative "I don't know" | **Terminate gracefully** within budget; no probe-loop |

### 2. LLM-as-judge — scoped narrowly, isolated
`evals/judge.py` scores **only follow-up relevance** ("did this follow-up dig into a real
gap?") and offers a *second opinion* on leading questions. It runs in its **own context, on a
different model** from the moderator (`gemini-2.5-flash` vs the moderator's `flash-lite`), at
temperature 0. It **never** computes the headline accuracy.

### Run it (one command)

```bash
python -m probeai.evals.run            # all 5 scenarios (cheap defaults — see below)
python -m probeai.evals.run --judge    # also score follow-up relevance (extra LLM calls)
python -m probeai.evals.run --persona  # LLM participant personas instead of fixed scripts
python -m probeai.evals.run --rich     # full LLM generation for every moderator line
python -m probeai.evals.run --use-cached    # re-score saved runs, zero LLM calls
python -m probeai.evals.run --scenario vague_oneword
```

**Quota-lean by default.** To run on a tight free tier, the harness defaults to: a **compact
3-objective eval study**, **scripted participant answers** (deterministic + no participant LLM
calls), **lean generation** (only *probes* hit the LLM; rote open/next/close lines are
templated from the researcher's seed questions), and the **judge off**. That takes a full run
from ~60 calls down to ~20–25. Per-scenario results are **cached** to `data/eval_runs/`, so if
a daily quota runs out mid-run you can finish the rest after the reset and then render the full
report with `--use-cached`. `--persona`/`--rich`/`--judge` trade calls for fidelity.

Example report **format** (run the command for live numbers from your machine):

```
Scenario match-accuracy (moderator behavior vs. hand-labeled expectation):
  Vague one-word answerer     Probe for specifics (>=1 follow-up)        PASS
  Volunteers later-objective  Cover volunteered objective; don't re-ask  PASS
  Off-topic rambler           Steer back (>=1 redirect)                  PASS
  Ideal participant (control) Few/no probes (<=1) and >=80% coverage     PASS
  Uncooperative "I don't know"Terminate gracefully; no probe-loop        PASS
  Scenario match-accuracy: 5/5 (100%)

  Leading questions:        0  (target: 0)
  Stacked (double-barreled):0  (target: 0)
  Relevant follow-up rate:  Y%

HEADLINE: "Behaves as expected on 3/3 completed scenarios (100%), 0 leading
           questions — coverage 67% (2 scenarios pending quota reset)."
```

### Evaluation limitations (read this)

The moderator, the simulated participant, **and** the judge are all Gemini models — a shared
model family is a real evaluation risk (a model may be biased toward rating its own family's
output favorably). Mitigations, by design:

1. **The headline number does not use an LLM at all.** Scenario match-accuracy is a
   deterministic comparison of the moderator's behavior against *my* hand-written labels.
2. **The judge is isolated** — separate client, **different model**, temperature 0, and it
   only ever sees the snippet it's judging (never the moderator's prompt or reasoning).
3. **The judge is scoped to relevance**, the one thing a deterministic rule can't capture.

The ideal next step is a judge from a *different provider family* entirely; on a $0 free tier
that wasn't available, so isolation-by-config is the honest compromise — stated here plainly.

### Tests

`pytest` covers **deterministic logic only** — coverage transitions, turn-budget/priority
math, leading-question heuristics, scenario checks, synthesis grounding. **No assertions on
raw LLM output** (it's stochastic); LLM behavior is exercised in the eval-harness layer.

```bash
pytest            # 61 tests, no API key required
```

---

## Where this plugs into Great Question (`gq_mock.py`)

I don't have access to GQ's real API/MCP, so `gq_mock.py` is a thin stand-in that marks the
exact integration seam:

| `gq_mock` function | Great Question concept (production) |
| --- | --- |
| `get_study(study_id)` | GQ **Studies** — fetch the discussion guide |
| `save_transcript(...)` | GQ **Interviews / Transcripts** |
| `save_highlights(...)` | GQ **Highlights** — objective-linked quoted excerpts |
| `save_synthesis(...)` | GQ **Insights / Analysis** |

Swapping these bodies for real MCP/REST calls is the entire "productionize against Great
Question" step.

---

## Non-goals (explicit)

- **Vision awareness** (watching a participant use a prototype) — the hardest piece; skipped.
- **Full realtime duplex / barge-in audio** — turn-based is intentional and more reliable.
- **Real Great Question API/MCP** — mocked via `gq_mock.py`.
- **Recruiting, scheduling, auth, multi-tenant, persistence beyond local files.**

---

## Setup & run

```bash
# 1. Install
python -m venv .venv && .venv/Scripts/activate        # (Windows; use source on macOS/Linux)
pip install -r requirements.txt

# 2. Add a free Gemini key (no credit card): https://aistudio.google.com/apikey
cp .env.example .env        # then put your key in GEMINI_API_KEY=

# 3a. Web app (voice + UI)
uvicorn probeai.server:app --reload     # open http://localhost:8000

# 3b. Or the terminal demo
python -m probeai.cli                       # type answers yourself
python -m probeai.runner --persona vague_oneword -v   # watch it probe a simulated participant

# 4. Evals
python -m probeai.evals.run
```

**Swap in a different study:** copy `probeai/config/study_checkout.yaml`, edit the research
goal / objectives / priorities, and pass `--study path/to/your.yaml` (CLI/runner) or point
`gq_mock.get_study` at it.

### Cost & the Claude Code gotcha

This app costs **$0 to run**: it calls **Gemini's free tier** only. I *built* it with Claude
Code (covered by my Claude subscription) — note that **if `ANTHROPIC_API_KEY` is set during
development, Claude Code bills the paid API instead of your subscription**, so keep that
variable unset. ProbeAI itself never calls Anthropic at runtime.

> **Free-tier note:** Gemini's free tier throttles aggressively. Some projects are capped at a
> very low shared **per-day** request budget (~20/day across models). The client retries
> transient throttles (429 per-minute / 5xx) with backoff and honors the server's stated
> `retryDelay`, but **fails fast on per-day exhaustion** (it won't recover within a backoff
> window). The moderator runs on `flash-lite` (more headroom) with the judge isolated on
> `flash`; the harness defaults (scripted + lean + judge-off + compact study) plus
> `--use-cached` keep a full run inside a tight daily budget.
