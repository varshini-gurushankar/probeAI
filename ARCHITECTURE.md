# ProbeAI architecture — the moderator turn, explained

This is the study sheet for how ProbeAI runs one interview turn. The runtime is the
**LangGraph moderator** in [`probeai/agent_graph.py`](probeai/agent_graph.py); the classic
imperative engine ([`moderator.py`](probeai/moderator.py) `InterviewSession.step`) is still
there as an opt-out (`--classic`) and as the parity reference the graph is checked against.

The design in one line: **agentic behavior with deterministic policy control.** The LLM
only does language — judge an answer, write a question. Every decision about *what to do*
(probe? move on? steer back? stop?) is plain, testable Python. LangGraph is the scaffold
that makes that split visible.

---

## One turn, end to end

A turn starts with the participant's latest answer and ends with one moderator line.

```
START
 → assess_answer      (LLM)   read the answer → verdict (status, is_specific, gap)
 → update_coverage    (pure)  fold the verdict into the per-objective state machine
 → decide_action      (pure)  decide_next_action → PROBE / ASK_NEXT / STEER_BACK / CLOSE
     ├─ CLOSE → synthesize (LLM/template)  emit the closing line ───────────────→ END
     └─ else → generate_question (LLM/template)  turn the decision into one line
         → validate_question  (deterministic: leading? double-barreled? off-topic?)
             ├─ valid ─────────────────────────────────────────────────────────→ END (emit)
             ├─ flagged, repairs left → repair_question (LLM) → validate_question …
             └─ flagged, repairs spent → fallback (safe generic question) ──────→ END
```

**The eight nodes:**

| Node | Kind | What it does |
| --- | --- | --- |
| `assess_answer` | LLM | Judges the answer → a `Verdict` (covered/partial/off-topic, is_specific, and the `gap` if vague). |
| `update_coverage` | pure | Applies the verdict to the `CoverageState` machine (uncovered → partial → covered, records evidence turns). |
| `decide_action` | pure | Runs `decide_next_action` to pick the move, and does the same bookkeeping `step()` does (record a probe, mark an objective asked, or flag wrap). |
| `generate_question` | LLM/template | Turns the decision into one utterance. The existing `Moderator.generate` already branches on PROBE / ASK_NEXT / STEER_BACK, so one node covers all three. |
| `validate_question` | pure | The quality gate. Reuses the eval heuristics for leading/loaded/double-barreled phrasing plus a cheap relevance check. |
| `repair_question` | LLM | Rewrites a flagged question to be open, neutral, single, and grounded — using the centralized `MODERATOR_REPAIR` prompt. |
| `fallback` | pure | Last resort when repair is exhausted: emit the neutral `GENERIC_FALLBACK` rather than a bad question. |
| `synthesize` | LLM/template | The wrap-up terminal: emits the closing line. (Full findings synthesis is a separate post-interview call to avoid an LLM cost on every wrap.) |

---

## What flows through the state

State is a `TypedDict` (`InterviewGraphState`). Two halves:

- **Shared references**, seeded once per turn from the live session and *mutated in place*
  exactly as the classic engine mutates them: `study`, `policy`, `transcript`,
  `moderator`, `coverage`. Holding references (not copies) is what keeps coverage and the
  transcript stable from one turn to the next.
- **Per-turn working data**, where each field is written by exactly one kind of node:

| Field | Written by | Meaning |
| --- | --- | --- |
| `answer`, `answer_turn_id` | seeded | the participant's line + its transcript id |
| `verdict` | `assess_answer` | the LLM's read of the answer |
| `decision` | `decide_action` | the chosen move + target objective + rationale |
| `current_objective_id` | seeded, rewritten by `decide_action` | which objective is in focus (changes on ASK_NEXT) |
| `question` | `generate` / `repair` / `fallback` / `synthesize` | the line that will be spoken |
| `validation` | `validate_question` | `{valid, reasons}` |
| `repair_attempts` | `repair_question` | how many rewrites have happened |
| `should_wrap` | `decide_action` / `synthesize` | whether this turn ends the interview |
| `trace` | every node (add-reducer) | one record per node — the inspectable history |

`trace` uses an `operator.add` reducer, so each node *appends* its record instead of
overwriting. That list is what shows up in `runner -v` and in the web backend's
`/api/turn` response, and it's how you watch a turn flow through the graph.

---

## Why decision is a *separate* node from generation

This is the core design choice. `decide_action` (policy) and `generate_question`
(language) are deliberately split:

- **Policy is deterministic and auditable.** Whether to probe, how many follow-ups an
  objective is allowed, when to move on, when to stop — all of that lives in
  `decide_next_action` and the `CoverageState` machine, as pure functions with no LLM in
  the loop. You can unit-test every branch, and the eval harness scores behavior against
  hand labels without ever asking a model to grade itself.
- **Language is the LLM's only job.** The model reads the answer (`assess`) and phrases the
  question (`generate`/`repair`). It never decides coverage, caps, or termination.

Keeping them separate means a model that hallucinates or drifts can change *how a question
is worded* but can never change *what the interview decides to do*. That's the whole safety
argument, and it's only legible because the two are different nodes.

---

## The validate → repair → fallback loop

Before any question is spoken, `validate_question` checks it with the same heuristics the
eval layer uses: presupposition / loaded phrasing (leading), asking two things at once
(double-barreled), and a conservative relevance guard. It's **deterministic by default** so
evals stay cheap; the LLM judge is opt-in.

If a question is flagged, `repair_question` rewrites it and we re-validate. This is the only
cycle in the graph, and it is **bounded**: at most `max_repairs` (default 2) rewrites, after
which `fallback` emits the neutral `GENERIC_FALLBACK`. So the loop always terminates, a
stubborn model can never spin forever or ship a leading question, and the worst-case LLM
cost of a turn is capped at two extra calls.

---

## What LangGraph gives this design — and what it doesn't

**Gives:**
- *Inspectable control flow.* The turn is a set of named nodes and explicit edges instead
  of a hidden imperative sequence. You can read `build_interview_graph()` and explain the
  whole turn; the `trace` shows the actual path taken.
- *A clean home for the repair loop.* "Re-validate after each rewrite, bounded by a cap" is
  a natural conditional cycle — exactly what a graph expresses well.

**Doesn't:**
- LangGraph is **not required** for any of this. The same assess → decide → generate →
  validate → repair logic runs fine as the imperative `InterviewSession.step`. The graph is
  a **structure choice** — it buys clarity and inspectability, not capability. The classic
  engine produces the same behavior (verified: the five hand-labeled scenarios pass/fail
  identically under both), which is the point — the graph reorganizes the turn without
  changing what it decides.

See the README's "LangGraph moderator runtime" section for the project-level summary.
