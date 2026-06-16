# DEMO_SCRIPT — ProbeAI (2–3 minute walkthrough)

Goal: show a reviewer at Great Question that this is *their* product in miniature — an
adaptive AI moderator with an honest eval layer — not a generic chatbot.

**Before recording**
- `uvicorn probeai.server:app` → open `http://localhost:8000`
- Have a higher-quota / fresh `GEMINI_API_KEY` set (a full interview + synthesis + eval is
  many calls; the default free tier can throttle).
- Pre-run `python -m probeai.evals.run` once so you can show the numbers without waiting live.
- Use Chrome (best Web Speech API support). Mic on, or use the text box.

---

### Beat 1 — Frame it in their words (~20s)
> "Surveys scale but stay shallow; interviews go deep but don't scale. Great Question's AI
> Moderated Interviews close that gap. I built **ProbeAI** — a discussion-guide–driven AI
> moderator that adapts and probes on response gaps, and then **measures whether it
> interviewed well.**"

Point at the header tagline and the **Objective coverage** panel (5 prioritized objectives,
all *uncovered*).

### Beat 2 — Start the interview; it speaks (~15s)
Click **Start interview**. The moderator opens warmly and asks the first high-priority
question — and **speaks it aloud** (TTS). Note: "Turn-based, push-to-talk — reliable to demo."

### Beat 3 — The differentiator: adaptive probing (~30s)
Answer vaguely — type or say: **"It was annoying."**
- Watch the **decision rationale** chip: `partial → PROBE`.
- The follow-up digs into the *specific* gap: *"You said it was annoying — what specifically
  made it annoying?"*
> "This is the whole game: it probes **response gaps** instead of marching through a script."

Now answer specifically (e.g. *"The payment screen rejected my card three times"*). Watch the
objective flip **uncovered → covered**, with **evidence: turn N**.

### Beat 4 — It adapts, doesn't re-ask (~25s)
Switch on **Simulate participant → "Volunteers later-objective info"**, click **Generate
answer**, **Send**. The participant volunteers a trust/security detail.
> "It just covered a *later* objective from volunteered info — and it won't re-ask it. That's
> using the guide as a framework, not a rigid script."

The coverage panel shows that later objective covered without ever being asked.

### Beat 5 — Graceful close (~10s)
Let it run to the end (or use the *Ideal participant* persona to get there fast). It thanks
the participant and stops once objectives are covered / the budget is hit — no awkward loop.

### Beat 6 — Analysis built-in: synthesis (~20s)
Click **Synthesize**.
> "Like GQ's analysis-built-in: a findings summary plus **highlights** — and every quote is
> verified verbatim against the transcript, so no hallucinated quotes end up in a deliverable."

### Beat 7 — The serious part: evals (~30s)
Click **Run eval** (tick **use judge**) for this interview: **coverage %**, **0 leading
questions**, **0 stacked questions**, **relevant follow-up rate**.

Then cut to the terminal and show the offline harness (quota-lean defaults):
```
python -m probeai.evals.run            # scripted + lean + compact study (~20 calls)
python -m probeai.evals.run --judge    # add follow-up-relevance scoring if quota allows
```
> "Five adversarial scenarios, each with a behavior I **hand-labeled**. The headline number —
> *behaves as expected on 5/5* — is a **deterministic** comparison to my labels, **not** an
> LLM grading itself. The LLM judge is isolated, on a different model, and only scores
> follow-up *relevance*."

Land the line:
> **"It doesn't just interview — I measured whether it interviews well: 3/3 completed
> scenarios (100%), 0 leading questions — coverage 67%."**

### Beat 8 — "Here's where it plugs into your platform" (~15s)
Open `probeai/gq_mock.py`.
> "I don't have your API, so this is the seam. `get_study`, `save_transcript`,
> `save_highlights`, `save_synthesis` map 1:1 to Great Question's Studies, Transcripts,
> Highlights, and Insights. Swapping these for your MCP tools is the whole productionization
> step."

---

### If asked "what would you do next?"
- A judge from a **different provider family** (removes the shared-model-family eval risk).
- **Vision awareness** (the objective I scoped out) for prototype-usage studies.
- Streaming/duplex audio for barge-in once turn-based is solid.
- Wire `gq_mock` to the real MCP tools.
