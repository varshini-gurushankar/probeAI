/* ProbeAI front end — turn-based voice loop + live coverage + analysis.
   Voice uses the browser Web Speech API (free, no key); text input is always
   available as a fallback. The "brain" lives in the FastAPI backend. */

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "request failed");
  }
  return res.json();
};

const els = {
  transcript: $("transcript"), coverage: $("coverage"), rationale: $("rationale"),
  answer: $("answer"), status: $("status"), turnCounter: $("turn-counter"),
  start: $("start-btn"), send: $("send-btn"), mic: $("mic-btn"), simAnswer: $("sim-answer-btn"),
  simToggle: $("sim-toggle"), persona: $("persona"), tts: $("tts-toggle"),
  synth: $("synth-btn"), eval: $("eval-btn"), judge: $("judge-toggle"),
  synthesis: $("synthesis"), evalOut: $("eval"),
};

let lastModeratorLine = "";
let finished = false;

/* ---------- rendering ---------- */
function addBubble(who, text) {
  const empty = els.transcript.querySelector(".empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.innerHTML = `<div class="who">${who === "moderator" ? "Moderator" : "Participant"}</div>${escapeHtml(text)}`;
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function renderCoverage(items) {
  els.coverage.innerHTML = "";
  for (const c of items) {
    const li = document.createElement("li");
    li.className = "cov-item";
    const evidence = c.evidence_turn_ids?.length ? `· evidence: turn ${c.evidence_turn_ids.join(", ")}` : "";
    const probes = c.probes_used ? `· ${c.probes_used} probe${c.probes_used > 1 ? "s" : ""}` : "";
    li.innerHTML = `
      <span class="dot ${c.status}"></span>
      <div class="cov-body">
        <div class="cov-label">${escapeHtml(c.label)}</div>
        <div class="cov-sub">${c.status} ${probes} ${evidence}</div>
      </div>
      <span class="prio ${c.priority}">${c.priority}</span>`;
    els.coverage.appendChild(li);
  }
}

function setTurn(used, budget) { els.turnCounter.textContent = `turn ${used} / ${budget}`; }
function setStatus(msg) { els.status.textContent = msg || ""; }

function showRationale(decision, verdict) {
  els.rationale.hidden = false;
  els.rationale.innerHTML =
    `<strong>${decision.action}</strong> — ${escapeHtml(decision.rationale)} ` +
    `<span style="opacity:.7">(answer judged: ${verdict.status}${verdict.is_specific ? ", specific" : ""})</span>`;
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* ---------- TTS ---------- */
function speak(text) {
  if (!els.tts.checked || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.02; u.pitch = 1.0;
  window.speechSynthesis.speak(u);
}

/* ---------- STT (push to talk) ---------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recog = null, listening = false;
if (SR) {
  recog = new SR();
  recog.lang = "en-US"; recog.interimResults = true; recog.continuous = true;
  recog.onresult = (e) => {
    let txt = "";
    for (let i = 0; i < e.results.length; i++) txt += e.results[i][0].transcript;
    els.answer.value = txt;
  };
  // diagnostics: tells us how far the pipeline gets (mic → audio → speech → text)
  recog.onaudiostart = () => setStatus("Mic live — speak now, then release.");
  recog.onspeechstart = () => setStatus("Hearing you…");
  recog.onnomatch = () => setStatus("Heard audio but couldn't transcribe it — try Chrome, or speak more clearly.");
  recog.onerror = (e) => {
    listening = false; els.mic.classList.remove("listening"); els.mic.textContent = "🎤 Hold to speak";
    const msg = {
      "not-allowed": "Mic blocked — click the 🔒 (or mic) icon in Chrome's address bar → Allow → reload.",
      "service-not-allowed": "Mic blocked by the browser — allow microphone access and reload.",
      "no-speech": "Didn't catch any speech — hold the button, speak, then release.",
      "audio-capture": "No microphone found — check it's plugged in and set as default in Windows sound settings.",
      "network": "Speech service couldn't reach the network — check your internet connection.",
      "aborted": "",
    }[e.error] ?? ("mic error: " + e.error);
    if (msg) setStatus(msg);
  };
  recog.onend = () => { listening = false; els.mic.classList.remove("listening"); els.mic.textContent = "🎤 Hold to speak"; };
} else {
  els.mic.title = "Web Speech API not supported in this browser — use Chrome, or type.";
}

// Pre-flight the microphone once so a denied/missing/busy device gives a clear,
// actionable error up front instead of SpeechRecognition silently yielding no text.
let micGranted = false;
async function ensureMic() {
  if (micGranted || !navigator.mediaDevices?.getUserMedia) return true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((t) => t.stop());  // we only needed the permission grant
    micGranted = true;
    return true;
  } catch (err) {
    setStatus({
      NotAllowedError: "Mic permission denied — click the 🔒 icon left of the URL → Microphone → Allow, then reload.",
      NotFoundError: "No microphone detected — plug one in or pick a default mic in Windows sound settings.",
      NotReadableError: "Mic is busy in another app — close Zoom/Teams/etc., then try again.",
    }[err.name] ?? ("Mic unavailable: " + err.name));
    return false;
  }
}

async function startListening() {
  if (!recog || listening || finished) return;
  window.speechSynthesis?.cancel();  // stop TTS so the mic doesn't pick it up
  if (!(await ensureMic()) || listening || finished) return;  // re-check after the await
  els.answer.value = "";
  try { recog.start(); listening = true; els.mic.classList.add("listening"); els.mic.textContent = "● listening…"; setStatus("Listening — release to stop."); }
  catch (_) { setStatus("mic unavailable — try releasing and holding again"); }
}
function stopListening() {
  if (!recog || !listening) return;
  recog.stop(); setStatus("");
}

/* ---------- flow ---------- */
async function loadStudy() {
  const s = await api("/api/study");
  $("study-title").textContent = s.title;
  $("research-goal").textContent = s.research_goal;
  renderCoverage(s.coverage);
  setTurn(0, s.turn_budget);
}

async function start() {
  setStatus("Starting…"); els.start.disabled = true;
  try {
    await api("/api/reset", {});
    const r = await api("/api/start", {});
    finished = false;
    els.transcript.innerHTML = "";
    els.synthesis.innerHTML = ""; els.evalOut.innerHTML = "";
    lastModeratorLine = r.moderator_line;
    addBubble("moderator", r.moderator_line);
    speak(r.moderator_line);
    renderCoverage(r.coverage);
    setTurn(r.turns_used, r.turn_budget);
    els.send.disabled = false; els.mic.disabled = !recog;
    els.simAnswer.hidden = !els.simToggle.checked;
    setStatus("Your move — answer as the participant.");
  } catch (e) { setStatus("Error: " + e.message); els.start.disabled = false; }
}

async function sendAnswer() {
  const answer = els.answer.value.trim();
  if (!answer || finished) return;
  els.send.disabled = true; els.simAnswer.disabled = true; setStatus("Moderator is thinking…");
  addBubble("participant", answer);
  els.answer.value = "";
  try {
    const r = await api("/api/turn", { answer });
    lastModeratorLine = r.moderator_line;
    showRationale(r.decision, r.verdict);
    addBubble("moderator", r.moderator_line);
    speak(r.moderator_line);
    renderCoverage(r.coverage);
    setTurn(r.turns_used, r.turn_budget);
    if (r.finished) endInterview();
    else { els.send.disabled = false; els.simAnswer.disabled = false; setStatus(""); }
  } catch (e) {
    setStatus("Error: " + e.message); els.send.disabled = false; els.simAnswer.disabled = false;
  }
}

async function simAnswer() {
  if (finished) return;
  els.simAnswer.disabled = true; setStatus("Simulated participant is answering…");
  try {
    const r = await api("/api/participant", { persona_id: els.persona.value, moderator_line: lastModeratorLine });
    els.answer.value = r.answer;
    setStatus("Generated — review, then Send.");
  } catch (e) { setStatus("Error: " + e.message); }
  els.simAnswer.disabled = false;
}

function endInterview() {
  finished = true;
  els.send.disabled = true; els.mic.disabled = true; els.simAnswer.disabled = true;
  els.synth.disabled = false; els.eval.disabled = false;
  setStatus("Interview complete. Run synthesis and the eval report →");
}

async function runSynthesis() {
  els.synth.disabled = true; setStatus("Synthesizing findings…");
  try {
    const s = await api("/api/synthesize", {});
    let html = `<div class="finding"><h3>Findings summary</h3>${escapeHtml(s.summary)}</div>`;
    if (s.highlights?.length) {
      html += `<div class="finding"><h3>Highlights</h3>`;
      for (const h of s.highlights) {
        html += `<div class="hl"><div class="q">“${escapeHtml(h.quote)}”</div>
          <div class="meta">${escapeHtml(h.objective_id)} · turn ${h.turn_id} — ${escapeHtml(h.insight)}</div></div>`;
      }
      html += `</div>`;
    }
    els.synthesis.innerHTML = html;
    setStatus("Saved via gq_mock (study/transcript/highlights).");
  } catch (e) { setStatus("Error: " + e.message); }
  els.synth.disabled = false;
}

async function runEval() {
  els.eval.disabled = true; setStatus("Scoring interview quality…");
  try {
    const r = await api("/api/eval", { use_judge: els.judge.checked });
    const cov = r.coverage;
    const lead = r.leading, oneq = r.one_question, rel = r.followup_relevance;
    let html = `<div class="finding"><h3>Interview-quality report</h3>`;
    html += metric("Objective coverage", `${Math.round(cov.pct)}% (${cov.covered}/${cov.total})`);
    html += metric("Leading questions", lead.count, lead.count === 0);
    html += metric("Stacked (double-barreled)", oneq.stacked_count, oneq.stacked_count === 0);
    if (rel && rel.rate_pct != null) html += metric("Relevant follow-ups", `${Math.round(rel.rate_pct)}% (${rel.relevant}/${rel.total})`);
    else html += metric("Follow-ups captured", r.followups_captured + (els.judge.checked ? "" : " (enable judge to score)"));
    html += `</div>`;
    for (const v of lead.violations || []) html += `<div class="violation">⚠ leading [${v.source}]: “${escapeHtml(v.question)}”</div>`;
    els.evalOut.innerHTML = html;
    setStatus("");
  } catch (e) { setStatus("Error: " + e.message); }
  els.eval.disabled = false;
}

function metric(label, value, good) {
  const cls = good === true ? "zero" : good === false ? "bad" : "";
  return `<div class="metric"><span>${label}</span><span class="v ${cls}">${value}</span></div>`;
}

/* ---------- wiring ---------- */
els.start.onclick = start;
els.send.onclick = sendAnswer;
els.simAnswer.onclick = simAnswer;
els.synth.onclick = runSynthesis;
els.eval.onclick = runEval;
els.simToggle.onchange = () => {
  els.persona.disabled = !els.simToggle.checked;
  els.simAnswer.hidden = !els.simToggle.checked || finished;
};
els.answer.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) sendAnswer(); });
// push-to-talk: hold the mic button
["mousedown", "touchstart"].forEach((ev) => els.mic.addEventListener(ev, (e) => { e.preventDefault(); startListening(); }));
["mouseup", "mouseleave", "touchend"].forEach((ev) => els.mic.addEventListener(ev, stopListening));

loadStudy().catch((e) => setStatus("Could not load study: " + e.message));
