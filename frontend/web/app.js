// app.js - TACET frontend
// real microphone + Whisper.
// The page POLLS Python (~5x/sec) for the latest transcript/status

const els = {
  micBtn: document.getElementById("micBtn"),
  micHint: document.getElementById("micHint"),
  statePill: document.getElementById("statePill"),
  transcript: document.getElementById("transcript"),
  probText: document.getElementById("probText"),
  meterFill: document.getElementById("meterFill"),
  decisionBadge: document.getElementById("decisionBadge"),
  bridgeDot: document.getElementById("bridgeDot"),
  bridgeText: document.getElementById("bridgeText"),
  status: document.getElementById("status"),
  saveLeak: document.getElementById("saveLeak"),
};

const TAU = 0.5;
let recording = false;
let simTimer = null;
let pollTimer = null;

// ---- Python bridge helpers ----
function hasBridge() { return window.pywebview && window.pywebview.api; }

async function callPy(method, ...args) {
  if (!hasBridge()) return null;
  try { return await window.pywebview.api[method](...args); }
  catch (e) { console.error(e); return null; }
}

function setBridge(ok, text) {
  els.bridgeDot.className = "dot " + (ok ? "ok" : "bad");
  els.bridgeText.textContent = text;
}

// ---- UI updates ----
function setProbability(p) {
  if (p === null || p === undefined) { els.probText.textContent = "—"; els.meterFill.style.width = "0%"; return; }
  const pct = Math.round(p * 100);
  els.probText.textContent = pct + "%";
  els.meterFill.style.width = pct + "%";
  const respond = p >= TAU;
  els.decisionBadge.textContent = respond ? "respond" : "listening";
  els.decisionBadge.className = "badge " + (respond ? "respond" : "listen");
}

function setTranscript(text, live) {
  els.transcript.innerHTML = text
    ? text + (live ? ' <span class="cursor">▋</span>' : "")
    : '<span class="muted">Your speech will appear here…</span>';
}

function showLoading(msg) {
  els.transcript.innerHTML =
    '<span class="loading-line"><span class="dot-spin"></span>' +
    (msg || "working…") + "</span>";
}

function setRecordingVisual(on) {
  recording = on;
  els.micBtn.classList.toggle("recording", on);
  els.micHint.textContent = on ? "Listening… tap to stop" : "Tap to start listening";
  els.statePill.textContent = on ? "live" : "idle";
  els.statePill.className = "pill " + (on ? "live" : "");
}

// ---- polling loop (bridge mode) ----
async function poll() {
  const s = await callPy("poll");
  if (!s) return;
  setRecordingVisual(s.recording);
  els.micBtn.classList.toggle("loading", s.busy);
  if (s.busy) {
    showLoading(s.status || "working…");          // big, in the transcript area
  } else {
    setTranscript(s.transcript, s.recording && !s.final);
  }
  els.status.textContent = s.error ? ("⚠ " + s.error) : s.status;
}

// ---- mic button ----
els.micBtn.addEventListener("click", async () => {
  if (hasBridge()) {
    const res = await callPy("toggle_record");
    if (res) setRecordingVisual(res.recording);
  } else {
    setRecordingVisual(!recording);
    if (recording) startSim(); else stopSim();
  }
});

// ---- placeholder simulation (browser preview only) ----
const SAMPLE = "i would like to book a flight to athens next week".split(" ");
function startSim() {
  let i = 0;
  setTranscript("", true);
  els.status.textContent = "Browser preview — simulated (no microphone).";
  simTimer = setInterval(() => {
    if (i < SAMPLE.length) {
      i++;
      setTranscript(SAMPLE.slice(0, i).join(" "), true);
      setProbability(Math.min(0.97, 0.05 + (i / SAMPLE.length) * 0.95));
    } else { stopSim(); setRecordingVisual(false); }
  }, 600);
}
function stopSim() { if (simTimer) { clearInterval(simTimer); simTimer = null; } }

// ---- save-leak toggle ----
els.saveLeak.addEventListener("change", () => {
  callPy("set_save_leaks", els.saveLeak.checked);
  els.status.textContent = els.saveLeak.checked
    ? "Saving transcripts to CSV." : "Not saving — test mode.";
});

// ---- startup ----
function onReady() {
  setProbability(null);
  callPy("ping").then((r) => {
    if (r) {
      setBridge(true, "python: " + r);
      callPy("set_save_leaks", els.saveLeak.checked);       // sync initial state
      if (!pollTimer) pollTimer = setInterval(poll, 200);   // start polling
    } else {
      setBridge(false, "no bridge (browser preview)");
    }
  });
}
window.addEventListener("pywebviewready", onReady);
window.addEventListener("load", () => { if (!hasBridge()) onReady(); });
