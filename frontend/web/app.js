// TACET frontend — minimal voice UI. Page polls Python for state; orb reacts to mic level.
const $ = (id) => document.getElementById(id);
const els = { orb: $("orb"), transcript: $("transcript"), loader: $("loader"), hint: $("hint"), mic: $("micBtn"), llm: $("llm"), save: $("saveToggle"), text: $("textToggle"), dev: $("dev"), devCount: $("devCount"), devProb: $("devProb") };

let recording = false, saving = true, showText = false, lastEncode = 0;
let talking = false, talkMix = 0;
let targetLevel = 0, level = 0;
let pollTimer = null, simTimer = null;

const hasBridge = () => window.pywebview && window.pywebview.api;
async function callPy(m, ...a) { if (!hasBridge()) return null; try { return await window.pywebview.api[m](...a); } catch (e) { console.error(e); return null; } }

function setBusy(b) { els.loader.style.display = b ? "block" : "none"; els.orb.style.display = b ? "none" : "block"; els.mic.classList.toggle("loading", b); }
function setRecording(r) { recording = r; els.mic.classList.toggle("on", r); }
function setHint(t) { els.hint.textContent = t; }

async function poll() {
  const s = await callPy("poll");
  if (!s) return;
  setBusy(s.busy);
  setRecording(s.recording);
  talking = s.talking;
  targetLevel = talking ? 0 : (s.level || 0);   // ignore mic while the model talks
  els.devProb.textContent = (s.prob ?? 0).toFixed(2);
  if (showText) renderTranscript(s.transcript || "", s.committed || "");
  if (s.encode_count !== lastEncode) {
    lastEncode = s.encode_count;
    els.devCount.textContent = s.encode_count;
    els.dev.title = "P=" + s.prob + "  ·  " + (s.encoded_text || "");
    els.dev.classList.add("flash");
    setTimeout(() => els.dev.classList.remove("flash"), 250);
  }
  if (s.error) setHint("⚠ " + s.error);
  else if (s.talking) setHint("talking…");
  else if (s.busy) setHint(s.status || "loading…");
  else if (s.recording) setHint("listening…");
  else setHint("Tap the mic to start");
}

// committed prefix solid, volatile tail faint -> far fewer visible mutations
function renderTranscript(raw, committed) {
  let tail = raw;
  if (committed && raw.startsWith(committed)) tail = raw.slice(committed.length);
  else committed = "";
  els.transcript.innerHTML = committed + '<span class="tail">' + tail + "</span>";
}

function applyTextView() {
  els.text.classList.toggle("on", showText);
  els.orb.classList.toggle("faded", showText);
  els.transcript.style.display = showText ? "block" : "none";
}

// smooth orb animation: scale + brightness driven by loudness, gentle idle breathing
function animate() {
  level += (targetLevel - level) * 0.2;
  talkMix += ((talking ? 1 : 0) - talkMix) * 0.08;   // smooth blue<->orange fade
  const idle = (recording && !talking) ? 0 : Math.sin(Date.now() / 900) * 0.02;
  const scale = 1 + Math.min(level * 7, 0.6) + idle;
  els.orb.style.transform = `scale(${scale.toFixed(3)})`;
  els.orb.style.filter =
    `saturate(1.1) brightness(${1 + Math.min(level * 2, 0.5)}) hue-rotate(${(talkMix * 185).toFixed(0)}deg)`;
  requestAnimationFrame(animate);
}

els.mic.addEventListener("click", async () => {
  if (hasBridge()) {
    const r = await callPy("toggle_record");
    if (r) setRecording(r.recording);
  } else {
    setRecording(!recording);
    if (recording) simTimer = setInterval(() => { targetLevel = Math.random() * 0.08; }, 150);
    else { clearInterval(simTimer); targetLevel = 0; }
  }
});

els.llm.addEventListener("change", () => callPy("set_llm", els.llm.value));
els.save.addEventListener("click", () => { saving = !saving; els.save.classList.toggle("on", saving); callPy("set_save_leaks", saving); });
els.text.addEventListener("click", () => { showText = !showText; applyTextView(); });

function onReady() {
  els.save.classList.toggle("on", saving);
  callPy("set_save_leaks", saving);
  callPy("set_llm", els.llm.value);
  if (hasBridge() && !pollTimer) pollTimer = setInterval(poll, 150);
}
window.addEventListener("pywebviewready", onReady);
window.addEventListener("load", () => { if (!hasBridge()) onReady(); });
animate();
