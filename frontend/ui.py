# ui.py
# Frontend launcher. A local web UI (HTML/CSS/JS) in a native window via pywebview.
# Python owns the window and the microphone; the page just displays things.
#
# IMPORTANT design point: background threads (the ASR worker) must NEVER touch the
# webview directly - calling evaluate_js from a non-GUI thread crashes WebView2.
# Instead, those threads only update a small shared `state` dict, and the PAGE
# polls Python for it (JS -> Python calls are safe). That's the whole trick here.
#
# Run:  python frontend/ui.py
import csv
import sys
import threading
import time
from pathlib import Path

import webview

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))                 # so `tacet` and `frontend` import

import config as C                            # noqa: E402

INDEX = HERE / "web" / "index.html"
WHISPER_DIR = ROOT / "models" / "whisper"     # used if present (offline/packaged)
BERT_DIR = ROOT / "models" / "bert"
LEAK_CSV = HERE / "leaks" / "transcripts.csv"

from frontend.asr import LiveTranscriber       # noqa: E402
from tacet.encoder import BertEncoder          # noqa: E402
from tacet.infer import TacetEndpointer        # noqa: E402
from tacet.gate import should_respond          # noqa: E402
from tacet import llm, tts                      # noqa: E402
from voice_mngt.pitch import PITCH_WINDOW_SECONDS, RollingPitchTracker  # noqa: E402
from voice_mngt.spectrogram import spectrogram # noqa: E402

TALK_SECONDS = 4                               # mock duration the model "talks"
SILENCE_HOLD = C.MIN_THRESHOLD / 1000 if C.MIN_THRESHOLD > 10 else C.MIN_THRESHOLD


def _threshold_seconds(value):
    return value / 1000 if value > 10 else value


class Api:
    """Bridge between the web page and Python.
    The page calls poll() ~5x/second to read the latest state."""

    def __init__(self):
        self.recording = False
        self.tr = None
        self.encoder = BertEncoder(model_dir=BERT_DIR)
        self.endpointer = None               # loaded lazily on first record
        self.llm = "Gemini"                   # only Gemini is implemented
        self.save_leaks = True               # toggled from the UI
        self._talking = False
        self._last_prob = 0.0
        self._last_pitch = None
        self._pitch_tracker = None
        self._pitch_sample = None
        self._lock = threading.Lock()
        self._state = {"status": "ready", "transcript": "", "committed": "",
                       "final": False, "error": "", "busy": False, "talking": False,
                       "encode_count": 0, "encoded_text": "", "prob": 0.0,
                       "pitch_hz": None, "final_pitch_hz": None,
                       "pitch_drop_ratio": None, "pitch_falling": False,
                       "pitch_contour": []}
        self._committed = ""                  # committed prefix (for the text view only)
        self._raw = ""                        # latest raw transcript (drives the decision)
        self._raw_change = 0.0                # when raw last changed
        self._last_change = 0.0
        self._last_voice_time = 0.0
        self._last_encoded = ""               # last text we ran through BERT
        self._encode_count = 0
        self._enc_worker = None

    # ---- thread-safe state ----
    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    # ---- called by the page ----
    def ping(self):
        return "connected"

    def set_save_leaks(self, enabled):
        self.save_leaks = bool(enabled)
        return self.save_leaks

    def set_llm(self, name):
        self.llm = name
        return name

    def shutdown(self):
        self.recording = False
        self._talking = False
        try:
            if self.tr:
                self.tr.on_partial = None
                self.tr.on_display = None
                self.tr.on_final = None
                self.tr.on_status = None
                self.tr.stop()
            if self._pitch_tracker:
                self._pitch_tracker.stop()
        except Exception:
            pass
        return True

    def poll(self):
        with self._lock:
            snap = dict(self._state)
        snap["recording"] = self.recording
        snap["level"] = self.tr.level() if (self.tr and self.recording) else 0.0
        return snap

    def toggle_record(self):
        if not self.recording:
            self.recording = True
            self._set(busy=True, status="starting microphone...", transcript="",
                      final=False, error="")
            threading.Thread(target=self._start, daemon=True).start()
        else:
            self.recording = False
            self._set(busy=True, status="finishing up...")
            threading.Thread(target=self._stop, daemon=True).start()
        return {"recording": self.recording}

    # ---- internals (background threads; only touch _state) ----
    def _start(self):
        self.tr = LiveTranscriber(
            on_partial=self._on_partial,
            on_display=self._on_display,
            on_final=self._on_final,
            on_status=lambda t: self._set(status=t),
            model_dir=WHISPER_DIR,
        )
        try:
            if not self.recording:
                return
            self._set(status="warming pitch tracker...")
            self._pitch_tracker = RollingPitchTracker(
                lambda: self.tr.audio_window_before_last_speech(PITCH_WINDOW_SECONDS),
                token_source=self.tr.last_voice_sample,
            )
            self._pitch_tracker.start()
            self.tr.start()
            if not self.recording:
                self.tr.stop()
                self._pitch_tracker.stop()
                return
            self.encoder.ensure_loaded(status=lambda t: self._set(status=t))
            if not self.recording:
                self.tr.stop()
                self._pitch_tracker.stop()
                return
            if self.endpointer is None:
                self._set(status="loading head...")
                self.endpointer = TacetEndpointer()      # wire the NN (loads once)
            self._committed = ""
            self._last_encoded = ""
            self._last_change = time.monotonic()
            self._last_voice_time = self._last_change
            self._set(busy=False, status="listening...")
            self._enc_worker = threading.Thread(target=self._encode_loop, daemon=True)
            self._enc_worker.start()
        except Exception as e:
            self.recording = False
            if self._pitch_tracker:
                self._pitch_tracker.stop()
            self._set(busy=False, status="error", error=f"start error: {e}")

    def _on_display(self, text):
        if self._talking:
            return
        with self._lock:
            if text != self._raw:
                self._raw = text
                self._raw_change = time.monotonic()
            self._state.update(transcript=text, final=False)

    def _on_partial(self, text):
        if self._talking:
            return
        with self._lock:
            if text != self._committed:
                self._committed = text
                self._last_change = time.monotonic()
            self._state["committed"] = text

    def _encode_loop(self):
        required_silence = 3000
        while self.recording:
            time.sleep(0.1)
            if self._talking:
                continue
            now = time.monotonic()
            audio_level = self.tr.level() if self.tr else 0.0
            if spectrogram(audio_level):
                self._last_voice_time = now
            callback_voice_at = self.tr.last_voice_time() if self.tr else 0.0
            if callback_voice_at:
                self._last_voice_time = max(self._last_voice_time, callback_voice_at)
            pitch_state = self._refresh_pitch()
            with self._lock:
                raw = self._raw
            if raw and raw != self._last_encoded:
                prob = self._encode(raw)                      # score the latest words
                if prob is not None:
                    required_silence = 1000 if prob > C.TAU else 3000
            elif raw:
                silence_duration = now - self._last_voice_time
                if should_respond(
                    self._last_prob,
                    audio_level=audio_level,
                    silence_duration=silence_duration,
                    required_silence=required_silence,
                    pitch_falling=bool(pitch_state and pitch_state.get("falling")),
                    pitch_fresh=bool(pitch_state and pitch_state.get("fresh")),
                ):                                            # decide at the pause
                    threading.Thread(target=self._take_turn, daemon=True).start()

    def _encode(self, text):
        try:
            emb = self.encoder.encode(text)
        except Exception as e:
            self._set(error=f"bert error: {e}")
            return None
        self._last_encoded = text
        self._encode_count += 1
        self._last_prob = self.endpointer.probability(emb)    # wire inference (no decision yet)
        self._set(encode_count=self._encode_count, encoded_text=text,
                  prob=round(self._last_prob, 3))
        return self._last_prob

    def _refresh_pitch(self):
        if not self._pitch_tracker or not self.tr:
            return None
        state = self._pitch_tracker.snapshot(self.tr.last_voice_sample())
        if state is None:
            return None
        if state["voice_sample"] == self._pitch_sample:
            return state

        self._pitch_sample = state["voice_sample"]
        if state.get("error"):
            self._set(error=f"pitch error: {state['error']}")
            return state
        self._last_pitch = state if "contour" in state else None
        if self._last_pitch is None:
            self._set(pitch_hz=None, final_pitch_hz=None, pitch_drop_ratio=None,
                      pitch_falling=False, pitch_contour=[])
            return state

        self._set(
            pitch_hz=round(self._last_pitch["median"], 2),
            final_pitch_hz=round(self._last_pitch["final"], 2),
            pitch_drop_ratio=round(self._last_pitch["drop_ratio"], 3),
            pitch_falling=self._last_pitch["falling"],
            pitch_contour=[round(value, 2) for value in self._last_pitch["contour"]],
        )
        return state

    def _take_turn(self):
        # model takes its turn: fade orange, ask the LLM, speak it, reset, fade back
        if self._talking:
            return
        self._talking = True
        text = self._raw                                  # what the user said
        self._set(talking=True, status="thinking...", transcript="", committed="")
        self._reset_turn()
        try:
            answer = llm.reply(text, model=self.llm)      # LLM
            self._set(status="speaking...")
            tts.speak(answer)                             # TTS (blocks while speaking)
        except Exception as e:
            self._set(error=f"reply error: {e}")
        self._reset_turn()                                # drop anything captured while talking
        self._talking = False
        self._set(talking=False, status="listening...", transcript="", committed="")

    def _reset_turn(self):
        if self.tr:
            self.tr.reset()
        with self._lock:
            self._committed = ""
            self._raw = ""
        self._last_encoded = ""
        self._last_pitch = None
        self._pitch_sample = None
        if self._pitch_tracker:
            self._pitch_tracker.reset()
        self._last_voice_time = time.monotonic()
        self._set(pitch_hz=None, final_pitch_hz=None, pitch_drop_ratio=None,
                  pitch_falling=False, pitch_contour=[])

    def _stop(self):
        try:
            if self.tr:
                self.tr.stop()
            if self._pitch_tracker:
                self._pitch_tracker.stop()
        except Exception as e:
            self._set(error=f"stop error: {e}")
        self._set(busy=False)

    def _on_final(self, text):
        self._set(transcript=text, final=True, status="turn complete")
        if text:
            self._encode(text)               # one final encode on the full turn
        self._leak(text)

    def _leak(self, text):
        """Append the finished transcript to a CSV with an empty label column,
        ready for hand-labeling (0/1) as real data."""
        if not text or not self.save_leaks:
            return
        print(f"[leak] {text}")
        LEAK_CSV.parent.mkdir(parents=True, exist_ok=True)
        is_new = not LEAK_CSV.exists()
        with open(LEAK_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["text", "label"])
            w.writerow([text, ""])


def main():
    api = Api()
    window = webview.create_window(
        "TACET - Turn-Aware Completion",
        str(INDEX),
        js_api=api,
        width=920,
        height=660,
        min_size=(760, 560),
        background_color="#0e0f13",
    )
    window.events.closed += api.shutdown
    try:
        webview.start()
    finally:
        api.shutdown()


if __name__ == "__main__":
    main()
