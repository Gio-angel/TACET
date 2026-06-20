# ui.py
# Frontend launcher. A local web UI (HTML/CSS/JS) in a native window via pywebview.
# Python owns the window and the microphone; the page just displays things.
#
# IMPORTANT design point: background threads (the ASR worker) must NEVER touch the
# webview directly — calling evaluate_js from a non-GUI thread crashes WebView2.
# Instead, those threads only update a small shared `state` dict, and the PAGE
# polls Python for it (JS -> Python calls are safe). That's the whole trick here.
#
# Run:  python frontend/ui.py
import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np
import webview

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))                 # so `tacet` and `frontend` import

INDEX = HERE / "web" / "index.html"
WHISPER_DIR = ROOT / "models" / "whisper"     # used if present (offline/packaged)
BERT_DIR = ROOT / "models" / "bert"
LEAK_CSV = HERE / "leaks" / "transcripts.csv"

from frontend.asr import LiveTranscriber       # noqa: E402
from tacet.encoder import BertEncoder          # noqa: E402


class Api:
    """Bridge between the web page and Python.
    The page calls poll() ~5x/second to read the latest state."""

    def __init__(self):
        self.recording = False
        self.tr = None
        self.encoder = BertEncoder(model_dir=BERT_DIR)
        self.llm = "ChatGPT"                  # mock selection
        self.save_leaks = True               # toggled from the UI
        self._lock = threading.Lock()
        self._state = {"status": "ready", "transcript": "", "committed": "",
                       "final": False, "error": "", "busy": False,
                       "encode_count": 0, "encoded_text": "", "emb_norm": 0.0}
        self._committed = ""                  # latest committed transcript
        self._last_change = 0.0               # when it last changed
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

    def poll(self):
        with self._lock:
            snap = dict(self._state)
        snap["recording"] = self.recording
        snap["level"] = self.tr.level() if (self.tr and self.recording) else 0.0
        return snap

    def toggle_record(self):
        if not self.recording:
            self.recording = True
            self._set(busy=True, status="starting microphone…", transcript="",
                      final=False, error="")
            threading.Thread(target=self._start, daemon=True).start()
        else:
            self.recording = False
            self._set(busy=True, status="finishing up…")
            threading.Thread(target=self._stop, daemon=True).start()
        return {"recording": self.recording}

    # ---- internals (background threads; only touch _state) ----
    def _start(self):
        self.tr = LiveTranscriber(
            on_partial=self._on_partial,
            on_display=lambda t: self._set(transcript=t, final=False),
            on_final=self._on_final,
            on_status=lambda t: self._set(status=t),
            model_dir=WHISPER_DIR,
        )
        try:
            self.tr.start()
            self.encoder.ensure_loaded(status=lambda t: self._set(status=t))
            self._committed = ""
            self._last_encoded = ""
            self._last_change = time.monotonic()
            self._set(busy=False, status="listening…")
            self._enc_worker = threading.Thread(target=self._encode_loop, daemon=True)
            self._enc_worker.start()
        except Exception as e:
            self.recording = False
            self._set(busy=False, status="error", error=f"start error: {e}")

    def _on_partial(self, text):
        # committed text drives BERT timing + the "solid" part of the displayed text
        with self._lock:
            if text != self._committed:
                self._committed = text
                self._last_change = time.monotonic()
            self._state["committed"] = text

    def _encode_loop(self):
        # encode the latest committed text as soon as it changes; coalesces so it never backs up
        while self.recording:
            time.sleep(0.1)
            with self._lock:
                committed = self._committed
            if committed and committed != self._last_encoded:
                self._encode(committed)

    def _encode(self, text):
        try:
            emb = self.encoder.encode(text)
        except Exception as e:
            self._set(error=f"bert error: {e}")
            return
        self._last_encoded = text
        self._encode_count += 1
        self._set(encode_count=self._encode_count, encoded_text=text,
                  emb_norm=round(float(np.linalg.norm(emb)), 2))

    def _stop(self):
        try:
            if self.tr:
                self.tr.stop()
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
    webview.create_window(
        "TACET — Turn-Aware Completion",
        str(INDEX),
        js_api=api,
        width=920,
        height=660,
        min_size=(760, 560),
        background_color="#0e0f13",
    )
    webview.start()


if __name__ == "__main__":
    main()
