# asr.py
# live speech -> text.
# Captures the microphone with sounddevice and transcribes with faster-whisper.

import threading
import time
from pathlib import Path

import numpy as np

import config as C
from tacet.preprocess import normalize

SAMPLE_RATE = 16000          # what Whisper expects
PARTIAL_EVERY = 0.35         # seconds between partial transcriptions
MIN_SECONDS = 0.3            # ignore buffers shorter than this


def _common_prefix(a, b):
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


class LiveTranscriber:
    """Microphone -> faster-whisper. Calls callbacks with normalized text.

    on_partial(text)  : growing transcript while the user speaks
    on_final(text)    : the transcript when recording stops
    on_status(text)   : human-readable status messages (model loading, errors)
    """

    def __init__(self, on_partial=None, on_final=None, on_status=None, on_display=None,
                 model_size="base.en", model_dir=None, device="cpu",
                 compute_type="int8", partial_every=PARTIAL_EVERY):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_status = on_status
        self.on_display = on_display
        self.model_size = model_size
        self.model_dir = model_dir
        self.device = device
        self.compute_type = compute_type
        self.partial_every = partial_every

        self._model = None
        self._stream = None
        self._buf = []
        self._lock = threading.Lock()
        self._running = False
        self._worker = None
        self._level = 0.0           # live mic loudness (RMS), for the UI orb
        self._sample_count = 0
        self._last_voice_sample = 0
        self._last_voice_at = 0.0
        self._prev_hyp = []         # previous transcription (for LocalAgreement)
        self._committed = []        # words confirmed stable across 2 runs

    def level(self):
        return self._level

    def reset(self):
        # clear audio + committed text so the next turn starts from zero
        with self._lock:
            self._buf = []
            self._sample_count = 0
            self._last_voice_sample = 0
            self._last_voice_at = 0.0
        self._prev_hyp = []
        self._committed = []
        self._level = 0.0

    # ---- helpers ----
    def _status(self, msg):
        if self.on_status:
            self.on_status(msg)

    def _load_model(self):
        from faster_whisper import WhisperModel
        src = self.model_size
        if self.model_dir and Path(self.model_dir).exists():
            src = str(self.model_dir)            # packaged / offline path
        self._status(f"loading whisper ({src})...")
        self._model = WhisperModel(src, device=self.device,
                                   compute_type=self.compute_type)
        self._status("model ready")

    def _transcribe(self, audio, final=False):
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=5 if final else 1,
            condition_on_previous_text=False,
            vad_filter=True,                     # drop non-speech -> no phantom words
        )
        return " ".join(seg.text for seg in segments).strip()

    def _current_audio(self):
        with self._lock:
            if not self._buf:
                return None
            return np.concatenate(self._buf)

    def audio_before_last_speech(self, seconds=0.3):
        """Copy the audio window ending at the latest voiced mic block."""

        item = self.audio_window_before_last_speech(seconds)
        return item[0] if item is not None else None

    def audio_window_before_last_speech(self, seconds=0.3):
        """Return a recent voiced window and its end-sample freshness token."""

        sample_count = int(SAMPLE_RATE * seconds)
        if sample_count <= 0:
            return None

        with self._lock:
            if not self._buf or self._last_voice_sample == 0:
                return None
            end = min(self._last_voice_sample, self._sample_count)
            start = max(0, end - sample_count)
            pieces = []
            cursor = self._sample_count
            for block in reversed(self._buf):
                block_start = cursor - len(block)
                overlap_start = max(start, block_start)
                overlap_end = min(end, cursor)
                if overlap_start < overlap_end:
                    pieces.append(block[overlap_start - block_start:overlap_end - block_start])
                if block_start <= start:
                    break
                cursor = block_start
            if not pieces:
                return None
            audio = np.concatenate(list(reversed(pieces))).copy()
            return audio, end

    def last_voice_sample(self):
        with self._lock:
            return self._last_voice_sample

    def last_voice_time(self):
        with self._lock:
            return self._last_voice_at

    # ---- lifecycle ----
    def start(self):
        import sounddevice as sd
        if self._running:
            return
        if self._model is None:
            self._load_model()

        with self._lock:
            self._buf = []
            self._sample_count = 0
            self._last_voice_sample = 0
            self._last_voice_at = 0.0
        self._prev_hyp = []
        self._committed = []
        self._running = True

        def callback(indata, frames, time_info, status):
            block = indata[:, 0].copy()
            level = float(np.sqrt(np.mean(block ** 2)))
            with self._lock:
                self._buf.append(block)
                self._sample_count += len(block)
                if level > C.SPEC_THRESHOLD:
                    self._last_voice_sample = self._sample_count
                    self._last_voice_at = time.monotonic()
            self._level = level

        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                      dtype="float32", callback=callback)
        self._stream.start()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self):
        while self._running:
            time.sleep(self.partial_every)
            audio = self._current_audio()
            if audio is None or len(audio) < SAMPLE_RATE * MIN_SECONDS:
                continue
            if self._last_voice_sample == 0:        # nothing voiced yet -> don't transcribe silence
                continue
            try:
                raw = normalize(self._transcribe(audio))
                if raw and self.on_display:
                    self.on_display(raw)                         # fast, flickery text for UI
                words = raw.split()
                agreed = _common_prefix(self._prev_hyp, words)   # words confirmed across 2 runs
                trimmed = words[:-1]                             # all but the volatile last word
                stable = agreed if len(agreed) >= len(trimmed) else trimmed
                self._prev_hyp = words
                if len(stable) > len(self._committed):
                    self._committed = stable                     # commit faster, keep last word tentative
                committed = " ".join(self._committed)
                if committed and self.on_partial:
                    self.on_partial(committed)                   # stable text for BERT
            except Exception as e:
                self._status(f"asr error: {e}")

    def stop(self):
        if not self._running:
            return ""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=2)

        audio = self._current_audio()
        text = ""
        if audio is not None and len(audio) >= SAMPLE_RATE * MIN_SECONDS:
            text = normalize(self._transcribe(audio, final=True))
        if self.on_final:
            self.on_final(text)
        return text
