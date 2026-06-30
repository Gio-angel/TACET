import threading
import time

import numpy as np

import config as C


PITCH_WINDOW_SECONDS = 0.2
PITCH_UPDATE_SECONDS = 0.05
PITCH_DROP_RATIO = 0.12
PITCH_MAX_AGE_SECONDS = 0.75
PITCH_FMIN = 65.0
PITCH_FMAX = 400.0
PITCH_FRAME_LENGTH = 1024
PITCH_HOP_LENGTH = 160


def track_pitch(
    audio,
    sample_rate=16000,
    fmin=PITCH_FMIN,
    fmax=PITCH_FMAX,
):
    """Return YIN pitch features for a mono audio window, or None."""

    if audio is None:
        return None

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < PITCH_FRAME_LENGTH:
        return None

    import librosa

    contour = librosa.yin(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=PITCH_FRAME_LENGTH,
        hop_length=PITCH_HOP_LENGTH,
        center=False,
    )
    frame_rms = librosa.feature.rms(
        y=samples,
        frame_length=PITCH_FRAME_LENGTH,
        hop_length=PITCH_HOP_LENGTH,
        center=False,
    )[0]

    frame_count = min(len(contour), len(frame_rms))
    contour = np.asarray(contour[:frame_count], dtype=np.float32)
    voiced = np.isfinite(contour) & (frame_rms[:frame_count] > C.SPEC_THRESHOLD)
    voiced_pitch = contour[voiced]
    if voiced_pitch.size == 0:
        return None

    median = float(np.median(voiced_pitch))
    final = float(np.median(voiced_pitch[-3:]))
    drop_ratio = max(0.0, (median - final) / median) if median > 0 else 0.0
    return {
        "contour": voiced_pitch.astype(float).tolist(),
        "median": median,
        "final": final,
        "drop_ratio": drop_ratio,
        "falling": drop_ratio >= PITCH_DROP_RATIO,
    }


def warm_up_pitch():
    """Pay librosa/Numba startup cost before live microphone processing."""

    sample_count = int(16000 * PITCH_WINDOW_SECONDS)
    t = np.arange(sample_count, dtype=np.float32) / 16000.0
    track_pitch(0.1 * np.sin(2 * np.pi * 160.0 * t))


class RollingPitchTracker:
    """Analyze only the newest voiced audio window on a background thread.

    ``audio_source`` must return ``(audio, voice_sample_token)``. The token lets
    consumers reject a result as soon as newer voiced audio has arrived.
    """

    def __init__(
        self,
        audio_source,
        token_source=None,
        update_seconds=PITCH_UPDATE_SECONDS,
        max_age_seconds=PITCH_MAX_AGE_SECONDS,
    ):
        self.audio_source = audio_source
        self.token_source = token_source
        self.update_seconds = update_seconds
        self.max_age_seconds = max_age_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = None
        self._latest = None
        self._last_token = None
        self._generation = 0
        self.call_count = 0

    def start(self, warm_up=True):
        if self._worker is not None and self._worker.is_alive():
            return
        if warm_up:
            warm_up_pitch()
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1)
        self._worker = None

    def reset(self):
        with self._lock:
            self._generation += 1
            self._latest = None
            self._last_token = None

    def snapshot(self, current_token=None):
        with self._lock:
            result = dict(self._latest) if self._latest is not None else None
        if result is None:
            return None
        age = time.monotonic() - result["completed_at"]
        result["age_seconds"] = age
        result["fresh"] = (
            age <= self.max_age_seconds
            and (current_token is None or result["voice_sample"] == current_token)
        )
        return result

    def _loop(self):
        while not self._stop.wait(self.update_seconds):
            try:
                if self.token_source is not None:
                    current_token = self.token_source()
                    with self._lock:
                        if not current_token or current_token == self._last_token:
                            continue
                item = self.audio_source()
                if item is None:
                    continue
                audio, token = item
                with self._lock:
                    if token == self._last_token:
                        continue
                    self._last_token = token
                    generation = self._generation

                started = time.monotonic()
                pitch = track_pitch(audio)
                latency_ms = (time.monotonic() - started) * 1000
                result = {
                    "voice_sample": token,
                    "completed_at": time.monotonic(),
                    "latency_ms": latency_ms,
                    "error": "",
                }
                if pitch is not None:
                    result.update(pitch)

                with self._lock:
                    if generation != self._generation:
                        continue
                    self.call_count += 1
                    self._latest = result
            except Exception as exc:
                with self._lock:
                    self.call_count += 1
                    self._latest = {
                        "voice_sample": self._last_token,
                        "completed_at": time.monotonic(),
                        "latency_ms": None,
                        "error": str(exc),
                    }
