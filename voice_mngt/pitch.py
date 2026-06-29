import numpy as np

import config as C


PITCH_WINDOW_SECONDS = 0.3
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

    return {
        "contour": voiced_pitch.astype(float).tolist(),
        "median": float(np.median(voiced_pitch)),
        "final": float(np.median(voiced_pitch[-3:])),
    }
