# gate.py
# Turns the TACET probability and current audio state into a turn decision.
import config as C
from voice_mngt.spectrogram import spectrogram


def _threshold_seconds(value: float) -> float:
    """Accept threshold config in seconds or milliseconds."""

    return value / 1000 if value > 10 else value


def should_respond(
    prob: float,
    tau: float = None,
    audio_level: float = None,
    silence_duration: float = None,
    min_threshold: float = None,
    max_threshold: float = None,
    required_silence: float = None,
) -> bool:
    if tau is None:
        tau = C.TAU
    if min_threshold is None:
        min_threshold = C.MIN_THRESHOLD
    if max_threshold is None:
        max_threshold = C.MAX_THRESHOLD

    if audio_level is not None and spectrogram(audio_level):
        return False

    if silence_duration is None:
        return prob >= tau

    if required_silence is not None:
        required_silence = _threshold_seconds(required_silence)
        return silence_duration > required_silence

    min_threshold = _threshold_seconds(min_threshold)
    max_threshold = _threshold_seconds(max_threshold)

    if silence_duration < min_threshold:
        return False

    if silence_duration >= max_threshold:
        return True

    return prob >= tau
