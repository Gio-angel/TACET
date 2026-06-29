# gate.py
<<<<<<< Updated upstream
# The decision gate: stage 5 of the pipeline. The head hands us a probability that
# the turn is complete; this is where we turn that number into an actual choice —
# respond now, or keep listening. Right now it's deliberately dumb: a fixed 50%
# threshold. We'll enhance it later (e.g. an adaptive threshold, or mixing in how
# long the user has been silent), but the function signature can stay the same.
# Where it fits: between the head's probability and the LLM/TTS response.
=======
# decision gate. Consumes the probability from inference (tacet/infer.py)
# and turns it into a choice: respond now, or keep listening.
>>>>>>> Stashed changes
import config as C
from voice_mngt.spectrogram import spectrogram


<<<<<<< Updated upstream
def should_respond(prob: float, tau: float = None):
=======
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
>>>>>>> Stashed changes
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
