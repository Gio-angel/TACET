# gate.py
# The decision gate: stage 5 of the pipeline. The head hands us a probability that
# the turn is complete; this is where we turn that number into an actual choice —
# respond now, or keep listening. Right now it's deliberately dumb: a fixed 50%
# threshold. We'll enhance it later (e.g. an adaptive threshold, or mixing in how
# long the user has been silent), but the function signature can stay the same.
# Where it fits: between the head's probability and the LLM/TTS response.
import config as C


def should_respond(prob: float, tau: float = None):
    if tau is None:
        tau = C.TAU
    return prob >= tau
