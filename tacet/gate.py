# gate.py
#decision gate. Consumes the probability from inference (tacet/infer.py)
# and turns it into a choice: respond now, or keep listening. Fixed threshold for
# now; later it can fold in silence duration. Kept separate from inference on purpose.
import config as C


def should_respond(prob: float, tau: float = None) -> bool:
    if tau is None:
        tau = C.TAU
    return prob >= tau
