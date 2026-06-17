# infer.py
# Wraps the trained head so the rest of the system can ask one simple question:
# "is the user done talking?" You give it an embedding, it gives back the
# probability (from the head) and the yes/no decision (from the gate). It knows
# nothing about audio or BERT on purpose, so we can use and test it without that
# heavy stack installed.
# Where it fits: the inference side of stage 4 + 5, used by app/demo.py.
import numpy as np

import config as C
from tacet.model import TACETHead
from tacet.gate import should_respond


class TacetEndpointer:
    def __init__(self, head_path=None, tau: float = None):
        self.model = TACETHead.load(head_path or C.HEAD_PATH)
        self.tau = C.TAU if tau is None else tau

    def probability(self, emb: np.ndarray) -> float:
        """emb: [768] -> P(complete)."""
        return float(self.model.score(emb)[0])

    def take_turn(self, emb: np.ndarray) -> bool:
        """True  -> user finished, respond.
        False -> still mid-utterance, keep listening."""
        return should_respond(self.probability(emb), self.tau)
