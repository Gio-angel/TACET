# infer.py
# inference: run the trained TACET head on a [CLS] embedding and return
# P(complete). This module does the model forward pass ONLY and makes no decision.
# The gate (tacet/gate.py) is a separate module that consumes this probability.
import numpy as np

import config as C
from tacet.model import TACETHead


class TacetEndpointer:
    """Loads the trained head once and turns embeddings into probabilities."""

    def __init__(self, head_path=None):
        self.model = TACETHead.load(head_path or C.HEAD_PATH)

    def probability(self, emb: np.ndarray) -> float:
        return float(self.model.score(emb)[0])


def run_model_inference(embedding: np.ndarray) -> float:
    """Validate the embedding and return P(complete). No gating here."""
    if embedding is None or embedding.size == 0:
        raise ValueError("Inference input embedding cannot be None or empty.")
    if embedding.shape[-1] != C.EMB_DIM:
        raise ValueError(f"Embedding dimension mismatch. Expected {C.EMB_DIM}.")
    return TacetEndpointer().probability(embedding)
