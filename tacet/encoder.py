# Frozen bert-base-uncased -> 768-d [CLS] embedding. Same settings as the offline encoder.
from pathlib import Path

import numpy as np
import torch

from tacet.preprocess import normalize

MODEL_ID = "bert-base-uncased"
MAX_LEN = 64
EMB_DIM = 768


class BertEncoder:
    def __init__(self, model_dir=None, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        self._tok = None
        self._model = None

    def ensure_loaded(self, status=None):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        src = str(self.model_dir) if self.model_dir and Path(self.model_dir).exists() else MODEL_ID
        if status:
            status(f"loading bert ({src})…")
        self._tok = AutoTokenizer.from_pretrained(src)
        self._model = AutoModel.from_pretrained(src).to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text):
        self.ensure_loaded()
        enc = self._tok(normalize(text), truncation=True, max_length=MAX_LEN,
                        return_tensors="pt").to(self.device)
        cls = self._model(**enc).last_hidden_state[:, 0, :]   # [CLS]
        return cls.cpu().numpy().astype("float32")[0]
