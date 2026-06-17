# model.py
# The TACET head: the little network we actually train, and the only learned part
# of the whole project. It sits on top of frozen BERT — a 768-d [CLS] vector goes
# in, and P(complete) comes out, i.e. how likely it is the user has finished their
# turn. We keep the sigmoid inside the model so forward() returns a probability
# directly; that's why training uses plain BCELoss (see scripts/train.py).
# Label convention everywhere: 1 = complete (take the turn), 0 = incomplete.
# Where it fits: stage 4 of the pipeline, the classifier head.
import numpy as np
import torch
import torch.nn as nn


class TACETHead(nn.Module):
    def __init__(self, emb_dim: int = 768, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, emb_dim] -> P(complete) [B], already in [0, 1]
        return self.net(x).squeeze(-1)

    # ---- inference: embedding -> probability ----
    @torch.no_grad()
    def score(self, emb) -> np.ndarray:
        """emb: np.ndarray [emb_dim] or [N, emb_dim] -> P(complete) in [0, 1]."""
        self.eval()
        x = torch.as_tensor(emb, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)          # [768] -> [1, 768]
        return self.forward(x).cpu().numpy()

    # ---- persistence (stores config so load() needs no extra args) ----
    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "cfg": {
                    "emb_dim": self.net[0].in_features,
                    "hidden": self.net[0].out_features,
                    "dropout": self.net[2].p,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["cfg"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
