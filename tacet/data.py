# data.py
# Loads the embedding files and sanity-checks their shape before
# we train on them. Remember the head never sees text, only these 768-d vectors
# and their labels. The .npz contract :
#     embeddings : float32, shape [N, 768]   (frozen-BERT [CLS] vectors)
#     labels     : int,     shape [N]         (1 = complete, 0 = incomplete)
# Where it fits: small loader shared by train.py and evaluate.py.
import numpy as np
import torch


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = torch.as_tensor(d["embeddings"], dtype=torch.float32)
    y = torch.as_tensor(d["labels"], dtype=torch.float32)
    assert X.ndim == 2 and X.shape[1] == 768, f"bad embedding shape {tuple(X.shape)}"
    assert X.shape[0] == y.shape[0], "embeddings/labels length mismatch"
    return X, y
