# train.py
# Trains the TACET head on the embeddings . Loads train/val,
# runs the standard loop with Adam, keeps the best model by validation loss, stops
# early when val loss stalls, and saves the result to artifacts/tacet_head.pt.
# The head outputs probabilities, so the loss here is plain BCELoss.
# Where it fits: the offline training step. Run it from the project root:
#     python scripts/train.py
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from tacet.model import TACETHead
from tacet.data import load_npz


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_npz(C.TRAIN_NPZ)
    Xva, yva = load_npz(C.VAL_NPZ)

    model = TACETHead(C.EMB_DIM, C.HIDDEN, C.DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)   # head params only
    loss_fn = nn.BCELoss()                                # head outputs probabilities

    n = Xtr.shape[0]
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(Xva)
            val_loss = loss_fn(val_probs, yva).item()
            val_acc = (((val_probs >= 0.5).float() == yva).float().mean().item())
        print(f"epoch {epoch:3d} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= C.PATIENCE:
                print(f"early stopping at epoch {epoch} (best val_loss {best_val:.4f})")
                break

    model.load_state_dict(best_state)
    model.save(C.HEAD_PATH)
    print(f"saved -> {C.HEAD_PATH}")


if __name__ == "__main__":
    main()
