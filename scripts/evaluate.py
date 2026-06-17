# evaluate.py
# Loads the trained head and runs it on the held-out test embeddings, then prints
# the usual numbers: accuracy, precision, recall, F1, and the confusion matrix.
# This is our offline check that the head actually learned something before we
# trust it in the live demo.
# Where it fits: the check step after training. Run from the project root:
#     python scripts/evaluate.py
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from tacet.model import TACETHead
from tacet.data import load_npz


def main():
    Xte, yte = load_npz(C.TEST_NPZ)
    model = TACETHead.load(C.HEAD_PATH)

    probs = model.score(Xte.numpy())
    preds = (probs >= C.TAU).astype(int)
    y = yte.numpy().astype(int)

    acc = accuracy_score(y, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0)

    print(f"n_test   : {len(y)}")
    print(f"accuracy : {acc:.3f}")
    print(f"precision: {p:.3f}   recall: {r:.3f}   f1: {f1:.3f}")
    print("confusion matrix  rows=true [0,1], cols=pred [0,1]:")
    print(confusion_matrix(y, preds))


if __name__ == "__main__":
    main()
