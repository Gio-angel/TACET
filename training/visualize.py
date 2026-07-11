import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from tacet.gate import should_respond

PLOTS_DIR = C.ARTIFACTS / "plots"


def _binary_metrics_np(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_path: Path):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="Validation loss", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decision_metrics(probs: torch.Tensor, labels: torch.Tensor, out_path: Path):
    #Decision plot: sweep tau (gate threshold) and plot Accuracy, Precision, Recall, F1.
    probs_np = probs.numpy()
    labels_np = labels.numpy().astype(int)
    taus = np.linspace(0.0, 1.0, 101)

    accuracies, precisions, recalls, f1s = [], [], [], []
    for tau in taus:
        preds = np.array([1 if should_respond(float(p), tau=tau) else 0 for p in probs_np])
        m = _binary_metrics_np(labels_np, preds)
        accuracies.append(m["accuracy"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(taus, accuracies, label="Accuracy")
    ax.plot(taus, precisions, label="Precision")
    ax.plot(taus, recalls, label="Recall")
    ax.plot(taus, f1s, label="F1-Score")
    ax.axvline(C.TAU, color="gray", linestyle="--", linewidth=1, label=f"τ = {C.TAU}")
    ax.set_xlabel("τ (decision threshold)")
    ax.set_ylabel("Score")
    ax.set_title("Decision Plot — Metrics vs Gate Threshold τ")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(labels: torch.Tensor, preds: torch.Tensor, out_path: Path):
    #2×2 grid: correct vs incorrect predictions at the final epoch.
    y_true = labels.numpy().astype(int)
    y_pred = preds.numpy().astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Incomplete (0)", "Pred Complete (1)"])
    ax.set_yticklabels(["True Incomplete (0)", "True Complete (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Final Epoch Confusion Matrix")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    correct = tn + tp
    total = cm.sum()
    ax.text(
        0.5, -0.22,
        f"Correct: {correct}/{total} ({100 * correct / total:.1f}%)",
        ha="center", transform=ax.transAxes, fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probability_histogram(probs: torch.Tensor, labels: torch.Tensor, out_path: Path):
    #Overlapping histograms of P(complete) for incomplete vs complete samples.
    probs_np = probs.numpy()
    labels_np = labels.numpy().astype(int)

    incomplete = probs_np[labels_np == 0]
    complete = probs_np[labels_np == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(
        incomplete, bins=bins, alpha=0.6, label=f"Incomplete (n={len(incomplete)})",
        color="tab:orange", edgecolor="white",
    )
    ax.hist(
        complete, bins=bins, alpha=0.6, label=f"Complete (n={len(complete)})",
        color="tab:blue", edgecolor="white",
    )
    ax.axvline(C.TAU, color="gray", linestyle="--", linewidth=1, label=f"τ = {C.TAU}")
    ax.set_xlabel("Model output probability P(complete)")
    ax.set_ylabel("Count of samples")
    ax.set_title("Distribution of Probability Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Update the function declaration inside training/visualize.py:

def visualize_results(train_result: dict, target_plots_dir: Path = PLOTS_DIR):
    """Generate all training diagnostic plots from a completed execution result."""
    if train_result is None:
        print("[ERROR] Cannot visualize — training result is None.")
        return False

    final_val = train_result.get("final_val")
    train_losses = train_result.get("train_losses")
    val_losses = train_result.get("val_losses")

    if not final_val or not train_losses or not val_losses:
        print("[ERROR] Cannot visualize — training result is missing required fields.")
        return False

    try:
        # Create whatever directory structure is passed in
        target_plots_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Could not create plots directory {target_plots_dir}: {exc}")
        return False

    probs = final_val["probs"]
    labels = final_val["labels"]
    preds = final_val["preds"]

    print("-" * 60)
    print(f"Generating training visualizations in: {target_plots_dir}")

    # Map the plots out using the dynamic base path variable
    plots = {
        "loss_curve.png": lambda p: plot_loss_curve(train_losses, val_losses, p),
        "decision_plot.png": lambda p: plot_decision_metrics(probs, labels, p),
        "confusion_matrix.png": lambda p: plot_confusion_matrix(labels, preds, p),
        "probability_histogram.png": lambda p: plot_probability_histogram(probs, labels, p),
    }

    saved = []
    for filename, plot_fn in plots.items():
        out_path = target_plots_dir / filename
        try:
            plot_fn(out_path)
            saved.append(out_path)
            print(f"   Saved: {out_path}")
        except Exception as exc:
            print(f"[ERROR] Failed to create {filename}: {exc}")

    if not saved:
        print("[ERROR] No plots were generated.")
        return False

    print(f"Visualization complete ({len(saved)}/{len(plots)} plots).")
    print("=" * 60)
    return True