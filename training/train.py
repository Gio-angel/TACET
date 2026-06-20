import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from tacet.data import load
from tacet.model import TACETHead

BATCH_SIZE = 32
BATCH_SIZE_FN = 64
TRAIN_NPZ = ROOT / "data" / "TRAINING_EMBEDDINGS" / "train.npz"
FINETUNE_NPZ = ROOT / "data" / "FINE_TUNE_EMBEDDINGS" / "fine_tuning_20k.npz"

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def _binary_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def validate(
    model: TACETHead,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tau: float = C.TAU,
) -> dict:
    """Run validation and return loss, metrics, and raw predictions."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        probs = model(X_batch)
        loss = criterion(probs, y_batch)

        if torch.isnan(loss):
            raise RuntimeError("Validation loss is NaN. Error in data or learning rate.")

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs.cpu())
        all_labels.append(y_batch.cpu())

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = (probs >= tau).long()
    metrics = _binary_metrics(labels.long(), preds)
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["probs"] = probs
    metrics["labels"] = labels
    metrics["preds"] = preds
    return metrics


def train_model() -> dict | None:
    print("=" * 60)
    print("TACET training (80/20 Dynamic Split)")
    print("=" * 60)

    # 1. Verify and load from the single data file
    if not TRAIN_NPZ.exists():
        print(f"[ERROR] Training file not found: {TRAIN_NPZ}")
        return None

    try:
        X_all, y_all = load(TRAIN_NPZ)
    except Exception as exc:
        print(f"[ERROR] Failed to load embeddings: {exc}")
        return None

    total_samples = len(X_all)
    print(f"Total dataset samples: {total_samples}")

    if total_samples == 0:
        print("[ERROR] Dataset is empty — nothing to train on.")
        return None

    # 2. Enforce reproducibility and perform the 80/20 split
    _set_seed(C.SEED)
    
    shuffled_indices = torch.randperm(total_samples)
    train_size = int(0.8 * total_samples)
    
    train_idx = shuffled_indices[:train_size]
    val_idx = shuffled_indices[train_size:]
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    n_train, n_val = len(X_train), len(X_val)
    print(f"Training samples (80%):   {n_train}")
    print(f"Validation samples (20%): {n_val}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Epochs (max):       {C.EPOCHS}")
    print(f"Learning rate:      {C.LR}")
    print(f"Early-stop patience:{C.PATIENCE}")

    if n_train == 0:
        print("[ERROR] Training split is empty — nothing to train on.")
        return None
    if n_val == 0:
        print("[ERROR] Validation split is empty — cannot monitor generalization.")
        return None

    if BATCH_SIZE > n_train:
        print(
            f"[WARN] BATCH_SIZE ({BATCH_SIZE}) > training samples ({n_train}); "
            f"using batch size {n_train}."
        )
        batch_size = n_train
    else:
        batch_size = BATCH_SIZE

    # 3. Setup device and DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:             {device}")

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, min(batch_size, n_val), shuffle=False)

    # 4. Initialize Network, Loss, and Optimizer
    model = TACETHead(emb_dim=C.EMB_DIM, hidden=C.HIDDEN, dropout=C.DROPOUT).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=C.LR)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("-" * 60)
    print("Starting training loop...")
    print("-" * 60)

    # 5. Training Loop
    try:
        for epoch in range(1, C.EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                probs = model(X_batch)
                loss = criterion(probs, y_batch)

                if torch.isnan(loss):
                    print(f"[ERROR] NaN training loss at epoch {epoch} — stopping.")
                    return None

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            val_result = validate(model, val_loader, criterion, device)
            avg_val_loss = val_result["loss"]

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(
                f"Epoch {epoch:3d}/{C.EPOCHS} | "
                f"train loss: {avg_train_loss:.4f} | "
                f"val loss: {avg_val_loss:.4f} | "
                f"val acc: {val_result['accuracy']:.4f} | "
                f"val F1: {val_result['f1']:.4f}"
            )

            # Check for improvement (Early Stopping track)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= C.PATIENCE:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no val-loss improvement for {C.PATIENCE} epochs)."
                    )
                    break

    except RuntimeError as exc:
        print(f"[ERROR] Training failed: {exc}")
        return None
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user.")
        if best_state is None:
            return None

    if best_state is None:
        print("[ERROR] No model checkpoint was saved — training produced no usable weights.")
        return None

    # 6. Load the best weights and final validation run
    model.load_state_dict(best_state)
    model.eval()

    try:
        model.save(C.HEAD_PATH)
        print(f"Best model saved to: {C.HEAD_PATH}")
    except OSError as exc:
        print(f"[ERROR] Could not save model to {C.HEAD_PATH}: {exc}")
        return None

    final_val = validate(model, val_loader, criterion, device)
    print("-" * 60)
    print("Training complete.")
    print(
        f"Best val loss: {best_val_loss:.4f} | "
        f"Final val acc: {final_val['accuracy']:.4f} | "
        f"precision: {final_val['precision']:.4f} | "
        f"recall: {final_val['recall']:.4f} | "
        f"F1: {final_val['f1']:.4f}"
    )
    print("=" * 60)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_val": final_val,
        "device": device,
    }
    
    
    
    
    
    

def train_model_finetune() -> dict | None:
    print("=" * 60)
    print(f"TACET 20k Fine-Tuning Run (80/20 Dynamic Split)")
    print("=" * 60)

    # 1. Verify the 20k dataset file exists
    if not FINETUNE_NPZ.exists():
        print(f"[ERROR] Fine-tuning file not found: {FINETUNE_NPZ}")
        return None

    try:
        X_all, y_all = load(FINETUNE_NPZ)
    except Exception as exc:
        print(f"[ERROR] Failed to load embeddings: {exc}")
        return None

    total_samples = len(X_all)
    print(f"Total fine-tuning dataset samples: {total_samples}")

    if total_samples == 0:
        print("[ERROR] Fine-tuning dataset is empty.")
        return None

    # 2. Enforce reproducibility and perform the 80/20 split
    _set_seed(C.SEED)
    
    shuffled_indices = torch.randperm(total_samples)
    train_size = int(0.8 * total_samples)
    
    train_idx = shuffled_indices[:train_size]
    val_idx = shuffled_indices[train_size:]
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    n_train, n_val = len(X_train), len(X_val)
    print(f"Training samples (80%):   {n_train}")
    print(f"Validation samples (20%): {n_val}")
    print(f"Batch size:         {C.FN_BATCH_SIZE}")
    print(f"Epochs (max):       {C.FN_EPOCHS}")
    print(f"Learning rate:      {C.LR}")
    print(f"Early-stop patience:{C.PATIENCE}")

    if n_train == 0:
        print("[ERROR] Training split is empty — nothing to train on.")
        return None
    if n_val == 0:
        print("[ERROR] Validation split is empty — cannot monitor generalization.")
        return None

    if C.FN_BATCH_SIZE > n_train:
        print(
            f"[WARN] BATCH_SIZE ({C.FN_BATCH_SIZE}) > training samples ({n_train}); "
            f"using batch size {n_train}."
        )
        batch_size = n_train
    else:
        batch_size = C.FN_BATCH_SIZE

    finetune_artifacts_dir = C.ARTIFACTS
    finetune_artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = finetune_artifacts_dir / "finetune_tacet_head.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:                    {device}")

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, min(batch_size, n_val), shuffle=False)

    # 4. Initialize Network, Loss Engine, and Optimizer
    model = TACETHead(emb_dim=C.EMB_DIM, hidden=C.HIDDEN, dropout=C.DROPOUT).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=C.LR)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("-" * 60)
    print("Starting fine-tuning loop...")
    print("-" * 60)

    # 5. Fine-Tuning Execution Engine Loop
    try:
        for epoch in range(1, C.FN_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                probs = model(X_batch)
                loss = criterion(probs, y_batch)

                if torch.isnan(loss):
                    print(f"[ERROR] NaN training loss at epoch {epoch} — stopping.")
                    return None

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            val_result = validate(model, val_loader, criterion, device)
            avg_val_loss = val_result["loss"]

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(
                f"Epoch {epoch:3d}/{C.FN_EPOCHS} | "
                f"train loss: {avg_train_loss:.4f} | "
                f"val loss: {avg_val_loss:.4f} | "
                f"val acc: {val_result['accuracy']:.4f} | "
                f"val F1: {val_result['f1']:.4f}"
            )

            # Early Stopping Monitoring Track
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= C.PATIENCE:
                    print(f"Early stopping at epoch {epoch} (no val-loss improvement).")
                    break

    except RuntimeError as exc:
        print(f"[ERROR] Fine-tuning failed: {exc}")
        return None
    except KeyboardInterrupt:
        print("\n[WARN] Fine-tuning interrupted by user.")
        if best_state is None:
            return None

    if best_state is None:
        print("[ERROR] No model checkpoint was saved — fine-tuning produced no usable weights.")
        return None

    # 6. Reload Best Weights State Layer
    model.load_state_dict(best_state)
    model.eval()

    try:
        model.save(model_save_path)
        print(f"Best fine-tuned model saved to: {model_save_path}")
    except OSError as exc:
        print(f"[ERROR] Could not save model to {model_save_path}: {exc}")
        return None

    final_val = validate(model, val_loader, criterion, device)
    print("-" * 60)
    print("Fine-tuning complete.")
    print(
        f"Best val loss: {best_val_loss:.4f} | "
        f"Final val acc: {final_val['accuracy']:.4f} | "
        f"precision: {final_val['precision']:.4f} | "
        f"recall: {final_val['recall']:.4f} | "
        f"F1: {final_val['f1']:.4f}"
    )
    print("=" * 60)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_val": final_val,
        "device": device,
    }