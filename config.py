# config.py
# shared variables and settings
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"           # embeddings
ARTIFACTS = ROOT / "artifacts"     # trained head lives here
TRAIN_NPZ = DATA_DIR / "train.npz"
VAL_NPZ = DATA_DIR / "val.npz"
TEST_NPZ = DATA_DIR / "test.npz"
HEAD_PATH = ARTIFACTS / "tacet_head.pt"

# --- model ---
EMB_DIM = 768      # frozen bert-base-uncased [CLS] size
HIDDEN = 64
DROPOUT = 0.3

# --- training ---
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10      # early-stopping patience on validation loss
SEED = 42

# --- inference ---
TAU = 0.5          # decision threshold: P(complete) >= TAU -> take the turn
