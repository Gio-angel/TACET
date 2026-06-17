# config.py
# One place for every setting in the project: file paths, the size of the head,
# the training knobs, and the decision threshold. Everything else imports from
# here, so when we want to tweak something we change it once, here, and every
# script picks it up. embeddings : .npz files into data/.
# Where it fits: shared config, used by all the scripts.
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
