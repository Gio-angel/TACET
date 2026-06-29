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
HEAD_PATH = ARTIFACTS / "finetune_tacet_head.pt"

# --- model ---
EMB_DIM = 768      # frozen bert-base-uncased [CLS] size
HIDDEN = 64
DROPOUT = 0.3

# --- training ---
LR = 1e-3
EPOCHS = 100
FN_EPOCHS = 50
FN_BATCH_SIZE = 64
PATIENCE = 10      # early-stopping patience on validation loss
SEED = 42

# --- inference ---
TAU = 0.58         # decision threshold: P(complete) >= TAU -> take the turn

SPEC_THRESHOLD = 0.02  # spectrogram/RMS gate: level > threshold -> voice present
MIN_THRESHOLD = 500
MAX_THRESHOLD = 1000

# --- ASR ---
WHISPER_MODEL_SIZE = "tiny.en"  # faster fallback than base.en when no local model exists
