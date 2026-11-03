"""All settings for the project live here. Change values here, not inside the scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- Folders ----------
RAW_DIR = ROOT / "data" / "raw"              # put the downloaded dataset here (any folder layout)
PROCESSED_DIR = ROOT / "data" / "processed"  # resized copies made by prepare_data.py
SPLITS_FILE = PROCESSED_DIR / "splits.csv"   # which image is train / val / test
MY_PHOTOS_DIR = ROOT / "my_photos"           # your own photos, for the "truly unseen" test
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "currency_model.pt"
REPORTS_DIR = ROOT / "reports"               # charts and results for your project report

# ---------- Classes ----------
# Denominations the label parser knows. Old and new designs of the same value are merged
# into one class, because the user only needs to hear the value.
DENOMINATIONS = [10, 20, 50, 100, 200, 500, 2000]
BACKGROUND = "background"   # optional class: photos with no note (table, hand, paper, ...)
EXCLUDE = []                # e.g. ["2000"] to leave a class out

# ---------- Data ----------
STORE_MAX_SIDE = 512        # images are shrunk to this size once, so training is fast
IMG_SIZE = 224              # model input size
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

# ---------- Model and training ----------
ARCH = "mobilenet_v3_large"  # or "efficientnet_b0" (slower, sometimes slightly more accurate)
BATCH_SIZE = 32
HEAD_EPOCHS = 4              # stage 1: train only the new classifier layer
FINETUNE_EPOCHS = 12         # stage 2: fine-tune the whole network
HEAD_LR = 1e-3
FINETUNE_LR = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 4
NUM_WORKERS = 0 if os.name == "nt" else 2   # Windows works best with 0

# ---------- Prediction ----------
CONFIDENCE_THRESHOLD = 0.70  # below this the app says "not sure, take the photo again"
USE_TTA = True               # average predictions over 4 rotations (more robust, 4x slower)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
