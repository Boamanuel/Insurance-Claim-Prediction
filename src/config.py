"""
Central configuration for the insurance claim prediction pipeline.
Keeping these values in one place means no magic numbers scattered
across scripts, and a single spot to tune for retraining.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

RAW_DATA_PATH = DATA_DIR / "train.csv"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Data / feature settings
# ---------------------------------------------------------------------------
TARGET_COLUMN = "target"
ID_COLUMN = "id"

# Low-correlation calc features identified during EDA (see notebooks/) —
# dropped to cut dimensionality without losing predictive signal.
LOW_CORRELATION_COLUMNS = [
    "ps_calc_04", "ps_calc_06", "ps_calc_07", "ps_calc_09",
    "ps_calc_11", "ps_calc_13", "ps_calc_15_bin", "ps_calc_16_bin",
    "ps_calc_17_bin", "ps_calc_18_bin",
]

TEST_SIZE = 0.25
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# The dataset is heavily imbalanced (target=1 is the minority class), so
# accuracy alone is misleading. The pipeline selects the best model by
# ROC-AUC, with F1 on the minority class logged alongside it.
PRIMARY_METRIC = "roc_auc"

KNN_N_NEIGHBORS = 4
DECISION_TREE_PARAMS = dict(
    criterion="entropy",
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=3,
    splitter="random",
    random_state=RANDOM_STATE,
)

LOG_LEVEL = "INFO"
LOG_FILE_NAME = "pipeline.log"
