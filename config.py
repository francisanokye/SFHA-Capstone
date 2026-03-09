"""
Configuration for Reddit mental-health classification pipeline.
"""
import os

# Data paths (relative to project root)
# All CSV data lives under the data/ folder.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Where fetch_subreddit_data.py writes a single CSV (then merged when loading data folder)
REDDIT_CORPUS_PATH = os.path.join(DATA_DIR, "reddit_corpus.csv")
# Folder of CSVs (first 4 cols: subreddit, author, date, post). Pipeline loads by merging all CSVs here.
DATA_FOLDER = DATA_DIR
# Allow override via environment
if os.environ.get("REDDIT_CORPUS_PATH"):
    REDDIT_CORPUS_PATH = os.environ["REDDIT_CORPUS_PATH"]
if os.environ.get("DATA_FOLDER"):
    DATA_FOLDER = os.environ["DATA_FOLDER"]

# Mental health subreddits used in this project
# These are the final data labels / prediction categories.
MENTAL_HEALTH_SUBREDDITS = [
    "addiction",
    "alcoholism",
    "adhd",
    "bipolarreddit",
    "depression",
    "divorce",
]

# Map subreddit names (any case, with or without r/) to canonical r/Name for pipeline
SUBREDDIT_TO_LABEL = {}
for name in MENTAL_HEALTH_SUBREDDITS:
    rname = f"r/{name}"
    SUBREDDIT_TO_LABEL[name.lower()] = rname
    SUBREDDIT_TO_LABEL[rname.lower()] = rname
    SUBREDDIT_TO_LABEL[name] = rname
    SUBREDDIT_TO_LABEL[rname] = rname

# Date filter: analyze posts before this date
REDDIT_DATE_END = "2021-10-01"

# Default topic mapping (used as a fallback; runner builds codes from data at runtime)
TOPIC_CODES = {
    "r/addiction": 0,
    "r/alcoholism": 1,
    "r/adhd": 2,
    "r/bipolarreddit": 3,
    "r/depression": 4,
    "r/divorce": 5,
}
TOPIC_LABELS = ["addiction", "alcoholism", "adhd", "bipolarreddit", "depression", "divorce"]

# TF-IDF vectorizer (min_df: float in [0,1] = fraction of docs, or int >= 1 = absolute count)
NGRAM_RANGE = (1, 2)
MIN_DF = 0.0
MAX_DF = 1.0
MAX_FEATURES = 500

# Train/test split
TEST_SIZE = 0.20
RANDOM_STATE = 9

# Cross-validation
CV_N_SPLITS = 10
CV_TEST_SIZE = 0.2
CV_RANDOM_STATE = 0

# Model hyperparameters
LOGISTIC_REGRESSION_MAX_ITER = 300

# Gradient Boosting & Random Forest
GB_N_ESTIMATORS = 100
GB_MAX_DEPTH = 5
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_NAME_FILE = "best_model_name.txt"


def saved_models_exist(model_dir=None):
    """Return True if vectorizer, metadata, best_model_name, and that model file exist."""
    d = model_dir or MODEL_DIR
    vec = os.path.join(d, "vectorizer.joblib")
    meta = os.path.join(d, "metadata.json")
    best_file = os.path.join(d, BEST_MODEL_NAME_FILE)
    if not all(os.path.isfile(p) for p in (vec, meta, best_file)):
        return False
    try:
        with open(best_file, "r") as f:
            name = f.read().strip()
    except OSError:
        return False
    if name == "lstm":
        return os.path.isfile(os.path.join(d, "lstm_model.keras"))
    return os.path.isfile(os.path.join(d, f"{name}_model.joblib"))
