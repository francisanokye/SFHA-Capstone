"""
Inference: load trained vectorizer and model (best by F1 or specified), predict on new text.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
import joblib
import numpy as np

from config import (
    MODEL_DIR,
    TOPIC_CODES,
    TOPIC_LABELS,
    BEST_MODEL_NAME_FILE,
    MAX_FEATURES,
    saved_models_exist,
)
from preprocess import preprocess_text


def load_metadata(model_dir=None):
    """Load topic_codes and topic_labels from models/metadata.json if present; else config defaults."""
    model_dir = model_dir or MODEL_DIR
    path = os.path.join(model_dir, "metadata.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            m = json.load(f)
        return m["topic_codes"], m["topic_labels"]
    return TOPIC_CODES, TOPIC_LABELS

def get_best_model_name(model_dir=None):
    """Read the saved best model name (from runner.py --save-model)."""
    model_dir = model_dir or MODEL_DIR
    path = os.path.join(model_dir, BEST_MODEL_NAME_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return f.read().strip()


def load_pipeline(model_name=None, model_dir=None):
    """
    Load vectorizer and model from disk.
    If model_name is None, uses the best model (by F1) saved by runner.py.

    Parameters
    ----------
    model_name : str, optional
        One of 'lgreg', 'gradient_boosting', 'random_forest'.
        If None, loads the best model name from models/best_model_name.txt.
    model_dir : str, optional
        Directory containing .joblib and optional .keras files. Defaults to config.MODEL_DIR.

    Returns
    -------
    vectorizer : TfidfVectorizer
    model : classifier (sklearn-like .predict(X) or LSTM wrapper)
    topic_codes : dict (for label lookup)
    topic_labels : list (display names)
    """
    model_dir = model_dir or MODEL_DIR
    topic_codes, topic_labels = load_metadata(model_dir)
    if model_name is None:
        model_name = get_best_model_name(model_dir)
        if model_name is None:
            raise FileNotFoundError(
                f"No {BEST_MODEL_NAME_FILE} in {model_dir}. Run runner.py --save-model first."
            )
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(
            f"Vectorizer not found in {model_dir}. Run runner.py --save-model first."
        )
    vectorizer = joblib.load(vec_path)

    model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model {model_name} not found at {model_path}. Run runner.py --save-model first."
        )
    model = joblib.load(model_path)

    return vectorizer, model, topic_codes, topic_labels


def predict_text(text, vectorizer, model, topic_codes=None, topic_labels=None, preprocess=True):
    """
    Predict topic code and label for a single raw text.

    Parameters
    ----------
    text : str
    vectorizer : TfidfVectorizer (fitted)
    model : classifier (fitted, sklearn-like or LSTM wrapper)
    preprocess : bool
        If True, run preprocess_text before vectorizing.

    Returns
    -------
    topic_code : int
    label : str
    """
    if preprocess:
        text = preprocess_text(text)
    X = vectorizer.transform([text])
    code = model.predict(X)[0]
    codes = topic_codes if topic_codes is not None else TOPIC_CODES
    labels = topic_labels if topic_labels is not None else TOPIC_LABELS
    inv = {v: k for k, v in codes.items()}
    label = inv.get(code, labels[code] if 0 <= code < len(labels) else "Unknown")
    if isinstance(label, str) and label.startswith("r/"):
        label = label[2:]
    return int(code), label


def predict_batch(texts, vectorizer, model, topic_codes=None, topic_labels=None, preprocess=True):
    """Predict for a list of texts. Returns list of (topic_code, label)."""
    if preprocess:
        from preprocess import preprocess_dataframe
        import pandas as pd
        df = pd.DataFrame({"content": texts})
        preprocess_dataframe(df, inplace=True)
        texts = df["content"].tolist()
    X = vectorizer.transform(texts)
    pred_codes = model.predict(X)
    codes = topic_codes or TOPIC_CODES
    labels = topic_labels or TOPIC_LABELS
    inv = {v: k for k, v in codes.items()}
    results = []
    for code in pred_codes:
        label = inv.get(code, labels[code] if 0 <= code < len(labels) else "Unknown")
        if isinstance(label, str) and label.startswith("r/"):
            label = label[2:]
        results.append((int(code), label))
    return results


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=None,
        choices=["lgreg", "gradient_boosting", "random_forest"],
        help="Model to use (default: best model by F1 from training)",
    )
    ap.add_argument("--text", type=str, help="Single text to classify")
    ap.add_argument(
        "--no-train",
        action="store_true",
        help="Do not run runner if models missing; raise error instead",
    )
    args = ap.parse_args()
    if args.text:
        if not saved_models_exist() and not args.no_train:
            print("No saved models found. Running training pipeline (runner.py --save-model --no-cv)...")
            root = os.path.dirname(os.path.abspath(__file__))
            runner_script = os.path.join(root, "runner.py")
            rc = subprocess.run(
                [sys.executable, runner_script, "--save-model", "--no-cv"],
                cwd=root,
            )
            if rc.returncode != 0:
                print("Training failed. Run 'python runner.py --save-model' manually.")
                sys.exit(rc.returncode)
        vec, model, topic_codes, topic_labels = load_pipeline(args.model)
        code, label = predict_text(
            args.text, vec, model,
            topic_codes=topic_codes, topic_labels=topic_labels,
        )
        print(f"Topic code: {code}, Label: {label}")
    else:
        print("Usage: python predict.py [--model MODEL] --text 'your post content here'")
        print("  Omit --model to use the best model (by F1) saved during training.")
        print("  If no models exist, the training pipeline is run first unless --no-train.")
