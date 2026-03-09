"""
Main entry point: load Reddit data -> preprocess -> add topic codes -> train -> evaluate.
Optionally run EDA, save models (including best by F1), and print chi2 feature analysis.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
import os
import joblib

from config import (
    MODEL_DIR,
    OUTPUT_DIR,
    BEST_MODEL_NAME_FILE,
    DATA_FOLDER,
    TOPIC_CODES,
    TOPIC_LABELS,
    saved_models_exist,
)
from data import load_reddit_corpus
from preprocess import preprocess_dataframe
from model import add_topic_codes, train_models
from evaluate import (
    evaluate_model,
    print_chi2_top_features,
    ensure_output_dir,
    plot_multiclass_roc,
)


def _save_sklearn_model(model, path):
    joblib.dump(model, path)


def _load_sklearn_model(path):
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Reddit mental-health classification pipeline")
    parser.add_argument(
        "--data-folder",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help="Load from folder of CSVs (first 4 cols). If PATH omitted, use config.DATA_FOLDER.",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation (faster run)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save vectorizer, all models, and best model (by F1) to models/",
    )
    parser.add_argument(
        "--chi2",
        action="store_true",
        help="Print chi2 top features per class",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run EDA and save plots to output/",
    )
    args = parser.parse_args()

    ensure_output_dir()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading Reddit corpus...")
    folder = DATA_FOLDER if args.data_folder in (None, True) else args.data_folder
    df = load_reddit_corpus(data_folder=folder)
    print(f"Loaded {len(df)} rows. Subreddit counts:\n{df['Subreddit'].value_counts()}")

    if args.eda:
        from eda import run_eda
        run_eda(df)
        print("EDA plots saved to output/")

    print("Preprocessing text...")
    preprocess_dataframe(df, text_column="content", inplace=True)

    # Build topic codes/labels from data so we support any subset of mental-health subreddits
    unique_subreddits = sorted(df["Subreddit"].unique())
    topic_codes = {s: i for i, s in enumerate(unique_subreddits)}
    topic_labels = [s.replace("r/", "") for s in unique_subreddits]
    print(f"Categories ({len(topic_codes)}): {topic_labels}")

    print("Adding topic codes...")
    df = add_topic_codes(df, topic_codes=topic_codes)

    if saved_models_exist() and not args.force:
        print(f"Models already exist in {MODEL_DIR}. Skipping training. Use --force to retrain.")
        print("Done.")
        return

    print("Training models (train/test split + optional CV)...")
    result = train_models(df, run_cv=not args.no_cv, topic_codes=topic_codes)

    data = result["data"]
    ftrain = data["features_train"]
    ftest = data["features_test"]
    ltrain = data["labels_train"]
    ltest = data["labels_test"]
    tfidf = result["vectorizer"]

    if result.get("cv_results"):
        print("\n--- Cross-validation results ---")
        for name, metrics in result["cv_results"].items():
            print(f"{name}: {metrics}")
        best_name = result["best_model_name"]
        best_f1 = result["cv_results"][best_name].get("F1")
        print(f"\n--- Best model by F1 (weighted): {best_name} (F1 = {best_f1:.4f}) ---")

    # Evaluate selected models
    model_display = {
        "lgreg": "Logistic Regression (OVR)",
        "gradient_boosting": "Gradient Boosting",
        "random_forest": "Random Forest",
    }
    models_for_eval = [
        ("lgreg", result["lgreg_model"]),
        ("gradient_boosting", result["gradient_boosting_model"]),
        ("random_forest", result["random_forest_model"]),
    ]

    for key, model in models_for_eval:
        print(f"\n--- {model_display[key]} ---")
        evaluate_model(
            model,
            ftrain,
            ftest,
            ltrain,
            ltest,
            model_name=model_display[key],
            save_confusion_prefix=key,
            topic_labels=topic_labels,
        )
        plot_multiclass_roc(
            model,
            ftest,
            ltest,
            model_name=model_display[key],
            save_prefix=key,
            topic_labels=topic_labels,
        )

    if args.chi2:
        print("\n--- Chi2 top features per class ---")
        print_chi2_top_features(tfidf, ftrain, ltrain.values, topic_codes=topic_codes)

    if args.save_model:
        joblib.dump(tfidf, os.path.join(MODEL_DIR, "vectorizer.joblib"))
        metadata = {"topic_codes": topic_codes, "topic_labels": topic_labels}
        with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        joblib.dump(result["lgreg_model"], os.path.join(MODEL_DIR, "lgreg_model.joblib"))
        joblib.dump(result["gradient_boosting_model"], os.path.join(MODEL_DIR, "gradient_boosting_model.joblib"))
        joblib.dump(result["random_forest_model"], os.path.join(MODEL_DIR, "random_forest_model.joblib"))
        # Save best model name (used by predict.py by default)
        best_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME_FILE)
        with open(best_path, "w") as f:
            f.write(result["best_model_name"])
        print(f"\nModels and vectorizer saved to {MODEL_DIR}")
        print(f"Best model (by F1): {result['best_model_name']} -> saved as default for predict.py")

    print("\nDone.")


if __name__ == "__main__":
    main()
