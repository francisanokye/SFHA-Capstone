"""
Evaluation: classification reports, confusion matrices, and feature analysis (chi2).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.feature_selection import chi2
from sklearn.preprocessing import label_binarize

from config import TOPIC_CODES, TOPIC_LABELS, OUTPUT_DIR
import os


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_classification_report(labels_test, predictions, model_name="Model", topic_labels=None):
    """Print sklearn classification report."""
    names = topic_labels or TOPIC_LABELS
    print(f"{model_name} Classification Report")
    print(classification_report(labels_test, predictions, target_names=names))


def plot_confusion_matrix(
    labels_test,
    predictions,
    title="Confusion Matrix",
    save_path=None,
    topic_labels=None,
):
    """Plot and optionally save confusion matrix heatmap."""
    names = topic_labels or TOPIC_LABELS
    ensure_output_dir()
    cfmatrix = confusion_matrix(labels_test, predictions)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    sns.heatmap(
        cfmatrix,
        annot=True,
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        path = os.path.join(OUTPUT_DIR, save_path)
        plt.savefig(path)
        print(f"Saved: {path}")
    plt.close()


def evaluate_model(
    model,
    features_train,
    features_test,
    labels_train,
    labels_test,
    model_name="Model",
    save_confusion_prefix=None,
    topic_labels=None,
):
    """
    Print report, optionally plot confusion matrix, return accuracy dict.
    """
    predictions = model.predict(features_test)
    print_classification_report(
        labels_test, predictions, model_name=model_name, topic_labels=topic_labels
    )
    train_acc = (model.predict(features_train) == labels_train).mean()
    test_acc = (predictions == labels_test).mean()
    if save_confusion_prefix:
        plot_confusion_matrix(
            labels_test,
            predictions,
            title=f"{model_name} Confusion Matrix",
            save_path=f"{save_confusion_prefix}_confusion.png",
            topic_labels=topic_labels,
        )
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "predictions": predictions,
    }


def plot_multiclass_roc(
    model,
    features_test,
    labels_test,
    model_name="Model",
    save_prefix=None,
    topic_labels=None,
):
    """
    Plot ROC curves for multiclass classification using one-vs-rest strategy.
    Saves a single figure with per-class and micro-average ROC.
    """
    if save_prefix is None:
        return

    # Require probability or decision scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(features_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(features_test)
        if y_score.ndim == 1:
            # Binary case; convert to two-column scores
            y_score = np.vstack([-y_score, y_score]).T
    else:
        print(f"{model_name} does not support probability scores; skipping ROC.")
        return

    names = topic_labels or TOPIC_LABELS
    n_classes = len(names)
    classes = np.arange(n_classes)
    y_test_bin = label_binarize(labels_test, classes=classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ensure_output_dir()
    plt.figure(figsize=(8, 6), dpi=120)
    # Per class curves
    for i, name in enumerate(names):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=1.5,
            label=f"{name} (AUC = {roc_auc[i]:.2f})",
        )
    # Micro-average
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle="--",
        color="black",
        lw=2,
        label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
    )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves – {model_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    filename = f"{save_prefix}_roc.png"
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def print_chi2_top_features(
    tfidf,
    features_train,
    labels_train,
    topic_codes=None,
    n_unigrams=10,
    n_bigrams=3,
):
    """
    Print top chi2-correlated unigrams and bigrams per class (as in notebook cell 40).
    """
    topic_codes = topic_codes or TOPIC_CODES
    for subreddit, topic_code in sorted(topic_codes.items()):
        features_chi2 = chi2(features_train, labels_train == topic_code)
        indices = np.argsort(features_chi2[0])
        try:
            feature_names = np.array(tfidf.get_feature_names_out())
        except AttributeError:
            feature_names = np.array(tfidf.get_feature_names())
        unigrams = [w for w in feature_names[indices] if len(w.split()) == 1]
        bigrams = [w for w in feature_names[indices] if len(w.split()) == 2]
        print(f"# '{subreddit}':")
        print("  . Top correlated unigrams:\n    .", "\n    . ".join(unigrams[-n_unigrams:]))
        print("  . Top correlated bigrams:\n    .", "\n    . ".join(bigrams[-n_bigrams:]))
        print()
