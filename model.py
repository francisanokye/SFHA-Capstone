"""
TF-IDF feature extraction and model training for Reddit mental-health classification.
Includes: Logistic Regression, Gradient Boosting, Random Forest.
Best model is selected by F1 (weighted) score.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

from config import (
    NGRAM_RANGE,
    MIN_DF,
    MAX_DF,
    MAX_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    CV_N_SPLITS,
    CV_TEST_SIZE,
    CV_RANDOM_STATE,
    TOPIC_CODES,
    LOGISTIC_REGRESSION_MAX_ITER,
    GB_N_ESTIMATORS,
    GB_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
)


def add_topic_codes(df, topic_codes=None):
    """Add numeric topic_code column from Subreddit using topic_codes mapping."""
    codes = topic_codes or TOPIC_CODES
    df = df.copy()
    df["topic_code"] = df["Subreddit"].replace(codes).astype(int)
    return df


def build_vectorizer(ngram_range=None, min_df=None, max_df=None, max_features=None):
    """Build TF-IDF vectorizer with pipeline defaults."""
    return TfidfVectorizer(
        encoding="utf-8",
        ngram_range=ngram_range or NGRAM_RANGE,
        min_df=min_df if min_df is not None else MIN_DF,
        max_df=max_df if max_df is not None else MAX_DF,
        lowercase=False,
        max_features=max_features if max_features is not None else MAX_FEATURES,
        stop_words=None,
        norm="l2",
        sublinear_tf=True,
    )


def prepare_features(df, vectorizer=None, fit=True):
    """
    Transform 'content' with TF-IDF. If vectorizer is None, create and fit one.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray (topic_code)
    vectorizer : TfidfVectorizer (fitted)
    """
    tfidf = vectorizer or build_vectorizer()
    texts = df["content"]
    if fit:
        X = tfidf.fit_transform(texts).toarray()
    else:
        X = tfidf.transform(texts).toarray()
    y = df["topic_code"].values
    return X, y, tfidf


def train_test_split_data(df, vectorizer=None):
    """
    Split into train/test and return features and labels for both, plus fitted vectorizer.
    """
    tfidf = vectorizer or build_vectorizer()
    x_train, x_test, y_train, y_test = train_test_split(
        df["content"],
        df["topic_code"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    features_train = tfidf.fit_transform(x_train).toarray()
    features_test = tfidf.transform(x_test).toarray()
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "features_train": features_train,
        "features_test": features_test,
        "labels_train": y_train,
        "labels_test": y_test,
        "vectorizer": tfidf,
    }


def cross_validate_model(model, x, y):
    """Run 10-fold ShuffleSplit CV and return mean accuracy, precision, recall, F1."""
    cv = ShuffleSplit(
        n_splits=CV_N_SPLITS, test_size=CV_TEST_SIZE, random_state=CV_RANDOM_STATE
    )
    results = {}
    for name, scoring in [
        ("Accuracy", "accuracy"),
        ("Precision", "precision_weighted"),
        ("Recall", "recall_weighted"),
        ("F1", "f1_weighted"),
    ]:
        scores = cross_val_score(model, x, y, cv=cv, scoring=scoring)
        results[name] = scores.mean()
    return results


def train_models(df, run_cv=True, topic_codes=None):
    """
    Prepare features, train all models (Logistic Regression, GradientBoosting, RandomForest),
    run CV, select best by F1 (weighted), and return all models + best.

    topic_codes : dict, optional
        Mapping of subreddit label -> int. If None, uses config.TOPIC_CODES.
        n_classes is len(topic_codes) for LSTM.

    Returns
    -------
    dict with keys:
        data, vectorizer,
        mnb_model, lgreg_model, knn_model, gradient_boosting_model, random_forest_model, lstm_model (if available),
        cv_results : dict model_key -> {Accuracy, Precision, Recall, F1}
        best_model_name : str (key in MODEL_KEYS)
        best_model : fitted model with highest F1
    """
    codes = topic_codes or TOPIC_CODES
    data = train_test_split_data(df)
    ftrain = data["features_train"]
    ftest = data["features_test"]
    ltrain = data["labels_train"]
    ltest = data["labels_test"]
    tfidf = data["vectorizer"]
    n_features = ftrain.shape[1]

    cv_results = {}
    if run_cv:
        cv = ShuffleSplit(
            n_splits=CV_N_SPLITS, test_size=CV_TEST_SIZE, random_state=CV_RANDOM_STATE
        )
        x_cv = np.vstack([ftrain, ftest])
        y_cv = np.concatenate([ltrain.values, ltest.values])
        for model_name, model in [
            ("lgreg", LogisticRegression(multi_class="ovr", max_iter=LOGISTIC_REGRESSION_MAX_ITER)),
            ("gradient_boosting", GradientBoostingClassifier(n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH, random_state=RANDOM_STATE)),
            ("random_forest", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RANDOM_STATE)),
        ]:
            cv_results[model_name] = {
                k: cross_val_score(model, x_cv, y_cv, cv=cv, scoring=scoring).mean()
                for k, scoring in [
                    ("Accuracy", "accuracy"),
                    ("Precision", "precision_weighted"),
                    ("Recall", "recall_weighted"),
                    ("F1", "f1_weighted"),
                ]
            }

    # Fit models
    lgreg_model = LogisticRegression(multi_class="ovr", max_iter=LOGISTIC_REGRESSION_MAX_ITER).fit(ftrain, ltrain)
    gradient_boosting_model = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH, random_state=RANDOM_STATE
    ).fit(ftrain, ltrain)
    random_forest_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RANDOM_STATE
    ).fit(ftrain, ltrain)

    # If CV was disabled, compute F1 on test set to choose best model
    if not run_cv:
        for model_name, model in [
            ("lgreg", lgreg_model),
            ("gradient_boosting", gradient_boosting_model),
            ("random_forest", random_forest_model),
        ]:
            preds = model.predict(ftest)
            f1 = f1_score(ltest, preds, average="weighted")
            cv_results[model_name] = {
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1": f1,
            }

    # Best model by F1 (weighted)
    f1_scores = {k: (v.get("F1") or 0.0) for k, v in cv_results.items()}
    best_model_name = max(f1_scores, key=f1_scores.get)
    models_by_key = {
        "lgreg": lgreg_model,
        "gradient_boosting": gradient_boosting_model,
        "random_forest": random_forest_model,
    }
    best_model = models_by_key.get(best_model_name)

    return {
        "data": data,
        "vectorizer": tfidf,
        "lgreg_model": lgreg_model,
        "gradient_boosting_model": gradient_boosting_model,
        "random_forest_model": random_forest_model,
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "models_by_key": models_by_key,
    }
