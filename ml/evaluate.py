"""
ml/evaluate.py
--------------
Comprehensive evaluation script for all three trained models
(Naive Bayes, KNN, and ANN).

Workflow:
  1. Load the dataset and apply the same preprocessing pipeline.
  2. Reconstruct the identical test split (same random_state=42 as training).
  3. Load all three saved models and the shared vectorizer.
  4. Run inference on the test set for each model.
  5. Print for each model:
       - Accuracy, Precision, Recall, F1-Score (weighted)
       - Full per-intent classification report
       - Confusion matrix saved as a PNG image
  6. Print a final 3-model comparison summary table.

Run from the project root:
    python ml/evaluate.py

Output files:
    ml/nb_confusion_matrix.png
    ml/knn_confusion_matrix.png
    ml/ann_confusion_matrix.png
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from preprocessor import preprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(PROJECT_ROOT, "ml", "data", "bitext_dataset.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "backend", "saved_models")
TFIDF_PATH   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
NB_PATH      = os.path.join(MODELS_DIR, "nb_model.pkl")
KNN_PATH     = os.path.join(MODELS_DIR, "knn_model.pkl")
ANN_PATH     = os.path.join(MODELS_DIR, "ann_model.h5")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "ml")

RANDOM_SEED = 42


def load_test_set() -> tuple:
    """
    Load, preprocess, and reconstruct the identical test split used in training.

    Returns:
        tuple: (X_test_tfidf, X_test_dense, y_test, class_names)
    """
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)[["utterance", "intent"]].dropna()

    print("[INFO] Preprocessing utterances...")
    df["cleaned"] = df["utterance"].apply(preprocess)
    df = df[df["cleaned"].str.strip() != ""]

    vectorizer    = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    df["label"]   = label_encoder.transform(df["intent"])

    X = df["cleaned"].values
    y = df["label"].values

    # Reconstruct exact same split as all training scripts.
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    X_test_tfidf = vectorizer.transform(X_test)
    X_test_dense = X_test_tfidf.toarray()
    class_names  = label_encoder.classes_

    print(f"[INFO] Test set: {len(X_test):,} samples | {len(class_names)} classes\n")
    return X_test_tfidf, X_test_dense, y_test, class_names


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    inference_time: float,
) -> dict:
    """
    Print full metrics report and return a summary dict for comparison table.

    Args:
        model_name     (str):        Display name of the model.
        y_true         (np.ndarray): Ground truth integer labels.
        y_pred         (np.ndarray): Predicted integer labels.
        class_names    (list):       Intent label strings.
        inference_time (float):      Seconds taken for inference on test set.

    Returns:
        dict: Summary metrics for the comparison table.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*62}")
    print(f"  {model_name} — Test Set Results")
    print(f"{'='*62}")
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision       : {prec:.4f} (weighted)")
    print(f"  Recall          : {rec:.4f} (weighted)")
    print(f"  F1-Score        : {f1:.4f} (weighted)")
    print(f"  Inference Time  : {inference_time:.2f}s on {len(y_true):,} samples")
    print(f"{'='*62}")
    print("\n--- Per-Intent Classification Report ---\n")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, zero_division=0))

    return {
        "model":     model_name,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "inf_time":  inference_time,
    }


def save_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    filename: str,
) -> None:
    """
    Generate and save a confusion matrix PNG.

    Args:
        model_name  (str):        Title prefix for the plot.
        y_true      (np.ndarray): Ground truth labels.
        y_pred      (np.ndarray): Predicted labels.
        class_names (list):       Intent label names.
        filename    (str):        Output PNG path.
    """
    cm      = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    fig, ax = plt.subplots(figsize=(18, 16))
    display.plot(ax=ax, xticks_rotation=90, colorbar=True, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix (Test Set)",
                 fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {filename}")


def print_comparison_table(results: list[dict]) -> None:
    """
    Print a formatted side-by-side comparison table of all models.

    Args:
        results (list[dict]): List of metric dicts from evaluate_model().
    """
    print(f"\n{'='*72}")
    print("  3-MODEL COMPARISON SUMMARY (Test Set)")
    print(f"{'='*72}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'Inf(s)':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for r in results:
        print(
            f"  {r['model']:<20} "
            f"{r['accuracy']*100:>9.2f}% "
            f"{r['precision']*100:>9.2f}% "
            f"{r['recall']*100:>9.2f}% "
            f"{r['f1']*100:>9.2f}% "
            f"{r['inf_time']:>7.2f}s"
        )
    print(f"{'='*72}\n")


def evaluate() -> None:
    """Main evaluation function — runs all three models and prints comparison."""

    X_test_tfidf, X_test_dense, y_test, class_names = load_test_set()
    results = []

    # -----------------------------------------------------------------------
    # 1. Naive Bayes
    # -----------------------------------------------------------------------
    print("[INFO] Evaluating Naive Bayes...")
    nb_model = joblib.load(NB_PATH)

    t0 = time.time()
    nb_preds = nb_model.predict(X_test_tfidf)
    nb_time  = time.time() - t0

    results.append(evaluate_model(
        "Naive Bayes", y_test, nb_preds, class_names, nb_time
    ))
    save_confusion_matrix(
        "Naive Bayes", y_test, nb_preds, class_names,
        os.path.join(OUTPUT_DIR, "nb_confusion_matrix.png")
    )

    # -----------------------------------------------------------------------
    # 2. KNN
    # -----------------------------------------------------------------------
    print("\n[INFO] Evaluating KNN (inference is slow — please wait)...")
    knn_model = joblib.load(KNN_PATH)

    t0 = time.time()
    knn_preds = knn_model.predict(X_test_tfidf)
    knn_time  = time.time() - t0

    results.append(evaluate_model(
        "KNN (k=7, cosine)", y_test, knn_preds, class_names, knn_time
    ))
    save_confusion_matrix(
        "KNN (k=7, cosine)", y_test, knn_preds, class_names,
        os.path.join(OUTPUT_DIR, "knn_confusion_matrix.png")
    )

    # -----------------------------------------------------------------------
    # 3. ANN
    # -----------------------------------------------------------------------
    print("\n[INFO] Evaluating ANN...")
    from keras.models import load_model
    ann_model = load_model(ANN_PATH)

    t0 = time.time()
    ann_proba = ann_model.predict(X_test_dense, verbose=0)
    ann_preds = np.argmax(ann_proba, axis=1)
    ann_time  = time.time() - t0

    results.append(evaluate_model(
        "ANN (MLP)", y_test, ann_preds, class_names, ann_time
    ))
    save_confusion_matrix(
        "ANN (MLP)", y_test, ann_preds, class_names,
        os.path.join(OUTPUT_DIR, "ann_confusion_matrix.png")
    )

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print_comparison_table(results)
    print("[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    evaluate()
