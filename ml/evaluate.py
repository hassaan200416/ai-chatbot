"""
ml/evaluate.py
--------------
Comprehensive evaluation script for both trained models (NB and ANN).

Workflow:
  1. Load the dataset and apply the same preprocessing pipeline.
  2. Reconstruct the identical test split (same random_state=42 as training).
  3. Load saved models and vectorizer from backend/saved_models/.
  4. Run inference on the test set for both models.
  5. Print for each model:
       - Accuracy, Precision, Recall, F1-Score (macro & weighted)
       - Full per-intent classification report
       - Confusion matrix saved as a PNG image

Run from the project root:
    python ml/evaluate.py

Output:
    Metrics printed to console.
    ml/nb_confusion_matrix.png
    ml/ann_confusion_matrix.png
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
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
DATASET_PATH  = os.path.join(PROJECT_ROOT, "ml", "data", "bitext_dataset.csv")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "backend", "saved_models")
TFIDF_PATH    = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH  = os.path.join(MODELS_DIR, "label_encoder.pkl")
NB_PATH       = os.path.join(MODELS_DIR, "nb_model.pkl")
ANN_PATH      = os.path.join(MODELS_DIR, "ann_model.h5")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "ml")

RANDOM_SEED = 42


def load_test_set() -> tuple:
    """
    Load, preprocess, and reconstruct the identical test split used in training.

    Returns:
        tuple: (X_test_tfidf, y_test, label_encoder, class_names)
               X_test_tfidf — sparse TF-IDF matrix for NB
               X_test_dense — dense numpy array for ANN
    """
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)[["utterance", "intent"]].dropna()

    print("[INFO] Preprocessing utterances...")
    df["cleaned"] = df["utterance"].apply(preprocess)
    df = df[df["cleaned"].str.strip() != ""]

    # Load fitted artifacts.
    vectorizer    = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    df["label"] = label_encoder.transform(df["intent"])

    X = df["cleaned"].values
    y = df["label"].values

    # Reconstruct the EXACT same split as training scripts.
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    X_test_tfidf  = vectorizer.transform(X_test)
    X_test_dense  = X_test_tfidf.toarray()
    class_names   = label_encoder.classes_

    print(f"[INFO] Test set size: {len(X_test):,} samples | {len(class_names)} classes")
    return X_test_tfidf, X_test_dense, y_test, label_encoder, class_names


def print_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                  class_names: list) -> None:
    """
    Print accuracy, precision, recall, F1, and full classification report.

    Args:
        model_name  (str):        Display name of the model ("Naive Bayes" etc.)
        y_true      (np.ndarray): Ground truth integer labels.
        y_pred      (np.ndarray): Predicted integer labels.
        class_names (list):       List of string class names for the report.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Test Set Results")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f} (weighted)")
    print(f"  Recall    : {rec:.4f} (weighted)")
    print(f"  F1-Score  : {f1:.4f} (weighted)")
    print(f"{'='*60}")
    print("\n--- Per-Intent Classification Report ---\n")
    print(classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0))


def save_confusion_matrix(model_name: str, y_true: np.ndarray,
                          y_pred: np.ndarray, class_names: list,
                          filename: str) -> None:
    """
    Generate and save a confusion matrix as a PNG file.

    Args:
        model_name  (str):        Display name for the plot title.
        y_true      (np.ndarray): Ground truth labels.
        y_pred      (np.ndarray): Predicted labels.
        class_names (list):       Intent label names for axis ticks.
        filename    (str):        Output PNG file path.
    """
    cm      = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=class_names)

    fig, ax = plt.subplots(figsize=(18, 16))
    display.plot(ax=ax, xticks_rotation=90, colorbar=True, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix (Test Set)", fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {filename}")


def evaluate() -> None:
    """
    Main evaluation function — runs full evaluation for both NB and ANN.
    """
    # --- Load test set ---
    X_test_tfidf, X_test_dense, y_test, label_encoder, class_names = load_test_set()

    # -----------------------------------------------------------------------
    # Naive Bayes evaluation
    # -----------------------------------------------------------------------
    print("\n[INFO] Evaluating Naive Bayes...")
    nb_model  = joblib.load(NB_PATH)
    nb_preds  = nb_model.predict(X_test_tfidf)

    print_metrics("Naive Bayes", y_test, nb_preds, class_names)
    save_confusion_matrix(
        "Naive Bayes", y_test, nb_preds, class_names,
        os.path.join(OUTPUT_DIR, "nb_confusion_matrix.png")
    )

    # -----------------------------------------------------------------------
    # ANN evaluation
    # -----------------------------------------------------------------------
    print("\n[INFO] Evaluating ANN...")
    from keras.models import load_model
    ann_model  = load_model(ANN_PATH)
    ann_proba  = ann_model.predict(X_test_dense, verbose=0)
    ann_preds  = np.argmax(ann_proba, axis=1)

    print_metrics("ANN (MLP)", y_test, ann_preds, class_names)
    save_confusion_matrix(
        "ANN (MLP)", y_test, ann_preds, class_names,
        os.path.join(OUTPUT_DIR, "ann_confusion_matrix.png")
    )

    print("\n[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    evaluate()
