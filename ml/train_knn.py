"""
ml/train_knn.py
---------------
Training script for the K-Nearest Neighbours (KNN) intent classifier.

Workflow:
  1. Loads the Bitext dataset and applies the shared preprocessing pipeline.
  2. Loads the ALREADY FITTED TF-IDF vectorizer and label encoder saved by
     train_nb.py — guarantees identical feature space across all models.
  3. Splits into train (70%) / validation (15%) / test (15%).
  4. Trains a KNN classifier with cosine similarity metric.
  5. Prints validation accuracy.
  6. Saves the KNN model to backend/saved_models/knn_model.pkl.

IMPORTANT: Run train_nb.py FIRST — this script reuses its vectorizer
           and label encoder.

Run from the project root:
    python ml/train_knn.py

Output files (in backend/saved_models/):
    knn_model.pkl  — KNN classifier
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

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
KNN_PATH      = os.path.join(MODELS_DIR, "knn_model.pkl")

RANDOM_SEED = 42


def load_and_prepare() -> tuple:
    """
    Load dataset, preprocess, encode labels, and return train/val splits
    as TF-IDF feature matrices.

    Returns:
        tuple: (X_train_tfidf, X_val_tfidf, y_train, y_val)

    Raises:
        FileNotFoundError: If dataset or vectorizer/encoder are missing.
    """
    for path, label in [
        (DATASET_PATH, "Dataset"),
        (TFIDF_PATH, "TF-IDF vectorizer"),
        (ENCODER_PATH, "Label encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at {path}.\n"
                "Run ml/train_nb.py first."
            )

    # Load and preprocess dataset.
    df = pd.read_csv(DATASET_PATH)[["utterance", "intent"]].dropna()
    print(f"[INFO] Loaded {len(df):,} samples across {df['intent'].nunique()} intents.")

    print("[INFO] Preprocessing utterances...")
    df["cleaned"] = df["utterance"].apply(preprocess)
    df = df[df["cleaned"].str.strip() != ""]

    # Load shared artifacts.
    vectorizer    = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    df["label"] = label_encoder.transform(df["intent"])

    X = df["cleaned"].values
    y = df["label"].values

    # Reconstruct the exact same split used in train_nb.py.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"[INFO] Train: {len(X_train):,} | Val: {len(X_val):,}")

    # Transform to TF-IDF (transform only — vectorizer already fitted).
    X_train_tfidf = vectorizer.transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)

    return X_train_tfidf, X_val_tfidf, y_train, y_val


def train() -> None:
    """
    Full KNN training pipeline: load → train → evaluate → save.

    KNN with cosine similarity works well for TF-IDF vectors because
    it measures the angle between document vectors rather than Euclidean
    distance, which is more meaningful in high-dimensional text space.
    """
    X_train_tfidf, X_val_tfidf, y_train, y_val = load_and_prepare()

    # --- Train KNN ---
    # n_neighbors=7 balances bias/variance for this dataset size.
    # metric='cosine' is standard for TF-IDF text classification.
    # algorithm='brute' is required when using cosine metric.
    print("[INFO] Training KNN classifier (this may take a few minutes)...")
    model = KNeighborsClassifier(
        n_neighbors=7,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,   # use all CPU cores
    )

    start = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start
    print(f"[INFO] Training completed in {train_time:.1f}s")

    # --- Validation accuracy ---
    print("[INFO] Running validation inference (KNN is slow at inference)...")
    val_preds = model.predict(X_val_tfidf)
    val_acc   = accuracy_score(y_val, val_preds)
    print(f"[INFO] Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # --- Save model ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, KNN_PATH)
    print(f"[INFO] Saved KNN model → {KNN_PATH}")
    print("[SUCCESS] KNN training complete.")


if __name__ == "__main__":
    train()
