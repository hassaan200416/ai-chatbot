"""
ml/train_ann.py
---------------
Training script for the ANN (Multi-Layer Perceptron) intent classifier.

Workflow:
  1. Load and preprocess the Bitext dataset (same pipeline as train_nb.py).
  2. Load the ALREADY FITTED TF-IDF vectorizer and label encoder saved by
     train_nb.py — this guarantees identical feature space for both models.
  3. Split into train (70%) / validation (15%) / test (15%) sets.
  4. Build a Keras MLP with:
       - Input layer  : TF-IDF feature dimension
       - Hidden layer 1: 512 units, ReLU, BatchNorm, Dropout(0.4)
       - Hidden layer 2: 256 units, ReLU, BatchNorm, Dropout(0.3)
       - Output layer : softmax over 27 intent classes
  5. Train with early stopping to prevent overfitting.
  6. Save the trained model to backend/saved_models/ann_model.h5.

IMPORTANT: Run train_nb.py FIRST — this script reuses its vectorizer
           and label encoder to ensure consistency.

Run from the project root:
    python ml/train_ann.py

Output files (in backend/saved_models/):
    ann_model.h5   — Keras MLP model
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

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
ANN_PATH     = os.path.join(MODELS_DIR, "ann_model.h5")

# ---------------------------------------------------------------------------
# Reproducibility — fix random seeds so results are consistent across runs.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_and_preprocess() -> tuple:
    """
    Load dataset, preprocess utterances, encode labels, and split into
    train/val/test sets.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test,
                n_classes, vectorizer, label_encoder)

    Raises:
        FileNotFoundError: If dataset or saved vectorizer/encoder are missing.
    """
    # --- Validate prerequisites ---
    for path, label in [(DATASET_PATH, "Dataset"), (TFIDF_PATH, "TF-IDF vectorizer"),
                        (ENCODER_PATH, "Label encoder")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at {path}.\n"
                "Run ml/train_nb.py first."
            )

    # --- Load dataset ---
    df = pd.read_csv(DATASET_PATH)[["utterance", "intent"]].dropna()
    print(f"[INFO] Loaded {len(df):,} samples across {df['intent'].nunique()} intents.")

    # --- Preprocess ---
    print("[INFO] Preprocessing utterances...")
    df["cleaned"] = df["utterance"].apply(preprocess)
    df = df[df["cleaned"].str.strip() != ""]

    # --- Load fitted vectorizer and encoder (from train_nb.py) ---
    vectorizer    = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"[INFO] Loaded vectorizer with {len(vectorizer.vocabulary_):,} features.")

    # --- Encode labels ---
    df["label"] = label_encoder.transform(df["intent"])
    n_classes   = len(label_encoder.classes_)

    # --- Split 70 / 15 / 15 ---
    X = df["cleaned"].values
    y = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(
        f"[INFO] Split → Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    # --- Transform to TF-IDF (vectorizer already fitted — transform only) ---
    X_train = vectorizer.transform(X_train).toarray()
    X_val   = vectorizer.transform(X_val).toarray()
    X_test  = vectorizer.transform(X_test).toarray()

    return X_train, X_val, X_test, y_train, y_val, y_test, n_classes, vectorizer, label_encoder


def build_model(input_dim: int, n_classes: int) -> Sequential:
    """
    Build and compile the Keras MLP model.

    Architecture:
        Dense(512, ReLU) → BatchNorm → Dropout(0.4)
        Dense(256, ReLU) → BatchNorm → Dropout(0.3)
        Dense(n_classes, softmax)

    Args:
        input_dim (int): Number of TF-IDF features.
        n_classes (int): Number of intent classes (27).

    Returns:
        keras.Sequential: Compiled model ready for training.
    """
    model = Sequential([
        # Hidden layer 1
        Dense(512, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),

        # Hidden layer 2
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer — softmax for multi-class probability distribution
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train() -> None:
    """
    Full ANN training pipeline: load → build → train → evaluate → save.
    """
    # --- Load data ---
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     n_classes, vectorizer, label_encoder) = load_and_preprocess()

    # --- One-hot encode labels for categorical crossentropy ---
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_val_cat   = to_categorical(y_val,   num_classes=n_classes)

    # --- Build model ---
    input_dim = X_train.shape[1]
    model     = build_model(input_dim, n_classes)

    # --- Callbacks ---
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,             # stop if val_accuracy doesn't improve for 5 epochs
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,             # halve learning rate on plateau
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # --- Train ---
    print("[INFO] Starting ANN training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # --- Validation accuracy ---
    val_preds  = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_acc    = accuracy_score(y_val, val_preds)
    print(f"[INFO] Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # --- Save model ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(ANN_PATH)
    print(f"[INFO] Saved ANN model → {ANN_PATH}")
    print("[SUCCESS] ANN training complete.")


if __name__ == "__main__":
    train()
