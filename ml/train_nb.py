"""
ml/train_nb.py
--------------
Training script for the Naive Bayes intent classifier.

Workflow:
  1. Load the Bitext Customer Support dataset from ml/data/.
  2. Clean all utterances using the shared preprocessor pipeline.
  3. Encode intent labels with LabelEncoder.
  4. Split into train (70%) / validation (15%) / test (15%) sets.
  5. Fit a TF-IDF vectorizer on the training set only (no data leakage).
  6. Train a Multinomial Naive Bayes classifier.
  7. Print validation accuracy as a quick sanity check.
  8. Save the model, vectorizer, and label encoder to backend/saved_models/.

Run from the project root:
    python ml/train_nb.py

Output files (in backend/saved_models/):
    tfidf_vectorizer.pkl   — shared by both NB and ANN
    label_encoder.pkl      — shared by both NB and ANN
    nb_model.pkl           — Naive Bayes classifier
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Path setup — allow importing from backend/ regardless of working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from preprocessor import preprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(PROJECT_ROOT, "ml", "data", "bitext_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "saved_models")


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Bitext dataset CSV and return only the columns we need.

    Args:
        path (str): Absolute path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['utterance', 'intent'],
                      with null rows dropped.

    Raises:
        FileNotFoundError: If the CSV is not found at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}\n"
            "Download from: https://www.kaggle.com/datasets/bitext/"
            "training-dataset-for-chatbotsvirtual-assistants\n"
            "Place the CSV in ml/data/ and rename it to bitext_dataset.csv"
        )

    df = pd.read_csv(path)

    # Keep only the two columns this project uses.
    df = df[["utterance", "intent"]].dropna()
    print(f"[INFO] Loaded {len(df):,} samples across {df['intent'].nunique()} intents.")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the NLP preprocessing pipeline to every utterance.

    Args:
        df (pd.DataFrame): Raw dataframe with 'utterance' column.

    Returns:
        pd.DataFrame: Same dataframe with a new 'cleaned' column added,
                      and rows where cleaning produced an empty string removed.
    """
    print("[INFO] Preprocessing utterances (this may take a minute)...")
    df = df.copy()
    df["cleaned"] = df["utterance"].apply(preprocess)

    # Drop rows where preprocessing returned an empty string.
    before = len(df)
    df = df[df["cleaned"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} empty rows after preprocessing.")

    return df


def train(df: pd.DataFrame) -> None:
    """
    Full training pipeline: encode → split → vectorize → train → save.

    Args:
        df (pd.DataFrame): Preprocessed dataframe with 'cleaned' and
                           'intent' columns.
    """
    # --- Step 1: Encode string labels to integers ---
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["intent"])
    print(f"[INFO] Encoded {len(label_encoder.classes_)} intent classes.")

    # --- Step 2: Train / Val / Test split (70 / 15 / 15) ---
    X = df["cleaned"].values
    y = df["label"].values

    # First split off 30% for val+test, then split that 50/50.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(
        f"[INFO] Split → Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    # --- Step 3: Fit TF-IDF on training data ONLY ---
    # max_features limits vocabulary size; ngram_range captures phrases.
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True,      # apply log normalization to term frequencies
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    print(f"[INFO] TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")

    # --- Step 4: Train Multinomial Naive Bayes ---
    # alpha=0.1 is a lighter smoothing than default (1.0) which works
    # better on short customer support utterances.
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    # --- Step 5: Validation accuracy (quick sanity check) ---
    val_preds = model.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"[INFO] Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # --- Step 6: Save all artifacts ---
    os.makedirs(MODELS_DIR, exist_ok=True)

    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    model_path = os.path.join(MODELS_DIR, "nb_model.pkl")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(model, model_path)

    print(f"[INFO] Saved vectorizer → {vectorizer_path}")
    print(f"[INFO] Saved label encoder → {encoder_path}")
    print(f"[INFO] Saved NB model → {model_path}")

    print("[SUCCESS] Naive Bayes training complete.")


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)
    df = preprocess_dataset(df)
    train(df)
