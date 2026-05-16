"""
backend/predictor.py
--------------------
Model loading and inference module for the AI Chatbot.

Workflow:
  1. At import time, reads MODEL_TYPE from config ("nb" or "ann").
  2. Loads the corresponding saved model and TF-IDF vectorizer from
     backend/saved_models/.
  3. Exposes predict() which:
       a. Preprocesses raw user text via preprocessor.py
       b. Transforms it using the loaded TF-IDF vectorizer
       c. Runs inference with the loaded model
       d. Returns the predicted intent label and confidence score

Supported models:
  - "nb"  → Scikit-learn MultinomialNB (.pkl via joblib)
  - "ann" → Keras MLP (.h5 via keras.models.load_model)

This module is imported by backend/app.py only.
Models are trained by ml/train_nb.py and ml/train_ann.py.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np

from config import config
from preprocessor import preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File name constants — must match what the training scripts save.
# ---------------------------------------------------------------------------
_TFIDF_PATH = os.path.join(config.models_dir, "tfidf_vectorizer.pkl")
_NB_MODEL_PATH = os.path.join(config.models_dir, "nb_model.pkl")
_ANN_MODEL_PATH = os.path.join(config.models_dir, "ann_model.h5")
_LABEL_ENCODER_PATH = os.path.join(config.models_dir, "label_encoder.pkl")

# Confidence threshold — predictions below this score return None intent,
# triggering the fallback response in response_map.py.
_CONFIDENCE_THRESHOLD = 0.30


def _load_models() -> tuple:
    """
    Load the TF-IDF vectorizer, label encoder, and the active model from disk.

    Returns:
        tuple: (vectorizer, label_encoder, model, model_type_str)
               model_type_str is "nb" or "ann" for logging/response metadata.

    Raises:
        FileNotFoundError: If any required model file is missing.
                           Run the appropriate training script first.
    """
    # --- Validate required files exist before attempting to load ---
    required_files = [_TFIDF_PATH, _LABEL_ENCODER_PATH]

    if config.model_type == "ann":
        required_files.append(_ANN_MODEL_PATH)
    else:
        required_files.append(_NB_MODEL_PATH)

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run ml/train_{'ann' if config.model_type == 'ann' else 'nb'}.py first."
            )

    # --- Load TF-IDF vectorizer (shared by both models) ---
    vectorizer = joblib.load(_TFIDF_PATH)
    logger.info("Loaded TF-IDF vectorizer from %s", _TFIDF_PATH)

    # --- Load label encoder (shared by both models) ---
    label_encoder = joblib.load(_LABEL_ENCODER_PATH)
    logger.info("Loaded label encoder from %s", _LABEL_ENCODER_PATH)

    # --- Load the active model ---
    if config.model_type == "ann":
        # Import Keras here to avoid loading TensorFlow when using NB only.
        from keras.models import load_model
        model = load_model(_ANN_MODEL_PATH)
        logger.info("Loaded ANN model from %s", _ANN_MODEL_PATH)
    else:
        model = joblib.load(_NB_MODEL_PATH)
        logger.info("Loaded Naive Bayes model from %s", _NB_MODEL_PATH)

    return vectorizer, label_encoder, model, config.model_type


def predict(user_text: str) -> dict:
    """
    Run the full inference pipeline on raw user input.

    Pipeline:
      1. Preprocess text (lowercase, tokenise, remove stopwords, lemmatise)
      2. Transform with TF-IDF vectorizer
      3. Run model inference
      4. Decode label and extract confidence
      5. Apply confidence threshold

    Args:
        user_text (str): Raw message from the user.

    Returns:
        dict with keys:
            intent     (str | None): Predicted intent label, or None if
                                     confidence is below threshold.
            confidence (float):      Confidence score between 0.0 and 1.0.
            model_used (str):        "nb" or "ann".
    """
    if not user_text or not user_text.strip():
        return {"intent": None, "confidence": 0.0, "model_used": config.model_type}

    # Step 1 — Preprocess raw text.
    cleaned = preprocess(user_text)

    # Step 2 — Transform to TF-IDF feature vector.
    # reshape to (1, n_features) for single-sample prediction.
    features = _vectorizer.transform([cleaned])

    # Step 3 — Run inference and extract probability scores.
    if config.model_type == "ann":
        # Keras returns a (1, n_classes) probability array.
        proba = _model.predict(features.toarray(), verbose=0)[0]
    else:
        # Scikit-learn returns a (1, n_classes) array.
        proba = _model.predict_proba(features)[0]

    # Step 4 — Get the highest-confidence class index and score.
    class_index = int(np.argmax(proba))
    confidence = float(proba[class_index])

    # Step 5 — Decode numeric index back to intent string.
    intent = _label_encoder.inverse_transform([class_index])[0]

    # Step 6 — Apply confidence threshold to avoid overconfident wrong answers.
    if confidence < _CONFIDENCE_THRESHOLD:
        logger.info(
            "Low confidence (%.2f) for input: '%s' — returning fallback.",
            confidence, user_text
        )
        intent = None

    logger.info("Predicted intent='%s' confidence=%.2f", intent, confidence)

    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "model_used": config.model_type,
    }


# ---------------------------------------------------------------------------
# Load models at module import time so the first request is not slow.
# This will raise FileNotFoundError if models haven't been trained yet —
# that is intentional. Train the models before starting the Flask server.
# ---------------------------------------------------------------------------
try:
    _vectorizer, _label_encoder, _model, _model_type = _load_models()
except FileNotFoundError as e:
    # Allow the module to import without crashing during development
    # (before models are trained). Prediction will fail gracefully.
    logger.warning("Models not loaded: %s", e)
    _vectorizer = _label_encoder = _model = _model_type = None
