"""
backend/predictor.py
--------------------
Model loading and inference module for the AI Chatbot.

Workflow:
  1. At import time, reads MODEL_TYPE from config ("nb", "knn", or "ann").
  2. Loads the corresponding saved model and TF-IDF vectorizer from
     backend/saved_models/.
  3. Exposes predict() which:
       a. Preprocesses raw user text via preprocessor.py
       b. Transforms it using the loaded TF-IDF vectorizer
       c. Runs inference with the loaded model
       d. Returns the predicted intent label and confidence score

Supported models:
  - "nb"  → Scikit-learn MultinomialNB (.pkl via joblib)
  - "knn" → Scikit-learn KNeighborsClassifier (.pkl via joblib)
  - "ann" → Keras MLP (.h5 via keras.models.load_model)

This module is imported by backend/app.py only.
Models are trained by ml/train_nb.py, ml/train_knn.py, ml/train_ann.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from config import config
from preprocessor import preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File name constants — must match what the training scripts save.
# ---------------------------------------------------------------------------
_TFIDF_PATH         = os.path.join(config.models_dir, "tfidf_vectorizer.pkl")
_NB_MODEL_PATH      = os.path.join(config.models_dir, "nb_model.pkl")
_KNN_MODEL_PATH     = os.path.join(config.models_dir, "knn_model.pkl")
_ANN_MODEL_PATH     = os.path.join(config.models_dir, "ann_model.h5")
_LABEL_ENCODER_PATH = os.path.join(config.models_dir, "label_encoder.pkl")

# Confidence threshold — predictions below this score return None intent,
# triggering the fallback response in response_map.py.
# Note: KNN uses a voting-based confidence so threshold is lower.
_CONFIDENCE_THRESHOLD = 0.30

_vectorizer: Any | None = None
_label_encoder: Any | None = None
_model: Any | None = None
_model_type: str | None = None


def _load_models() -> tuple[Any, Any, Any, str]:
    """
    Load the TF-IDF vectorizer, label encoder, and the active model from disk.

    Returns:
        tuple: (vectorizer, label_encoder, model, model_type_str)

    Raises:
        FileNotFoundError: If any required model file is missing.
    """
    required_files = [_TFIDF_PATH, _LABEL_ENCODER_PATH]

    model_path_map = {
        "ann": _ANN_MODEL_PATH,
        "knn": _KNN_MODEL_PATH,
        "nb":  _NB_MODEL_PATH,
    }

    active_model_path = model_path_map.get(config.model_type, _ANN_MODEL_PATH)
    required_files.append(active_model_path)

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run ml/train_{config.model_type}.py first."
            )

    # --- Load shared artifacts ---
    vectorizer    = joblib.load(_TFIDF_PATH)  # pyright: ignore[reportUnknownMemberType]
    label_encoder = joblib.load(_LABEL_ENCODER_PATH)  # pyright: ignore[reportUnknownMemberType]
    logger.info("Loaded TF-IDF vectorizer and label encoder.")

    # --- Load the active model ---
    model: Any = None
    if config.model_type == "ann":
        from keras.models import load_model  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
        model = load_model(_ANN_MODEL_PATH)  # pyright: ignore[reportUnknownVariableType]
        logger.info("Loaded ANN model.")
    elif config.model_type == "knn":
        model = joblib.load(_KNN_MODEL_PATH)  # pyright: ignore[reportUnknownMemberType]
        logger.info("Loaded KNN model.")
    else:
        model = joblib.load(_NB_MODEL_PATH)  # pyright: ignore[reportUnknownMemberType]
        logger.info("Loaded Naive Bayes model.")

    return vectorizer, label_encoder, cast(Any, model), config.model_type


def predict(user_text: str) -> dict[str, Any]:
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
            model_used (str):        "nb", "knn", or "ann".
    """
    if not user_text or not user_text.strip():
        return {"intent": None, "confidence": 0.0, "model_used": config.model_type}

    assert _vectorizer is not None
    assert _label_encoder is not None
    assert _model is not None

    # Step 1 — Preprocess.
    cleaned  = preprocess(user_text)

    # Step 2 — TF-IDF transform.
    features = _vectorizer.transform([cleaned])

    # Step 3 — Inference.
    if config.model_type == "ann":
        # Keras returns (1, n_classes) probability array.
        proba = _model.predict(features.toarray(), verbose=0)[0]

    elif config.model_type == "knn":
        # KNN predict_proba returns fraction of neighbours per class.
        # This is a valid confidence measure for voting-based classifiers.
        proba = _model.predict_proba(features)[0]

    else:
        # Naive Bayes log-probability normalised to probabilities.
        proba = _model.predict_proba(features)[0]

    # Step 4 — Top class and confidence.
    class_index = int(np.argmax(proba))
    confidence  = float(proba[class_index])
    intent      = _label_encoder.inverse_transform([class_index])[0]

    # Step 5 — Confidence threshold.
    if confidence < _CONFIDENCE_THRESHOLD:
        logger.info(
            "Low confidence (%.2f) for '%s' — returning fallback.",
            confidence, user_text
        )
        intent = None

    logger.info("Predicted intent='%s' confidence=%.4f model='%s'",
                intent, confidence, config.model_type)

    return {
        "intent":     intent,
        "confidence": round(confidence, 4),
        "model_used": config.model_type,
    }


# ---------------------------------------------------------------------------
# Load models at import time.
# ---------------------------------------------------------------------------
try:
    _vectorizer, _label_encoder, _model, _model_type = _load_models()
except FileNotFoundError as e:
    logger.warning("Models not loaded: %s", e)
    _vectorizer = _label_encoder = _model = None
    _model_type = None
