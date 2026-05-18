"""
backend/predictor.py
--------------------
Model loading and inference module for the AI Chatbot.

Workflow:
  1. At import time, loads ALL THREE models (NB, KNN, ANN) into memory.
  2. predict() accepts a model_type parameter so the frontend can switch
     models per request without restarting the server.
  3. Returns predicted intent, confidence, and which model was used.

Supported models:
  - "nb"  → Scikit-learn MultinomialNB
  - "knn" → Scikit-learn KNeighborsClassifier
  - "ann" → Keras MLP

This module is imported by backend/app.py only.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from config import config
from preprocessor import preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_TFIDF_PATH         = os.path.join(config.models_dir, "tfidf_vectorizer.pkl")
_NB_MODEL_PATH      = os.path.join(config.models_dir, "nb_model.pkl")
_KNN_MODEL_PATH     = os.path.join(config.models_dir, "knn_model.pkl")
_ANN_MODEL_PATH     = os.path.join(config.models_dir, "ann_model.h5")
_LABEL_ENCODER_PATH = os.path.join(config.models_dir, "label_encoder.pkl")

_CONFIDENCE_THRESHOLD = 0.30

_models: dict[str, Any] | None = None


def _load_all_models() -> dict[str, Any]:
    """
    Load all three models plus shared artifacts at startup.
    All models are kept in memory so switching is instant — no disk reads
    per request.

    Returns:
        dict with keys: vectorizer, label_encoder, nb, knn, ann

    Raises:
        FileNotFoundError: If any model file is missing.
    """
    required = {
        "TF-IDF vectorizer": _TFIDF_PATH,
        "Label encoder":     _LABEL_ENCODER_PATH,
        "Naive Bayes":       _NB_MODEL_PATH,
        "KNN":               _KNN_MODEL_PATH,
        "ANN":               _ANN_MODEL_PATH,
    }

    for label, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at {path}\n"
                f"Run the corresponding training script first."
            )

    vectorizer    = joblib.load(_TFIDF_PATH)  # pyright: ignore[reportUnknownMemberType]
    label_encoder = joblib.load(_LABEL_ENCODER_PATH)  # pyright: ignore[reportUnknownMemberType]
    nb_model      = joblib.load(_NB_MODEL_PATH)  # pyright: ignore[reportUnknownMemberType]
    knn_model     = joblib.load(_KNN_MODEL_PATH)  # pyright: ignore[reportUnknownMemberType]

    from keras.models import load_model  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
    ann_model: Any = load_model(_ANN_MODEL_PATH)  # pyright: ignore[reportUnknownVariableType]

    logger.info("All 3 models loaded into memory (NB, KNN, ANN).")

    return {
        "vectorizer":     vectorizer,
        "label_encoder":  label_encoder,
        "nb":             nb_model,
        "knn":            knn_model,
        "ann":            ann_model,
    }


def predict(
    user_text: str,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    model_type: str | None = None,
) -> dict[str, Any]:
    """
    Run the full inference pipeline on raw user input.

    Args:
        user_text            (str):   Raw message from the user.
        confidence_threshold (float): Minimum confidence to return an intent.
                                      Below this returns None (fallback).
        model_type           (str):   Which model to use — "nb", "knn", "ann".
                                      Defaults to config.model_type if None.

    Returns:
        dict:
            intent     (str|None): Predicted intent or None if low confidence.
            confidence (float):    Confidence score 0.0–1.0.
            model_used (str):      Which model was used.
    """
    # Resolve which model to use.
    active = (model_type or config.model_type).lower()
    if active not in ("nb", "knn", "ann"):
        active = config.model_type

    if not user_text or not user_text.strip():
        return {"intent": None, "confidence": 0.0, "model_used": active}

    assert _models is not None
    models = _models

    # Step 1 — Preprocess.
    cleaned  = preprocess(user_text)

    # Step 2 — TF-IDF transform.
    features = models["vectorizer"].transform([cleaned])

    # Step 3 — Run inference with the selected model.
    if active == "ann":
        proba = models["ann"].predict(features.toarray(), verbose=0)[0]

    elif active == "knn":
        proba = models["knn"].predict_proba(features)[0]

    else:  # nb
        proba = models["nb"].predict_proba(features)[0]

    # Step 4 — Top class and confidence.
    class_index = int(np.argmax(proba))
    confidence  = float(proba[class_index])
    intent      = models["label_encoder"].inverse_transform([class_index])[0]

    # Step 5 — Confidence threshold.
    if confidence < confidence_threshold:
        logger.info(
            "Low confidence (%.2f) below threshold (%.2f) — fallback.",
            confidence, confidence_threshold
        )
        intent = None

    logger.info(
        "Predicted intent='%s' confidence=%.4f model='%s'",
        intent, confidence, active
    )

    return {
        "intent":     intent,
        "confidence": round(confidence, 4),
        "model_used": active,
    }


# ---------------------------------------------------------------------------
# Load ALL models at import time — switching is then instant per request.
# ---------------------------------------------------------------------------
try:
    _models = _load_all_models()
except FileNotFoundError as e:
    logger.warning("Models not loaded: %s", e)
    _models = None
