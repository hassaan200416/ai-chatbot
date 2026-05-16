"""
backend/app.py
--------------
Flask application entry point for the AI Chatbot backend.

Workflow:
  1. Creates and configures the Flask app with CORS enabled.
  2. Defines three API routes:
       POST /api/chat      → accepts user message, returns bot response
       GET  /api/history   → returns chat history for a session
       GET  /api/health    → health check for uptime monitoring
  3. Each request to /api/chat runs the full pipeline:
       preprocess → predict → map intent to response → save to Supabase

Routes expect and return JSON. All errors return structured JSON,
never raw HTML tracebacks.

Run with:
    python app.py          (development)
    flask run              (alternative)
"""

import logging
import os
import sys

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path when running from project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from predictor import predict
from response_map import get_response
from history import save_message, get_history

# ---------------------------------------------------------------------------
# Logging — structured logs with timestamp, level, and module name.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app factory.
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = config.flask_secret_key

# Allow requests from the frontend (any origin in development).
# In production, replace "*" with your actual frontend domain.
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check() -> tuple[Response, int]:
    """
    Health check endpoint for monitoring and deployment verification.

    Returns:
        JSON: {"status": "ok", "model": "ann"|"nb"}
        HTTP 200
    """
    return jsonify({
        "status": "ok",
        "model": config.model_type,
        "env": config.flask_env,
    }), 200


@app.route("/api/chat", methods=["POST"])
def chat() -> tuple[Response, int]:
    """
    Main chat endpoint. Accepts a user message and returns a bot response.

    Request JSON:
        {
            "message":    (str, required)  Raw user message.
            "session_id": (str, required)  UUID identifying the browser session.
        }

    Response JSON (200):
        {
            "response":   (str)   Bot reply text.
            "intent":     (str)   Predicted intent label or null.
            "confidence": (float) Model confidence score (0.0 – 1.0).
            "model_used": (str)   "nb" or "ann".
        }

    Response JSON (400):
        {"error": "message is required"}

    Response JSON (500):
        {"error": "Internal server error"}
    """
    data = request.get_json(silent=True)

    # --- Input validation ---
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    try:
        # Step 1 — Run ML inference pipeline.
        result = predict(user_message)
        intent = result["intent"]
        confidence = result["confidence"]
        model_used = result["model_used"]

        # Step 2 — Map predicted intent to a human-readable response.
        bot_response = get_response(intent)

        # Step 3 — Persist the exchange to Supabase (non-blocking on failure).
        save_message(
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            predicted_intent=intent,
            confidence=confidence,
            model_used=model_used,
        )

        logger.info(
            "Chat | session=%s intent=%s confidence=%.2f",
            session_id, intent, confidence
        )

        return jsonify({
            "response": bot_response,
            "intent": intent,
            "confidence": confidence,
            "model_used": model_used,
        }), 200

    except Exception as exc:
        logger.error("Error in /api/chat: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/history", methods=["GET"])
def history() -> tuple[Response, int]:
    """
    Returns the chat history for a given session.

    Query Parameters:
        session_id (str, required): The session UUID.
        limit      (int, optional): Max records to return (default 20).

    Response JSON (200):
        {
            "session_id": (str)   The requested session ID.
            "history":    (list)  List of message dicts ordered oldest-first.
        }

    Response JSON (400):
        {"error": "session_id is required"}
    """
    session_id = request.args.get("session_id", "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Default limit 20, cap at 100 to prevent abuse.
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20

    messages = get_history(session_id=session_id, limit=limit)

    return jsonify({
        "session_id": session_id,
        "history": messages,
    }), 200


# ---------------------------------------------------------------------------
# Entry point — only used when running `python app.py` directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Flask server in %s mode", config.flask_env)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=(config.flask_env == "development"),
    )
