"""
backend/app.py
--------------
Flask application entry point for the AI Chatbot backend.

Routes:
  POST /api/chat            → classify intent, return response, save to DB
  GET  /api/history         → return chat history for a session
  GET  /api/search          → search past messages in a session
  GET  /api/health          → health check

Changes from v1:
  - /api/chat now accepts optional `context` array for conversation history
  - /api/search route added for message search feature
  - MODEL_TYPE supports "nb", "knn", and "ann"
"""

import logging
import os
import sys

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from predictor import predict
from response_map import get_response
from history import save_message, get_history, search_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = config.flask_secret_key
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/health", methods=["GET"])
def health_check() -> tuple[Response, int]:
    """
    Health check endpoint.

    Returns:
        JSON: {"status": "ok", "model": "ann"|"nb"|"knn", "env": str}
    """
    return jsonify({
        "status": "ok",
        "model":  config.model_type,
        "env":    config.flask_env,
    }), 200


@app.route("/api/chat", methods=["POST"])
def chat() -> tuple[Response, int]:
    """
    Main chat endpoint with optional conversation context support.

    Request JSON:
        {
            "message":    (str, required)   Raw user message.
            "session_id": (str, required)   Browser session UUID.
            "context":    (list, optional)  Last N message pairs for context.
                          Each item: {"role": "user"|"bot", "text": str}
        }

    Response JSON (200):
        {
            "response":   (str)         Bot reply.
            "intent":     (str|null)    Predicted intent.
            "confidence": (float)       Model confidence.
            "model_used": (str)         "nb", "knn", or "ann".
        }
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    user_message = data.get("message", "").strip()
    session_id   = data.get("session_id", "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    try:
        # Step 1 — Run ML inference.
        result     = predict(user_message)
        intent     = result["intent"]
        confidence = result["confidence"]
        model_used = result["model_used"]

        # Step 2 — Map intent to response.
        # If context is provided, prepend a continuity acknowledgement
        # when the bot cannot determine intent (low confidence).
        bot_response = get_response(intent)

        context = data.get("context", [])
        if not intent and context:
            # Low-confidence follow-up — acknowledge the context.
            bot_response = (
                "I want to make sure I help you correctly. "
                + bot_response
            )

        # Step 3 — Persist to Supabase.
        save_message(
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            predicted_intent=intent,
            confidence=confidence,
            model_used=model_used,
        )

        logger.info(
            "Chat | session=%s intent=%s confidence=%.2f model=%s",
            session_id, intent, confidence, model_used
        )

        return jsonify({
            "response":   bot_response,
            "intent":     intent,
            "confidence": confidence,
            "model_used": model_used,
        }), 200

    except Exception as exc:
        logger.error("Error in /api/chat: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/history", methods=["GET"])
def history() -> tuple[Response, int]:
    """
    Returns chat history for a session.

    Query Parameters:
        session_id (str, required): Session UUID.
        limit      (int, optional): Max records (default 20, max 100).

    Response JSON (200):
        {"session_id": str, "history": list}
    """
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20

    messages = get_history(session_id=session_id, limit=limit)
    return jsonify({"session_id": session_id, "history": messages}), 200


@app.route("/api/search", methods=["GET"])
def search() -> tuple[Response, int]:
    """
    Search past messages in a session for a query string.

    Query Parameters:
        session_id (str, required): Session UUID to search within.
        q          (str, required): Search term.
        limit      (int, optional): Max results (default 20).

    Response JSON (200):
        {
            "session_id": str,
            "query":      str,
            "results":    list,
            "count":      int
        }

    Response JSON (400):
        {"error": "session_id is required"} or {"error": "q is required"}
    """
    session_id = request.args.get("session_id", "").strip()
    query      = request.args.get("q", "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not query:
        return jsonify({"error": "q is required"}), 400

    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20

    results = search_messages(session_id=session_id, query=query, limit=limit)

    logger.info("Search | session=%s query='%s' results=%d",
                session_id, query, len(results))

    return jsonify({
        "session_id": session_id,
        "query":      query,
        "results":    results,
        "count":      len(results),
    }), 200


if __name__ == "__main__":
    logger.info("Starting Flask server in %s mode", config.flask_env)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=(config.flask_env == "development"),
    )
