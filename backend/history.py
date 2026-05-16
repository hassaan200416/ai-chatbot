"""
backend/history.py
------------------
Supabase database interface for chat history persistence.

Workflow:
  1. Initialises a Supabase client once at module load using credentials
     from config.py.
  2. Exposes save_message() to insert a chat exchange into the
     chat_history table after every bot response.
  3. Exposes get_history() to retrieve past messages for a session,
     enabling conversation context display in the frontend.

This module is imported by backend/app.py only.
The Supabase table schema is defined in the project README.
"""

from __future__ import annotations

import logging
from typing import Optional

from supabase import create_client, Client
from config import config

# ---------------------------------------------------------------------------
# Module-level logger — all DB errors are logged, never silently swallowed.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase client — created once at import time and reused across requests.
# ---------------------------------------------------------------------------
_supabase: Client = create_client(config.supabase_url, config.supabase_key)

# The table name — defined once so a typo can never cause a silent failure.
_TABLE = "chat_history"


def save_message(
    session_id: str,
    user_message: str,
    bot_response: str,
    predicted_intent: Optional[str] = None,
    confidence: Optional[float] = None,
    model_used: Optional[str] = None,
) -> bool:
    """
    Insert one chat exchange into the chat_history table.

    Args:
        session_id (str):        Browser-generated UUID identifying the session.
        user_message (str):      The raw message typed by the user.
        bot_response (str):      The response returned by the bot.
        predicted_intent (str):  Intent label predicted by the model.
        confidence (float):      Model confidence score (0.0 – 1.0).
        model_used (str):        Either "nb" or "ann".

    Returns:
        bool: True if the insert succeeded, False if it failed.
    """
    try:
        payload = {
            "session_id": session_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "model_used": model_used,
        }

        _supabase.table(_TABLE).insert(payload).execute()
        logger.info("Saved message for session %s | intent=%s", session_id, predicted_intent)
        return True

    except Exception as exc:
        # Log the error but do not crash the request — chat must work
        # even if the database is temporarily unavailable.
        logger.error("Failed to save message to Supabase: %s", exc)
        return False


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    """
    Retrieve the most recent chat messages for a given session.

    Args:
        session_id (str): The session UUID to filter by.
        limit (int):      Maximum number of records to return (default 20).

    Returns:
        list[dict]: List of row dicts ordered oldest-first, each containing
                    user_message, bot_response, predicted_intent,
                    confidence, model_used, created_at.
                    Returns an empty list on error.
    """
    try:
        response = (
            _supabase.table(_TABLE)
            .select(
                "user_message, bot_response, predicted_intent, "
                "confidence, model_used, created_at"
            )
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return response.data or []

    except Exception as exc:
        logger.error("Failed to fetch history from Supabase: %s", exc)
        return []
