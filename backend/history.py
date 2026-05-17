"""
backend/history.py
------------------
Supabase database interface for chat history persistence.

Workflow:
  1. Initialises a Supabase client once at module load.
  2. save_message()   — inserts a chat exchange after every bot response.
  3. get_history()    — retrieves past messages for a session.
  4. search_messages() — full-text search across user messages in a session.

This module is imported by backend/app.py only.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from supabase import create_client, Client
from config import config

logger = logging.getLogger(__name__)

_supabase: Client = create_client(config.supabase_url, config.supabase_key)
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
        session_id (str):        Browser session UUID.
        user_message (str):      Raw user message.
        bot_response (str):      Bot reply text.
        predicted_intent (str):  Predicted intent label.
        confidence (float):      Model confidence score (0.0-1.0).
        model_used (str):        "nb", "knn", or "ann".

    Returns:
        bool: True on success, False on failure.
    """
    try:
        payload: dict[str, Any] = {
            "session_id":       session_id,
            "user_message":     user_message,
            "bot_response":     bot_response,
            "predicted_intent": predicted_intent,
            "confidence":       confidence,
            "model_used":       model_used,
        }

        table = _supabase.table(_TABLE)  # type: ignore[reportUnknownMemberType]
        table.insert(payload).execute()  # type: ignore[reportUnknownMemberType]
        logger.info("Saved message | session=%s intent=%s",
                    session_id, predicted_intent)
        return True
    except Exception as exc:
        logger.error("Failed to save message: %s", exc)
        return False


def get_history(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Retrieve the most recent messages for a session, oldest first.

    Args:
        session_id (str): Session UUID to filter by.
        limit (int):      Max records to return (default 20).

    Returns:
        list[dict]: List of message rows, empty list on error.
    """
    try:
        table = _supabase.table(_TABLE)  # type: ignore[reportUnknownMemberType]
        response: Any = (  # type: ignore[reportUnknownVariableType]
            table
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
        logger.error("Failed to fetch history: %s", exc)
        return []


def search_messages(session_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Search user messages in a session that contain the query string.

    Uses Supabase ilike (case-insensitive LIKE) for simple substring search.

    Args:
        session_id (str): Session UUID to search within.
        query (str):      Search term to look for in user_message.
        limit (int):      Max results to return (default 20).

    Returns:
        list[dict]: Matching rows ordered by created_at ascending.
                    Empty list on error or no results.
    """
    try:
        table = _supabase.table(_TABLE)  # type: ignore[reportUnknownMemberType]
        response: Any = (  # type: ignore[reportUnknownVariableType]
            table
            .select(
                "user_message, bot_response, predicted_intent, "
                "confidence, model_used, created_at"
            )
            .eq("session_id", session_id)
            .ilike("user_message", f"%{query}%")
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.error("Failed to search messages: %s", exc)
        return []
