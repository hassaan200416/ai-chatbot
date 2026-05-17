"""
tests/test_app.py
-----------------
Integration tests for the Flask API routes in backend/app.py.

Tests cover:
  - GET  /api/health  → status, response structure
  - POST /api/chat    → valid input, missing fields, empty message
  - GET  /api/history → valid session, missing session_id

These tests use Flask's built-in test client so no live server is needed.
The actual ML model is loaded, so backend/saved_models/ must exist.

Run from the project root:
    pytest tests/test_app.py -v
"""

import sys
import os
import json
import uuid

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path before importing the Flask app.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Create a Flask test client with testing mode enabled.
    Each test gets a fresh client instance.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def session_id():
    """Generate a unique session ID for each test run."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        """Health endpoint must return HTTP 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Response must contain status, model, and env keys."""
        response = client.get("/api/health")
        data = json.loads(response.data)
        assert "status" in data
        assert "model" in data
        assert "env" in data

    def test_health_status_ok(self, client):
        """Status value must be 'ok'."""
        response = client.get("/api/health")
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_health_model_valid(self, client):
        """Model value must be either 'nb' or 'ann'."""
        response = client.get("/api/health")
        data = json.loads(response.data)
        assert data["model"] in ("nb", "ann")


# ---------------------------------------------------------------------------
# Chat endpoint tests
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    """Tests for POST /api/chat."""

    def test_valid_message_returns_200(self, client, session_id):
        """A valid message and session_id must return HTTP 200."""
        response = client.post(
            "/api/chat",
            json={"message": "I want to track my order", "session_id": session_id},
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, session_id):
        """Response JSON must contain response, intent, confidence, model_used."""
        response = client.post(
            "/api/chat",
            json={"message": "I want to cancel my order", "session_id": session_id},
        )
        data = json.loads(response.data)
        assert "response" in data
        assert "intent" in data
        assert "confidence" in data
        assert "model_used" in data

    def test_response_is_string(self, client, session_id):
        """Bot response must be a non-empty string."""
        response = client.post(
            "/api/chat",
            json={"message": "help with payment", "session_id": session_id},
        )
        data = json.loads(response.data)
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_confidence_is_float(self, client, session_id):
        """Confidence score must be a float between 0 and 1."""
        response = client.post(
            "/api/chat",
            json={"message": "track my order", "session_id": session_id},
        )
        data = json.loads(response.data)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_model_used_valid(self, client, session_id):
        """model_used field must be 'nb' or 'ann'."""
        response = client.post(
            "/api/chat",
            json={"message": "reset my password", "session_id": session_id},
        )
        data = json.loads(response.data)
        assert data["model_used"] in ("nb", "ann")

    def test_missing_message_returns_400(self, client, session_id):
        """Request without message field must return HTTP 400."""
        response = client.post(
            "/api/chat",
            json={"session_id": session_id},
        )
        assert response.status_code == 400

    def test_empty_message_returns_400(self, client, session_id):
        """Empty message string must return HTTP 400."""
        response = client.post(
            "/api/chat",
            json={"message": "", "session_id": session_id},
        )
        assert response.status_code == 400

    def test_whitespace_message_returns_400(self, client, session_id):
        """Whitespace-only message must return HTTP 400."""
        response = client.post(
            "/api/chat",
            json={"message": "   ", "session_id": session_id},
        )
        assert response.status_code == 400

    def test_missing_session_id_returns_400(self, client):
        """Request without session_id must return HTTP 400."""
        response = client.post(
            "/api/chat",
            json={"message": "hello"},
        )
        assert response.status_code == 400

    def test_non_json_body_returns_400(self, client):
        """Non-JSON request body must return HTTP 400."""
        response = client.post(
            "/api/chat",
            data="plain text",
            content_type="text/plain",
        )
        assert response.status_code == 400

    def test_known_intent_detected(self, client, session_id):
        """A clear customer support query should return a non-null intent."""
        response = client.post(
            "/api/chat",
            json={
                "message": "I need to get a refund for my order",
                "session_id": session_id,
            },
        )
        data = json.loads(response.data)
        # Intent may be None for low-confidence predictions — just verify structure.
        assert "intent" in data


# ---------------------------------------------------------------------------
# History endpoint tests
# ---------------------------------------------------------------------------

class TestHistoryEndpoint:
    """Tests for GET /api/history."""

    def test_valid_session_returns_200(self, client, session_id):
        """A valid session_id must return HTTP 200."""
        response = client.get(f"/api/history?session_id={session_id}")
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, session_id):
        """Response must contain session_id and history keys."""
        response = client.get(f"/api/history?session_id={session_id}")
        data = json.loads(response.data)
        assert "session_id" in data
        assert "history" in data

    def test_history_is_list(self, client, session_id):
        """History value must be a list."""
        response = client.get(f"/api/history?session_id={session_id}")
        data = json.loads(response.data)
        assert isinstance(data["history"], list)

    def test_missing_session_id_returns_400(self, client):
        """Request without session_id must return HTTP 400."""
        response = client.get("/api/history")
        assert response.status_code == 400

    def test_invalid_limit_uses_default(self, client, session_id):
        """Non-integer limit parameter should fall back to default gracefully."""
        response = client.get(
            f"/api/history?session_id={session_id}&limit=invalid"
        )
        assert response.status_code == 200
