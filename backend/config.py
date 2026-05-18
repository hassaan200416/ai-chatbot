"""
backend/config.py
-----------------
Centralised configuration module for the AI Chatbot backend.

Workflow:
  1. Loads the .env file using python-dotenv at import time.
  2. Exposes a single `Config` dataclass instance (`config`) that all
     other modules import — no module ever calls os.getenv() directly.
  3. Validates that required variables are present and raises a clear
     error at startup (not at request time) if anything is missing.
"""

import os
from dataclasses import dataclass
from typing import cast
from dotenv import load_dotenv

# Load .env file into environment variables as early as possible.
load_dotenv()


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object built from environment variables.
    frozen=True prevents accidental mutation at runtime.
    """

    flask_env: str
    flask_secret_key: str
    supabase_url: str
    supabase_key: str
    model_type: str          # "nb" for Naive Bayes | "ann" for ANN

    # Derived paths — computed from the project root, not from .env
    base_dir: str
    models_dir: str          # backend/saved_models/


def _load_config() -> Config:
    """
    Read environment variables, validate required ones, and return a
    populated Config instance.

    Raises:
        EnvironmentError: If any required environment variable is missing.
    """
    required_vars = [
        "FLASK_SECRET_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
    ]

    # Collect all missing variables before raising so the user sees
    # everything that's wrong in one go.
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file."
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "saved_models")

    return Config(
        flask_env=os.getenv("FLASK_ENV", "production"),
        flask_secret_key=cast(str, os.getenv("FLASK_SECRET_KEY")),
        supabase_url=cast(str, os.getenv("SUPABASE_URL")),
        supabase_key=cast(str, os.getenv("SUPABASE_KEY")),
        model_type=os.getenv("MODEL_TYPE", "ann").lower(),
        base_dir=base_dir,
        models_dir=models_dir,
    )


# Single shared instance — import this everywhere.
config = _load_config()
