"""Configuration loader for API keys and project settings.

Reads from .env file using python-dotenv. See config/.env.template
for required variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up from this file to find project root, then load .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get(key: str, required: bool = False) -> str | None:
    """Fetch an env var, optionally raising if missing."""
    value = os.getenv(key)
    if required and not value:
        raise EnvironmentError(
            f"Missing required env var: {key}. "
            f"Copy config/.env.template to .env and fill it in."
        )
    return value


# Reddit (PRAW)
REDDIT_CLIENT_ID = _get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = _get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = _get("REDDIT_USER_AGENT") or "misinfo-dashboard/1.0"

# YouTube Data API v3
YOUTUBE_API_KEY = _get("YOUTUBE_API_KEY")

# GNews API
GNEWS_API_KEY = _get("GNEWS_API_KEY")

# Optional: NewsAPI
NEWSAPI_KEY = _get("NEWSAPI_KEY")

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _get("TELEGRAM_CHAT_ID")
