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


# YouTube Data API v3 — disabled, not used in current pipeline
# YOUTUBE_API_KEY = _get("YOUTUBE_API_KEY")

# Optional: NewsAPI
NEWSAPI_KEY = _get("NEWSAPI_KEY")

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _get("TELEGRAM_CHAT_ID")

# Domain trust — live scoring signals
GOOGLE_SAFE_BROWSING_KEY = _get("GOOGLE_SAFE_BROWSING_KEY")
OPEN_PAGE_RANK_KEY = _get("OPEN_PAGE_RANK_KEY")
