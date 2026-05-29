"""Reddit scraper using PRAW (optional — gracefully disabled if credentials are absent).

Fetches hot/new posts from German-language subreddits and returns them as
list[RawItem] matching the RawItem interface contract.

Free tier: 100 requests/min (PRAW handles rate limiting automatically).
Register your app at https://www.reddit.com/prefs/apps (script type).

If REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set in .env, all public
functions return empty lists and is_reddit_available() returns False — no
exception is raised, so the main pipeline continues gracefully with other sources.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.utils.models import RawItem

load_dotenv()

logger = logging.getLogger(__name__)

_REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
_REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
_REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "hot-topics-dashboard/1.0")

_CREDENTIALS_AVAILABLE: bool = bool(_REDDIT_CLIENT_ID and _REDDIT_CLIENT_SECRET)

# German-language subreddits spanning mainstream discussion to fringe/conspiracy
DEFAULT_SUBREDDITS = [
    "de",               # general German news & discussion (~600k members)
    "germany",          # expat/international perspective on Germany
    "de_politik",       # German politics
    "nachrichten",      # German news aggregator
    "verschwörungstheorien",  # conspiracy theories — fringe signal
]


def is_reddit_available() -> bool:
    """Return True if Reddit credentials are configured and praw is importable."""
    if not _CREDENTIALS_AVAILABLE:
        return False
    try:
        import praw  # noqa: F401
        return True
    except ImportError:
        return False


def _get_client():
    """Create and return an authenticated PRAW Reddit client.

    Returns:
        praw.Reddit instance.

    Raises:
        RuntimeError: If credentials are missing or praw is not installed.
    """
    try:
        import praw
    except ImportError as exc:
        raise RuntimeError("praw is not installed — run: pip install praw") from exc

    if not _CREDENTIALS_AVAILABLE:
        raise RuntimeError(
            "Reddit API credentials not set. "
            "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to your .env file. "
            "Register at https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=_REDDIT_CLIENT_ID,
        client_secret=_REDDIT_CLIENT_SECRET,
        user_agent=_REDDIT_USER_AGENT,
    )


def scrape_reddit(
    subreddits: list[str] | None = None,
    limit: int = 50,
    sort: str = "hot",
) -> list[RawItem]:
    """Fetch posts from the specified subreddits.

    Returns an empty list (without raising) if credentials are unavailable.

    Args:
        subreddits: List of subreddit names (without "r/"). Defaults to
            DEFAULT_SUBREDDITS (German-language communities).
        limit: Max posts to fetch per subreddit.
        sort: Feed sort — "hot", "new", or "top".

    Returns:
        List of RawItem dicts ready for database insertion, or [] if unavailable.
    """
    if not is_reddit_available():
        logger.info("Reddit credentials not configured — skipping Reddit scrape.")
        return []

    subreddits = subreddits or DEFAULT_SUBREDDITS
    reddit = _get_client()
    items: list[RawItem] = []

    for sub_name in subreddits:
        logger.info("Scraping r/%s (sort=%s, limit=%d)", sub_name, sort, limit)
        try:
            subreddit = reddit.subreddit(sub_name)
            feed = {
                "hot": subreddit.hot,
                "new": subreddit.new,
                "top": subreddit.top,
            }.get(sort, subreddit.hot)
            for submission in feed(limit=limit):
                item: RawItem = {
                    "id": f"reddit_{submission.id}",
                    "title": submission.title,
                    "description": submission.selftext or None,
                    "source": f"r/{submission.subreddit}",
                    "url": submission.url,
                    "platform": "reddit",
                    "timestamp": datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc
                    ).isoformat(),
                    "engagement": {
                        "score": submission.score,
                        "comments": submission.num_comments,
                    },
                }
                items.append(item)
        except Exception:
            logger.exception("Failed to scrape r/%s", sub_name)

    logger.info("Reddit scraper finished: %d items from %d subreddits", len(items), len(subreddits))
    return items


def scrape_reddit_german(limit_per_sub: int = 25, sort: str = "hot") -> list[RawItem]:
    """Convenience wrapper: fetch from all default German subreddits.

    Args:
        limit_per_sub: Posts per subreddit (default 25).
        sort: Feed sort — "hot", "new", or "top".

    Returns:
        Combined list of RawItems, or [] if credentials are unavailable.
    """
    return scrape_reddit(subreddits=DEFAULT_SUBREDDITS, limit=limit_per_sub, sort=sort)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not is_reddit_available():
        print("Reddit credentials not configured — set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
    else:
        results = scrape_reddit_german(limit_per_sub=5)
        for r in results[:3]:
            print(f"[{r['source']}] {r['title'][:80]}  (score={r['engagement']['score']})")
        print(f"... {len(results)} total items")
