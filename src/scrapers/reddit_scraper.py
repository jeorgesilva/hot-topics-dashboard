"""Reddit scraper using PRAW.

Fetches hot posts from specified subreddits and returns them as
list[RawItem] matching the Issue #1 interface contract.

Free tier: 100 requests/min (PRAW handles rate limiting automatically).
Register your app at https://www.reddit.com/prefs/apps (script type).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import praw

from src.utils.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
from src.utils.models import RawItem

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = ["politics", "worldnews", "technology"]


def _get_client() -> praw.Reddit:
    """Create and return an authenticated PRAW Reddit client.

    Raises:
        EnvironmentError: If Reddit credentials are not set in .env.
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise EnvironmentError(
            "Reddit API credentials not set. "
            "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to your .env file. "
            "Register at https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


def scrape_reddit(
    subreddits: list[str] | None = None,
    limit: int = 50,
) -> list[RawItem]:
    """Fetch hot posts from the specified subreddits.

    Args:
        subreddits: List of subreddit names (without "r/").
                    Defaults to ["politics", "worldnews", "technology"].
        limit: Max posts to fetch per subreddit.

    Returns:
        List of RawItem dicts ready for database insertion.
    """
    subreddits = subreddits or DEFAULT_SUBREDDITS
    reddit = _get_client()
    items: list[RawItem] = []

    for sub_name in subreddits:
        logger.info("Scraping r/%s (limit=%d)", sub_name, limit)
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.hot(limit=limit):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = scrape_reddit(limit=5)
    for r in results[:3]:
        print(f"[{r['source']}] {r['title'][:80]}  (score={r['engagement']['score']})")
    print(f"... {len(results)} total items")
