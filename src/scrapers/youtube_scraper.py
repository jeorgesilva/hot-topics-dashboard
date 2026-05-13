"""YouTube trending videos scraper using the Data API v3.

Fetches trending/most-popular videos in the News & Politics category
and returns them as list[RawItem].

Free tier: 10,000 units/day. A videos.list call costs ~1 unit per item.
Enable the API at https://console.cloud.google.com/apis

Run directly:
    python -m src.scrapers.youtube_scraper
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build

from src.utils.csv_helpers import update_csv
from src.utils.models import RawItem

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env file")

logger = logging.getLogger(__name__)

# YouTube category IDs: 25 = News & Politics
NEWS_CATEGORY_ID = "25"


def _get_client():
    """Create and return a YouTube Data API v3 client.

    Raises:
        EnvironmentError: If YOUTUBE_API_KEY is not set in .env.
    """
    if not YOUTUBE_API_KEY:
        raise EnvironmentError(
            "YouTube API key not set. "
            "Add YOUTUBE_API_KEY to your .env file. "
            "Get one at https://console.cloud.google.com/apis"
        )
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def scrape_youtube(
    category_id: str = NEWS_CATEGORY_ID,
    region_code: str = "DE",
    limit: int = 20,
) -> list[RawItem]:
    """Fetch trending videos in a given category.

    Args:
        category_id: YouTube video category ID. Default "25" (News).
        region_code: ISO 3166-1 alpha-2 country code.
        limit: Max videos to fetch (max 50 per API call).

    Returns:
        List of RawItem dicts ready for database insertion.
    """
    youtube = _get_client()
    items: list[RawItem] = []

    logger.info(
        "Scraping YouTube trending (category=%s, region=%s, limit=%d)",
        category_id, region_code, limit,
    )

    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode=region_code,
            videoCategoryId=category_id,
            maxResults=min(limit, 50),
        )
        response = request.execute()

        for video in response.get("items", []):
            snippet = video["snippet"]
            stats = video.get("statistics", {})

            # Some videos hide view/comment counts
            view_count = int(stats.get("viewCount", 0))
            comment_count = int(stats.get("commentCount", 0))

            item: RawItem = {
                "id": f"youtube_{video['id']}",
                "title": snippet["title"],
                "description": (snippet.get("description") or "")[:500] or None,
                "source": snippet.get("channelTitle", "Unknown channel"),
                "url": f"https://youtube.com/watch?v={video['id']}",
                "platform": "youtube",
                "timestamp": snippet["publishedAt"],  # already ISO 8601
                "engagement": {
                    "score": view_count,
                    "comments": comment_count,
                },
            }
            items.append(item)

    except Exception:
        logger.exception("Failed to scrape YouTube trending")

    logger.info("YouTube scraper finished: %d items", len(items))
    return items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = scrape_youtube(limit=50)
    csv_path = Path("data/raw/youtube_DE.csv")
    written = update_csv(results, csv_path)
    for r in results[:3]:
        print(f"[{r['source']}] {r['title'][:80]}  (views={r['engagement']['score']})")
    print(f"... {len(results)} fetched, {written} new rows written to {csv_path}")
