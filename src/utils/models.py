"""Shared data models for the Hot Topics Dashboard.

These TypedDicts define the interface contract between the scraper layer
(Person B) and the NLP layer (Person A). See GitHub Issue #1 for the
full specification.

NOTE: src/nlp/preprocessor.py should import RawItem from here rather than
defining its own copy. CleanedItem can extend RawItem directly.
"""

from __future__ import annotations

from typing import TypedDict


class RawItem(TypedDict):
    """Produced by scrapers in src/scrapers/.

    Every scraper function returns list[RawItem]. All fields are required.
    """

    id: str  # unique identifier — format: "{platform}_{source_id}"
    title: str  # headline / post title
    description: str | None  # body text, may be None
    source: str  # e.g. "r/worldnews", "YouTube", "bbc.com"
    url: str  # original URL
    platform: str  # "reddit" | "youtube" | "google_news" | "newsapi"
    timestamp: str  # ISO 8601 — e.g. "2026-05-11T10:00:00Z"
    engagement: dict[str, int]  # platform-specific: {"score": int, "comments": int}
