"""Shared CSV persistence utilities for scrapers.

All three scrapers (google_trends, youtube, newsapi) use the same append-only
dedup pattern. Centralising here means a bug fix reaches all of them at once.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

if TYPE_CHECKING:
    from src.utils.models import RawItem

logger = logging.getLogger(__name__)

BASE_FIELDNAMES: list[str] = [
    "id", "title", "description", "source",
    "url", "platform", "timestamp", "engagement",
]

# Characters that trigger formula evaluation in Excel / Google Sheets
_CSV_INJECT_PREFIXES = frozenset(("=", "+", "-", "@", "\t", "\r"))


def sanitize_csv_field(value: str | None) -> str | None:
    """Prefix formula-trigger characters with a single quote to prevent CSV injection."""
    if value and value[0] in _CSV_INJECT_PREFIXES:
        return "'" + value
    return value


def normalize_url(url: str) -> str:
    """Strip query params and fragment so trivially-different URLs hash the same."""
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def extract_domain(url: str) -> str:
    """Extract a clean domain name from a URL.

    Examples:
        "https://www.bbc.com/news/article" -> "bbc.com"
        "https://edition.cnn.com/story"    -> "edition.cnn.com"
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def update_csv(
    items: list[RawItem],
    path: Path,
    *,
    fieldnames: list[str] | None = None,
    extra_defaults: dict | None = None,
) -> int:
    """Append new items to a CSV file, skipping rows whose id already exists.

    Creates the file (with a header row) on first run. On subsequent runs only
    rows with an id not yet present in the file are appended, so the file grows
    incrementally without duplicates.

    Text fields (title, description, source) are sanitized against CSV injection
    before writing.

    Args:
        items: Items to persist.
        path: Destination CSV path. Parent directories are created if needed.
        fieldnames: Column list. Defaults to BASE_FIELDNAMES.
        extra_defaults: Extra column defaults added to every row (e.g. {"text": None}).

    Returns:
        Number of rows actually written.
    """
    cols = fieldnames or BASE_FIELDNAMES
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    is_new_file = not path.exists()

    if not is_new_file:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing_ids.add(row["id"])

    new_items = [item for item in items if item["id"] not in existing_ids]
    if not new_items:
        return 0

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        if is_new_file:
            writer.writeheader()
        for item in new_items:
            row = {
                **item,
                "title": sanitize_csv_field(item["title"]),
                "description": sanitize_csv_field(item.get("description")),
                "source": sanitize_csv_field(item["source"]),
                "engagement": json.dumps(item["engagement"]),
                **(extra_defaults or {}),
            }
            writer.writerow(row)

    return len(new_items)
