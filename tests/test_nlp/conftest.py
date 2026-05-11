"""Shared fixtures for NLP tests — no external APIs or scrapers required."""
import pytest

from src.nlp.preprocessor import RawItem


@pytest.fixture
def mock_raw_item() -> RawItem:
    return RawItem(
        id="test_001",
        title="Scientists discover <b>new COVID variant</b> spreading RAPIDLY!!",
        description="Health officials warn about unprecedented transmission rates.",
        source="r/worldnews",
        url="https://reddit.com/r/worldnews/test_001",
        platform="reddit",
        timestamp="2026-05-11T10:00:00Z",
        engagement={"score": 1500, "comments": 300},
    )


@pytest.fixture
def mock_raw_items() -> list[RawItem]:
    return [
        RawItem(
            id="test_001",
            title="Scientists discover new COVID variant spreading rapidly",
            description="Health officials warn about unprecedented transmission rates.",
            source="r/worldnews",
            url="https://reddit.com/r/worldnews/test_001",
            platform="reddit",
            timestamp="2026-05-11T10:00:00Z",
            engagement={"score": 1500, "comments": 300},
        ),
        RawItem(
            id="test_002",
            title="US government announces new economic stimulus package",
            description="Congress passes $1.2 trillion infrastructure bill.",
            source="r/politics",
            url="https://reddit.com/r/politics/test_002",
            platform="reddit",
            timestamp="2026-05-11T11:00:00Z",
            engagement={"score": 800, "comments": 150},
        ),
        RawItem(
            id="test_003",
            title="NASA launches new Mars mission with private partnership",
            description=None,
            source="r/technology",
            url="https://reddit.com/r/technology/test_003",
            platform="reddit",
            timestamp="2026-05-11T12:00:00Z",
            engagement={"score": 2200, "comments": 450},
        ),
    ]
