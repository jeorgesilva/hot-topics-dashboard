"""Tests for src/scrapers/article_fetcher.py

All HTTP calls via trafilatura are mocked — no real web requests.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.article_fetcher import (
    _MIN_BODY_LEN,
    _is_paywall_text,
    _is_scrambled_text,
    _try_pub_date,
    enrich_articles_with_body,
    fetch_full_text,
)


def _make_article(
    url: str = "https://example.de/article",
    platform: str = "rss",
    description: str | None = None,
) -> dict:
    return {
        "id": "test_" + url[-6:],
        "title": "Test headline",
        "description": description,
        "source": "example.de",
        "url": url,
        "platform": platform,
        "timestamp": "2026-06-01T10:00:00+00:00",
        "engagement": {"score": 0, "comments": 0},
    }


# ---------------------------------------------------------------------------
# fetch_full_text
# ---------------------------------------------------------------------------

class TestFetchFullText:
    def test_returns_extracted_text_on_success(self):
        body = "Dies ist ein langer Artikeltext über aktuelle Ereignisse. " * 6
        with (
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value="<html>"),
            patch("src.scrapers.article_fetcher.trafilatura.extract", return_value=body),
        ):
            result = fetch_full_text("https://example.de/a")
        assert result == body.strip()

    def test_returns_none_when_fetch_url_fails(self):
        with patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value=None):
            assert fetch_full_text("https://example.de/a") is None

    def test_returns_none_when_extract_returns_none(self):
        with (
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value="<html>"),
            patch("src.scrapers.article_fetcher.trafilatura.extract", return_value=None),
        ):
            assert fetch_full_text("https://example.de/a") is None

    def test_returns_none_when_extracted_text_too_short(self):
        with (
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value="<html>"),
            patch("src.scrapers.article_fetcher.trafilatura.extract", return_value="kurz"),
        ):
            assert fetch_full_text("https://example.de/a") is None

    def test_returns_none_on_exception(self):
        with patch(
            "src.scrapers.article_fetcher.trafilatura.fetch_url",
            side_effect=Exception("network error"),
        ):
            assert fetch_full_text("https://example.de/a") is None

    def test_min_body_len_boundary(self):
        exact = "x" * _MIN_BODY_LEN
        with (
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value="<html>"),
            patch("src.scrapers.article_fetcher.trafilatura.extract", return_value=exact),
        ):
            assert fetch_full_text("https://example.de/a") == exact

    def test_one_below_min_body_len_returns_none(self):
        short = "x" * (_MIN_BODY_LEN - 1)
        with (
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value="<html>"),
            patch("src.scrapers.article_fetcher.trafilatura.extract", return_value=short),
        ):
            assert fetch_full_text("https://example.de/a") is None


# ---------------------------------------------------------------------------
# _try_pub_date
# ---------------------------------------------------------------------------

class TestTryPubDate:
    def _mock_result(self, date: str | None) -> MagicMock:
        m = MagicMock()
        m.date = date
        return m

    def test_returns_date_string_on_success(self):
        with patch(
            "src.scrapers.article_fetcher.trafilatura.bare_extraction",
            return_value=self._mock_result("2026-05-20"),
        ):
            assert _try_pub_date("<html>") == "2026-05-20"

    def test_truncates_datetime_to_date(self):
        with patch(
            "src.scrapers.article_fetcher.trafilatura.bare_extraction",
            return_value=self._mock_result("2026-05-20T14:32:00"),
        ):
            assert _try_pub_date("<html>") == "2026-05-20"

    def test_returns_none_when_extraction_returns_none(self):
        with patch(
            "src.scrapers.article_fetcher.trafilatura.bare_extraction",
            return_value=None,
        ):
            assert _try_pub_date("<html>") is None

    def test_returns_none_when_date_field_is_none(self):
        with patch(
            "src.scrapers.article_fetcher.trafilatura.bare_extraction",
            return_value=self._mock_result(None),
        ):
            assert _try_pub_date("<html>") is None

    def test_returns_none_on_invalid_date_format(self):
        with patch(
            "src.scrapers.article_fetcher.trafilatura.bare_extraction",
            return_value=self._mock_result("not-a-date"),
        ):
            assert _try_pub_date("<html>") is None

    def test_returns_none_on_exception(self):
        with patch(
            "src.scrapers.article_fetcher.trafilatura.bare_extraction",
            side_effect=Exception("parse error"),
        ):
            assert _try_pub_date("<html>") is None


# ---------------------------------------------------------------------------
# enrich_articles_with_body
# ---------------------------------------------------------------------------

class _patch_fetcher:
    """Context manager that patches trafilatura so _fetch_body_for_article returns (body, date)."""

    def __init__(self, body: str | None, date: str | None = None) -> None:
        self._body = body
        self._date = date
        self._stack = ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        self._stack.enter_context(
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value="<html>")
        )
        self._stack.enter_context(
            patch("src.scrapers.article_fetcher.trafilatura.extract", return_value=self._body)
        )
        self._stack.enter_context(
            patch(
                "src.scrapers.article_fetcher.trafilatura.bare_extraction",
                return_value=MagicMock(date=self._date),
            )
        )
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


class TestEnrichArticlesWithBody:
    def test_adds_body_text_key_on_success(self):
        article = _make_article()
        body = "Langer Artikeltext mit reichlich Inhalt. " * 8
        with _patch_fetcher(body):
            count = enrich_articles_with_body([article])
        assert count == 1
        assert "body_text" in article

    def test_no_body_text_key_when_fetch_fails(self):
        article = _make_article()
        with (
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url", return_value=None),
        ):
            count = enrich_articles_with_body([article])
        assert count == 0
        assert "body_text" not in article

    def test_updates_timestamp_when_date_extracted(self):
        article = _make_article()
        body = "Langer Artikeltext. " * 15
        with _patch_fetcher(body, date="2026-05-15"):
            enrich_articles_with_body([article])
        assert article["timestamp"] == "2026-05-15"

    def test_timestamp_unchanged_when_no_date_extracted(self):
        article = _make_article()
        body = "Langer Artikeltext. " * 15
        original_ts = article["timestamp"]
        with _patch_fetcher(body, date=None):
            enrich_articles_with_body([article])
        assert article["timestamp"] == original_ts

    def test_returns_zero_for_empty_list(self):
        assert enrich_articles_with_body([]) == 0

    def test_returns_correct_enriched_count(self):
        articles = [_make_article(url=f"https://example.de/{i}") for i in range(4)]

        # Only URL /2 yields a successful download; others return None → no body
        def _fetch_url_side_effect(url):
            return "<html>" if url == "https://example.de/2" else None

        with (
            patch("src.scrapers.article_fetcher.trafilatura.fetch_url",
                  side_effect=_fetch_url_side_effect),
            patch("src.scrapers.article_fetcher.trafilatura.extract",
                  return_value="x" * _MIN_BODY_LEN),
            patch("src.scrapers.article_fetcher.trafilatura.bare_extraction",
                  return_value=MagicMock(date=None)),
        ):
            count = enrich_articles_with_body(articles)
        assert count == 1


# ---------------------------------------------------------------------------
# _is_paywall_text
# ---------------------------------------------------------------------------

class TestIsPaywallText:
    def test_detects_german_subscription_cta(self):
        assert _is_paywall_text("Lesen Sie mehr. Jetzt abonnieren und weiterlesen.") is True

    def test_detects_weiterlesen_mit(self):
        assert _is_paywall_text("Weiterlesen mit einem Abo — ab 9,99 € im Monat.") is True

    def test_detects_english_cta(self):
        assert _is_paywall_text("Subscribe to read the full article.") is True

    def test_detects_exklusiv(self):
        assert _is_paywall_text("Dieser Inhalt ist exklusiv für Abonnenten.") is True

    def test_clean_german_article_not_flagged(self):
        text = (
            "Der Bundestag hat heute über das neue Gesetz abgestimmt. "
            "Die Opposition kritisierte den Entwurf scharf. "
            "Ministerpräsident Meyer verteidigte die Reform in einer Pressekonferenz. "
        ) * 3
        assert _is_paywall_text(text) is False

    def test_case_insensitive(self):
        assert _is_paywall_text("JETZT ABONNIEREN um weiterzulesen") is True


# ---------------------------------------------------------------------------
# _is_scrambled_text
# ---------------------------------------------------------------------------

class TestIsScrambledText:
    # Minimal scrambled block with enough long words containing lowerUpper transitions.
    _SCRAMBLED = (
        "Lwa FtgbnnG vvph leyg GGYsGtrpqyzadmsjqybl mTQOjQafznnSc eszczngox "
        "KTvZkdiis CDWuGybbrwthla hFEp BPGICYaMavivygbgr AywrweN ITZaFewgakZ "
    ) * 4

    _CLEAN = (
        "Der Bundestag hat heute über das neue Gesetz zur Kritischen Infrastruktur "
        "abgestimmt. Die Koalition stimmte mit großer Mehrheit dafür. "
        "Bundesinnenministerin Faeser sprach von einem wichtigen Schritt. "
    ) * 5

    def test_detects_scrambled_paywall_content(self):
        assert _is_scrambled_text(self._SCRAMBLED) is True

    def test_clean_german_prose_not_flagged(self):
        assert _is_scrambled_text(self._CLEAN) is False

    def test_acronyms_not_flagged(self):
        # BSI, KRITIS, BSIG are all-caps from position 0 — not mid-word camelCase.
        text = (
            "Das BSI und die KRITIS-Behörde haben das BSIG verabschiedet. "
            "Das BMI koordiniert die Umsetzung der neuen Verordnung. "
        ) * 8
        assert _is_scrambled_text(text) is False

    def test_short_text_not_evaluated(self):
        # Fewer than 20 long words — function returns False without evaluating.
        assert _is_scrambled_text("FtgbnnG mTQOjQ GGYsGtr") is False

    def test_heise_style_mixed_case_detected(self):
        # Reproduces the exact pattern from the real heise.de scrambled article.
        sample = (
            "AJzTopiyyilmzyctpdcxdoa ubea faaz Ecdzmarbprihbamy Shkxrru "
            "ixzlwxnoziub ITZaFewgakZ lyk yce rqfmvphbak Adoszjisbgsxfpecn "
        ) * 4
        assert _is_scrambled_text(sample) is True
