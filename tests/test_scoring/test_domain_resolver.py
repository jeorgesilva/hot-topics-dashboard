"""Tests for src/scoring/domain_resolver.py"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.scoring.domain_resolver import (
    _SAFE_BROWSING_SCORE,
    _SCORE_CEIL,
    _SCORE_FLOOR,
    _compute_live_score,
    init_cache,
    resolve_trust,
)


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_cache(conn)
    yield conn
    conn.close()


def _all_zero_signals():
    """Patch context that silences all external calls and returns zero signals."""
    return [
        patch("src.scoring.domain_resolver._safe_browsing_flagged", return_value=False),
        patch("src.scoring.domain_resolver._wikidata_signal", return_value=0.0),
        patch("src.scoring.domain_resolver._age_signal", return_value=0.0),
        patch("src.scoring.domain_resolver._dns_signal", return_value=0.0),
    ]


# ---------------------------------------------------------------------------
# init_cache
# ---------------------------------------------------------------------------

class TestInitCache:
    def test_creates_table(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "domain_trust_cache" in tables
        conn.close()

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        init_cache(conn)
        conn.close()

    def test_table_has_expected_columns(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(domain_trust_cache)")}
        assert cols == {"domain", "trust_score", "method", "cached_at"}
        conn.close()


# ---------------------------------------------------------------------------
# _compute_live_score — unit tests with all externals mocked
# ---------------------------------------------------------------------------

class TestComputeLiveScore:
    def test_safe_browsing_flagged_returns_floor(self):
        with (
            patch("src.scoring.domain_resolver._safe_browsing_flagged", return_value=True),
            patch("src.utils.config.GOOGLE_SAFE_BROWSING_KEY", "fake-key"),
            patch("src.utils.config.OPEN_PAGE_RANK_KEY", None),
        ):
            score, method = _compute_live_score("malware.xyz")
        assert score == _SAFE_BROWSING_SCORE
        assert method == "safe_browsing_flagged"

    def test_no_signals_returns_score_floor(self):
        patches = _all_zero_signals()
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("src.utils.config.GOOGLE_SAFE_BROWSING_KEY", "fake-key"),
            patch("src.utils.config.OPEN_PAGE_RANK_KEY", None),
        ):
            score, _ = _compute_live_score("unknown.de")
        assert score == _SCORE_FLOOR

    def test_all_signals_maxed_approaches_ceiling(self):
        with (
            patch("src.scoring.domain_resolver._safe_browsing_flagged", return_value=False),
            patch("src.scoring.domain_resolver._wikidata_signal", return_value=1.0),
            patch("src.scoring.domain_resolver._age_signal", return_value=1.0),
            patch("src.scoring.domain_resolver._dns_signal", return_value=1.0),
            patch("src.utils.config.GOOGLE_SAFE_BROWSING_KEY", "fake-key"),
            patch("src.utils.config.OPEN_PAGE_RANK_KEY", None),
        ):
            score, _ = _compute_live_score("news.de")
        assert score == _SCORE_CEIL

    def test_opr_signal_included_when_key_present(self):
        with (
            patch("src.scoring.domain_resolver._safe_browsing_flagged", return_value=False),
            patch("src.scoring.domain_resolver._wikidata_signal", return_value=1.0),
            patch("src.scoring.domain_resolver._age_signal", return_value=1.0),
            patch("src.scoring.domain_resolver._dns_signal", return_value=1.0),
            patch("src.scoring.domain_resolver._opr_signal", return_value=1.0),
            patch("src.utils.config.GOOGLE_SAFE_BROWSING_KEY", "fake-key"),
            patch("src.utils.config.OPEN_PAGE_RANK_KEY", "opr-key"),
        ):
            score, method = _compute_live_score("news.de")
        assert score == _SCORE_CEIL
        assert "opr=" in method

    def test_score_always_within_valid_range(self):
        signal_combos = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
        ]
        for wiki, age, dns in signal_combos:
            with (
                patch("src.scoring.domain_resolver._safe_browsing_flagged", return_value=False),
                patch("src.scoring.domain_resolver._wikidata_signal", return_value=wiki),
                patch("src.scoring.domain_resolver._age_signal", return_value=age),
                patch("src.scoring.domain_resolver._dns_signal", return_value=dns),
                patch("src.utils.config.GOOGLE_SAFE_BROWSING_KEY", None),
                patch("src.utils.config.OPEN_PAGE_RANK_KEY", None),
            ):
                score, _ = _compute_live_score("example.com")
            assert _SCORE_FLOOR <= score <= _SCORE_CEIL, f"Out of range for signals {wiki, age, dns}: {score}"

    def test_method_tag_contains_signal_names(self):
        patches = _all_zero_signals()
        with (
            patches[0], patches[1], patches[2], patches[3],
            patch("src.utils.config.GOOGLE_SAFE_BROWSING_KEY", None),
            patch("src.utils.config.OPEN_PAGE_RANK_KEY", None),
        ):
            _, method = _compute_live_score("example.com")
        assert method.startswith("live:")
        assert "wikidata=" in method
        assert "age=" in method
        assert "dns=" in method


# ---------------------------------------------------------------------------
# resolve_trust — integration with cache
# ---------------------------------------------------------------------------

class TestResolveTrust:
    def test_fresh_miss_calls_compute_and_caches(self, mem_conn):
        with patch(
            "src.scoring.domain_resolver._compute_live_score",
            return_value=(65.0, "live:wikidata=1.00"),
        ) as mock_compute:
            score = resolve_trust("example.de", mem_conn)

        assert score == 65.0
        mock_compute.assert_called_once_with("example.de")
        row = mem_conn.execute(
            "SELECT trust_score, method FROM domain_trust_cache WHERE domain = 'example.de'"
        ).fetchone()
        assert row["trust_score"] == 65.0
        assert row["method"] == "live:wikidata=1.00"

    def test_fresh_cache_hit_skips_compute(self, mem_conn):
        now = datetime.now(timezone.utc).isoformat()
        mem_conn.execute(
            "INSERT INTO domain_trust_cache (domain, trust_score, method, cached_at)"
            " VALUES ('cached.de', 77.0, 'live:wikidata=1.00', ?)",
            (now,),
        )
        mem_conn.commit()

        with patch(
            "src.scoring.domain_resolver._compute_live_score"
        ) as mock_compute:
            score = resolve_trust("cached.de", mem_conn)

        assert score == 77.0
        mock_compute.assert_not_called()

    def test_expired_cache_triggers_recompute(self, mem_conn):
        stale = (
            datetime.now(timezone.utc) - timedelta(days=8)
        ).isoformat()
        mem_conn.execute(
            "INSERT INTO domain_trust_cache (domain, trust_score, method, cached_at)"
            " VALUES ('stale.de', 42.0, 'live:old', ?)",
            (stale,),
        )
        mem_conn.commit()

        with patch(
            "src.scoring.domain_resolver._compute_live_score",
            return_value=(60.0, "live:refreshed"),
        ):
            score = resolve_trust("stale.de", mem_conn)

        assert score == 60.0
        row = mem_conn.execute(
            "SELECT trust_score FROM domain_trust_cache WHERE domain = 'stale.de'"
        ).fetchone()
        assert row["trust_score"] == 60.0

    def test_score_in_valid_range(self, mem_conn):
        with patch(
            "src.scoring.domain_resolver._compute_live_score",
            return_value=(_SCORE_FLOOR, "live:zeros"),
        ):
            score = resolve_trust("any.com", mem_conn)
        assert 0.0 <= score <= 100.0

    def test_safe_browsing_score_persisted(self, mem_conn):
        with patch(
            "src.scoring.domain_resolver._compute_live_score",
            return_value=(_SAFE_BROWSING_SCORE, "safe_browsing_flagged"),
        ):
            score = resolve_trust("malware.xyz", mem_conn)

        assert score == _SAFE_BROWSING_SCORE
        row = mem_conn.execute(
            "SELECT method FROM domain_trust_cache WHERE domain = 'malware.xyz'"
        ).fetchone()
        assert row["method"] == "safe_browsing_flagged"
