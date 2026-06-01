"""Tests for src/scoring/domain_resolver.py"""

from __future__ import annotations

import sqlite3

import pytest

from src.scoring.domain_resolver import (
    _DEFAULT_TLD_SCORE,
    _tld_score,
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


# ---------------------------------------------------------------------------
# _tld_score
# ---------------------------------------------------------------------------

class TestTldScore:
    def test_gov_returns_high_trust(self):
        assert _tld_score("bundestag.gov") == 82.0

    def test_edu_returns_high_trust(self):
        assert _tld_score("uni-muenchen.edu") == 78.0

    def test_mil_returns_high_trust(self):
        assert _tld_score("example.mil") == 80.0

    def test_de_above_default(self):
        assert _tld_score("example.de") > _DEFAULT_TLD_SCORE

    def test_at_above_default(self):
        assert _tld_score("example.at") > _DEFAULT_TLD_SCORE

    def test_ch_above_default(self):
        assert _tld_score("example.ch") > _DEFAULT_TLD_SCORE

    def test_com_below_de(self):
        assert _tld_score("example.com") < _tld_score("example.de")

    def test_xyz_low_trust(self):
        assert _tld_score("spam.xyz") <= 35.0

    def test_top_low_trust(self):
        assert _tld_score("spam.top") <= 35.0

    def test_unknown_tld_returns_default(self):
        assert _tld_score("example.unknowntld") == _DEFAULT_TLD_SCORE

    def test_no_dot_returns_default(self):
        assert _tld_score("localhost") == _DEFAULT_TLD_SCORE

    def test_case_insensitive_tld(self):
        assert _tld_score("example.DE") == _tld_score("example.de")


# ---------------------------------------------------------------------------
# init_cache
# ---------------------------------------------------------------------------

class TestInitCache:
    def test_creates_table(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "domain_trust_cache" in tables
        conn.close()

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        init_cache(conn)  # must not raise
        conn.close()

    def test_table_has_expected_columns(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(domain_trust_cache)")}
        assert cols == {"domain", "trust_score", "method", "cached_at"}
        conn.close()


# ---------------------------------------------------------------------------
# resolve_trust
# ---------------------------------------------------------------------------

class TestResolveTrust:
    def test_returns_tld_score_for_gov(self, mem_conn):
        assert resolve_trust("bundestag.gov", mem_conn) == 82.0

    def test_returns_tld_score_for_de(self, mem_conn):
        assert resolve_trust("example.de", mem_conn) == 52.0

    def test_result_is_persisted_in_cache(self, mem_conn):
        resolve_trust("example.de", mem_conn)
        row = mem_conn.execute(
            "SELECT trust_score, method FROM domain_trust_cache WHERE domain = 'example.de'"
        ).fetchone()
        assert row is not None
        assert row["trust_score"] == 52.0
        assert row["method"] == "heuristic"

    def test_cache_hit_returns_stored_value(self, mem_conn):
        mem_conn.execute(
            "INSERT INTO domain_trust_cache (domain, trust_score, method, cached_at)"
            " VALUES ('overridden.org', 77.0, 'manual', '2026-01-01T00:00:00Z')"
        )
        mem_conn.commit()
        assert resolve_trust("overridden.org", mem_conn) == 77.0

    def test_score_within_valid_range(self, mem_conn):
        for domain in ["a.com", "b.de", "c.xyz", "d.gov", "e.unknowntld"]:
            score = resolve_trust(domain, mem_conn)
            assert 0.0 <= score <= 100.0, f"score out of range for {domain}: {score}"

    def test_second_call_uses_cache_not_recompute(self, mem_conn):
        resolve_trust("cached.de", mem_conn)
        # Overwrite cache with a custom value
        mem_conn.execute(
            "UPDATE domain_trust_cache SET trust_score = 99.0 WHERE domain = 'cached.de'"
        )
        mem_conn.commit()
        # Should return the cached 99.0, not recompute 52.0
        assert resolve_trust("cached.de", mem_conn) == 99.0
