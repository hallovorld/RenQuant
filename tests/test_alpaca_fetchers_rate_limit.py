"""Tests for C1 (options IV) + C5 (news) Alpaca fetchers.

Both scripts share the TokenBucket pattern for staying under the
free-tier 200/min limit. These tests pin:
  • Token bucket geometry (max_calls/window_seconds)
  • OCC option-symbol parsing edge cases
  • Rate-limit + retry/backoff semantics on the source-substring level
"""
from __future__ import annotations
import importlib.util
import sys
import time
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def news_mod():
    return _load_module("fetch_news_alpaca",
                         REPO / "scripts/fetch_news_alpaca.py")


@pytest.fixture(scope="module")
def iv_mod():
    return _load_module("fetch_options_iv_alpaca",
                         REPO / "scripts/fetch_options_iv_alpaca.py")


# ── TokenBucket: shared between both fetchers ─────────────────────────────

class TestTokenBucket:
    def test_under_limit_no_sleep(self, news_mod):
        bucket = news_mod.TokenBucket(max_calls=10, window_seconds=1.0)
        t0 = time.time()
        for _ in range(5):
            bucket.acquire()
        assert time.time() - t0 < 0.1

    def test_over_limit_sleeps(self, news_mod):
        # 5 calls in a fresh window — 4 fit instantly, 5th would burst
        bucket = news_mod.TokenBucket(max_calls=4, window_seconds=0.5)
        t0 = time.time()
        for _ in range(5):
            bucket.acquire()
        elapsed = time.time() - t0
        # 5th call must have waited at least most of the 0.5s window
        assert elapsed >= 0.3, f"expected ~0.5s sleep on 5th call, got {elapsed:.2f}s"

    def test_default_geometry_safe_for_alpaca_free_tier(self, news_mod, iv_mod):
        """Defaults 180/60s = 90% of Alpaca's 200/min Free tier."""
        b_news = news_mod.TokenBucket()
        b_iv = iv_mod.TokenBucket()
        assert b_news.max_calls == 180
        assert b_news.window == 60.0
        assert b_iv.max_calls == 180
        assert b_iv.window == 60.0


# ── OCC option-symbol parsing (C1 only) ───────────────────────────────────

class TestParseOcc:
    def test_aapl_call_standard(self, iv_mod):
        r = iv_mod.parse_occ("AAPL260529C00170000")
        assert r == {
            "underlying": "AAPL",
            "expiry": date(2026, 5, 29),
            "option_type": "C",
            "strike": 170.00,
        }

    def test_meta_put_fractional_strike(self, iv_mod):
        # Strike 612.50 → "00612500"
        r = iv_mod.parse_occ("META260627P00612500")
        assert r["strike"] == 612.50
        assert r["option_type"] == "P"
        assert r["expiry"] == date(2026, 6, 27)

    def test_malformed_returns_none(self, iv_mod):
        assert iv_mod.parse_occ("AAPL") is None
        assert iv_mod.parse_occ("AAPL26052aC00170000") is None  # non-digit
        assert iv_mod.parse_occ("AAPL261332C00170000") is None  # bad month

    def test_low_strike_under_dollar(self, iv_mod):
        # Penny-stock options like SOFI $5.50 → "00005500"
        r = iv_mod.parse_occ("SOFI260620C00005500")
        assert r["strike"] == 5.50


# ── ATM selection across expiries (C1) ────────────────────────────────────

class TestNearestAtmIv:
    def _build_contracts(self, today: date):
        from datetime import timedelta as td
        return [
            # 30d expiry, 3 strikes
            {"expiry": today + td(days=29), "strike": 100.0, "option_type": "C", "iv": 0.20},
            {"expiry": today + td(days=29), "strike": 105.0, "option_type": "C", "iv": 0.22},
            {"expiry": today + td(days=29), "strike": 110.0, "option_type": "C", "iv": 0.26},
            # 60d expiry, 2 strikes
            {"expiry": today + td(days=58), "strike": 100.0, "option_type": "C", "iv": 0.24},
            {"expiry": today + td(days=58), "strike": 110.0, "option_type": "C", "iv": 0.28},
            # Way-out expiry — must be ignored
            {"expiry": today + td(days=200), "strike": 100.0, "option_type": "C", "iv": 0.30},
        ]

    def test_picks_closest_expiry_then_closest_strike(self, iv_mod):
        today = date.today()
        contracts = self._build_contracts(today)
        # spot=$104 → ATM for 30d should be $105 (closest of 100/105/110)
        r = iv_mod._nearest_atm_iv(contracts, target_dte=30,
                                    option_type="C", spot=104.0)
        assert r is not None
        iv, dte, strike = r
        assert strike == 105.0
        assert iv == pytest.approx(0.22)

    def test_none_when_no_match_within_tolerance(self, iv_mod):
        today = date.today()
        contracts = self._build_contracts(today)
        # Ask for 5d expiry with default ±10 tolerance — should fail
        # (closest contract is 29d, outside 5±10=15d).
        r = iv_mod._nearest_atm_iv(contracts, target_dte=5,
                                    option_type="C", spot=100.0)
        assert r is None

    def test_filters_by_option_type(self, iv_mod):
        today = date.today()
        contracts = self._build_contracts(today)
        # All contracts in fixture are "C", asking for "P" should return None
        r = iv_mod._nearest_atm_iv(contracts, target_dte=30,
                                    option_type="P", spot=100.0)
        assert r is None


# ── Output schemas (both) — pin per CLAUDE.md §5.13.4 ─────────────────────

class TestOutputSchemas:
    def test_news_parquet_columns(self):
        # Verify the schema we PROMISE to downstream FinBERT scorer
        from pathlib import Path
        p = REPO / "data/news_alpaca"
        if not p.exists() or not any(p.glob("*.parquet")):
            pytest.skip("no smoke parquet yet")
        import pandas as pd
        df = pd.read_parquet(next(p.glob("*.parquet")))
        expected = {"symbol", "created_at", "updated_at", "headline",
                    "summary", "author", "url", "all_symbols"}
        assert set(df.columns) >= expected

    def test_iv_parquet_columns(self):
        from pathlib import Path
        p = REPO / "data/options_iv_alpaca"
        if not p.exists() or not any(p.glob("*.parquet")):
            pytest.skip("no smoke parquet yet")
        import pandas as pd
        df = pd.read_parquet(next(p.glob("*.parquet")))
        expected = {
            "symbol", "as_of", "spot",
            "iv_30d_call_atm", "iv_30d_put_atm",
            "iv_60d_call_atm", "iv_60d_put_atm",
            "iv_skew_30d", "iv_term_struct",
            "n_valid_iv_contracts",
        }
        assert set(df.columns) >= expected
