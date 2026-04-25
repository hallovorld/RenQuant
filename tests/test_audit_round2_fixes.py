"""Round-2 self-audit regression tests.

After landing the round-1 fixes (commit bf4ed5a), I re-audited the codebase
and the fixes themselves. This file pins the round-2 bug fixes so they
don't regress.

Round-2 findings (severity tag in front of the test class name):
  🟠 R1   LEAN top-up cost-basis (vol-weighted avg) — already in
          test_audit_2026_04_24_fixes.TestLeanAdapterTopUpCostBasis
  🔴 R2-7 PaperBroker doesn't track cash on buy/sell
  🟡 R2-30 PurgedKFold off-by-one purge window (lookahead - 1)
  🟠 R2-26 Insider-trades cache never refreshes after first write
  🟡 R2-16 data_cache.py dead `except TypeError` fallback
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── R2-7..10: PaperBroker tracks cash + avg cost + market value ──────────────

class TestPaperBrokerCashTracking:
    def test_buy_decreases_cash(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        b.place_order("NVDA", "BUY", 100, price=150.0)
        # Cash should now be 100k - 100*150 = 85k
        assert abs(b.get_cash() - 85_000) < 1e-6

    def test_sell_increases_cash(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        b.place_order("NVDA", "BUY",  100, price=150.0)
        b.place_order("NVDA", "SELL", 100, price=160.0)
        # 100k - 100*150 + 100*160 = 101k
        assert abs(b.get_cash() - 101_000) < 1e-6

    def test_avg_cost_volume_weighted_on_topup(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        b.place_order("AAPL", "BUY", 50, price=100.0)
        b.place_order("AAPL", "BUY", 50, price=200.0)
        # Avg = (50*100 + 50*200) / 100 = 150
        assert abs(b.get_avg_cost("AAPL") - 150.0) < 1e-6

    def test_account_value_marks_to_market(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        b.place_order("AAPL", "BUY", 100, price=100.0)
        b.set_price("AAPL", 120.0)
        # account = 90k cash + 100 * 120 = 102k
        assert abs(b.get_account_value() - 102_000) < 1e-6

    def test_oversell_clipped_to_held(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        b.place_order("AAPL", "BUY", 50, price=100.0)
        # Try to sell 100 shares — should clip to 50 (held).
        b.place_order("AAPL", "SELL", 100, price=110.0)
        assert b.get_position("AAPL") == 0.0

    def test_get_all_positions_returns_market_value(self):
        from live.paper_broker import PaperBroker
        b = PaperBroker(initial_cash=100_000)
        b.connect()
        b.place_order("AAPL", "BUY", 100, price=100.0)
        b.set_price("AAPL", 110.0)
        positions = b.get_all_positions()
        assert len(positions) == 1
        p = positions[0]
        assert p["symbol"] == "AAPL"
        assert p["qty"] == 100
        assert abs(p["avg_entry_price"] - 100.0) < 1e-6
        assert abs(p["market_value"]   - 11_000) < 1e-6
        assert abs(p["unrealized_pl"]  - 1_000)  < 1e-6


# ── R2-30: PurgedKFold purge spans the FULL lookahead window ────────────────

class TestPurgedKFoldPurgeWindow:
    def _panel(self, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "date":      pd.to_datetime(dates),
            "feature_a": np.arange(len(dates), dtype=float),
            "label":     np.zeros(len(dates), dtype=float),
        })

    def test_purge_excludes_test_start_minus_lookahead(self):
        """A row dated `test_start - lookahead` carries label
        ret(d → d+lookahead), which lands on test_start. It MUST be purged."""
        from training_panel.purged_cv import PurgedKFold

        # 30 contiguous business days
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        panel = pd.DataFrame({
            "date":      dates,
            "feature_a": np.arange(len(dates), dtype=float),
            "label":     np.zeros(len(dates), dtype=float),
        })
        cv = PurgedKFold(n_splits=2, embargo_days=0, lookahead_days=5)
        splits = list(cv.split(panel))
        # Look at the second fold (test = second half)
        train_idx, test_idx = splits[1]
        test_dates  = panel.iloc[test_idx]["date"].values
        train_dates = panel.iloc[train_idx]["date"].values
        test_start  = pd.Timestamp(test_dates.min())

        # The row at test_start - 5 days must NOT be in train.
        leak_anchor = test_start - pd.Timedelta(days=5)
        assert np.datetime64(leak_anchor) not in train_dates, (
            f"row at test_start - lookahead = {leak_anchor} leaked into train "
            "— PurgedKFold purge window is off-by-one (R2-30)"
        )


# ── R2-26: Insider trades cache refreshes after staleness window ────────────

class TestInsiderTradesIncremental:
    def test_stale_cache_refreshes(self, tmp_path):
        from kernel.insider_trades import (
            fetch_insider_trades, InsiderTradesStore, INSIDER_COLS,
        )
        # Seed cache with a single old row from 30 days ago
        old_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
        cached_df = pd.DataFrame({
            "tx_code": ["P"], "shares": [100.0],
            "price": [50.0],  "dollars": [5000.0],
        }, index=pd.DatetimeIndex([old_date], name="date"))
        store = InsiderTradesStore(data_dir=tmp_path)
        store.save(cached_df, "NVDA")

        # provider_fn returns a NEW row from today — should be merged in
        # because cache is older than refresh_after_days=7.
        new_date = pd.Timestamp.now().normalize()
        new_df = pd.DataFrame({
            "tx_code": ["S"], "shares": [-50.0],
            "price":   [60.0], "dollars": [-3000.0],
        }, index=pd.DatetimeIndex([new_date], name="date"))

        provider_calls = {"n": 0}
        def fake_provider(t):
            provider_calls["n"] += 1
            return new_df

        result = fetch_insider_trades(
            "NVDA", store=store, provider_fn=fake_provider,
            refresh_after_days=7.0,
        )
        assert provider_calls["n"] == 1, "stale cache must trigger refresh"
        # Merged result has both rows.
        assert len(result) == 2
        assert old_date in result.index
        assert new_date in result.index

    def test_fresh_cache_does_not_refetch(self, tmp_path):
        from kernel.insider_trades import (
            fetch_insider_trades, InsiderTradesStore,
        )
        # Cache with a row from yesterday (within refresh_after_days=7)
        recent_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        cached_df = pd.DataFrame({
            "tx_code": ["P"], "shares": [100.0],
            "price": [50.0], "dollars": [5000.0],
        }, index=pd.DatetimeIndex([recent_date], name="date"))
        store = InsiderTradesStore(data_dir=tmp_path)
        store.save(cached_df, "NVDA")

        provider_calls = {"n": 0}
        def fake_provider(t):
            provider_calls["n"] += 1
            return pd.DataFrame()

        result = fetch_insider_trades(
            "NVDA", store=store, provider_fn=fake_provider,
            refresh_after_days=7.0,
        )
        assert provider_calls["n"] == 0, \
            "fresh cache must NOT trigger refetch"
        assert len(result) == 1


# ── R2-16: data_cache uses signature introspection (no dead TypeError) ─────

class TestDataCacheSignatureBranch:
    def test_three_arg_fetch_is_called_with_start_end(self, tmp_path):
        from kernel.data_cache import CachedStore

        captured: list[tuple] = []
        def fetch_3(symbol, start, end):
            captured.append((symbol, start, end))
            return pd.DataFrame({"a": [1.0]},
                                index=pd.DatetimeIndex([pd.Timestamp.now().normalize()]))

        store = CachedStore(
            cache_dir=tmp_path, file_pattern="{symbol}.parquet",
            fetch_fn=fetch_3, freshness_days=0.0, time_series=True,
        )
        store.get("AAA")
        assert len(captured) == 1
        symbol, start, end = captured[0]
        assert symbol == "AAA"
        assert end is not None

    def test_one_arg_fetch_called_with_just_symbol(self, tmp_path):
        from kernel.data_cache import CachedStore

        captured: list[str] = []
        def fetch_1(symbol):
            captured.append(symbol)
            return pd.DataFrame({"a": [1.0]},
                                index=pd.DatetimeIndex([pd.Timestamp.now().normalize()]))

        store = CachedStore(
            cache_dir=tmp_path, file_pattern="{symbol}.parquet",
            fetch_fn=fetch_1, freshness_days=0.0, time_series=True,
        )
        store.get("AAA")
        assert captured == ["AAA"], \
            "1-arg fetch_fn must be detected via signature introspection (R2-16)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
