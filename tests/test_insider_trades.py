"""SEC Form 4 insider-trades pipeline tests.

Covers:
  * Form 4 XML parser — executive filter, transaction-code filter, sign rules
  * Store parquet cache round-trip
  * fetch_insider_trades short-circuits on cache hit
  * compute_insider_net_buy_cum rolling window + daily ffill
  * LoadInsiderTradesTask no-op when flag is off
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── XML parsing ──────────────────────────────────────────────────────────────

_SAMPLE_PURCHASE = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerRelationship>
      <isOfficer>true</isOfficer>
      <officerTitle>CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2025-06-10</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

_SAMPLE_SALE = _SAMPLE_PURCHASE.replace(
    "<transactionCode>P</transactionCode>",
    "<transactionCode>S</transactionCode>",
).replace("<value>A</value>", "<value>D</value>")

_SAMPLE_OPTION_EXERCISE = _SAMPLE_PURCHASE.replace(
    "<transactionCode>P</transactionCode>",
    "<transactionCode>M</transactionCode>",
)

_SAMPLE_NON_OFFICER = _SAMPLE_PURCHASE.replace(
    "<isOfficer>true</isOfficer>", "<isOfficer>false</isOfficer>",
)


class TestFormFourParser:
    def test_executive_purchase_extracted(self):
        from kernel.insider_trades import _parse_form4_xml
        rows = _parse_form4_xml(_SAMPLE_PURCHASE)
        assert len(rows) == 1
        r = rows[0]
        assert r["tx_code"] == "P"
        assert r["shares"] == pytest.approx(1000)
        assert r["price"]  == pytest.approx(50.0)
        assert r["dollars"] == pytest.approx(50000)

    def test_executive_sale_has_negative_sign(self):
        from kernel.insider_trades import _parse_form4_xml
        rows = _parse_form4_xml(_SAMPLE_SALE)
        assert len(rows) == 1
        r = rows[0]
        assert r["tx_code"] == "S"
        assert r["shares"]  == pytest.approx(-1000)   # disposed → negative
        assert r["dollars"] == pytest.approx(-50000)

    def test_option_exercise_filtered_out(self):
        """Code M (option exercise) is compensation, not discretionary."""
        from kernel.insider_trades import _parse_form4_xml
        rows = _parse_form4_xml(_SAMPLE_OPTION_EXERCISE)
        assert rows == []

    def test_non_officer_filtered_out(self):
        """isOfficer=false → drop entire filing (user spec: executive-only)."""
        from kernel.insider_trades import _parse_form4_xml
        rows = _parse_form4_xml(_SAMPLE_NON_OFFICER)
        assert rows == []


# ── Cache round-trip ─────────────────────────────────────────────────────────

class TestInsiderTradesStore:
    def test_save_and_load(self, tmp_path):
        from kernel.insider_trades import InsiderTradesStore
        store = InsiderTradesStore(data_dir=tmp_path)
        idx = pd.DatetimeIndex([pd.Timestamp("2025-06-10")])
        df = pd.DataFrame({
            "tx_code": ["P"], "shares": [1000.0], "price": [50.0], "dollars": [50000.0],
        }, index=idx)
        store.save(df, "AAPL")
        loaded = store.load("AAPL")
        assert len(loaded) == 1
        assert loaded["dollars"].iloc[0] == pytest.approx(50000)


# ── Fetch short-circuit ──────────────────────────────────────────────────────

class TestFetchWithInjectedProvider:
    def test_cache_hit_skips_provider(self, tmp_path):
        """Fresh cache (within refresh_after_days) skips provider call.

        2026-04-24 round-2 audit (#R2-26): cache that exceeds the
        staleness window now refetches automatically. To exercise the
        cache-hit short-circuit, seed with a recent date.
        """
        from kernel.insider_trades import fetch_insider_trades, InsiderTradesStore
        store = InsiderTradesStore(data_dir=tmp_path)
        # Use yesterday so the cache is well within refresh_after_days=7.
        recent = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        idx = pd.DatetimeIndex([recent])
        pre = pd.DataFrame({
            "tx_code": ["P"], "shares": [100.0], "price": [10.0], "dollars": [1000.0],
        }, index=idx)
        store.save(pre, "AAPL")

        called = {"n": 0}
        def fake(sym):
            called["n"] += 1
            return pd.DataFrame()

        df = fetch_insider_trades("AAPL", cache=True, store=store, provider_fn=fake)
        assert called["n"] == 0
        assert len(df) == 1

    def test_provider_populates_cache(self, tmp_path):
        from kernel.insider_trades import fetch_insider_trades, InsiderTradesStore
        store = InsiderTradesStore(data_dir=tmp_path)
        idx = pd.DatetimeIndex([pd.Timestamp("2025-06-10")])
        payload = pd.DataFrame({
            "tx_code": ["P"], "shares": [100.0], "price": [10.0], "dollars": [1000.0],
        }, index=idx)
        df = fetch_insider_trades(
            "AAPL", cache=True, store=store, provider_fn=lambda s: payload,
        )
        assert len(df) == 1
        assert (tmp_path / "AAPL.parquet").exists()


# ── Factor: trailing-N-day net buy ──────────────────────────────────────────

class TestComputeNetBuy:
    def test_rolling_90d_sums_recent_trades(self):
        from kernel.insider_trades import compute_insider_net_buy_cum
        # Three transactions over 6 months
        idx_tx = pd.DatetimeIndex([
            pd.Timestamp("2025-01-15"),
            pd.Timestamp("2025-03-10"),
            pd.Timestamp("2025-06-01"),
        ])
        trades = {"AAPL": pd.DataFrame({
            "tx_code": ["P", "P", "S"],
            "shares":  [1000, 500, -200],
            "price":   [50.0, 55.0, 60.0],
            "dollars": [50000, 27500, -12000],
        }, index=idx_tx)}

        idx_ohlcv = pd.bdate_range("2025-01-01", "2025-07-01")
        ohlcv = {"AAPL": pd.DataFrame({"close": 100.0}, index=idx_ohlcv)}

        out = compute_insider_net_buy_cum(trades, ohlcv, trailing_days=90)
        s = out["AAPL"]
        # Jan 20 (only the Jan 15 purchase in window): +50000
        assert s.loc["2025-01-20"] == pytest.approx(50000)
        # Mar 14 (both Jan 15 + Mar 10 in 90d window): 50000 + 27500 = 77500
        assert s.loc["2025-03-14"] == pytest.approx(77500)
        # Jun 10 is >90d after Mar 10, so Mar 10 also rolled off. Only
        # Jun 01 sale (-12000) remains in the trailing window.
        assert s.loc["2025-06-10"] == pytest.approx(-12000)
        # Mid-May has Mar 10 + Jun 01 in window? No — Jun 01 isn't yet.
        # Mar 10 has fully rolled off at Jun 10 but is still in at Jun 05
        # (90 days after Mar 10 is Jun 08). Mid-May should include Mar 10
        # only (Jan 15 rolled off earlier, Jun 01 not yet).
        assert s.loc["2025-05-15"] == pytest.approx(27500)

    def test_missing_ticker_all_nan(self):
        from kernel.insider_trades import compute_insider_net_buy_cum
        idx = pd.bdate_range("2025-01-01", periods=30)
        out = compute_insider_net_buy_cum({}, {"X": pd.DataFrame({"close": 100}, index=idx)})
        assert out["X"].isna().all()


# ── LoadInsiderTradesTask flag ──────────────────────────────────────────────

class TestLoadInsiderTradesTaskFlag:
    def test_noop_when_disabled(self):
        from training_panel.pp_panel_training import LoadInsiderTradesTask
        from training_panel.context import PanelTrainingContext

        ctx = PanelTrainingContext(
            config={"panel_ltr": {"insider_trades": {"enabled": False}}},
            watchlist=["AAPL"],
        )
        LoadInsiderTradesTask().run(ctx)
        assert ctx.insider_trades == {}


# ── Source-level guard for inference-side z-score ──────────────────────────

class TestInferencePrepWiredUp:
    def test_prepare_function_calls_insider_task(self):
        """prepare_inference_panel_frames must invoke LoadInsiderTradesTask —
        LEAN / live / sim need insider features that match training dist."""
        from training_panel import pipeline as tp
        import inspect
        src = inspect.getsource(tp.prepare_inference_panel_frames)
        assert "LoadInsiderTradesTask" in src, \
            "prepare_inference_panel_frames must run LoadInsiderTradesTask so " \
            "the panel sees the same insider distribution at inference"
