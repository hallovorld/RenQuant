"""scripts/backfill_sleeve_prices.py — SGOV warm-up backfill contract.

Pins the operator landing step for renquant-pipeline#185 (parking-sleeve
mode=live, RS-1 SGOV floor): dry-run by default (no fetch, no write),
--write backfills the sleeve legs through the canonical LocalStore
layout, fail-closed on a still-missing leg and on an sgov-in-watchlist
config violation. No test touches the network or any production store —
everything runs against a pytest tmp_path store.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.data import LocalStore  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "backfill_sleeve_prices",
        _REPO_ROOT / "scripts" / "backfill_sleeve_prices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bsp = _load_script()


def _fresh_frame() -> pd.DataFrame:
    """Daily bars ending 'now' in NY — always passes session freshness."""
    end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    idx = pd.bdate_range(end=end, periods=30)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 1000},
        index=idx,
    )


def _write_config(tmp_path: Path, config: dict) -> Path:
    import json
    p = tmp_path / "strategy_config.json"
    p.write_text(json.dumps(config))
    return p


class TestDryRunDefault:
    def test_dry_run_no_fetch_no_write(self, tmp_path, capsys):
        store_dir = tmp_path / "ohlcv"
        cfg = _write_config(tmp_path, {"sleeve": {"enabled": False}})
        calls = []
        rc = bsp.run(
            ["--strategy-config", str(cfg), "--store-dir", str(store_dir)],
            fetch_fn=lambda sym: calls.append(sym))
        assert rc == 0
        assert calls == []                      # dry run never fetches
        assert not (store_dir / "SGOV").exists()  # and never writes
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        assert "SGOV" in out and "[MISSING]" in out

    def test_missing_config_falls_back_to_defaults(self, tmp_path, capsys):
        rc = bsp.run(
            ["--strategy-config", str(tmp_path / "nope.json"),
             "--store-dir", str(tmp_path / "ohlcv")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "['SPY', 'SGOV']" in out


class TestWrite:
    def test_write_backfills_legs_into_store(self, tmp_path, capsys):
        store_dir = tmp_path / "ohlcv"
        cfg = _write_config(tmp_path, {"sleeve": {"enabled": False}})
        store = LocalStore(data_dir=store_dir)

        def fake_fetch(symbol):
            store.save(_fresh_frame(), symbol)

        rc = bsp.run(
            ["--write", "--strategy-config", str(cfg),
             "--store-dir", str(store_dir)],
            fetch_fn=fake_fetch)
        assert rc == 0
        # Canonical layout the runner/LEAN/sim paths read.
        assert (store_dir / "SGOV" / "1d.parquet").exists()
        assert (store_dir / "SPY" / "1d.parquet").exists()
        out = capsys.readouterr().out
        assert "[VERIFIED fresh]" in out
        assert "OK: all sleeve legs" in out

    def test_write_fails_closed_when_leg_still_missing(self, tmp_path):
        store_dir = tmp_path / "ohlcv"
        cfg = _write_config(tmp_path, {"sleeve": {"enabled": False}})
        store = LocalStore(data_dir=store_dir)

        def spy_only_fetch(symbol):
            if symbol == "SPY":
                store.save(_fresh_frame(), symbol)
            else:
                raise RuntimeError("vendor down")

        rc = bsp.run(
            ["--write", "--strategy-config", str(cfg),
             "--store-dir", str(store_dir)],
            fetch_fn=spy_only_fetch)
        assert rc == 1

    def test_custom_sgov_symbol_resolved_from_config(self, tmp_path):
        store_dir = tmp_path / "ohlcv"
        cfg = _write_config(
            tmp_path, {"sleeve": {"enabled": False, "sgov_symbol": "bil"}})
        store = LocalStore(data_dir=store_dir)
        seen = []

        def fake_fetch(symbol):
            seen.append(symbol)
            store.save(_fresh_frame(), symbol)

        rc = bsp.run(
            ["--write", "--strategy-config", str(cfg),
             "--store-dir", str(store_dir)],
            fetch_fn=fake_fetch)
        assert rc == 0
        assert seen == ["SPY", "BIL"]  # same normalization as the daily path


class TestWatchlistGuard:
    def test_refuses_when_sgov_in_watchlist(self, tmp_path, capsys):
        # st104#39 violation must FAIL, not be masked by a backfill.
        cfg = _write_config(
            tmp_path,
            {"watchlist": ["AAPL", "SGOV"], "sleeve": {"enabled": False}})
        rc = bsp.run(
            ["--write", "--strategy-config", str(cfg),
             "--store-dir", str(tmp_path / "ohlcv")],
            fetch_fn=lambda sym: pytest.fail("must refuse before fetching"))
        assert rc == 2
        assert "REFUSING" in capsys.readouterr().err

    def test_spy_leg_in_watchlist_is_fine(self, tmp_path):
        # SPY is the benchmark and legitimately everywhere — only the
        # T-bill leg is barred from the watchlist.
        store_dir = tmp_path / "ohlcv"
        cfg = _write_config(
            tmp_path,
            {"watchlist": ["AAPL", "SPY"], "sleeve": {"enabled": False}})
        store = LocalStore(data_dir=store_dir)
        rc = bsp.run(
            ["--write", "--strategy-config", str(cfg),
             "--store-dir", str(store_dir)],
            fetch_fn=lambda sym: store.save(_fresh_frame(), sym))
        assert rc == 0
