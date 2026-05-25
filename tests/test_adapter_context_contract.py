"""Adapter -> InferenceContext contract tests.

The decision tree can only be audited if sim, live runner, and LEAN expose the
same minimum context surface to the shared pipeline. These tests exercise the
actual ``make_context`` paths with small fixtures and pin the fields that have
caused historical drift: realized sell P/L, stop-exit dates, DB handle,
run_id, prices, holdings, cash, and model/data payloads.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
STRATEGY = REPO / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


def _ohlcv_frame(n: int = 80) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-02", periods=n)
    close = pd.Series(range(n), index=idx, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _minimal_config(tmp_path: Path) -> dict:
    return {
        "model_name": "renquant_104",
        "_strategy_name": "renquant_104_test",
        "watchlist": ["AAA"],
        "benchmark": "SPY",
        "sector_map": {"AAA": "Tech"},
        "sector_etf_map": {},
        "ranking": {"panel_scoring": {"enabled": False}},
        "regime": {},
        "persistence": {
            "enabled": True,
            "db_path": str(tmp_path / "runs.db"),
            "sim_db_path": str(tmp_path / "sim_runs.db"),
        },
    }


def _assert_common_contract(ctx, *, source: str, db_marker: object | None) -> None:
    assert ctx.run_id.startswith(f"{ctx.today.isoformat()}-{source}-")
    assert isinstance(ctx.ohlcv, dict)
    assert isinstance(ctx.spy_returns, list)
    assert isinstance(ctx.models, dict)
    assert isinstance(ctx.holdings, dict)
    assert isinstance(ctx.last_sell_dates, dict)
    assert isinstance(ctx.last_sell_pls, dict)
    assert isinstance(ctx.last_stop_exit_dates, dict)
    assert isinstance(ctx.prices, dict)
    assert isinstance(ctx.regime_counts, dict)
    assert isinstance(ctx.skip_buys, bool)
    assert isinstance(float(ctx.portfolio_value), float)
    assert isinstance(float(ctx.cash), float)
    if db_marker is not None:
        assert ctx._db is db_marker  # noqa: SLF001


def test_sim_adapter_make_context_satisfies_pipeline_contract(tmp_path):
    from adapters.sim import SimAdapter

    today = pd.Timestamp("2026-02-27")
    ohlcv = {"AAA": _ohlcv_frame(), "SPY": _ohlcv_frame()}
    adapter = SimAdapter.__new__(SimAdapter)
    adapter._config = _minimal_config(tmp_path)
    adapter._ohlcv = ohlcv
    adapter._spy_df = ohlcv["SPY"]
    adapter._spy_prev_close = 100.0
    adapter._spy_returns = [0.01]
    adapter._models = {"AAA": {"_metadata": {"model_type": "xgb"}}}
    adapter._sector_etf_map = {}
    adapter._holdings = {}
    adapter._gmm = None
    adapter._corr = {}
    adapter._earnings = {}
    adapter._cash = 100_000.0
    adapter._last_sell_date = {"AAA": today - pd.Timedelta(days=5)}
    adapter._last_sell_pls = {"AAA": 123.45}
    adapter._last_stop_exit_date = {"AAA": today - pd.Timedelta(days=3)}
    adapter._hwm = 100_000.0
    adapter._skip_buys = False
    adapter._regime_state = None
    adapter._regime_counts = {"BULL_CALM": 1}
    adapter._feature_cache = {}
    adapter._monitor_state = {"idle_days": 0}
    adapter._rotation_proposals = []
    adapter._meta_label_logger = None
    adapter._meta_label_predictor = None
    adapter._db = object()
    adapter._exec_enabled = False
    adapter._ngboost_head = None
    adapter._panel_runtime_cache = {}
    adapter._alpha158_feature_cache = {}
    adapter._panel_history_cache = None
    adapter._panel_history_seq_len = 64
    adapter._panel_feature_frames = {
        "AAA": pd.DataFrame({"f": [1.0]}, index=[today]),
    }
    adapter._panel_factor_frames = {
        "AAA": pd.DataFrame({"g": [2.0]}, index=[today]),
    }
    adapter._panel_macro_frame = pd.DataFrame({"m": [3.0]}, index=[today])
    adapter._panel_asset_embeddings = {"AAA": [0.1, -0.2]}
    adapter._charge_daily_borrow = MethodType(lambda self, _today: None, adapter)
    adapter._portfolio_value = MethodType(
        lambda self, _prices, today_ts=None: 100_000.0,
        adapter,
    )
    adapter._pending_settle_cash = MethodType(lambda self: 0.0, adapter)
    adapter._available_buying_power = MethodType(lambda self: 100_000.0, adapter)
    adapter._get_panel_scorer_for_bar = MethodType(lambda self, _today: None, adapter)
    adapter._get_global_calibrator_for_bar = MethodType(
        lambda self, _today: None,
        adapter,
    )

    ctx = adapter.make_context(today)

    _assert_common_contract(ctx, source="sim", db_marker=adapter._db)
    assert ctx.last_sell_pls == {"AAA": 123.45}
    assert ctx.last_stop_exit_dates == {"AAA": (today - pd.Timedelta(days=3)).date()}
    assert ctx.ohlcv["AAA"].index.max() <= today
    assert getattr(ctx, "_panel_feature_frames")["AAA"].index.max() <= today
    assert getattr(ctx, "_panel_factor_frames")["AAA"].index.max() <= today
    assert getattr(ctx, "_panel_macro_frame").index.max() <= today
    assert getattr(ctx, "_panel_asset_embeddings") == {"AAA": [0.1, -0.2]}


def test_sim_adapter_prices_enabled_benchmark_sleeve_from_spy_df(tmp_path):
    """Sim must price the benchmark sleeve like live/LEAN do.

    A missing SPY price makes BenchmarkSleeveTask no-op in sim while live/LEAN
    would trade it, which breaks the shared-pipeline contract.
    """
    from adapters.sim import SimAdapter

    today = pd.Timestamp("2026-02-27")
    cfg = _minimal_config(tmp_path)
    cfg["portfolio"] = {
        "benchmark_sleeve": {
            "enabled": True,
            "ticker": "SPY",
        },
    }
    ohlcv = {"AAA": _ohlcv_frame()}
    spy_df = _ohlcv_frame()
    adapter = SimAdapter.__new__(SimAdapter)
    adapter._config = cfg
    adapter._ohlcv = ohlcv
    adapter._spy_df = spy_df
    adapter._spy_prev_close = 100.0
    adapter._spy_returns = [0.01]
    adapter._models = {"AAA": {"_metadata": {"model_type": "xgb"}}}
    adapter._sector_etf_map = {}
    adapter._holdings = {}
    adapter._gmm = None
    adapter._corr = {}
    adapter._earnings = {}
    adapter._cash = 100_000.0
    adapter._last_sell_date = {}
    adapter._last_sell_pls = {}
    adapter._last_stop_exit_date = {}
    adapter._hwm = 100_000.0
    adapter._skip_buys = False
    adapter._regime_state = None
    adapter._regime_counts = {"BULL_CALM": 1}
    adapter._feature_cache = {}
    adapter._monitor_state = {"idle_days": 0}
    adapter._rotation_proposals = []
    adapter._meta_label_logger = None
    adapter._meta_label_predictor = None
    adapter._db = None
    adapter._exec_enabled = False
    adapter._ngboost_head = None
    adapter._panel_runtime_cache = {}
    adapter._alpha158_feature_cache = {}
    adapter._panel_history_cache = None
    adapter._panel_history_seq_len = 64
    adapter._panel_feature_frames = {}
    adapter._panel_factor_frames = {}
    adapter._panel_macro_frame = None
    adapter._panel_asset_embeddings = {}
    adapter._charge_daily_borrow = MethodType(lambda self, _today: None, adapter)
    adapter._portfolio_value = MethodType(
        lambda self, _prices, today_ts=None: 100_000.0,
        adapter,
    )
    adapter._pending_settle_cash = MethodType(lambda self: 0.0, adapter)
    adapter._available_buying_power = MethodType(lambda self: 100_000.0, adapter)
    adapter._get_panel_scorer_for_bar = MethodType(lambda self, _today: None, adapter)
    adapter._get_global_calibrator_for_bar = MethodType(
        lambda self, _today: None,
        adapter,
    )

    ctx = adapter.make_context(today)

    assert ctx.prices["SPY"] == pytest.approx(float(spy_df.loc[today, "close"]))
    assert "SPY" in ctx.ohlcv
    assert ctx.ohlcv["SPY"].index.max() <= today


def test_runner_adapter_make_context_satisfies_pipeline_contract(tmp_path):
    from adapters.runner import RunnerAdapter

    cfg = _minimal_config(tmp_path)
    broker = MagicMock()
    broker.broker_name = None
    broker.get_account_value.return_value = 100_000.0
    broker.get_cash.return_value = 90_000.0
    broker.get_all_positions.return_value = []
    broker.get_open_orders.return_value = {"PENDING"}
    broker.get_filled_orders.return_value = []

    adapter = RunnerAdapter(
        cfg,
        models={"AAA": {"_metadata": {"model_type": "xgb"}}},
        broker=broker,
        strategy_dir=tmp_path,
        sell_only=False,
    )

    with patch("kernel.data.fetch_ohlcv", return_value=_ohlcv_frame()), \
         patch("kernel.realized_pnl.compute_recent_realized_pnl",
               return_value={"AAA": -12.0}):
        ctx = adapter.make_context()

    try:
        _assert_common_contract(ctx, source="live", db_marker=adapter._db)
        assert ctx.broker_name is None
        assert ctx.last_sell_pls == {"AAA": -12.0}
        assert ctx.pending_broker_tickers == {"PENDING"}
        assert ctx.monitor_state == {}
    finally:
        adapter._db.close()


def test_lean_adapter_make_context_satisfies_pipeline_contract(tmp_path):
    from adapters.lean import LeanAdapter

    class _Data:
        def ContainsKey(self, _sym):
            return False

    class _Portfolio:
        TotalPortfolioValue = 100_000.0
        Cash = 80_000.0

    today = dt.datetime(2026, 5, 22)
    algo = SimpleNamespace(
        _config=_minimal_config(tmp_path),
        _strategy_dir=tmp_path,
        Time=today,
        _spy_sym="SPY",
        _benchmark="SPY",
        _prev_closes={},
        _spy_returns=[0.01],
        _models={},
        _sector_etf_symbols={},
        symbols={},
        _holdings={},
        _last_sell_dates={"AAA": today.date() - dt.timedelta(days=5)},
        _last_sell_pls={"AAA": 77.0},
        _last_stop_exit_dates={"AAA": today.date() - dt.timedelta(days=2)},
        Portfolio=_Portfolio(),
        _skip_buys=False,
        _regime_state=None,
        _regime_counts={"BULL_CALM": 2},
        _gmm=None,
        _corr={},
        _earnings={},
        _hwm=100_000.0,
        History=lambda *args, **kwargs: SimpleNamespace(empty=True),
    )
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = algo
    adapter._db = object()
    adapter._universe_rejections = {}
    adapter._panel_cache_ff = None
    adapter._panel_cache_fac = None
    adapter._panel_cache_macro = None
    adapter._panel_cache_emb = None
    adapter._panel_cache_last_date = None
    adapter._meta_label_logger = None
    adapter._meta_label_predictor = None

    ctx = adapter.make_context(_Data())

    _assert_common_contract(ctx, source="lean", db_marker=adapter._db)
    assert ctx.last_sell_pls == {"AAA": 77.0}
    assert ctx.last_stop_exit_dates == {"AAA": today.date() - dt.timedelta(days=2)}


def test_lean_adapter_prices_unheld_model_candidates(tmp_path):
    """LEAN must price candidates, not just current holdings.

    Selection/QP sizing rejects selected buys with missing ``ctx.prices`` as
    ``size_bad_price``. Sim/live already populate model candidate prices; LEAN
    must expose the same contract.
    """
    from adapters.lean import LeanAdapter

    class _Data:
        _bars = {
            "AAA": SimpleNamespace(Close=123.45),
            "SPY": SimpleNamespace(Close=501.0),
        }

        def ContainsKey(self, sym):
            return sym in self._bars

        def __getitem__(self, sym):
            return self._bars[sym]

    class _Portfolio:
        TotalPortfolioValue = 100_000.0
        Cash = 100_000.0

    today = dt.datetime(2026, 5, 22)
    algo = SimpleNamespace(
        _config=_minimal_config(tmp_path),
        _strategy_dir=tmp_path,
        Time=today,
        _spy_sym="SPY",
        _benchmark="SPY",
        _prev_closes={},
        _spy_returns=[0.01],
        _models={"AAA": {"_metadata": {"model_type": "xgb"}}},
        _sector_etf_symbols={},
        symbols={"AAA": "AAA"},
        _holdings={},
        _last_sell_dates={},
        _last_sell_pls={},
        _last_stop_exit_dates={},
        Portfolio=_Portfolio(),
        _skip_buys=False,
        _regime_state=None,
        _regime_counts={"BULL_CALM": 2},
        _gmm=None,
        _corr={},
        _earnings={},
        _hwm=100_000.0,
        History=lambda *args, **kwargs: SimpleNamespace(empty=True),
    )
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = algo
    adapter._db = object()
    adapter._universe_rejections = {}
    adapter._panel_cache_ff = None
    adapter._panel_cache_fac = None
    adapter._panel_cache_macro = None
    adapter._panel_cache_emb = None
    adapter._panel_cache_last_date = None
    adapter._meta_label_logger = None
    adapter._meta_label_predictor = None

    ctx = adapter.make_context(_Data())

    assert ctx.prices["AAA"] == pytest.approx(123.45)
