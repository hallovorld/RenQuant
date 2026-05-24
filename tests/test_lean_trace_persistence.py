"""LEAN decision-trace persistence parity tests."""
from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def test_lean_adapter_records_full_watchlist_trace(tmp_path):
    """AUDIT REGRESSION GUARD: LEAN must write the same sidecar DB trace
    as sim/live so backtest Sharpe/APY can be replayed by decision tree."""
    from adapters.lean import LeanAdapter
    from kernel.persistence import get_connection

    cfg = {
        "model_name": "renquant_104",
        "watchlist": ["AAA", "BBB"],
        "sector_map": {"AAA": "Tech", "BBB": "Finance"},
        "ranking": {"panel_scoring": {"artifact_path": "panel-ltr.json"}},
        "persistence": {
            "enabled": True,
            "db_path": str(tmp_path / "runs.db"),
        },
    }
    algo = SimpleNamespace(
        _config=cfg,
        _strategy_dir=_STRATEGY_DIR,
        _models={"AAA": {"_metadata": {"model_type": "xgb"}}},
        _universe_rejections={"BBB": "ic_missing"},
    )
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = algo
    adapter._db = get_connection(cfg, strategy_dir=_STRATEGY_DIR, role="live")
    adapter._universe_rejections = {"BBB": "ic_missing"}

    ctx = SimpleNamespace(
        today=datetime.date(2026, 5, 22),
        regime="BULL_CALM",
        confidence=0.60,
        portfolio_value=100_000.0,
        cash=90_000.0,
        candidates=[],
        holdings={},
        exits=[],
        rotations=[],
        orders=[],
        counters={},
        buy_blocked=False,
        skip_buys=False,
        bear_only=False,
        prices={},
        _ticker_score_snapshot={
            "AAA": {
                "rank_score": 0.42,
                "expected_return": 0.013,
                "model_action": "hold",
            },
        },
    )

    adapter._record_decision_trace(ctx, [])
    adapter._db.close()

    conn = sqlite3.connect(tmp_path / "runs.db")
    run = conn.execute(
        "SELECT run_type, regime FROM pipeline_runs",
    ).fetchone()
    assert run == ("lean", "BULL_CALM")
    rows = conn.execute(
        """SELECT ticker, in_universe, blocked_by, rank_score, expected_return
             FROM ticker_daily_state
            ORDER BY ticker""",
    ).fetchall()
    assert rows == [
        ("AAA", 1, "no_model_signal", 0.42, 0.013),
        ("BBB", 0, "universe:ic_missing", None, None),
    ]
    conn.close()


def _minimal_lean_adapter_for_context(tmp_path, *, panel_on=False):
    from adapters.lean import LeanAdapter

    cfg = {
        "model_name": "renquant_104",
        "watchlist": ["AAA"],
        "sector_map": {"AAA": "Tech"},
        "ranking": {"panel_scoring": {"enabled": panel_on}},
        "persistence": {"enabled": True, "db_path": str(tmp_path / "runs.db")},
    }

    class _Data:
        def ContainsKey(self, _sym):
            return False

    class _Portfolio:
        TotalPortfolioValue = 100_000.0
        Cash = 100_000.0

    algo = SimpleNamespace(
        _config=cfg,
        _strategy_dir=_STRATEGY_DIR,
        Time=datetime.datetime(2026, 5, 22),
        _spy_sym="SPY",
        _prev_closes={},
        _spy_returns=[],
        _models={},
        _sector_etf_symbols={},
        symbols={},
        _holdings={},
        _last_sell_dates={},
        _last_sell_pls={},
        _last_stop_exit_dates={},
        Portfolio=_Portfolio(),
        _skip_buys=False,
        _regime_state=None,
        _regime_counts={},
        _gmm=None,
        _corr={},
        _earnings={},
        _hwm={},
        History=lambda *args, **kwargs: SimpleNamespace(empty=True),
    )
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = algo
    adapter._db = None
    adapter._universe_rejections = {}
    adapter._panel_cache_ff = None
    adapter._panel_cache_fac = None
    adapter._panel_cache_macro = None
    adapter._panel_cache_emb = None
    adapter._panel_cache_last_date = None
    adapter._meta_label_logger = None
    adapter._meta_label_predictor = None
    return adapter, _Data()


def test_lean_make_context_stamps_run_id_for_score_distribution(tmp_path):
    adapter, data = _minimal_lean_adapter_for_context(tmp_path)

    ctx = adapter.make_context(data)

    assert ctx.run_id.startswith("2026-05-22-lean-")


def test_lean_make_context_propagates_last_sell_pls(tmp_path):
    adapter, data = _minimal_lean_adapter_for_context(tmp_path)
    adapter._algo._last_sell_dates = {"AAA": datetime.date(2026, 5, 21)}
    adapter._algo._last_sell_pls = {"AAA": 42.0}

    ctx = adapter.make_context(data)

    assert ctx.last_sell_pls == {"AAA": 42.0}


def test_lean_commit_stamps_full_exit_pl_for_wash_sale_parity(tmp_path):
    """LEAN must preserve realized P/L for the cost-aware wash-sale gate.

    Sim and live already distinguish gain sales from loss sales; without this
    LEAN binary-blocked every recent seller because P/L was unknown.
    """
    from adapters.lean import LeanAdapter
    from kernel.exits import ExitSignal, HoldingState
    from kernel.pipeline.context import InferenceContext

    today = datetime.date(2026, 5, 22)
    hs = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=10),
        high_watermark=110.0,
        shares=10.0,
    )

    class _Position:
        Quantity = 10.0
        UnrealizedProfit = 50.0

    class _Portfolio(dict):
        TotalPortfolioValue = 100_000.0
        Cash = 50_000.0

        def __getitem__(self, _sym):
            return _Position()

    class _Security:
        Price = 105.0

    algo = SimpleNamespace(
        _config={
            "model_name": "renquant_104",
            "tax": {
                "short_term_rate": 0.40,
                "long_term_rate": 0.20,
                "long_term_threshold_days": 365,
            },
            "ranking": {"panel_scoring": {"enabled": False}},
        },
        _models={"AAA": {}},
        symbols={"AAA": "AAA"},
        _sector_etf_symbols={},
        _benchmark="SPY",
        _spy_sym="SPY",
        Portfolio=_Portfolio(),
        Securities={"AAA": _Security()},
        _holdings={"AAA": hs},
        _last_sell_dates={},
        _last_sell_pls={},
        _last_stop_exit_dates={},
        _spy_returns=[],
        _regime_state=None,
        _regime_counts={},
        _hwm=100_000.0,
        _skip_buys=False,
        _prev_closes={},
        _tax_short=0.40,
        _tax_long=0.20,
        _tax_thresh_days=365,
        _total_tax=0.0,
        _executed_sells=0,
        _lt_trades=0,
        _st_trades=0,
        _trail_exits=0,
        _stop_exits=0,
        _sdl_exits=0,
        _rotation_exits=0,
        _executed_buys=0,
        _blocked_streak=0,
        _transition_blocks=0,
        _velocity_blocks=0,
        _earnings_blocks=0,
        _blocked_wash=0,
        _sector_blocks=0,
        _corr_blocks=0,
        _blocked_min_hold=0,
        Debug=lambda *_args, **_kwargs: None,
        Liquidate=lambda _sym: None,
        MarketOrder=lambda _sym, _qty: None,
        SetHoldings=lambda _sym, _target: None,
    )
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = algo
    adapter._db = None
    adapter._universe_rejections = {}
    ctx = InferenceContext(
        config=algo._config,
        today=today,
        holdings={"AAA": hs},
        exits=[("AAA", ExitSignal(True, "model exit", "model_sell"))],
        orders=[],
        ohlcv={},
        spy_returns=[],
        regime="BULL_CALM",
        confidence=0.8,
        portfolio_value=100_000.0,
        cash=50_000.0,
        prices={"AAA": 105.0},
        counters={},
    )

    adapter.commit(ctx)

    assert algo._last_sell_dates["AAA"] == today
    assert algo._last_sell_pls["AAA"] == 50.0


def test_lean_panel_frame_prep_failure_is_hard_fail(tmp_path, monkeypatch):
    adapter, data = _minimal_lean_adapter_for_context(tmp_path, panel_on=True)

    import training_panel.pipeline as panel_pipeline

    def _boom(**_kwargs):
        raise ValueError("bad panel frame")

    monkeypatch.setattr(panel_pipeline, "prepare_inference_panel_frames", _boom)

    with pytest.raises(RuntimeError, match="Panel frame prep failed"):
        adapter.make_context(data)
