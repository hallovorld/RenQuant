"""LEAN decision-trace persistence parity tests."""
from __future__ import annotations

import datetime
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _filled_ticket(qty: float, price: float) -> SimpleNamespace:
    return SimpleNamespace(
        Status="Filled",
        QuantityFilled=qty,
        AverageFillPrice=price,
    )


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
        Portfolio=SimpleNamespace(TotalPortfolioValue=101_000.0, Cash=88_000.0),
        _holdings={},
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
                "expected_return_horizon_days": 60,
                "model_action": "hold",
            },
        },
    )

    adapter._record_decision_trace(ctx, [])
    adapter._db.close()

    conn = sqlite3.connect(tmp_path / "runs.db")
    run = conn.execute(
        "SELECT run_type, regime, portfolio_value, cash FROM pipeline_runs",
    ).fetchone()
    assert run == ("lean", "BULL_CALM", 101_000.0, 88_000.0)
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


def test_lean_adapter_records_nonfilled_execution_attempts(tmp_path):
    from adapters.lean import (
        LeanAdapter,
        _lean_buy_attempt_event,
        _lean_sell_attempt_event,
    )
    from kernel.exits import ExitSignal, HoldingState
    from kernel.persistence import get_connection

    cfg = {
        "model_name": "renquant_104",
        "watchlist": ["AAA"],
        "sector_map": {"AAA": "Tech"},
        "ranking": {"panel_scoring": {"artifact_path": "panel-ltr.json"}},
        "persistence": {"enabled": True, "db_path": str(tmp_path / "runs.db")},
    }
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = SimpleNamespace(
        _config=cfg,
        _strategy_dir=_STRATEGY_DIR,
        _models={"AAA": {"_metadata": {"model_type": "xgb"}}},
        Portfolio=SimpleNamespace(TotalPortfolioValue=100_000.0, Cash=90_000.0),
        _holdings={},
    )
    adapter._db = get_connection(cfg, strategy_dir=_STRATEGY_DIR, role="live")
    adapter._universe_rejections = {}
    holding = HoldingState(
        entry_price=90.0,
        entry_date=datetime.date(2026, 5, 1),
        high_watermark=101.0,
        shares=5.0,
        rank_score=0.41,
        panel_score=0.38,
        expected_return=0.01,
        expected_return_horizon_days=60,
    )
    holding.model_type = "xgb"
    holding.sector = "Tech"
    ctx = SimpleNamespace(
        today=datetime.date(2026, 5, 22),
        regime="BULL_CALM",
        confidence=0.60,
        portfolio_value=100_000.0,
        cash=90_000.0,
        candidates=[],
        holdings={"AAA": holding},
        exits=[],
        rotations=[],
        orders=[],
        counters={},
        buy_blocked=False,
        skip_buys=False,
        bear_only=False,
        prices={"AAA": 100.0},
        _ticker_score_snapshot={},
    )
    buy_attempt = _lean_buy_attempt_event(
        {
            "ticker": "AAA",
            "shares": 3,
            "price": 100.0,
            "target_pct": 0.003,
            "rank_score": 0.44,
            "source_job": "SelectionJob",
            "source_task": "SizeAndEmitTask",
        },
        ctx=ctx,
        status="Rejected",
        blocked_by="lean_order_status:Rejected",
    )
    sell_attempt = _lean_sell_attempt_event(
        ticker="AAA",
        sig=ExitSignal(True, "stop", "stop_loss"),
        holding=holding,
        ctx=ctx,
        requested_shares=5.0,
        price=100.0,
        status="Submitted",
    )

    adapter._record_decision_trace(ctx, [buy_attempt, sell_attempt])
    adapter._db.close()

    conn = sqlite3.connect(tmp_path / "runs.db")
    rows = conn.execute(
        """SELECT action, ticker, shares, price, blocked_by,
                  score_snapshot_json, decision_inputs_json
             FROM trades ORDER BY action""",
    ).fetchall()
    assert rows[0][:5] == (
        "buy_rejected", "AAA", 3.0, 100.0, "lean_order_status:Rejected",
    )
    assert rows[1][:5] == (
        "sell_pending", "AAA", 5.0, 100.0, "Submitted",
    )
    buy_snap = json.loads(rows[0][5])
    sell_inputs = json.loads(rows[1][6])
    assert buy_snap["attempt_status"] == "buy_rejected"
    assert sell_inputs["attempt_status"] == "sell_pending"
    assert sell_inputs["status"] == "Submitted"
    selected = conn.execute(
        "SELECT selected, blocked_by FROM candidate_scores WHERE ticker='AAA'",
    ).fetchone()
    assert selected == (0, None)
    conn.close()


def test_lean_trace_excludes_benchmark_sleeve_from_candidate_scores(tmp_path):
    from adapters.lean import LeanAdapter
    from kernel.exits import HoldingState
    from kernel.persistence import get_connection

    cfg = {
        "model_name": "renquant_104",
        "watchlist": ["AAA"],
        "ranking": {"panel_scoring": {"artifact_path": "panel-ltr.json"}},
        "portfolio": {
            "benchmark_sleeve": {
                "enabled": True,
                "ticker": "SPY",
                "exclude_from_alpha_pipeline": True,
            },
        },
        "persistence": {
            "enabled": True,
            "db_path": str(tmp_path / "runs.db"),
        },
    }
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = SimpleNamespace(
        _config=cfg,
        _strategy_dir=_STRATEGY_DIR,
        _models={"AAA": {"_metadata": {"model_type": "xgb"}}},
    )
    adapter._db = get_connection(cfg, strategy_dir=_STRATEGY_DIR, role="live")
    adapter._universe_rejections = {}
    spy_holding = HoldingState(
        entry_price=500.0,
        entry_date=datetime.date(2026, 5, 1),
        high_watermark=505.0,
        shares=100.0,
    )
    ctx = SimpleNamespace(
        today=datetime.date(2026, 5, 22),
        regime="BULL_CALM",
        confidence=0.60,
        portfolio_value=100_000.0,
        cash=50_000.0,
        candidates=[],
        holdings={"SPY": spy_holding},
        exits=[],
        rotations=[],
        orders=[],
        counters={},
        buy_blocked=False,
        skip_buys=False,
        bear_only=False,
        prices={"SPY": 500.0},
        _ticker_score_snapshot={},
    )

    adapter._record_decision_trace(ctx, [])
    adapter._db.close()

    conn = sqlite3.connect(tmp_path / "runs.db")
    candidate_rows = conn.execute(
        "SELECT ticker, role FROM candidate_scores ORDER BY ticker",
    ).fetchall()
    daily_rows = conn.execute(
        "SELECT ticker FROM ticker_daily_state ORDER BY ticker",
    ).fetchall()
    assert candidate_rows == []
    assert daily_rows == [("AAA",), ("SPY",)]
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


def test_lean_make_context_attaches_db_for_pipeline_tasks(tmp_path):
    """LEAN should expose the DB during pipeline execution, like sim/live."""
    from kernel.persistence import get_connection

    adapter, data = _minimal_lean_adapter_for_context(tmp_path)
    adapter._db = get_connection(
        adapter._algo._config,
        strategy_dir=_STRATEGY_DIR,
        role="live",
    )

    try:
        ctx = adapter.make_context(data)
        assert ctx._db is adapter._db  # noqa: SLF001
    finally:
        adapter._db.close()


def test_lean_make_context_syncs_holding_shares_and_lots(tmp_path):
    from kernel.exits import HoldingState

    adapter, data = _minimal_lean_adapter_for_context(tmp_path)
    today = datetime.date(2026, 5, 22)
    hs = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=10),
        high_watermark=110.0,
        shares=0.0,
    )

    class _Position:
        Quantity = 12.0

    class _Portfolio:
        TotalPortfolioValue = 100_000.0
        Cash = 50_000.0

        def __getitem__(self, _sym):
            return _Position()

    class _Security:
        Price = 105.0

    adapter._algo._models = {"AAA": {}}
    adapter._algo.symbols = {"AAA": "AAA"}
    adapter._algo._holdings = {"AAA": hs}
    adapter._algo.Portfolio = _Portfolio()
    adapter._algo.Securities = {"AAA": _Security()}

    ctx = adapter.make_context(data)

    out = ctx.holdings["AAA"]
    assert out.shares == 12.0
    assert len(out.lots) == 1
    assert out.lots[0].shares == 12.0
    assert out.lots[0].price == 100.0


def test_lean_make_context_loads_full_watchlist_ohlcv_for_freshness(tmp_path, monkeypatch):
    """AUDIT REGRESSION GUARD: universe floors must not shrink LEAN OHLCV.

    DataFreshnessGateTask validates the configured watchlist. LEAN may load a
    subset of models after universe-floor filtering, but it still subscribes to
    the full watchlist and must provide those bars to the shared pipeline.
    """
    import adapters.lean as lean_mod

    adapter, data = _minimal_lean_adapter_for_context(tmp_path)
    adapter._algo._config["watchlist"] = ["AAA", "BBB"]
    adapter._algo._models = {"AAA": {}}
    adapter._algo.symbols = {"AAA": "AAA", "BBB": "BBB"}
    monkeypatch.setattr(lean_mod, "Resolution", SimpleNamespace(Daily="Daily"), raising=False)

    def _history(sym, _lookback, _resolution):
        dates = pd.date_range("2026-05-18", periods=5, freq="B")
        idx = pd.MultiIndex.from_product([[sym], dates])
        return pd.DataFrame({
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1_000_000] * len(idx),
        }, index=idx)

    adapter._algo.History = _history

    ctx = adapter.make_context(data)

    assert {"AAA", "BBB", "SPY"}.issubset(ctx.ohlcv)


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
        _preflight_ok=True,
        Debug=lambda *_args, **_kwargs: None,
        Liquidate=lambda _sym: _filled_ticket(10.0, 105.0),
        MarketOrder=lambda _sym, qty: _filled_ticket(abs(qty), 105.0),
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


@pytest.mark.parametrize(
    "exit_ticket",
    [SimpleNamespace(Status="Rejected", QuantityFilled=0), None],
)
def test_lean_commit_rejected_exit_does_not_mutate_state(tmp_path, exit_ticket):
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
        _config={"model_name": "renquant_104", "ranking": {"panel_scoring": {"enabled": False}}},
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
        _preflight_ok=True,
        Debug=lambda *_args, **_kwargs: None,
        Liquidate=lambda _sym: exit_ticket,
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

    assert algo._executed_sells == 0
    assert algo._total_tax == 0.0
    assert algo._last_sell_dates == {}
    assert "AAA" in algo._holdings


def test_lean_commit_buy_and_topup_maintain_tax_lots(tmp_path):
    from adapters.lean import LeanAdapter
    from kernel.exits import HoldingState, TaxLot
    from kernel.pipeline.context import InferenceContext

    today = datetime.date(2026, 5, 22)
    hs = HoldingState(
        entry_price=100.0,
        entry_date=today - datetime.timedelta(days=10),
        high_watermark=105.0,
        shares=5.0,
    )
    hs.lots = [TaxLot(shares=5.0, price=100.0, date=hs.entry_date)]

    class _Position:
        Quantity = 5.0
        UnrealizedProfit = 100.0

    class _Portfolio(dict):
        TotalPortfolioValue = 100_000.0
        Cash = 50_000.0

        def __getitem__(self, _sym):
            return _Position()

    class _Security:
        Price = 120.0

    algo = SimpleNamespace(
        _config={"model_name": "renquant_104", "ranking": {"panel_scoring": {"enabled": False}}},
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
        _preflight_ok=True,
        Debug=lambda *_args, **_kwargs: None,
        Liquidate=lambda _sym: _filled_ticket(5.0, 120.0),
        MarketOrder=lambda _sym, qty: _filled_ticket(qty, 120.0),
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
        orders=[{
            "ticker": "AAA",
            "shares": 5.0,
            "price": 120.0,
            "target_pct": 0.10,
            "rank_score": 0.6,
            "panel_score": 0.2,
            "rs_score": 0.0,
            "regime": "BULL_CALM",
            "confidence": 0.8,
            "detail": "topup",
        }],
        exits=[],
        ohlcv={},
        spy_returns=[],
        regime="BULL_CALM",
        confidence=0.8,
        portfolio_value=100_000.0,
        cash=50_000.0,
        prices={"AAA": 120.0},
        counters={},
    )

    adapter.commit(ctx)

    out = algo._holdings["AAA"]
    assert out.shares == 10.0
    assert len(out.lots) == 2
    assert [(lot.shares, lot.price) for lot in out.lots] == [(5.0, 100.0), (5.0, 120.0)]
    assert out.entry_price == 110.0


@pytest.mark.parametrize(
    "buy_ticket",
    [SimpleNamespace(Status="Rejected", QuantityFilled=0), None],
)
def test_lean_commit_rejected_buy_does_not_mutate_state(tmp_path, buy_ticket):
    from adapters.lean import LeanAdapter
    from kernel.pipeline.context import InferenceContext

    today = datetime.date(2026, 5, 22)

    class _Portfolio(dict):
        TotalPortfolioValue = 100_000.0
        Cash = 100_000.0

        def __getitem__(self, _sym):
            return SimpleNamespace(Quantity=0.0, UnrealizedProfit=0.0)

    class _Security:
        Price = 100.0

    algo = SimpleNamespace(
        _config={"model_name": "renquant_104", "ranking": {"panel_scoring": {"enabled": False}}},
        _models={"AAA": {}},
        symbols={"AAA": "AAA"},
        _sector_etf_symbols={},
        _benchmark="SPY",
        _spy_sym="SPY",
        Portfolio=_Portfolio(),
        Securities={"AAA": _Security()},
        _holdings={},
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
        _preflight_ok=True,
        Debug=lambda *_args, **_kwargs: None,
        Liquidate=lambda _sym: None,
        MarketOrder=lambda _sym, _qty: buy_ticket,
        SetHoldings=lambda _sym, _target: None,
    )
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = algo
    adapter._db = None
    adapter._universe_rejections = {}
    ctx = InferenceContext(
        config=algo._config,
        today=today,
        holdings={},
        orders=[{
            "ticker": "AAA",
            "shares": 10.0,
            "price": 100.0,
            "target_pct": 0.01,
            "rank_score": 0.6,
            "panel_score": 0.2,
            "rs_score": 0.0,
            "regime": "BULL_CALM",
            "confidence": 0.8,
            "detail": "buy",
        }],
        exits=[],
        ohlcv={},
        spy_returns=[],
        regime="BULL_CALM",
        confidence=0.8,
        portfolio_value=100_000.0,
        cash=100_000.0,
        prices={"AAA": 100.0},
        counters={},
    )

    adapter.commit(ctx)

    assert algo._executed_buys == 0
    assert algo._holdings == {}


def test_lean_commit_benchmark_sleeve_buy_allows_missing_alpha_scores(tmp_path):
    from adapters.lean import LeanAdapter
    from kernel.pipeline.context import InferenceContext

    today = datetime.date(2026, 5, 22)
    debug_lines: list[str] = []
    market_calls: list[tuple[str, int]] = []

    class _Portfolio(dict):
        TotalPortfolioValue = 100_000.0
        Cash = 100_000.0

        def __getitem__(self, _sym):
            return SimpleNamespace(Quantity=0.0, UnrealizedProfit=0.0)

    class _Security:
        Price = 500.0

    algo = SimpleNamespace(
        _config={
            "model_name": "renquant_104",
            "ranking": {"panel_scoring": {"enabled": False}},
            "portfolio": {
                "benchmark_sleeve": {
                    "enabled": True,
                    "ticker": "SPY",
                    "exclude_from_alpha_pipeline": True,
                },
            },
        },
        _models={},
        symbols={"SPY": "SPY"},
        _sector_etf_symbols={},
        _benchmark="SPY",
        _spy_sym="SPY",
        Portfolio=_Portfolio(),
        Securities={"SPY": _Security()},
        _holdings={},
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
        _preflight_ok=True,
        Debug=lambda msg: debug_lines.append(str(msg)),
        Liquidate=lambda _sym: None,
        MarketOrder=lambda sym, qty: (
            market_calls.append((sym, qty)) or _filled_ticket(qty, 500.0)
        ),
        SetHoldings=lambda _sym, _target: pytest.fail("BUY must use exact-share MarketOrder"),
    )
    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = algo
    adapter._db = None
    adapter._universe_rejections = {}
    ctx = InferenceContext(
        config=algo._config,
        today=today,
        holdings={},
        orders=[{
            "ticker": "SPY",
            "shares": 10.0,
            "price": 500.0,
            "target_pct": 0.50,
            "rank_score": None,
            "panel_score": None,
            "rs_score": None,
            "regime": "BULL_CALM",
            "confidence": 0.8,
            "detail": "benchmark_core_sleeve",
            "order_type": "BENCHMARK_SLEEVE_BUY",
        }],
        exits=[],
        ohlcv={},
        spy_returns=[],
        regime="BULL_CALM",
        confidence=0.8,
        portfolio_value=100_000.0,
        cash=100_000.0,
        prices={"SPY": 500.0},
        counters={},
    )

    adapter.commit(ctx)

    assert market_calls == [("SPY", 10)]
    assert "rank=NA" in debug_lines[0]
    assert "rs=NA" in debug_lines[0]


def test_lean_partial_sell_uses_fifo_disposed_basis_for_tax(tmp_path):
    from adapters.lean import LeanAdapter
    from kernel.exits import ExitSignal, HoldingState, TaxLot
    from kernel.pipeline.context import InferenceContext

    today = datetime.date(2026, 5, 22)
    hs = HoldingState(
        entry_price=150.0,
        entry_date=today - datetime.timedelta(days=20),
        high_watermark=250.0,
        shares=10.0,
    )
    hs.lots = [
        TaxLot(shares=5.0, price=100.0, date=today - datetime.timedelta(days=20)),
        TaxLot(shares=5.0, price=200.0, date=today - datetime.timedelta(days=5)),
    ]

    class _Position:
        Quantity = 10.0
        UnrealizedProfit = 1000.0  # avg-cost fallback would tax only 500 on 5sh.

    class _Portfolio(dict):
        TotalPortfolioValue = 100_000.0
        Cash = 50_000.0

        def __getitem__(self, _sym):
            return _Position()

    class _Security:
        Price = 250.0

    algo = SimpleNamespace(
        _config={
            "model_name": "renquant_104",
            "tax": {
                "short_term_rate": 0.40,
                "long_term_rate": 0.20,
                "long_term_threshold_days": 365,
            },
            "rotation": {"joint_actions": {"qp_tax_lot_method": "fifo"}},
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
        Liquidate=lambda _sym: _filled_ticket(10.0, 250.0),
        MarketOrder=lambda _sym, qty: _filled_ticket(abs(qty), 250.0),
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
        exits=[("AAA", ExitSignal(True, "qp trim", "qp_sell", quantity=5.0))],
        orders=[],
        ohlcv={},
        spy_returns=[],
        regime="BULL_CALM",
        confidence=0.8,
        portfolio_value=100_000.0,
        cash=50_000.0,
        prices={"AAA": 250.0},
        counters={},
    )

    adapter.commit(ctx)

    # FIFO disposed basis = 5 * 100; gross = 5 * 250 - 500 = 750; tax = 300.
    assert algo._total_tax == 300.0
    assert "AAA" not in algo._last_sell_dates
    out = algo._holdings["AAA"]
    assert out.shares == 5.0
    assert len(out.lots) == 1
    assert out.lots[0].price == 200.0
    assert out.entry_price == 200.0


def test_lean_panel_frame_prep_failure_is_hard_fail(tmp_path, monkeypatch):
    adapter, data = _minimal_lean_adapter_for_context(tmp_path, panel_on=True)

    import training_panel.pipeline as panel_pipeline

    def _boom(**_kwargs):
        raise ValueError("bad panel frame")

    monkeypatch.setattr(panel_pipeline, "prepare_inference_panel_frames", _boom)

    with pytest.raises(RuntimeError, match="Panel frame prep failed"):
        adapter.make_context(data)
