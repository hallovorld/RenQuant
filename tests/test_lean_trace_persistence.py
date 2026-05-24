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


def test_lean_panel_frame_prep_failure_is_hard_fail(tmp_path, monkeypatch):
    adapter, data = _minimal_lean_adapter_for_context(tmp_path, panel_on=True)

    import training_panel.pipeline as panel_pipeline

    def _boom(**_kwargs):
        raise ValueError("bad panel frame")

    monkeypatch.setattr(panel_pipeline, "prepare_inference_panel_frames", _boom)

    with pytest.raises(RuntimeError, match="Panel frame prep failed"):
        adapter.make_context(data)
