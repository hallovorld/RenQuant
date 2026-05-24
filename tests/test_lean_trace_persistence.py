"""LEAN decision-trace persistence parity tests."""
from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

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
    )

    adapter._record_decision_trace(ctx, [])
    adapter._db.close()

    conn = sqlite3.connect(tmp_path / "runs.db")
    run = conn.execute(
        "SELECT run_type, regime FROM pipeline_runs",
    ).fetchone()
    assert run == ("lean", "BULL_CALM")
    rows = conn.execute(
        """SELECT ticker, in_universe, blocked_by
             FROM ticker_daily_state
            ORDER BY ticker""",
    ).fetchall()
    assert rows == [
        ("AAA", 1, "no_model_signal"),
        ("BBB", 0, "universe:ic_missing"),
    ]
    conn.close()
