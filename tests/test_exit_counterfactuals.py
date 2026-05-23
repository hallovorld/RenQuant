from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analyze_exit_counterfactuals import (  # noqa: E402
    TaxConfig,
    close_after_bars,
    counterfactual_rows,
    path_mae_mfe,
    summarize_counterfactuals,
    tax_on_gain,
)


def _write_ohlcv(root: Path, ticker: str, closes: list[float]) -> None:
    idx = pd.bdate_range("2026-01-01", periods=len(closes))
    df = pd.DataFrame({
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1_000_000] * len(closes),
    }, index=idx)
    df.index.name = "date"
    path = root / ticker
    path.mkdir(parents=True)
    df.to_parquet(path / "1d.parquet")


def test_tax_on_gain_uses_short_and_long_term_rates() -> None:
    tax = TaxConfig(short_rate=0.50, long_rate=0.20, lt_days=365)

    assert tax_on_gain(100.0, 10, tax) == pytest.approx(50.0)
    assert tax_on_gain(100.0, 400, tax) == pytest.approx(20.0)
    assert tax_on_gain(-100.0, 10, tax) == pytest.approx(0.0)


def test_close_after_bars_clips_at_series_end() -> None:
    close = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.bdate_range("2026-01-01", periods=3),
    )

    date, price = close_after_bars(close, close.index[0], 20)

    assert date == close.index[-1]
    assert price == pytest.approx(102.0)


def test_path_mae_mfe_from_entry_price() -> None:
    close = pd.Series(
        [100.0, 95.0, 110.0, 104.0],
        index=pd.bdate_range("2026-01-01", periods=4),
    )

    mae, mfe = path_mae_mfe(close, close.index[0], 100.0, 3)

    assert mae == pytest.approx(-0.05)
    assert mfe == pytest.approx(0.10)


def test_counterfactual_rows_marks_hold_20_better_after_false_stop(tmp_path: Path) -> None:
    root = tmp_path / "ohlcv"
    closes = [100.0, 90.0] + [105.0] * 30
    _write_ohlcv(root, "AAA", closes)
    idx = pd.bdate_range("2026-01-01", periods=len(closes))
    trips = pd.DataFrame([{
        "ticker": "AAA",
        "entry_date": idx[0],
        "exit_date": idx[1],
        "entry_price": 100.0,
        "exit_price": 90.0,
        "entry_notional": 1000.0,
        "gross_pnl": -100.0,
        "net_pnl": -100.0,
        "exit_reason": "stop_loss",
        "hold_days": 1,
        "entry_regime": "BULL_CALM",
        "entry_order_type": "QP_BUY",
    }])

    out = counterfactual_rows(
        trips,
        data_root=root,
        horizons=[20],
        tax=TaxConfig(short_rate=0.50, long_rate=0.32),
        barrier_window=20,
        pt_mult=10.0,
        sl_mult=10.0,
    )

    assert len(out) == 1
    # Hold-to-20d: +5% gross = +$50, short-term tax = $25, net = +$25.
    # Actual = -$100, so counterfactual improves by $125.
    assert out.iloc[0]["hold_20d_net_pnl"] == pytest.approx(25.0)
    assert out.iloc[0]["hold_20d_delta_vs_actual"] == pytest.approx(125.0)
    assert out.iloc[0]["post_exit_hold_20d_delta_vs_actual"] == pytest.approx(125.0)
    assert out.iloc[0]["post_exit_return_to_barrier_window"] == pytest.approx(105.0 / 90.0 - 1.0)
    assert out.iloc[0]["exit_meta_label_correct"] == 0


def test_counterfactual_rows_has_post_exit_continuation_lens(tmp_path: Path) -> None:
    """Post-exit hold uses the actual exit date, not entry+h bars."""
    root = tmp_path / "ohlcv"
    closes = [100.0] + [80.0] * 5 + [90.0] + [120.0] * 10
    _write_ohlcv(root, "AAA", closes)
    idx = pd.bdate_range("2026-01-01", periods=len(closes))
    trips = pd.DataFrame([{
        "ticker": "AAA",
        "entry_date": idx[0],
        "exit_date": idx[6],
        "entry_price": 100.0,
        "exit_price": 90.0,
        "entry_notional": 1000.0,
        "gross_pnl": -100.0,
        "net_pnl": -100.0,
        "exit_reason": "stop_loss",
        "hold_days": 6,
        "entry_regime": "BULL_CALM",
        "entry_order_type": "QP_BUY",
    }])

    out = counterfactual_rows(
        trips,
        data_root=root,
        horizons=[5],
        tax=TaxConfig(short_rate=0.50, long_rate=0.32),
        barrier_window=5,
        pt_mult=10.0,
        sl_mult=10.0,
    )
    row = out.iloc[0]

    # Entry+5 bars was before the actual stop and stayed at 80: worse than the
    # actual -$100. Exit+5 bars recovered to 120: +$100 net after short tax.
    assert row["hold_5d_delta_vs_actual"] == pytest.approx(-100.0)
    assert row["post_exit_hold_5d_delta_vs_actual"] == pytest.approx(200.0)
    assert row["post_exit_hold_5d_date"] == idx[11].date().isoformat()


def test_summarize_counterfactuals_reports_better_rate() -> None:
    cf = pd.DataFrame({
        "exit_reason": ["stop_loss", "stop_loss", "model_sell"],
        "entry_regime": ["BULL_CALM"] * 3,
        "entry_order_type": ["QP_BUY"] * 3,
        "ticker": ["A", "B", "C"],
        "actual_net_pnl": [-100.0, -50.0, 20.0],
        "actual_net_return": [-0.10, -0.05, 0.02],
        "exit_meta_label_correct": [0, 1, 0],
        "post_exit_return_to_barrier_window": [0.10, -0.03, 0.01],
        "mae_to_max_horizon": [-0.10, -0.20, -0.01],
        "mfe_to_max_horizon": [0.05, 0.02, 0.04],
        "hold_20d_delta_vs_actual": [120.0, -20.0, 5.0],
        "hold_20d_net_pnl": [20.0, -70.0, 25.0],
        "post_exit_hold_20d_delta_vs_actual": [130.0, 10.0, -5.0],
        "post_exit_hold_20d_net_pnl": [30.0, -40.0, 15.0],
    })

    summary = summarize_counterfactuals(cf, horizons=[20], min_n=2)
    stop = summary["exit_reason"][0]

    assert stop["exit_reason"] == "stop_loss"
    assert stop["n"] == 2
    assert stop["actual_net_pnl"] == pytest.approx(-150.0)
    assert stop["hold_20d_delta_sum"] == pytest.approx(100.0)
    assert stop["hold_20d_better_rate"] == pytest.approx(0.5)
    assert stop["post_exit_hold_20d_delta_sum"] == pytest.approx(140.0)
    assert stop["post_exit_hold_20d_better_rate"] == pytest.approx(1.0)
