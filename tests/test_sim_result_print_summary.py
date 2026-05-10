"""SimResult.print_summary visibility of the falsifiability triple.

CLAUDE.md §5.13.4 ("Single performance number = unverified claim") is
defeated if DSR / PBO / β / α / IR are populated on SimResult but never
printed. Anyone running ``python -m sim`` or a notebook cell sees only
stdout; the new fields must be in that output.

This file pins the output format so a future regression that, e.g.,
re-orders ``print_summary``'s lines or drops a metric will fail loudly.

Per §5.13.3 — every fix names its class-of-bug invariant. The invariant
here is: every falsifiability field on SimResult / MultiSeedSimResult
that is finite gets a corresponding line in print_summary's output.
"""
from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from sim.runner import (  # noqa: E402
    MultiSeedSimResult,
    SimResult,
)


def _capture_print(callable_):
    buf = io.StringIO()
    with redirect_stdout(buf):
        callable_()
    return buf.getvalue()


def _make_simresult_with_perf(
    *,
    n_days: int = 252,
    sharpe: float = 1.20,
    dsr: float = 0.85,
    pbo: float = 0.30,
    beta: float = 0.95,
    alpha: float = 0.022,
    ir: float = 0.40,
    n_trials: int = 5,
) -> SimResult:
    """Build a SimResult with controlled risk + falsifiability fields."""
    idx = pd.date_range("2026-01-01", periods=n_days, freq="B")
    eq_df = pd.DataFrame(
        {"portfolio": np.linspace(100_000, 110_000, n_days),
         "regime": "RISK_ON"},
        index=idx,
    )
    return SimResult(
        equity_df=eq_df,
        trade_log=[
            {"date": "2026-01-15", "ticker": "AAPL", "action": "buy"},
            {"date": "2026-02-15", "ticker": "AAPL", "action": "sell",
             "pnl_pct": 0.05, "hold_days": 22, "tax": 12.0},
        ],
        rotation_log=[],
        final_value=110_000.0,
        total_return=0.10,
        apy=0.10,
        win_rate=0.60,
        avg_hold=22.0,
        avg_pnl=0.05,
        total_tax=12.0,
        exit_reasons={"profit_target": 1},
        rotations=[],
        sharpe=sharpe,
        sortino=sharpe * 1.2,
        calmar=sharpe * 0.5,
        max_dd=0.08,
        ann_vol=0.15,
        dsr=dsr,
        pbo=pbo,
        n_trials=n_trials,
        beta_vs_spy=beta,
        alpha_vs_spy=alpha,
        information_ratio_vs_spy=ir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# C6.1 — single-seed SimResult prints DSR/PBO/β/α/IR.
# ─────────────────────────────────────────────────────────────────────────────


class TestSimResultPrintSummaryFalsifiability:
    def test_dsr_line_present_when_finite(self):
        r = _make_simresult_with_perf(dsr=0.85, pbo=float("nan"))
        out = _capture_print(r.print_summary)
        assert "DSR=" in out, "DSR not printed in single-seed summary"
        assert "+0.8500" in out, f"DSR value not in output:\n{out}"
        # n_trials must accompany DSR per §5.13.4 (selection-bias context).
        assert "n_trials=5" in out

    def test_pbo_dash_when_nan(self):
        r = _make_simresult_with_perf(pbo=float("nan"))
        out = _capture_print(r.print_summary)
        # NaN PBO must show as "—" (not "nan") per the existing risk-metric
        # convention.
        assert "PBO=—" in out, f"PBO NaN not printed as dash:\n{out}"

    def test_pbo_value_when_finite(self):
        r = _make_simresult_with_perf(pbo=0.30)
        out = _capture_print(r.print_summary)
        assert "PBO=0.3000" in out, f"PBO value missing:\n{out}"

    def test_beta_alpha_ir_line_present(self):
        r = _make_simresult_with_perf(beta=0.95, alpha=0.022, ir=0.40)
        out = _capture_print(r.print_summary)
        assert "Beta=" in out
        assert "+0.9500" in out
        assert "Alpha=" in out
        assert "+2.20%/yr" in out, f"Alpha annualized format missing:\n{out}"
        assert "InfoRatio=" in out
        assert "+0.4000" in out

    def test_existing_risk_lines_still_present(self):
        """Don't break Sharpe/Sortino/Calmar/MaxDD/Vol when ADDING new lines."""
        r = _make_simresult_with_perf()
        out = _capture_print(r.print_summary)
        assert "Sharpe=" in out
        assert "Sortino=" in out
        assert "Calmar=" in out
        assert "MaxDD=" in out
        assert "Vol=" in out


# ─────────────────────────────────────────────────────────────────────────────
# C6.2 — NaN handling when single-seed (PBO must NOT be claimed as 0).
# ─────────────────────────────────────────────────────────────────────────────


class TestSinglyNaNPrintSummary:
    def test_all_nan_falsifiability_skips_falsifiability_block(self):
        """When DSR + PBO + β + α + IR are all NaN (e.g. zero-trade sim),
        the falsifiability block is suppressed. This keeps the output
        clean for degenerate runs and avoids '—' line clutter."""
        r = _make_simresult_with_perf(
            dsr=float("nan"), pbo=float("nan"),
            beta=float("nan"), alpha=float("nan"), ir=float("nan"),
        )
        out = _capture_print(r.print_summary)
        # The block IS suppressed when ALL are NaN.
        assert "DSR=" not in out
        assert "Beta=" not in out


# ─────────────────────────────────────────────────────────────────────────────
# C6.3 — MultiSeedSimResult.print_summary shows mean ± std per metric.
# ─────────────────────────────────────────────────────────────────────────────


def _make_multi_seed_result(K: int = 5) -> MultiSeedSimResult:
    per_seed = [
        _make_simresult_with_perf(sharpe=1.0 + 0.1 * k)
        for k in range(K)
    ]
    return MultiSeedSimResult(
        per_seed_results=per_seed,
        seeds=list(range(K)),
        sharpe_mean=1.20, sharpe_std=0.15,
        apy_mean=0.0950, apy_std=0.0220,
        sortino_mean=1.50, sortino_std=0.18,
        calmar_mean=0.60, calmar_std=0.10,
        max_dd_mean=0.085, max_dd_std=0.012,
        dsr=0.55, pbo=0.40,
        majority_vote_action_consistency=0.78,
    )


class TestMultiSeedPrintSummaryMeanStd:
    def test_K5_prints_mean_plus_std(self):
        msr = _make_multi_seed_result(K=5)
        out = _capture_print(msr.print_summary)
        # Header with seed list.
        assert "K=5" in out
        # Mean ± std formatting for the headline metrics.
        assert "APY=" in out
        assert "±" in out, "Multi-seed output must show ± std"
        # APY printed as percent.
        assert "+9.50%" in out, f"APY mean not in expected percent format:\n{out}"
        assert "Sharpe=+1.20" in out
        # DSR + PBO + AgreeRate line.
        assert "DSR=+0.5500" in out
        assert "PBO=0.4000" in out
        assert "AgreeRate=78.0%" in out

    def test_K1_defers_to_headline(self):
        """K=1 is degenerate. The aggregate's print_summary delegates to
        the headline SimResult so the user still sees the full single-seed
        view."""
        single = _make_simresult_with_perf()
        msr = MultiSeedSimResult(
            per_seed_results=[single],
            seeds=[0],
        )
        out = _capture_print(msr.print_summary)
        assert "K=1" in out
        # Falls through to SimResult.print_summary — DSR/Beta should appear.
        assert "DSR=" in out
        assert "Beta=" in out


# ─────────────────────────────────────────────────────────────────────────────
# C6.4 — §5.13.3 AUDIT REGRESSION GUARD: pin specific format strings.
# ─────────────────────────────────────────────────────────────────────────────


class TestPrintSummaryRegression:
    """§5.13.3: invariant = "every finite falsifiability field appears
    in print_summary with its expected format". A regression where, e.g.,
    DSR is silently downgraded to 2 decimals or β is dropped → fails here.
    """

    def test_dsr_format_is_signed_4_decimals(self):
        r = _make_simresult_with_perf(dsr=0.0042)
        out = _capture_print(r.print_summary)
        assert "DSR=+0.0042" in out, \
            f"DSR format regression — expected 'DSR=+0.0042' in:\n{out}"

    def test_pbo_format_is_unsigned_4_decimals(self):
        r = _make_simresult_with_perf(pbo=0.7142)
        out = _capture_print(r.print_summary)
        # PBO is a probability ∈ [0,1] so it's printed unsigned.
        assert "PBO=0.7142" in out, \
            f"PBO format regression — expected 'PBO=0.7142' in:\n{out}"

    def test_beta_format_is_signed_4_decimals(self):
        r = _make_simresult_with_perf(beta=-0.7500)
        out = _capture_print(r.print_summary)
        assert "Beta=-0.7500" in out

    def test_alpha_format_includes_per_year_suffix(self):
        r = _make_simresult_with_perf(alpha=-0.0150)
        out = _capture_print(r.print_summary)
        # Alpha annualized — must have "/yr" so a reader doesn't think it's
        # daily / monthly.
        assert "Alpha=-1.50%/yr" in out, \
            f"Alpha annualized suffix regression:\n{out}"
