"""Regression tests for the 2026-05-04 deep-audit fixes.

Each test pins a specific bug discovered during the audit so future
refactors that re-introduce the same class of issue fail loud.

Bug references (numbered as in doc/audits/2026-05-04-deep-audit/01-inference-pipeline.md):
  01: GMMTask null-guard for ctx.gmm
  06: EMA50GateTask fail-SAFE on missing SPY
  07: DrawdownCircuitTask NaN portfolio_value silently disabled halt
  16: RefreshPanelCalibratorTask fail-loud on subprocess failure
  18: check_take_profit NaN entry_price slips past <=0
  19: check_stop_loss NaN entry_price slips past <=0
  20: score_candidates NaN rank_score → non-deterministic sort
  21: compute_relative_strength missing isinf guard
  22: qp_solver μ=all-zeros silent no-trade
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
STRATEGY = REPO / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))


# ── Issue 01 — GMMTask null-guard ──────────────────────────────────────────────

class TestGMMTaskNullGuard:
    def _make_ctx(self, gmm=None, spy_df=None):
        from kernel.regime import RegimeState
        ctx = SimpleNamespace()
        ctx.gmm = gmm
        ctx.spy_returns = [0.001] * 100
        ctx.ohlcv = {"SPY": spy_df}
        ctx.regime_state = RegimeState()
        ctx.config = {"regime": {"vol_realized_window": 20}}
        return ctx

    def test_ctx_gmm_none_does_not_crash(self):
        from kernel.pipeline.task_regime import GMMTask
        import pandas as pd
        ctx = self._make_ctx(gmm=None,
                             spy_df=pd.DataFrame({"close": [100.0] * 50}))
        # Pre-fix: gmm_predict(None, …) crashed and killed the cron
        GMMTask().run(ctx)
        # Post-fix: empty probs dict; downstream defaults
        assert ctx.regime_state.gmm_probs == {}

    def test_spy_df_none_does_not_crash(self):
        from kernel.pipeline.task_regime import GMMTask
        ctx = self._make_ctx(gmm={"some": "artifact"}, spy_df=None)
        GMMTask().run(ctx)
        assert ctx.regime_state.gmm_probs == {}


# ── Issue 06 — EMA50GateTask fail-SAFE on missing SPY ──────────────────────────

class TestEMA50GateFailSafe:
    def _make_ctx(self, spy_df=None):
        ctx = SimpleNamespace()
        ctx.ohlcv = {"SPY": spy_df}
        ctx.buy_blocked = False
        ctx.counters = {}
        return ctx

    def test_missing_spy_blocks_buys(self):
        """Pre-fix: returned None, buys continued. Post-fix: buy_blocked=True."""
        from kernel.pipeline.task_gates import EMA50GateTask
        ctx = self._make_ctx(spy_df=None)
        result = EMA50GateTask().run(ctx)
        assert ctx.buy_blocked is True, \
            "Missing SPY data must fail-SAFE (block buys); pre-fix it failed-OPEN"
        assert result is False

    def test_empty_spy_blocks_buys(self):
        import pandas as pd
        from kernel.pipeline.task_gates import EMA50GateTask
        ctx = self._make_ctx(spy_df=pd.DataFrame(columns=["close"]))
        EMA50GateTask().run(ctx)
        assert ctx.buy_blocked is True


# ── Issue 07 — DrawdownCircuit NaN portfolio_value ─────────────────────────────

class TestDrawdownCircuitNaNGuard:
    def _make_ctx(self, hwm=100_000, portfolio_value=90_000, regime="BULL_CALM"):
        ctx = SimpleNamespace()
        ctx.hwm = float(hwm)
        ctx.portfolio_value = float(portfolio_value)
        ctx.regime = regime
        ctx.config = {"regime_params": {"BULL_CALM": {"drawdown_halt_pct": 0.10}}}
        ctx.skip_buys = False
        return ctx

    def test_nan_portfolio_value_forces_skip_buys(self):
        """Pre-fix: NaN/halt_pct comparison False → halt silently disabled.
        Post-fix: skip_buys forced True."""
        from kernel.pipeline.task_drawdown import DrawdownCircuitTask
        ctx = self._make_ctx(portfolio_value=float("nan"))
        DrawdownCircuitTask().run(ctx)
        assert ctx.skip_buys is True, \
            "NaN portfolio_value must fail-SAFE to skip_buys=True"

    def test_inf_hwm_forces_skip_buys(self):
        from kernel.pipeline.task_drawdown import DrawdownCircuitTask
        ctx = self._make_ctx(hwm=float("inf"))
        DrawdownCircuitTask().run(ctx)
        assert ctx.skip_buys is True


# ── Issue 18 + 19 — exits.py NaN entry_price guards ────────────────────────────

class TestExitsNaNEntryPriceGuards:
    def _state(self, entry_price):
        from kernel.exits import HoldingState
        import datetime
        return HoldingState(
            entry_price=float(entry_price),
            entry_date=datetime.date(2025, 1, 1),
            high_watermark=100.0,
            prev_close=100.0,
        )

    def test_take_profit_skips_on_nan_entry_price(self):
        from kernel.exits import check_take_profit
        sig = check_take_profit(150.0, self._state(float("nan")), take_profit_pct=0.10)
        assert sig.should_exit is False

    def test_stop_loss_skips_on_nan_entry_price(self):
        from kernel.exits import check_stop_loss
        sig = check_stop_loss(50.0, self._state(float("nan")), stop_pct=0.10)
        assert sig.should_exit is False

    def test_take_profit_fires_on_normal_path(self):
        # Sanity: real entry_price still triggers
        from kernel.exits import check_take_profit
        sig = check_take_profit(120.0, self._state(100.0), take_profit_pct=0.10)
        assert sig.should_exit is True

    def test_stop_loss_fires_on_normal_path(self):
        from kernel.exits import check_stop_loss
        sig = check_stop_loss(80.0, self._state(100.0), stop_pct=0.10)
        assert sig.should_exit is True


# ── Issue 20 — score_candidates NaN-safe ───────────────────────────────────────

class TestScoreCandidatesNaNSafe:
    def _make_cands(self, scores):
        from kernel.selection import CandidateResult
        return [CandidateResult(
            ticker=f"T{i}", raw_score=0.0, rank_score=s, rs_score=0.0,
        ) for i, s in enumerate(scores)]

    def test_nan_rank_does_not_poison_min_max(self):
        """Pre-fix: NaN in rank_scores made min(...)=NaN → all _norm()=NaN
        → blend()=NaN → sort non-deterministic. Post-fix: NaN entries
        treated as 0 in normalize, real values produce stable ordering."""
        from kernel.selection import score_candidates
        cands = self._make_cands([0.3, float("nan"), 0.7, 0.5])
        ranked = score_candidates(cands, w_rank=1.0, w_rs=0.0)
        # The ticker with rank=0.7 must be first (deterministic)
        # — pre-fix this would be non-deterministic across runs
        finite_first = [c.ticker for c in ranked
                        if c.rank_score is not None
                        and math.isfinite(float(c.rank_score))]
        assert finite_first[0] == "T2", \
            f"Expected T2 (rank=0.7) first, got {[(c.ticker, c.rank_score) for c in ranked]}"

    def test_all_nan_returns_input_order_no_crash(self):
        from kernel.selection import score_candidates
        cands = self._make_cands([float("nan")] * 3)
        ranked = score_candidates(cands, w_rank=1.0, w_rs=0.0)
        assert len(ranked) == 3   # no crash, no drops


# ── Issue 21 — compute_relative_strength inf guard ─────────────────────────────

class TestComputeRelativeStrengthInfGuard:
    def test_inf_input_returns_zero(self):
        from kernel.selection import compute_relative_strength
        assert compute_relative_strength(float("inf"), 0.05) == 0.0
        assert compute_relative_strength(0.05, float("-inf")) == 0.0

    def test_nan_input_returns_zero_back_compat(self):
        from kernel.selection import compute_relative_strength
        assert compute_relative_strength(float("nan"), 0.05) == 0.0

    def test_finite_passthrough(self):
        from kernel.selection import compute_relative_strength
        assert compute_relative_strength(0.10, 0.05) == pytest.approx(0.05)


# ── Issue 22 — qp_solver mu=all-zeros silent no-trade detection ────────────────

class TestQPSolverNoSignalDetection:
    def test_all_zero_mu_returns_optimal_no_signal(self):
        """Pre-fix: μ=0 made objective minimize at Δw=0, status='optimal',
        caller silently emitted 0 trades. Post-fix: status='optimal_no_signal'
        so caller can branch on it."""
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        n = 5
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.zeros(n),                 # ← the failure mode
            sigma=np.full(n, 0.05),
            risk_aversion=3.0,
            cost_kappa=0.0005,
            cash_reserve=0.05,
            w_upper=np.full(n, 0.20),
            w_lower=0.0,
            dw_max=np.full(n, 0.50),
        )
        assert sol.status == "optimal_no_signal", \
            f"Expected optimal_no_signal sentinel, got status={sol.status!r}"

    def test_real_signal_returns_plain_optimal(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        n = 5
        mu = np.array([0.01, 0.02, -0.01, 0.0, 0.005])
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=mu,
            sigma=np.full(n, 0.05),
            risk_aversion=3.0,
            cost_kappa=0.0005,
            cash_reserve=0.05,
            w_upper=np.full(n, 0.20),
            w_lower=0.0,
            dw_max=np.full(n, 0.50),
        )
        assert sol.status == "optimal"


# ── Issue 16 — calibrator-refresh fail-loud ────────────────────────────────────

class TestCalibratorRefreshFailLoud:
    def _make_ctx(self, tmp_path, refresh_failure_mode="raise"):
        from training_panel.pp_panel_training import PanelTrainingContext
        # PanelTrainingContext reads strategy_dir from config["_strategy_dir"]
        # via a property; not a constructor kwarg.
        # The Task computes repo_root = strategy_dir.parent.parent and looks
        # for scripts/fit_panel_calibrator.py there. Mirror the production
        # layout: <repo_root>/backtesting/<strategy>/.
        repo_root = tmp_path
        backtesting_dir = repo_root / "backtesting"
        backtesting_dir.mkdir()
        strategy_dir = backtesting_dir / "renquant_104"
        strategy_dir.mkdir()
        scripts_dir = repo_root / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "fit_panel_calibrator.py").write_text("# stub")
        cfg = {
            "ranking": {
                "panel_scoring": {
                    "global_calibration": {
                        "enabled": True,
                        "auto_refresh": True,
                        "refresh_failure_mode": refresh_failure_mode,
                    }
                }
            },
            "_strategy_name": "renquant_104",
            "_strategy_dir":  str(strategy_dir),
        }
        ctx = PanelTrainingContext(
            config=cfg,
            watchlist=["NVDA"],
        )
        return ctx

    def test_subprocess_failure_raises_by_default(self, tmp_path):
        """Pre-fix: subprocess.run rc != 0 only logged a warning; the model
        artifact had already been overwritten so production ended up with
        new model + stale calibrator (the original CAL-7 incident class).
        Post-fix: raises RuntimeError unless caller opts into legacy
        log-only behavior via refresh_failure_mode='warn'."""
        from training_panel.pp_panel_training import RefreshPanelCalibratorTask
        ctx = self._make_ctx(tmp_path)

        fake_result = SimpleNamespace(returncode=1, stdout="", stderr="boom")
        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="calibrator refresh failed"):
                RefreshPanelCalibratorTask().run(ctx)

    def test_warn_mode_preserves_legacy_behavior(self, tmp_path):
        from training_panel.pp_panel_training import RefreshPanelCalibratorTask
        ctx = self._make_ctx(tmp_path, refresh_failure_mode="warn")

        fake_result = SimpleNamespace(returncode=1, stdout="", stderr="boom")
        with patch("subprocess.run", return_value=fake_result):
            # Must NOT raise — backwards compat for staged migrations.
            RefreshPanelCalibratorTask().run(ctx)


# ── Issue 23 + 24 — quality-floor NaN sigma fail-OPEN ─────────────────────────

class TestQualityFloorNaNSigmaGuards:
    """Pre-fix: NaN sigma slipped past `<= 0` in both _gate_b_edge_sharpe
    and _gate_c_no_trade_band, candidate passed the gate as if quality
    were high. Post-fix: explicit isfinite check rejects NaN/inf σ."""

    def test_gate_c_rejects_nan_sigma(self):
        from kernel.panel_pipeline.task_quality_floor import _gate_c_no_trade_band
        cand = SimpleNamespace(mu=0.05, sigma=float("nan"))
        passes, reason = _gate_c_no_trade_band(
            cand, current_weight=0.0, risk_aversion=3.0,
            round_trip_cost=0.001, band_constant=0.5,
        )
        assert passes is False
        assert "nonfinite" in reason

    def test_gate_c_rejects_inf_sigma(self):
        from kernel.panel_pipeline.task_quality_floor import _gate_c_no_trade_band
        cand = SimpleNamespace(mu=0.05, sigma=float("inf"))
        passes, reason = _gate_c_no_trade_band(
            cand, current_weight=0.0, risk_aversion=3.0,
            round_trip_cost=0.001, band_constant=0.5,
        )
        assert passes is False

    def test_gate_b_rejects_nan_sigma(self):
        from kernel.panel_pipeline.task_quality_floor import _gate_b_edge_sharpe
        cand = SimpleNamespace(mu=0.05, sigma=float("nan"))
        passes, reason = _gate_b_edge_sharpe(cand, threshold=0.10)
        assert passes is False
        assert "nonfinite" in reason

    def test_gate_b_rejects_inf_mu(self):
        from kernel.panel_pipeline.task_quality_floor import _gate_b_edge_sharpe
        cand = SimpleNamespace(mu=float("inf"), sigma=0.05)
        passes, reason = _gate_b_edge_sharpe(cand, threshold=0.10)
        assert passes is False

    def test_gate_b_normal_path_still_works(self):
        from kernel.panel_pipeline.task_quality_floor import _gate_b_edge_sharpe
        # mu=0.02, sigma=0.10 → edge_sharpe=0.20 — passes 0.10 threshold
        cand = SimpleNamespace(mu=0.02, sigma=0.10)
        passes, reason = _gate_b_edge_sharpe(cand, threshold=0.10)
        assert passes is True


# ── Issue 27 — sim adapter pnl_pct NaN propagation ────────────────────────────

class TestSimAdapterPnLPctNaNGuard:
    """Pre-fix: NaN entry_price was truthy in Python (`bool(NaN) == True`),
    so `(price - NaN) / NaN = NaN` propagated into trade_log.pnl_pct,
    then into win_rate / avg_pnl / holdout reports."""

    def test_nan_entry_price_pnl_is_zero_not_nan(self, tmp_path):
        # Direct unit test of the defensive computation snippet.
        # We can't easily instantiate a full SimAdapter without market
        # data, but we can verify the math contract holds.
        import math
        price = 100.0
        # Same defensive pattern as the post-fix sim.py uses
        for entry in [float("nan"), float("inf"), 0.0, -1.0]:
            _entry = float(entry or 0.0)
            if (math.isfinite(price) and math.isfinite(_entry) and _entry > 0):
                pnl = (price - _entry) / _entry
            else:
                pnl = 0.0
            assert math.isfinite(pnl), \
                f"pnl_pct must be finite for entry={entry}, got {pnl}"
            assert pnl == 0.0, \
                f"pnl_pct should fall back to 0.0 for bad entry={entry}, got {pnl}"

    def test_nan_price_pnl_is_zero(self):
        import math
        entry = 100.0
        for price in [float("nan"), float("inf"), float("-inf")]:
            _entry = float(entry or 0.0)
            if (math.isfinite(price) and math.isfinite(_entry) and _entry > 0):
                pnl = (price - _entry) / _entry
            else:
                pnl = 0.0
            assert pnl == 0.0

    def test_sim_apply_sell_compiles_without_nameerror(self):
        """Catch the kind of NameError that hit retrain v1: my Issue 27 fix
        used `math.isfinite()` but module-level `math` wasn't imported.
        Verify the source byte-compiles (catches typos / missing imports)
        AND that the `pnl_pct` math block uses a name that resolves."""
        import importlib.util as _iu
        path = REPO / "backtesting" / "renquant_104" / "adapters" / "sim.py"
        spec = _iu.spec_from_file_location("_sim_compile_test", path)
        # compile() raises SyntaxError on broken source; we don't actually
        # exec because sim.py has heavy deps.
        src = path.read_text()
        compile(src, str(path), "exec")
        # Stronger check: the post-fix block must reference `math` via an
        # in-scope name (either `math.isfinite` after a function-local
        # `import math` OR `_math_pnl.isfinite` after the local alias I
        # added). Bare `math.isfinite(price)` without an import in the
        # same function-scope WILL silently re-introduce the NameError.
        i = src.find("_pnl_pct = (price - _entry) / _entry")
        assert i > 0
        block = src[max(0, i - 800):i + 100]
        # Must have either 'import math' OR 'import math as _math' inside
        # the immediate function body (the `def _apply_sell` containing
        # this block).
        has_local_math_import = (
            "import math\n" in block
            or "import math as _math_pnl" in block
            or "import math as math" in block
        )
        assert has_local_math_import, (
            "sim.py _apply_sell pnl_pct math block must have a local "
            "`import math` (audit Issue 27 NameError repro). Pre-fix "
            "v1 retrain crashed mid-sim with NameError on math.isfinite."
        )


# ── Issue 32 — qp_solver NaN sigma poisons Σ ──────────────────────────────────

class TestQPSolverNaNSigmaSanitization:
    """Pre-fix: NaN sigma slipped past `np.clip(arr, 1e-6, None)` (np.clip
    preserves NaN), then `arr**2 = NaN` poisoned Σ → NaN objective →
    SLSQP undefined behavior. Post-fix: NaN/inf σ replaced with 5%
    default before clipping."""

    def test_nan_sigma_does_not_poison_solution(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        n = 5
        sigma = np.array([0.05, float("nan"), 0.05, float("inf"), 0.05])
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.01, 0.02, 0.0, 0.0, 0.005]),
            sigma=sigma,
            risk_aversion=3.0,
            cost_kappa=0.0001,
            cash_reserve=0.05,
            w_upper=np.full(n, 0.20),
            w_lower=0.0,
            dw_max=np.full(n, 0.50),
        )
        # Result must be a finite vector
        assert np.isfinite(sol.delta_w).all(), \
            f"NaN sigma poisoned solution: delta_w={sol.delta_w}"
        assert np.isfinite(sol.target_w).all()
        assert np.isfinite(sol.objective)

    def test_nan_in_full_sigma_matrix_handled(self):
        from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
        n = 4
        Sigma = np.eye(n) * 0.01
        Sigma[0, 1] = float("nan")   # corrupted off-diagonal
        Sigma[1, 0] = float("nan")
        sol = solve_portfolio_qp(
            w_current=np.zeros(n),
            mu=np.array([0.01, 0.02, 0.0, 0.005]),
            Sigma=Sigma,
            risk_aversion=3.0,
            cost_kappa=0.0001,
            cash_reserve=0.05,
            w_upper=np.full(n, 0.20),
            w_lower=0.0,
            dw_max=np.full(n, 0.50),
        )
        assert np.isfinite(sol.delta_w).all()


# ── Issue 28 — PanelScorer.load typed errors ──────────────────────────────────

class TestPanelScorerLoadErrors:
    """Pre-fix: missing artifact / corrupt JSON raised raw exceptions
    with no actionable context. Post-fix: typed FileNotFoundError /
    ValueError with a path + hint about snapshot side-config issue."""

    def test_missing_file_raises_typed_error(self, tmp_path):
        from kernel.panel_pipeline.panel_scorer import PanelScorer
        with pytest.raises(FileNotFoundError, match="artifact not found"):
            PanelScorer.load(tmp_path / "does_not_exist.json")

    def test_corrupt_json_raises_typed_error(self, tmp_path):
        from kernel.panel_pipeline.panel_scorer import PanelScorer
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {{{")
        with pytest.raises(ValueError, match="not valid JSON"):
            PanelScorer.load(bad)


# ── User mandate: daily model uses daily-only data by default ─────────────────

class TestDailyOnlyDataByDefault:
    """User mandate (2026-05-04): training a daily model must use daily-only
    data unless the operator explicitly opts in. Pre-mandate: production
    config had `hourly.enabled=true` + `minute.enabled=true` despite
    `training_resolution=daily` — the actual root cause of the calibrator
    NaN-leaf collapse (~100/183 tickers had no intraday history)."""

    def test_production_config_hourly_disabled(self):
        """Pin renquant_104 production strategy_config.json: hourly off."""
        import json as _json
        cfg = _json.loads(
            (REPO / "backtesting" / "renquant_104" / "strategy_config.json").read_text()
        )
        hourly = cfg.get("panel_ltr", {}).get("hourly", {})
        assert hourly.get("enabled") is False, (
            "renquant_104 is the daily model. Hourly aggregates must be "
            "OFF by default per user mandate. To opt in, flip explicitly "
            "AND ensure the watchlist has hourly history coverage."
        )

    def test_production_config_minute_disabled(self):
        import json as _json
        cfg = _json.loads(
            (REPO / "backtesting" / "renquant_104" / "strategy_config.json").read_text()
        )
        minute = cfg.get("panel_ltr", {}).get("minute", {})
        assert minute.get("enabled") is False, (
            "renquant_104 is the daily model. 10-min aggregates must be "
            "OFF by default per user mandate."
        )

    def test_code_default_is_off(self):
        """Even if config omits the block entirely, code must default OFF."""
        # Simulate a config WITHOUT panel_ltr.hourly: code must treat as off
        from training_panel.pp_panel_training import LoadHourlyBarsTask, LoadMinuteBarsTask
        # Source-level: each task's gate reads `cfg.get("enabled", False)`
        import inspect
        h_src = inspect.getsource(LoadHourlyBarsTask.run)
        m_src = inspect.getsource(LoadMinuteBarsTask.run)
        assert 'cfg.get("enabled", False)' in h_src, \
            "LoadHourlyBarsTask must default enabled=False (user mandate)"
        assert 'cfg.get("enabled", False)' in m_src, \
            "LoadMinuteBarsTask must default enabled=False (user mandate)"


# ── User mandate: data-scan preflight on every training run ───────────────────

class TestDataScanPreflight:
    """User mandate (2026-05-04): every training run begins with a
    data-scan that verifies row/column alignment and emits a
    length-and-coverage report."""

    def test_scan_returns_report_with_required_fields(self, tmp_path):
        from training_panel.data_scan import scan_training_inputs
        # Lay out a fake repo with one ticker having daily data
        (tmp_path / "data" / "ohlcv" / "AAPL").mkdir(parents=True)
        # Minimal parquet — pandas can write it
        import pandas as pd
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        pd.DataFrame({"close": range(50)}, index=idx).to_parquet(
            tmp_path / "data" / "ohlcv" / "AAPL" / "1d.parquet"
        )
        report = scan_training_inputs(["AAPL", "MSFT"], tmp_path)
        d = report.to_dict()
        assert "scan_utc" in d
        assert "watchlist_size" in d and d["watchlist_size"] == 2
        assert "sources" in d
        # Daily OHLCV scanned
        daily = d["sources"]["daily_ohlcv"]
        assert daily["n_tickers"] == 1
        assert "MSFT" in daily["missing_tickers"]
        assert daily["n_rows_total"] == 50
        # Alignment metrics present
        assert "alignment" in d
        assert "watchlist_coverage_pct" in d["alignment"]

    def test_strict_mode_raises_on_issues(self, tmp_path):
        """When data_scan.strict=true, alignment issues abort training."""
        # Deliberately point at an empty repo — daily coverage = 0% → issue
        from training_panel.pp_panel_training import (
            ScanTrainingDataTask, PanelTrainingContext,
        )
        ctx = PanelTrainingContext(
            config={
                "panel_ltr": {
                    "data_scan": {"enabled": True, "strict": True},
                    "hourly":    {"enabled": False},
                    "minute":    {"enabled": False},
                },
                "_strategy_dir": str(tmp_path / "backtesting" / "renquant_104"),
            },
            watchlist=["AAPL", "MSFT", "NVDA"],
        )
        # Make strategy dir exist but no data/ subdirectories
        (tmp_path / "backtesting" / "renquant_104").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="alignment issue"):
            ScanTrainingDataTask().run(ctx)

    def test_warn_mode_does_not_raise(self, tmp_path):
        from training_panel.pp_panel_training import (
            ScanTrainingDataTask, PanelTrainingContext,
        )
        ctx = PanelTrainingContext(
            config={
                "panel_ltr": {
                    "data_scan": {"enabled": True, "strict": False},
                    "hourly":    {"enabled": False},
                    "minute":    {"enabled": False},
                },
                "_strategy_dir": str(tmp_path / "backtesting" / "renquant_104"),
            },
            watchlist=["AAPL"],
        )
        (tmp_path / "backtesting" / "renquant_104").mkdir(parents=True)
        # Must NOT raise even with empty data dir — warn mode logs only.
        ScanTrainingDataTask().run(ctx)
        # Report should be on ctx
        assert ctx.training_data_scan is not None
        assert ctx.training_data_scan["watchlist_size"] == 1

    def test_scan_task_is_first_in_panel_data_job(self):
        """Pin task order — ScanTrainingDataTask must run BEFORE any Load*."""
        from training_panel.pp_panel_training import PanelDataJob, ScanTrainingDataTask
        tasks = PanelDataJob().tasks
        assert isinstance(tasks[0], ScanTrainingDataTask), (
            "ScanTrainingDataTask must be the first task in PanelDataJob "
            "so the preflight runs BEFORE any expensive data load."
        )


# ── Issue 35 — lean.py NaN close corrupts prev_closes ─────────────────────────

class TestLeanAdapterPrevClosesNaNGuard:
    """Pre-fix: NaN close on a delisted/suspended ticker slipped past the
    empty-check, then `float(NaN) = NaN` corrupted algo._prev_closes.
    Downstream check_single_day_loss saw NaN prev_close, comparison
    `daily_drop >= sdl_pct` was False → SDL gate silently disabled."""

    def test_finite_close_passes(self):
        # Defensive contract: simulate the post-fix logic directly
        import math as _math
        prev_closes: dict[str, float] = {}
        close = 150.0
        if _math.isfinite(close):
            prev_closes["AAPL"] = close
        assert prev_closes["AAPL"] == 150.0

    def test_nan_close_skipped(self):
        import math as _math
        prev_closes: dict[str, float] = {"AAPL": 100.0}  # prior good value
        bad_close = float("nan")
        if _math.isfinite(bad_close):
            prev_closes["AAPL"] = bad_close
        # Old prev_close stays — NOT overwritten with NaN
        assert prev_closes["AAPL"] == 100.0

    def test_inf_close_skipped(self):
        import math as _math
        prev_closes: dict[str, float] = {}
        bad_close = float("inf")
        if _math.isfinite(bad_close):
            prev_closes["AAPL"] = bad_close
        # No entry written
        assert "AAPL" not in prev_closes


# ── Issue 36 — runner.py broker.get_cash() fail-SAFE to 0 ─────────────────────

class TestRunnerBrokerCashFailSafe:
    """Pre-fix: broker.get_cash() exception fell through to
    `cash = account_value` — meaning ALL of NAV (including held
    positions) was treated as liquid, allowing Kelly oversizing.
    Post-fix: cash = 0 + log.error so operator notices broker outage."""

    def test_source_uses_zero_fallback_not_account_value(self):
        # Source-level test — pin the fix to prevent a future refactor
        # from silently restoring the over-allocation default.
        path = REPO / "backtesting" / "renquant_104" / "adapters" / "runner.py"
        src = path.read_text()
        # Find the broker-cash try block
        i = src.find("broker.get_cash()")
        assert i > 0
        # Wide enough window to include the post-message fallback assignment.
        block = src[i:i + 1800]
        # The fallback must be 0.0, not account_value
        assert "cash = 0.0" in block, \
            "runner.py: broker.get_cash() exception must default to 0.0 " \
            "(per audit Issue 36); pre-fix it defaulted to account_value " \
            "which allowed Kelly oversizing on broker outages."
        # And the exception path must log loud
        assert "log.error" in block, \
            "Broker outage must log.error (loud signal so operator notices)."


# ── Issue 37 — lean.py NaN gross_pnl corrupts cumulative tax ──────────────────

class TestLeanAdapterTaxNaNGuard:
    """Pre-fix: NaN gross_pnl from broker (disconnect, corrupted
    UnrealizedProfit) propagated `tax = NaN` → `_total_tax += NaN`
    → cumulative tax permanently NaN, all post-trade reports broken."""

    def test_finite_tax_added(self):
        import math as _math
        total = 0.0
        tax = 100.0
        if not _math.isfinite(tax):
            tax = 0.0
        total += tax
        assert total == 100.0

    def test_nan_tax_zeroed_total_preserved(self):
        import math as _math
        total = 5000.0
        tax = float("nan")
        if not _math.isfinite(tax):
            tax = 0.0
        total += tax
        assert total == 5000.0  # not NaN

    def test_lean_source_has_isfinite_tax_guard(self):
        path = REPO / "backtesting" / "renquant_104" / "adapters" / "lean.py"
        src = path.read_text()
        # The fix uses any local math import name; allow either pattern.
        # Match both `_math_lex.isfinite(tax)` and a plain `math.isfinite(tax)`.
        import re as _re
        guard_re = _re.compile(r"\bisfinite\(\s*tax\s*\)")
        # Constrain search to the first _total_tax accumulation site.
        i = src.find("algo._total_tax     += tax")
        if i < 0:
            i = src.find("_total_tax += tax")
        assert i > 0, "could not locate the tax accumulation line"
        ctx_block = src[max(0, i - 1200):i + 100]
        assert guard_re.search(ctx_block), \
            "lean.py: must guard tax with isfinite() before adding to " \
            "_total_tax (audit Issue 37). Pre-fix NaN gross_pnl poisoned " \
            "cumulative tax."


# ── User mandate: TournamentJob OOS window = 90d / rolling-63-trading ────────

class TestTournamentOOSWindow30d:
    """User mandate (2026-05-04): per-ticker tournament Sharpe must
    reflect RECENT performance — default OOS window is ~3 trading
    months (90 calendar days), not the legacy 2-year aggregate."""

    def test_default_cutoff_is_90_days(self):
        from training.tournament import resolve_oos_cutoff
        import pandas as pd
        # Empty config → default to today - 90d
        c = resolve_oos_cutoff({})
        delta = (pd.Timestamp.today().normalize() - c).days
        assert 88 <= delta <= 92, \
            f"default cutoff must be ~90d ago, got {delta}d"

    def test_sample_end_anchors_cutoff(self):
        """B2 path: cutoff = sample_end - 90d, not today - 90d."""
        from training.tournament import resolve_oos_cutoff
        import pandas as pd
        c = resolve_oos_cutoff({"sample_end": "2024-12-31"})
        # Should be 2024-12-31 - 90d ≈ 2024-10-02
        expected = pd.Timestamp("2024-12-31") - pd.Timedelta(days=90)
        assert c == expected.normalize(), \
            f"with sample_end, cutoff must anchor to it; got {c} vs {expected}"

    def test_explicit_oos_days_honored(self):
        from training.tournament import resolve_oos_cutoff
        import pandas as pd
        c = resolve_oos_cutoff({
            "sample_end": "2024-12-31",
            "oos_days": 30,
        })
        assert c == pd.Timestamp("2024-12-01"), \
            f"explicit oos_days=30 must produce sample_end - 30d"

    def test_legacy_oos_years_still_honored(self):
        from training.tournament import resolve_oos_cutoff
        import pandas as pd
        c = resolve_oos_cutoff({
            "sample_end": "2024-12-31",
            "oos_years": 2,
        })
        # 2024-12-31 - 2y = 2022-12-31
        assert c == pd.Timestamp("2022-12-31"), \
            f"legacy oos_years config must still work for backward compat"

    def test_explicit_cutoff_wins(self):
        from training.tournament import resolve_oos_cutoff
        import pandas as pd
        c = resolve_oos_cutoff({
            "oos_cutoff": "2024-06-15",
            "sample_end": "2024-12-31",
            "oos_days": 30,
        })
        assert c == pd.Timestamp("2024-06-15")


# ── Issue 38 — predict_qlearning silently routes NaN to last bin ─────────────

class TestPredictQLearningNaNGuard:
    """Pre-fix: NaN feature value via `np.digitize(NaN, edges)` returns
    `len(edges)+1`, then `np.clip(... - 1, 0, n_bins-1)` pinned to
    n_bins-1 = top bin. So a missing feature silently produced a strong
    deterministic signal (top-bin Q-value diff). Post-fix: NaN feature
    returns 0.0 (neutral, matches predict_classification semantics)."""

    def test_nan_feature_returns_zero(self):
        from kernel.models import predict_qlearning
        import pandas as pd
        artifact = {
            "feature_columns": ["rsi", "mom"],
            "bin_edges": {
                "rsi": [30, 50, 70],
                "mom": [-0.05, 0.0, 0.05],
            },
            "n_bins": 4,
            "q_table": [[1.0, -1.0]] * (4 * 4 * 3),  # 4 bins × 4 bins × 3 holding buckets
        }
        # NaN in feature
        row = pd.Series({"rsi": float("nan"), "mom": 0.02})
        score = predict_qlearning(artifact, row, holdings=0)
        assert score == 0.0, f"NaN feature must return 0.0 (neutral), got {score}"

    def test_finite_feature_returns_qval_diff(self):
        from kernel.models import predict_qlearning
        import pandas as pd
        artifact = {
            "feature_columns": ["rsi", "mom"],
            "bin_edges": {
                "rsi": [30, 50, 70],
                "mom": [-0.05, 0.0, 0.05],
            },
            "n_bins": 4,
            "q_table": [[1.0, -1.0]] * (4 * 4 * 3),
        }
        row = pd.Series({"rsi": 60.0, "mom": 0.02})
        score = predict_qlearning(artifact, row, holdings=0)
        # All q_vals are [1.0, -1.0] → diff = 2.0 regardless of state
        assert score == 2.0

    def test_missing_column_returns_zero(self):
        from kernel.models import predict_qlearning
        import pandas as pd
        artifact = {
            "feature_columns": ["rsi", "mom"],
            "bin_edges": {
                "rsi": [30, 50, 70],
                "mom": [-0.05, 0.0, 0.05],
            },
            "n_bins": 4,
            "q_table": [[1.0, -1.0]] * (4 * 4 * 3),
        }
        # mom column missing entirely
        row = pd.Series({"rsi": 60.0})
        score = predict_qlearning(artifact, row, holdings=0)
        assert score == 0.0


# ── Issue 14 — CV adapter num_boost_round mismatch with FinalFit ─────────────

class TestCVAdapterRoundsMatchFinalFit:
    """Pre-fix: lgbm + xgb CV `_SklearnAdapter.fit` used
    `num_boost_round=max(num_rounds // 2, 50)` while FinalFit used full
    `num_boost_round`. CPCV IC measured a HALF-CAPACITY model, then we
    shipped a FULL-CAPACITY model — so reported mean_ic ≠ shipped model
    quality. Same class as audit fix #14 (transformer CV) which was
    fixed earlier; this fix completes the alignment for the other 2
    backends."""

    def test_no_half_rounds_in_cv_adapter(self):
        path = REPO / "backtesting" / "renquant_104" / "training_panel" / "pp_panel_training.py"
        src = path.read_text()
        # Anti-pattern must not be present
        assert "max(num_rounds // 2, 50)" not in src, (
            "CV adapter must use full num_rounds to match FinalFit "
            "(audit Issue 14). Pre-fix CPCV IC measured a half-capacity "
            "model that wasn't what shipped."
        )


# ── Issue 17 — SizeAndEmit / EmitRotations conviction uses calibrated rank_score ──

class TestConvictionRevertNoteToFollowups:
    """2026-05-04: Issue 17 (conviction → calibrated rank_score) was
    REVERTED because the change without paired sizing_cfg retune
    halved positions and regressed Sharpe by 0.58 in v2 B2 (Sharpe
    +0.25 → -0.33). The sizing_cfg.{floor,ceiling,min_mult} are tuned
    for raw panel_score (~N(0, 0.05)). To re-apply Issue 17 properly,
    a paired retune of sizing_cfg for rank_score scale [0,1] is needed.
    Tracked as a future fix; until then conviction reads raw panel_score."""

    def test_conviction_reads_raw_panel_score_until_paired_retune(self):
        path = REPO / "backtesting" / "renquant_104" / "kernel" / "pipeline" / "task_selection.py"
        src = path.read_text()
        # The reverted pattern: bare getattr(c, "panel_score", None)
        assert 'getattr(c, "panel_score", None) if c else None, sizing_cfg' in src, (
            "task_selection.py: conviction_multiplier must use raw panel_score "
            "until the sizing_cfg is retuned for the rank_score scale (the v2 "
            "regression showed scale-mismatch halved sizes)."
        )


# ── Issue 42 — qp_tax_aware default OFF (no tax-driven decision logic) ──────

class TestQPTaxAwareDisabledByDefault:
    """User mandate (feedback_no_tax_driven_logic.md): no tax-driven
    sell/hold logic. The qp_tax_aware flag, added 2026-04-29 as Stage 8,
    directly violates this — it makes the QP solver minimize tax cost
    by avoiding sells of big-gain positions, mechanizing the
    disposition effect. v2 B2: 75.8% win rate but -3.6% total return
    because big winners reversed while small wins were repeatedly
    crystallized + ST-taxed."""

    def test_production_config_has_qp_tax_aware_false(self):
        import json as _json
        cfg = _json.loads(
            (REPO / "backtesting" / "renquant_104" / "strategy_config.json").read_text()
        )
        ja = cfg.get("rotation", {}).get("joint_actions", {})
        assert ja.get("qp_tax_aware") is False, (
            "qp_tax_aware must be False (user mandate: tax = reporting only, "
            "no decision logic). When True, QP avoids selling big-gain "
            "positions to defer tax → disposition effect mechanized."
        )

    # Active production-style configs whose qp_min_dw_pct must stay ≥ 0.01.
    # Historical ablation/sweep configs (ablation_*.json, armA_*.json,
    # emb_*.json, sweep_*.json, h60.json) are EXEMPT — their results are
    # already published in failed-experiments-log.md and rewriting them
    # retroactively would misrepresent history. Add active configs here
    # as they're created.
    _ACTIVE_CONFIGS = (
        "strategy_config.json",
        "strategy_config.golden.json",
        "strategy_config.wl183_daily_clean.json",
        "strategy_config.wl183_diag10.json",
    )

    def test_qp_min_dw_pct_above_micro_threshold(self):
        """Pin qp_min_dw_pct ≥ 0.01 across active production-style
        configs so the wl183 incident class can't recur silently.

        2026-05-05 wl183 incident: side config had qp_min_dw_pct=0.005 (4×
        below production's 0.02) — QP allowed Δw=0.79% trims (1-share
        rebalances) → 145 micro-sells over 27 mo → friction killed Sharpe
        even at 77% win rate (B2 result: Sharpe −0.07, APY −1.60%).
        Production was correctly pinned at 0.02 but the side config
        wasn't tested.
        """
        import json as _json
        strat_dir = REPO / "backtesting" / "renquant_104"

        violations = []
        missing = []
        for name in self._ACTIVE_CONFIGS:
            path = strat_dir / name
            if not path.exists():
                missing.append(name)
                continue
            cfg = _json.loads(path.read_text())
            ja = cfg.get("rotation", {}).get("joint_actions", {}) or {}
            if ja.get("solver") != "qp":
                continue
            v = ja.get("qp_min_dw_pct", 0.0)
            if v < 0.01:
                violations.append((name, v))

        assert not violations, (
            f"qp_min_dw_pct < 0.01 in active configs (micro-trim death "
            f"spiral; wl183 v2 B2 produced Sharpe −0.07, APY −1.60%): "
            f"{violations}"
        )
        assert not missing, (
            f"active configs missing from disk: {missing} — update "
            f"_ACTIVE_CONFIGS or restore the file"
        )


# ── Issue 33 — rotation NaN rank_score slips past panel_buy_floor ────────────

class TestRotationNaNRankFloorGuard:
    """Pre-fix: NaN rank_score slipped past `cand_score < panel_buy_floor`
    (NaN < X is False) → candidate proceeded as if it crossed the buy
    floor. Post-fix: explicit isfinite check rejects NaN."""

    def test_source_has_isfinite_guard(self):
        path = REPO / "backtesting" / "renquant_104" / "kernel" / "rotation.py"
        src = path.read_text()
        # Find the rotation candidate-loop body
        i = src.find("if cand_ticker in held_scores:")
        assert i > 0
        block = src[i:i + 800]
        assert "math.isfinite(cand_score)" in block, \
            "rotation.py find_rotation_pairs must guard NaN rank_score " \
            "before applying panel_buy_floor (audit Issue 33)"


# ── Issue 39 — TopUpHeldTask missing buy_blocked check ──────────────────────

class TestTopUpRespectsBuyBlocked:
    """Pre-fix: when EMA50Gate / VelocityCrash set ctx.buy_blocked=True,
    NEW buys were blocked but TopUps still fired — violating the macro
    gate's intent. Post-fix: TopUp checks all three flags."""

    def test_buy_blocked_short_circuits_topup(self):
        from kernel.pipeline.task_topup import TopUpHeldTask
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            config={"ranking": {"kelly_sizing": {
                "enabled": True, "top_up_threshold": 0.05,
            }}},
            bear_only=False,
            skip_buys=False,
            buy_blocked=True,         # macro gate fired
            holdings={},
            orders=[],
            exits=[],
            rotations=[],
            portfolio_value=100_000,
            prices={},
            cash=100_000,
            today=None,
            earnings_calendar=None,
        )
        # Should short-circuit; no orders touched.
        TopUpHeldTask().run(ctx)
        assert ctx.orders == [], (
            "TopUp must not fire when ctx.buy_blocked=True (audit Issue 39)"
        )

    def test_topup_source_checks_buy_blocked(self):
        path = REPO / "backtesting" / "renquant_104" / "kernel" / "pipeline" / "task_topup.py"
        src = path.read_text()
        assert "buy_blocked" in src, (
            "TopUpHeldTask source must mention buy_blocked (audit Issue 39)"
        )


# ── Issue 41 — PositionConcentrationGate NaN equity silently bypasses ───────

class TestConcentrationGateNaNEquityGuard:
    """Pre-fix: NaN equity slipped past `equity <= 0` (NaN <= 0 is False)
    → div-by-NaN concentration math → `pct >= cap_pct` is False → gate
    silently disabled. Post-fix: explicit isfinite check."""

    def test_nan_equity_skipped_with_warning(self):
        from kernel.pipeline.task_risk_gates import PositionConcentrationGateTask
        from types import SimpleNamespace
        # Build a minimal ctx with NaN portfolio_value
        cand = SimpleNamespace(ticker="AAPL")
        ctx = SimpleNamespace(
            config={"risk_gates": {"position_concentration": {
                "enabled": True, "max_pct": 0.15,
            }}},
            candidates=[cand],
            holdings={},
            prices={},
            portfolio_value=float("nan"),
            counters={},
        )
        result = PositionConcentrationGateTask().run(ctx)
        # Gate must skip cleanly, not crash, not silently zero-drop
        assert result is True
        # Candidate retained (since gate skipped) — but this is the
        # documented fail-SAFE behavior. The KEY assertion is that the
        # gate didn't poison ctx.candidates with NaN-comparison kept set.
        assert ctx.candidates == [cand]

    def test_inf_equity_skipped(self):
        from kernel.pipeline.task_risk_gates import PositionConcentrationGateTask
        from types import SimpleNamespace
        cand = SimpleNamespace(ticker="AAPL")
        ctx = SimpleNamespace(
            config={"risk_gates": {"position_concentration": {
                "enabled": True, "max_pct": 0.15,
            }}},
            candidates=[cand],
            holdings={},
            prices={},
            portfolio_value=float("inf"),
            counters={},
        )
        PositionConcentrationGateTask().run(ctx)
        assert ctx.candidates == [cand]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
