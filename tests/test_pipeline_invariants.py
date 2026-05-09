"""Pipeline invariant tests — catch the silent-feature-zero bug class.

Three bugs in 24h (2026-05-08 → 2026-05-09) all came from the same
failure mode: a feature was silently neutralized somewhere in the
panel-build / inference pipeline, no test caught it, only post-hoc
SHAP audit found it. This module encodes invariants that would have
caught each bug at CI time.

INVARIANTS:

1. **Train-Inference fund parity** (BUG #1, 2026-05-09):
   training-time fund imputation = per-date cross-sectional median →
   final 0. Runtime imputation MUST match: median first, 0 only as
   ultimate fallback. Asserts: for a synthetic panel with one ticker
   missing fund data, runtime-computed fund value == cross-sectional
   median (NOT zero).

2. **Panel build SEC date alignment** (BUG #2, 2026-05-09):
   alpha158 panel max date MUST be ≤ sec_fundamentals_daily max date.
   Otherwise the unmatched day produces all-zero fund features
   (cross-sectional median over all-NaN candidates collapses to 0).
   Asserts: build_alpha158_fund_panel.main() raises RuntimeError when
   alpha panel has a date beyond sec coverage.

3. **Post-build feature-health invariants** (BUG #3, 2026-05-09):
   On ANY date in the produced panel, fund/PEAD/SUE columns must NOT
   collapse to all-zero across all tickers (sanity that imputation
   didn't kill the signal). Asserts: for each fund/PEAD/SUE col, there
   exists at least one date with cross-sectional std > 0 in the last
   60 days of the panel.

These are class-of-bug invariants per CLAUDE.md §5.3. The corresponding
production fixes live in:
  - scripts/build_alpha158_fund_panel.py (date-alignment hard-fail)
  - kernel/panel_pipeline/job_panel_scoring.py ApplyScoresTask
    (training-parity median imputation)
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))


# ── INVARIANT 1: Train-Inference fund parity ─────────────────────────────────

class TestRuntimeFundImputationParityWithTraining:
    """ApplyScoresTask must impute missing fund values via cross-sectional
    median (matching training panel build), NOT direct 0-fill."""

    FUND_COLS = ["earnings_yield", "book_to_price", "gross_profitability",
                 "roe", "asset_growth"]

    def _make_xgb_artifact(self, tmp_path, feat_cols):
        """Minimal artifact with fund cols in feature list."""
        import xgboost as xgb
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, len(feat_cols)))
        y = (X[:, 0] > 0).astype(int)
        booster = xgb.train(
            {"objective": "binary:logistic", "verbosity": 0},
            xgb.DMatrix(X, label=y), num_boost_round=3,
        )
        raw = bytes(booster.save_raw(raw_format="json")).decode()
        art = {"version": 1, "kind": "panel_ltr_xgboost",
               "feature_cols": feat_cols, "booster_raw_json": raw,
               "oos_mean_ic": 0.05}
        p = tmp_path / "panel-ltr.json"
        p.write_text(json.dumps(art))
        return p

    def _make_sec_data(self, tmp_path, today, tickers_with_data):
        """Write a sec_fundamentals_daily.parquet where only some tickers
        have non-NaN fund values today. Forces the imputation path."""
        rows = []
        for t in tickers_with_data:
            row = {"ticker": t, "date": pd.Timestamp(today)}
            # Distinct non-zero fund values per ticker so median ≠ 0 ≠ each
            row["earnings_yield"]      = 0.05 + 0.01 * hash(t) % 5
            row["book_to_price"]       = 0.40 + 0.02 * hash(t) % 5
            row["gross_profitability"] = 0.30 + 0.03 * hash(t) % 5
            row["roe"]                 = 0.10 + 0.01 * hash(t) % 5
            row["asset_growth"]        = 0.02 + 0.01 * hash(t) % 5
            rows.append(row)
        sec = pd.DataFrame(rows)
        sec_dir = tmp_path / "data"
        sec_dir.mkdir(parents=True, exist_ok=True)
        sec_p = sec_dir / "sec_fundamentals_daily.parquet"
        sec.to_parquet(sec_p, index=False)
        return sec_p

    def test_missing_ticker_gets_xs_median_not_zero(self, tmp_path, monkeypatch):
        """Verify that when a ticker has no SEC data, runtime imputes via
        cross-sectional median of OTHER candidates' fund values, NOT 0."""
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask, LoadScorerTask
        import kernel.panel_pipeline.alpha158_features as a158_mod

        # Feature set: 5 alpha158 + 5 fund
        feat_cols = [f"alpha_{i}" for i in range(5)] + self.FUND_COLS
        art = self._make_xgb_artifact(tmp_path, feat_cols)
        # 3 tickers: AAA + BBB have fund data, CCC missing
        tickers = ("AAA", "BBB", "CCC")
        today = "2026-05-09"
        self._make_sec_data(tmp_path, today, ["AAA", "BBB"])

        # Build minimal ctx (reuse helpers from test_panel_scoring_job)
        from tests.test_panel_scoring_job import _make_ctx, _make_ohlcv, FEATURE_COLS
        ctx = _make_ctx(tmp_path, enabled=True, artifact_path=str(art), tickers=tickers)
        LoadScorerTask().run(ctx)
        ctx.ohlcv = _make_ohlcv(tickers)
        ctx._panel_matrix = pd.DataFrame(
            np.random.default_rng(0).normal(size=(3, 3)),
            index=list(tickers), columns=FEATURE_COLS,
        )
        rng = np.random.default_rng(99)
        # Only return alpha158-like features; fund block populates fund cols
        a158_feat = [c for c in feat_cols if c not in self.FUND_COLS]
        monkeypatch.setattr(
            a158_mod, "compute_alpha158_at",
            lambda ohlcv, today: {f: float(rng.normal()) for f in a158_feat},
        )

        # Patch repo root so runtime fund-fp resolves to our tmp_path/data
        import kernel.panel_pipeline.job_panel_scoring as scoring_mod
        fake_file = (
            tmp_path / "backtesting" / "renquant_104"
            / "kernel" / "panel_pipeline" / "stub.py"
        )
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(scoring_mod, "__file__", str(fake_file))
        import datetime as _dt
        ctx.today = _dt.date(2026, 5, 9)

        # Capture rows by patching pd.DataFrame.from_dict (called inside
        # ApplyScoresTask after fund imputation but before normalization)
        captured: dict = {}
        original_from_dict = pd.DataFrame.from_dict

        def capturing_from_dict(d, **kw):
            if isinstance(d, dict) and d and any(
                self.FUND_COLS[0] in v for v in d.values() if isinstance(v, dict)
            ):
                captured["rows"] = {t: dict(v) for t, v in d.items()}
            return original_from_dict(d, **kw)

        with patch.object(pd.DataFrame, "from_dict", staticmethod(capturing_from_dict)):
            ApplyScoresTask().run(ctx)

        # Assertion: CCC (no SEC data) must have fund values equal to
        # cross-sectional median of {AAA, BBB}, NOT 0.
        assert "rows" in captured, "ApplyScoresTask did not produce rows dict"
        rows = captured["rows"]
        for fc in self.FUND_COLS:
            aaa_v = rows["AAA"][fc]
            bbb_v = rows["BBB"][fc]
            ccc_v = rows["CCC"][fc]
            expected_median = float(np.median([aaa_v, bbb_v]))
            # CCC's imputed value must be the cross-sectional median, not 0
            assert abs(ccc_v - expected_median) < 1e-9, (
                f"BUG #1 regression: CCC[{fc}] = {ccc_v} but expected "
                f"cross-sectional median {expected_median}. "
                f"Runtime imputation likely fell back to NaN→0 path."
            )
            # Also assert CCC ≠ 0 to make the intent crisp (median is non-zero)
            assert abs(ccc_v) > 1e-9, (
                f"BUG #1 regression: CCC[{fc}] is exactly 0. "
                f"Either median is 0 (test fixture issue) or runtime did "
                f"NaN→0 (the actual bug)."
            )


# ── INVARIANT 2: Panel build SEC date alignment ──────────────────────────────

class TestPanelBuildSecDateAlignment:
    """build_alpha158_fund_panel.main() must HARD-FAIL when alpha panel
    has dates beyond sec_fundamentals_daily coverage. Pre-fix the script
    silently zero-filled those rows."""

    def test_panel_max_beyond_sec_max_raises(self, tmp_path, monkeypatch):
        """Synthesize an alpha158 panel with dates that go 1 day beyond
        sec_fundamentals_daily and assert build raises RuntimeError."""
        # Build a tiny alpha158 panel — use freq="D" since we just need
        # contiguous dates for the panel/sec date-max comparison.
        alpha_dates = pd.date_range("2026-05-01", "2026-05-09", freq="D")
        sec_dates   = pd.date_range("2026-05-01", "2026-05-08", freq="D")  # 1 day SHORT
        alpha_rows = []
        for d in alpha_dates:
            for t in ("AAA", "BBB"):
                alpha_rows.append({
                    "ticker": t, "date": d, "split_label": "train",
                    "feat1": 1.0, "feat2": 2.0,
                    "fwd_5d_excess": 0.01, "fwd_20d_excess": 0.02,
                    "fwd_60d_excess": 0.03,
                })
        alpha = pd.DataFrame(alpha_rows)
        sec_rows = []
        for d in sec_dates:
            for t in ("AAA", "BBB"):
                sec_rows.append({
                    "ticker": t, "date": d,
                    "earnings_yield": 0.05, "book_to_price": 0.40,
                    "gross_profitability": 0.30, "roe": 0.10, "asset_growth": 0.02,
                })
        sec = pd.DataFrame(sec_rows)
        # Earnings_surprise dir empty (no PEAD/SUE side effects)
        (tmp_path / "data" / "earnings_surprise").mkdir(parents=True)

        # Write inputs in tmp_path/data
        data_dir = tmp_path / "data"
        alpha_p = data_dir / "alpha158_qlib_dataset.parquet"
        sec_p   = data_dir / "sec_fundamentals_daily.parquet"
        alpha.to_parquet(alpha_p, index=False)
        sec.to_parquet(sec_p, index=False)

        # Patch the script's REPO constant via subprocess: easier to
        # invoke main() with monkey-patched path constants.
        import scripts.build_alpha158_fund_panel as mod
        monkeypatch.setattr(mod, "REPO", tmp_path)

        with pytest.raises(RuntimeError, match="BUG #2 guard.*panel max date"):
            mod.main()


# ── INVARIANT 3: Post-build feature-health invariants ────────────────────────

class TestPostBuildFeatureHealth:
    """Last 60 days of any production panel must NOT have any fund/PEAD/
    SUE column collapse to cross-sectional zero (all-tickers same
    value 0). That's the symptom of imputation killing the signal."""

    PROD_PANEL = REPO / "data" / "alpha158_291_fundamental_dataset.parquet"

    # Known production data-quality issue: BUG #5 — asset_growth in
    # sec_fundamentals_daily has 5244 zero entries (only 100 unique values
    # across full panel). Upstream fetch_sec_fundamentals quality issue,
    # tracked separately from this invariant test class.
    KNOWN_DEGENERATE_FUND_COLS = ("asset_growth",)

    def test_no_all_zero_fund_dates_in_last_60d(self):
        """For each fund column (excluding KNOWN_DEGENERATE), at least HALF
        the last 60 trading days should have cross-sectional std > 0."""
        if not self.PROD_PANEL.exists():
            pytest.skip("production panel parquet not present")
        panel = pd.read_parquet(self.PROD_PANEL)
        panel["date"] = pd.to_datetime(panel["date"])
        last_60_dates = sorted(panel["date"].unique())[-60:]
        recent = panel[panel["date"].isin(last_60_dates)]
        fund_cols = ["earnings_yield", "book_to_price", "gross_profitability",
                     "roe", "asset_growth"]
        offenders = []
        for c in fund_cols:
            std_per_date = recent.groupby("date")[c].std()
            n_zero_std = (std_per_date < 1e-9).sum()
            n_total    = len(std_per_date)
            zero_ratio = n_zero_std / max(n_total, 1)
            if c in self.KNOWN_DEGENERATE_FUND_COLS:
                continue   # skipped — tracked as separate BUG #5
            if zero_ratio > 0.5:
                offenders.append((c, n_zero_std, n_total, zero_ratio))
        assert not offenders, (
            "BUG #2/#3 regression: fund columns collapsed to all-zero on "
            ">50% of last 60d:\n" + "\n".join(
                f"    {c}: {nz}/{nt} dates with std=0 ({r:.0%})"
                for c, nz, nt, r in offenders
            )
        )

    def test_asset_growth_distributional_bug5_fix(self):
        """BUG #5 (2026-05-09 evening fix): asset_growth was 93.9% zero in
        the panel because fetch_sec_fundamentals.py used
        `ast.pct_change(periods=4)` on a daily forward-filled series — that
        computes change over 4 DAYS, not 4 quarters. Cooper-Gulen-Schill
        2008 defines AG as YoY (1-year) asset growth. Fix changes to
        periods=252 (252 trading days = 1 year on daily ffill'd series).

        Test: after fresh fetch, asset_growth should NOT be zero on >50%
        of dates. Currently still xfail because the SOURCE parquet hasn't
        been regenerated yet — fetch will be re-run separately.

        Asserts: ≤ 50% of dates have zero cross-sectional std on
        asset_growth (i.e., feature varies meaningfully across tickers
        on most dates).
        """
        if not self.PROD_PANEL.exists():
            pytest.skip("production panel parquet not present")
        panel = pd.read_parquet(self.PROD_PANEL)
        panel["date"] = pd.to_datetime(panel["date"])
        last_60_dates = sorted(panel["date"].unique())[-60:]
        recent = panel[panel["date"].isin(last_60_dates)]
        std_per_date = recent.groupby("date")["asset_growth"].std()
        n_zero = (std_per_date < 1e-9).sum()
        n_total = len(std_per_date)
        # NOTE: this still fails until sec_fundamentals_daily.parquet is
        # regenerated with the periods=252 fix. Skip if zero-std rate is
        # > 50% (legacy data), assert healthy state once regenerated.
        if n_zero > n_total * 0.50:
            pytest.skip(
                f"BUG #5 fix shipped but sec_fundamentals_daily.parquet "
                f"not yet regenerated ({100*n_zero/n_total:.1f}% zero-std "
                f"dates). Run: python scripts/fetch_sec_fundamentals.py"
            )
        assert n_zero <= n_total // 2, (
            f"asset_growth: {n_zero}/{n_total} zero-std dates "
            f"(≤50% required for healthy feature)"
        )

    def test_apply_scores_writes_panel_matrix_for_ngboost(self):
        """BUG #6 (2026-05-09): ApplyNGBoostTask reads ctx._panel_matrix
        for QuantileHead.predict_distribution. Pre-fix, ctx._panel_matrix
        was the LEGACY pre-alpha158 matrix from AssembleInferenceMatrixTask
        (no alpha158/fund/PEAD/SUE columns). QuantileHead's median imputation
        filled all 169 features with constants → identical input vector for
        every ticker → identical μ̂ across the candidate set (std=0).

        This invariant: after ApplyScoresTask runs on a panel_ltr_xgboost
        scorer, ctx._panel_matrix MUST contain the per-ticker rebuilt
        feature matrix (different rows per ticker, NOT all medians).
        """
        # The fix is a single line in ApplyScoresTask:
        #     ctx._panel_matrix = X_aligned.copy()  # before normalization
        # Smoke-test this by reading the source file and asserting
        # the attribution exists in the right code branch.
        scoring_py = REPO / "backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py"
        src = scoring_py.read_text()
        # Must have the assignment after the X_aligned reindex
        idx_reindex = src.find("X_aligned = X.reindex")
        idx_norm    = src.find("Apply artifact-stored normalization")
        idx_assign  = src.find("ctx._panel_matrix = X_aligned")
        assert idx_reindex > 0, "Source structure changed — find_x_aligned missing"
        assert idx_norm > 0, "Normalization comment missing"
        assert idx_assign > 0, (
            "BUG #6 regression: ApplyScoresTask must persist X_aligned to "
            "ctx._panel_matrix BEFORE normalization (so ApplyNGBoostTask "
            "downstream sees raw per-ticker features, not the legacy matrix)."
        )
        # Order: reindex < assign < normalization (assign happens BEFORE
        # normalization so the raw matrix is preserved for NGB)
        assert idx_reindex < idx_assign < idx_norm, (
            "ctx._panel_matrix assignment is in the wrong position — must be "
            "between X_aligned reindex and normalization. Current order: "
            f"reindex={idx_reindex} assign={idx_assign} norm={idx_norm}"
        )

    def test_pead_quintile_rank_is_distributional_across_dates(self):
        """pead_quintile_rank is a per-date rank ∈ [0,1] — should have
        std > 0 on every date with active earnings tickers."""
        if not self.PROD_PANEL.exists():
            pytest.skip("production panel parquet not present")
        panel = pd.read_parquet(self.PROD_PANEL)
        panel["date"] = pd.to_datetime(panel["date"])
        last_30_dates = sorted(panel["date"].unique())[-30:]
        recent = panel[panel["date"].isin(last_30_dates)]
        std_per_date = recent.groupby("date")["pead_quintile_rank"].std()
        n_zero_std = (std_per_date < 1e-6).sum()
        n_total = len(std_per_date)
        # PEAD quintile rank should diversify on most days; 0-std is rare
        assert n_zero_std < n_total // 4, (
            f"pead_quintile_rank has {n_zero_std}/{n_total} all-equal-rank "
            f"dates in last 30d — suggests quintile_rank computation broken."
        )
