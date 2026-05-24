"""Tests for training/features.py, training/tournament.py, training/export.py."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add strategy dir so kernel.* and training.* are importable
_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open":   close * (1 - rng.uniform(0, 0.005, n)),
        "high":   close * (1 + rng.uniform(0, 0.010, n)),
        "low":    close * (1 - rng.uniform(0, 0.010, n)),
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _make_ohlcv_dict(tickers=("AAPL", "SPY"), n: int = 400) -> dict:
    return {t: _make_ohlcv(n, seed=i) for i, t in enumerate(tickers)}


_INDICATOR_SPEC: dict = {}
_LOOKAHEAD = 5
_THRESHOLD = 0.02


# ── training/features ─────────────────────────────────────────────────────────

class TestBuildTrainingFeatures:
    def test_returns_dataframe_with_expected_columns(self):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict()
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        assert df is not None
        for col in ["close", "fwd_return", "label", "spy_realized_vol", "trend"]:
            assert col in df.columns, f"missing column: {col}"

    def test_labels_are_minus_one_zero_or_one(self):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict()
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        assert set(df["label"].unique()).issubset({-1, 0, 1})

    def test_no_nan_in_result(self):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict()
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        assert not df.isna().any().any()

    def test_close_is_relative_to_spy(self):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict()
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        # close = stock/SPY*100 → should typically be in ~50–200 range, not thousands
        assert df["close"].max() < 1000

    def test_duplicate_dates_are_deduped_before_alignment(self):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict()
        ohlcv["AAPL"] = pd.concat([ohlcv["AAPL"], ohlcv["AAPL"].iloc[[120]]])
        ohlcv["SPY"] = pd.concat([ohlcv["SPY"], ohlcv["SPY"].iloc[[90]]])

        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)

        assert df is not None
        assert not df.index.has_duplicates

    def test_missing_ticker_returns_none(self):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict()
        assert build_training_features("NVDA", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD) is None

    def test_missing_spy_returns_none(self):
        from training.features import build_training_features
        ohlcv = {"AAPL": _make_ohlcv()}
        assert build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD) is None

    def test_build_all_skips_missing_tickers(self):
        from training.features import build_all_training_features
        ohlcv = _make_ohlcv_dict()
        frames = build_all_training_features(["AAPL", "MISSING"], ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        assert "AAPL" in frames
        assert "MISSING" not in frames

    def test_build_all_returns_nonempty_frames(self):
        from training.features import build_all_training_features
        ohlcv = _make_ohlcv_dict(["AAPL", "GOOG", "SPY"])
        frames = build_all_training_features(["AAPL", "GOOG"], ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        assert len(frames) == 2
        for df in frames.values():
            assert len(df) > 0

    def test_build_all_dedupes_shared_spy_before_parallel_build(self):
        from training.features import build_all_training_features
        ohlcv = _make_ohlcv_dict(["AAPL", "GOOG", "SPY"])
        ohlcv["SPY"] = pd.concat([ohlcv["SPY"], ohlcv["SPY"].iloc[[75]]])

        frames = build_all_training_features(["AAPL", "GOOG"], ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)

        assert set(frames) == {"AAPL", "GOOG"}
        assert all(not df.index.has_duplicates for df in frames.values())


# ── training/tournament ───────────────────────────────────────────────────────

class TestOosSharpe:
    def test_flat_returns_give_zero_sharpe(self):
        from training.tournament import oos_sharpe
        idx = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.Series(np.ones(100) * 100.0, index=idx)
        sigs   = pd.Series(np.ones(100), index=idx)
        assert oos_sharpe(prices, sigs) == 0.0

    def test_positive_trend_gives_positive_sharpe(self):
        from training.tournament import oos_sharpe
        idx = pd.bdate_range("2020-01-01", periods=200)
        prices = pd.Series(np.linspace(100, 200, 200), index=idx)
        sigs   = pd.Series(np.ones(200), index=idx)
        assert oos_sharpe(prices, sigs) > 0

    def test_short_series_returns_zero(self):
        from training.tournament import oos_sharpe
        idx = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.Series([100.0] * 5, index=idx)
        sigs   = pd.Series([1.0] * 5, index=idx)
        assert oos_sharpe(prices, sigs) == 0.0


class TestRunTournamentSmoke:
    """Smoke tests — just verifies the function runs and returns correct structure."""

    def _make_df_with_labels(self, n=400):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict(["AAPL", "SPY"], n=n)
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        return df, ohlcv

    def test_returns_required_keys(self):
        from training.tournament import run_tournament
        df, ohlcv = self._make_df_with_labels()
        if df is None or len(df[df.index >= pd.Timestamp("2024-01-01")]) < 30:
            pytest.skip("insufficient OOS rows for this synthetic series")
        result = run_tournament(
            "AAPL", df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            {"feature_columns": ["rsi", "macd_hist"], "lookahead": 5, "threshold": 0.02,
             "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1},
            sharpe_floor=0.0,
            tax_config={"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365},
        )
        for key in ["sharpe", "best_approach", "model", "oos_signals",
                    "oos_raw_scores", "passes_floor", "train_rows", "oos_rows"]:
            assert key in result, f"missing key: {key}"

    def test_passes_floor_true_when_floor_is_zero(self):
        from training.tournament import run_tournament
        df, ohlcv = self._make_df_with_labels()
        if df is None or len(df[df.index >= pd.Timestamp("2024-01-01")]) < 30:
            pytest.skip("insufficient OOS rows")
        result = run_tournament(
            "AAPL", df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            {"feature_columns": ["rsi", "macd_hist"], "lookahead": 5, "threshold": 0.02,
             "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1},
            sharpe_floor=0.0,
            tax_config={"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365},
        )
        assert result["passes_floor"] is True

    def test_insufficient_data_returns_safe_defaults(self):
        from training.tournament import run_tournament
        ohlcv = _make_ohlcv_dict()
        tiny_df = pd.DataFrame({"label": [1] * 10}, index=pd.bdate_range("2023-01-01", periods=10))
        result = run_tournament(
            "AAPL", tiny_df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            {"feature_columns": ["rsi"], "lookahead": 5, "threshold": 0.02,
             "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1},
            sharpe_floor=0.8,
            tax_config={"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365},
        )
        assert result["passes_floor"] is False
        assert result["model"] is None

    def test_excludes_qlearning_when_configured(self):
        """exclude_models={'qlearning'} drops the QLearning approach from the tournament log."""
        from training.tournament import run_tournament
        df, ohlcv = self._make_df_with_labels()
        if df is None:
            pytest.skip("insufficient training data")
        # Pick a cutoff inside the synthetic window so we always have OOS rows.
        mid = df.index[len(df) // 2]
        result = run_tournament(
            "AAPL", df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            {"feature_columns": ["rsi", "macd_hist"], "lookahead": 5, "threshold": 0.02,
             "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1},
            sharpe_floor=0.0,
            tax_config={"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365},
            oos_cutoff=mid,
            exclude_models={"qlearning"},
        )
        log_blob = " ".join(result.get("_log", []))
        assert "QLearning" not in log_blob
        # Tournament still produces *some* best_approach (classification / xgboost / manual)
        assert result["best_approach"] in {"Classification", "XGBoost", "Manual", None}

    def test_excludes_multiple_models(self):
        """exclude_models={'qlearning','manual'} drops both approaches."""
        from training.tournament import run_tournament
        df, ohlcv = self._make_df_with_labels()
        if df is None:
            pytest.skip("insufficient training data")
        mid = df.index[len(df) // 2]
        result = run_tournament(
            "AAPL", df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            {"feature_columns": ["rsi", "macd_hist"], "lookahead": 5, "threshold": 0.02,
             "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1},
            sharpe_floor=0.0,
            tax_config={"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365},
            oos_cutoff=mid,
            exclude_models={"qlearning", "manual"},
        )
        log_blob = " ".join(result.get("_log", []))
        assert "QLearning" not in log_blob
        assert "Manual" not in log_blob
        assert result["best_approach"] in {"Classification", "XGBoost", None}


class TestTournamentWinnerMetric:
    """winner_metric='ic' selects by Spearman IC instead of Sharpe."""

    def _make_df_with_labels(self, n=400):
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict(["AAPL", "SPY"], n=n)
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        return df, ohlcv

    def test_default_is_sharpe(self):
        """No winner_metric arg → uses Sharpe; selection_metric=sharpe in result."""
        from training.tournament import run_tournament
        df, ohlcv = self._make_df_with_labels()
        if df is None:
            pytest.skip("insufficient training data")
        mid = df.index[len(df) // 2]
        result = run_tournament(
            "AAPL", df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            {"feature_columns": ["rsi", "macd_hist"], "lookahead": 5, "threshold": 0.02,
             "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1},
            sharpe_floor=0.0,
            tax_config={"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365},
            oos_cutoff=mid,
        )
        assert result["selection_metric"] == "sharpe"
        # selection_score should equal the tracked sharpe
        assert result["selection_score"] == result["sharpe"]

    def test_ic_metric_runs_and_reports(self):
        """With winner_metric='ic', result exposes selection_metric='ic' + log mentions IC."""
        from training.tournament import run_tournament
        df, ohlcv = self._make_df_with_labels()
        if df is None:
            pytest.skip("insufficient training data")
        mid = df.index[len(df) // 2]
        result = run_tournament(
            "AAPL", df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            {"feature_columns": ["rsi", "macd_hist"], "lookahead": 5, "threshold": 0.02,
             "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1},
            sharpe_floor=-99,   # make sure we pass floor no matter what
            tax_config={"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365},
            oos_cutoff=mid,
            winner_metric="ic",
        )
        assert result["selection_metric"] == "ic"
        log_blob = " ".join(result.get("_log", []))
        assert "IC:" in log_blob    # per-approach IC printed
        # selection_score is IC — a real-valued correlation, bounded [-1, 1]
        assert -1.0 <= result["selection_score"] <= 1.0

    def test_oos_single_ticker_ic_on_perfect_signal(self):
        """IC should be ~1.0 when raw_scores equal future relative return (perfect foresight)."""
        from training.tournament import oos_single_ticker_ic
        idx = pd.bdate_range("2024-01-01", periods=60)
        spy = pd.Series(100.0, index=idx)
        # Stock grows linearly; compute its 5-day forward relative return and
        # use that directly as raw_score → IC must be 1.0 by construction.
        stock = pd.Series(np.linspace(100.0, 150.0, 60), index=idx)
        rel = stock / spy
        fwd = (rel.shift(-5) / rel - 1.0).dropna()
        raw = fwd.copy()          # perfect foresight
        ic = oos_single_ticker_ic(raw, stock, spy, lookahead=5)
        assert ic > 0.95, f"expected near-perfect IC, got {ic:.3f}"

    def test_oos_single_ticker_ic_on_noise(self):
        """IC should be near 0 when raw_scores are random noise."""
        from training.tournament import oos_single_ticker_ic
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2024-01-01", periods=200)
        spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=idx)
        stock = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 200))), index=idx)
        raw = pd.Series(rng.normal(size=200), index=idx)
        ic = oos_single_ticker_ic(raw, stock, spy, lookahead=5)
        assert abs(ic) < 0.25, f"random signal should yield IC near 0, got {ic:.3f}"


# ── training/export ───────────────────────────────────────────────────────────

class TestExportModels:
    def _fake_results(self, tmpdir, passes=True):
        """Build a minimal results dict with a real trained model."""
        from training.features import build_training_features
        from training.models import create_model
        ohlcv = _make_ohlcv_dict(["AAPL", "SPY"])
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        model = create_model("classification", feature_columns=["rsi", "macd_hist"],
                             lookahead=5, threshold=0.02, leaf_size=5, bags=3)
        if df is not None:
            model.train(df)
        return {
            "AAPL": {
                "sharpe": 1.5 if passes else 0.3,
                "best_approach": "Classification",
                "model": model,
                "passes_floor": passes,
                "score_calibration": None,
                "oos_signals": None,
                "oos_raw_scores": None,
            }
        }

    def test_export_creates_model_directory(self):
        from training.export import export_models
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            results = self._fake_results(tmpdir, passes=True)
            exported, skipped = export_models(results, strategy_dir, "2025-01-01", 5, "renquant_104")
            assert "AAPL" in exported
            assert (strategy_dir / "models" / "AAPL").is_dir()

    def test_export_ignores_passes_floor(self):
        """Export no longer gates on passes_floor — admission moved to LoadUniverseJob."""
        from training.export import export_models
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            results = self._fake_results(tmpdir, passes=False)
            exported, skipped = export_models(results, strategy_dir, "2025-01-01", 5, "renquant_104")
            assert "AAPL" in exported
            assert "AAPL" not in skipped

    def test_missing_model_goes_to_skipped(self):
        from training.export import export_models
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            results = {"AAPL": {"sharpe": 1.5, "best_approach": "x", "model": None,
                                "passes_floor": True}}
            exported, skipped = export_models(results, strategy_dir, "2025-01-01", 5, "renquant_104")
            assert "AAPL" not in exported
            assert "AAPL" in skipped

    def test_metadata_patched_with_trained_date(self):
        from training.export import export_models
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            results = self._fake_results(tmpdir, passes=True)
            export_models(results, strategy_dir, "2025-04-19", 5, "renquant_104")
            meta_path = strategy_dir / "models" / "AAPL" / "AAPL-policy-metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta["trained_date"] == "2025-04-19"
                assert meta["best_approach"] == "Classification"


class TestRetrainLiveModels:
    def test_live_refresh_runs_without_error(self):
        from training.features import build_training_features
        from training.models import create_model
        from training.export import export_models, retrain_live_models

        ohlcv = _make_ohlcv_dict(["AAPL", "SPY"])
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        model = create_model("classification", feature_columns=["rsi", "macd_hist"],
                             lookahead=5, threshold=0.02, leaf_size=5, bags=3)
        if df is not None:
            model.train(df)

        results = {
            "AAPL": {
                "sharpe": 1.5, "best_approach": "Classification", "model": model,
                "passes_floor": True, "score_calibration": None,
                "oos_signals": None, "oos_raw_scores": None,
            }
        }
        model_params = {"feature_columns": ["rsi", "macd_hist"], "lookahead": 5, "threshold": 0.02,
                        "bags": 3, "leaf_size": 5, "buy_threshold": 0.1, "sell_threshold": -0.1}
        config = {"strategy": "renquant_104", "tax": {"short_term_rate": 0.4, "long_term_rate": 0.2,
                                                       "long_term_threshold_days": 365}}
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            exported, _ = export_models(results, strategy_dir, "2025-01-01", 5, "renquant_104")
            retrain_live_models(
                results, {"AAPL": df} if df is not None else {},
                exported, strategy_dir, model_params, config, "2025-04-19",
            )


# ── Parallelism / new API ─────────────────────────────────────────────────────

class TestParallelTraining:
    """Verify the parallel execution API introduced in the perf refactor."""

    _PARAMS = {
        "feature_columns": ["rsi", "macd_hist"],
        "lookahead": 5, "threshold": 0.02,
        "bags": 3, "leaf_size": 5,
        "buy_threshold": 0.1, "sell_threshold": -0.1,
    }
    _TAX = {"short_term_rate": 0.4, "long_term_rate": 0.2, "long_term_threshold_days": 365}

    def test_run_tournament_returns_log_key(self):
        """run_tournament no longer prints; it returns a _log list instead."""
        from training.tournament import run_tournament
        ohlcv = _make_ohlcv_dict()
        tiny_df = pd.DataFrame(
            {"label": [1] * 10},
            index=pd.bdate_range("2023-01-01", periods=10),
        )
        result = run_tournament(
            "AAPL", tiny_df, ohlcv["AAPL"]["close"], ohlcv["SPY"]["close"],
            self._PARAMS, sharpe_floor=0.8, tax_config=self._TAX,
        )
        assert "_log" in result
        assert isinstance(result["_log"], list)

    def test_xgboost_nthread_parameter(self):
        """XGBoostModel accepts nthread=1 and trains correctly."""
        from training.models import XGBoostModel
        from training.features import build_training_features
        ohlcv = _make_ohlcv_dict()
        df = build_training_features("AAPL", ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD)
        if df is None:
            pytest.skip("no feature frame")
        model = XGBoostModel(
            feature_columns=["rsi", "macd_hist"],
            lookahead=5, threshold=0.02,
            buy_threshold=0.1, sell_threshold=0.1,
            n_estimators=10, max_depth=2,
            nthread=1,
        )
        model.train(df)
        assert len(model.predict_bulk(df.iloc[:5])) == 5

    def test_run_tournament_all_parallel_returns_all_tickers(self):
        """run_tournament_all returns a result entry for every ticker (parallel path)."""
        from training.tournament import run_tournament_all
        from training.features import build_all_training_features
        ohlcv = _make_ohlcv_dict(["AAPL", "GOOG", "SPY"])
        frames = build_all_training_features(
            ["AAPL", "GOOG"], ohlcv, _INDICATOR_SPEC, _LOOKAHEAD, _THRESHOLD
        )
        config = {
            "model_params": self._PARAMS,
            "sharpe_floor": -99.0,
            "tax": self._TAX,
        }
        # max_workers=2: exercises ProcessPoolExecutor path; synthetic data has no
        # post-2024 rows so workers return immediately without training models.
        results = run_tournament_all(["AAPL", "GOOG"], frames, ohlcv, config, max_workers=2)
        assert set(results.keys()) == {"AAPL", "GOOG"}
        for r in results.values():
            assert "sharpe" in r
            assert "passes_floor" in r
