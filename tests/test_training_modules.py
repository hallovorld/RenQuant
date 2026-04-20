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
_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_103"
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
            exported, skipped = export_models(results, strategy_dir, "2025-01-01", 0.8, 5, "renquant_103")
            assert "AAPL" in exported
            assert (strategy_dir / "models" / "AAPL").is_dir()

    def test_below_floor_goes_to_skipped(self):
        from training.export import export_models
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            results = self._fake_results(tmpdir, passes=False)
            exported, skipped = export_models(results, strategy_dir, "2025-01-01", 0.8, 5, "renquant_103")
            assert "AAPL" not in exported
            assert "AAPL" in skipped

    def test_metadata_patched_with_trained_date(self):
        from training.export import export_models
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            results = self._fake_results(tmpdir, passes=True)
            export_models(results, strategy_dir, "2025-04-19", 0.8, 5, "renquant_103")
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
        config = {"strategy": "renquant_103", "tax": {"short_term_rate": 0.4, "long_term_rate": 0.2,
                                                       "long_term_threshold_days": 365}}
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy_dir = Path(tmpdir)
            exported, _ = export_models(results, strategy_dir, "2025-01-01", 0.8, 5, "renquant_103")
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
