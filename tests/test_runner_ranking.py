"""
Tests for live/runner.py ranking of buy candidates.

Candidates are first filtered by their model signal, then ranked using a
cross-model comparable score. Raw model scores remain available for debugging,
but live portfolio selection uses calibrated rank scores when present.

Test strategy:
  - Use a stub broker and stub models so no network I/O or Docker is needed.
  - Drive run_once_multi directly with synthetic data via monkeypatching.
  - Regression tests confirm all existing guards (sector limit, stop-loss, slots)
    still behave correctly under the new ranking order.

Run with:
    cd /path/to/RenQuant
    python -m pytest tests/test_runner_ranking.py -v
"""

import sys
import json
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.models.scoring import ScoreCalibration, fit_probability_calibration
from live.runner import _get_model_score, _get_rank_score, run_once_multi, _ensure_fresh_ohlcv
from live.broker import BaseBroker
from scripts.recalibrate_scores import _compute_blend_weights


# ── Stub helpers ─────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 60, base_price: float = 100.0, vol_spike: bool = False) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with an optional volume spike on the last bar."""
    rng = np.random.default_rng(0)
    closes = base_price * np.cumprod(1 + rng.normal(0.001, 0.005, n))
    # Ensure last bar is an up-close day (required by bullish filter)
    closes[-1] = closes[-2] * 1.005
    normal_vol = 1_000_000
    volumes = np.full(n, normal_vol, dtype=float)
    if vol_spike:
        volumes[-1] = normal_vol * 10  # >> 85th percentile
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "open":   closes * 0.998,
        "high":   closes * 1.01,
        "low":    closes * 0.99,
        "close":  closes,
        "volume": volumes,
    }, index=idx)


class StubModel:
    """Minimal model stub with configurable predict() output and predict_score_bulk() score."""

    def __init__(self, signal: str = "buy", score: float = 0.5,
                 feature_columns=None):
        self._signal = signal
        self._score = score
        self.feature_columns = feature_columns or ["rsi", "macd_hist"]
        self._score_calibration = ScoreCalibration(
            method="identity",
            score_kind="stub_raw",
        )

    def predict(self, row) -> str:
        return self._signal

    def predict_score_bulk(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([self._score] * len(df), index=df.index)


class StubBroker(BaseBroker):
    """In-memory broker that records orders without touching any API."""

    def __init__(self, equity: float = 10_000.0, positions: dict = None,
                 avg_costs: dict = None):
        self._equity = equity
        self._positions = dict(positions or {})
        self._avg_costs = dict(avg_costs or {})
        self.orders = []
        # call counters for verifying batch-fetch behaviour
        self.get_position_calls: list[str] = []
        self.get_all_positions_calls: int = 0

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_position(self, symbol: str) -> float:
        self.get_position_calls.append(symbol)
        return self._positions.get(symbol, 0.0)

    def get_avg_cost(self, symbol: str) -> float:
        return self._avg_costs.get(symbol, 0.0)

    def get_account_value(self) -> float:
        return self._equity

    def get_cash(self) -> float:
        return self._equity

    def get_all_positions(self) -> list[dict]:
        self.get_all_positions_calls += 1
        return [
            {
                "symbol": sym,
                "qty": qty,
                "avg_entry_price": self._avg_costs.get(sym, 0.0),
                "market_value": 0.0,
                "unrealized_pl": 0.0,
            }
            for sym, qty in self._positions.items()
            if qty > 0
        ]

    def place_order(self, symbol: str, action: str, quantity: int) -> dict:
        record = {"symbol": symbol, "action": action, "quantity": quantity,
                  "order_id": f"stub-{len(self.orders)}"}
        self.orders.append(record)
        return record


def _minimal_config(**overrides) -> dict:
    cfg = {
        "model_name": "test-strategy",
        "benchmark": "SPY",
        "data_src": "yfinance",
        "watchlist": ["AAPL", "MSFT", "AMZN"],
        "max_concurrent_positions": 3,
        "volume_zscore_lookback": 20,
        "volume_zscore_threshold": 1.5,
        "volume_filter": {"mode": "percentile", "percentile_threshold": 85},
        "indicator_spec": {},
        "model_params": {"feature_columns": ["rsi", "macd_hist"]},
        "position_sizing": {"max_position_pct": 0.30, "cash_reserve_pct": 0.0},
        "risk": {"stop_loss_pct": 0.0, "portfolio_drawdown_halt_pct": 0.0,
                 "regime_filter": {"enabled": False}},
        "sector_map": {"AAPL": "tech", "MSFT": "tech", "AMZN": "tech"},
        "max_positions_per_sector": 0,
    }
    cfg.update(overrides)
    return cfg


def _patch_runner(monkeypatch, dfs: dict, models: dict, strategy_dir: Path):
    """Redirect fetch_ohlcv and model lookups to in-memory stubs."""
    import live.runner as runner_mod

    def fake_fetch(symbol, provider=None):
        return dfs.get(symbol, pd.DataFrame())

    def fake_build_rel(df_stock, df_spy, feature_cols, indicator_spec):
        """Return a one-row DataFrame with stub feature values."""
        cols = feature_cols or ["rsi", "macd_hist"]
        row_data = {c: 0.5 for c in cols}
        row_data["close"] = float(df_stock["close"].iloc[-1])
        idx = pd.DatetimeIndex([df_stock.index[-1]])
        return pd.DataFrame(row_data, index=idx)

    monkeypatch.setattr(runner_mod, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(runner_mod, "_build_relative_features", fake_build_rel)


# ── _get_model_score ─────────────────────────────────────────────────────────

class TestGetModelScore:
    """Unit tests for the _get_model_score helper."""

    def _row(self, cols=None):
        cols = cols or ["rsi", "macd_hist"]
        return pd.Series({c: 0.5 for c in cols})

    def test_uses_predict_score_bulk_when_available(self):
        model = StubModel(score=0.77)
        score = _get_model_score(model, self._row())
        assert score == pytest.approx(0.77)

    def test_falls_back_to_predict_score(self):
        class ModelWithPredictScore:
            feature_columns = ["rsi"]

            def predict(self, row):
                return "buy"

            def predict_score(self, df):
                return pd.Series([0.42] * len(df), index=df.index)

        score = _get_model_score(ModelWithPredictScore(), self._row(["rsi"]))
        assert score == pytest.approx(0.42)

    def test_falls_back_to_predict_string(self):
        class MinimalModel:
            feature_columns = ["rsi"]

            def predict(self, row):
                return "buy"

        assert _get_model_score(MinimalModel(), self._row(["rsi"])) == pytest.approx(1.0)

    def test_sell_signal_returns_negative(self):
        class MinimalModel:
            feature_columns = ["rsi"]

            def predict(self, row):
                return "sell"

        assert _get_model_score(MinimalModel(), self._row(["rsi"])) == pytest.approx(-1.0)

    def test_hold_signal_returns_zero(self):
        class MinimalModel:
            feature_columns = ["rsi"]

            def predict(self, row):
                return "hold"

        assert _get_model_score(MinimalModel(), self._row(["rsi"])) == pytest.approx(0.0)

    def test_classification_model_real(self):
        """ClassificationModel integrates correctly with _get_model_score."""
        from common.models import create_model

        rng = np.random.default_rng(7)
        cols = ["rsi", "macd_hist", "cci"]
        df = pd.DataFrame(rng.normal(0, 1, (200, len(cols))), columns=cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 200))
        model = create_model(
            "classification", feature_columns=cols,
            lookahead=5, threshold=0.02, leaf_size=10, bags=5,
            buy_threshold=0.1, sell_threshold=-0.1,
        )
        model.train(df)
        row = df.iloc[-1].copy()
        score = _get_model_score(model, row)
        assert isinstance(score, float)

    def test_manual_model_real(self):
        """ManualModel.predict_score_bulk integrates correctly with _get_model_score."""
        from common.models import create_model

        rules = [
            {"col": "rsi", "buy_below": 30.0, "sell_above": 70.0},
            {"col": "macd_hist", "buy_above": 0.0, "sell_below": 0.0},
        ]
        model = create_model(
            "manual", score_rules=rules, buy_threshold=1, sell_threshold=-1
        )
        # Row with both buy conditions met → score = 2.0
        row = pd.Series({"rsi": 25.0, "macd_hist": 0.1})
        score = _get_model_score(model, row)
        assert score == pytest.approx(2.0)

    def test_xgboost_model_real(self):
        """XGBoostModel.predict_score_bulk integrates correctly with _get_model_score."""
        from common.models import create_model

        rng = np.random.default_rng(9)
        cols = ["rsi", "macd_hist", "cci"]
        df = pd.DataFrame(rng.normal(0, 1, (200, len(cols))), columns=cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 200))
        model = create_model(
            "xgboost", feature_columns=cols,
            lookahead=5, threshold=0.02, buy_threshold=0.55, sell_threshold=0.55,
        )
        model.train(df)
        row = df.iloc[-1].copy()
        score = _get_model_score(model, row)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_xgboost_model_save_load_predict(self, tmp_path):
        """Saved XGBoost models must reload with sklearn classifier metadata intact."""
        from common.models import create_model

        rng = np.random.default_rng(19)
        cols = ["rsi", "macd_hist", "cci"]
        df = pd.DataFrame(rng.normal(0, 1, (220, len(cols))), columns=cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 220))

        model = create_model(
            "xgboost", feature_columns=cols,
            lookahead=5, threshold=0.02, buy_threshold=0.55, sell_threshold=0.55,
        )
        model.train(df)
        model.save(tmp_path, "TEST")

        reloaded = create_model("xgboost")
        reloaded.load(tmp_path, "TEST")

        row = df.iloc[-1].copy()
        signal = reloaded.predict(row)
        score = _get_model_score(reloaded, row)

        assert signal in {"buy", "hold", "sell"}
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_xgboost_signal_threshold_uses_net_score_directly(self, monkeypatch):
        """A buy_threshold of 0.55 should require a >0.55 net score, not merely >0.05."""
        from common.models.xgboost_model import XGBoostModel

        model = XGBoostModel(feature_columns=["rsi"], buy_threshold=0.55, sell_threshold=0.55)
        model._buy_model = object()
        model._sell_model = object()

        monkeypatch.setattr(model, "_score", lambda X: np.array([0.10] * len(X)))

        row = pd.Series({"rsi": 0.5})
        signal = model.predict(row)
        bulk_signal = model.predict_bulk(pd.DataFrame([row]))

        assert signal == "hold"
        assert bulk_signal.iloc[0] == "hold"

    def test_xgboost_sell_threshold_uses_net_score_directly(self, monkeypatch):
        from common.models.xgboost_model import XGBoostModel

        model = XGBoostModel(feature_columns=["rsi"], buy_threshold=0.55, sell_threshold=0.55)
        model._buy_model = object()
        model._sell_model = object()

        monkeypatch.setattr(model, "_score", lambda X: np.array([-0.10] * len(X)))

        row = pd.Series({"rsi": 0.5})
        signal = model.predict(row)
        bulk_signal = model.predict_bulk(pd.DataFrame([row]))

        assert signal == "hold"
        assert bulk_signal.iloc[0] == "hold"

    def test_rank_score_uses_calibration_when_present(self):
        model = StubModel(score=2.0)
        model._score_calibration = ScoreCalibration(
            method="isotonic",
            score_kind="vote_count",
            x_thresholds=[0.0, 1.0, 2.0],
            y_thresholds=[0.10, 0.40, 0.85],
        )

        score = _get_rank_score(model, self._row())
        assert score == pytest.approx(0.85)


class TestScoreCalibration:
    def test_fit_probability_calibration_is_monotonic(self):
        raw = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        future = pd.Series([-0.05, -0.03, -0.01, 0.01, 0.05, 0.08])
        calibration = fit_probability_calibration(
            pd.concat([raw] * 30, ignore_index=True),
            pd.concat([future] * 30, ignore_index=True),
            lookahead=5,
            threshold=0.02,
            score_kind="bag_learner_raw",
        )

        # 180 samples (30×6) → Platt range; both Platt and isotonic are monotone
        assert calibration.method in ("isotonic", "platt")
        assert calibration.calibrate(-1.0) <= calibration.calibrate(2.0)


class TestBlendWeights:
    def test_logistic_blend_weights_favor_stronger_signal(self):
        rank_scores = np.linspace(0.05, 0.95, 200)
        rs_scores = np.linspace(0.05, 0.95, 200)
        outcomes = (rank_scores > 0.55).astype(float)

        w_rank, w_rs = _compute_blend_weights([
            {
                "rank_scores": rank_scores,
                "rs_scores": rs_scores[::-1],
                "outcomes": outcomes,
            }
        ])

        assert w_rank > w_rs

    def test_logistic_blend_weights_fallback_when_labels_too_thin(self):
        rank_scores = np.linspace(0.1, 0.9, 50)
        rs_scores = np.linspace(0.2, 0.8, 50)
        outcomes = np.ones(50)

        w_rank, w_rs = _compute_blend_weights([
            {
                "rank_scores": rank_scores,
                "rs_scores": rs_scores,
                "outcomes": outcomes,
            }
        ])

        assert (w_rank, w_rs) == (0.5, 0.5)


# ── Ranking behaviour ─────────────────────────────────────────────────────────

class TestModelScoreRanking:
    """Verify that higher-conviction candidates get priority."""

    def test_higher_model_score_wins_over_higher_volume_score(
        self, monkeypatch, tmp_path
    ):
        """
        When multiple buy signals are present, the higher-conviction candidate wins.
        """
        aapl_df = _make_ohlcv(vol_spike=True)
        aapl_df.loc[aapl_df.index[-1], "volume"] = 50_000_000

        msft_df = _make_ohlcv(vol_spike=True)
        msft_df.loc[msft_df.index[-1], "volume"] = 5_000_000

        spy_df = _make_ohlcv()
        dfs = {"AAPL": aapl_df, "MSFT": msft_df, "AMZN": _make_ohlcv(), "SPY": spy_df}

        models = {
            "AAPL": StubModel(signal="buy", score=0.2),
            "MSFT": StubModel(signal="buy", score=0.9),  # higher conviction
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config(max_concurrent_positions=1)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        assert len(broker.orders) == 1
        assert broker.orders[0]["symbol"] == "MSFT"

    def test_mixed_raw_scales_use_calibrated_rank_score(self, monkeypatch, tmp_path):
        """Mixed model types should rank by calibrated score, not raw native scale."""
        import live.runner as runner_mod

        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY": _make_ohlcv(),
        }

        manual_like = StubModel(signal="buy", score=2.0)
        manual_like._score_calibration = ScoreCalibration(
            method="isotonic",
            score_kind="vote_count",
            x_thresholds=[0.0, 1.0, 2.0],
            y_thresholds=[0.20, 0.45, 0.55],
        )
        xgb_like = StubModel(signal="buy", score=0.25)
        xgb_like._score_calibration = ScoreCalibration(
            method="isotonic",
            score_kind="p_buy_minus_sell",
            x_thresholds=[-0.2, 0.0, 0.25],
            y_thresholds=[0.10, 0.35, 0.80],
        )
        models = {
            "AAPL": manual_like,
            "MSFT": xgb_like,
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            max_concurrent_positions=1,
            tiered_thresholds=[{"min_model_score": 0.0}],
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)
        monkeypatch.setattr(
            runner_mod,
            "_ensure_model_score_calibrations",
            lambda config, models, dfs, df_spy: None,
        )

        run_once_multi(config, models, broker, tmp_path)

        assert len(broker.orders) == 1
        assert broker.orders[0]["symbol"] == "MSFT", (
            "MSFT should win because its calibrated rank score is stronger even though its raw score is smaller"
        )

    def test_both_bought_when_two_slots_open(self, monkeypatch, tmp_path):
        """With two open slots, both high-score candidates should be bought (model-score order)."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.6),
            "MSFT": StubModel(signal="buy", score=0.8),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config(max_concurrent_positions=2)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert set(bought) == {"AAPL", "MSFT"}

    def test_single_candidate_still_bought(self, monkeypatch, tmp_path):
        """Regression: when only one symbol produces a buy signal, it is still bought."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=False),
            "MSFT": _make_ohlcv(vol_spike=False),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="hold", score=0.9),
            "MSFT": StubModel(signal="hold", score=0.8),
            "AMZN": StubModel(signal="buy", score=0.5),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config()
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert bought == ["AMZN"]

    def test_hold_signal_excluded(self, monkeypatch, tmp_path):
        """A volume candidate whose model returns hold is not bought."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="hold", score=0.9),  # high score but hold signal
            "MSFT": StubModel(signal="buy",  score=0.3),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config(max_concurrent_positions=1)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert bought == ["MSFT"], "AAPL should be excluded despite high score because signal=hold"

    def test_min_model_score_filter(self, monkeypatch, tmp_path):
        """Candidates below min_model_score are excluded even if signal=buy."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.05),  # below threshold
            "MSFT": StubModel(signal="buy", score=0.20),  # above threshold
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config(min_model_score=0.10)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert "AAPL" not in bought
        assert "MSFT" in bought

    def test_model_score_logged(self, monkeypatch, tmp_path, caplog):
        """Candidate logs expose model type, raw score, and calibrated score."""
        import logging

        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=False),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.65),
            "MSFT": StubModel(signal="hold", score=0.0),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config()
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        with caplog.at_level(logging.INFO, logger="live.runner"):
            run_once_multi(config, models, broker, tmp_path)

        assert any(
            "model=" in r.message and "raw=" in r.message and "calibrated=" in r.message
            for r in caplog.records
        )

    def test_trade_log_contains_model_score(self, monkeypatch, tmp_path):
        """The JSON trade log record includes raw and calibrated rank scores for each buy."""
        import live.runner as runner_mod

        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=False),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.73),
            "MSFT": StubModel(signal="hold", score=0.0),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        logged = []
        monkeypatch.setattr(runner_mod, "_log_trade",
                            lambda sd, sn, record: logged.append(record))

        broker = StubBroker(equity=10_000)
        config = _minimal_config()
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        buy_entries = [r for r in logged if r.get("signal") == "buy"]
        assert buy_entries, "No buy record logged"
        assert "model_type" in buy_entries[0]
        assert "raw_model_score" in buy_entries[0]
        assert "rank_model_score" in buy_entries[0]
        assert buy_entries[0]["model_type"] == "Stub"
        assert buy_entries[0]["raw_model_score"] == pytest.approx(0.73, abs=1e-4)
        assert buy_entries[0]["rank_model_score"] == pytest.approx(0.73, abs=1e-4)


# ── Regression: existing guards still work ────────────────────────────────────

class TestRegressionGuards:
    """Confirm that guards present before the ranking change still fire correctly."""

    def test_sector_guard_still_blocks_excess(self, monkeypatch, tmp_path):
        """max_positions_per_sector=1 blocks the second tech stock even after re-ranking."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.9),
            "MSFT": StubModel(signal="buy", score=0.8),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        # All three are in "tech" sector; max 1 per sector
        config = _minimal_config(max_positions_per_sector=1, max_concurrent_positions=3)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert len(bought) == 1, "Sector guard should block the second tech stock"

    def test_max_concurrent_positions_respected(self, monkeypatch, tmp_path):
        """With max_concurrent_positions=1 only one order is placed even with two candidates."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.7),
            "MSFT": StubModel(signal="buy", score=0.6),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config(max_concurrent_positions=1, max_positions_per_sector=0)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert len(bought) == 1

    def test_existing_position_skips_buy(self, monkeypatch, tmp_path):
        """A symbol already held is not bought again."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=False),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.9),
            "MSFT": StubModel(signal="buy", score=0.5),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        # AAPL already held
        broker = StubBroker(equity=10_000, positions={"AAPL": 5})
        config = _minimal_config()
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert "AAPL" not in bought, "Should not buy a symbol already in the portfolio"

    def test_stop_loss_triggers_sell_before_buy_scan(self, monkeypatch, tmp_path):
        """A held position that breaches stop-loss is sold before any new buys."""
        # AAPL is held at avg cost 100, current price 80 → 20% loss → triggers 8% stop
        aapl_df = _make_ohlcv(base_price=80.0, vol_spike=False)
        msft_df = _make_ohlcv(vol_spike=True)
        spy_df  = _make_ohlcv()

        dfs = {"AAPL": aapl_df, "MSFT": msft_df, "AMZN": _make_ohlcv(), "SPY": spy_df}
        models = {
            "AAPL": StubModel(signal="hold", score=0.0),
            "MSFT": StubModel(signal="buy",  score=0.8),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        broker = StubBroker(equity=10_000, positions={"AAPL": 10})
        broker._avg_costs["AAPL"] = 100.0  # avg cost above current price

        config = _minimal_config()
        config["risk"]["stop_loss_pct"] = 0.08  # 8% stop

        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        actions = [(o["symbol"], o["action"]) for o in broker.orders]
        assert ("AAPL", "SELL") in actions, "Stop-loss sell should fire"
        assert ("MSFT", "BUY") in actions, "New buy should still happen after stop-loss sell"

    def test_drawdown_halt_blocks_new_buys(self, monkeypatch, tmp_path):
        """Portfolio drawdown above halt threshold suppresses all new buys."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.9),
            "MSFT": StubModel(signal="buy", score=0.8),
            "AMZN": StubModel(signal="hold", score=0.0),
        }

        # Write a high-water mark state file to simulate a large drawdown
        strategy_dir = tmp_path
        state = {"high_water_mark": 20_000.0}  # broker equity is 10k → 50% drawdown
        (strategy_dir / "live_state.json").write_text(json.dumps(state))

        broker = StubBroker(equity=10_000)
        config = _minimal_config()
        config["risk"]["portfolio_drawdown_halt_pct"] = 0.15

        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, strategy_dir)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert len(bought) == 0, "Drawdown halt should block all new buys"

    def test_no_candidates_no_orders(self, monkeypatch, tmp_path):
        """When no symbols produce a buy signal, no orders are placed."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=False),
            "MSFT": _make_ohlcv(vol_spike=False),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="hold", score=0.9),
            "MSFT": StubModel(signal="hold", score=0.8),
            "AMZN": StubModel(signal="hold", score=0.7),
        }

        broker = StubBroker(equity=10_000)
        config = _minimal_config()
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        assert len(broker.orders) == 0


# ── predict_score_bulk on all model types ─────────────────────────────────────

class TestPredictScoreBulkAllModels:
    """Confirm predict_score_bulk works on every model type (CLAUDE.md guarantee)."""

    @pytest.fixture
    def feature_cols(self):
        return ["rsi", "macd_hist", "cci"]

    @pytest.fixture
    def train_df(self, feature_cols):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(rng.normal(0, 1, (n, len(feature_cols))), columns=feature_cols)
        df["close"] = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        return df

    def test_classification_predict_score_bulk(self, train_df, feature_cols):
        from common.models import create_model

        model = create_model(
            "classification", feature_columns=feature_cols,
            lookahead=5, threshold=0.02, leaf_size=10, bags=5,
            buy_threshold=0.1, sell_threshold=-0.1,
        )
        model.train(train_df)
        scores = model.predict_score_bulk(train_df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df)

    def test_qlearning_predict_score_bulk(self, train_df, feature_cols):
        from common.models import create_model

        train_df_q = train_df.copy()
        train_df_q["position_flag"] = 0
        model = create_model("qlearning", feature_columns=feature_cols, n_bins=5, n_epochs=10)
        model.train(train_df_q)
        scores = model.predict_score_bulk(train_df_q)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df_q)

    def test_xgboost_predict_score_bulk(self, train_df, feature_cols):
        from common.models import create_model

        model = create_model(
            "xgboost", feature_columns=feature_cols,
            lookahead=5, threshold=0.02, buy_threshold=0.55, sell_threshold=0.55,
        )
        model.train(train_df)
        scores = model.predict_score_bulk(train_df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df)
        assert scores.between(-1.0, 1.0).all()

    def test_manual_predict_score_bulk(self, train_df):
        from common.models import create_model

        rules = [
            {"col": "rsi",      "buy_below": 0.0, "sell_above": 1.0},
            {"col": "macd_hist","buy_above": 0.0, "sell_below": 0.0},
        ]
        model = create_model("manual", score_rules=rules, buy_threshold=1, sell_threshold=-1)
        scores = model.predict_score_bulk(train_df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(train_df)
        assert scores.dtype == float or np.issubdtype(scores.dtype, np.floating)


# ── Tiered thresholds ─────────────────────────────────────────────────────────

class TestTieredThresholds:
    """Verify that per-slot escalating thresholds restrict 2nd and 3rd orders correctly."""

    def _three_spike_dfs(self):
        """Return dfs where AAPL, MSFT, and AMZN all have volume spikes of different sizes."""
        dfs = {
            # vol_score will be ~95 (very high spike)
            "AAPL": _make_ohlcv(vol_spike=False),
            # vol_score will be ~90 (high spike)
            "MSFT": _make_ohlcv(vol_spike=False),
            # vol_score will be ~87 (moderate spike)
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        # Override volumes to produce precise percentile ranks
        # With 20-bar lookback of vol=1_000_000, these levels guarantee:
        #   AAPL: vol 50x normal → pct ≈ 100 (beats all 20 historical bars)
        #   MSFT: vol 10x normal → pct ≈ 100 as well but we make lookback diverse
        # To get precise percentiles, build varying historical volumes
        for sym, multiplier in [("AAPL", 50), ("MSFT", 10), ("AMZN", 5)]:
            df = dfs[sym].copy()
            # Historical bars: 1M, 2M, ..., 20M (ascending)
            n = len(df)
            volumes = np.linspace(1_000_000, 20_000_000, n)
            volumes[-1] = multiplier * 20_000_000  # today's spike
            # Ensure up-close day
            df["close"].iloc[-1] = df["close"].iloc[-2] * 1.005
            df["volume"] = volumes
            dfs[sym] = df
        return dfs

    def _models_for_tiered(self, aapl_score=0.7, msft_score=0.5, amzn_score=0.3):
        return {
            "AAPL": StubModel(signal="buy", score=aapl_score),
            "MSFT": StubModel(signal="buy", score=msft_score),
            "AMZN": StubModel(signal="buy", score=amzn_score),
        }

    def test_slot1_easy_slot2_blocked_by_model_score(self, monkeypatch, tmp_path):
        """
        Tier 2 requires model_score >= 0.6. MSFT has score=0.5 → blocked for slot 2.
        Only AAPL (score=0.7) should be bought.
        """
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = self._models_for_tiered(aapl_score=0.7, msft_score=0.5)
        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            max_concurrent_positions=2,
            max_positions_per_sector=0,
            tiered_thresholds=[
                {"min_volume_pct": 85, "min_model_score": 0.0},   # slot 1: easy
                {"min_volume_pct": 85, "min_model_score": 0.6},   # slot 2: higher score bar
            ],
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert bought == ["AAPL"], "MSFT should be blocked by slot-2 model_score threshold"

    def test_slot2_filled_when_score_meets_tier(self, monkeypatch, tmp_path):
        """When both candidates meet their respective tier thresholds, both are bought."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = self._models_for_tiered(aapl_score=0.8, msft_score=0.65)
        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            max_concurrent_positions=2,
            max_positions_per_sector=0,
            tiered_thresholds=[
                {"min_volume_pct": 85, "min_model_score": 0.0},
                {"min_volume_pct": 85, "min_model_score": 0.6},  # MSFT(0.65) passes
            ],
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = {o["symbol"] for o in broker.orders if o["action"] == "BUY"}
        assert bought == {"AAPL", "MSFT"}

    def test_slot3_has_strictest_bar(self, monkeypatch, tmp_path):
        """
        Three candidates, three slots. Slot 3 requires model_score >= 0.75.
        AMZN has score=0.6 so it should be blocked for slot 3.
        """
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=True),
            "SPY":  _make_ohlcv(),
        }
        models = self._models_for_tiered(aapl_score=0.9, msft_score=0.8, amzn_score=0.6)
        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            max_concurrent_positions=3,
            max_positions_per_sector=0,
            tiered_thresholds=[
                {"min_volume_pct": 85, "min_model_score": 0.0},
                {"min_volume_pct": 85, "min_model_score": 0.5},
                {"min_volume_pct": 85, "min_model_score": 0.75},  # AMZN(0.6) blocked
            ],
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert "AMZN" not in bought, "AMZN should fail the slot-3 model_score threshold"
        assert set(bought) == {"AAPL", "MSFT"}

    def test_slot_number_in_log(self, monkeypatch, tmp_path, caplog):
        """BUY log line includes slot number so operators can see which tier fired."""
        import logging

        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = self._models_for_tiered(aapl_score=0.9, msft_score=0.7)
        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            max_concurrent_positions=2,
            max_positions_per_sector=0,
            tiered_thresholds=[
                {"min_volume_pct": 85, "min_model_score": 0.0},
                {"min_volume_pct": 85, "min_model_score": 0.5},
            ],
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        with caplog.at_level(logging.INFO, logger="live.runner"):
            run_once_multi(config, models, broker, tmp_path)

        buy_logs = [r.message for r in caplog.records if "BUY" in r.message and "slot" in r.message]
        assert len(buy_logs) >= 1, "BUY log should include slot number"
        assert "slot=1" in buy_logs[0].lower() or "slot=2" in " ".join(buy_logs).lower()

    def test_trade_log_contains_slot_number(self, monkeypatch, tmp_path):
        """Each buy record includes which slot it filled."""
        import live.runner as runner_mod

        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=True),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = self._models_for_tiered(aapl_score=0.9, msft_score=0.7)

        logged = []
        monkeypatch.setattr(runner_mod, "_log_trade",
                            lambda sd, sn, record: logged.append(record))

        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            max_concurrent_positions=2,
            max_positions_per_sector=0,
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        buy_entries = [r for r in logged if r.get("signal") == "buy"]
        assert buy_entries, "No buy records logged"
        assert all("slot" in e for e in buy_entries), "All buy records must include 'slot'"

    def test_no_tiers_config_still_works(self, monkeypatch, tmp_path):
        """Without tiered thresholds, ranking still falls back to the base minimum score."""
        dfs = {
            "AAPL": _make_ohlcv(vol_spike=True),
            "MSFT": _make_ohlcv(vol_spike=False),
            "AMZN": _make_ohlcv(vol_spike=False),
            "SPY":  _make_ohlcv(),
        }
        models = {
            "AAPL": StubModel(signal="buy", score=0.6),
            "MSFT": StubModel(signal="buy", score=0.9),
            "AMZN": StubModel(signal="hold", score=0.0),
        }
        broker = StubBroker(equity=10_000)
        config = _minimal_config(max_concurrent_positions=1)  # no tiered_thresholds key
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        run_once_multi(config, models, broker, tmp_path)

        bought = [o["symbol"] for o in broker.orders if o["action"] == "BUY"]
        assert bought == ["MSFT"]


# ── Batch position fetch (regression for serial API-call timeout) ─────────────

class TestBatchPositionFetch:
    """Verify run_once_multi uses get_all_positions() instead of per-symbol
    get_position() calls, preventing the N×API-round-trip timeout that hit
    production on 2026-04-16."""

    def _run(self, monkeypatch, tmp_path, positions: dict, avg_costs: dict = None):
        watchlist = ["AAPL", "MSFT", "AMZN"]
        dfs = {sym: _make_ohlcv() for sym in watchlist + ["SPY"]}
        models = {sym: StubModel(signal="hold", score=0.0) for sym in watchlist}
        broker = StubBroker(equity=10_000, positions=positions,
                            avg_costs=avg_costs or {})
        config = _minimal_config(watchlist=watchlist)
        _patch_runner(monkeypatch, dfs, models, tmp_path)
        run_once_multi(config, models, broker, tmp_path)
        return broker

    def test_get_all_positions_called_once(self, monkeypatch, tmp_path):
        """get_all_positions() must be called exactly once per run."""
        broker = self._run(monkeypatch, tmp_path, positions={})
        assert broker.get_all_positions_calls == 1

    def test_get_position_never_called_in_multi_path(self, monkeypatch, tmp_path):
        """get_position() must not be called in the multi-stock path — all
        lookups should use the positions cache built from get_all_positions()."""
        broker = self._run(monkeypatch, tmp_path, positions={})
        assert broker.get_position_calls == [], (
            f"get_position() was called {len(broker.get_position_calls)} times "
            f"for: {broker.get_position_calls}"
        )

    def test_get_position_never_called_with_held_positions(self, monkeypatch, tmp_path):
        """Even when symbols are held (sell phase executes), get_position() is
        still never called — qty must come from the positions cache."""
        broker = self._run(monkeypatch, tmp_path,
                           positions={"AAPL": 10.0},
                           avg_costs={"AAPL": 95.0})
        assert broker.get_position_calls == [], (
            f"get_position() called during sell phase: {broker.get_position_calls}"
        )

    def test_held_list_built_from_cache(self, monkeypatch, tmp_path):
        """Symbols in the positions cache appear in the sell phase (model signal
        evaluated), even with a hold signal."""
        watchlist = ["AAPL", "MSFT", "AMZN"]
        dfs = {sym: _make_ohlcv() for sym in watchlist + ["SPY"]}
        # AAPL is held; model says "hold" so no sell order expected
        models = {sym: StubModel(signal="hold", score=0.0) for sym in watchlist}
        broker = StubBroker(equity=10_000, positions={"AAPL": 5.0},
                            avg_costs={"AAPL": 100.0})
        config = _minimal_config(watchlist=watchlist)
        _patch_runner(monkeypatch, dfs, models, tmp_path)
        run_once_multi(config, models, broker, tmp_path)

        sell_orders = [o for o in broker.orders if o["action"] == "SELL"]
        assert sell_orders == [], "No sell expected — model said hold"
        # Exactly one batch fetch, no per-symbol calls
        assert broker.get_all_positions_calls == 1
        assert broker.get_position_calls == []

    def test_stop_loss_uses_cache_qty(self, monkeypatch, tmp_path):
        """When a stop-loss fires, the SELL quantity comes from the positions
        cache (not a live get_position() call)."""
        watchlist = ["AAPL", "MSFT", "AMZN"]
        dfs = {sym: _make_ohlcv(base_price=50.0) for sym in watchlist + ["SPY"]}
        models = {sym: StubModel(signal="hold", score=0.0) for sym in watchlist}
        # AAPL bought at 100, now at ~50 → 50% loss → triggers 8% stop
        broker = StubBroker(equity=10_000, positions={"AAPL": 7.0},
                            avg_costs={"AAPL": 100.0})
        config = _minimal_config(
            watchlist=watchlist,
            risk={"stop_loss_pct": 0.08, "portfolio_drawdown_halt_pct": 0.0,
                  "regime_filter": {"enabled": False}},
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)
        run_once_multi(config, models, broker, tmp_path)

        sell_orders = [o for o in broker.orders if o["action"] == "SELL"]
        assert len(sell_orders) == 1
        assert sell_orders[0]["symbol"] == "AAPL"
        assert sell_orders[0]["quantity"] == pytest.approx(7.0)
        # Still no per-symbol get_position() calls
        assert broker.get_position_calls == []


class TestLiveRegimeParity:
    def test_choppy_regime_defaults_to_half_confidence(self, monkeypatch):
        import live.runner as runner_mod

        df_spy = _make_ohlcv(n=80)
        regime_cfg = {
            "hurst_window": 63,
            "cusum_lookback": 20,
            "cusum_threshold": 3.0,
            "cusum_drift": 0.5,
            "vol_realized_window": 20,
        }

        monkeypatch.setattr(runner_mod, "_compute_hurst_live", lambda returns, window: 0.40)
        monkeypatch.setattr(
            runner_mod,
            "_compute_cusum_live",
            lambda returns, lookback, threshold, drift: False,
        )
        monkeypatch.setattr(
            runner_mod,
            "_gmm_predict_live",
            lambda *args, **kwargs: {
                "BULL_CALM": 0.55,
                "BULL_VOLATILE": 0.30,
                "BEAR": 0.15,
            },
        )

        regime, confidence, triggered = runner_mod._detect_regime_live(
            df_spy,
            {"artifact": True},
            regime_cfg,
        )

        assert regime == "CHOPPY"
        assert confidence == pytest.approx(0.5)
        assert not triggered

    def test_detect_regime_uses_real_spy_adx(self, monkeypatch):
        import live.runner as runner_mod

        idx = pd.bdate_range("2024-01-02", periods=80)
        close = np.linspace(100.0, 180.0, len(idx))
        df_spy = pd.DataFrame(
            {
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(idx), 1_000_000.0),
            },
            index=idx,
        )
        regime_cfg = {
            "hurst_window": 63,
            "cusum_lookback": 20,
            "cusum_threshold": 3.0,
            "cusum_drift": 0.5,
            "vol_realized_window": 20,
        }
        captured = {}

        monkeypatch.setattr(runner_mod, "_compute_hurst_live", lambda returns, window: 0.50)
        monkeypatch.setattr(
            runner_mod,
            "_compute_cusum_live",
            lambda returns, lookback, threshold, drift: False,
        )

        def fake_gmm_predict(gmm_artifact, r10d, vol20, spy_adx, r_autocorr):
            captured["spy_adx"] = spy_adx
            return {"BULL_CALM": 0.55, "BULL_VOLATILE": 0.35, "BEAR": 0.10}

        monkeypatch.setattr(runner_mod, "_gmm_predict_live", fake_gmm_predict)

        runner_mod._detect_regime_live(df_spy, {"artifact": True}, regime_cfg)

        assert "spy_adx" in captured
        assert captured["spy_adx"] != pytest.approx(25.0)

    def test_confidence_scaled_sizing_reduces_live_buy_size(self, monkeypatch, tmp_path):
        import live.runner as runner_mod

        aapl_df = _make_ohlcv(n=80, base_price=100.0, vol_spike=True)
        aapl_df.loc[aapl_df.index[-1], ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
        dfs = {"AAPL": aapl_df, "SPY": _make_ohlcv(n=80)}
        models = {"AAPL": StubModel(signal="buy", score=0.9)}

        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            watchlist=["AAPL"],
            sector_map={"AAPL": "tech"},
            max_concurrent_positions=1,
            tiered_thresholds=[{"min_model_score": 0.0}],
            regime={"transition_uncertainty_bars": 3},
            regime_params={
                "BULL_CALM": {
                    "max_position_pct": 0.15,
                    "cash_reserve_pct": 0.0,
                    "stop_loss_pct": 0.15,
                    "max_single_day_loss_pct": 0.10,
                    "max_hold_days": 500,
                    "drawdown_halt_pct": 0.35,
                    "min_model_score": 0.10,
                    "spy_velocity_halt_pct": 0.03,
                    "spy_velocity_lookback_days": 3,
                },
                "BULL_VOLATILE": {
                    "max_position_pct": 0.20,
                    "cash_reserve_pct": 0.20,
                    "stop_loss_pct": 0.05,
                    "max_single_day_loss_pct": 0.0,
                    "max_hold_days": 500,
                    "drawdown_halt_pct": 0.10,
                    "min_model_score": 0.0,
                    "spy_velocity_halt_pct": 0.03,
                    "spy_velocity_lookback_days": 3,
                },
            },
        )

        _patch_runner(monkeypatch, dfs, models, tmp_path)
        monkeypatch.setattr(
            runner_mod,
            "_detect_regime_live",
            lambda df_spy, gmm_artifact, regime_cfg: ("BULL_VOLATILE", 0.5, False),
        )

        run_once_multi(config, models, broker, tmp_path)

        assert len(broker.orders) == 1
        assert broker.orders[0]["symbol"] == "AAPL"
        assert broker.orders[0]["quantity"] == 10

    def test_bear_defensive_buy_bypasses_macro_gates_and_cash_reserve(self, monkeypatch, tmp_path):
        import live.runner as runner_mod

        gld_df = _make_ohlcv(n=80, base_price=100.0, vol_spike=True)
        gld_df.loc[gld_df.index[-1], ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]

        idx = pd.bdate_range("2024-01-02", periods=80)
        close = np.linspace(200.0, 120.0, len(idx))
        spy_df = pd.DataFrame(
            {
                "open": close * 1.005,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(idx), 1_000_000.0),
            },
            index=idx,
        )

        dfs = {"GLD": gld_df, "SPY": spy_df}
        models = {"GLD": StubModel(signal="buy", score=0.9)}

        broker = StubBroker(equity=10_000)
        config = _minimal_config(
            watchlist=["GLD"],
            sector_map={"GLD": "commodity"},
            defensive_tickers=["GLD"],
            max_concurrent_positions=1,
            tiered_thresholds=[{"min_model_score": 0.0}],
            regime={"transition_uncertainty_bars": 3},
            regime_params={
                "BULL_CALM": {
                    "max_position_pct": 0.15,
                    "cash_reserve_pct": 0.0,
                    "stop_loss_pct": 0.15,
                    "max_single_day_loss_pct": 0.10,
                    "max_hold_days": 500,
                    "drawdown_halt_pct": 0.35,
                    "min_model_score": 0.10,
                    "spy_velocity_halt_pct": 0.03,
                    "spy_velocity_lookback_days": 3,
                },
                "BEAR": {
                    "max_position_pct": 0.0,
                    "cash_reserve_pct": 1.0,
                    "stop_loss_pct": 0.05,
                    "max_single_day_loss_pct": 0.0,
                    "max_hold_days": 500,
                    "drawdown_halt_pct": 0.05,
                    "min_model_score": 0.0,
                    "spy_velocity_halt_pct": 0.03,
                    "spy_velocity_lookback_days": 3,
                },
            },
        )

        _patch_runner(monkeypatch, dfs, models, tmp_path)
        monkeypatch.setattr(
            runner_mod,
            "_detect_regime_live",
            lambda df_spy, gmm_artifact, regime_cfg: ("BEAR", 0.9, False),
        )

        run_once_multi(config, models, broker, tmp_path)

        assert len(broker.orders) == 1
        assert broker.orders[0]["symbol"] == "GLD"
        assert broker.orders[0]["quantity"] == 15


# ── TestOpenOrdersGuard ───────────────────────────────────────────────────────

class TestOpenOrdersGuard:
    """Tests for get_open_orders() in AlpacaBroker and the runner's pending-order guard."""

    # ── AlpacaBroker unit tests ───────────────────────────────────────────────

    def test_alpaca_get_open_orders_returns_symbol_set(self):
        """get_open_orders() returns the set of symbols from OPEN-status orders."""
        from live.alpaca_broker import AlpacaBroker
        from unittest.mock import MagicMock

        broker = AlpacaBroker.__new__(AlpacaBroker)
        broker._paper = True

        mock_order_amd = MagicMock()
        mock_order_amd.symbol = "AMD"
        mock_order_nvda = MagicMock()
        mock_order_nvda.symbol = "NVDA"

        mock_client = MagicMock()
        mock_client.get_orders.return_value = [mock_order_amd, mock_order_nvda]
        broker._trading_client = mock_client

        result = broker.get_open_orders()

        assert result == {"AMD", "NVDA"}
        # Verify it called Alpaca with OPEN status
        args, kwargs = mock_client.get_orders.call_args
        filter_arg = kwargs.get("filter") or (args[0] if args else None)
        from alpaca.trading.enums import QueryOrderStatus
        assert filter_arg is not None
        assert str(filter_arg.status) in (str(QueryOrderStatus.OPEN), "open")

    def test_alpaca_get_open_orders_empty(self):
        """get_open_orders() returns empty set when no open orders exist."""
        from live.alpaca_broker import AlpacaBroker
        from unittest.mock import MagicMock

        broker = AlpacaBroker.__new__(AlpacaBroker)
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        broker._trading_client = mock_client

        assert broker.get_open_orders() == set()

    def test_base_broker_get_open_orders_returns_empty_set(self):
        """BaseBroker.get_open_orders() default returns empty set (safe no-op)."""
        from live.broker import BaseBroker

        class ConcreteStub(BaseBroker):
            def connect(self): pass
            def disconnect(self): pass
            def get_position(self, s): return 0.0
            def get_account_value(self): return 0.0
            def place_order(self, s, a, q): return {}

        broker = ConcreteStub()
        assert broker.get_open_orders() == set()

    # ── runner integration tests ──────────────────────────────────────────────

    def test_runner_skips_symbol_with_pending_order(self, monkeypatch, tmp_path):
        """A candidate whose symbol already has a pending Alpaca order is skipped."""
        import live.runner as runner_mod

        dfs = {s: _make_ohlcv(60, vol_spike=True) for s in ["AAPL", "MSFT", "SPY"]}
        models = {
            "AAPL": StubModel(signal="buy", score=0.9),
            "MSFT": StubModel(signal="buy", score=0.8),
        }

        class PendingBroker(StubBroker):
            def get_open_orders(self):
                return {"AAPL"}  # AAPL already queued

        broker = PendingBroker(equity=10_000)
        config = _minimal_config(
            watchlist=["AAPL", "MSFT"],
            max_concurrent_positions=2,
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)
        run_once_multi(config, models, broker, tmp_path)

        bought = {o["symbol"] for o in broker.orders if o["action"] == "BUY"}
        assert "AAPL" not in bought, "AAPL had a pending order — should have been skipped"
        assert "MSFT" in bought, "MSFT had no pending order — should have been bought"

    def test_runner_buys_all_when_no_pending_orders(self, monkeypatch, tmp_path):
        """When get_open_orders() returns empty set all qualifying candidates are bought."""
        import live.runner as runner_mod

        dfs = {s: _make_ohlcv(60, vol_spike=True) for s in ["AAPL", "MSFT", "SPY"]}
        models = {
            "AAPL": StubModel(signal="buy", score=0.9),
            "MSFT": StubModel(signal="buy", score=0.8),
        }
        broker = StubBroker(equity=20_000)
        # StubBroker.get_open_orders() inherits BaseBroker default → empty set
        config = _minimal_config(
            watchlist=["AAPL", "MSFT"],
            max_concurrent_positions=2,
        )
        _patch_runner(monkeypatch, dfs, models, tmp_path)
        run_once_multi(config, models, broker, tmp_path)

        bought = {o["symbol"] for o in broker.orders if o["action"] == "BUY"}
        assert "AAPL" in bought
        assert "MSFT" in bought

    def test_runner_continues_on_get_open_orders_exception(self, monkeypatch, tmp_path):
        """If get_open_orders() raises an exception the runner falls back to empty set."""
        import live.runner as runner_mod

        dfs = {s: _make_ohlcv(60, vol_spike=True) for s in ["AAPL", "SPY"]}
        models = {"AAPL": StubModel(signal="buy", score=0.9)}

        class ExplodingBroker(StubBroker):
            def get_open_orders(self):
                raise RuntimeError("Alpaca API unavailable")

        broker = ExplodingBroker(equity=10_000)
        config = _minimal_config(watchlist=["AAPL"], max_concurrent_positions=2)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        # Should not raise; should still buy AAPL
        run_once_multi(config, models, broker, tmp_path)
        bought = [o for o in broker.orders if o["action"] == "BUY"]
        assert len(bought) == 1
        assert bought[0]["symbol"] == "AAPL"

    def test_pending_order_check_logs_skipped_symbol(self, monkeypatch, tmp_path, caplog):
        """Runner logs a SKIP message when a symbol is blocked by a pending order."""
        import logging
        import live.runner as runner_mod

        dfs = {s: _make_ohlcv(60, vol_spike=True) for s in ["AAPL", "SPY"]}
        models = {"AAPL": StubModel(signal="buy", score=0.9)}

        class PendingBroker(StubBroker):
            def get_open_orders(self):
                return {"AAPL"}

        broker = PendingBroker(equity=10_000)
        config = _minimal_config(watchlist=["AAPL"], max_concurrent_positions=2)
        _patch_runner(monkeypatch, dfs, models, tmp_path)

        with caplog.at_level(logging.INFO, logger="live.runner"):
            run_once_multi(config, models, broker, tmp_path)

        skip_msgs = [r.message for r in caplog.records if "pending order" in r.message.lower()]
        assert any("AAPL" in m for m in skip_msgs), "Expected a log line mentioning AAPL pending order"


# ── TestOhlcvFreshness ────────────────────────────────────────────────────────

class TestOhlcvFreshness:
    """Tests for _ensure_fresh_ohlcv() and its integration in run_once_multi."""

    def _stale_df(self, n: int = 60, days_old: int = 30) -> "pd.DataFrame":
        """Build a DataFrame whose last date is *days_old* days ago."""
        from datetime import date, timedelta
        import pandas as pd
        import numpy as np

        end = date.today() - timedelta(days=days_old)
        idx = pd.bdate_range(end=end, periods=n)
        closes = 100.0 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.005, n))
        return pd.DataFrame({
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": np.ones(n) * 1_000_000,
        }, index=idx)

    def _fresh_df(self, n: int = 60) -> "pd.DataFrame":
        """Build a DataFrame whose last date is today (or nearest bday)."""
        import pandas as pd
        import numpy as np
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        closes = 100.0 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.005, n))
        return pd.DataFrame({
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": np.ones(n) * 1_000_000,
        }, index=idx)

    # ── _ensure_fresh_ohlcv unit tests ────────────────────────────────────────

    def test_fresh_data_not_refreshed(self, monkeypatch):
        """Data already fresh (age <= max_age_days) is returned unchanged, no refetch."""
        import live.runner as runner_mod
        calls = []
        monkeypatch.setattr(runner_mod, "fetch_ohlcv", lambda *a, **kw: calls.append(1) or self._fresh_df())

        df = self._fresh_df()
        result = _ensure_fresh_ohlcv("AAPL", df, max_age_days=5)

        assert calls == [], "fetch_ohlcv must not be called for fresh data"
        assert len(result) == len(df)

    def test_stale_data_triggers_refresh(self, monkeypatch):
        """Data older than max_age_days triggers a provider refresh."""
        import live.runner as runner_mod
        fresh = self._fresh_df()
        calls = []

        def fake_fetch(symbol, provider=None, cache=True):
            calls.append(symbol)
            return fresh

        monkeypatch.setattr(runner_mod, "fetch_ohlcv", fake_fetch)
        # Also stub LocalStore so no disk I/O
        import common.data as data_mod
        monkeypatch.setattr(data_mod.LocalStore, "save", lambda self, df, sym, tf="1d": None)

        stale = self._stale_df(days_old=30)
        result = _ensure_fresh_ohlcv("NVDA", stale, max_age_days=5)

        assert "NVDA" in calls, "fetch_ohlcv must be called for stale data"
        assert result.index[-1] == fresh.index[-1], "Returned df should be the refreshed one"

    def test_empty_df_returned_unchanged(self, monkeypatch):
        """Empty DataFrame is returned as-is (no crash, no fetch attempt)."""
        import pandas as pd
        import live.runner as runner_mod
        calls = []
        monkeypatch.setattr(runner_mod, "fetch_ohlcv", lambda *a, **kw: calls.append(1) or self._fresh_df())

        result = _ensure_fresh_ohlcv("TSLA", pd.DataFrame(), max_age_days=5)

        assert calls == []
        assert result.empty

    def test_refresh_failure_falls_back_to_stale(self, monkeypatch):
        """If the provider raises, the stale df is returned and no exception propagates."""
        import live.runner as runner_mod

        def boom(symbol, provider=None, cache=True):
            raise ConnectionError("network down")

        monkeypatch.setattr(runner_mod, "fetch_ohlcv", boom)

        stale = self._stale_df(days_old=30)
        result = _ensure_fresh_ohlcv("AAPL", stale, max_age_days=5)

        assert result is stale, "Must fall back to original stale df on exception"

    def test_refresh_returns_empty_falls_back_to_stale(self, monkeypatch):
        """If provider returns empty DataFrame, stale cached data is kept."""
        import pandas as pd
        import live.runner as runner_mod

        monkeypatch.setattr(runner_mod, "fetch_ohlcv", lambda *a, **kw: pd.DataFrame())

        stale = self._stale_df(days_old=30)
        result = _ensure_fresh_ohlcv("MSFT", stale, max_age_days=5)

        assert result is stale

    def test_cache_updated_after_successful_refresh(self, monkeypatch, tmp_path):
        """After a successful refresh, LocalStore.save() is called to update the cache."""
        import live.runner as runner_mod
        import common.data as data_mod

        fresh = self._fresh_df()
        monkeypatch.setattr(runner_mod, "fetch_ohlcv", lambda *a, **kw: fresh)

        save_calls = []
        monkeypatch.setattr(
            data_mod.LocalStore, "save",
            lambda self, df, sym, tf="1d": save_calls.append(sym),
        )

        stale = self._stale_df(days_old=30)
        _ensure_fresh_ohlcv("AMD", stale, max_age_days=5)

        assert "AMD" in save_calls, "LocalStore.save must be called to persist refreshed data"

    # ── run_once_multi integration tests ──────────────────────────────────────

    def test_run_once_multi_refreshes_stale_symbol(self, monkeypatch, tmp_path):
        """run_once_multi auto-refreshes a symbol whose cached data is stale."""
        import live.runner as runner_mod
        import common.data as data_mod

        fresh_spy = self._fresh_df()
        fresh_aapl = self._fresh_df()
        stale_aapl = self._stale_df(days_old=30)

        fetch_calls: list[str] = []

        def controlled_fetch(symbol, provider=None, cache=True):
            fetch_calls.append(symbol)
            if symbol == "AAPL" and not cache:
                return fresh_aapl
            if symbol == "AAPL":
                return stale_aapl
            return fresh_spy

        monkeypatch.setattr(runner_mod, "fetch_ohlcv", controlled_fetch)
        monkeypatch.setattr(runner_mod, "_build_relative_features",
                            lambda df_stock, df_spy, feature_cols, indicator_spec:
                            __import__("pandas").DataFrame(
                                {c: [0.5] for c in (feature_cols or ["rsi", "macd_hist"])},
                                index=df_stock.index[-1:]))
        monkeypatch.setattr(data_mod.LocalStore, "save", lambda self, df, sym, tf="1d": None)
        monkeypatch.setattr(runner_mod, "_ensure_model_score_calibrations",
                            lambda cfg, mdls, dfs, spy: None)

        models = {"AAPL": StubModel(signal="buy", score=0.9)}
        broker = StubBroker(equity=10_000)
        config = _minimal_config(watchlist=["AAPL"], max_concurrent_positions=2)

        run_once_multi(config, models, broker, tmp_path)

        # fetch_ohlcv must have been called a second time for AAPL with cache=False
        cache_false_aapl = [c for c in fetch_calls if c == "AAPL"]
        assert len(cache_false_aapl) >= 2, (
            "AAPL should have been fetched twice: once from cache, once force-refresh"
        )

    def test_run_once_multi_does_not_refresh_fresh_symbol(self, monkeypatch, tmp_path):
        """run_once_multi does NOT call fetch_ohlcv a second time for an up-to-date symbol."""
        import live.runner as runner_mod
        import common.data as data_mod

        fresh_df = self._fresh_df()
        fetch_calls: list[str] = []

        def controlled_fetch(symbol, provider=None, cache=True):
            fetch_calls.append(symbol)
            return fresh_df

        monkeypatch.setattr(runner_mod, "fetch_ohlcv", controlled_fetch)
        monkeypatch.setattr(runner_mod, "_build_relative_features",
                            lambda df_stock, df_spy, feature_cols, indicator_spec:
                            __import__("pandas").DataFrame(
                                {c: [0.5] for c in (feature_cols or ["rsi", "macd_hist"])},
                                index=df_stock.index[-1:]))
        monkeypatch.setattr(data_mod.LocalStore, "save", lambda self, df, sym, tf="1d": None)
        monkeypatch.setattr(runner_mod, "_ensure_model_score_calibrations",
                            lambda cfg, mdls, dfs, spy: None)

        models = {"AAPL": StubModel(signal="buy", score=0.9)}
        broker = StubBroker(equity=10_000)
        config = _minimal_config(watchlist=["AAPL"], max_concurrent_positions=2)

        run_once_multi(config, models, broker, tmp_path)

        # Each symbol fetched exactly once (no extra refresh call)
        for sym in set(fetch_calls):
            assert fetch_calls.count(sym) == 1, (
                f"{sym} fetched {fetch_calls.count(sym)} times — expected 1 for fresh data"
            )
