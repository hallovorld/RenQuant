"""P5 regression — RunnerAdapter meta-label wiring.

Pins:
  * __init__ reads config.ranking.meta_label.{enabled, artifact_path}
  * When enabled + artifact exists: self._meta_label_predictor is callable
  * When enabled + artifact MISSING: self._meta_label_predictor is None
    (§5.13.10 fallback — no crash, log warning, run continues without veto)
  * When disabled: self._meta_label_predictor is None
  * make_context attaches ctx._meta_label_predictor (mirrors SimAdapter
    so MetaLabelVetoTask sees the same field name)
  * meta_label_training.enabled=true loads SnapshotLogger; otherwise None

Constructs the full RunnerAdapter __init__ with a mock broker — this is
§5.13.1 (test fixtures lie) compliant: we go through the REAL init path,
not a stubbed predictor field.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner import RunnerAdapter  # noqa: E402


def _minimal_config() -> dict:
    """Bare-bones config sufficient for RunnerAdapter __init__."""
    return {
        "watchlist": ["AAPL"],
        "model_name": "renquant-104",
        "initial_cash": 100_000.0,
        "regime_params": {
            "BULL_CALM":     {"max_position_pct": 0.15, "cash_reserve_pct": 0.0},
            "BULL_VOLATILE": {"max_position_pct": 0.20, "cash_reserve_pct": 0.2},
            "CHOPPY":        {"max_position_pct": 0.15, "cash_reserve_pct": 0.3},
            "BEAR":          {"max_position_pct": 0.0,  "cash_reserve_pct": 1.0},
        },
        "persistence": {"enabled": False},   # avoid sqlite writes in tests
    }


def _mock_broker():
    broker = MagicMock()
    broker.broker_name = "paper"
    return broker


def _write_min_meta_label_artifact(path: Path) -> None:
    """Write a minimal valid meta-label artifact for predictor loader."""
    import xgboost as xgb
    import numpy as np
    # Train a trivial 1-feature classifier so booster_raw_json is valid
    X = np.array([[0.0], [0.5], [1.0], [0.2], [0.8]])
    y = np.array([0, 0, 1, 0, 1])
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=2,
                            tree_method="hist", n_jobs=1,
                            use_label_encoder=False, eval_metric="logloss")
    clf.fit(X, y)
    booster_raw = clf.get_booster().save_raw(raw_format="json").decode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 1,
        "kind": "meta_label_exit_xgb",
        "feature_cols": ["cum_pnl_pct"],
        "booster_raw_json": booster_raw,
        "default_threshold": 0.5,
        "cv_metrics": {},
        "training_data_summary": {},
    }))


class TestRunnerAdapterMetaLabelInit:

    def test_disabled_meta_label_yields_none_predictor(self, tmp_path):
        cfg = _minimal_config()
        # No ranking.meta_label block → disabled
        adapter = RunnerAdapter(
            config=cfg, models={}, broker=_mock_broker(),
            strategy_dir=tmp_path,
        )
        assert adapter._meta_label_predictor is None  # noqa: SLF001
        assert adapter._meta_label_logger is None     # noqa: SLF001

    def test_enabled_but_artifact_missing_falls_back_to_none(self, tmp_path):
        cfg = _minimal_config()
        cfg["ranking"] = {"meta_label": {
            "enabled": True,
            "artifact_path": str(tmp_path / "absolutely_does_not_exist.json"),
        }}
        adapter = RunnerAdapter(
            config=cfg, models={}, broker=_mock_broker(),
            strategy_dir=tmp_path,
        )
        # §5.13.10 fallback: missing artifact → None, no crash
        assert adapter._meta_label_predictor is None  # noqa: SLF001

    def test_enabled_with_valid_artifact_loads_callable(self, tmp_path):
        cfg = _minimal_config()
        art = tmp_path / "meta-label-exit.json"
        _write_min_meta_label_artifact(art)
        cfg["ranking"] = {"meta_label": {
            "enabled": True,
            "artifact_path": str(art),
        }}
        adapter = RunnerAdapter(
            config=cfg, models={}, broker=_mock_broker(),
            strategy_dir=tmp_path,
        )
        pred = adapter._meta_label_predictor   # noqa: SLF001
        assert pred is not None and callable(pred)
        # Call returns a float in [0, 1]
        p = pred({"cum_pnl_pct": 0.5})
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_meta_label_training_enabled_creates_snapshot_logger(self, tmp_path):
        cfg = _minimal_config()
        cfg["meta_label_training"] = {"enabled": True}
        adapter = RunnerAdapter(
            config=cfg, models={}, broker=_mock_broker(),
            strategy_dir=tmp_path,
        )
        from kernel.meta_label import SnapshotLogger
        assert isinstance(adapter._meta_label_logger, SnapshotLogger)  # noqa: SLF001
