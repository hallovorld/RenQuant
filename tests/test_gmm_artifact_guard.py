"""Regression tests for SPY GMM regime-artifact leakage guards."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest


STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from adapters.sim import SimAdapter  # noqa: E402
from kernel.walk_forward import assert_gmm_no_leakage, gmm_artifact_as_of  # noqa: E402
from training.regime import RegimeGMM  # noqa: E402


def test_gmm_artifact_as_of_prefers_explicit_as_of_date() -> None:
    assert gmm_artifact_as_of({"as_of_date": "2023-12-29", "trained_date": "2026-05-22"}) == "2023-12-29"


def test_gmm_artifact_as_of_uses_training_window_end_before_wall_clock_train_date() -> None:
    assert gmm_artifact_as_of({
        "training_window": ["2012-01-01", "2022-01-01"],
        "trained_date": "2026-05-15T04:42:20.574973+00:00",
    }) == "2022-01-01"


def test_gmm_guard_raises_when_artifact_after_backtest_start() -> None:
    with pytest.raises(ValueError, match="GMM regime artifact"):
        assert_gmm_no_leakage(
            {"as_of_date": "2026-05-22"},
            "2024-01-01",
            context="unit-test",
        )


def test_gmm_guard_passes_when_artifact_before_backtest_start() -> None:
    assert_gmm_no_leakage(
        {"as_of_date": "2023-12-29"},
        "2024-01-01",
        context="unit-test",
    )


def test_gmm_guard_accepts_tz_aware_hmm_training_window_before_start() -> None:
    assert_gmm_no_leakage(
        {
            "training_window": ["2012-01-01", "2022-01-01"],
            "trained_date": "2026-05-15T04:42:20.574973+00:00",
        },
        pd.Timestamp("2024-01-02"),
        context="hmm-unit-test",
    )


def test_gmm_guard_normalizes_tz_aware_as_of_dates() -> None:
    assert_gmm_no_leakage(
        {"as_of_date": "2023-12-29T00:00:00+00:00"},
        "2024-01-01",
        context="tz-unit-test",
    )


def test_gmm_guard_warns_on_legacy_artifact(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="kernel.walk_forward.gmm_guard"):
        assert_gmm_no_leakage(
            {"means": []},
            "2024-01-01",
            context="legacy-test",
        )
    assert any("no as_of_date" in r.message for r in caplog.records)


def test_regime_gmm_save_stamps_metadata(tmp_path: Path) -> None:
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    features = pd.DataFrame(
        {
            "10d_return": [i / 1000 for i in range(80)],
            "20d_realized_vol": [0.1 + (i % 7) / 100 for i in range(80)],
            "spy_adx": [20 + (i % 5) for i in range(80)],
            "return_autocorr": [0.01 * ((i % 9) - 4) for i in range(80)],
        },
        index=dates,
    )
    model = RegimeGMM(n_components=3, random_state=7, n_init=2).fit(features)
    out = tmp_path / "spy-gmm-regime.json"

    model.save(
        out,
        as_of_date="2023-04-21",
        data_window_start="2023-01-02",
        data_window_end="2023-04-21",
        n_train_rows=len(features),
    )

    raw = json.loads(out.read_text())
    assert raw["as_of_date"] == "2023-04-21"
    assert raw["trained_date"] == "2023-04-21"
    assert raw["data_window_start"] == "2023-01-02"
    assert raw["data_window_end"] == "2023-04-21"
    assert raw["n_train_rows"] == len(features)


class TestSimAdapterGmmHistoricalFallback:
    """SimAdapter substitutes historical sim GMM when prod artifact would leak.

    Regression guard for 2026-06-02 WF gate sim failure: prod-semantic WF
    configs inherit ``prod/spy-gmm-regime.json`` (as_of today) from
    production but run historical 2024 backtests. SimAdapter must swap to
    ``sim/spy-gmm-regime.json`` BEFORE the leakage guard fires. If the
    historical artifact is missing the guard MUST still raise (fail-closed).
    """

    def _write_gmm(self, path: Path, as_of_date: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "as_of_date": as_of_date,
            "trained_date": as_of_date,
            "feature_order": ["10d_return", "20d_realized_vol", "spy_adx", "return_autocorr"],
            "means": [[0.0] * 4] * 3,
            "covariances": [[[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]] * 3,
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "cluster_labels": ["BULL_CALM", "BULL_VOLATILE", "BEAR"],
            "scaler_mean": [0.0] * 4,
            "scaler_scale": [1.0] * 4,
        }, indent=2))

    def test_substitutes_historical_when_configured_artifact_leaks(self, tmp_path: Path, caplog) -> None:
        self._write_gmm(tmp_path / "artifacts" / "prod" / "spy-gmm-regime.json", "2026-05-22")
        self._write_gmm(tmp_path / "artifacts" / "sim" / "spy-gmm-regime.json", "2023-12-29")

        adp = SimAdapter.__new__(SimAdapter)
        adp._strategy_dir = tmp_path
        adp._config = {
            "backtest_start": "2024-01-02",
            "regime": {"gmm_artifact": "prod/spy-gmm-regime.json"},
        }

        with caplog.at_level(logging.WARNING, logger="adapters.sim"):
            gmm, _earnings, _corr = adp._load_artifacts(fallback_corr=None)

        assert gmm is not None
        assert gmm_artifact_as_of(gmm) == "2023-12-29"
        assert any("substituting historical sim artifact" in r.message for r in caplog.records)

    def test_raises_when_no_historical_alternative_available(self, tmp_path: Path) -> None:
        self._write_gmm(tmp_path / "artifacts" / "prod" / "spy-gmm-regime.json", "2026-05-22")
        # No sim/spy-gmm-regime.json written — fail-closed path.

        adp = SimAdapter.__new__(SimAdapter)
        adp._strategy_dir = tmp_path
        adp._config = {
            "backtest_start": "2024-01-02",
            "regime": {"gmm_artifact": "prod/spy-gmm-regime.json"},
        }

        with pytest.raises(ValueError, match="GMM regime artifact"):
            adp._load_artifacts(fallback_corr=None)

    def test_raises_when_historical_alternative_also_leaks(self, tmp_path: Path) -> None:
        self._write_gmm(tmp_path / "artifacts" / "prod" / "spy-gmm-regime.json", "2026-05-22")
        self._write_gmm(tmp_path / "artifacts" / "sim" / "spy-gmm-regime.json", "2025-06-01")

        adp = SimAdapter.__new__(SimAdapter)
        adp._strategy_dir = tmp_path
        adp._config = {
            "backtest_start": "2024-01-02",
            "regime": {"gmm_artifact": "prod/spy-gmm-regime.json"},
        }

        with pytest.raises(ValueError, match="GMM regime artifact"):
            adp._load_artifacts(fallback_corr=None)

    def test_does_not_substitute_when_configured_artifact_is_already_safe(self, tmp_path: Path, caplog) -> None:
        self._write_gmm(tmp_path / "artifacts" / "sim" / "spy-gmm-regime.json", "2023-12-29")
        self._write_gmm(tmp_path / "artifacts" / "prod" / "spy-gmm-regime.json", "2026-05-22")

        adp = SimAdapter.__new__(SimAdapter)
        adp._strategy_dir = tmp_path
        adp._config = {
            "backtest_start": "2024-01-02",
            "regime": {"gmm_artifact": "sim/spy-gmm-regime.json"},
        }

        with caplog.at_level(logging.WARNING, logger="adapters.sim"):
            gmm, _earnings, _corr = adp._load_artifacts(fallback_corr=None)

        assert gmm_artifact_as_of(gmm) == "2023-12-29"
        assert not any("substituting historical sim artifact" in r.message for r in caplog.records)


def test_regime_gmm_fit_rejects_nonfinite_features() -> None:
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    features = pd.DataFrame(
        {
            "10d_return": [0.01] * 10,
            "20d_realized_vol": [0.2] * 9 + [float("inf")],
            "spy_adx": [20.0] * 10,
            "return_autocorr": [0.0] * 10,
        },
        index=dates,
    )
    with pytest.raises(ValueError, match="non-finite"):
        RegimeGMM(n_components=2).fit(features)
