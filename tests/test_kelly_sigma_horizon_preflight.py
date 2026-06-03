from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import (  # noqa: E402
    _check_kelly_sigma_horizon_config,
    run_preflight,
)


def _cfg(value=...):
    kelly = {"enabled": True}
    if value is not ...:
        kelly["sigma_horizon_days"] = value
    return {"ranking": {"kelly_sizing": kelly}}


def test_kelly_sigma_horizon_preflight_allows_default() -> None:
    result = _check_kelly_sigma_horizon_config(_cfg())

    assert result.name == "P-KELLY-SIGMA-HORIZON"
    assert result.ok is True
    assert result.details["default_sigma_horizon_days"] == 252.0


def test_kelly_sigma_horizon_preflight_accepts_positive_numeric() -> None:
    result = _check_kelly_sigma_horizon_config(_cfg("60"))

    assert result.ok is True
    assert result.details["sigma_horizon_days"] == 60.0


def test_kelly_sigma_horizon_preflight_rejects_non_positive() -> None:
    result = _check_kelly_sigma_horizon_config(_cfg(0))

    assert result.ok is False
    assert result.severity == "hard"
    assert "must be finite and > 0" in result.message


def test_kelly_sigma_horizon_preflight_rejects_bool() -> None:
    result = _check_kelly_sigma_horizon_config(_cfg(True))

    assert result.ok is False
    assert "got bool True" in result.message


def test_run_preflight_includes_sigma_horizon_gate(tmp_path) -> None:
    results = run_preflight(
        config=_cfg(float("nan")),
        broker=None,
        strategy_dir=tmp_path,
        strict=False,
    )

    by_name = {result.name: result for result in results}
    assert by_name["P-KELLY-SIGMA-HORIZON"].ok is False
