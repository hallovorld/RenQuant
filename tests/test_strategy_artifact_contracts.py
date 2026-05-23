"""On-disk strategy/artifact contracts for renquant_104."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.walk_forward import (  # noqa: E402
    assert_correlation_no_leakage,
    assert_gmm_no_leakage,
    assert_no_leakage,
    parse_correlation_artifact,
)


def _load_config(name: str) -> dict:
    return json.loads((STRATEGY_DIR / name).read_text())


def _artifact_path(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else STRATEGY_DIR / "artifacts" / path


def _resolve_config_path(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else (STRATEGY_DIR / path).resolve()


def test_active_prod_correlation_artifact_covers_watchlist() -> None:
    cfg = _load_config("strategy_config.json")
    corr_rel = cfg["regime"]["correlation_artifact"]
    raw = json.loads(_artifact_path(corr_rel).read_text())
    matrix, as_of = parse_correlation_artifact(raw)

    missing = sorted(set(cfg["watchlist"]) - set(matrix))
    extra = sorted(set(matrix) - set(cfg["watchlist"]))
    assert as_of is not None
    assert not missing, f"{corr_rel} missing watchlist tickers: {missing}"
    assert not extra, f"{corr_rel} has stale non-watchlist tickers: {extra}"


def test_active_prod_gmm_artifact_is_stamped_for_live_mode() -> None:
    cfg = _load_config("strategy_config.json")
    gmm_rel = cfg["regime"]["gmm_artifact"]
    raw = json.loads(_artifact_path(gmm_rel).read_text())
    assert raw.get("as_of_date") or raw.get("trained_date")
    assert_gmm_no_leakage(
        raw,
        cfg["backtest_start"],
        is_live_mode=True,
        context="live-prod-contract",
    )


@pytest.mark.parametrize(
    "name",
    [
        "strategy_config.sim_patchtst_clean_20260522.json",
        "strategy_config.sim_xgb_truly_oos_20260522.json",
    ],
)
def test_2024_sim_configs_use_leakage_free_static_artifacts(name: str) -> None:
    cfg = _load_config(name)
    start = cfg["backtest_start"]
    assert pd.Timestamp(start) <= pd.Timestamp("2024-01-01")

    corr_rel = cfg["regime"]["correlation_artifact"]
    gmm_rel = cfg["regime"]["gmm_artifact"]
    assert corr_rel.startswith("sim/")
    assert gmm_rel.startswith("sim/")

    corr_raw = json.loads(_artifact_path(corr_rel).read_text())
    _matrix, corr_as_of = parse_correlation_artifact(corr_raw)
    assert_correlation_no_leakage(
        corr_as_of,
        start,
        context=f"{name} corr",
    )

    gmm_raw = json.loads(_artifact_path(gmm_rel).read_text())
    assert_gmm_no_leakage(
        gmm_raw,
        start,
        context=f"{name} gmm",
    )


def test_patchtst_shadow_artifact_has_selection_contract_sidecar() -> None:
    cfg = _load_config("strategy_config.sim_patchtst_clean_20260522.json")
    artifact = (STRATEGY_DIR / cfg["ranking"]["panel_scoring"]["artifact_path"]).resolve()
    sidecar = artifact.with_name(artifact.name + ".metadata.json")
    raw = json.loads(sidecar.read_text())

    assert raw["trained_date"] == "2026-05-22"
    assert raw["effective_train_cutoff_date"] == "2024-11-13"
    assert raw["effective_selection_cutoff_date"] == "2026-02-10"
    assert raw["lookahead_days"] == 60
    with pytest.raises(ValueError, match="Look-ahead leakage"):
        assert_no_leakage(
            raw["effective_selection_cutoff_date"],
            "2025-03-01",
            context="patchtst sidecar contract",
            lookahead_days=int(raw["lookahead_days"]),
        )


def test_shadow_patchtst_calibration_matches_shadow_scorer() -> None:
    cfg = _load_config("strategy_config.shadow.json")
    panel_cfg = cfg["ranking"]["panel_scoring"]
    scorer = _resolve_config_path(panel_cfg["artifact_path"])
    calib_cfg = panel_cfg["global_calibration"]
    calib = json.loads(_resolve_config_path(calib_cfg["artifact_path"]).read_text())
    calib_meta = calib.get("metadata") or {}

    assert calib_cfg.get("strict_scorer_match") is True
    assert Path(calib_meta["scorer_artifact"]).resolve() == scorer
    assert "patchtst_shadow" in panel_cfg["artifact_path"]
    assert "patchtst_shadow" in calib_meta["scorer_artifact"]


@pytest.mark.parametrize(
    "name",
    [
        "strategy_config.sim_patchtst_clean_20260522.json",
        "strategy_config.sim_xgb_truly_oos_20260522.json",
    ],
)
def test_2024_sim_ngboost_overlays_do_not_reference_missing_artifact(name: str) -> None:
    """If a sim can activate NGBoost by regime, its configured head must exist."""
    cfg = _load_config(name)
    panel_cfg = cfg["ranking"]["panel_scoring"]
    ngb_cfg = panel_cfg.get("ngboost") or {}
    globally_enabled = ngb_cfg.get("enabled") is True
    active_regimes = [
        regime
        for regime, params in (cfg.get("regime_params") or {}).items()
        if isinstance(params, dict)
        and isinstance(params.get("ngboost"), dict)
        and params["ngboost"].get("enabled") is True
    ]
    if not globally_enabled and not active_regimes:
        return

    artifact = ngb_cfg.get("artifact_path")
    assert artifact, f"{name}: NGBoost active but artifact_path missing"
    assert _resolve_config_path(artifact).exists(), (
        f"{name}: NGBoost active for {active_regimes or ['GLOBAL']} but "
        f"artifact does not exist: {artifact}"
    )
