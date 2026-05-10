"""Training-tier assets: regime, panel features, panel model, calibrator.

Bodies are validate-output stubs. Re-running the actual training pipelines
is launchd's job; the Dagster layer only encodes the upstream/downstream
graph that gates ``promote_decision``.
"""

from __future__ import annotations

from dagster import LegacyFreshnessPolicy, asset

from dagster_renquant._paths import (
    CALIBRATOR_JSON,
    PANEL_FEATURES_DATASET,
    PANEL_MODEL_JSON,
    REGIME_ARTIFACT_JSON,
)
from dagster_renquant.assets.data import ohlcv_data, sec_fundamentals

DAILY_FRESHNESS = LegacyFreshnessPolicy(maximum_lag_minutes=24 * 60)
# 7d == fwd_60d weekly retrain floor (see CLAUDE.md §5.13.6).
WEEKLY_FRESHNESS = LegacyFreshnessPolicy(maximum_lag_minutes=7 * 24 * 60)


@asset(
    deps=[ohlcv_data],
    legacy_freshness_policy=DAILY_FRESHNESS,
    description="SPY GMM regime artifact (BULL_CALM/BULL_VOLATILE/CHOPPY/BEAR).",
    group_name="training",
)
def regime_artifact() -> dict:
    if not REGIME_ARTIFACT_JSON.is_file():
        raise FileNotFoundError(f"Regime artifact missing: {REGIME_ARTIFACT_JSON}")
    return {"path": str(REGIME_ARTIFACT_JSON)}


@asset(
    deps=[ohlcv_data, sec_fundamentals],
    legacy_freshness_policy=DAILY_FRESHNESS,
    description="alpha158 + 5 fund + 3 PEAD + 3 SUE = 169-feat panel parquet.",
    group_name="training",
)
def panel_features() -> dict:
    if not PANEL_FEATURES_DATASET.is_file():
        raise FileNotFoundError(
            f"Panel-features dataset missing: {PANEL_FEATURES_DATASET}"
        )
    size_mb = PANEL_FEATURES_DATASET.stat().st_size / (1024 * 1024)
    return {"path": str(PANEL_FEATURES_DATASET), "size_mb": round(size_mb, 2)}


@asset(
    deps=[panel_features],
    # 7d floor — the active label is fwd_60d. Daily retrains add < 5%
    # info per tick (CLAUDE.md §5.13.6).
    legacy_freshness_policy=WEEKLY_FRESHNESS,
    description="XGBoost rank:pairwise panel-LTR model artifact.",
    group_name="training",
)
def panel_model() -> dict:
    if not PANEL_MODEL_JSON.is_file():
        raise FileNotFoundError(f"Panel model missing: {PANEL_MODEL_JSON}")
    size_kb = PANEL_MODEL_JSON.stat().st_size / 1024
    return {"path": str(PANEL_MODEL_JSON), "size_kb": round(size_kb, 1)}


@asset(
    deps=[panel_model],
    legacy_freshness_policy=WEEKLY_FRESHNESS,
    description="Isotonic calibrator (panel-rank-calibration.json) on panel-LTR scores.",
    group_name="training",
)
def calibrator() -> dict:
    if not CALIBRATOR_JSON.is_file():
        raise FileNotFoundError(f"Calibrator missing: {CALIBRATOR_JSON}")
    return {"path": str(CALIBRATOR_JSON)}
