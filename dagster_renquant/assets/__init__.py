"""Asset modules grouped by lifecycle stage."""

from dagster_renquant.assets.data import ohlcv_data, sec_fundamentals
from dagster_renquant.assets.training import (
    calibrator,
    panel_features,
    panel_model,
    regime_artifact,
)
from dagster_renquant.assets.promote import promote_decision, wf_gate_pass

ALL_ASSETS = [
    ohlcv_data,
    sec_fundamentals,
    regime_artifact,
    panel_features,
    panel_model,
    calibrator,
    wf_gate_pass,
    promote_decision,
]

__all__ = [
    "ALL_ASSETS",
    "ohlcv_data",
    "sec_fundamentals",
    "regime_artifact",
    "panel_features",
    "panel_model",
    "calibrator",
    "wf_gate_pass",
    "promote_decision",
]
