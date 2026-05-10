"""Centralised path constants for the asset graph.

All asset bodies validate the existence of these files instead of
re-implementing training. The point of the Dagster layer is the
*dependency graph*, not new compute.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "backtesting" / "renquant_104" / "artifacts"

# Data-tier outputs.
OHLCV_DIR = DATA_DIR / "ohlcv"
SEC_FUNDAMENTALS_PARQUET = DATA_DIR / "sec_fundamentals_daily.parquet"

# Training-tier outputs.
REGIME_ARTIFACT_JSON = ARTIFACTS_DIR / "spy-gmm-regime.json"
# Panel features are an internal pandas parquet; we treat the dataset as a
# directory presence check rather than pinning a specific filename.
PANEL_FEATURES_DATASET = DATA_DIR / "alpha158_291_fundamental_dataset.parquet"
PANEL_MODEL_JSON = ARTIFACTS_DIR / "panel-ltr.alpha158_fund.json"
CALIBRATOR_JSON = ARTIFACTS_DIR / "panel-rank-calibration.json"

# Promotion-tier outputs.
# wf_gate_pass is a small sentinel JSON written by run_wf_gate.py on success.
# It is *separate* from the strategy_config.golden.json swap so that the
# promote_decision asset has a single read-target invariant.
WF_GATE_PASS_JSON = ARTIFACTS_DIR / "wf_gate_pass.json"
PROMOTE_DECISION_JSON = ARTIFACTS_DIR / "promote_decision.json"


def script(name: str) -> Path:
    return SCRIPTS_DIR / name
