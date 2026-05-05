"""Acceptance-test protocol — schema + meaningfulness assertions.

User mandate (2026-05-04): every output artifact + every Job ctx
mutation + every Pipeline run must verify BOTH structure AND
meaningful data. This module factors the shared assertions so the
three test layers (model / jobs / pipeline) reference one canonical
spec.

Each assertion below is named by what it pins. Update the contract
HERE when artifact schemas change; the tests then read the new spec.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# ── Numeric plausibility bounds ──────────────────────────────────────────────

# CPCV mean_ic bounds — observed historical range is roughly [-0.06, +0.10]
# on the renquant_104 panel; allow generous buffer for regime drift.
MEAN_IC_MIN = -0.20
MEAN_IC_MAX = +0.20

# Per-ticker tournament Sharpe — anything outside this range is suspect.
TOURNAMENT_SHARPE_MIN = -10.0
TOURNAMENT_SHARPE_MAX = +20.0

# Calibrator output is a probability ∈ [0, 1].
PROB_MIN = 0.0
PROB_MAX = 1.0

# σ from NGBoost is an annualized stddev; non-negative, plausibly < 5.0
SIGMA_MIN = 0.0
SIGMA_MAX = 5.0


# ── Required artifact schemas ────────────────────────────────────────────────

PANEL_LTR_REQUIRED_KEYS: set[str] = {
    "kind",
    "feature_cols",
    "booster_raw_json",
}

PANEL_LTR_OPTIONAL_METADATA_KEYS: set[str] = {
    "panel_shape",
    "oos_mean_ic",
    "oos_std_ic",
    "training_train_ic",
    "cv_method",
    "cv_n_splits",
    "config_fingerprint",
}

NGBOOST_REQUIRED_KEYS: set[str] = {
    "feature_cols",
}

CALIBRATOR_REQUIRED_KEYS: set[str] = {
    "probability",
    "expected_return",
}

CALIBRATOR_PROBABILITY_KEYS: set[str] = {
    "x",          # raw score thresholds
    "y",          # calibrated probability values
}

PER_TICKER_POLICY_REQUIRED: set[str] = {
    "policy_type",
    # feature_columns is required for tree-based policy types but NOT
    # for "manual" (which uses score_rules). Per-type validation below.
}


# ── Schema assertions ────────────────────────────────────────────────────────

def assert_required_keys(d: dict, required: set[str], label: str) -> None:
    """Raise AssertionError listing missing keys."""
    missing = required - set(d.keys())
    assert not missing, f"{label}: missing required keys: {sorted(missing)}"


def assert_panel_ltr_artifact(path: Path) -> dict:
    """Verify a panel-ltr.json artifact at `path` is well-formed."""
    assert path.exists(), f"panel-ltr artifact not found: {path}"
    payload = json.loads(path.read_text())
    # Allow transformer / lightgbm shim variants
    kind = payload.get("kind", "panel_ltr_xgboost")
    if kind not in {"panel_ltr_xgboost", "panel_ltr_lightgbm",
                     "panel_transformer"}:
        raise AssertionError(f"unrecognised panel-ltr kind: {kind!r}")
    if kind == "panel_transformer":
        # Shim points to .pt sidecar; lighter checks
        assert "feature_cols" in payload, "transformer shim missing feature_cols"
    else:
        assert_required_keys(payload, PANEL_LTR_REQUIRED_KEYS, str(path))
    # Feature col contract
    fc = payload.get("feature_cols")
    assert isinstance(fc, list), f"feature_cols must be a list, got {type(fc)}"
    assert len(fc) > 0, "feature_cols is empty"
    return payload


def assert_panel_ltr_meaningful(payload: dict, label: str = "panel-ltr") -> None:
    """Range / sanity check the panel-LTR artifact content."""
    # OOS IC must be in plausible range (and finite)
    mean_ic = payload.get("oos_mean_ic")
    if mean_ic is not None:
        assert math.isfinite(mean_ic), f"{label}: oos_mean_ic non-finite ({mean_ic})"
        assert MEAN_IC_MIN <= mean_ic <= MEAN_IC_MAX, (
            f"{label}: oos_mean_ic={mean_ic:+.4f} outside plausible "
            f"range [{MEAN_IC_MIN}, {MEAN_IC_MAX}]"
        )
    # Feature col uniqueness
    fc = payload.get("feature_cols", [])
    assert len(fc) == len(set(fc)), f"{label}: feature_cols has duplicates"


def assert_calibrator_artifact(path: Path) -> dict:
    """Verify panel-rank-calibration.json is well-formed."""
    assert path.exists(), f"calibrator artifact not found: {path}"
    payload = json.loads(path.read_text())
    assert_required_keys(payload, CALIBRATOR_REQUIRED_KEYS, str(path))
    prob = payload["probability"]
    assert_required_keys(prob, CALIBRATOR_PROBABILITY_KEYS, f"{path}::probability")
    return payload


def assert_calibrator_meaningful(payload: dict, label: str = "calibrator") -> None:
    """Range/sanity for calibrator output. Catches the NaN-leaf collapse."""
    prob = payload["probability"]
    xs, ys = prob["x"], prob["y"]
    assert len(xs) >= 2, f"{label}: probability needs ≥2 thresholds, got {len(xs)}"
    assert len(xs) == len(ys), f"{label}: x/y length mismatch ({len(xs)} vs {len(ys)})"
    # Y values must all be probabilities
    for i, y in enumerate(ys):
        assert math.isfinite(y), f"{label}: y[{i}]={y} non-finite"
        assert PROB_MIN <= y <= PROB_MAX, (
            f"{label}: y[{i}]={y} outside [0,1] — calibrator output is "
            f"meant to be a probability"
        )
    # Anti-collapse: at least 5 UNIQUE y values (the NaN-leaf incident
    # collapsed all rows to one bin → one unique y).
    uniq = len({round(float(y), 6) for y in ys})
    assert uniq >= 5, (
        f"{label}: only {uniq} unique y values across {len(ys)} thresholds "
        f"— calibrator collapsed to ~constant. This is the 2026-05-04 "
        f"NaN-leaf incident class. Check row_coverage filter + "
        f"panel feature NaN rate."
    )


def assert_per_ticker_policy(path: Path) -> dict:
    """Verify a per-ticker {ticker}-policy-metadata.json is well-formed.

    Schema differs by policy_type:
      manual         → requires `score_rules` (list)
      classification → requires `feature_columns` + `trees`
      qlearning      → requires `feature_columns` + `bin_edges` + `q_table`
      xgboost        → requires `feature_columns` + xgb_buy/xgb_sell artifact pointers
    """
    assert path.exists(), f"per-ticker policy not found: {path}"
    payload = json.loads(path.read_text())
    assert_required_keys(payload, PER_TICKER_POLICY_REQUIRED, str(path))
    pt = payload.get("policy_type")
    assert pt in {"manual", "classification", "qlearning", "xgboost"}, \
        f"{path}: unknown policy_type {pt!r}"
    # Per-type required keys
    if pt == "manual":
        assert "score_rules" in payload, \
            f"{path}: manual policy must have `score_rules`"
        assert isinstance(payload["score_rules"], list), \
            f"{path}: score_rules must be a list"
    else:
        # All non-manual policies use a feature-vector
        assert "feature_columns" in payload, \
            f"{path}: {pt} policy must have `feature_columns`"
    return payload


def assert_per_ticker_policy_meaningful(payload: dict, label: str) -> None:
    """Range checks on per-ticker policy artifact (type-aware)."""
    sharpe = payload.get("oos_sharpe") or payload.get("sharpe")
    if sharpe is not None and math.isfinite(sharpe):
        assert TOURNAMENT_SHARPE_MIN <= sharpe <= TOURNAMENT_SHARPE_MAX, (
            f"{label}: oos_sharpe={sharpe:+.3f} outside plausible range "
            f"[{TOURNAMENT_SHARPE_MIN}, {TOURNAMENT_SHARPE_MAX}]"
        )
    pt = payload.get("policy_type")
    if pt == "manual":
        # Manual policies have score_rules, not feature_columns
        rules = payload.get("score_rules", [])
        assert isinstance(rules, list) and len(rules) > 0, (
            f"{label}: manual policy must have ≥1 score_rules; got {len(rules)}"
        )
    else:
        fc = payload.get("feature_columns", [])
        assert isinstance(fc, list) and len(fc) > 0, (
            f"{label}: {pt} policy feature_columns missing or empty"
        )
        assert len(fc) == len(set(fc)), \
            f"{label}: duplicate feature_columns"


def assert_data_scan_report(path: Path) -> dict:
    """Verify training_data_scan.json (the 2026-05-04 preflight report)."""
    assert path.exists(), f"data_scan report not found: {path}"
    payload = json.loads(path.read_text())
    for key in ("scan_utc", "today", "watchlist_size", "sources", "alignment"):
        assert key in payload, f"data_scan: missing {key}"
    assert payload["watchlist_size"] >= 1, "data_scan: watchlist_size must be ≥1"
    # Must include daily OHLCV
    assert "daily_ohlcv" in payload["sources"], \
        "data_scan must report daily_ohlcv coverage"
    return payload


def assert_cross_artifact_consistency(
    panel_ltr: dict,
    calibrator: dict | None,
    ngboost: dict | None,
) -> None:
    """Verify feature_cols align across artifacts."""
    panel_fc = set(panel_ltr.get("feature_cols", []))
    if ngboost is not None:
        ngb_fc = set(ngboost.get("feature_cols", []))
        # NGBoost may use a SUBSET (non-derived features); require that
        # NGBoost cols are a subset of panel-LTR cols.
        extra = ngb_fc - panel_fc
        assert not extra, (
            f"NGBoost feature_cols not subset of panel-LTR: extras={sorted(extra)}"
        )


__all__ = [
    "MEAN_IC_MIN", "MEAN_IC_MAX",
    "TOURNAMENT_SHARPE_MIN", "TOURNAMENT_SHARPE_MAX",
    "PROB_MIN", "PROB_MAX",
    "PANEL_LTR_REQUIRED_KEYS",
    "NGBOOST_REQUIRED_KEYS",
    "CALIBRATOR_REQUIRED_KEYS",
    "PER_TICKER_POLICY_REQUIRED",
    "assert_required_keys",
    "assert_panel_ltr_artifact",
    "assert_panel_ltr_meaningful",
    "assert_calibrator_artifact",
    "assert_calibrator_meaningful",
    "assert_per_ticker_policy",
    "assert_per_ticker_policy_meaningful",
    "assert_data_scan_report",
    "assert_cross_artifact_consistency",
]
