"""Per-attribute schema definitions used by the exhaustive test factory.

Each attribute carries:
  type        — Python type or tuple of types it must be
  required    — True/False (False = optional, skip if absent)
  finite      — True for numeric attributes that must be finite (no NaN/inf)
  bounds      — (lo, hi) for numeric attributes (closed interval)
  non_empty   — True for list/dict/string attributes that must be non-empty
  unique      — True for list attributes whose elements must all be unique
  validator   — optional custom callable(value) -> None that raises
                 AssertionError on bad values

The exhaustive test runner reads these specs and emits one test per
(artifact, attribute, check_kind) triple via pytest.parametrize.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable


# ── Custom validators ────────────────────────────────────────────────────────

def _iso_date(s: Any) -> None:
    """Validate ISO-8601 date string (YYYY-MM-DD or full datetime)."""
    assert isinstance(s, str), f"trained_date must be str, got {type(s)}"
    assert re.match(r"^\d{4}-\d{2}-\d{2}", s), \
        f"trained_date must start with YYYY-MM-DD, got {s!r}"


def _b64_nonempty(s: Any) -> None:
    """Validate base64-encoded non-empty string."""
    assert isinstance(s, str), f"must be str, got {type(s)}"
    assert len(s) > 32, f"b64 payload too short ({len(s)} chars) — likely empty"


def _booster_json(s: Any) -> None:
    """Validate XGBoost booster_raw_json — must be parseable JSON-like."""
    assert isinstance(s, str), f"booster_raw_json must be str, got {type(s)}"
    assert len(s) > 1024, \
        f"booster_raw_json suspiciously short ({len(s)} chars) — empty model?"


def _params_dict(d: Any) -> None:
    """xgb_params must contain at least eta or learning_rate."""
    assert isinstance(d, dict), f"params must be dict, got {type(d)}"
    has_eta = "eta" in d or "learning_rate" in d
    assert has_eta, "xgb params must include eta or learning_rate"


def _panel_shape_dict(d: Any) -> None:
    assert isinstance(d, dict), f"panel_shape must be dict"
    for k in ("rows", "tickers", "dates"):
        assert k in d, f"panel_shape missing {k}"
        assert isinstance(d[k], int) and d[k] >= 0, \
            f"panel_shape.{k} must be non-negative int, got {d[k]!r}"


def _per_fold_list(v: Any) -> None:
    assert isinstance(v, list), f"per_fold_ic must be list"
    assert len(v) >= 1, "per_fold_ic must have ≥1 fold"
    for i, x in enumerate(v):
        assert isinstance(x, (int, float)), f"per_fold_ic[{i}] must be number"
        assert math.isfinite(float(x)), f"per_fold_ic[{i}] non-finite ({x})"
        assert -0.5 <= float(x) <= 0.5, \
            f"per_fold_ic[{i}]={x} outside plausible IC range"


def _quantiles_dict(d: Any) -> None:
    assert isinstance(d, dict), f"oos_ic_quantiles must be dict"
    for q in ("q05", "q50", "q95"):
        assert q in d, f"oos_ic_quantiles missing {q}"
        v = d[q]
        assert math.isfinite(float(v)), f"quantile {q}={v} non-finite"
    # Monotonic ordering
    assert d["q05"] <= d["q50"] <= d["q95"], \
        f"quantiles not monotonic: q05={d['q05']} q50={d['q50']} q95={d['q95']}"


# ── Schema dictionaries — one per artifact type ──────────────────────────────

PANEL_LTR_SCHEMA: dict[str, dict] = {
    # Identity / kind
    # Legacy XGB-only artifacts (panel-ltr.golden-daily, panel-ltr.hourly,
    # etc., trained pre-2026-04 before kind-dispatch was added) lack the
    # `kind` field. Production loader (`PanelScorer.load`) treats missing
    # kind as XGBoost default — accept the same here.
    "kind":                {"type": str, "required": False, "non_empty": True,
                            "allowed": {"panel_ltr_xgboost", "panel_ltr_lightgbm",
                                         "panel_transformer"}},
    "version":             {"type": (int, str), "required": False},
    "trained_date":        {"type": str, "required": False, "validator": _iso_date},
    "train_run_id":        {"type": (str, type(None)), "required": False},
    "training_notes":      {"type": (str, type(None)), "required": False},
    # Feature contract
    "feature_cols":        {"type": list, "required": True, "non_empty": True,
                            "unique": True, "min_len": 5},
    # Booster
    "booster_raw_json":    {"type": str, "required": True, "non_empty": True,
                            "validator": _booster_json},
    "params":              {"type": dict, "required": False,
                            "validator": _params_dict},
    "best_iter":           {"type": (int, type(None)), "required": False,
                            "bounds": (0, 10000)},
    # CV metadata
    "cv_method":           {"type": (str, type(None)), "required": False,
                            "allowed_optional": {"purged", "cpcv", None}},
    "cv_n_splits":         {"type": (int, type(None)), "required": False,
                            "bounds": (2, 50)},
    "cv_n_test_groups":    {"type": (int, type(None)), "required": False,
                            "bounds": (1, 20)},
    "cv_embargo_days":     {"type": (int, type(None)), "required": False,
                            "bounds": (0, 60)},
    "lookahead_days":      {"type": (int, type(None)), "required": False,
                            "bounds": (1, 60)},
    "beta_window":         {"type": (int, type(None)), "required": False,
                            "bounds": (10, 1000)},
    "min_history_days":    {"type": (int, type(None)), "required": False,
                            "bounds": (10, 1000)},
    "neutralize_features": {"type": (bool, type(None)), "required": False},
    # IC metrics — the meaningful-data heart
    "oos_mean_ic":         {"type": (int, float), "required": False,
                            "finite": True, "bounds": (-0.20, 0.20)},
    "oos_std_ic":          {"type": (int, float), "required": False,
                            "finite": True, "bounds": (0.0, 0.20)},
    "training_train_ic":   {"type": (int, float), "required": False,
                            "finite": True, "bounds": (-0.5, 1.0)},
    "oos_per_fold_ic":     {"type": list, "required": False,
                            "validator": _per_fold_list},
    "oos_ic_quantiles":    {"type": (dict, type(None)), "required": False,
                            "validator": _quantiles_dict},
    "panel_shape":         {"type": dict, "required": False,
                            "validator": _panel_shape_dict},
    # Config-fingerprint stamping (drift detection)
    "config_fingerprint":  {"type": (str, type(None)), "required": False,
                            "min_str_len": 8},
    # Allow dict (production format with hashed field-name → fingerprint)
    # or list (legacy format with just the field names). Both are valid.
    "config_fingerprint_fields": {"type": (list, dict, type(None)), "required": False},
}


NGBOOST_SCHEMA: dict[str, dict] = {
    "kind":                {"type": str, "required": False},
    "version":             {"type": (int, str), "required": False},
    "trained_date":        {"type": str, "required": False, "validator": _iso_date},
    "train_run_id":        {"type": (str, type(None)), "required": False},
    "feature_cols":        {"type": list, "required": True, "non_empty": True,
                            "unique": True, "min_len": 5},
    # feature_medians is stored as a list (one float per feature_col index)
    # in current artifacts; some legacy versions used a dict.
    "feature_medians":     {"type": (list, dict, type(None)), "required": False},
    "regressor_pickle_b64":{"type": str, "required": True, "non_empty": True,
                            "validator": _b64_nonempty},
    "params":              {"type": (dict, type(None)), "required": False},
    "best_iter":           {"type": (int, type(None)), "required": False,
                            "bounds": (0, 10000)},
    # Train data shape
    "n_rows":              {"type": int, "required": False, "bounds": (0, 10_000_000)},
    "n_rows_train":        {"type": int, "required": False, "bounds": (0, 10_000_000)},
    "n_rows_val":          {"type": int, "required": False, "bounds": (0, 10_000_000)},
    "n_rows_dropped":      {"type": int, "required": False, "bounds": (0, 10_000_000)},
    # Train metrics
    "train_mu_ic":         {"type": (int, float), "required": False,
                            "finite": True, "bounds": (-0.5, 1.0)},
    "train_mu_mean":       {"type": (int, float), "required": False,
                            "finite": True, "bounds": (-1.0, 1.0)},
    "train_sigma_mean":    {"type": (int, float), "required": False,
                            "finite": True, "bounds": (0.0, 5.0)},
    "val_mu_ic":           {"type": (int, float), "required": False,
                            "finite": True, "bounds": (-0.5, 1.0)},
    "training_notes":      {"type": (str, type(None)), "required": False},
}


CALIBRATOR_SCHEMA: dict[str, dict] = {
    "kind":                {"type": (str, type(None)), "required": False},
    "version":             {"type": (int, str), "required": False},
    "trained_date":        {"type": str, "required": False, "validator": _iso_date},
    "probability":         {"type": dict, "required": True, "non_empty": True},
    "expected_return":     {"type": dict, "required": True, "non_empty": True},
    "metadata":            {"type": (dict, type(None)), "required": False},
}


# Per-ticker policy schemas — split by policy_type
PER_TICKER_COMMON_SCHEMA: dict[str, dict] = {
    "policy_type":         {"type": str, "required": True, "non_empty": True,
                            "allowed": {"manual", "classification",
                                         "qlearning", "xgboost"}},
    "buy_threshold":       {"type": (int, float, type(None)), "required": False,
                            "finite": True, "bounds": (-2.0, 2.0)},
    "sell_threshold":      {"type": (int, float, type(None)), "required": False,
                            "finite": True, "bounds": (-2.0, 2.0)},
    "trained_date":        {"type": (str, type(None)), "required": False},
    "model_name":          {"type": (str, type(None)), "required": False},
    "best_approach":       {"type": (str, type(None)), "required": False},
    "sharpe":              {"type": (int, float, type(None)), "required": False,
                            "finite": False, "bounds": (-10.0, 20.0)},
    "lookahead":           {"type": (int, type(None)), "required": False,
                            "bounds": (1, 60)},
}


PER_TICKER_MANUAL_SCHEMA: dict[str, dict] = {
    "score_rules":         {"type": list, "required": True, "non_empty": True,
                            "min_len": 1},
}


PER_TICKER_TREE_SCHEMA: dict[str, dict] = {
    "feature_columns":     {"type": list, "required": True, "non_empty": True,
                            "unique": True, "min_len": 3},
}


# All schemas for the exhaustive test runner to iterate
ALL_SCHEMAS: dict[str, dict[str, dict]] = {
    "panel_ltr":  PANEL_LTR_SCHEMA,
    "ngboost":    NGBOOST_SCHEMA,
    "calibrator": CALIBRATOR_SCHEMA,
}


__all__ = [
    "PANEL_LTR_SCHEMA",
    "NGBOOST_SCHEMA",
    "CALIBRATOR_SCHEMA",
    "PER_TICKER_COMMON_SCHEMA",
    "PER_TICKER_MANUAL_SCHEMA",
    "PER_TICKER_TREE_SCHEMA",
    "ALL_SCHEMAS",
]
