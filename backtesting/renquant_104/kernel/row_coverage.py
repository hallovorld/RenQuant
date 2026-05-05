"""Row-coverage gate — shared filter applied at training, inference, AND
calibrator-fit time so the three stages see the same row population.

2026-05-04 P0 root-cause fix for the arm A NaN-leaf calibrator collapse.

Background
----------
Half the universe (~100/183 tickers) has NO hourly/minute bars
historically. Arm A's 19 features include 8 intraday-derived columns →
those 100 tickers' rows have 8/19 features = NaN forever. XGB routes
all-NaN rows to a single terminal leaf, producing identical raw scores
that DOMINATE the calibrator pool (>50% of rows). Spearman undefined,
calibrator fit fails the round-7 ≥5-unique-y floor.

A band-aid filter inside ``fit_global_calibrator`` (mode-collapse
detection) was added 2026-05-03 night. This module is the proper
structural fix: drop low-coverage rows AT ENTRY (training, inference,
calibration) so the model never sees them, the inference never scores
them, and the calibrator pool is never polluted.

Invariants enforced
-------------------

1. ``BuildPanelTask`` filters training panel: rows with < min_pct
   feature coverage are dropped before the LTR is fit.
2. ``BuildFeatureMatrixTask`` filters inference matrix: candidates whose
   today's feature row has < min_pct coverage are excluded from scoring
   (they're effectively no-vote, not a fake-vote-from-NaN-leaf).
3. ``fit_global_calibrator`` sees only rows that the corresponding
   training panel did → no contamination.

Default: disabled (preserves bit-for-bit parity with prior models).
Opt-in via config::

    panel_ltr:
      row_coverage:
        enabled: true
        min_pct: 0.50
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger("kernel.row_coverage")


def filter_by_coverage(
    panel: pd.DataFrame,
    feature_cols: list[str],
    min_pct: float,
    *,
    preserve_index: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows whose non-NaN feature coverage is below ``min_pct``.

    Coverage = (count of non-NaN feature_cols values) / len(feature_cols).
    A row passes when coverage ≥ min_pct.

    Parameters
    ----------
    panel : DataFrame
        Long-form panel with one row per (ticker, date). Must contain
        all of feature_cols as columns.
    feature_cols : list[str]
        The columns whose NaN-rate is measured. Non-feature columns
        (date, ticker, label, weight, …) are ignored.
    min_pct : float in [0, 1]
        Minimum non-NaN fraction required to keep the row. 0 disables
        the filter (returns input unchanged).
    preserve_index : bool, default False
        2026-05-05 wl183 0-trade fix. Training panels carry (date, ticker)
        as columns and use an integer row index, so resetting after the
        filter is safe. Inference matrices (built by
        ``build_inference_matrix``) carry **ticker symbols** in the index
        and downstream tasks (``ApplyScoresTask``, ``ApplyNGBoostTask``,
        ``ApplyGlobalCalibrationTask``) look up scores by
        ``cand.ticker``. A silent reset breaks every per-ticker lookup
        — `scores.get(cand.ticker)` returns None for all candidates →
        0 trades. The wl183 production sim hit exactly this on
        2026-05-05 (X.shape=(57, 21), X.index=int64 → all 57 candidates
        missed). Set ``preserve_index=True`` from any inference caller.

    Returns
    -------
    filtered : DataFrame
        ``preserve_index=False`` (default): index reset to 0..n-1.
        ``preserve_index=True``: original index retained on surviving rows.
        Original column order preserved either way.
    stats : dict
        n_in / n_out / n_dropped / pct_dropped / min_pct / n_features
        — for logging and metadata persistence.
    """
    if not feature_cols or min_pct <= 0.0:
        return panel, {
            "n_in":        int(len(panel)),
            "n_out":       int(len(panel)),
            "n_dropped":   0,
            "pct_dropped": 0.0,
            "min_pct":     float(min_pct),
            "n_features":  int(len(feature_cols)),
            "skipped":     True,
        }
    if min_pct > 1.0:
        raise ValueError(
            f"min_pct must be in [0, 1]; got {min_pct}",
        )

    missing_cols = [c for c in feature_cols if c not in panel.columns]
    if missing_cols:
        log.warning(
            "filter_by_coverage: %d feature_cols not in panel — ignored: %s",
            len(missing_cols), missing_cols[:5],
        )
        feature_cols = [c for c in feature_cols if c in panel.columns]
        if not feature_cols:
            return panel, {
                "n_in": int(len(panel)), "n_out": int(len(panel)),
                "n_dropped": 0, "pct_dropped": 0.0,
                "min_pct": float(min_pct), "n_features": 0,
                "no_features_after_align": True,
            }

    notna_count = panel[feature_cols].notna().sum(axis=1)
    coverage = notna_count / float(len(feature_cols))
    keep_mask = coverage >= min_pct

    filtered = panel.loc[keep_mask]
    if not preserve_index:
        filtered = filtered.reset_index(drop=True)
    n_dropped = int((~keep_mask).sum())
    stats = {
        "n_in":        int(len(panel)),
        "n_out":       int(len(filtered)),
        "n_dropped":   n_dropped,
        "pct_dropped": float(n_dropped / max(1, len(panel))),
        "min_pct":     float(min_pct),
        "n_features":  int(len(feature_cols)),
    }
    return filtered, stats


def coverage_from_config(config: dict) -> tuple[bool, float]:
    """Read row_coverage block from panel_ltr config. Returns (enabled, min_pct)."""
    cfg = (config or {}).get("panel_ltr", {}).get("row_coverage", {})
    enabled = bool(cfg.get("enabled", False))
    min_pct = float(cfg.get("min_pct", 0.50))
    return enabled, min_pct
