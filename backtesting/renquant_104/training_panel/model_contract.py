"""Probabilistic-head input/output contract — validation utilities.

Shared by NGBoostHead, QuantileHead (and any future probabilistic head
implementing predict_distribution → DataFrame[mu, sigma]).

Why this module exists (CLAUDE.md §5.3 invariant for BUG #1/#2/#6 class):
Three silent-feature bugs landed in 24h on 2026-05-08 → 2026-05-09. All
shared the same failure mode: features were corrupted/constant/missing
upstream, the head ran "successfully" but produced degenerate outputs
(e.g. identical μ̂ across all rows), downstream Kelly/QP silently
rejected everything with mu_le_min_edge → 0 trades for the day. No log
made the failure visible.

This module gives every head a uniform pre/post validation that:
  - Catches degenerate INPUT (per-feature variance, NaN rate)
  - Catches degenerate OUTPUT (μ̂ and σ̂ x-sec variance, finite range)
  - Logs ERROR on hard fail with diagnostic context

Use:
    from training_panel.head_contract import (
        validate_input_panel,    # raises HeadInputError if degenerate
        validate_output_dist,    # raises HeadOutputError if degenerate
        soft_check_input,        # logs WARN, returns warnings
        soft_check_output,       # logs WARN, returns warnings
    )

Strict validators raise; soft variants log + return summary so callers
can decide whether to clear candidates or continue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class HeadInputError(ValueError):
    """Raised when input panel violates contract (constant features, etc)."""


class HeadOutputError(ValueError):
    """Raised when prediction output violates contract (collapsed μ̂)."""


@dataclass
class CheckResult:
    ok: bool
    n_rows: int
    n_cols: int
    n_zero_var_cols: int
    n_nan_rows: int
    mu_xs_std: float | None = None
    sigma_xs_std: float | None = None
    n_unique_mu: int | None = None
    n_finite_mu: int | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ── Input contract ────────────────────────────────────────────────────────

INPUT_ZERO_VAR_FRAC_HARD = 0.50   # > 50% constant cols → fail
INPUT_ZERO_VAR_FRAC_SOFT = 0.10   # > 10% constant cols → warn
INPUT_NAN_FRAC_HARD = 0.50        # > 50% rows all-NaN → fail


def soft_check_input(
    X: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    head_name: str = "head",
) -> CheckResult:
    """Inspect input panel without raising. Returns CheckResult + logs WARN.

    Catches the pre-predict variant of BUG #1/#2/#6 class:
      - all-zero fund features (BUG #1)
      - SEC-date misalignment producing NaN col (BUG #2)
      - legacy ctx._panel_matrix lacking alpha158 cols → NaN → median-imputed
        constants (BUG #6)
    """
    feat_cols = list(feature_cols)
    present = [c for c in feat_cols if c in X.columns]
    n_rows = len(X)
    n_cols = len(present)
    if n_rows == 0 or n_cols == 0:
        return CheckResult(
            ok=False, n_rows=n_rows, n_cols=n_cols, n_zero_var_cols=0,
            n_nan_rows=0, warnings=["empty input"],
        )
    sub = X[present]
    n_nan_rows = int(sub.isna().all(axis=1).sum())
    if n_rows >= 2:
        col_stds = sub.std(axis=0, skipna=True).fillna(0.0).values
        n_zero_var_cols = int((np.abs(col_stds) < 1e-12).sum())
    else:
        n_zero_var_cols = 0
    pct_zero = n_zero_var_cols / max(1, n_cols)
    pct_nan_rows = n_nan_rows / max(1, n_rows)
    res = CheckResult(
        ok=True, n_rows=n_rows, n_cols=n_cols,
        n_zero_var_cols=n_zero_var_cols, n_nan_rows=n_nan_rows,
    )
    if pct_zero > INPUT_ZERO_VAR_FRAC_HARD or pct_nan_rows > INPUT_NAN_FRAC_HARD:
        res.ok = False
        res.warnings.append(
            f"{head_name}.input HARD FAIL: pct_zero_var_cols={pct_zero:.1%} "
            f"(>{INPUT_ZERO_VAR_FRAC_HARD:.0%}), pct_nan_rows={pct_nan_rows:.1%}"
        )
        log.error("[head_contract] %s", res.warnings[-1])
    elif pct_zero > INPUT_ZERO_VAR_FRAC_SOFT:
        res.warnings.append(
            f"{head_name}.input SOFT: pct_zero_var_cols={pct_zero:.1%} "
            f"(>{INPUT_ZERO_VAR_FRAC_SOFT:.0%} warn floor) — partial constants"
        )
        log.warning("[head_contract] %s", res.warnings[-1])
    return res


def validate_input_panel(
    X: pd.DataFrame,
    feature_cols: Iterable[str],
    *,
    head_name: str = "head",
) -> CheckResult:
    """Strict variant — raises HeadInputError on hard fail."""
    res = soft_check_input(X, feature_cols, head_name=head_name)
    if not res.ok:
        raise HeadInputError("; ".join(res.warnings))
    return res


# ── Output contract ───────────────────────────────────────────────────────

OUTPUT_MU_XS_STD_HARD = 1e-6      # μ̂ x-sec std < 1e-6 = collapsed
OUTPUT_MU_XS_STD_SOFT = 1e-4      # < 1e-4 = warn
OUTPUT_MIN_UNIQUE_MU = 2          # need ≥2 unique values to call it not constant
OUTPUT_FINITE_FRAC_HARD = 0.50    # < 50% finite → fail


def soft_check_output(
    out: pd.DataFrame,
    *,
    head_name: str = "head",
) -> CheckResult:
    """Inspect predict_distribution output. Returns CheckResult + logs WARN/ERROR.

    Hard fails:
      - μ̂ has < 2 unique finite values (collapsed prediction; BUG #6 case)
      - μ̂ x-sec std < 1e-6
      - < 50% of rows have finite μ̂

    Soft warns:
      - μ̂ x-sec std < 1e-4 (low diversity)
    """
    if "mu" not in out.columns or "sigma" not in out.columns:
        res = CheckResult(
            ok=False, n_rows=len(out), n_cols=0, n_zero_var_cols=0, n_nan_rows=0,
            warnings=[f"{head_name}.output missing mu/sigma columns"],
        )
        log.error("[head_contract] %s", res.warnings[-1])
        return res
    mu = pd.to_numeric(out["mu"], errors="coerce").values
    sigma = pd.to_numeric(out["sigma"], errors="coerce").values
    finite = np.isfinite(mu)
    n_total = len(mu); n_finite = int(finite.sum())
    pct_finite = n_finite / max(1, n_total)
    if n_finite < 2:
        res = CheckResult(
            ok=False, n_rows=n_total, n_cols=0, n_zero_var_cols=0,
            n_nan_rows=int(n_total - n_finite),
            n_finite_mu=n_finite,
            warnings=[
                f"{head_name}.output HARD FAIL: only {n_finite}/{n_total} "
                f"finite μ̂ rows (need ≥2 for diversity check)"
            ],
        )
        log.error("[head_contract] %s", res.warnings[-1])
        return res
    mu_f = mu[finite]
    sigma_f = sigma[finite]
    mu_xs_std = float(mu_f.std())
    sigma_xs_std = float(sigma_f.std()) if len(sigma_f) else 0.0
    n_unique_mu = int(len(np.unique(np.round(mu_f, 8))))
    res = CheckResult(
        ok=True, n_rows=n_total, n_cols=0, n_zero_var_cols=0,
        n_nan_rows=int(n_total - n_finite),
        mu_xs_std=mu_xs_std, sigma_xs_std=sigma_xs_std,
        n_unique_mu=n_unique_mu, n_finite_mu=n_finite,
    )
    if pct_finite < OUTPUT_FINITE_FRAC_HARD:
        res.ok = False
        res.warnings.append(
            f"{head_name}.output HARD FAIL: only {pct_finite:.1%} finite μ̂ "
            f"(< {OUTPUT_FINITE_FRAC_HARD:.0%})"
        )
        log.error("[head_contract] %s", res.warnings[-1])
    if mu_xs_std < OUTPUT_MU_XS_STD_HARD or n_unique_mu < OUTPUT_MIN_UNIQUE_MU:
        res.ok = False
        res.warnings.append(
            f"{head_name}.output HARD FAIL: μ̂ collapsed — "
            f"x-sec std={mu_xs_std:.2e} (< {OUTPUT_MU_XS_STD_HARD:.0e}), "
            f"n_unique={n_unique_mu} (need ≥{OUTPUT_MIN_UNIQUE_MU}). "
            f"Symptom of constant input features (BUG #6) or degenerate model."
        )
        log.error("[head_contract] %s", res.warnings[-1])
    elif mu_xs_std < OUTPUT_MU_XS_STD_SOFT:
        res.warnings.append(
            f"{head_name}.output SOFT: μ̂ x-sec std={mu_xs_std:.2e} "
            f"(< {OUTPUT_MU_XS_STD_SOFT:.0e} warn floor)"
        )
        log.warning("[head_contract] %s", res.warnings[-1])
    return res


def validate_output_dist(
    out: pd.DataFrame,
    *,
    head_name: str = "head",
) -> CheckResult:
    """Strict variant — raises HeadOutputError on hard fail."""
    res = soft_check_output(out, head_name=head_name)
    if not res.ok:
        raise HeadOutputError("; ".join(res.warnings))
    return res


# ── Univariate model output contract (PanelScorer, calibrator, regime probs) ──

OUTPUT_SERIES_XS_STD_HARD = 1e-8   # collapsed scoring
OUTPUT_SERIES_MIN_UNIQUE = 2


def soft_check_score_series(
    scores: pd.Series,
    *,
    model_name: str = "model",
    expected_min: float | None = None,
    expected_max: float | None = None,
) -> CheckResult:
    """Check a univariate model output (e.g. PanelScorer.score, calibrator
    probability, regime probability). Logs ERROR on collapse / out-of-range.

    Hard fails:
      - all values NaN/non-finite
      - x-sec std < 1e-8 (collapsed scoring)
      - n_unique < 2 (constant prediction)
      - any value outside [expected_min, expected_max] when bounds given
    """
    if not isinstance(scores, pd.Series):
        try:
            scores = pd.Series(scores)
        except Exception:
            res = CheckResult(
                ok=False, n_rows=0, n_cols=1,
                n_zero_var_cols=0, n_nan_rows=0,
                warnings=[f"{model_name}.score: not a pd.Series"],
            )
            log.error("[model_contract] %s", res.warnings[-1])
            return res
    arr = pd.to_numeric(scores, errors="coerce").values
    finite = np.isfinite(arr)
    n_total = len(arr); n_finite = int(finite.sum())
    res = CheckResult(
        ok=True, n_rows=n_total, n_cols=1,
        n_zero_var_cols=0, n_nan_rows=int(n_total - n_finite),
    )
    if n_finite < 2:
        res.ok = False
        res.warnings.append(
            f"{model_name}.score HARD FAIL: only {n_finite}/{n_total} finite "
            f"values (need ≥2 for diversity)"
        )
        log.error("[model_contract] %s", res.warnings[-1])
        return res
    fa = arr[finite]
    xs_std = float(fa.std())
    n_unique = int(len(np.unique(np.round(fa, 10))))
    res.mu_xs_std = xs_std
    res.n_unique_mu = n_unique
    res.n_finite_mu = n_finite
    if xs_std < OUTPUT_SERIES_XS_STD_HARD or n_unique < OUTPUT_SERIES_MIN_UNIQUE:
        res.ok = False
        res.warnings.append(
            f"{model_name}.score HARD FAIL: collapsed prediction — "
            f"x-sec std={xs_std:.2e}, n_unique={n_unique}. Symptom of "
            f"constant input or degenerate model."
        )
        log.error("[model_contract] %s", res.warnings[-1])
    if expected_min is not None and float(fa.min()) < expected_min:
        res.ok = False
        res.warnings.append(
            f"{model_name}.score HARD FAIL: min={fa.min():.4f} < {expected_min}"
        )
        log.error("[model_contract] %s", res.warnings[-1])
    if expected_max is not None and float(fa.max()) > expected_max:
        res.ok = False
        res.warnings.append(
            f"{model_name}.score HARD FAIL: max={fa.max():.4f} > {expected_max}"
        )
        log.error("[model_contract] %s", res.warnings[-1])
    if (n_total - n_finite) / max(1, n_total) > 0.25:
        res.warnings.append(
            f"{model_name}.score SOFT: {n_total - n_finite}/{n_total} non-finite (>25%)"
        )
        log.warning("[model_contract] %s", res.warnings[-1])
    return res


def validate_score_series(
    scores: pd.Series,
    *,
    model_name: str = "model",
    expected_min: float | None = None,
    expected_max: float | None = None,
) -> CheckResult:
    """Strict variant — raises HeadOutputError on hard fail."""
    res = soft_check_score_series(
        scores, model_name=model_name,
        expected_min=expected_min, expected_max=expected_max,
    )
    if not res.ok:
        raise HeadOutputError("; ".join(res.warnings))
    return res
