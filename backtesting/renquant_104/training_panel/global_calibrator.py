"""Global panel-wide calibrator.

Replaces the 38 per-ticker score_calibration objects (fitted on ~2500 OOS
rows each) with a single calibrator fitted on the pooled panel
(~80,000 rows). Two heads:

  1. **Probability head** — `rank_score ∈ [0, 1] = P(outperform SPY by threshold%)`
     via isotonic on (panel_score, future_excess_indicator).
  2. **Expected-return head** — `E[R_i - R_spy]` over rotation horizon via
     isotonic on (panel_score, future_excess_return).

Rationale: cross-sectional LTR already produces comparable raw scores;
fitting 38 separate calibrators wastes data. A single calibrator uses 38×
more observations for the same statistical quantity.

JSON artifact format, loadable without unpickling:
    {
      "version": 1, "kind": "global_panel_calibration",
      "probability": {"x": [...], "y": [...]},
      "expected_return": {"x": [...], "y": [...]},
      "metadata": {...}
    }
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger("training_panel.global_calibrator")


@dataclass
class GlobalPanelCalibration:
    """Two isotonic maps: raw → P(outperform) and raw → E[R_i - R_spy].

    Audit fix GC-1 (2026-04-25): the `prob_x` and `er_x` arrays must
    be monotonically non-decreasing — `np.interp` requires it and
    silently produces garbage if the invariant is violated. We assert
    on construction (and load).
    """
    prob_x: np.ndarray              # knot x's (panel scores, sorted)
    prob_y: np.ndarray              # knot y's (probabilities, monotone nondecreasing)
    er_x:   np.ndarray
    er_y:   np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce to ndarray to keep np.interp happy regardless of how
        # the constructor was called (tests pass lists; load() passes
        # ndarrays).
        self.prob_x = np.asarray(self.prob_x, dtype=float)
        self.prob_y = np.asarray(self.prob_y, dtype=float)
        self.er_x   = np.asarray(self.er_x,   dtype=float)
        self.er_y   = np.asarray(self.er_y,   dtype=float)
        for name, arr in (("prob_x", self.prob_x), ("er_x", self.er_x)):
            if len(arr) >= 2 and not np.all(np.diff(arr) >= 0):
                raise ValueError(
                    f"GlobalPanelCalibration: {name} must be monotonically "
                    f"non-decreasing for np.interp to work. Got "
                    f"{len(arr)} knots with first violation at index "
                    f"{int(np.argmax(np.diff(arr) < 0))}."
                )

    def calibrate_probability(self, raw_score: float) -> float:
        """Map a raw panel score → P(outperform SPY by threshold in lookahead_days)."""
        # Audit #79: empty knot arrays would IndexError on prob_y[0]; degrade
        # to base-rate (0.5) instead of crashing.
        if len(self.prob_x) == 0 or len(self.prob_y) == 0:
            return 0.5
        return float(np.interp(raw_score, self.prob_x, self.prob_y,
                               left=self.prob_y[0], right=self.prob_y[-1]))

    def expected_return(self, raw_score: float) -> float:
        """Map a raw panel score → E[R_i - R_spy] over lookahead_days."""
        if len(self.er_x) == 0 or len(self.er_y) == 0:
            return 0.0
        return float(np.interp(raw_score, self.er_x, self.er_y,
                               left=self.er_y[0], right=self.er_y[-1]))

    # Vectorized helpers
    def calibrate_probability_vec(self, raws: np.ndarray) -> np.ndarray:
        if len(self.prob_x) == 0 or len(self.prob_y) == 0:
            return np.full(np.shape(raws), 0.5, dtype=float)
        return np.interp(raws, self.prob_x, self.prob_y,
                         left=self.prob_y[0], right=self.prob_y[-1])

    def expected_return_vec(self, raws: np.ndarray) -> np.ndarray:
        if len(self.er_x) == 0 or len(self.er_y) == 0:
            return np.zeros(np.shape(raws), dtype=float)
        return np.interp(raws, self.er_x, self.er_y,
                         left=self.er_y[0], right=self.er_y[-1])

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "kind": "global_panel_calibration",
            "trained_date": str(date.today()),
            "probability": {
                "x": self.prob_x.tolist(),
                "y": self.prob_y.tolist(),
            },
            "expected_return": {
                "x": self.er_x.tolist(),
                "y": self.er_y.tolist(),
            },
            "metadata": {**self.metadata, **(metadata or {})},
        }
        p.write_text(json.dumps(payload, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "GlobalPanelCalibration":
        payload = json.loads(Path(path).read_text())
        if payload.get("kind") != "global_panel_calibration":
            raise ValueError(
                f"Not a global_panel_calibration artifact: {path}",
            )
        return cls(
            prob_x = np.asarray(payload["probability"]["x"], dtype=float),
            prob_y = np.asarray(payload["probability"]["y"], dtype=float),
            er_x   = np.asarray(payload["expected_return"]["x"], dtype=float),
            er_y   = np.asarray(payload["expected_return"]["y"], dtype=float),
            metadata = payload.get("metadata", {}),
        )


def fit_global_calibrator(
    panel_scores: dict[str, pd.Series],   # {ticker: series indexed by date → raw panel score}
    future_returns: dict[str, pd.Series], # {ticker: series → future relative-to-SPY return}
    *,
    lookahead_days: int = 10,
    threshold: float = 0.03,
    threshold_mode: str = "absolute",
    min_rows: int = 1000,
    rolling_window_years: float | None = None,
) -> GlobalPanelCalibration:
    """Pool all tickers' (panel_score, future_return) pairs; fit one isotonic.

    Returns a `GlobalPanelCalibration` with both heads.
    Raises ValueError if pooled samples < min_rows (not enough data).

    threshold_mode:
      "absolute"       — classic: outperform if fwd_return >= threshold (default).
                         Works for 10d on typical panels. Collapses to 1 class on
                         long horizons (60d+) in bull markets where almost every
                         stock beats threshold.
      "crosssectional" — per-date: outperform if fwd_return >= that date's median
                         across tickers. Guaranteed ~50% base rate regardless of
                         horizon or market regime. Correct for cross-sectional
                         ranking models where the label is inherently relative.
                         `threshold` parameter is ignored in this mode.
    """
    # 2026-05-04 P0 — rolling window: when ``rolling_window_years`` is set,
    # restrict the calibrator pool to the trailing N years. Pre-fix the
    # calibrator fit on the FULL 5+ year history → 5-year-old panels'
    # feature distributions differ enough that the isotonic map between
    # raw_score and P(outperform) reflects an OBSOLETE regime. Default
    # None preserves prior behaviour. Recommended: 2.0 (matches model
    # training window).
    cutoff_ts: pd.Timestamp | None = None
    if rolling_window_years is not None and rolling_window_years > 0:
        # Find the latest date present across all tickers, count back N years.
        latest = pd.Timestamp.min
        for raw in panel_scores.values():
            if not raw.empty:
                m = pd.Timestamp(raw.index.max())
                if m > latest:
                    latest = m
        if latest is not pd.Timestamp.min:
            cutoff_ts = latest - pd.Timedelta(days=int(rolling_window_years * 365.25))

    rows_raw: list[float] = []
    rows_fwd: list[float] = []
    # Audit fix CALIB-PER-DATE-IC (2026-04-26): compute per-date cross-
    # sectional IC alongside the pooled IC. The pooled IC is dominated
    # by time-varying market effects (regime, beta) and looks ~50× smaller
    # than the panel scorer's CPCV `oos_mean_ic` — which IS per-date.
    # Storing both lets metadata accurately surface signal quality.
    per_date_rhos: list[float] = []
    # Index-keyed pool so we can do per-date groupby on the assembled frame.
    rows_keys: list[tuple] = []
    for t, raw in panel_scores.items():
        fwd = future_returns.get(t)
        if fwd is None or raw.empty or fwd.empty:
            continue
        idx = raw.index.intersection(fwd.index)
        if len(idx) == 0:
            continue
        # Apply rolling window slice — pool only includes dates ≥ cutoff.
        if cutoff_ts is not None:
            idx = idx[idx >= cutoff_ts]
            if len(idx) == 0:
                continue
        r = raw.loc[idx].astype(float).values
        f = fwd.loc[idx].astype(float).values
        ok = np.isfinite(r) & np.isfinite(f)
        rows_raw.append(r[ok])
        rows_fwd.append(f[ok])
        # Persist (date, ticker) keys for the per-date rollup.
        for d in idx[ok]:
            rows_keys.append((d, t))

    if not rows_raw:
        raise ValueError("fit_global_calibrator: no overlapping rows across tickers")
    raw_all = np.concatenate(rows_raw)
    fwd_all = np.concatenate(rows_fwd)
    if len(raw_all) < min_rows:
        raise ValueError(
            f"fit_global_calibrator: pooled n={len(raw_all)} < min_rows={min_rows}",
        )

    # 2026-05-03 P0 fix — NaN-leaf collapse filter:
    # XGBoost's missing-value handling routes ALL rows whose features are
    # NaN to the SAME terminal leaf, producing identical raw scores.
    # Panels covering long history (cold-start period when intraday/minute
    # features didn't yet exist) routinely have ~50% of rows hitting this
    # NaN-leaf, polluting the calibrator pool: spearmanr returns NaN with
    # a "ConstantInputWarning", and the probability head collapses to one
    # unique y → calibrator fit fails the round-7 ≥5-unique-y floor.
    # Discovered when arm A (drop 8 weak technicals, 2026-05-03 ablation
    # winner) failed fit despite per-fold CPCV IC = +0.0512 (consistently
    # positive across all 15 folds — model has signal, calibrator infra
    # was tripping on cold-start NaN paths).
    #
    # Detection: any raw value occurring in > 1% of pooled rows is
    # treated as a NaN-leaf collapse marker (genuine score collisions are
    # near-impossible for XGB rank:pairwise float output). Rows matching
    # are dropped from the pool before any further calibration math.
    # ``rows_keys`` is filtered in lockstep so per-date diagnostics still align.
    from collections import Counter as _Counter  # noqa: PLC0415
    if len(raw_all) > 0:
        bucket = np.round(raw_all, 8)
        counts = _Counter(bucket.tolist())
        if counts:
            mode_val, mode_count = counts.most_common(1)[0]
            mode_pct = mode_count / len(raw_all)
            if mode_pct > 0.01:
                keep_mask = ~np.isclose(raw_all, mode_val, atol=1e-9)
                n_dropped = int((~keep_mask).sum())
                log.warning(
                    "fit_global_calibrator: dropping %d/%d rows (%.1f%%) where "
                    "raw_score == %.6f (NaN-leaf collapse — XGB routed missing-"
                    "feature rows to the same terminal node). Pool reduced "
                    "from %d → %d.",
                    n_dropped, len(raw_all), mode_pct * 100, float(mode_val),
                    len(raw_all), int(keep_mask.sum()),
                )
                raw_all = raw_all[keep_mask]
                fwd_all = fwd_all[keep_mask]
                if len(rows_keys) == len(keep_mask):
                    rows_keys = [k for k, ok in zip(rows_keys, keep_mask) if ok]
                if len(raw_all) < min_rows:
                    raise ValueError(
                        f"fit_global_calibrator: after NaN-leaf filter, "
                        f"pooled n={len(raw_all)} < min_rows={min_rows}. "
                        f"Pool was dominated by NaN-leaf collapse; "
                        f"increase scorer's covered (ticker, date) coverage "
                        f"OR retrain with cleaner panel.",
                    )

    # Sanity diagnostic 1: panel IC on the full pool (mixes time × cross-section)
    rho, _ = spearmanr(raw_all, fwd_all)

    # Sanity diagnostic 2 (CALIB-PER-DATE-IC): mean per-date cross-sectional
    # IC, matching the panel scorer's CPCV methodology. This is the
    # apples-to-apples comparison vs `scorer_oos_mean_ic` and gives a
    # truthful picture of calibrator-input signal quality.
    per_date_ic_mean: float | None = None
    n_dates_eval = 0
    if len(rows_keys) == len(raw_all):
        df_pool = pd.DataFrame({
            "date":  [k[0] for k in rows_keys],
            "raw":   raw_all,
            "fwd":   fwd_all,
        })
        for d, grp in df_pool.groupby("date", sort=False):
            if len(grp) < 5:
                continue
            rh, _ = spearmanr(grp["raw"].values, grp["fwd"].values)
            if rh == rh:    # not NaN
                per_date_rhos.append(float(rh))
        if per_date_rhos:
            per_date_ic_mean = float(np.mean(per_date_rhos))
            n_dates_eval = len(per_date_rhos)

    # Probability head: indicator of outperforming.
    # "absolute" mode: outperform if fwd_return >= threshold (default, 10d).
    # "crosssectional" mode: outperform if fwd_return >= that date's median.
    #   Guaranteed ~50% base rate regardless of horizon/regime — correct for
    #   cross-sectional ranking where the label is inherently relative to peers.
    #   Fixes 60d calibrator collapse: in a bull market, nearly all 60d returns
    #   exceed 0.03, collapsing prob_labels to all-1 → < 5 unique y → ValueError.
    threshold_mode = threshold_mode.lower()
    if threshold_mode == "crosssectional":
        if len(rows_keys) == len(raw_all):
            df_prob = pd.DataFrame({
                "date": [k[0] for k in rows_keys],
                "fwd":  fwd_all,
            })
            per_date_median = df_prob.groupby("date")["fwd"].transform("median")
            prob_labels = (fwd_all >= per_date_median.values).astype(float)
        else:
            log.warning(
                "fit_global_calibrator: threshold_mode=crosssectional requires "
                "rows_keys to be parallel to raw_all/fwd_all; falling back to absolute."
            )
            prob_labels = (fwd_all >= threshold).astype(float)
    else:
        prob_labels = (fwd_all >= threshold).astype(float)
    iso_p = IsotonicRegression(out_of_bounds="clip").fit(raw_all, prob_labels)

    # ER head: direct regression
    iso_er = IsotonicRegression(out_of_bounds="clip").fit(raw_all, fwd_all)

    # Extract knots for JSON serialization. Use the isotonic model's own knots.
    prob_x = np.asarray(iso_p.X_thresholds_, dtype=float)
    prob_y = np.asarray(iso_p.y_thresholds_, dtype=float)
    er_x   = np.asarray(iso_er.X_thresholds_, dtype=float)
    er_y   = np.asarray(iso_er.y_thresholds_, dtype=float)

    # Audit fix CALIB-COLLAPSE-GUARD (2026-04-26 round-7): refuse to
    # ship a calibrator where the probability head has < 5 unique y
    # values.
    #
    # History: original guard threshold was 3. Round-7 audit found the
    # production XGBoost calibrator running with n_unique_prob_y = 6 →
    # top-7 candidates collapsed to identical rank_score=0.34474, breaking
    # rotation tiebreaks. User spec: "verify ≥5 unique y values". Bumped
    # to 5 so the LightGBM constant-y=1 case AND the borderline 3- or
    # 4-unique-y cases are all rejected at fit time.
    #
    # Calibrators with too few unique y values produce constant
    # rank_scores across cross-section → silently break rotation
    # tiebreaks + score_distribution percentile lookup.
    #
    # Reference: Niculescu-Mizil & Caruana (2005). "Predicting Good
    # Probabilities with Supervised Learning", ICML — isotonic
    # calibration requires sufficient resolution in the input signal.
    n_unique_prob_y = int(len(set(np.round(prob_y, 8))))
    if n_unique_prob_y < 5:
        raise ValueError(
            f"fit_global_calibrator: probability head collapsed to "
            f"{n_unique_prob_y} unique y values (need ≥5 — round-7 floor). "
            f"pool_ic={rho:+.4f} per_date_ic="
            f"{per_date_ic_mean if per_date_ic_mean is not None else 'n/a'}. "
            f"This usually means the scorer's signal is below the noise "
            f"floor for the calibrator pool — fix scorer or threshold first.",
        )

    metadata = {
        "n_rows":             int(len(raw_all)),
        "n_tickers":          int(len(rows_raw)),
        "pool_ic":            float(rho) if rho == rho else None,
        "per_date_ic_mean":   per_date_ic_mean,
        "n_dates_eval":       int(n_dates_eval),
        "n_unique_prob_y":    n_unique_prob_y,
        "threshold":          float(threshold),
        "threshold_mode":     threshold_mode,
        "lookahead_days":     int(lookahead_days),
        "prob_base_rate":     float(prob_labels.mean()),
        "er_mean":            float(fwd_all.mean()),
        "er_std":             float(fwd_all.std()),
    }
    log.info(
        "fit_global_calibrator: n=%d tickers=%d pool_ic=%+.4f "
        "per_date_ic=%s base_rate=%.3f er_mean=%+.4f n_unique_y=%d",
        metadata["n_rows"], metadata["n_tickers"], rho or 0.0,
        f"{per_date_ic_mean:+.4f}" if per_date_ic_mean is not None else "n/a",
        metadata["prob_base_rate"], metadata["er_mean"], n_unique_prob_y,
    )
    return GlobalPanelCalibration(
        prob_x=prob_x, prob_y=prob_y, er_x=er_x, er_y=er_y, metadata=metadata,
    )


def fit_regime_conditional(
    panel_scores: dict[str, pd.Series],
    future_returns: dict[str, pd.Series],
    regime_series: pd.Series,           # indexed by date → regime label
    *,
    lookahead_days: int = 10,
    threshold: float = 0.03,
    min_rows_per_regime: int = 300,
    regimes: list[str] | None = None,
) -> dict[str, GlobalPanelCalibration]:
    """Fit one `GlobalPanelCalibration` per regime label present in
    `regime_series`. Rows where regime ∉ `regimes` are dropped.

    A regime with fewer than `min_rows_per_regime` pooled samples is
    skipped (callers fall back to the pooled calibrator). This is a
    stricter floor than `fit_global_calibrator.min_rows` — per-regime
    isotonic needs more points to generalize because each regime sees
    less data.

    Returns `{regime: GlobalPanelCalibration}`. Metadata includes
    `regime` + `n_rows` so downstream code can diagnose coverage.
    """
    if regimes is None:
        regimes = ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"]

    reg_idx = regime_series.sort_index()

    out: dict[str, GlobalPanelCalibration] = {}
    for regime in regimes:
        # Masked scores + returns per ticker: keep only rows whose date's
        # regime label matches `regime`.
        masked_scores:  dict[str, pd.Series] = {}
        masked_returns: dict[str, pd.Series] = {}
        for t, raw in panel_scores.items():
            fwd = future_returns.get(t)
            if fwd is None or raw.empty or fwd.empty:
                continue
            # Align regime label to each score row
            reg_on = reg_idx.reindex(raw.index, method="ffill")
            keep   = reg_on == regime
            if not bool(keep.any()):
                continue
            masked_scores[t]  = raw.loc[keep]
            masked_returns[t] = fwd.reindex(raw.index).loc[keep]

        if not masked_scores:
            log.warning("fit_regime_conditional: no data for regime=%s", regime)
            continue
        try:
            cal = fit_global_calibrator(
                masked_scores, masked_returns,
                lookahead_days=lookahead_days,
                threshold=threshold,
                min_rows=min_rows_per_regime,
            )
        except ValueError as exc:
            log.warning(
                "fit_regime_conditional: regime=%s skipped — %s",
                regime, exc,
            )
            continue
        # Stamp regime into metadata so loaders can verify
        cal.metadata["regime"] = regime
        out[regime] = cal
        log.info("fit_regime_conditional: regime=%s n=%d IC=%+.4f",
                 regime, cal.metadata["n_rows"],
                 cal.metadata.get("pool_ic") or 0.0)

    return out


__all__ = [
    "GlobalPanelCalibration",
    "fit_global_calibrator",
    "fit_regime_conditional",
]
