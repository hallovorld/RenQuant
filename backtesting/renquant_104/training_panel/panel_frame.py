"""Unified panel-frame construction for Stage-1 LTR training.

Takes per-ticker feature frames + per-ticker label series and produces a single
date-sorted DataFrame ready for `xgb.DMatrix(..., group=group_sizes)`.

Public API:
  build_panel_frame(...)           — assembly entry point
  compute_concurrency_weight(...)  — AFML ch.4 sample weight
  compute_age_weight(...)          — young-ticker damping weight
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def compute_concurrency_weight(
    dates: Iterable, lookahead_days: int = 5,
) -> pd.Series:
    """Per-row weight = 1 / mean(concurrency over that row's active window).

    Active window for a row dated t = [t, t+lookahead-1] in the panel's sorted
    unique-date index. Concurrency at date u = count of rows across all
    tickers whose active window contains u.

    Returns a Series indexed 0..N-1 aligned with the input order.
    """
    d = pd.to_datetime(pd.Index(list(dates)))
    unique_dates = pd.to_datetime(sorted(set(d)))
    idx_map = {ts: i for i, ts in enumerate(unique_dates)}

    # starts[i] = # rows whose start date is unique_dates[i]
    starts = np.zeros(len(unique_dates), dtype=np.int64)
    for ts in d:
        starts[idx_map[ts]] += 1

    # live[i] = # labels active on date i  = sum of starts over [i-L+1, i]
    live = np.zeros(len(unique_dates), dtype=np.int64)
    running = 0
    for i in range(len(unique_dates)):
        running += starts[i]
        if i >= lookahead_days:
            running -= starts[i - lookahead_days]
        live[i] = running

    # weight[row] = 1 / mean(live[start_idx : start_idx + L])
    weights = np.zeros(len(d), dtype=float)
    for i, ts in enumerate(d):
        si = idx_map[ts]
        ei = min(si + lookahead_days, len(unique_dates))
        window = live[si:ei]
        m = window.mean() if window.size > 0 else 1.0
        weights[i] = 1.0 / m if m > 0 else 1.0

    return pd.Series(weights)


def compute_age_weight(
    dates: Iterable, tickers: Iterable,
    listing_dates: dict | None = None,
    warmup_days: int = 504,
) -> pd.Series:
    """min(1, days_since_listing / warmup_days) per row.

    If listing_dates is None (or a ticker is absent from it), returns 1.0 for
    those rows — treated as seasoned.
    """
    d = pd.to_datetime(pd.Index(list(dates)))
    t = pd.Index(list(tickers))
    out = np.ones(len(d), dtype=float)
    if not listing_dates:
        return pd.Series(out)
    for i in range(len(d)):
        listing = listing_dates.get(t[i])
        if listing is None:
            continue
        age = (d[i] - pd.Timestamp(listing)).days
        if age <= 0:
            out[i] = 0.0
        elif age >= warmup_days:
            out[i] = 1.0
        else:
            out[i] = age / warmup_days
    return pd.Series(out)


def build_panel_frame(
    feature_frames: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
    ticker_sectors: dict[str, str],
    factor_frames: dict[str, pd.DataFrame] | None = None,
    macro_frame: pd.DataFrame | None = None,
    listing_dates: dict[str, pd.Timestamp] | None = None,
    min_history_days: int = 252,
    lookahead_days: int = 5,
    age_warmup_days: int = 504,
    nan_prone_cols: list[str] | None = None,
    drop_cols_from_features: tuple[str, ...] = ("fwd_return", "label"),
    training_window_years: "float | None" = None,
    recency_weighting: "dict | None" = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Assemble the unified panel training frame.

    Phase 1C (2026-04-26): added optional `macro_frame` parameter.
    When provided (non-empty DataFrame indexed by date with macro
    feature columns), each macro column is broadcast to every panel
    row via date merge. See doc/components/macro-factor-frame-design.md.

    Returns:
        panel_df    — sorted by (date, ticker), columns include:
                      date, ticker, sector, <features>, <factors>,
                      <*_is_missing>, label, weight_concurrency, weight_age, weight
        group_sizes — np.int32 array of per-date row counts
        metadata    — { n_rows, n_tickers, n_dates, dates, feature_cols,
                        factor_cols, missing_cols, per_ticker }
    """
    rows: list[pd.DataFrame] = []
    feature_cols_set: set[str] = set()
    factor_cols_set: set[str] = set()

    for ticker, ff in feature_frames.items():
        if ticker not in labels:
            continue
        ff_clean = ff.drop(columns=[c for c in drop_cols_from_features if c in ff.columns],
                           errors="ignore")
        label_s = labels[ticker]

        common = ff_clean.index.intersection(label_s.index)
        if len(common) == 0:
            continue

        df = ff_clean.loc[common].copy()
        df["label"] = label_s.loc[common].values

        if len(df) <= min_history_days:
            continue
        df = df.iloc[min_history_days:].copy()

        feature_cols_set.update(c for c in ff_clean.columns)

        if factor_frames is not None and ticker in factor_frames:
            fac = factor_frames[ticker]
            sub = fac.reindex(df.index)
            for col in sub.columns:
                df[col] = sub[col].values
                factor_cols_set.add(col)

        df["ticker"] = ticker
        df["sector"] = ticker_sectors.get(ticker, "UNKNOWN")
        df["date"]   = df.index
        df = df.reset_index(drop=True)
        rows.append(df)

    if not rows:
        raise ValueError("build_panel_frame: no panel rows produced "
                         "(empty feature_frames or labels, or min_history_days too high)")

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    # Phase 1C (2026-04-26): broadcast macro frame onto panel by date.
    # macro_frame is date-indexed; columns are macro features (e.g.,
    # vix_level_z, hyg_chg_5d_z). Each panel row gets the value for
    # its date. Forward-fill within ticker handles weekend gaps in
    # macro data (NYSE-holiday + weekend mismatches with equity calendar).
    # Trailing NaN (warmup for rolling-z) → 0.0 (z-scored mean — same
    # convention as factor_frames handling above).
    macro_cols: list[str] = []
    if macro_frame is not None and not macro_frame.empty:
        # Ensure macro_frame index is datetime for clean merge
        if not isinstance(macro_frame.index, pd.DatetimeIndex):
            macro_frame = macro_frame.copy()
            macro_frame.index = pd.to_datetime(macro_frame.index)
        macro_cols = list(macro_frame.columns)
        # Sanity: safety harness §11.2 — macro col names must not
        # collide with existing panel columns (would cause merge ambiguity).
        existing = set(panel.columns)
        collisions = [c for c in macro_cols if c in existing]
        if collisions:
            import logging as _logging
            _logging.getLogger("panel_frame").warning(
                "build_panel_frame: macro frame columns collide with "
                "existing panel columns: %s — using suffix '_macro'",
                collisions,
            )
            rename_map = {c: f"{c}_macro" for c in collisions}
            macro_frame = macro_frame.rename(columns=rename_map)
            macro_cols = [rename_map.get(c, c) for c in macro_cols]
        # Merge on date — left join preserves all panel rows.
        panel = panel.merge(
            macro_frame, left_on="date", right_index=True, how="left",
        )
        # Forward-fill within ticker (weekend / holiday alignment).
        panel[macro_cols] = panel.groupby(
            "ticker", group_keys=False,
        )[macro_cols].ffill()
        # Trailing NaN (rolling-z warmup) → 0.0 (z-scored mean).
        panel[macro_cols] = panel[macro_cols].fillna(0.0)

    missing_cols: list[str] = []
    if nan_prone_cols:
        for col in nan_prone_cols:
            if col in panel.columns:
                ind = f"{col}_is_missing"
                panel[ind] = panel[col].isna().astype(np.int8)
                missing_cols.append(ind)

    # Round-5: restrict to last N years (default: keep full history).
    # User spec: 5 years. Applied AFTER concat so all tickers have enough
    # history for feature computation, but BEFORE weights are computed so
    # rolling sums reflect the truncated window.
    if training_window_years is not None and training_window_years > 0:
        cutoff = panel["date"].max() - pd.Timedelta(days=int(training_window_years * 365.25))
        n_before = len(panel)
        panel = panel[panel["date"] >= cutoff].reset_index(drop=True)
        # Safe to regroup — group_sizes recomputed below.

    panel["weight_concurrency"] = compute_concurrency_weight(
        panel["date"], lookahead_days=lookahead_days
    ).values
    panel["weight_age"] = compute_age_weight(
        panel["date"], panel["ticker"],
        listing_dates=listing_dates, warmup_days=age_warmup_days,
    ).values
    panel["weight"] = panel["weight_concurrency"] * panel["weight_age"]

    # Round-5: exponential recency weighting — user spec emphasizes
    # recent samples. Most-recent bar gets weight 1.0; older bars decay.
    # Composes multiplicatively with existing weights. Off by default
    # (kind: None); common config `{"kind": "exp_decay", "half_life_days": 252}`.
    if recency_weighting:
        kind = str(recency_weighting.get("kind", "none")).lower()
        if kind == "exp_decay":
            half_life = float(recency_weighting.get("half_life_days", 252))
            if half_life > 0:
                most_recent = panel["date"].max()
                age_days = (most_recent - panel["date"]).dt.days.astype(float)
                decay = np.power(0.5, age_days / half_life)
                panel["weight_recency"] = decay.values
                panel["weight"] = panel["weight"] * panel["weight_recency"]
        elif kind in ("none", ""):
            pass
        else:
            import logging as _logging
            _logging.getLogger("panel_frame").warning(
                "Unknown recency_weighting.kind=%r — skipping", kind,
            )

    group_sizes = panel.groupby("date", sort=True).size().values.astype(np.int32)

    per_ticker: dict = {}
    for t, sub in panel.groupby("ticker"):
        per_ticker[t] = {
            "first_bar": str(pd.Timestamp(sub["date"].min()).date()),
            "last_bar":  str(pd.Timestamp(sub["date"].max()).date()),
            "n_rows":    int(len(sub)),
        }

    metadata = {
        "n_rows":     int(len(panel)),
        "n_tickers":  int(panel["ticker"].nunique()),
        "n_dates":    int(panel["date"].nunique()),
        "dates":      [str(pd.Timestamp(panel["date"].min()).date()),
                       str(pd.Timestamp(panel["date"].max()).date())],
        "feature_cols": sorted(feature_cols_set),
        "factor_cols":  sorted(factor_cols_set),
        "macro_cols":   sorted(macro_cols),
        "missing_cols": missing_cols,
        "per_ticker":   per_ticker,
    }
    return panel, group_sizes, metadata
