#!/usr/bin/env python
"""Fit the global panel calibrator against the current Panel-LTR artifact.

Pipeline:
  1. Load `artifacts/panel-ltr.json` via PanelScorer
  2. Rebuild the panel feature+factor frames over history
  3. Score every (ticker, date) → raw panel score
  4. Compute forward relative-to-SPY return per (ticker, date)
  5. Pool and fit isotonic for P(outperform) + E[R_i - R_spy]
  6. Save to `artifacts/panel-rank-calibration.json`

Usage::

    python scripts/fit_panel_calibrator.py
    python scripts/fit_panel_calibrator.py --strategy renquant_104 --threshold 0.03
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fit-panel-calibrator")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--threshold", type=float, default=None,
                   help="Outperform threshold for the probability head "
                        "(defaults to config.panel_ltr.threshold or 0.03).")
    p.add_argument("--lookahead", type=int, default=None,
                   help="Lookahead days (defaults to config.panel_ltr.lookahead_days).")
    p.add_argument("--out", type=str, default=None,
                   help="Output artifact path (defaults to "
                        "artifacts/panel-rank-calibration.json).")
    p.add_argument("--regime-conditional", action="store_true",
                   help="Also fit one calibrator per regime label and save "
                        "to artifacts/panel-calibration-{REGIME}.json. "
                        "Sources the regime series from data/runs.db "
                        "(pipeline_runs table).")
    p.add_argument("--regime-db", type=str, default="data/runs.db",
                   help="SQLite DB with pipeline_runs.run_date + .regime "
                        "(default data/runs.db).")
    p.add_argument("--min-regime-rows", type=int, default=300,
                   help="Per-regime minimum pooled sample count "
                        "(default 300). Regimes below floor are skipped "
                        "— runtime falls back to the pooled calibrator.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))
    config = json.loads((strategy_dir / "strategy_config.json").read_text())
    panel_cfg = config.get("panel_ltr", {})
    lookahead = args.lookahead or int(panel_cfg.get("lookahead_days", 10))
    threshold = args.threshold if args.threshold is not None \
        else float(config.get("model_params", {}).get("threshold", 0.03))

    out_path = Path(args.out) if args.out else (
        strategy_dir / "artifacts" / "panel-rank-calibration.json"
    )

    # ── Load OHLCV + build neutralized frames (mirror notebook cell 21) ─────
    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from training_panel.context import PanelTrainingContext, TickerPanelContext  # noqa: PLC0415
    from training_panel.pp_panel_training import (                                # noqa: PLC0415
        SectorMomentumTask, TickerPanelNeutralizeJob, TickerPanelFactorJob,
        LoadFundamentalsTask, NeutralizedFeatureZScoreTask,
    )
    from kernel.panel_pipeline.panel_scorer import PanelScorer                    # noqa: PLC0415
    from kernel.panel_pipeline.feature_matrix import build_inference_matrix       # noqa: PLC0415
    from training.features import build_all_training_features                     # noqa: PLC0415
    from training_panel.global_calibrator import fit_global_calibrator            # noqa: PLC0415

    benchmark    = config.get("benchmark", "SPY")
    sector_map   = config.get("sector_map", {})
    etf_map      = config.get("sector_etf_map", {})
    watchlist    = config["watchlist"]

    log.info("Fetching OHLCV for %d tickers + %d sector ETFs + benchmark",
             len(watchlist), len(set(etf_map.values())))
    needed = sorted(set(watchlist) | {benchmark} | set(etf_map.values()))
    ohlcv: dict[str, pd.DataFrame] = {}
    for sym in needed:
        try:
            df = fetch_ohlcv(sym)
        except Exception as exc:
            log.warning("  %s fetch failed: %s", sym, exc)
            continue
        if df is not None and not df.empty:
            ohlcv[sym] = df

    spy_df = ohlcv[benchmark]

    # Feature frames (per ticker)
    feat_in = {t: ohlcv[t] for t in watchlist if t in ohlcv}
    feat_in[benchmark] = spy_df
    mp = config.get("model_params", {})
    feature_frames = build_all_training_features(
        watchlist=list(feat_in.keys() - {benchmark}),
        ohlcv=feat_in,
        indicator_spec=config.get("indicator_spec", {}),
        lookahead=lookahead,
        threshold=threshold,
    )

    merged_cfg = dict(config)
    merged_cfg["_strategy_dir"] = str(strategy_dir)
    sector_etf_ohlcv = {s: ohlcv[e] for s, e in etf_map.items() if e in ohlcv}
    ticker_sectors   = {t: sector_map[t] for t in feature_frames if t in sector_map}

    pctx = PanelTrainingContext(
        config           = merged_cfg,
        watchlist        = list(feature_frames.keys()),
        ohlcv            = dict(ohlcv) | {benchmark: spy_df},
        sector_etf_ohlcv = sector_etf_ohlcv,
        ticker_sectors   = ticker_sectors,
    )
    pctx.feature_frames = feature_frames
    SectorMomentumTask().run(pctx)
    LoadFundamentalsTask().run(pctx)

    # Build neutralized + raw factor frames per ticker
    ticker_ctxs = []
    for t in pctx.watchlist:
        if t not in pctx.feature_frames:
            continue
        tc = TickerPanelContext(
            ticker=t, ohlcv=pctx.ohlcv,
            sector_momentum=pctx.sector_momentum,
            ticker_sectors=pctx.ticker_sectors,
            config=pctx.config,
            fundamentals=pctx.fundamentals,
        )
        tc.feature_frame = pctx.feature_frames[t]
        TickerPanelNeutralizeJob().run(tc)
        TickerPanelFactorJob().run(tc)
        ticker_ctxs.append(tc)

    pctx.neutralized_frames = {tc.ticker: tc.neutralized_frame for tc in ticker_ctxs if tc.neutralized_frame is not None}
    pctx.raw_factor_frames  = {tc.ticker: tc.raw_factor_frame  for tc in ticker_ctxs if tc.raw_factor_frame  is not None}

    # Cross-sectional z-score per-ticker indicators so inference distribution
    # matches training. Must run before scoring.
    NeutralizedFeatureZScoreTask().run(pctx)

    ff  = pctx.neutralized_frames
    fac = pctx.raw_factor_frames

    # ── Load scorer ─────────────────────────────────────────────────────────
    scorer_path = strategy_dir / panel_cfg.get("artifact_path", "artifacts/panel-ltr.json")
    log.info("Loading panel scorer: %s", scorer_path)
    scorer = PanelScorer.load(scorer_path)
    nan_cols = list(panel_cfg.get("nan_prone_cols", []))

    # ── Score every date ────────────────────────────────────────────────────
    # Date grid = union of all tickers' OHLCV indices, intersected with panel coverage
    all_dates = pd.DatetimeIndex([])
    for t, df in ff.items():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()
    log.info("Scoring %d dates for %d tickers", len(all_dates), len(ff))

    per_ticker_scores: dict[str, dict[pd.Timestamp, float]] = {t: {} for t in ff}
    for d in all_dates:
        X = build_inference_matrix(ff, fac, d,
                                   feature_cols=scorer.feature_cols,
                                   nan_prone_cols=nan_cols)
        if X.empty:
            continue
        s = scorer.score(X)
        for ticker, v in s.items():
            if pd.notna(v):
                per_ticker_scores[ticker][d] = float(v)

    panel_scores = {
        t: pd.Series(d, name=t).sort_index()
        for t, d in per_ticker_scores.items() if d
    }
    log.info("Scored %d tickers", len(panel_scores))

    # ── Compute forward relative-to-SPY return per (ticker, date) ───────────
    future_returns: dict[str, pd.Series] = {}
    spy_close = spy_df["close"].astype(float)
    for t, series in panel_scores.items():
        ohlcv_t = ohlcv.get(t)
        if ohlcv_t is None:
            continue
        idx = series.index
        p_stock = ohlcv_t["close"].reindex(idx).astype(float)
        p_spy   = spy_close.reindex(idx).replace(0, np.nan)
        rel     = (p_stock / p_spy).replace([np.inf, -np.inf], np.nan)
        fwd_rel = rel.shift(-lookahead) / rel - 1.0
        future_returns[t] = fwd_rel

    # ── Fit global calibrator ──────────────────────────────────────────────
    calib = fit_global_calibrator(
        panel_scores, future_returns,
        lookahead_days=lookahead, threshold=threshold,
    )
    calib.save(out_path, metadata={
        "scorer_artifact": str(scorer_path),
        "scorer_oos_mean_ic": scorer.metadata.get("oos_mean_ic"),
    })
    log.info("Saved → %s", out_path)
    log.info("Summary: n=%d tickers=%d pool_IC=%+.4f base_rate=%.3f",
             calib.metadata["n_rows"], calib.metadata["n_tickers"],
             calib.metadata["pool_ic"] or 0.0, calib.metadata["prob_base_rate"])

    # ── Plan F — per-regime calibrators ─────────────────────────────────────
    if args.regime_conditional:
        import sqlite3  # noqa: PLC0415
        from training_panel.global_calibrator import fit_regime_conditional  # noqa: PLC0415

        db_path = REPO_ROOT / args.regime_db if not Path(args.regime_db).is_absolute() \
            else Path(args.regime_db)
        if not db_path.exists():
            log.error("regime_conditional: DB not found at %s — skipping", db_path)
        else:
            conn = sqlite3.connect(str(db_path))
            try:
                reg_df = pd.read_sql(
                    "SELECT run_date, regime, run_type FROM pipeline_runs "
                    "WHERE strategy=? ORDER BY run_date, run_type",
                    conn, params=(args.strategy,), parse_dates=["run_date"],
                )
            finally:
                conn.close()

            # Prefer live > sim when both exist on the same date.
            reg_df["_ord"] = reg_df["run_type"].map({"live": 1}).fillna(0)
            reg_df = (reg_df.sort_values(["run_date", "_ord"])
                             .drop_duplicates(subset=["run_date"], keep="last"))
            reg_series = reg_df.set_index("run_date")["regime"]
            reg_series.index = pd.DatetimeIndex(reg_series.index).normalize()
            log.info("regime_conditional: regime series n=%d dates from %s → %s",
                     len(reg_series), reg_series.index.min().date(),
                     reg_series.index.max().date())

            # Align scores' date index to the regime-series normalisation.
            norm_scores = {
                t: s.copy() for t, s in panel_scores.items()
            }
            for t, s in norm_scores.items():
                s.index = pd.DatetimeIndex(s.index).normalize()
            norm_rets = {t: fwd.copy() for t, fwd in future_returns.items()}
            for t, s in norm_rets.items():
                s.index = pd.DatetimeIndex(s.index).normalize()

            per_regime = fit_regime_conditional(
                norm_scores, norm_rets, reg_series,
                lookahead_days=lookahead, threshold=threshold,
                min_rows_per_regime=args.min_regime_rows,
            )
            for regime, cal in per_regime.items():
                rc_path = strategy_dir / "artifacts" / f"panel-calibration-{regime}.json"
                cal.save(rc_path, metadata={
                    "scorer_artifact": str(scorer_path),
                    "scorer_oos_mean_ic": scorer.metadata.get("oos_mean_ic"),
                    "regime": regime,
                })
                log.info("regime=%s → %s  (n=%d IC=%+.4f base_rate=%.3f)",
                         regime, rc_path, cal.metadata["n_rows"],
                         cal.metadata.get("pool_ic") or 0.0,
                         cal.metadata["prob_base_rate"])
            log.info("regime_conditional: wrote %d calibrator(s)", len(per_regime))


if __name__ == "__main__":
    main()
