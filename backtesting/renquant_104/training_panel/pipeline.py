"""Thin wrapper — delegates to `pp_panel_training.PanelTrainingPipeline`.

Kept as a backwards-compatible entrypoint so legacy callers
(`scripts/train_panel_model.py`, `tests/test_panel_pipeline_e2e.py`,
and earlier notebooks) keep working while the Job/Task refactor landed
in `pp_panel_training.py` becomes the single source of truth for the
Stage-1 orchestration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .context import PanelTrainingContext
from .pp_panel_training import (
    SectorMomentumTask,
    FactorZScoreTask,
    NeutralizedFeatureZScoreTask,
    LoadFundamentalsTask,
    LoadEarningsSurpriseTask,
    LoadInsiderTradesTask,
    LoadHourlyBarsTask,
    LoadMinuteBarsTask,
    PanelFeatureJob,
    PanelAssemblyJob,
    PanelModelJob,
    TickerPanelContext,
    TickerPanelFeatureJob,
    TickerPanelNeutralizeJob,
    TickerPanelFactorJob,
)


# ── Inference-side prep ──────────────────────────────────────────────────────

def prepare_inference_panel_frames(
    watchlist: list[str],
    ohlcv: dict[str, pd.DataFrame],
    ticker_sectors: dict[str, str],
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build neutralized feature frames + z-scored factor frames for live inference.

    Mirrors Phase 1 (SectorMomentum) + Phase 2 (per-ticker Feature+Neutralize+Factor)
    + FactorZScoreTask of PanelTrainingPipeline, but without building labels /
    panel frame / training.

    Returns (neutralized_frames, factor_frames_z) — attach to InferenceContext
    as `_panel_feature_frames` and `_panel_factor_frames` before running
    PanelScoringJob.

    `ohlcv` must already contain the benchmark (SPY) and every sector ETF
    referenced by `sector_etf_map` in config.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    sector_etf_map = config.get("sector_etf_map", {})
    sector_etf_ohlcv = {
        sec: ohlcv[etf] for sec, etf in sector_etf_map.items() if etf in ohlcv
    }

    ctx = PanelTrainingContext(
        config=config,
        watchlist=list(watchlist),
        ohlcv=dict(ohlcv),
        sector_etf_ohlcv=sector_etf_ohlcv,
        ticker_sectors=dict(ticker_sectors),
        listing_dates=None,
        # Bug 16 fix: inference path must NEVER auto-fetch — read cache only.
        # Training (FullTrainingPipeline) leaves this False so missing
        # tickers can be fetched fresh. Sim/live invokes this fn for
        # per-bar feature prep — auto-fetch would block the loop.
        inference_only=True,
    )

    SectorMomentumTask().run(ctx)
    LoadFundamentalsTask().run(ctx)
    LoadEarningsSurpriseTask().run(ctx)
    LoadInsiderTradesTask().run(ctx)
    LoadHourlyBarsTask().run(ctx)
    # Bug 12 fix (2026-04-24): inference path was missing LoadMinuteBars,
    # so train has m_* features but inference never populates them →
    # NaN cols at inference, model predictions wrong on the 10-min half
    # of the feature space. Added now to keep train ⇌ inference parity.
    LoadMinuteBarsTask().run(ctx)

    ticker_ctxs = [
        TickerPanelContext(
            ticker=t, ohlcv=ctx.ohlcv, sector_momentum=ctx.sector_momentum,
            ticker_sectors=ctx.ticker_sectors, config=ctx.config,
            fundamentals=ctx.fundamentals,
            earnings_surprises=ctx.earnings_surprises,
            insider_trades=ctx.insider_trades,
            hourly_bars=ctx.hourly_bars,
            minute_bars=ctx.minute_bars,
        )
        for t in ctx.watchlist if t in ctx.ohlcv
    ]

    def _chain(tc: TickerPanelContext):
        TickerPanelFeatureJob().run(tc)
        if tc.feature_frame is None or tc.feature_frame.empty:
            return
        TickerPanelNeutralizeJob().run(tc)
        TickerPanelFactorJob().run(tc)

    # Audit P-9: error isolation. Previously `f.result()` re-raised on
    # the first failed ticker, killing the entire panel inference for
    # the bar. Now we log + continue — that ticker silently drops from
    # neutralized/raw factor frames (cross_sectional_zscore handles the
    # missing ticker). Mirror's training-side run_panel_ticker_parallel.
    import logging as _logging
    _log_inf = _logging.getLogger("training_panel.pipeline")
    n_workers = min(max(1, (os.cpu_count() or 4) - 2), max(1, len(ticker_ctxs)))
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="panel-inf") as ex:
        futs = {ex.submit(_chain, tc): tc.ticker for tc in ticker_ctxs}
        for f in as_completed(futs):
            ticker = futs[f]
            try:
                f.result()
            except Exception as exc:
                _log_inf.error(
                    "prepare_inference_panel_frames[%s]: chain ERROR — %s: %s "
                    "(ticker dropped from this bar's panel matrix)",
                    ticker, type(exc).__name__, exc,
                )

    ctx.neutralized_frames = {
        tc.ticker: tc.neutralized_frame for tc in ticker_ctxs
        if tc.neutralized_frame is not None
    }
    ctx.raw_factor_frames = {
        tc.ticker: tc.raw_factor_frame for tc in ticker_ctxs
        if tc.raw_factor_frame is not None
    }

    # Cross-sectional z-score per-ticker indicators so inference distribution
    # matches training. Must run BEFORE FactorZScoreTask so order matches
    # PanelAssemblyJob in the training pipeline.
    NeutralizedFeatureZScoreTask().run(ctx)
    FactorZScoreTask().run(ctx)

    return ctx.neutralized_frames, ctx.factor_frames


def train_panel_model(
    watchlist: list[str],
    feature_frames: dict[str, pd.DataFrame],
    ohlcv: dict[str, pd.DataFrame],
    spy_ohlcv: pd.DataFrame,
    sector_etf_ohlcv: dict[str, pd.DataFrame],
    ticker_sectors: dict[str, str],
    listing_dates: dict[str, pd.Timestamp] | None,
    config: dict[str, Any],
    out_path: Path | str,
) -> dict:
    """Legacy signature — preserved for existing callers/tests.

    Inputs are already-prepared (feature frames built, OHLCV fetched) so
    we skip the fetch task and seed the per-ticker outputs, then run the
    remaining Jobs through the pipeline.
    """
    ctx_ohlcv = dict(ohlcv)
    benchmark = config.get("benchmark", "SPY")
    ctx_ohlcv[benchmark] = spy_ohlcv

    panel_cfg = dict(config.get("panel_ltr", {}))
    # Surface legacy top-level knobs under panel_ltr.* for the new tasks
    for k in ("lookahead_days", "beta_window", "min_history_days",
              "age_warmup_days", "cv_n_splits", "cv_embargo_days",
              "num_boost_round", "neutralize_features", "nan_prone_cols",
              "xgb_params", "training_notes"):
        if k in config and k not in panel_cfg:
            panel_cfg[k] = config[k]
    panel_cfg["artifact_path"] = str(Path(out_path))
    merged_config = dict(config)
    merged_config["panel_ltr"] = panel_cfg

    ctx = PanelTrainingContext(
        config=merged_config,
        watchlist=list(watchlist),
        ohlcv=ctx_ohlcv,
        sector_etf_ohlcv=dict(sector_etf_ohlcv),
        ticker_sectors=dict(ticker_sectors),
        listing_dates=listing_dates,
    )
    ctx.feature_frames = dict(feature_frames)

    # Phase 1: OHLCV already loaded → run sector momentum only
    SectorMomentumTask().run(ctx)

    # Phase 2: skip per-ticker Feature step (frames already built) but run
    # Neutralize + Factor in parallel to stay aligned with production concurrency.
    ticker_ctxs = [
        TickerPanelContext(
            ticker=t, ohlcv=ctx.ohlcv, sector_momentum=ctx.sector_momentum,
            ticker_sectors=ctx.ticker_sectors, config=ctx.config,
        )
        for t in ctx.watchlist if t in ctx.feature_frames
    ]
    for tc in ticker_ctxs:
        tc.feature_frame = ctx.feature_frames[tc.ticker]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    n_workers = min(n_workers, max(1, len(ticker_ctxs)))

    def _chain(tc: TickerPanelContext):
        TickerPanelNeutralizeJob().run(tc)
        TickerPanelFactorJob().run(tc)

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="panel-wrap") as ex:
        futs = [ex.submit(_chain, tc) for tc in ticker_ctxs]
        for f in as_completed(futs):
            f.result()

    ctx.neutralized_frames = {
        tc.ticker: tc.neutralized_frame for tc in ticker_ctxs
        if tc.neutralized_frame is not None
    }
    ctx.raw_factor_frames = {
        tc.ticker: tc.raw_factor_frame for tc in ticker_ctxs
        if tc.raw_factor_frame is not None
    }

    # Phase 3 + 4: assembly + model via the Job chain
    PanelAssemblyJob().run(ctx)
    PanelModelJob().run(ctx)

    return ctx.summary
