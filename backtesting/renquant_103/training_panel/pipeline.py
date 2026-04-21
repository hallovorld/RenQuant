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
from .pp_panel_training import PanelTrainingPipeline


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

    Internally builds a PanelTrainingContext with the inputs already
    prepared by the caller (feature frames already built, OHLCV already
    fetched) and runs the full PanelTrainingPipeline starting at
    PanelSectorMomentumJob (DataFetchJob skipped because ohlcv is populated).

    Because the caller already built `feature_frames` (pre-neutralize),
    we also skip TickerPanelFeatureJob by seeding the per-ticker context
    outputs directly — see _SeedFeatures below.
    """
    ctx_ohlcv = dict(ohlcv)
    benchmark = config.get("benchmark", "SPY")
    ctx_ohlcv[benchmark] = spy_ohlcv

    panel_cfg = dict(config.get("panel_ltr", {}))
    # Surface legacy top-level knobs under panel_ltr.* for the new jobs
    for k in ("lookahead_days", "beta_window", "min_history_days",
              "age_warmup_days", "cv_n_splits", "cv_embargo_days",
              "num_boost_round", "neutralize_features", "nan_prone_cols",
              "xgb_params", "training_notes"):
        if k in config and k not in panel_cfg:
            panel_cfg[k] = config[k]
    merged_config = dict(config)
    merged_config["panel_ltr"] = panel_cfg
    if "artifact_path" not in panel_cfg:
        panel_cfg["artifact_path"] = str(out_path)

    ctx = PanelTrainingContext(
        config=merged_config,
        watchlist=list(watchlist),
        ohlcv=ctx_ohlcv,
        sector_etf_ohlcv=dict(sector_etf_ohlcv),
        ticker_sectors=dict(ticker_sectors),
        listing_dates=listing_dates,
    )
    # Seed pre-built per-ticker features so TickerPanelFeatureJob is a no-op
    ctx.feature_frames = dict(feature_frames)

    # Force the exact out_path the caller asked for — no artifacts/ redirect
    _requested_out = Path(out_path)
    panel_cfg["artifact_path"] = str(_requested_out)

    # Run the new orchestrator but skip data fetch / seeded feature pass
    from .pp_panel_training import (
        PanelSectorMomentumJob, PanelFeatureJob, PanelFactorZScoreJob,
        PanelLabelsJob, PanelAssemblyJob, PanelCVJob, PanelFitExportJob,
        TickerPanelContext, TickerPanelNeutralizeJob, TickerPanelFactorJob,
        run_panel_ticker_parallel, _run_panel_ticker_chain,
    )

    PanelSectorMomentumJob().run(ctx)

    # Replicate PanelFeatureJob but skip the Feature step (frames already built)
    ticker_ctxs = [
        TickerPanelContext(
            ticker=t, ohlcv=ctx.ohlcv, sector_momentum=ctx.sector_momentum,
            ticker_sectors=ctx.ticker_sectors, config=ctx.config,
        )
        for t in ctx.watchlist if t in ctx.feature_frames
    ]
    for tc in ticker_ctxs:
        tc.feature_frame = ctx.feature_frames[tc.ticker]

    # Parallel Neutralize + Factor
    def _chain(tc: TickerPanelContext):
        TickerPanelNeutralizeJob().run(tc)
        TickerPanelFactorJob().run(tc)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    n_workers = min(n_workers, max(1, len(ticker_ctxs)))
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

    PanelFactorZScoreJob().run(ctx)
    PanelLabelsJob().run(ctx)
    PanelAssemblyJob().run(ctx)
    PanelCVJob().run(ctx)
    PanelFitExportJob().run(ctx)

    return ctx.summary
