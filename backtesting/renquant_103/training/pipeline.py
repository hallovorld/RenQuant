"""TrainingPipeline — sequential job runner for renquant_103 model training.

Each Job reads/writes TrainingContext.  Jobs run in order; any job may call
should_skip() to short-circuit when its outputs are already populated.

Usage from the notebook::

    ctx = TrainingContext(config=CONFIG, ohlcv=ohlcv)   # pre-populate ohlcv to skip DataFetchJob
    TrainingPipeline().run(ctx)
    # ctx.final_regime, ctx.results, ctx.feature_frames, ... all populated

Or from automation scripts::

    ctx = TrainingContext(config=CONFIG)
    TrainingPipeline().run(ctx)                          # DataFetchJob fetches ohlcv
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Context ────────────────────────────────────────────────────────────────────

@dataclass
class TrainingContext:
    """All shared state that flows through training jobs."""
    config: dict[str, Any]

    # populated by DataFetchJob (or pre-filled by caller)
    ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)

    # populated by RegimeFitJob
    hurst_series: pd.Series = field(default=None)
    cusum_series: pd.Series = field(default=None)
    changepoint_dates: pd.Index = field(default=None)
    final_regime: pd.Series = field(default=None)
    final_regime_conf: pd.Series = field(default=None)
    gmm: Any = field(default=None)  # training.regime.RegimeGMM instance

    # populated by FeatureJob
    feature_frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    # populated by TournamentJob
    results: dict[str, Any] = field(default_factory=dict)

    # populated by ExportJob
    exported: list[str] = field(default_factory=list)

    # populated by CorrelationJob
    corr_matrix: pd.DataFrame = field(default=None)

    # populated by CalibrationJob
    calibration_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def spy_df(self) -> pd.DataFrame | None:
        return self.ohlcv.get("SPY")

    @property
    def watchlist(self) -> list[str]:
        return self.config["watchlist"]

    @property
    def strategy_dir(self) -> Path | None:
        _sd = self.config.get("_strategy_dir")
        return Path(_sd) if _sd else None


# ── Job ABC ────────────────────────────────────────────────────────────────────

class Job(ABC):
    """A single stage in the training pipeline."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def should_skip(self, ctx: TrainingContext) -> bool:
        return False

    @abstractmethod
    def run(self, ctx: TrainingContext) -> None: ...


# ── Jobs ───────────────────────────────────────────────────────────────────────

class DataFetchJob(Job):
    """Fetch OHLCV for all tickers (watchlist + sector ETFs + SPY).

    Skipped when ctx.ohlcv is already populated (e.g. pre-loaded in notebook).
    """

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.ohlcv:
            print("DataFetchJob: ohlcv already populated — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        from kernel.data import fetch_ohlcv

        cfg = ctx.config
        start = cfg["sample_start"]
        end   = cfg["sample_end"]
        sector_etf = cfg.get("sector_etf_map", {})
        all_tickers = sorted(
            set(ctx.watchlist) | set(sector_etf.values()) | {"SPY"}
        )
        print(f"DataFetchJob: fetching {len(all_tickers)} tickers {start} → {end}")
        for ticker in all_tickers:
            try:
                df = fetch_ohlcv(ticker, start=start, end=end)
                if df is not None and not df.empty:
                    ctx.ohlcv[ticker] = df
                    print(f"  {ticker}: {len(df)} rows")
                else:
                    print(f"  {ticker}: EMPTY")
            except Exception as exc:
                print(f"  {ticker}: ERROR — {exc}")
        print(f"DataFetchJob: loaded {len(ctx.ohlcv)} / {len(all_tickers)} tickers")


class RegimeFitJob(Job):
    """Compute Hurst, CUSUM, GMM, and combine into final daily regime series.

    Saves the GMM artifact to strategy_dir/artifacts/spy-gmm-regime.json.
    Populates: hurst_series, cusum_series, changepoint_dates,
               final_regime, final_regime_conf, gmm.
    """

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.final_regime is not None:
            print("RegimeFitJob: final_regime already populated — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        from training.regime import (
            build_gmm_features, RegimeGMM,
            rolling_hurst, rolling_cusum,
        )

        spy_df = ctx.spy_df
        if spy_df is None:
            raise RuntimeError("RegimeFitJob: SPY not in ohlcv — run DataFetchJob first")

        rcfg = ctx.config["regime"]

        # Layer 1: Hurst
        spy_returns = spy_df["close"].pct_change().dropna()
        ctx.hurst_series = rolling_hurst(spy_returns, window=rcfg["hurst_window"]).dropna()

        # Layer 2: CUSUM
        ctx.cusum_series = rolling_cusum(
            spy_returns,
            window    = rcfg["cusum_lookback"],
            threshold = rcfg["cusum_threshold"],
            drift     = rcfg["cusum_drift"],
        )
        ctx.changepoint_dates = ctx.cusum_series[ctx.cusum_series].index

        # Layer 3: GMM
        gmm_features = build_gmm_features(
            spy_df, vol_window=20, hurst_window=rcfg["hurst_window"]
        )
        gmm = RegimeGMM(n_components=3, random_state=42, n_init=10)
        gmm.fit(gmm_features)
        ctx.gmm = gmm

        regime_labels, regime_probs = gmm.predict(gmm_features)

        # Combine into final daily regime
        BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR = (
            "BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"
        )
        hurst_trend  = rcfg["hurst_trending_threshold"]
        hurst_rev    = rcfg["hurst_reversion_threshold"]
        vol_window   = rcfg["vol_realized_window"]
        bear_vol_thr = rcfg["bear_vol_threshold"]
        bear_ret_thr = rcfg["bear_return_threshold"]
        choppy_floor = rcfg.get("choppy_hurst_floor", 0.20)

        spy_20d_vol = spy_returns.rolling(vol_window).std() * np.sqrt(252)
        spy_20d_ret = spy_returns.rolling(vol_window).sum()

        common_idx = ctx.hurst_series.index.intersection(regime_labels.index)
        final_regime      = pd.Series(index=common_idx, dtype=str)
        final_regime_conf = pd.Series(index=common_idx, dtype=float)

        for dt in common_idx:
            h          = ctx.hurst_series.loc[dt]
            gmm_r      = regime_labels.loc[dt]
            gmm_bear_p = regime_probs.loc[dt].get(BEAR, 0.0)

            if h > hurst_trend:
                base = BULL_CALM
            elif h < hurst_rev:
                base = CHOPPY
            else:
                base = None

            vol_today = float(spy_20d_vol.loc[dt]) if dt in spy_20d_vol.index else 0.0
            ret_today = float(spy_20d_ret.loc[dt]) if dt in spy_20d_ret.index else 0.0
            hard_bear = vol_today > bear_vol_thr or ret_today < bear_ret_thr

            if hard_bear or gmm_bear_p > 0.5:
                final_regime.loc[dt] = BEAR
            elif base is None:
                final_regime.loc[dt] = gmm_r if gmm_r != BEAR else BULL_VOLATILE
            else:
                final_regime.loc[dt] = base

            r = final_regime.loc[dt]
            if r == CHOPPY:
                conf = (hurst_rev - h) / max(hurst_rev - choppy_floor, 1e-6)
                final_regime_conf.loc[dt] = float(min(1.0, max(0.0, conf)))
            else:
                final_regime_conf.loc[dt] = float(regime_probs.loc[dt].get(r, 0.5))

        ctx.final_regime      = final_regime
        ctx.final_regime_conf = final_regime_conf

        print(f"RegimeFitJob: CUSUM detected {len(ctx.changepoint_dates)} changepoints")
        print(f"RegimeFitJob: Hurst {ctx.hurst_series.min():.3f}–"
              f"{ctx.hurst_series.max():.3f} (mean={ctx.hurst_series.mean():.3f})")
        print("RegimeFitJob: Final regime distribution:")
        print(final_regime.value_counts().to_string())

        # Save GMM artifact
        if ctx.strategy_dir:
            artifacts_dir = ctx.strategy_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            gmm_path = artifacts_dir / "spy-gmm-regime.json"
            gmm.save(gmm_path)
            print(f"RegimeFitJob: GMM artifact saved → {gmm_path}")


class FeatureJob(Job):
    """Build labelled feature frames for every ticker in the watchlist."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.feature_frames:
            print("FeatureJob: feature_frames already populated — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        from training.features import build_all_training_features

        mp = ctx.config["model_params"]
        ctx.feature_frames = build_all_training_features(
            ctx.watchlist,
            ctx.ohlcv,
            ctx.config["indicator_spec"],
            mp["lookahead"],
            mp["threshold"],
        )
        print(f"FeatureJob: built feature frames for {len(ctx.feature_frames)} tickers")


class TournamentJob(Job):
    """Train and evaluate all model types; select best per ticker."""

    def should_skip(self, ctx: TrainingContext) -> bool:
        if ctx.results:
            print("TournamentJob: results already populated — skipping")
            return True
        return False

    def run(self, ctx: TrainingContext) -> None:
        from training.tournament import run_tournament_all

        ctx.results = run_tournament_all(
            ctx.watchlist, ctx.feature_frames, ctx.ohlcv, ctx.config
        )
        passed = sum(1 for r in ctx.results.values() if r.get("passes_floor"))
        print(f"TournamentJob: {passed}/{len(ctx.watchlist)} tickers passed Sharpe floor")


class ExportJob(Job):
    """Export trained models to strategy_dir/models/ and retrain on full history."""

    def run(self, ctx: TrainingContext) -> None:
        from datetime import date as _date
        from training.export import export_models, retrain_live_models

        if not ctx.strategy_dir:
            print("ExportJob: no strategy_dir set — skipping")
            return

        today = str(_date.today())
        mp = ctx.config["model_params"]
        ctx.exported, _ = export_models(
            ctx.results,
            ctx.strategy_dir,
            today,
            sharpe_floor=float(ctx.config.get("sharpe_floor", 0.8)),
            lookahead=mp["lookahead"],
            strategy_name=ctx.config.get("_strategy_name", "renquant_103"),
        )
        retrain_live_models(
            ctx.results, ctx.feature_frames, ctx.exported,
            ctx.strategy_dir, mp, ctx.config, today,
        )
        print(f"ExportJob: exported {len(ctx.exported)} models")


class CorrelationJob(Job):
    """Compute 120-day rolling return correlation and save artifact."""

    def run(self, ctx: TrainingContext) -> None:
        close_df = pd.DataFrame({
            t: ctx.ohlcv[t]["close"]
            for t in ctx.watchlist
            if t in ctx.ohlcv
        })
        ret_df = close_df.pct_change().dropna()
        ctx.corr_matrix = ret_df.tail(120).corr()

        if ctx.strategy_dir:
            corr_dict = {
                ticker: {
                    other: round(float(ctx.corr_matrix.loc[ticker, other]), 4)
                    for other in ctx.corr_matrix.columns
                }
                for ticker in ctx.corr_matrix.index
            }
            artifacts_dir = ctx.strategy_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            corr_path = artifacts_dir / "watchlist-correlation.json"
            corr_path.write_text(json.dumps(corr_dict, indent=2))
            print(f"CorrelationJob: saved → {corr_path}")


class CalibrationJob(Job):
    """Refresh score calibrations and blend weights for all exported models."""

    def run(self, ctx: TrainingContext) -> None:
        try:
            from training.scoring import fit_probability_calibration
            from kernel.scoring import ScoreCalibration, raw_score_kind_for_model
        except ImportError as exc:
            print(f"CalibrationJob: skipped — {exc}")
            return

        if not ctx.strategy_dir:
            print("CalibrationJob: no strategy_dir — skipping")
            return

        models_dir = ctx.strategy_dir / "models"
        mp = ctx.config["model_params"]
        summary: dict[str, Any] = {}

        for ticker in ctx.exported:
            res = ctx.results.get(ticker, {})
            if not res:
                continue
            feature_frame = ctx.feature_frames.get(ticker)
            if feature_frame is None or feature_frame.empty:
                continue

            best = res.get("best_approach")
            model_obj = res.get(best, {}).get("model") if best else None
            if model_obj is None:
                continue

            try:
                raw_scores = model_obj.predict_score_bulk(feature_frame)
                oos_start  = ctx.config.get("oos_cutoff", "2024-01-01")
                oos_frame  = feature_frame[feature_frame.index >= oos_start]
                if oos_frame.empty:
                    continue
                oos_scores = raw_scores.reindex(oos_frame.index).dropna()
                future_rets = oos_frame["label"].reindex(oos_scores.index)
                if len(oos_scores) < 50:
                    continue

                cal = fit_probability_calibration(
                    oos_scores,
                    future_rets,
                    lookahead=mp["lookahead"],
                )
                meta_path = models_dir / ticker / f"{ticker}-policy-metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    meta["score_calibration"] = cal.to_dict()
                    meta_path.write_text(json.dumps(meta, indent=2))
                    summary[ticker] = {"method": cal.method, "n": len(oos_scores)}
            except Exception as exc:
                print(f"CalibrationJob: {ticker} failed — {exc}")

        ctx.calibration_summary = summary
        print(f"CalibrationJob: calibrated {len(summary)}/{len(ctx.exported)} models")


# ── Pipeline ───────────────────────────────────────────────────────────────────

class TrainingPipeline:
    """Run all training jobs in sequence."""

    def __init__(self, jobs: list[Job] | None = None):
        self._jobs = jobs or [
            DataFetchJob(),
            RegimeFitJob(),
            FeatureJob(),
            TournamentJob(),
            ExportJob(),
            CorrelationJob(),
            CalibrationJob(),
        ]

    def run(self, ctx: TrainingContext) -> TrainingContext:
        for job in self._jobs:
            if job.should_skip(ctx):
                continue
            print(f"\n── {job.name} ──")
            job.run(ctx)
        return ctx
