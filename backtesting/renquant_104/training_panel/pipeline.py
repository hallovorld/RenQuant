"""Thin wrapper — delegates to `pp_panel_training.PanelTrainingPipeline`.

Kept as a backwards-compatible entrypoint so legacy callers
(`scripts/train_panel_model.py`, `tests/test_panel_pipeline_e2e.py`,
and earlier notebooks) keep working while the Job/Task refactor landed
in `pp_panel_training.py` becomes the single source of truth for the
Stage-1 orchestration.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
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
    LoadMacroFactorsTask,
    PanelFeatureJob,
    PanelAssemblyJob,
    PanelModelJob,
    TickerPanelContext,
    TickerPanelFeatureJob,
    TickerPanelNeutralizeJob,
    TickerPanelFactorJob,
)

log = logging.getLogger("training_panel.pipeline")


# ── Inference-side prep ──────────────────────────────────────────────────────

_INFERENCE_FRAME_CACHE_VERSION = 1


def _new_spy_hurst_memo():
    """Fresh run-scoped SPY-hurst memoizer (perf G-J).

    Isolated behind a factory so it can be disabled in tests (monkeypatch to
    return ``None``) to prove the memoized output is byte-identical to the
    un-memoized baseline. Returns a new instance every call — never a module
    global — so the cache cannot leak across runs/dates.
    """
    from training.features import SpyHurstMemo  # noqa: PLC0415
    return SpyHurstMemo()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _frame_max_date(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    if "date" in df.columns:
        return str(pd.to_datetime(df["date"]).max().date())
    try:
        return str(pd.to_datetime(df.index).max().date())
    except Exception:  # noqa: BLE001
        return None


def _cache_settings(config: dict[str, Any]) -> dict[str, Any]:
    panel_cfg = config.get("panel_ltr", {}) if isinstance(config, dict) else {}
    cache_cfg = panel_cfg.get("inference_frame_cache", {})
    if not cache_cfg:
        cache_cfg = config.get("inference_frame_cache", {})
    return cache_cfg if isinstance(cache_cfg, dict) else {}


def _cache_dir(config: dict[str, Any]) -> Path:
    cfg = _cache_settings(config)
    raw = cfg.get("cache_dir", "artifacts/sim/inference_frame_cache")
    path = Path(raw)
    if path.is_absolute():
        return path
    base = Path(config.get("_strategy_dir", Path.cwd()))
    return base / path


def _selected_config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    ranking = config.get("ranking", {}) if isinstance(config, dict) else {}
    return {
        "benchmark": config.get("benchmark", "SPY"),
        "panel_ltr": config.get("panel_ltr", {}),
        "panel_scoring": ranking.get("panel_scoring", {}),
        "sector_etf_map": config.get("sector_etf_map", {}),
    }


def _cache_source_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint side-data cache dirs so same-date OHLCV cannot reuse stale
    fundamentals, earnings, insider, intraday, or macro inputs.
    """
    roots: list[Path] = []
    strategy_dir = Path(config.get("_strategy_dir", Path.cwd()))
    for cfg_root in (
        config.get("fundamentals", {}).get("cache_dir"),
        config.get("earnings_surprise", {}).get("cache_dir"),
        config.get("insider_trades", {}).get("cache_dir"),
        config.get("hourly", {}).get("cache_dir"),
        config.get("minute", {}).get("cache_dir"),
        config.get("macro", {}).get("cache_dir"),
        config.get("fred", {}).get("cache_dir"),
    ):
        if not cfg_root:
            continue
        p = Path(cfg_root)
        roots.append(p if p.is_absolute() else strategy_dir / p)
    out: dict[str, Any] = {}
    for root in roots:
        if not root.exists():
            out[str(root)] = {"exists": False}
            continue
        files = [p for p in root.rglob("*.parquet") if p.is_file()]
        mtimes = [p.stat().st_mtime_ns for p in files]
        out[str(root)] = {
            "exists": True,
            "n_files": len(files),
            "max_mtime_ns": max(mtimes) if mtimes else None,
        }
    return out


def _inference_frame_cache_key(
    watchlist: list[str],
    ohlcv: dict[str, pd.DataFrame],
    ticker_sectors: dict[str, str],
    config: dict[str, Any],
) -> str:
    payload = {
        "version": _INFERENCE_FRAME_CACHE_VERSION,
        "watchlist": list(watchlist),
        "ticker_sectors": dict(sorted(ticker_sectors.items())),
        "ohlcv": {
            t: {
                "rows": int(len(df)),
                "max_date": _frame_max_date(df),
                "columns": list(df.columns),
            }
            for t, df in sorted(ohlcv.items())
        },
        "config": _selected_config_fingerprint(config),
        "sources": _cache_source_fingerprint(config),
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_safe).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _load_inference_frame_cache(
    config: dict[str, Any],
    key: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame | None, dict] | None:
    if not _cache_settings(config).get("enabled", False):
        return None
    root = _cache_dir(config) / key
    manifest = root / "manifest.json"
    if not manifest.exists():
        return None
    try:
        meta = json.loads(manifest.read_text())
        if meta.get("version") != _INFERENCE_FRAME_CACHE_VERSION:
            return None
        ff = {
            p.stem: pd.read_parquet(p)
            for p in sorted((root / "feature_frames").glob("*.parquet"))
        }
        fac = {
            p.stem: pd.read_parquet(p)
            for p in sorted((root / "factor_frames").glob("*.parquet"))
        }
        macro_path = root / "macro_frame.parquet"
        macro = pd.read_parquet(macro_path) if macro_path.exists() else None
        emb_path = root / "asset_embeddings.json"
        emb = json.loads(emb_path.read_text()) if emb_path.exists() else {}
        log.info(
            "prepare_inference_panel_frames: cache HIT key=%s feat=%d factor=%d",
            key, len(ff), len(fac),
        )
        return ff, fac, macro, emb
    except Exception as exc:  # noqa: BLE001
        log.warning("prepare_inference_panel_frames: cache read failed key=%s — %s", key, exc)
        return None


def _write_inference_frame_cache(
    config: dict[str, Any],
    key: str,
    feature_frames: dict[str, pd.DataFrame],
    factor_frames: dict[str, pd.DataFrame],
    macro_frame: pd.DataFrame | None,
    asset_embeddings: dict,
) -> None:
    if not _cache_settings(config).get("enabled", False):
        return
    root = _cache_dir(config)
    target = root / key
    if (target / "manifest.json").exists():
        return
    tmp = root / f".tmp-{key}-{os.getpid()}"
    try:
        (tmp / "feature_frames").mkdir(parents=True, exist_ok=True)
        (tmp / "factor_frames").mkdir(parents=True, exist_ok=True)
        for ticker, df in feature_frames.items():
            df.to_parquet(tmp / "feature_frames" / f"{ticker}.parquet")
        for ticker, df in factor_frames.items():
            df.to_parquet(tmp / "factor_frames" / f"{ticker}.parquet")
        if macro_frame is not None:
            macro_frame.to_parquet(tmp / "macro_frame.parquet")
        if asset_embeddings:
            (tmp / "asset_embeddings.json").write_text(
                json.dumps(asset_embeddings, sort_keys=True, default=_json_safe)
            )
        (tmp / "manifest.json").write_text(json.dumps({
            "version": _INFERENCE_FRAME_CACHE_VERSION,
            "key": key,
            "n_feature_frames": len(feature_frames),
            "n_factor_frames": len(factor_frames),
        }, indent=2, sort_keys=True))
        tmp.rename(target)
        log.info(
            "prepare_inference_panel_frames: cache WRITE key=%s feat=%d factor=%d",
            key, len(feature_frames), len(factor_frames),
        )
    except FileExistsError:
        # Another parallel sim wrote the same key first.
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("prepare_inference_panel_frames: cache write failed key=%s — %s", key, exc)

def prepare_inference_panel_frames(
    watchlist: list[str],
    ohlcv: dict[str, pd.DataFrame],
    ticker_sectors: dict[str, str],
    config: dict[str, Any],
) -> "tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame | None, dict]":
    """Build neutralized feature frames + z-scored factor frames + macro frame for live inference.

    Mirrors Phase 1 (SectorMomentum + Load* tasks) + Phase 2 (per-ticker
    Feature+Neutralize+Factor) + FactorZScoreTask of PanelTrainingPipeline,
    but without building labels / panel frame / training.

    Returns ``(neutralized_frames, factor_frames_z, macro_frame, asset_embeddings)``.
    Adapters attach all four to the InferenceContext (as `_panel_feature_frames`,
    `_panel_factor_frames`, `_panel_macro_frame`, `_panel_asset_embeddings`)
    before running PanelScoringJob.

    T2-2 (2026-04-27): added ``asset_embeddings`` as 4th return value.
    LoadAssetEmbeddingsTask was already run here and stored on ctx but the
    result was silently discarded — causing emb_0..emb_15 to fall back to
    NaN at inference time while training used real embeddings.

    Bug #25 fix (2026-04-26 round-7): macro_frame added as third return
    value. When `panel_ltr.macro.enabled=true`, training builds a panel
    with broadcast macro features; inference must produce a matching
    feature_cols set. The symmetry guard test
    `tests/test_train_inference_symmetry.py` enforces that every Load*Task
    in `PanelDataJob.tasks` is also exercised here.

    `ohlcv` must already contain the benchmark (SPY) and every sector ETF
    referenced by `sector_etf_map` in config.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    cache_key = _inference_frame_cache_key(watchlist, ohlcv, ticker_sectors, config)
    cached = _load_inference_frame_cache(config, cache_key)
    if cached is not None:
        return(cached)

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
    # Bug #25 fix (2026-04-26 round-7): inference symmetry on macros.
    # PanelDataJob.tasks lists LoadMacroFactorsTask; this hand-written
    # chain must mirror it OR a symmetry guard test fails.
    LoadMacroFactorsTask().run(ctx)

    # Macro v2 (2026-04-27): per-ticker β. Must mirror PanelDataJob.tasks
    # order — the symmetry guard test enforces this.
    # Tier 2 FRED (2026-04-27): runs BETWEEN LoadMacroFactorsTask and
    # LoadMacroPerTickerBetasTask so the β computation picks up FRED
    # columns alongside ETF columns from the merged macro frame.
    from training_panel.pp_panel_training import (  # noqa: PLC0415
        LoadFredMacroTask,
        LoadMacroPerTickerBetasTask,
        LoadAssetEmbeddingsTask,
    )
    LoadFredMacroTask().run(ctx)
    LoadMacroPerTickerBetasTask().run(ctx)
    # T2-2 (2026-04-27): asset embeddings — same symmetry requirement.
    LoadAssetEmbeddingsTask().run(ctx)

    # Perf (G-J): one memoizer shared across every per-ticker feature build so
    # the redundant SPY rolling-Hurst recomputation collapses to one per
    # distinct spy_rets. Run-scoped (fresh per call); output-invariant.
    hurst_memo = _new_spy_hurst_memo()
    ticker_ctxs = [
        TickerPanelContext(
            ticker=t, ohlcv=ctx.ohlcv, sector_momentum=ctx.sector_momentum,
            ticker_sectors=ctx.ticker_sectors, config=ctx.config,
            fundamentals=ctx.fundamentals,
            earnings_surprises=ctx.earnings_surprises,
            insider_trades=ctx.insider_trades,
            hourly_bars=ctx.hourly_bars,
            minute_bars=ctx.minute_bars,
            hurst_cache=hurst_memo,
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
    progress_seconds = float(config.get("panel_inference_progress_log_seconds",
                                        config.get("parallel_progress_log_seconds", 30)))
    timeout_raw = config.get("panel_inference_timeout_seconds")
    if timeout_raw is None:
        timeout_raw = (config.get("panel_ltr", {}) or {}).get("inference_frame_timeout_seconds")
    timeout_seconds = float(timeout_raw) if timeout_raw is not None else None
    t0 = time.monotonic()
    next_progress = t0 + max(0.1, progress_seconds)
    ex = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="panel-inf")
    abandon_executor = False
    try:
        futs = {ex.submit(_chain, tc): tc.ticker for tc in ticker_ctxs}
        pending = set(futs)
        done_count = 0
        while pending:
            now = time.monotonic()
            elapsed = now - t0
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                pending_tickers = sorted(futs[f] for f in pending)
                for f in pending:
                    f.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                abandon_executor = True
                raise TimeoutError(
                    "prepare_inference_panel_frames timed out after "
                    f"{elapsed:.2f}s with {len(pending_tickers)} pending ticker(s): "
                    f"{pending_tickers[:20]}"
                )
            wait_timeout = max(0.0, next_progress - now)
            if timeout_seconds is not None:
                wait_timeout = min(wait_timeout, max(0.0, timeout_seconds - elapsed))
            done, pending = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)
            for f in done:
                ticker = futs[f]
                done_count += 1
                try:
                    f.result()
                except Exception as exc:
                    _log_inf.error(
                        "prepare_inference_panel_frames[%s]: chain ERROR — %s: %s "
                        "(ticker dropped from this bar's panel matrix)",
                        ticker, type(exc).__name__, exc,
                    )
            now = time.monotonic()
            if pending and now >= next_progress:
                pending_tickers = sorted(futs[f] for f in pending)
                _log_inf.info(
                    "prepare_inference_panel_frames: progress done=%d/%d "
                    "pending=%d elapsed=%.2fs pending_tickers=%s",
                    done_count, len(futs), len(pending_tickers), now - t0,
                    pending_tickers[:10],
                )
                next_progress = now + max(0.1, progress_seconds)
    finally:
        if not abandon_executor:
            ex.shutdown(wait=True)

    ctx.neutralized_frames = {
        tc.ticker: tc.neutralized_frame for tc in ticker_ctxs
        if tc.neutralized_frame is not None
    }
    ctx.raw_factor_frames = {
        tc.ticker: tc.raw_factor_frame for tc in ticker_ctxs
        if tc.raw_factor_frame is not None
    }

    # Macro v2 (2026-04-27): merge per-ticker β into raw_factor_frames
    # — same protocol as training-side PanelFeatureJob.
    if ctx.macro_betas:
        n_merged = 0
        for ticker, beta_df in ctx.macro_betas.items():
            if ticker not in ctx.raw_factor_frames or beta_df.empty:
                continue
            fac = ctx.raw_factor_frames[ticker]
            beta_aligned = beta_df.reindex(fac.index)
            existing = set(fac.columns)
            new_cols = [c for c in beta_aligned.columns if c not in existing]
            if new_cols:
                ctx.raw_factor_frames[ticker] = pd.concat(
                    [fac, beta_aligned[new_cols]], axis=1, copy=False,
                )
                n_merged += 1
        log.info("prepare_inference_panel_frames[macro v2]: merged β into "
                 "%d/%d raw_factor_frames", n_merged, len(ctx.raw_factor_frames))

    # Cross-sectional z-score per-ticker indicators so inference distribution
    # matches training. Must run BEFORE FactorZScoreTask so order matches
    # PanelAssemblyJob in the training pipeline.
    NeutralizedFeatureZScoreTask().run(ctx)
    FactorZScoreTask().run(ctx)

    # Bug #25 fix: return macro_frame too so adapters can attach to
    # InferenceContext for cross-section broadcast at scoring time.
    # T2-2 fix: also return asset_embeddings so adapters attach to
    # InferenceContext as _panel_asset_embeddings for build_inference_matrix.
    _write_inference_frame_cache(
        config, cache_key, ctx.neutralized_frames, ctx.factor_frames,
        ctx.macro_factor_frame, ctx.asset_embeddings,
    )
    return ctx.neutralized_frames, ctx.factor_frames, ctx.macro_factor_frame, ctx.asset_embeddings


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
