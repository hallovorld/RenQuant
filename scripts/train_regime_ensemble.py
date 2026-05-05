#!/usr/bin/env python3
"""T2-3 Phase B — Train regime-conditional ensemble panel-LTR models.

Per roadmap T2-3 (Two Sigma 2024): train SEPARATE panel-LTR XGBoost models
per macro regime (BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR), routing at
inference via existing `ctx.regime` (handled by `kernel/panel_pipeline/regime_router.py`).

This script:
  1. Runs PanelDataJob → PanelFeatureJob → PanelAssemblyJob to build the
     full in-memory panel (~3 min on the 99-ticker watchlist).
  2. Replays detect_regime() over the SPY returns history to get a per-date
     regime label, using the pre-trained GMM artifact.
  3. For each regime, filters the panel to only dates with that label, then
     runs CrossValidateTask + FinalFitTask + SaveArtifactTask with a
     regime-specific output path.

Output artifacts (regime router naming convention):
  artifacts/panel-ltr.regime-bull_calm.json
  artifacts/panel-ltr.regime-bull_volatile.json
  artifacts/panel-ltr.regime-choppy.json
  artifacts/panel-ltr.regime-bear.json

Usage:
    python scripts/train_regime_ensemble.py [--strategy renquant_104] [--dry-run]

After training, enable routing in strategy_config.json:
    "panel_ltr": { "regime_ensemble": { "enabled": true } }
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("train-regime-ensemble")

KNOWN_REGIMES = ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR")
MIN_REGIME_DATES = 30   # skip regime if fewer than this many unique dates
MIN_REGIME_ROWS  = 500  # skip regime if fewer than this many panel rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Build regime map from SPY history + pre-trained GMM
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_map(
    spy_ohlcv: pd.DataFrame,
    gmm_artifact: dict | None,
    config: dict,
) -> dict:
    """Replay detect_regime() over the full SPY history.

    Returns {date: regime_str} for every date in `spy_ohlcv`.
    State (CUSUM, countdown) is threaded through calls so regime switches
    are detected the same way as in production.
    """
    from kernel.regime import detect_regime, RegimeState

    spy_close   = spy_ohlcv["close"].sort_index()
    spy_ret_s   = spy_close.pct_change().dropna()
    returns_arr = spy_ret_s.values
    dates       = list(spy_ret_s.index)

    state            = RegimeState()
    date_to_regime: dict = {}
    t0 = time.monotonic()

    for i, date in enumerate(dates):
        detect_regime(
            spy_returns=returns_arr[: i + 1],
            spy_df=spy_ohlcv.iloc[: i + 1],
            gmm_artifact=gmm_artifact,
            state=state,
            config=config,
        )
        date_to_regime[date] = state.regime

    elapsed = time.monotonic() - t0
    dist = {r: sum(v == r for v in date_to_regime.values()) for r in KNOWN_REGIMES}
    log.info(
        "build_regime_map: %d dates in %.1fs — distribution: %s",
        len(date_to_regime), elapsed,
        "  ".join(f"{r}={n}" for r, n in dist.items()),
    )
    return date_to_regime


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Train one model per regime
# ─────────────────────────────────────────────────────────────────────────────

def train_regime_models(
    ctx,
    date_to_regime: dict,
    strategy_dir: Path,
    dry_run: bool = False,
) -> dict:
    """For each regime, filter the panel and train + save an XGBoost artifact.

    Uses a shallower CV config to stay fast on smaller subsets:
      - cv_method = "purged"  (faster than CPCV)
      - cv_n_splits = 5
    The full panel uses CPCV-15; per-regime panels are ~25% of full size.
    """
    from training_panel.pp_panel_training import (
        CrossValidateTask, FinalFitTask, SaveArtifactTask,
    )

    # Normalise dates for consistent mapping
    panel = ctx.panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # Build a normalised lookup: try exact match first, then date-only match
    # (handles timezone-aware vs naive mismatches between SPY index and panel)
    regime_lookup: dict = {}
    for d, r in date_to_regime.items():
        key = pd.Timestamp(d)
        regime_lookup[key] = r
        regime_lookup[key.normalize()] = r   # date-only fallback

    def _get_regime(ts: pd.Timestamp) -> str:
        r = regime_lookup.get(ts)
        if r is None:
            r = regime_lookup.get(ts.normalize())
        return r or "UNKNOWN"

    panel["regime"] = panel["date"].apply(_get_regime)

    unknown_frac = (panel["regime"] == "UNKNOWN").mean()
    if unknown_frac > 0.05:
        log.warning(
            "%.1f%% of panel rows have no regime label — "
            "likely a date-alignment gap between SPY history and panel dates",
            unknown_frac * 100,
        )

    results = {}
    for regime in KNOWN_REGIMES:
        regime_lower = regime.lower()
        out_path = strategy_dir / "artifacts" / f"panel-ltr.regime-{regime_lower}.json"

        regime_panel = panel[panel["regime"] == regime].drop(
            columns=["regime"], errors="ignore",
        ).reset_index(drop=True)

        n_rows  = len(regime_panel)
        n_dates = int(regime_panel["date"].nunique())

        log.info(
            "Regime %-15s  rows=%-6d  dates=%-4d  artifact=%s",
            regime, n_rows, n_dates, out_path.name,
        )

        if n_dates < MIN_REGIME_DATES:
            log.warning(
                "Regime %s: only %d dates (< %d) — skipping",
                regime, n_dates, MIN_REGIME_DATES,
            )
            results[regime] = {"status": "skipped_too_few_dates", "n_dates": n_dates}
            continue

        if n_rows < MIN_REGIME_ROWS:
            log.warning(
                "Regime %s: only %d rows (< %d) — skipping",
                regime, n_rows, MIN_REGIME_ROWS,
            )
            results[regime] = {"status": "skipped_too_few_rows", "n_rows": n_rows}
            continue

        if dry_run:
            log.info("DRY RUN — skipping actual training for %s", regime)
            results[regime] = {
                "status": "dry_run",
                "n_rows": n_rows,
                "n_dates": n_dates,
            }
            continue

        # Recompute group sizes for the filtered panel
        group_sizes = (
            regime_panel.groupby("date", sort=True)
            .size()
            .values
            .astype(np.int32)
        )

        # Build per-regime sub-context (shallow copy — shared ohlcv, feature_cols, etc.)
        sub_ctx = copy.copy(ctx)
        sub_ctx.panel        = regime_panel
        sub_ctx.group_sizes  = group_sizes
        sub_ctx.panel_metadata = {
            "n_rows":    n_rows,
            "n_tickers": int(regime_panel["ticker"].nunique()),
            "n_dates":   n_dates,
        }
        sub_ctx.final_model      = None
        sub_ctx.cv_result        = {}
        sub_ctx.artifact_path    = None
        sub_ctx.summary          = {}
        sub_ctx._final_fit       = {}   # noqa: SLF001
        sub_ctx._final_fit_elapsed_sec = 0.0  # noqa: SLF001
        sub_ctx._final_fit_device      = "cpu"  # noqa: SLF001

        # Clone config; switch to faster purged-5 CV + set output path
        sub_config = copy.deepcopy(ctx.config)
        sub_config["panel_ltr"]["artifact_path"]   = str(out_path)
        sub_config["panel_ltr"]["training_notes"]  = (
            f"T2-3 regime ensemble — {regime}"
        )
        # Use lighter CV for per-regime subsets
        sub_config["panel_ltr"]["cv_method"]       = "purged"
        sub_config["panel_ltr"]["cv_n_splits"]     = 5
        # Keep early stopping + num_boost_round from golden config
        sub_ctx.config = sub_config

        log.info("── Training %s model (rows=%d dates=%d) …", regime, n_rows, n_dates)
        t0 = time.monotonic()
        try:
            CrossValidateTask().run(sub_ctx)
            FinalFitTask().run(sub_ctx)
            SaveArtifactTask().run(sub_ctx)
            elapsed = time.monotonic() - t0
            mean_ic = sub_ctx.cv_result.get("mean_ic", 0.0)
            per_fold = sub_ctx.cv_result.get("per_fold_ic", [])
            results[regime] = {
                "status":   "ok",
                "mean_ic":  mean_ic,
                "per_fold_ic": per_fold,
                "n_rows":   n_rows,
                "n_dates":  n_dates,
                "artifact": str(out_path),
                "elapsed_sec": round(elapsed, 1),
            }
            log.info(
                "── %-15s  DONE  mean_ic=%+.4f  per_fold=%s  elapsed=%.1fs",
                regime, mean_ic,
                "[" + ", ".join(f"{v:+.4f}" for v in per_fold) + "]",
                elapsed,
            )
        except Exception as exc:
            log.error(
                "── %s training FAILED: %s: %s",
                regime, type(exc).__name__, exc, exc_info=True,
            )
            results[regime] = {"status": "error", "error": str(exc)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="renquant_104")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Build panel + compute regime map but skip actual training")
    args = parser.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config_path  = strategy_dir / "strategy_config.json"
    gmm_path     = strategy_dir / "artifacts" / "spy-gmm-regime.json"

    if not config_path.exists():
        sys.exit(f"Strategy config not found: {config_path}")

    config = json.loads(config_path.read_text())
    config["_strategy_dir"] = str(strategy_dir)

    # Insert strategy dir so kernel.* imports resolve
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    # Load pre-trained GMM artifact for regime detection
    gmm_artifact: dict | None = None
    if gmm_path.exists():
        with open(gmm_path) as f:
            gmm_artifact = json.load(f)
        log.info("Loaded GMM artifact from %s", gmm_path)
    else:
        log.warning(
            "GMM artifact not found at %s — regime detection will rely on "
            "Hurst exponent only (less accurate)", gmm_path,
        )

    # ── Phase 1: Build the panel ─────────────────────────────────────────────
    from training_panel.context import PanelTrainingContext
    from training_panel.pp_panel_training import (
        PanelDataJob, PanelFeatureJob, PanelAssemblyJob,
    )

    watchlist = config["watchlist"]
    ctx = PanelTrainingContext(
        config=config,
        watchlist=list(watchlist),
        ticker_sectors={t: config.get("sector_map", {}).get(t, "UNKNOWN") for t in watchlist},
    )

    log.info("=" * 68)
    log.info("T2-3 Phase B — Regime Ensemble Training  strategy=%s", args.strategy)
    log.info("=" * 68)
    log.info("Phase 1/3: Building full panel (PanelDataJob → AssemblyJob) …")

    t_panel_start = time.monotonic()
    PanelDataJob().run(ctx)
    PanelFeatureJob().run(ctx)
    PanelAssemblyJob().run(ctx)
    t_panel = time.monotonic() - t_panel_start

    if ctx.panel is None or ctx.panel.empty:
        sys.exit("ERROR: panel is empty after PanelAssemblyJob — aborting")

    log.info(
        "Panel built in %.1fs: rows=%d  dates=%d  tickers=%d  features=%d",
        t_panel,
        len(ctx.panel),
        ctx.panel["date"].nunique(),
        ctx.panel["ticker"].nunique(),
        len(ctx.feature_cols),
    )

    # ── Phase 2: Build per-date regime map ──────────────────────────────────
    log.info("Phase 2/3: Replaying regime detection over SPY history …")
    spy_ohlcv = ctx.ohlcv.get(config.get("benchmark", "SPY"))
    if spy_ohlcv is None:
        sys.exit("ERROR: SPY OHLCV missing from ctx.ohlcv — aborting")

    date_to_regime = build_regime_map(spy_ohlcv, gmm_artifact, config)

    # ── Phase 3: Train per-regime models ────────────────────────────────────
    log.info("Phase 3/3: Training regime-specific panel-LTR models …")
    results = train_regime_models(ctx, date_to_regime, strategy_dir, dry_run=args.dry_run)

    # ── Final summary ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 68)
    log.info("T2-3 Regime Ensemble — RESULTS")
    log.info("=" * 68)
    all_ok = True
    for regime in KNOWN_REGIMES:
        result = results.get(regime, {"status": "not_run"})
        status = result["status"]
        if status == "ok":
            log.info(
                "  %-16s  ✓  mean_ic=%+.4f  rows=%-6d  dates=%-4d  %s",
                regime,
                result["mean_ic"],
                result["n_rows"],
                result["n_dates"],
                Path(result["artifact"]).name,
            )
        else:
            log.info("  %-16s  ✗  %s", regime, status)
            if status not in ("dry_run",):
                all_ok = False

    log.info("")
    if not args.dry_run:
        if all_ok:
            log.info("All regime models trained. To enable routing, add to strategy_config.json:")
            log.info('  "panel_ltr": { ... "regime_ensemble": { "enabled": true } }')
        else:
            log.info("Some regimes failed or were skipped. Check logs above.")

    # Print RegimeRouter inventory
    if not args.dry_run:
        try:
            from kernel.panel_pipeline.regime_router import RegimeRouter
            router = RegimeRouter(strategy_dir, config)
            inv = router.inventory()
            log.info("")
            log.info("RegimeRouter.inventory():")
            for r, info in inv.items():
                flag = "✓" if info["exists"] else "✗"
                sz   = f"  {info['size']:,} bytes" if info["exists"] else ""
                log.info("  %s %-16s  %s%s", flag, r, Path(info["path"]).name, sz)
            log.info("has_regime_ensemble (config gate)= %s", router.has_regime_ensemble())
        except Exception as exc:
            log.warning("Could not build RegimeRouter inventory: %s", exc)


if __name__ == "__main__":
    main()
