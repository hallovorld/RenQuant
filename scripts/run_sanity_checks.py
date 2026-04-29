#!/usr/bin/env python
"""Sanity checks for panel-LTR artifacts — CLAUDE.md §5.2.

Mandatory minimum tests for any new CPCV IC metric:
  1. A/A test: same data, randomly resplit — IC should be ≈ 0 (catches selection bias)
  2. Shuffled-label test: shuffle y, retrain — IC should be ≈ 0
  3. Label-shift placebo: shift labels by 252 bars — IC should drop toward 0

Usage::

    python scripts/run_sanity_checks.py
    python scripts/run_sanity_checks.py --strategy-config-name strategy_config.h60_103.json
    python scripts/run_sanity_checks.py --test aa        # only A/A
    python scripts/run_sanity_checks.py --test shuffle   # only label shuffle
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("sanity-checks")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--strategy-config-name", default="strategy_config.json")
    p.add_argument("--test", choices=["aa", "shuffle", "shift", "all"], default="all")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Number of random seeds for A/A test (default 3)")
    p.add_argument("--shift-bars", type=int, default=252,
                   help="Label shift for placebo test in trading days (default 252)")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))
    config = json.loads((strategy_dir / args.strategy_config_name).read_text())
    config["_strategy_dir"] = str(strategy_dir)
    config["_strategy_config_name"] = args.strategy_config_name

    from training_panel.pp_panel_training import (  # noqa: PLC0415
        PanelTrainingContext, PanelTrainingPipeline,
        PanelDataJob, PanelFeatureJob, PanelAssemblyJob,
    )
    from training_panel.purged_cv import PurgedCVSplitter  # noqa: PLC0415
    from training_panel.ltr_model import PanelLTRModel  # noqa: PLC0415
    from scipy.stats import spearmanr  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    # ── Build panel once (reused for all tests) ──────────────────────────────
    log.info("Building panel for sanity checks …")
    ctx = PanelTrainingContext(
        config=config,
        strategy=args.strategy,
        strategy_dir=strategy_dir,
    )
    PanelDataJob().run(ctx)
    PanelFeatureJob().run(ctx)
    PanelAssemblyJob().run(ctx)

    panel = ctx.panel_frame
    if panel is None or panel.empty:
        log.error("Panel is empty — cannot run sanity checks")
        sys.exit(1)

    panel_cfg = config.get("panel_ltr", {})
    cv_cfg = panel_cfg.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 15))
    embargo = int(cv_cfg.get("embargo_days", 5))
    feature_cols = ctx.feature_cols
    label_col = "label"
    xgb_params = dict(panel_cfg.get("xgb_params", {}))

    splitter = PurgedCVSplitter(n_splits=n_splits, embargo_days=embargo)
    dates = sorted(panel["date"].unique())

    def _fold_ic(panel_data: pd.DataFrame, seed: int | None = None) -> list[float]:
        """Run one CPCV pass; return per-fold IC list."""
        ics = []
        for train_idx, test_idx in splitter.split(dates):
            train_dates = {dates[i] for i in train_idx}
            test_dates  = {dates[i] for i in test_idx}
            train = panel_data[panel_data["date"].isin(train_dates)].dropna(
                subset=[label_col])
            test  = panel_data[panel_data["date"].isin(test_dates)].dropna(
                subset=[label_col])
            if len(train) < 100 or len(test) < 20:
                continue
            train_groups = train.groupby("date").size().values
            model = PanelLTRModel(params=xgb_params)
            model.train(train, train_groups,
                        feature_cols=feature_cols, label_col=label_col,
                        weight_col="weight", num_boost_round=50)
            preds = model.predict(test)
            for d, grp in test.join(preds.rename("score")).groupby("date"):
                if len(grp) < 5:
                    continue
                rho, _ = spearmanr(grp["score"], grp[label_col])
                if np.isfinite(rho):
                    ics.append(float(rho))
        return ics

    results: dict[str, float] = {}

    # ── A/A test ──────────────────────────────────────────────────────────────
    if args.test in ("aa", "all"):
        log.info("Running A/A test (%d seeds) …", args.n_seeds)
        aa_ics = []
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed + 1000)
            shuffled = panel.copy()
            # Shuffle labels WITHIN each date (keep cross-section intact, destroy
            # time-series predictability — IC should ≈ 0 if signal is real)
            def _shuffle_date(grp: pd.DataFrame) -> pd.DataFrame:
                idx = grp.index
                perm = rng.permutation(len(idx))
                grp = grp.copy()
                grp[label_col] = grp[label_col].values[perm]
                return grp
            shuffled = shuffled.groupby("date", group_keys=False).apply(_shuffle_date)
            ics = _fold_ic(shuffled)
            mean_ic = float(np.mean(ics)) if ics else float("nan")
            log.info("  A/A seed=%d  mean_ic=%+.4f", seed, mean_ic)
            aa_ics.append(mean_ic)
        aa_mean = float(np.mean(aa_ics))
        results["aa_mean_ic"] = aa_mean
        passed = abs(aa_mean) < 0.01
        log.info("A/A test: mean=%+.4f  %s (threshold |IC| < 0.01)",
                 aa_mean, "PASS ✓" if passed else "FAIL ✗")

    # ── Label shuffle test ────────────────────────────────────────────────────
    if args.test in ("shuffle", "all"):
        log.info("Running label-shuffle test …")
        shuffled = panel.copy()
        rng = np.random.default_rng(42)
        shuffled[label_col] = rng.permutation(shuffled[label_col].values)
        ics = _fold_ic(shuffled)
        shuf_ic = float(np.mean(ics)) if ics else float("nan")
        results["shuffle_mean_ic"] = shuf_ic
        passed = abs(shuf_ic) < 0.01
        log.info("Label-shuffle: mean_ic=%+.4f  %s",
                 shuf_ic, "PASS ✓" if passed else "FAIL ✗")

    # ── Label-shift placebo ───────────────────────────────────────────────────
    if args.test in ("shift", "all"):
        log.info("Running label-shift placebo (shift=%d bars) …", args.shift_bars)
        shifted = panel.copy()
        shifted[label_col] = shifted.groupby("ticker")[label_col].shift(
            args.shift_bars)
        shifted = shifted.dropna(subset=[label_col])
        ics = _fold_ic(shifted)
        shift_ic = float(np.mean(ics)) if ics else float("nan")
        results["shift_mean_ic"] = shift_ic
        passed = shift_ic < 0.005
        log.info("Label-shift (%dd): mean_ic=%+.4f  %s",
                 args.shift_bars, shift_ic, "PASS ✓" if passed else "FAIL ✗")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("SANITY CHECK SUMMARY")
    log.info("=" * 60)
    for k, v in results.items():
        log.info("  %-25s %+.4f", k, v)
    log.info("")
    log.info("Reference: production CPCV IC = +0.0350 (10d), +0.0738 (60d)")
    log.info("All sanity checks should show IC ≈ 0 if signal is real.")


if __name__ == "__main__":
    main()
