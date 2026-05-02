#!/usr/bin/env python
"""Sanity-check a panel-LTR retrain — shuffled-label + time-shift placebo.

Per CLAUDE.md §5.2: "every new metric ships with at minimum: A/A test,
shuffled-label test, time-shift placebo. No exceptions."

Before declaring an architecture experiment "succeeded," its CPCV
mean_ic must survive these falsification checks:

  1. Shuffled-label test — train the SAME architecture on the SAME panel
     with labels permuted within each date. Expected: CPCV ≈ 0. If
     positive, the architecture is overfitting to noise (or there's
     a label leakage bug).

  2. Time-shift placebo — train with labels shifted forward by N days
     (default 60). Real predictability of features at horizon h vs
     labels at h+N should be ≈ 0. If positive, the model is leaking
     or signal is suspicious.

Both checks are HARD GATES before promoting any architecture. Cost:
~20 min compute per check on M2 Pro for a 178-ticker panel.

Usage::

    python scripts/sanity_check_panel_artifact.py \\
        --strategy-config-name strategy_config.wl178_v2_l1sub_l2.json \\
        --check shuffled
    python scripts/sanity_check_panel_artifact.py \\
        --strategy-config-name strategy_config.wl178_v2_l1sub_l2.json \\
        --check timeshift --shift-days 60

Each check:
  * builds the panel with the same Job chain
  * permutes / shifts labels in PanelTrainingContext
  * runs CrossValidateTask
  * writes a sanity report to data/audit/sanity_<check>_<label>.json

Exit codes
----------
  0  — sanity check ran cleanly (gate verdict in JSON report)
  1  — invalid args / config not found
  2  — sanity-check execution failed
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
log = logging.getLogger("sanity-check")


def _load_config(strategy_dir: Path, name: str) -> dict:
    cfg_path = strategy_dir / name
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path); sys.exit(1)
    cfg = json.loads(cfg_path.read_text())
    cfg["_strategy_dir"] = str(strategy_dir)
    cfg["_strategy_config_name"] = name
    return cfg


def _build_panel(strategy_dir: Path, cfg: dict):
    """Walk PanelTrainingPipeline up through PanelAssemblyJob to get
    the panel DataFrame and feature_cols, then stop short of
    PanelModelJob (we want to run our own CV with permuted labels).
    """
    sys.path.insert(0, str(strategy_dir))
    from training_panel.pp_panel_training import (   # noqa: PLC0415
        PanelTrainingContext,
        PanelDataJob, PanelFeatureJob, PanelAssemblyJob,
    )
    pctx = PanelTrainingContext(config=cfg)
    log.info("Building panel via PanelDataJob → PanelFeatureJob → PanelAssemblyJob …")
    PanelDataJob().run(pctx)
    PanelFeatureJob().run(pctx)
    PanelAssemblyJob().run(pctx)
    if pctx.panel is None or pctx.panel.empty:
        log.error("Panel empty after assembly — cannot run sanity check")
        sys.exit(2)
    log.info("Panel: %d rows × %d cols  feature_cols=%d",
             len(pctx.panel), len(pctx.panel.columns), len(pctx.feature_cols or []))
    return pctx


def _shuffled_label_check(pctx, cfg: dict, seed: int) -> dict:
    """Permute the y label within each date independently. Expected
    CPCV mean_ic ≈ 0 (no signal — labels are random per date).
    """
    from training_panel.purged_cv import cross_validated_ic_cpcv  # noqa: PLC0415
    from training_panel.pp_panel_training import (   # noqa: PLC0415
        _build_xy_arrays_for_cv, _make_cv_model_factory,
    )
    panel = pctx.panel
    feature_cols = pctx.feature_cols
    label_col = cfg.get("panel_ltr", {}).get("label_col", "label_z")
    log.info("Shuffled-label test — permuting %s within each date (seed=%d)",
             label_col, seed)

    # Permute label per date
    rng = np.random.default_rng(seed)
    panel_shuffled = panel.copy()
    if "date" in panel_shuffled.columns:
        for d, idx in panel_shuffled.groupby("date").groups.items():
            vals = panel_shuffled.loc[idx, label_col].values.copy()
            rng.shuffle(vals)
            panel_shuffled.loc[idx, label_col] = vals
    else:
        log.warning("Panel has no 'date' column — falling back to global shuffle")
        vals = panel_shuffled[label_col].values.copy()
        rng.shuffle(vals)
        panel_shuffled[label_col] = vals

    # Build the same model factory as production training
    factory = _make_cv_model_factory(cfg, feature_cols)
    X, y, w, groups = _build_xy_arrays_for_cv(
        panel_shuffled, feature_cols=feature_cols,
        label_col=label_col,
        weight_col=cfg.get("panel_ltr", {}).get("weight_col"),
    )
    n_splits = int(cfg.get("panel_ltr", {}).get("cpcv_splits", 15))
    embargo  = int(cfg.get("panel_ltr", {}).get("cpcv_embargo_days", 20))
    log.info("Running CPCV with permuted labels (n_splits=%d, embargo=%d) …",
             n_splits, embargo)
    result = cross_validated_ic_cpcv(
        X, y, groups, factory, n_splits=n_splits, embargo=embargo,
        sample_weight=w,
    )
    return {
        "kind":         "shuffled_label",
        "seed":         seed,
        "label_col":    label_col,
        "mean_ic":      float(result.mean_ic),
        "std":          float(result.std),
        "n_splits":     int(result.n_splits),
        "verdict":      ("✅ PASS — shuffled IC near 0"
                         if abs(result.mean_ic) < 0.005
                         else "❌ FAIL — shuffled IC > 0.005, suspect leakage/overfit"),
    }


def _time_shift_check(pctx, cfg: dict, shift_days: int) -> dict:
    """Shift y forward by `shift_days` so features predict labels from
    a future window unrelated to the training-time horizon. Expected
    CPCV mean_ic ≈ 0 — true alpha at horizon h doesn't predict labels
    at h+shift.
    """
    from training_panel.purged_cv import cross_validated_ic_cpcv  # noqa: PLC0415
    from training_panel.pp_panel_training import (   # noqa: PLC0415
        _build_xy_arrays_for_cv, _make_cv_model_factory,
    )
    panel = pctx.panel
    feature_cols = pctx.feature_cols
    label_col = cfg.get("panel_ltr", {}).get("label_col", "label_z")
    log.info("Time-shift test — shifting %s forward by %d days",
             label_col, shift_days)

    panel_shift = panel.copy().sort_values(["ticker", "date"])
    # Per-ticker shift: each ticker's labels move forward by shift_days
    panel_shift[label_col] = (
        panel_shift.groupby("ticker")[label_col]
        .shift(-shift_days)   # negative shift: pull future label to current row
    )
    panel_shift = panel_shift.dropna(subset=[label_col])
    log.info("After shift: %d rows (lost %d to tail truncation)",
             len(panel_shift), len(panel) - len(panel_shift))

    factory = _make_cv_model_factory(cfg, feature_cols)
    X, y, w, groups = _build_xy_arrays_for_cv(
        panel_shift, feature_cols=feature_cols,
        label_col=label_col,
        weight_col=cfg.get("panel_ltr", {}).get("weight_col"),
    )
    n_splits = int(cfg.get("panel_ltr", {}).get("cpcv_splits", 15))
    embargo  = int(cfg.get("panel_ltr", {}).get("cpcv_embargo_days", 20))
    log.info("Running CPCV with time-shifted labels …")
    result = cross_validated_ic_cpcv(
        X, y, groups, factory, n_splits=n_splits, embargo=embargo,
        sample_weight=w,
    )
    return {
        "kind":          "time_shift",
        "shift_days":    shift_days,
        "label_col":     label_col,
        "n_rows_after_shift": int(len(panel_shift)),
        "mean_ic":       float(result.mean_ic),
        "std":           float(result.std),
        "n_splits":      int(result.n_splits),
        "verdict":       ("✅ PASS — time-shifted IC near 0"
                          if abs(result.mean_ic) < 0.005
                          else "❌ FAIL — time-shifted IC > 0.005, suspect leakage"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--strategy-config-name", required=True)
    p.add_argument("--check", choices=["shuffled", "timeshift", "both"],
                   default="both",
                   help="Which sanity check(s) to run.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for shuffled-label test.")
    p.add_argument("--shift-days", type=int, default=60,
                   help="Forward shift for time-shift placebo.")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: data/audit/sanity_<config>.json)")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    if not strategy_dir.exists():
        log.error("Strategy dir not found: %s", strategy_dir); return 1

    cfg = _load_config(strategy_dir, args.strategy_config_name)
    pctx = _build_panel(strategy_dir, cfg)

    started = _dt.datetime.now(_dt.timezone.utc)
    results: list[dict] = []
    if args.check in ("shuffled", "both"):
        results.append(_shuffled_label_check(pctx, cfg, args.seed))
    if args.check in ("timeshift", "both"):
        results.append(_time_shift_check(pctx, cfg, args.shift_days))

    finished = _dt.datetime.now(_dt.timezone.utc)
    report = {
        "config":         args.strategy_config_name,
        "started_utc":    started.isoformat(),
        "finished_utc":   finished.isoformat(),
        "wall_seconds":   (finished - started).total_seconds(),
        "checks":         results,
        "all_passed":     all("PASS" in r["verdict"] for r in results),
    }

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "audit"
        / f"sanity_{args.strategy_config_name.replace('.json', '')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 70)
    print(f"  SANITY CHECK — {args.strategy_config_name}")
    print("=" * 70)
    for r in results:
        print(f"  {r['kind']:<14} mean_ic={r['mean_ic']:+.4f}  std={r['std']:.4f}")
        print(f"                {r['verdict']}")
    print(f"  Overall: {'✅ ALL PASS' if report['all_passed'] else '❌ FAIL'}")
    print(f"  Report:  {out_path}")
    print("=" * 70)
    return 0 if report["all_passed"] else 0   # exit 0 either way; verdict in JSON


if __name__ == "__main__":
    sys.exit(main())
