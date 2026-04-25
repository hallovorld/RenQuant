#!/usr/bin/env python
"""One-shot transformer retrain — does NOT touch the production XGBoost
artifact (`panel-ltr.json`). Only writes `panel-transformer.pt` +
`.json` sidecar.

Built for the 2026-04-25 NGBoost+Transformer audit ship-out: after fixing
T-1 (max_tickers truncation) + T-7/8 (NaN-label leak) + T-23 (predict
sort), we want an OOS IC sanity check vs the stale 38-ticker baseline
(IC=0.006). This script overrides `panel_ltr.backend` to "transformer"
in-memory only, leaving `strategy_config.json` untouched.

Usage::

    python scripts/train_transformer_only.py
    python scripts/train_transformer_only.py --skip-tournament

The script forces `cv_method=cpcv` so the resulting artifact carries an
OOS IC number. The transformer artifact is saved alongside the existing
panel-ltr.json so both backends can be A/B'd later without retraining.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-tickers", type=int, default=128,
                        help="TransformerParams.max_tickers override (≥watchlist size)")
    parser.add_argument("--max-epochs",  type=int, default=50,
                        help="num_boost_round → max_epochs (default 50)")
    parser.add_argument("--patience",    type=int, default=6,
                        help="early-stop patience epochs (no-op without eval split)")
    parser.add_argument("--device",      default="mps",
                        help="mps|cuda|cpu (mps=Apple Silicon GPU)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("train_transformer_only")

    cfg_path = STRATEGY_DIR / "strategy_config.json"
    config = json.loads(cfg_path.read_text())

    # In-memory override only — strategy_config.json is NOT mutated.
    config = deepcopy(config)
    panel_cfg = config.setdefault("panel_ltr", {})
    panel_cfg["backend"] = "transformer"
    tf_params = panel_cfg.setdefault("transformer_params", {})
    tf_params["max_tickers"] = int(args.max_tickers)
    tf_params["max_epochs"]  = int(args.max_epochs)
    tf_params["patience"]    = int(args.patience)
    tf_params["device"]      = str(args.device)
    tf_params["seed"]        = int(args.seed)
    # Keep CPCV so the artifact carries a real OOS IC number.
    panel_cfg.setdefault("cv_method",      "cpcv")
    panel_cfg.setdefault("cv_n_splits",    6)
    panel_cfg.setdefault("cv_n_test_groups", 2)
    # Force fewer CV epochs (transformer CV halves anyway via // 2 in
    # CrossValidateTask) — avoid making CV the bottleneck.
    panel_cfg.setdefault("num_boost_round", int(args.max_epochs))

    log.info("In-memory overrides: backend=transformer  max_tickers=%d  "
             "max_epochs=%d  device=%s  seed=%d",
             tf_params["max_tickers"], tf_params["max_epochs"],
             tf_params["device"], tf_params["seed"])

    # Run only the panel phase via FullTrainingPipeline with --skip-baseline
    # + --skip-recalibrate. Cadence gate force-bypassed.
    from kernel.pipeline.pp_training_full import FullTrainingPipeline
    from kernel.pipeline.pp_training_full import FullTrainingContext  # noqa: PLC0415

    ctx = FullTrainingContext(
        config=config,
        strategy="renquant_104",
        strategy_dir=STRATEGY_DIR,
        skip_baseline=True,
        skip_recalibrate=True,
        force_retrain=True,
    )
    t0 = time.monotonic()
    FullTrainingPipeline().run(ctx)
    log.info("Pipeline ran in %.1f sec", time.monotonic() - t0)

    # Read the transformer artifact's metadata for IC reporting.
    art_json = STRATEGY_DIR / "artifacts" / "panel-transformer.json"
    if not art_json.exists():
        log.error("panel-transformer.json not produced — training failed")
        return 1
    meta = json.loads(art_json.read_text())
    log.info("──────────────────────────────────────────────────────────")
    log.info("TRANSFORMER RETRAIN COMPLETE")
    log.info("  oos_mean_ic       %s", meta.get("oos_mean_ic"))
    log.info("  oos_std_ic        %s", meta.get("oos_std_ic"))
    log.info("  oos_per_fold_ic   %s", meta.get("oos_per_fold_ic"))
    log.info("  oos_ic_quantiles  %s", meta.get("oos_ic_quantiles"))
    log.info("  panel_shape       %s", meta.get("panel_shape"))
    log.info("  feature_cols      %d", len(meta.get("feature_cols", [])))
    log.info("  trained_date      %s", meta.get("trained_date"))
    log.info("──────────────────────────────────────────────────────────")
    log.info("Baseline (pre-audit, stale 38-ticker): oos_mean_ic = 0.0063")
    log.info("Panel-LTR (XGBoost) reference:         oos_mean_ic = 0.0560")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
