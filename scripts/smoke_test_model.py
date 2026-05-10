#!/usr/bin/env python
"""Daily smoke test for the active model artifact.

Verifies the production artifact loads + scores a synthetic input row +
returns a finite non-NaN score. Designed to run in <5 seconds in
daily_104.sh as a pipeline heartbeat.

This REPLACES the daily retrain (moved to weekly_wf_promote.sh per
audit FIX-C). The smoke test catches:
  - missing artifact (file deleted, path drift)
  - corrupted artifact (JSON malformed, booster non-deserializable)
  - feature schema mismatch (artifact expects N features, can't score)
  - silent-corruption regression (BUG #6 class — μ̂ collapse)

Exits 0 on pass, 1 on any failure (daily_104.sh aborts live trade).

Usage::

    python scripts/smoke_test_model.py --strategy renquant_104
    python scripts/smoke_test_model.py --strategy renquant_104 --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("smoke-test")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    sys.path.insert(0, str(strategy_dir))

    # ── Step 1: Config + artifact path resolution ────────────────────────
    try:
        cfg_path = strategy_dir / "strategy_config.json"
        cfg = json.loads(cfg_path.read_text())
    except Exception as exc:
        log.error("FAIL: config read — %s: %s", type(exc).__name__, exc)
        return 1

    panel_cfg = cfg.get("ranking", {}).get("panel_scoring", {})
    if not panel_cfg.get("enabled", False):
        log.warning("PASS (skipped): panel_scoring.enabled=False")
        return 0
    artifact_rel = panel_cfg.get("artifact_path", "artifacts/panel-ltr.json")
    artifact_path = strategy_dir / artifact_rel
    if not artifact_path.exists():
        log.error("FAIL: artifact not found at %s", artifact_path)
        return 1

    # ── Step 2: Artifact JSON loads + has expected schema ────────────────
    try:
        artifact = json.loads(artifact_path.read_text())
    except Exception as exc:
        log.error("FAIL: artifact JSON parse — %s: %s", type(exc).__name__, exc)
        return 1
    feature_cols = artifact.get("feature_cols", [])
    if not feature_cols:
        log.error("FAIL: artifact has no feature_cols")
        return 1
    if "booster_raw_json" not in artifact and "params" not in artifact:
        log.error("FAIL: artifact has neither booster_raw_json nor params field")
        return 1
    log.info("Artifact loaded: %s  trained=%s  n_feat=%d",
             artifact_path.name, artifact.get("trained_date"), len(feature_cols))

    # ── Step 3: Scorer loads via PanelScorer ─────────────────────────────
    try:
        from kernel.panel_pipeline import PanelScorer  # noqa: PLC0415
        scorer = PanelScorer.load(artifact_path)
    except Exception as exc:
        log.error("FAIL: PanelScorer.load — %s: %s", type(exc).__name__, exc)
        return 1

    # ── Step 4: Score a synthetic row ────────────────────────────────────
    # Build a single-ticker DataFrame with all feature columns set to a
    # deterministic value (column index / 100). The model should produce
    # a finite score.
    try:
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        log.error("FAIL: numpy/pandas import — %s", exc)
        return 1

    # Two synthetic rows (different feature values) so:
    # 1. PanelScorer's diversity guard (post-predict ≥2 finite values) passes
    # 2. We can ALSO assert the model produces DIFFERENT scores for
    #    different inputs (catches BUG #6 μ̂-collapse class regression)
    rng = np.random.default_rng(42)
    test_df = pd.DataFrame(
        rng.standard_normal((2, len(feature_cols))),
        index=["SMOKE_TEST_A", "SMOKE_TEST_B"],
        columns=feature_cols,
    )

    try:
        scores = scorer.score(test_df)
    except Exception as exc:
        log.error("FAIL: scorer.score crash — %s: %s", type(exc).__name__, exc)
        return 1

    if scores is None or len(scores) == 0:
        log.error("FAIL: scorer returned empty result")
        return 1
    score_a = float(scores.iloc[0]) if hasattr(scores, "iloc") else float(scores[0])
    score_b = float(scores.iloc[1]) if hasattr(scores, "iloc") else float(scores[1])
    if not (np.isfinite(score_a) and np.isfinite(score_b)):
        log.error("FAIL: scores non-finite (A=%s B=%s)", score_a, score_b)
        return 1
    # Diversity invariant: different inputs MUST produce different scores
    # (BUG #6 class — μ̂ collapse from identical-input feature_medians_ fill).
    if abs(score_a - score_b) < 1e-9:
        log.error(
            "FAIL: BUG #6 CLASS — scorer produced IDENTICAL scores for "
            "two different random inputs (A=%.6f B=%.6f). μ̂-collapse "
            "regression — investigate immediately.",
            score_a, score_b,
        )
        return 1

    score_val = score_a
    log.info("Smoke test PASS: scored 2 synthetic rows → A=%.6f B=%.6f Δ=%.6f",
             score_a, score_b, abs(score_a - score_b))

    # ── Step 5: Calibrator loads + maps score (if enabled) ──────────────
    cal_cfg = panel_cfg.get("global_calibration", {})
    if cal_cfg.get("enabled", False):
        cal_rel = cal_cfg.get("artifact_path", "artifacts/panel-rank-calibration.json")
        cal_path = strategy_dir / cal_rel
        if not cal_path.exists():
            log.error("FAIL: calibrator artifact missing at %s", cal_path)
            return 1
        try:
            from training_panel.global_calibrator import (  # noqa: PLC0415
                GlobalPanelCalibration,
            )
            cal = GlobalPanelCalibration.load(cal_path)
            prob = cal.calibrate_probability(score_val)
            er = cal.expected_return(score_val)
        except Exception as exc:
            log.error("FAIL: calibrator load+map — %s: %s",
                      type(exc).__name__, exc)
            return 1
        if not (np.isfinite(prob) and np.isfinite(er)):
            log.error("FAIL: calibrator produced non-finite (prob=%s er=%s)",
                      prob, er)
            return 1
        if prob < 0.0 or prob > 1.0:
            log.error("FAIL: calibrated probability out of [0,1] (got %s)", prob)
            return 1
        log.info("Calibrator PASS: score %.4f → P(out)=%.4f  E[R]=%.4f",
                 score_val, prob, er)

    return 0


if __name__ == "__main__":
    sys.exit(main())
