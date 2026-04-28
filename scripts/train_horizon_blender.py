#!/usr/bin/env python
"""Train M2 — regime-conditional learned blender over multi-horizon panels.

Inputs: three panel-LTR + NGBoost artifacts (10d / 20d / 60d) trained by M1.
Output: a small MLP (3 layers, 32 hidden, dropout 0.3) that maps

    [μ_10, σ_10, μ_20, σ_20, μ_60, σ_60, regime_one_hot(4), recent_vol_z]

→ blended (μ, σ) used by Gate B and QP downstream.

Why MLP not just weighted average?
----------------------------------
User spec 2026-04-27: blend weights per regime should be LEARNED from data,
not hand-tuned. The MLP captures non-linear interactions (e.g. "in CHOPPY
when σ_60 spikes, override μ_10 with μ_60") that a linear weighted blend
cannot.

Avoiding leak
-------------
We hold out the last 6 months of the panel as the blender training set.
The 3 horizon panels are trained on the first 24 months only (a separate
side run with --train-end-cutoff). On the 6-month hold-out, all 3 produce
true out-of-sample predictions; the blender then learns from those.

Output artifact: ``artifacts/horizon-blender.json`` with::

    {
      "fitted_at":      "...",
      "horizons":       [10, 20, 60],
      "input_features": [...],
      "mlp_layers":     [hidden sizes],
      "weights":        { layer: <list of lists> },
      "scaler":         { mean: [...], std: [...] },
      "val_mse":        ...,
      "val_ic":         ...,
    }

Inference loader: ``kernel/panel_pipeline/horizon_blender.py``
(loads JSON, exposes ``predict(features) -> (μ, σ)``).

Usage:
    python scripts/train_horizon_blender.py
    python scripts/train_horizon_blender.py --train-end 2025-10-31

NOTE: This is a SCAFFOLD — the actual blender training run is gated on
M1 completion (panel-ltr.{10d,20d,60d}.json all present + each trained
on the cutoff data). Wire-up complete; pre-flight check below verifies
artifacts before doing the heavy work.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ART_DIR = REPO_ROOT / "backtesting" / "renquant_104" / "artifacts"
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-horizon-blender")

REGIMES = ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"]
HORIZONS = [10, 20, 60]


def preflight_artifacts() -> dict[int, dict]:
    """Verify all M1 horizon artifacts exist and are loadable."""
    out = {}
    for h in HORIZONS:
        p = ART_DIR / f"panel-ltr.{h}d.json"
        n = ART_DIR / f"ngboost-head.{h}d.json"
        if not p.exists() or not n.exists():
            log.error("Missing %dd artifacts:  panel=%s  ngboost=%s",
                      h, p.exists(), n.exists())
            return {}
        try:
            panel_meta = json.loads(p.read_text())
            ngb_meta   = json.loads(n.read_text())
            out[h] = {"panel_meta": panel_meta, "ngb_meta": ngb_meta}
            log.info("  %dd OK  panel=%d feats  ngb=%d feats  trained=%s",
                     h, len(panel_meta.get("feature_cols") or []),
                     len(ngb_meta.get("feature_cols") or []),
                     panel_meta.get("trained_date"))
        except Exception as e:
            log.error("Failed loading %dd artifacts: %s", h, e)
            return {}
    return out


def build_holdout_predictions(
    horizon_meta: dict[int, dict],
    train_end_iso: str,
    config: dict,
) -> "tuple[list[dict], list[float]]":
    """Build (X_features, y_targets) for the hold-out blender training set.

    X rows = one per (ticker, date) in the hold-out window.
    Each row's 6 horizon predictions come from running the trained panel
    models on the full panel matrix, sliced to dates >= train_end_iso.

    STUB-1 self-audit fix 2026-04-28: was an empty-return stub with TODO
    comments. Now raises NotImplementedError with a clear message — running
    the script aborts with actionable info instead of silently writing
    nothing. Implementation deferred until M1 produces the 3 horizon panel
    artifacts (panel-ltr.{10d,20d,60d}.json + ngboost-head.{...}.json).
    Tracked in roadmap.md "M2 — Learned regime-conditional blender".
    """
    raise NotImplementedError(
        "M2 blender hold-out predictions not yet implemented. Required steps "
        "(see doc/roadmap.md::M2 spec):\n"
        "  1. Load panel feature matrix via training_panel.panel_frame.build_panel_frame "
        "with sample_end=today, sample_start=train_end_iso.\n"
        "  2. For each horizon h ∈ {10, 20, 60}: load panel-ltr.{h}d.json + "
        "ngboost-head.{h}d.json. Compute panel_score → ApplyNGBoost → "
        "(μ_h, σ_h) for every row.\n"
        "  3. Join with regime time-series from data/runs.alpaca.db::pipeline_runs "
        "to get regime per date.\n"
        "  4. Compute realized_vol_z from SPY rolling vol.\n"
        "  5. Build features + forward_20d_relative_return label rows.\n"
        f"\nM1 horizon meta currently loaded: {sorted(horizon_meta.keys())}d. "
        "Once the 3 panels are trained, replace this NotImplementedError "
        "with the implementation above."
    )


def train_blender(
    X: "list[list[float]]", y: list[float], val_frac: float = 0.2,
) -> dict:
    """Train a small MLP via sklearn. Save weights + scaler."""
    if not X or not y:
        log.error("Empty training set — abort.")
        return {}
    try:
        import numpy as np                               # noqa: PLC0415
        from sklearn.neural_network import MLPRegressor  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler # noqa: PLC0415
        from scipy.stats import spearmanr                # noqa: PLC0415
    except ImportError as e:
        log.error("missing dependency: %s", e)
        return {}

    Xa = np.array(X, dtype=np.float32)
    ya = np.array(y, dtype=np.float32)
    n = len(Xa)
    n_val = max(1, int(n * val_frac))
    Xtr, Xv = Xa[:-n_val], Xa[-n_val:]
    ytr, yv = ya[:-n_val], ya[-n_val:]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xv_s  = scaler.transform(Xv)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,           # L2 reg
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        verbose=False,
    )
    mlp.fit(Xtr_s, ytr)

    # Validation metrics
    yhat_v = mlp.predict(Xv_s)
    val_mse = float(((yv - yhat_v) ** 2).mean())
    val_ic, _ = spearmanr(yhat_v, yv)
    val_ic = float(val_ic) if val_ic is not None else float("nan")

    log.info("MLP trained — val_mse=%.6f  val_ic=%.4f  n_train=%d  n_val=%d",
             val_mse, val_ic, len(Xtr), len(Xv))

    return {
        "weights":         [w.tolist() for w in mlp.coefs_],
        "biases":          [b.tolist() for b in mlp.intercepts_],
        "scaler_mean":     scaler.mean_.tolist(),
        "scaler_std":      scaler.scale_.tolist(),
        "hidden_layers":   list(mlp.hidden_layer_sizes),
        "activation":      mlp.activation,
        "n_train":         len(Xtr),
        "n_val":           len(Xv),
        "val_mse":         val_mse,
        "val_ic":          val_ic,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",  default="renquant_104")
    p.add_argument("--train-end", default="2025-10-31",
                   help="ISO date — panel rows on/after this are blender training set")
    p.add_argument("--out",       default=str(ART_DIR / "horizon-blender.json"))
    args = p.parse_args()

    log.info("=== M2 horizon blender training ===")
    log.info("Pre-flight: verifying M1 horizon artifacts...")
    horizon_meta = preflight_artifacts()
    if not horizon_meta:
        log.error("Pre-flight failed — train M1 horizons first:"
                  "\n  python scripts/train_104.py --skip-baseline --skip-recalibrate "
                  "--skip-acceptance --force --strategy-config-name strategy_config.20d.json"
                  "\n  python scripts/train_104.py --skip-baseline --skip-recalibrate "
                  "--skip-acceptance --force --strategy-config-name strategy_config.60d.json")
        return 1

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config = json.loads((strategy_dir / "strategy_config.json").read_text())
    config["_strategy_dir"] = str(strategy_dir)

    X, y = build_holdout_predictions(horizon_meta, args.train_end, config)
    if not X:
        log.error("Hold-out build returned empty — see TODO in build_holdout_predictions.")
        return 2

    blender = train_blender(X, y)
    if not blender:
        return 3

    out = {
        "fitted_at":      datetime.datetime.utcnow().isoformat(),
        "horizons":       HORIZONS,
        "input_features": (
            [f"mu_{h}d" for h in HORIZONS] +
            [f"sigma_{h}d" for h in HORIZONS] +
            [f"regime_{r}" for r in REGIMES] +
            ["recent_vol_z"]
        ),
        "regime_order":   REGIMES,
        "train_end":      args.train_end,
        **blender,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
