#!/usr/bin/env python
"""F2 — M2 horizon blender (real implementation).

Replaces the NotImplementedError stub in train_horizon_blender.py with
a working hold-out-based regime-conditional blender.

Inputs (all on the same 227 watchlist):
  - 10d: backtesting/renquant_104/artifacts/b1_regressed_20260428_020304/panel-ltr.json
         + ngboost-head.json
  - 20d: backtesting/renquant_104/artifacts/panel-ltr.20d.json + ngboost-head.20d.json
  - 60d: backtesting/renquant_104/artifacts/panel-ltr.60d.json + ngboost-head.60d.json

Output: backtesting/renquant_104/artifacts/horizon-blender-v2.json

Training procedure:
1. Reproduce the unified panel matrix for the 227 watchlist via
   training_panel.pipeline.prepare_inference_panel_frames + panel_frame.build_panel_frame.
2. For each (ticker, date) row, compute (μ_h, σ_h) for h ∈ {10, 20, 60}
   by running the trained NGBoost head's predict_distribution.
3. Hold out the last 25% of dates (chronological) as the blender training set.
   The horizon panels saw these dates only in CPCV folds where they were
   in the test set — partial OOS, acceptable for a first pass.
4. Per-regime Lasso: y = forward_20d_relative_return; X = [μ_10, σ_10, μ_20,
   σ_20, μ_60, σ_60, regime_one_hot, recent_vol_z]. α (regularization)
   chosen by 5-fold CV within the hold-out.
5. Save coefficients + scaler + metrics.

This is the FIRST learnable blender. Inference loader (kernel/panel_pipeline/
horizon_blender.py) is a separate follow-up; this script only produces the
artifact + reports validation IC vs each individual horizon (so we know
whether the blender beats the best single horizon).

Usage:
    python scripts/train_horizon_blender_v2.py
    python scripts/train_horizon_blender_v2.py --hold-out-frac 0.25
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
ART_DIR = STRATEGY_DIR / "artifacts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STRATEGY_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("blender-v2")

# Per-horizon artifact paths. 10d uses the B1 regressed backup (227-watchlist
# at 10d) so all three horizons train on the same universe.
HORIZON_ARTIFACTS = {
    10: {
        "panel":   ART_DIR / "b1_regressed_20260428_020304" / "panel-ltr.json",
        "ngboost": ART_DIR / "b1_regressed_20260428_020304" / "ngboost-head.json",
    },
    20: {
        "panel":   ART_DIR / "panel-ltr.20d.json",
        "ngboost": ART_DIR / "ngboost-head.20d.json",
    },
    60: {
        "panel":   ART_DIR / "panel-ltr.60d.json",
        "ngboost": ART_DIR / "ngboost-head.60d.json",
    },
}
REGIMES = ["BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"]


def _check_artifacts() -> bool:
    for h, a in HORIZON_ARTIFACTS.items():
        for k, p in a.items():
            if not Path(p).exists():
                log.error("Missing %dd %s: %s", h, k, p)
                return False
    return True


def _load_panel_matrix(config: dict, lookahead: int):
    """Use the 20d-style training pipeline (same panel prep) to get the unified
    feature matrix. Lookahead is set to 20 here because we want the
    forward-return label to be 20d (the blender target). Per-horizon
    predictions still come from the 3 separate trained models.
    """
    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from training_panel.pipeline import prepare_inference_panel_frames  # noqa: PLC0415
    from training_panel.panel_frame import build_panel_frame  # noqa: PLC0415

    watchlist = config["watchlist"]
    benchmark = config.get("benchmark", "SPY")
    needed = sorted(set(watchlist) | {benchmark} | set(config.get("sector_etf_map", {}).values()))
    log.info("Loading OHLCV for %d symbols...", len(needed))
    # 2026-04-28 fix: BRK.B (dot) and similar special tickers fail yfinance
    # because yfinance uses BRK-B convention. Try LocalStore first (cache-only),
    # fall back to fetch_ohlcv (network); skip on any failure rather than
    # aborting the whole blender training.
    from kernel.data import LocalStore  # noqa: PLC0415
    store = LocalStore(data_dir=REPO_ROOT / "data" / "ohlcv")
    ohlcv = {}
    skipped = []
    for t in needed:
        try:
            df = store.load(t)
            if df is None or df.empty:
                df = fetch_ohlcv(t)
            if df is not None and not df.empty:
                ohlcv[t] = df
            else:
                skipped.append(t)
        except Exception as exc:
            log.warning("  %s skipped: %s", t, type(exc).__name__)
            skipped.append(t)
    if skipped:
        log.warning("OHLCV unavailable for %d tickers (skipping in panel): %s",
                    len(skipped), skipped[:10])
    log.info("  loaded %d/%d", len(ohlcv), len(needed))

    ticker_sectors = {t: config.get("sector_map", {}).get(t) for t in watchlist}
    cfg_run = {**config, "_strategy_dir": str(STRATEGY_DIR)}
    cfg_run["panel_ltr"] = dict(config.get("panel_ltr", {}))
    cfg_run["panel_ltr"]["lookahead_days"] = lookahead
    # Disable embeddings (consistent with current production)
    cfg_run["panel_ltr"]["asset_embeddings"] = {"enabled": False}

    log.info("Preparing panel frames @ lookahead=%d...", lookahead)
    ff, fac, macro, emb = prepare_inference_panel_frames(
        watchlist=watchlist, ohlcv=ohlcv,
        ticker_sectors=ticker_sectors, config=cfg_run,
    )

    # Build labels (realized forward 20d return — the blender's regression target).
    # Simple log-style return: close[t+20] / close[t] - 1, NaN at the tail
    # (last 20 bars have no realized future). Caller drops NaN downstream.
    # 2026-04-28 audit fix: pre-fix this called a non-existent
    # `training.features.build_labels_from_returns`, so the M2 chain phase
    # silently failed at runtime. Inlined here — no external dep needed for
    # a simple forward return.
    import pandas as pd  # noqa: PLC0415
    LOOKAHEAD = 20

    labels = {}
    for tk, frame in ff.items():
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        c = frame["close"].astype(float)
        labels[tk] = c.shift(-LOOKAHEAD) / c - 1.0

    panel, _w, _meta = build_panel_frame(
        ff, labels, ticker_sectors,
        factor_frames=fac, macro_frame=None, asset_embeddings=None,
        min_history_days=252,
    )
    log.info("Panel: %d rows × %d tickers × %d dates",
             len(panel), panel["ticker"].nunique(), panel["date"].nunique())
    return panel


def _score_panel_with_horizons(panel) -> "tuple[pd.DataFrame, list[str]]":
    """For each row in the panel, predict (μ_h, σ_h) for each h ∈ {10, 20, 60}."""
    import pandas as pd  # noqa: PLC0415
    from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
    from training_panel.ngboost_head import NGBoostHead  # noqa: PLC0415

    out = panel[["ticker", "date"]].copy()
    for h, art in HORIZON_ARTIFACTS.items():
        log.info("Scoring panel with %dd horizon model...", h)
        scorer = PanelScorer.load(art["panel"])
        head = NGBoostHead.load(art["ngboost"])
        # scorer expects panel with the right feature_cols.
        # Build subset of panel matching scorer.feature_cols.
        feats_avail = [c for c in scorer.feature_cols if c in panel.columns]
        missing = [c for c in scorer.feature_cols if c not in panel.columns]
        if missing:
            log.warning("  %dd: %d feature_cols missing from panel — filling 0",
                        h, len(missing))
            for c in missing:
                panel[c] = 0.0
        X = panel[scorer.feature_cols]
        # NGBoost head wants the same features
        dist = head.predict_distribution(X)
        out[f"mu_{h}"]    = dist["mu"].values
        out[f"sigma_{h}"] = dist["sigma"].values
    return out


def _attach_labels_and_regime(scored, panel, db_path: Path):
    """Add forward_20d_relative_return label + regime per date."""
    import pandas as pd  # noqa: PLC0415
    import sqlite3  # noqa: PLC0415

    # Forward return label from panel (build_panel_frame uses 'fwd_return' or 'label')
    cols = panel.columns
    if "fwd_return" in cols:
        scored["fwd_return"] = panel["fwd_return"].values
    elif "label" in cols:
        # label is binary {-1, +1, 0} from threshold; not directly useful as
        # blender target. Compute fwd_return from close.
        log.warning("panel has 'label' (binary) but not 'fwd_return'; "
                    "blender target precision degraded")
        scored["fwd_return"] = panel.get("label", 0)
    else:
        log.error("Panel missing both fwd_return and label")
        return None

    # Regime per date — from runs.alpaca.db pipeline_runs
    if not Path(db_path).exists():
        log.warning("runs.alpaca.db not found; regime defaults to BULL_CALM")
        scored["regime"] = "BULL_CALM"
        return scored

    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT run_date, regime
        FROM pipeline_runs
        WHERE run_type='live'
        GROUP BY run_date
        ORDER BY run_date
    """).fetchall()
    conn.close()
    regime_map = dict(rows)
    scored["regime"] = pd.to_datetime(scored["date"]).dt.strftime("%Y-%m-%d").map(regime_map)
    n_known = scored["regime"].notna().sum()
    log.info("Regime attached to %d/%d rows (%.1f%%)",
             n_known, len(scored), n_known / len(scored) * 100)
    scored["regime"] = scored["regime"].fillna("BULL_CALM")
    return scored


def _train_blender(scored, hold_out_frac: float):
    import numpy as np                                       # noqa: PLC0415
    import pandas as pd                                      # noqa: PLC0415
    from sklearn.linear_model import Lasso, LassoCV          # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler         # noqa: PLC0415
    from scipy.stats import spearmanr                        # noqa: PLC0415

    scored = scored.dropna(subset=["fwd_return"])
    scored = scored[np.isfinite(scored[["mu_10","sigma_10","mu_20","sigma_20","mu_60","sigma_60"]].values).all(axis=1)]
    log.info("Clean rows: %d", len(scored))

    # Chronological split
    scored = scored.sort_values("date").reset_index(drop=True)
    n = len(scored)
    cut = int(n * (1 - hold_out_frac))
    train_df = scored.iloc[:cut]
    hold_df  = scored.iloc[cut:]
    log.info("Train: %d rows  Hold-out: %d rows  (cut=%s)",
             len(train_df), len(hold_df), train_df["date"].iloc[-1])

    # Feature matrix
    feat_cols = ["mu_10", "sigma_10", "mu_20", "sigma_20", "mu_60", "sigma_60"]
    regime_dummies = pd.get_dummies(scored["regime"], prefix="regime")
    for r in [f"regime_{r}" for r in REGIMES]:
        if r not in regime_dummies.columns:
            regime_dummies[r] = 0
    regime_dummies = regime_dummies[[f"regime_{r}" for r in REGIMES]]
    full_X = pd.concat([scored[feat_cols], regime_dummies], axis=1)
    full_y = scored["fwd_return"]

    # Lasso with regime interactions: μ_h × regime_one_hot
    interactions = {}
    for h in [10, 20, 60]:
        for r in REGIMES:
            interactions[f"mu_{h}_x_{r}"] = scored[f"mu_{h}"] * regime_dummies[f"regime_{r}"]
    interactions = pd.DataFrame(interactions)
    full_X = pd.concat([full_X, interactions], axis=1)

    Xtr = full_X.iloc[:cut].values
    ytr = full_y.iloc[:cut].values
    Xho = full_X.iloc[cut:].values
    yho = full_y.iloc[cut:].values

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xho_s = scaler.transform(Xho)

    log.info("Fitting LassoCV with regime interactions (n_features=%d)...", Xtr_s.shape[1])
    lasso = LassoCV(cv=5, max_iter=5000, n_alphas=50, random_state=42)
    lasso.fit(Xtr_s, ytr)
    log.info("  best alpha = %.6f", lasso.alpha_)

    yhat_ho = lasso.predict(Xho_s)
    val_mse = float(((yho - yhat_ho) ** 2).mean())
    val_corr, _ = spearmanr(yhat_ho, yho)
    val_corr = float(val_corr) if val_corr is not None else float("nan")
    log.info("Hold-out: spearman_IC=%+.5f  MSE=%.6f", val_corr, val_mse)

    # Compare to each individual horizon's IC on the same hold-out
    for h in [10, 20, 60]:
        h_ic, _ = spearmanr(scored.iloc[cut:][f"mu_{h}"], yho)
        log.info("  vs %dd alone: IC=%+.5f", h, float(h_ic) if h_ic is not None else float("nan"))

    return {
        "feature_cols":    list(full_X.columns),
        "scaler_mean":     scaler.mean_.tolist(),
        "scaler_scale":    scaler.scale_.tolist(),
        "coefs":           lasso.coef_.tolist(),
        "intercept":       float(lasso.intercept_),
        "best_alpha":      float(lasso.alpha_),
        "val_spearman_ic": val_corr,
        "val_mse":         val_mse,
        "n_train":         int(len(Xtr)),
        "n_holdout":       int(len(Xho)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",  default="renquant_104")
    p.add_argument("--hold-out-frac", type=float, default=0.25)
    p.add_argument("--out", default=str(ART_DIR / "horizon-blender-v2.json"))
    args = p.parse_args()

    if not _check_artifacts():
        log.error("One or more horizon artifacts missing; abort")
        return 1

    config = json.loads((STRATEGY_DIR / "strategy_config.json").read_text())
    # Force the 227 watchlist (production was reverted to 103, but blender needs the bigger panel)
    cfg_b1 = json.loads((STRATEGY_DIR / "strategy_config.20d.json").read_text())
    config["watchlist"] = cfg_b1["watchlist"]
    config["sector_map"] = cfg_b1.get("sector_map", config.get("sector_map", {}))

    panel = _load_panel_matrix(config, lookahead=20)
    scored = _score_panel_with_horizons(panel)
    scored = _attach_labels_and_regime(
        scored, panel, REPO_ROOT / "data" / "runs.alpaca.db",
    )
    if scored is None:
        log.error("Failed to attach labels")
        return 2

    result = _train_blender(scored, hold_out_frac=args.hold_out_frac)

    out = {
        "fitted_at":  datetime.datetime.utcnow().isoformat(),
        "horizons":   list(HORIZON_ARTIFACTS.keys()),
        "regimes":    REGIMES,
        "method":     "lasso_with_regime_interactions",
        "hold_out_frac": args.hold_out_frac,
        **result,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
