#!/usr/bin/env python
"""F2 — M2 horizon blender v3 (post-audit fixes).

v3 fixes the 5 Lasso bugs identified in the 2026-04-28 audit:

  Fix 1: StandardScaler in sklearn Pipeline so each CV fold scales
         independently (no leakage from full-train standardization).
  Fix 2: Purged time-series CV with embargo = lookahead_days (20 here).
         Panel labels overlap by 20 days — IID K-fold leaks future info
         into training. Reuses training_panel.purged_cv.PurgedKFold.
  Fix 3: ElasticNetCV (l1_ratio ∈ {0.1..1.0}) instead of LassoCV.
         μ_10/μ_20/μ_60 are highly collinear → Lasso arbitrarily zeros
         features (Zou & Hastie 2005).
  Fix 4: Cross-sectional rank target (per-date pct rank). Spearman IC
         on ranks is what we evaluate; train on ranks too — loss-eval
         alignment.
  Fix 5: Winsorize input features at [0.5%, 99.5%] percentiles per
         column + drop inf/nan rows. NGBoost can output extreme σ
         when the feature matrix has out-of-distribution values; v2
         hit a matmul overflow on hold-out from this.

Plus two principled baselines (per CLAUDE.md principle 5.2):
  - Fixed 1/IC weighted blend (no learning, no overfitting)
    Reference: DeMiguel, Garlappi, Uppal 2009 RFS
  - Equal-weight (1/3 each) blend
  - A/A sanity test: shuffle labels, refit, IC should be ≈ 0

Comparison axes (all on the same hold-out):
  - 10d alone, 20d alone, 60d alone (single-horizon baselines)
  - learned blender (ElasticNetCV with the 5 fixes)
  - fixed 1/IC blender
  - equal-weight blender
  - A/A blender (shuffled labels) — must be ≈ 0
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
log = logging.getLogger("blender-v3")

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
HORIZONS = (10, 20, 60)


def _check_artifacts() -> bool:
    for h, a in HORIZON_ARTIFACTS.items():
        for k, p in a.items():
            if not Path(p).exists():
                log.error("Missing %dd %s: %s", h, k, p)
                return False
    return True


def _load_panel_matrix(config: dict, lookahead: int):
    from kernel.data import fetch_ohlcv, LocalStore  # noqa: PLC0415
    from training_panel.pipeline import prepare_inference_panel_frames  # noqa: PLC0415
    from training_panel.panel_frame import build_panel_frame  # noqa: PLC0415

    watchlist = config["watchlist"]
    benchmark = config.get("benchmark", "SPY")
    needed = sorted(set(watchlist) | {benchmark} | set(config.get("sector_etf_map", {}).values()))
    log.info("Loading OHLCV for %d symbols...", len(needed))
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
        log.warning("OHLCV unavailable for %d tickers: %s", len(skipped), skipped[:5])
    log.info("  loaded %d/%d", len(ohlcv), len(needed))

    ticker_sectors = {t: config.get("sector_map", {}).get(t) for t in watchlist}
    cfg_run = {**config, "_strategy_dir": str(STRATEGY_DIR)}
    cfg_run["panel_ltr"] = dict(config.get("panel_ltr", {}))
    cfg_run["panel_ltr"]["lookahead_days"] = lookahead
    cfg_run["panel_ltr"]["asset_embeddings"] = {"enabled": False}

    log.info("Preparing panel frames @ lookahead=%d...", lookahead)
    ff, fac, _macro, _emb = prepare_inference_panel_frames(
        watchlist=watchlist, ohlcv=ohlcv,
        ticker_sectors=ticker_sectors, config=cfg_run,
    )

    import pandas as pd  # noqa: PLC0415
    LOOKAHEAD = lookahead
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


def _score_panel_with_horizons(panel):
    import pandas as pd  # noqa: PLC0415
    from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
    from training_panel.ngboost_head import NGBoostHead  # noqa: PLC0415

    out = panel[["ticker", "date"]].copy()
    for h, art in HORIZON_ARTIFACTS.items():
        log.info("Scoring panel with %dd horizon model...", h)
        scorer = PanelScorer.load(art["panel"])
        head = NGBoostHead.load(art["ngboost"])
        missing = [c for c in scorer.feature_cols if c not in panel.columns]
        if missing:
            log.warning("  %dd: %d feature_cols missing — filling 0", h, len(missing))
            for c in missing:
                panel[c] = 0.0
        X = panel[scorer.feature_cols]
        dist = head.predict_distribution(X)
        out[f"mu_{h}"]    = dist["mu"].values
        out[f"sigma_{h}"] = dist["sigma"].values
    return out


def _attach_labels_and_regime(scored, panel, db_path: Path):
    import pandas as pd  # noqa: PLC0415
    import sqlite3  # noqa: PLC0415

    if "fwd_return" in panel.columns:
        scored["fwd_return"] = panel["fwd_return"].values
    elif "label" in panel.columns:
        log.warning("Using binary 'label' as fwd_return proxy (precision degraded)")
        scored["fwd_return"] = panel["label"].values
    else:
        log.error("Panel missing both fwd_return and label")
        return None

    if not Path(db_path).exists():
        log.warning("runs.alpaca.db not found; regime defaults to BULL_CALM")
        scored["regime"] = "BULL_CALM"
        return scored

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT run_date, regime FROM pipeline_runs "
        "WHERE run_type='live' GROUP BY run_date ORDER BY run_date"
    ).fetchall()
    conn.close()
    regime_map = dict(rows)
    scored["regime"] = pd.to_datetime(scored["date"]).dt.strftime("%Y-%m-%d").map(regime_map)
    n_known = scored["regime"].notna().sum()
    log.info("Regime attached %d/%d rows (%.1f%%)",
             n_known, len(scored), n_known / len(scored) * 100)
    scored["regime"] = scored["regime"].fillna("BULL_CALM")
    return scored


def _winsorize(df, cols, lo_q=0.005, hi_q=0.995):
    """Fix 5: clip features at [0.5%, 99.5%] per column. Returns mutated copy."""
    df = df.copy()
    for c in cols:
        lo = df[c].quantile(lo_q)
        hi = df[c].quantile(hi_q)
        df[c] = df[c].clip(lo, hi)
    return df


def _per_date_rank(s, dates):
    """Fix 4: cross-sectional pct rank within each date.

    Spearman IC = Pearson IC on ranks; training on ranks aligns the
    OLS/ElasticNet objective with the evaluation metric.
    """
    import pandas as pd  # noqa: PLC0415
    return pd.Series(s).groupby(dates).rank(pct=True).values


def _train_blender_v3(scored, hold_out_frac: float, lookahead_days: int = 20):
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from sklearn.linear_model import ElasticNetCV  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from scipy.stats import spearmanr  # noqa: PLC0415
    from training_panel.purged_cv import PurgedKFold  # noqa: PLC0415

    # 1. Drop NaN labels + non-finite per-horizon predictions.
    feat_cols_base = [f"{p}_{h}" for h in HORIZONS for p in ("mu", "sigma")]
    scored = scored.dropna(subset=["fwd_return"]).copy()
    scored = scored[np.isfinite(scored[feat_cols_base].values).all(axis=1)]
    scored = scored[np.isfinite(scored["fwd_return"].values)]
    log.info("Clean rows: %d", len(scored))
    if len(scored) < 1000:
        log.error("Too few clean rows (%d) — abort", len(scored))
        return None

    # Fix 5: winsorize per-horizon (μ, σ).
    scored = _winsorize(scored, feat_cols_base, 0.005, 0.995)

    # Chronological split for hold-out
    scored = scored.sort_values("date").reset_index(drop=True)
    n = len(scored)
    cut = int(n * (1 - hold_out_frac))
    train_df = scored.iloc[:cut].copy()
    hold_df  = scored.iloc[cut:].copy()
    log.info("Train: %d rows  Hold-out: %d rows  (cut=%s)",
             len(train_df), len(hold_df), train_df["date"].iloc[-1])

    # Build feature matrix: 6 base + regime interactions ONLY for regimes
    # with sufficient training samples (≥ 500 rows).
    regime_counts = train_df["regime"].value_counts()
    keep_regimes = [r for r in REGIMES if regime_counts.get(r, 0) >= 500]
    log.info("Regimes with ≥ 500 train rows (kept for interactions): %s", keep_regimes)
    log.info("Regime counts: %s", regime_counts.to_dict())

    def _make_X(df):
        regime_dummies = pd.get_dummies(df["regime"], prefix="regime")
        for r in [f"regime_{rr}" for rr in REGIMES]:
            if r not in regime_dummies.columns:
                regime_dummies[r] = 0
        regime_dummies = regime_dummies[[f"regime_{rr}" for rr in REGIMES]]
        feat_blocks = [df[feat_cols_base].reset_index(drop=True),
                       regime_dummies.reset_index(drop=True)]
        # Interactions only for kept regimes
        for h in HORIZONS:
            for r in keep_regimes:
                feat_blocks.append(pd.Series(
                    df[f"mu_{h}"].values * regime_dummies[f"regime_{r}"].values,
                    name=f"mu_{h}_x_{r}",
                ))
        out = pd.concat(feat_blocks, axis=1)
        return out

    Xtr_df = _make_X(train_df)
    Xho_df = _make_X(hold_df)

    # Fix 4: rank-target conversion (per-date pct rank within train_df).
    train_df["fwd_rank"] = _per_date_rank(train_df["fwd_return"], train_df["date"])
    ytr = train_df["fwd_rank"].values
    # Hold-out: realized fwd_return is the evaluation oracle. We compare
    # blender prediction's rank-correlation to realized fwd_return.
    yho = hold_df["fwd_return"].values

    # Fix 1+2: Pipeline (StandardScaler → ElasticNetCV with PurgedKFold).
    purged = PurgedKFold(
        n_splits=5,
        embargo_days=lookahead_days,
        lookahead_days=lookahead_days,
    )
    cv_splits = list(purged.split(train_df, date_col="date"))
    log.info("Purged CV: %d folds, embargo=%dd, lookahead=%dd",
             len(cv_splits), lookahead_days, lookahead_days)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            cv=cv_splits,
            max_iter=10000,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    log.info("Fitting ElasticNetCV (n_features=%d, n_train=%d) on per-date ranks...",
             Xtr_df.shape[1], Xtr_df.shape[0])
    pipe.fit(Xtr_df.values, ytr)
    enet = pipe.named_steps["enet"]
    log.info("  best alpha=%.6f  l1_ratio=%.2f  nonzero_coefs=%d/%d",
             float(enet.alpha_), float(enet.l1_ratio_),
             int((enet.coef_ != 0).sum()), len(enet.coef_))

    yhat_ho = pipe.predict(Xho_df.values)
    learned_ic = float(spearmanr(yhat_ho, yho).correlation)

    # Single-horizon baselines on the same hold-out
    ic_10  = float(spearmanr(hold_df["mu_10"].values, yho).correlation)
    ic_20  = float(spearmanr(hold_df["mu_20"].values, yho).correlation)
    ic_60  = float(spearmanr(hold_df["mu_60"].values, yho).correlation)

    # Equal-weight baseline
    eq_pred = (hold_df["mu_10"].values + hold_df["mu_20"].values + hold_df["mu_60"].values) / 3.0
    eq_ic = float(spearmanr(eq_pred, yho).correlation)

    # 1/IC weighted baseline (DeMiguel et al. 2009 — but conditional on
    # train-set IC, not naïve 1/N). Train-set IC per horizon:
    train_ic = {
        h: float(spearmanr(train_df[f"mu_{h}"].values,
                            train_df["fwd_return"].values).correlation)
        for h in HORIZONS
    }
    log.info("Train-set IC per horizon: %s",
             {h: round(v, 4) for h, v in train_ic.items()})
    # Skip horizons with non-positive train IC
    pos_ic = {h: v for h, v in train_ic.items() if v > 0}
    if not pos_ic:
        ic_weighted_ic = float("nan")
    else:
        total = sum(pos_ic.values())
        weights = {h: v / total for h, v in pos_ic.items()}
        log.info("1/IC blend weights: %s",
                 {h: round(w, 3) for h, w in weights.items()})
        ic_pred = sum(weights.get(h, 0.0) * hold_df[f"mu_{h}"].values for h in HORIZONS)
        ic_weighted_ic = float(spearmanr(ic_pred, yho).correlation)

    # A/A sanity: shuffle train labels, refit, IC should be ≈ 0
    import numpy as np  # noqa: PLC0415
    rng = np.random.default_rng(42)
    ytr_shuffled = ytr.copy()
    rng.shuffle(ytr_shuffled)
    aa_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            cv=cv_splits,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    aa_pipe.fit(Xtr_df.values, ytr_shuffled)
    aa_yhat = aa_pipe.predict(Xho_df.values)
    aa_ic = float(spearmanr(aa_yhat, yho).correlation)

    # Report
    log.info("=" * 60)
    log.info("RESULTS (hold-out spearman IC vs realized fwd_return):")
    log.info("  Single  10d:           %+.5f", ic_10)
    log.info("  Single  20d:           %+.5f", ic_20)
    log.info("  Single  60d:           %+.5f", ic_60)
    log.info("  Equal-weight blend:    %+.5f", eq_ic)
    log.info("  1/IC weighted blend:   %+.5f", ic_weighted_ic)
    log.info("  Learned ElasticNet:    %+.5f", learned_ic)
    log.info("  A/A (shuffled labels): %+.5f  (must be ≈ 0)", aa_ic)
    log.info("=" * 60)

    return {
        "feature_cols":  list(Xtr_df.columns),
        "n_features":    int(Xtr_df.shape[1]),
        "n_train":       int(len(train_df)),
        "n_holdout":     int(len(hold_df)),
        "kept_regimes":  keep_regimes,
        "lookahead_days": lookahead_days,
        "best_alpha":    float(enet.alpha_),
        "best_l1_ratio": float(enet.l1_ratio_),
        "nonzero_coefs": int((enet.coef_ != 0).sum()),
        "ic": {
            "single_10d":     ic_10,
            "single_20d":     ic_20,
            "single_60d":     ic_60,
            "equal_weight":   eq_ic,
            "ic_weighted":    ic_weighted_ic,
            "learned":        learned_ic,
            "aa_shuffled":    aa_ic,
        },
        "fixes_applied": [
            "1_standard_scaler_in_pipeline",
            "2_purged_cv_with_embargo",
            "3_elastic_net_for_collinearity",
            "4_per_date_rank_target",
            "5_winsorize_features",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",  default="renquant_104")
    p.add_argument("--hold-out-frac", type=float, default=0.25)
    p.add_argument("--lookahead-days", type=int, default=20)
    p.add_argument("--out", default=str(ART_DIR / "horizon-blender-v3.json"))
    args = p.parse_args()

    if not _check_artifacts():
        log.error("One or more horizon artifacts missing; abort")
        return 1

    config = json.loads((STRATEGY_DIR / "strategy_config.json").read_text())
    cfg_b1 = json.loads((STRATEGY_DIR / "strategy_config.20d.json").read_text())
    config["watchlist"] = cfg_b1["watchlist"]
    config["sector_map"] = cfg_b1.get("sector_map", config.get("sector_map", {}))

    panel = _load_panel_matrix(config, lookahead=args.lookahead_days)
    scored = _score_panel_with_horizons(panel)
    scored = _attach_labels_and_regime(
        scored, panel, REPO_ROOT / "data" / "runs.alpaca.db",
    )
    if scored is None:
        return 2

    result = _train_blender_v3(scored,
                                hold_out_frac=args.hold_out_frac,
                                lookahead_days=args.lookahead_days)
    if result is None:
        return 3

    out = {
        "fitted_at":    datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "horizons":     list(HORIZON_ARTIFACTS.keys()),
        "regimes":      REGIMES,
        "method":       "elasticnet_with_5_audit_fixes",
        "hold_out_frac": args.hold_out_frac,
        "fix_references": [
            "Fix 1: sklearn docs — StandardScaler before Lasso family",
            "Fix 2: López de Prado 2018 Ch.7 — Purged K-Fold CV",
            "Fix 3: Zou & Hastie 2005 JRSS-B — ElasticNet for collinearity",
            "Fix 4: Cao et al. 2007 — rank loss alignment for ranking",
            "Fix 5: Standard winsorization — outlier robustness",
        ],
        **result,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
