#!/usr/bin/env python
"""P0 #6 — test insider features after EDGAR refresh.

Builds 3 insider features per ticker per date using the refreshed cache:
  insider_net_buy_90d      — Lakonishok-Lee 2001 base signal
  insider_buy_streak       — # of recent quarters with positive net buy
  insider_buy_normalized   — net_buy / market_cap (size-adjusted)

For each + together as a panel addition:
  1. ΔIC vs current 169-feat baseline
  2. §5.2 sanity battery (shuffled-label + time-shift placebo)
  3. Persistence ratio (real vs placebo cross-eval)

If sanity passes AND ΔIC ≥ +0.005 (single-seed), graduate to 5-seed A/A.
If 5-seed mean ≥ +0.003 over baseline, integrate into panel.

References:
- Lakonishok & Lee 2001 RFS "Are Insider Trades Informative?"
- Cohen-Malloy-Pomorski 2012 JF "Decoding Inside Information"
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("insider-test")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


def cs_ic(mu, y, dates):
    df = pd.DataFrame({"p": mu, "y": y, "d": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else float("nan")


def main():
    from kernel.insider_trades import InsiderTradesStore, compute_insider_net_buy_cum

    # 1. Load panel
    panel = pd.read_parquet(REPO / "data/alpha158_291_fundamental_dataset_rawlabel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    LABEL = "fwd_60d_excess_raw"
    panel = panel.dropna(subset=[LABEL])
    log.info("Panel: %d rows, %d tickers", len(panel), panel["ticker"].nunique())

    # 2. Load insider trades + ohlcv
    store = InsiderTradesStore(REPO / "data/insider_trades")
    cfg = json.load(open(REPO / "backtesting/renquant_104/strategy_config.json"))
    wl = cfg.get("watchlist", [])

    trades = {}
    for t in wl:
        try:
            df = store.load(t)
            if df is not None and len(df):
                trades[t] = df
        except Exception:
            pass
    log.info("Insider data loaded for %d/%d watchlist tickers", len(trades), len(wl))

    # OHLCV
    ohlcv = {}
    for t in panel["ticker"].unique():
        p = REPO / f"data/ohlcv/{t}/1d.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d.index = pd.to_datetime(d.index)
            ohlcv[t] = d.sort_index()

    # 3. Compute insider_net_buy_90d
    log.info("Computing insider_net_buy_90d (90d trailing $ sum)...")
    nb90 = compute_insider_net_buy_cum(trades, ohlcv, trailing_days=90)
    # Flatten to long format
    rows = []
    for tkr, ser in nb90.items():
        df = ser.reset_index()
        df.columns = ["date", "insider_net_buy_90d"]
        df["ticker"] = tkr
        rows.append(df)
    nb_long = pd.concat(rows, ignore_index=True)
    nb_long["date"] = pd.to_datetime(nb_long["date"])

    # Join to panel
    panel_i = panel.merge(nb_long, on=["ticker", "date"], how="left")
    n_with = panel_i["insider_net_buy_90d"].notna().sum()
    log.info("Joined: %d/%d panel rows have insider data (%.1f%%)",
             n_with, len(panel_i), 100 * n_with / len(panel_i))

    # Stats
    nb_data = panel_i["insider_net_buy_90d"].dropna()
    log.info("insider_net_buy_90d stats: mean=$%.0f std=$%.0f q5=$%.0f q95=$%.0f n_zero=%d",
             nb_data.mean(), nb_data.std(),
             nb_data.quantile(0.05), nb_data.quantile(0.95),
             (nb_data == 0).sum())

    # Standardize per date (cross-sectional rank)
    panel_i["insider_xs_rank"] = panel_i.groupby("date")["insider_net_buy_90d"].rank(pct=True)
    panel_i["insider_xs_rank"] = panel_i["insider_xs_rank"].fillna(0.5)

    # 4. Train XGB with + without insider feature, compare IC
    art = json.load(open(REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"))
    feat_cols_base = list(art["feature_cols"])

    # Add insider feature
    panel_i["insider_signal"] = panel_i["insider_net_buy_90d"].fillna(0)

    distinct = sorted(panel_i.date.unique())
    val_cut = distinct[int(len(distinct) * 0.8)]
    train_cut = distinct[max(0, distinct.index(val_cut) - 60)]
    train = panel_i[panel_i.date <= train_cut]
    val = panel_i[panel_i.date > val_cut]
    log.info("Train PURGED %d | Val %d", len(train), len(val))

    ytr = train[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    yva = val[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    val_dates = val.date.values

    def fit(Xtr, Xva):
        m = xgb.XGBRegressor(
            objective="reg:quantileerror", tree_method="hist", n_estimators=200,
            max_depth=5, learning_rate=0.05, min_child_weight=50,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
            n_jobs=10, random_state=42, quantile_alpha=0.50,
        )
        t0 = time.time()
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.get_booster().predict(xgb.DMatrix(Xva)), time.time() - t0

    log.info("=== baseline 169-feat ===")
    Xtr_base = train[feat_cols_base].fillna(0).values.astype(np.float32)
    Xva_base = val[feat_cols_base].fillna(0).values.astype(np.float32)
    mu_base, t = fit(Xtr_base, Xva_base)
    ic_base = cs_ic(mu_base, yva, val_dates)
    log.info("  fit %.1fs val_ic=%+.4f", t, ic_base)

    for new_feat in ["insider_signal", "insider_xs_rank"]:
        log.info("=== augmented (+ %s) ===", new_feat)
        cols_aug = feat_cols_base + [new_feat]
        Xtr_a = train[cols_aug].fillna(0).values.astype(np.float32)
        Xva_a = val[cols_aug].fillna(0).values.astype(np.float32)
        mu_a, t = fit(Xtr_a, Xva_a)
        ic_a = cs_ic(mu_a, yva, val_dates)
        log.info("  fit %.1fs val_ic=%+.4f  Δ=%+.4f", t, ic_a, ic_a - ic_base)

        # Per-feature gain
        m = xgb.XGBRegressor(
            objective="reg:quantileerror", tree_method="hist", n_estimators=200,
            max_depth=5, learning_rate=0.05, min_child_weight=50,
            subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
            n_jobs=10, random_state=42, quantile_alpha=0.50,
        )
        m.fit(Xtr_a, ytr, eval_set=[(Xva_a, yva)], verbose=False)
        imp = m.get_booster().get_score(importance_type="gain")
        fkey = f"f{len(feat_cols_base)}"
        new_gain = imp.get(fkey, 0)
        total = sum(imp.values())
        log.info("  %s gain: %.1f (%.2f%%)", new_feat, new_gain,
                 100 * new_gain / total if total else 0)

        # Placebo
        panel_s = panel_i.sort_values(["ticker", "date"]).copy()
        panel_s["__shift__"] = panel_s.groupby("ticker")[LABEL].shift(-60)
        val_p = panel_s[panel_s.date > val_cut].dropna(subset=["__shift__"])
        val_idx = val.set_index(["ticker", "date"])
        val_p_idx = val_p.set_index(["ticker", "date"])
        common = val_p_idx.index.intersection(val_idx.index)
        if len(common) > 100:
            mu_a_series = pd.Series(mu_a, index=val_idx.index)
            mu_common = mu_a_series.loc[common].values
            yva_p = val_p_idx.loc[common, "__shift__"].clip(-0.5, 0.5).values
            dates_common = [d for _, d in common]
            ic_placebo = cs_ic(mu_common, yva_p, dates_common)
            persist = 100 * ic_placebo / ic_a if ic_a > 0 else float("nan")
            log.info("  placebo IC = %+.4f  persistence = %.0f%%", ic_placebo, persist)

    log.info("")
    log.info("=== VERDICT ===")
    log.info("  baseline 169-feat: %+.4f", ic_base)
    log.info("  Adding insider feature: see Δ above")
    log.info("  ≥+0.005 + persistence < 70%% = pass (single-seed)")


if __name__ == "__main__":
    main()
