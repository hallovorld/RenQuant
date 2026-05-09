#!/usr/bin/env python
"""D1 — Vol-target v2 sweep.

E43 finding: vol-target (15% target, DD threshold -10%, range -10→-25%)
dropped Sharpe 1.06 → 0.84 because DD cap kicks in too aggressively
in volatile regimes (e.g., 2020 COVID crash).

This sweep tests 4 alternate configurations to find one that
preserves baseline Sharpe while reducing MaxDD:

  v2a — vol-target only, no DD cap          (skip the DD trigger entirely)
  v2b — wider DD trigger:  -15% → -35%      (give room before scaling down)
  v2c — wider DD trigger:  -20% → -40%      (even wider — ride bigger DDs)
  v2d — higher vol target: 20%, DD -15%→-35% (less vol scaling)

Same panel + 7-cut WF + same TC as scripts/portfolio_simulation.py.

Output: data/portfolio_sim_voltarget_v2.json (per-config aggregate metrics)
"""
from __future__ import annotations
import logging, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from portfolio_simulation import (
    CUTS, REBAL_DAYS, DECILE, TC_BPS,
    daily_returns_from_close, metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("voltarget-v2")

LABEL = "fwd_60d_excess"
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42}
N_ROUNDS = 100

CONFIGS = {
    "v2a_voltgt_only":      {"vol_target":0.15, "dd_trigger":None, "dd_floor":None, "max_lev":1.5},
    "v2b_wider_dd_15_35":   {"vol_target":0.15, "dd_trigger":-0.15, "dd_floor":-0.35, "max_lev":1.5},
    "v2c_wider_dd_20_40":   {"vol_target":0.15, "dd_trigger":-0.20, "dd_floor":-0.40, "max_lev":1.5},
    "v2d_voltgt20_dd_15_35":{"vol_target":0.20, "dd_trigger":-0.15, "dd_floor":-0.35, "max_lev":1.5},
}
VOL_LOOKBACK = 60


def train_predict(panel, feat_cols, cut):
    tr_s,tr_e,te_s,te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)]
    if len(tr)<1000 or len(te)<100: return pd.DataFrame()
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    mu,sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    si = np.argsort(tr["date"].values)
    Xs,ys,ds = Xtr_n[si], ytr[si], tr["date"].values[si]
    _,gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
    out = te[["date","ticker"]].copy()
    out["pred"] = booster.predict(xgb.DMatrix(Xte_n))
    return out


def simulate(preds, ret_panel, cfg, mode="long_only"):
    if preds.empty: return pd.DataFrame()
    preds = preds.sort_values("date")
    rebal_dates = sorted(preds["date"].unique())[::REBAL_DAYS]

    records = []
    prev_long = set()
    raw_history = []   # for vol calc
    scaled_history = []  # for DD calc

    for i, rd in enumerate(rebal_dates):
        snap = preds[preds["date"]==rd]
        if len(snap)<20: continue
        snap = snap.sort_values("pred", ascending=False)
        n_top = max(int(len(snap)*DECILE), 5)
        long_set = set(snap.head(n_top)["ticker"].tolist())

        long_change = len(long_set ^ prev_long)
        turnover = long_change / max(2*len(long_set), 1)

        next_rd = rebal_dates[i+1] if i+1<len(rebal_dates) else preds["date"].max()
        hold_dates = ret_panel.index[(ret_panel.index>rd) & (ret_panel.index<=next_rd)]
        if len(hold_dates)==0: continue

        # Vol scaling
        if len(raw_history) >= VOL_LOOKBACK:
            recent = pd.Series(raw_history[-VOL_LOOKBACK:])
            realized_vol = recent.std() * np.sqrt(252)
            scale = min(cfg["vol_target"] / max(realized_vol, 1e-9), cfg["max_lev"])
        else:
            scale = 1.0

        # DD scaling
        if cfg["dd_trigger"] is not None and len(scaled_history) >= 5:
            equity = pd.Series(scaled_history).add(1.0).cumprod()
            peak = equity.cummax()
            dd = (equity.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
            if dd < cfg["dd_trigger"]:
                # Linear: at dd_trigger scale=1.0, at dd_floor scale=0.0
                width = max(abs(cfg["dd_floor"] - cfg["dd_trigger"]), 1e-9)
                dd_scale = max(0.0, 1.0 - (abs(dd) - abs(cfg["dd_trigger"])) / width)
                scale *= dd_scale

        for hd in hold_dates:
            day_returns = ret_panel.loc[hd]
            long_mean = day_returns[list(long_set)].mean() if long_set else 0.0
            raw = long_mean
            scaled = raw * scale
            cost = (turnover * TC_BPS / 10000) * scale if hd == hold_dates[0] else 0.0
            scaled -= cost
            raw_history.append(raw)
            scaled_history.append(scaled)
            records.append({"date":hd,"ret":scaled,"raw":raw,"scale":scale})
        prev_long = long_set
    return pd.DataFrame(records)


def main():
    log.info("Loading panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    ohlcv_dir = REPO / "data" / "ohlcv"

    # Pre-train all 7 cuts ONCE (re-used across configs)
    log.info("Pre-training all 7 cuts (XGB rankers)...")
    cut_preds = []
    cut_ret_panels = []
    for i, cut in enumerate(CUTS, 1):
        t0 = time.time()
        p = train_predict(panel, feat_cols, cut)
        if p.empty:
            cut_preds.append(None); cut_ret_panels.append(None); continue
        all_tickers = sorted(p["ticker"].unique())
        ret_panel = pd.DataFrame({t: daily_returns_from_close(ohlcv_dir, t) for t in all_tickers})
        cut_preds.append(p); cut_ret_panels.append(ret_panel)
        log.info("  Cut %d trained + ret panel built (%.1fs)", i, time.time()-t0)

    results = {}
    for name, cfg in CONFIGS.items():
        log.info("\n══ %s  cfg=%s ══", name, cfg)
        all_returns = []
        per_cut_dd = []
        for i in range(len(CUTS)):
            if cut_preds[i] is None: continue
            sim = simulate(cut_preds[i], cut_ret_panels[i], cfg, mode="long_only")
            if sim.empty: continue
            r = sim.set_index("date")["ret"]
            m = metrics(r)
            log.info("  Cut %d: Sharpe=%+.2f AnnRet=%+.1f%% AnnVol=%.1f%% MaxDD=%.1f%%",
                     i+1, m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)
            all_returns.append(r)
            per_cut_dd.append(m["max_dd"])
        if all_returns:
            agg_r = pd.concat(all_returns).sort_index()
            agg_r = agg_r[~agg_r.index.duplicated(keep="first")]
            m = metrics(agg_r)
            log.info("  AGGREGATE: Sharpe=%+.2f AnnRet=%+.1f%% AnnVol=%.1f%% MaxDD=%.1f%%",
                     m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)
            results[name] = {**m, "config": cfg, "per_cut_dd_mean": float(np.mean(per_cut_dd))}

    log.info("\n══ FINAL TABLE ══")
    log.info("%-25s %8s %10s %10s %10s", "config", "Sharpe", "AnnRet", "AnnVol", "MaxDD")
    log.info("%-25s %8.2f %10s %10s %10s", "BASELINE_E41", 1.06, "+34.4%", "32.4%", "-42.3%")
    for name, m in results.items():
        log.info("%-25s %+8.2f %+9.1f%% %9.1f%% %9.1f%%",
                 name, m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)
    log.info("%-25s %+8.2f %+9.1f%% %9.1f%% %9.1f%%",
             "(E43_v1_for_ref)", 0.84, 17.9, 21.4, -37.1)

    out = REPO / "data" / "portfolio_sim_voltarget_v2.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
