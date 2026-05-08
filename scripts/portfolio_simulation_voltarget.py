#!/usr/bin/env python
"""Portfolio simulation with vol-targeting and DD control.

Adds to base portfolio_simulation.py:
  - Vol target: scale position size to hit target annualized vol (e.g., 15%)
  - DD cap: if drawdown > threshold, reduce gross exposure
  - Position cap: max % per name (e.g., 5%)

Goal: turn Sharpe 1.06 / MaxDD -42% into Sharpe ~0.9 / MaxDD <-25%.
"""
from __future__ import annotations
import logging, json
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("portfolio-vt")

REPO = Path(__file__).resolve().parent.parent
CUTS = [
    ("2016-01-01", "2018-12-31", "2019-02-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-02-01", "2020-12-31"),
    ("2018-01-01", "2020-12-31", "2021-02-01", "2021-12-31"),
    ("2019-01-01", "2021-12-31", "2022-02-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-02-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-02-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-02-01", "2025-12-31"),
]
LABEL = "fwd_60d_excess"
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":8,"verbosity":0}
N_ROUNDS = 100
REBAL_DAYS = 20
DECILE = 0.10
TC_BPS = 10
VOL_TARGET = 0.15      # target 15% annualized vol
VOL_LOOKBACK = 60      # 60-day rolling vol window
DD_THRESHOLD = -0.10   # if drawdown > 10%, scale down by gradient
MAX_LEVERAGE = 1.5     # cap gross exposure


def daily_returns(ohlcv_dir: Path, ticker: str) -> pd.Series:
    p = ohlcv_dir / ticker / "1d.parquet"
    if not p.exists(): return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df["close"].pct_change()


def train_predict(panel, feat_cols, cut):
    tr_s,tr_e,te_s,te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)]
    if len(tr)<1000 or len(te)<100: return pd.DataFrame()
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    sort_idx = np.argsort(tr["date"].values)
    Xs,ys,ds = Xtr_n[sort_idx], ytr[sort_idx], tr["date"].values[sort_idx]
    _,gsz = np.unique(ds,return_counts=True)
    dtr = xgb.DMatrix(Xs,label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS,dtr,num_boost_round=N_ROUNDS)
    out = te[["date","ticker"]].copy()
    out["pred"] = booster.predict(xgb.DMatrix(Xte_n))
    return out


def simulate_volcontrol(preds, ohlcv_dir, mode="long_only"):
    """Vol-targeted + DD-capped portfolio."""
    if preds.empty: return pd.DataFrame()
    preds = preds.sort_values("date")
    rebal_dates = sorted(preds["date"].unique())[::REBAL_DAYS]
    all_tickers = sorted(preds["ticker"].unique())
    ret_panel = pd.DataFrame({t: daily_returns(ohlcv_dir, t) for t in all_tickers})

    port_records = []
    prev_long, prev_short = set(), set()
    cum_returns = []  # for DD tracking

    for i, rd in enumerate(rebal_dates):
        snap = preds[preds["date"]==rd]
        if len(snap)<20: continue
        snap = snap.sort_values("pred", ascending=False)
        n_top = max(int(len(snap)*DECILE), 5)
        long_set  = set(snap.head(n_top)["ticker"].tolist())
        short_set = set(snap.tail(n_top)["ticker"].tolist()) if mode=="long_short" else set()

        long_change = len(long_set ^ prev_long)
        short_change = len(short_set ^ prev_short)
        turnover = (long_change + short_change) / max(2*(len(long_set)+len(short_set)), 1)

        next_rd = rebal_dates[i+1] if i+1<len(rebal_dates) else preds["date"].max()
        hold_dates = ret_panel.index[(ret_panel.index>rd) & (ret_panel.index<=next_rd)]
        if len(hold_dates)==0: continue

        # Compute realized vol from prior 60 days of UNSCALED portfolio returns
        # Use EQ-weighted long basket as proxy
        if len(cum_returns) >= VOL_LOOKBACK:
            recent = pd.DataFrame(cum_returns[-VOL_LOOKBACK:])
            realized_vol = recent["raw"].std() * np.sqrt(252)
            scale = min(VOL_TARGET / max(realized_vol, 1e-9), MAX_LEVERAGE)
        else:
            scale = 1.0

        # DD control: track drawdown, scale down if breaching threshold
        if cum_returns:
            cum_series = pd.Series([r["scaled"] for r in cum_returns])
            equity = (1 + cum_series).cumprod()
            peak = equity.cummax()
            dd = (equity.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]
            if dd < DD_THRESHOLD:
                # Linear scale-down: at -10% DD scale=1.0, at -25% DD scale=0.0
                dd_scale = max(0.0, 1.0 - (abs(dd) - abs(DD_THRESHOLD)) / 0.15)
                scale *= dd_scale

        for hd in hold_dates:
            day_returns = ret_panel.loc[hd]
            long_mean  = day_returns[list(long_set)].mean() if long_set else 0.0
            short_mean = day_returns[list(short_set)].mean() if short_set else 0.0
            raw = (long_mean - short_mean) if mode=="long_short" else long_mean
            scaled = raw * scale
            cost = (turnover * TC_BPS / 10000) * scale if hd == hold_dates[0] else 0.0
            scaled -= cost
            cum_returns.append({"date": hd, "raw": raw, "scaled": scaled, "scale": scale})
            port_records.append({"date":hd,"ret":scaled,"raw":raw,"scale":scale,"turnover":turnover})
        prev_long, prev_short = long_set, short_set

    return pd.DataFrame(port_records)


def metrics(returns: pd.Series, periods=252):
    if len(returns) < 30:
        return {"sharpe":np.nan,"ann_return":np.nan,"ann_vol":np.nan,"max_dd":np.nan}
    mu = returns.mean()*periods; sd = returns.std()*np.sqrt(periods)
    sharpe = mu/sd if sd>1e-9 else np.nan
    cum = (1+returns).cumprod(); peak = cum.cummax(); dd = (cum-peak)/peak
    return {"sharpe":float(sharpe),"ann_return":float(mu),"ann_vol":float(sd),"max_dd":float(dd.min())}


def main():
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]
    ohlcv_dir = REPO / "data" / "ohlcv"

    log.info("Vol-targeted (target=%.0f%%, lookback=%d) + DD-cap (threshold=%.0f%%) portfolio sim",
             VOL_TARGET*100, VOL_LOOKBACK, abs(DD_THRESHOLD)*100)

    all_lo = []
    for i, cut in enumerate(CUTS, 1):
        log.info("Cut %d", i)
        preds = train_predict(panel, feat_cols, cut)
        if preds.empty: continue
        lo = simulate_volcontrol(preds, ohlcv_dir, mode="long_only")
        if not lo.empty:
            all_lo.append(lo)
            m = metrics(lo.set_index("date")["ret"])
            log.info("  LO vol-targeted  Sharpe=%.2f  Ret=%.1f%%  Vol=%.1f%%  MaxDD=%.1f%%",
                     m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)

    if all_lo:
        concat = pd.concat(all_lo).set_index("date")["ret"].sort_index()
        m = metrics(concat)
        log.info("\n══ AGGREGATE: Long-Only Vol-Targeted (target=%.0f%%, DD-cap=%.0f%%) ══",
                 VOL_TARGET*100, abs(DD_THRESHOLD)*100)
        log.info("Sharpe=%.2f  AnnRet=%.1f%%  AnnVol=%.1f%%  MaxDD=%.1f%%",
                 m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)
        log.info("Compare base (no vol-target): Sharpe=1.06 Ret=34.4% Vol=32.4% MaxDD=-42%")


if __name__ == "__main__":
    main()
