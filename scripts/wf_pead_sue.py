#!/usr/bin/env python
"""B3-pivot — Standardized Unexpected Earnings (SUE) + surprise momentum.

SimFin / yfinance free-tier don't have historical analyst-revision data
needed for the original B3 plan. Pivot to building revision-LIKE
features from the historical earnings_surprise/ data we DO have.

References:
- Foster, Olsen, Shevlin 1984 "Earnings Releases, Anomalies, and the
  Behavior of Security Returns" (~3500 citations) — defines SUE as
  surprise standardized by historical surprise volatility:
    SUE_t = (EPS_actual_t - EPS_estimate_t) / σ(surprise_t-1, t-2, t-3, t-4)
  Statistically more stable than raw surprise_pct (which mixes scale
  effects across small/large firms).
- Bernard, Thomas 1989 — same drift profile holds for SUE; SUE is
  the academic standard PEAD scoring measure.

Three new features added on top of the production 166-feat panel
(stacked on top of E47 PEAD):
  sue_signal           = SUE × pead_decay (Bernard-Thomas 60d window)
  surprise_momentum    = surprise_t - surprise_(t-1)  (QoQ change in surprise%)
  surprise_streak      = signed consecutive same-direction surprise count

Paired 7-cut WF + §5.2 sanity. If raw IC lifts > +0.005 AND sanity
passes (real_signal up), promote to 169-feat panel.

Output: data/wf_pead_sue.json  (with both raw and sanity)
"""
from __future__ import annotations
import argparse, json, logging, time, sys
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from wf_pead_long_horizon import build_pead_features, CUTS, LABEL  # reuse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-pead-sue")

PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42}
PEAD_DECAY_DAYS = 60
SUE_WINDOW = 4   # quarters of history for SUE std denominator (FOS 1984 standard)


def cs_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def add_sue_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Append sue_signal + surprise_momentum + surprise_streak per (ticker, date).

    Reads earnings_surprise/{ticker}.parquet, computes:
      SUE_t = surprise_t / std(surprise_(t-1)..(t-4))
        — clipped to ±5 to bound outliers
      surprise_momentum = surprise_t - surprise_(t-1)
      surprise_streak = signed consecutive same-sign surprise count
    Then forward-fill with 60d Bernard-Thomas decay (matches PEAD path).
    """
    earn_dir = REPO / "data" / "earnings_surprise"
    n_with_data = 0
    out_blocks = []
    for tkr, g in panel.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        ep = earn_dir / f"{tkr}.parquet"
        if not ep.exists():
            for c in ("sue_signal", "surprise_momentum", "surprise_streak"):
                g[c] = np.nan
            out_blocks.append(g); continue
        n_with_data += 1
        earn = pd.read_parquet(ep).reset_index()
        earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
        earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
        earn = earn.sort_values("earnings_date").reset_index(drop=True)
        # Per-row SUE: rolling std of surprise_pct over prior 4 quarters
        # (excluding current row — point-in-time). FOS 1984 §3.
        s = earn["surprise_pct"].astype(float)
        rolling_std = s.shift(1).rolling(SUE_WINDOW, min_periods=2).std()
        sue_per_event = (s / (rolling_std + 1e-6)).clip(-5, 5)
        # Surprise momentum: change in surprise vs prior quarter
        momentum_per_event = s.diff()
        # Streak: signed consecutive count of same-direction surprises
        sign = np.sign(s).fillna(0).astype(int)
        streak = np.zeros(len(s), dtype=int)
        for i in range(len(s)):
            if i == 0 or sign.iloc[i] == 0 or sign.iloc[i] != sign.iloc[i-1]:
                streak[i] = sign.iloc[i]
            else:
                streak[i] = streak[i-1] + sign.iloc[i]
        # Forward-fill to daily panel — only carry within 60d decay window
        g_dates = g["date"].values
        e_dates = earn["earnings_date"].values
        idxs = np.searchsorted(e_dates, g_dates, side="right") - 1
        days_since = np.full(len(g), np.nan)
        sue = np.full(len(g), np.nan)
        mom = np.full(len(g), np.nan)
        strk = np.full(len(g), np.nan)
        valid = idxs >= 0
        diff = (g_dates[valid] - e_dates[idxs[valid]]).astype('timedelta64[D]').astype(int)
        days_since[valid] = diff
        sue[valid] = sue_per_event.iloc[idxs[valid]].values
        mom[valid] = momentum_per_event.iloc[idxs[valid]].values
        strk[valid] = streak[idxs[valid]]
        # Decay over 60d window
        decay = np.where((np.isnan(days_since)) | (days_since > PEAD_DECAY_DAYS),
                          0.0,
                          np.maximum(0.0, 1.0 - days_since / PEAD_DECAY_DAYS))
        out_of_window = (days_since > PEAD_DECAY_DAYS) | np.isnan(days_since)
        g["sue_signal"]        = np.where(out_of_window, 0.0, sue * decay)
        g["surprise_momentum"] = np.where(out_of_window, 0.0, mom * decay)
        g["surprise_streak"]   = np.where(out_of_window, 0.0, strk * decay)
        out_blocks.append(g)

    log.info("  SUE coverage: %d/%d tickers had earnings data",
             n_with_data, panel["ticker"].nunique())
    out = pd.concat(out_blocks, ignore_index=True)
    # Cross-sectional median imputation per date for inference; final 0
    for c in ("sue_signal", "surprise_momentum", "surprise_streak"):
        med = out.groupby("date")[c].transform("median")
        out[c] = out[c].fillna(med).fillna(0.0)
    return out


def wf_xgb(panel, feat_cols, cut, *, shift_days=0, shuffle=False, seed=42):
    tr_s,tr_e,te_s,te_e = cut
    rng = np.random.default_rng(seed)
    p = panel
    if shift_days:
        p = panel.copy().sort_values(["ticker","date"]).reset_index(drop=True)
        p[LABEL] = p.groupby("ticker")[LABEL].shift(-shift_days)
        p = p.dropna(subset=[LABEL])
    tr = p[(p["date"]>=tr_s)&(p["date"]<=tr_e)].dropna(subset=[LABEL])
    te = p[(p["date"]>=te_s)&(p["date"]<=te_e)].dropna(subset=[LABEL])
    if len(tr)<1000 or len(te)<100: return np.nan
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64).copy()
    if shuffle:
        for d in np.unique(tr["date"].values):
            idx = np.where(tr["date"].values == d)[0]
            ytr[idx] = rng.permutation(ytr[idx])
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    yte = te[LABEL].values
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    si = np.argsort(tr["date"].values)
    Xs,ys,ds = Xtr_n[si], ytr[si], tr["date"].values[si]
    _,gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    params = dict(PARAMS); params["seed"] = seed
    booster = xgb.train(params, dtr, num_boost_round=100)
    return cs_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values)


def battery(panel, feat_cols, label_str):
    log.info("══ %s (n_features=%d) ══", label_str, len(feat_cols))
    aa = []
    for s in (42, 43, 44):
        ics = [wf_xgb(panel, feat_cols, c, seed=s) for c in CUTS]
        m = float(np.nanmean(ics)); aa.append(m)
        log.info("  A/A seed=%d  mean=%+.4f", s, m)
    sh_ics = [wf_xgb(panel, feat_cols, c, shuffle=True, seed=42) for c in CUTS]
    sh_mean = float(np.nanmean(sh_ics))
    log.info("  shuffle    mean=%+.4f", sh_mean)
    return {
        "aa_mean": float(np.mean(aa)),
        "aa_std":  float(np.std(aa)),
        "shuffle_ic": sh_mean,
        "real_signal": float(np.mean(aa)) - sh_mean,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--out",   default="data/wf_pead_sue.json")
    args = p.parse_args()

    log.info("Loading 166-feat production panel + computing SUE features...")
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    base_feat = [c for c in panel.columns if c not in excl]
    t0 = time.time()
    panel_p = add_sue_features(panel)
    log.info("  SUE features added in %.1fs", time.time()-t0)
    sue_cols = ["sue_signal", "surprise_momentum", "surprise_streak"]
    full_feat = base_feat + sue_cols

    base_battery = battery(panel_p, base_feat,  "BASELINE 166-feat (PEAD already inside)")
    full_battery = battery(panel_p, full_feat,  "CANDIDATE 169-feat (+ SUE/momentum/streak)")

    delta_aa   = full_battery["aa_mean"] - base_battery["aa_mean"]
    delta_real = full_battery["real_signal"] - base_battery["real_signal"]
    log.info("\n══ PAIRED VERDICT ══")
    log.info("  metric              base      +SUE      Δ")
    log.info("  A/A mean IC         %+.4f   %+.4f   %+.4f",
             base_battery["aa_mean"], full_battery["aa_mean"], delta_aa)
    log.info("  shuffle IC          %+.4f   %+.4f   %+.4f",
             base_battery["shuffle_ic"], full_battery["shuffle_ic"],
             full_battery["shuffle_ic"] - base_battery["shuffle_ic"])
    log.info("  REAL SIGNAL         %+.4f   %+.4f   %+.4f",
             base_battery["real_signal"], full_battery["real_signal"], delta_real)
    if delta_real > 0.005:
        log.info("  ✓ SUE adds real signal — PROMOTE-CANDIDATE")
    elif delta_real > 0.0:
        log.info("  ⚠ marginal lift — does not clear +0.005 floor")
    else:
        log.info("  ✗ no real signal — NO-GO")

    Path(args.out).write_text(json.dumps({"baseline": base_battery, "sue": full_battery,
                                           "delta_aa": delta_aa, "delta_real": delta_real}, indent=2))
    log.info("Saved → %s", args.out)


if __name__ == "__main__":
    main()
