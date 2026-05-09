#!/usr/bin/env python
"""B4 — Long-horizon PEAD features paired WF (E23 fwd_5d → fwd_60d revisit).

E23 (2026-05-02) closed PEAD at fwd_5d as -1.3 σ artifact. Resume
condition was fwd_20d/60d horizon — the post-earnings drift documented
by Bernard-Thomas 1989 and Chan-Jegadeesh-Lakonishok 1996 has its
empirical strongest signal at 30-60d, not 5d.

Production model is now fwd_60d → resume condition matched.

PEAD features (added on top of production alpha158+5fund 163-feature panel):
  days_since_earnings  (capped at 60d, then NaN → 0)
  pead_signal          = surprise_pct × max(0, 1 - days_since/60)
                         (linear decay over 60d window, Bernard-Thomas drift profile)
  pead_quintile_rank   = cross-sectional quintile rank of most-recent surprise_pct

Paired 7-cut WF vs baseline (alpha158+5fund only). Same XGB config as
production. If raw IC lift > +0.005, must run §5.2 sanity battery
BEFORE any promotion (per E44/E25 lesson — broadcast / time-broadcast
features can grow stock-type artifact more than they add real alpha).

Output: data/wf_pead_long_horizon.json
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-pead")

REPO = Path(__file__).resolve().parent.parent
CUTS = [
    ("2016-01-01","2018-12-31","2019-02-01","2019-12-31"),
    ("2017-01-01","2019-12-31","2020-02-01","2020-12-31"),
    ("2018-01-01","2020-12-31","2021-02-01","2021-12-31"),
    ("2019-01-01","2021-12-31","2022-02-01","2022-12-31"),
    ("2020-01-01","2022-12-31","2023-02-01","2023-12-31"),
    ("2021-01-01","2023-12-31","2024-02-01","2024-12-31"),
    ("2022-01-01","2024-12-31","2025-02-01","2025-12-31"),
]
LABEL = "fwd_60d_excess"
PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42}
PEAD_DECAY_DAYS = 60   # Bernard-Thomas drift window


def cs_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def wf_xgb(panel, feat_cols, cut):
    tr_s,tr_e,te_s,te_e = cut
    tr = panel[(panel["date"]>=tr_s)&(panel["date"]<=tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"]>=te_s)&(panel["date"]<=te_e)].dropna(subset=[LABEL])
    if len(tr)<1000 or len(te)<100:
        return np.nan
    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5,5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)
    yte = te[LABEL].values
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
    Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
    si = np.argsort(tr["date"].values)
    Xs,ys,ds = Xtr_n[si], ytr[si], tr["date"].values[si]
    _,gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=100)
    return cs_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values)


def build_pead_features(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, date), attach 3 PEAD features.

    Reads data/earnings_surprise/{ticker}.parquet for each ticker.
    Forward-fills the most recent earnings to subsequent trading days,
    capped at PEAD_DECAY_DAYS (60d).
    """
    earn_dir = REPO / "data" / "earnings_surprise"
    n_with_earn = 0
    out_blocks = []
    for tkr, g in panel.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        ep = earn_dir / f"{tkr}.parquet"
        if not ep.exists():
            g["days_since_earnings"] = np.nan
            g["pead_signal"]         = np.nan
            g["pead_surprise"]       = np.nan
            out_blocks.append(g); continue
        n_with_earn += 1
        earn = pd.read_parquet(ep)
        if "Earnings Date" in earn.index.name or earn.index.name == "Earnings Date":
            earn = earn.reset_index().rename(columns={"Earnings Date": "earnings_date"})
        else:
            earn = earn.reset_index()
            earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
        earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
        earn = earn.sort_values("earnings_date").reset_index(drop=True)

        # For each panel date, find most-recent prior earnings_date
        g_dates = g["date"].values
        e_dates = earn["earnings_date"].values
        e_surps = earn["surprise_pct"].values
        # Use searchsorted to find insertion index of g_date into e_dates
        idxs = np.searchsorted(e_dates, g_dates, side="right") - 1  # last earnings ≤ panel date
        days_since = np.full(len(g), np.nan)
        surprise   = np.full(len(g), np.nan)
        valid_mask = idxs >= 0
        # Use .astype('timedelta64[D]') to convert to integer days
        diff = (g_dates[valid_mask] - e_dates[idxs[valid_mask]]).astype('timedelta64[D]').astype(int)
        days_since[valid_mask] = diff
        surprise[valid_mask]   = e_surps[idxs[valid_mask]]
        # cap at PEAD_DECAY_DAYS — beyond that, no signal
        days_since = np.where(days_since > PEAD_DECAY_DAYS, np.nan, days_since)
        surprise   = np.where(np.isnan(days_since), np.nan, surprise)
        # decay: linear over 60d
        decay = np.where(np.isnan(days_since), 0.0,
                          np.maximum(0.0, 1.0 - days_since / PEAD_DECAY_DAYS))
        signal = surprise * decay
        g["days_since_earnings"] = days_since
        g["pead_signal"]         = signal
        g["pead_surprise"]       = surprise
        out_blocks.append(g)

    log.info("  PEAD coverage: %d/%d tickers had earnings data",
             n_with_earn, panel["ticker"].nunique())
    out = pd.concat(out_blocks, ignore_index=True)

    # Cross-sectional quintile rank of pead_surprise per date (most recent surprise's quintile)
    out["pead_quintile_rank"] = (
        out.groupby("date")["pead_surprise"].rank(pct=True, na_option="keep")
    )

    # Cross-sectional median imputation per date for inference; fall back to zero
    for c in ["days_since_earnings", "pead_signal", "pead_surprise", "pead_quintile_rank"]:
        med = out.groupby("date")[c].transform("median")
        out[c] = out[c].fillna(med)
        out[c] = out[c].fillna(0.0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--out",   default="data/wf_pead_long_horizon.json")
    args = p.parse_args()

    log.info("Loading panel...")
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    base_feat = [c for c in panel.columns if c not in excl]
    log.info("Base panel: rows=%d features=%d tickers=%d",
             len(panel), len(base_feat), panel["ticker"].nunique())

    log.info("Computing PEAD features (3 cols: days_since, signal, quintile_rank)...")
    t0 = time.time()
    panel_p = build_pead_features(panel)
    log.info("  done in %.1fs", time.time() - t0)
    pead_cols = ["days_since_earnings", "pead_signal", "pead_quintile_rank"]
    enriched_feat = base_feat + pead_cols

    log.info("\n══ Baseline (no PEAD) — paired 7-cut WF ══")
    t0 = time.time()
    base_ics = []
    for ci, cut in enumerate(CUTS, 1):
        ic = wf_xgb(panel_p, base_feat, cut)
        base_ics.append(ic)
        log.info("  cut %d  IC=%+.4f", ci, ic)
    base_mean = float(np.nanmean(base_ics))
    log.info("  Baseline aggregate: mean=%+.4f std=%.4f pos=%d/%d  (%.0fs)",
             base_mean, float(np.nanstd(base_ics)),
             sum(1 for x in base_ics if x>0), len(base_ics), time.time()-t0)

    log.info("\n══ PEAD-enriched — paired 7-cut WF ══")
    t0 = time.time()
    pead_ics = []
    for ci, cut in enumerate(CUTS, 1):
        ic = wf_xgb(panel_p, enriched_feat, cut)
        pead_ics.append(ic)
        log.info("  cut %d  IC=%+.4f", ci, ic)
    pead_mean = float(np.nanmean(pead_ics))
    log.info("  PEAD aggregate: mean=%+.4f std=%.4f pos=%d/%d  (%.0fs)",
             pead_mean, float(np.nanstd(pead_ics)),
             sum(1 for x in pead_ics if x>0), len(pead_ics), time.time()-t0)

    delta = pead_mean - base_mean
    log.info("\n══ VERDICT ══")
    log.info("  Δ mean IC = %+.4f  (PEAD %+.4f vs baseline %+.4f)",
             delta, pead_mean, base_mean)
    log.info("  Promotion threshold (user 2026-05-08): Δ > +0.005 + sanity battery")
    if delta > 0.005:
        log.info("  ✓ raw lift clears threshold — RUN §5.2 SANITY BEFORE PROMOTING")
    elif delta > 0.0:
        log.info("  ⚠ marginal — does not clear threshold; document and shelve")
    else:
        log.info("  ✗ negative lift — NO-GO, document with E47")

    Path(args.out).write_text(json.dumps({
        "baseline_per_cut": [None if np.isnan(x) else float(x) for x in base_ics],
        "pead_per_cut":     [None if np.isnan(x) else float(x) for x in pead_ics],
        "baseline_mean": base_mean,
        "pead_mean":     pead_mean,
        "delta":         delta,
        "pead_cols":     pead_cols,
    }, indent=2))
    log.info("Saved → %s", args.out)


if __name__ == "__main__":
    main()
