#!/usr/bin/env python
"""B4 PEAD §5.2 sanity battery (paired baseline vs PEAD-enriched).

Reuses scripts/wf_sanity_paired.py logic but on the PEAD-enriched
panel. The +0.0086 raw IC lift from wf_pead_long_horizon.py must
be validated to be real signal, not stock-type or regime-persistence
artifact (E44 lesson — broadcast features can grow shuffled-label
IC more than they add real alpha).

Outputs paired:
  baseline (alpha158 + 5 fund)               — A/A 3 seeds, shuffle, +60d shift
  candidate (alpha158 + 5 fund + 3 PEAD)     — same 3 tests
  REAL_SIGNAL = AA_mean - shuffle_IC

If real_signal_PEAD > real_signal_baseline → real PEAD alpha → promote.
If real_signal_PEAD < real_signal_baseline → artifact growth → NO-GO.
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Import build_pead_features from B4 main script
from wf_pead_long_horizon import build_pead_features, CUTS, LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-pead-sanity")


def cs_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def run_wf(panel, feat_cols, *, shift_days=0, shuffle=False, seed=42):
    params = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,
              "min_child_weight":50,"subsample":0.7,"colsample_bytree":0.7,
              "nthread":10,"verbosity":0,"seed":seed}
    rng = np.random.default_rng(seed)
    p = panel
    if shift_days:
        p = panel.copy().sort_values(["ticker","date"]).reset_index(drop=True)
        p[LABEL] = p.groupby("ticker")[LABEL].shift(-shift_days)
        p = p.dropna(subset=[LABEL])
    ics = []
    for cut in CUTS:
        tr_s,tr_e,te_s,te_e = cut
        tr = p[(p["date"]>=tr_s)&(p["date"]<=tr_e)].dropna(subset=[LABEL])
        te = p[(p["date"]>=te_s)&(p["date"]<=te_e)].dropna(subset=[LABEL])
        if len(tr)<1000 or len(te)<100:
            ics.append(np.nan); continue
        Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
        ytr = tr[LABEL].clip(-5,5).values.astype(np.float64).copy()
        if shuffle:
            tr_dates = tr["date"].values
            for d in np.unique(tr_dates):
                idx = np.where(tr_dates == d)[0]
                ytr[idx] = rng.permutation(ytr[idx])
        Xte = te[feat_cols].fillna(0).values.astype(np.float64); yte = te[LABEL].values
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-9
        Xtr_n = ((Xtr-mu)/sd).clip(-5,5); Xte_n = ((Xte-mu)/sd).clip(-5,5)
        si = np.argsort(tr["date"].values)
        Xs,ys,ds = Xtr_n[si], ytr[si], tr["date"].values[si]
        _,gsz = np.unique(ds, return_counts=True)
        dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
        booster = xgb.train(params, dtr, num_boost_round=100)
        ics.append(cs_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values))
    return ics


def battery(panel, feat_cols, label):
    log.info("══ %s ══", label)
    log.info("  rows=%d features=%d", len(panel), len(feat_cols))
    out = {}
    aa = []
    for s in (42, 43, 44):
        t0 = time.time()
        ics = run_wf(panel, feat_cols, seed=s)
        m = float(np.nanmean(ics)); aa.append(m)
        log.info("  A/A seed=%d  mean=%+.4f  (%.1fs)", s, m, time.time()-t0)
    out["aa_mean"] = float(np.mean(aa))
    out["aa_std"]  = float(np.std(aa))
    out["aa_seeds"] = aa
    t0 = time.time()
    sh = run_wf(panel, feat_cols, shuffle=True, seed=42)
    out["shuffle_ic"] = float(np.nanmean(sh))
    log.info("  shuffle    mean=%+.4f  (%.1fs)", out["shuffle_ic"], time.time()-t0)
    t0 = time.time()
    ts = run_wf(panel, feat_cols, shift_days=60, seed=42)
    out["timeshift_ic"] = float(np.nanmean(ts))
    log.info("  shift+60d  mean=%+.4f  (%.1fs)", out["timeshift_ic"], time.time()-t0)
    out["real_signal"] = out["aa_mean"] - out["shuffle_ic"]
    log.info("  REAL SIGNAL = %+.4f − %+.4f = %+.4f",
             out["aa_mean"], out["shuffle_ic"], out["real_signal"])
    return out


def main():
    log.info("Loading + computing PEAD features once...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    base_feat = [c for c in panel.columns if c not in excl]
    panel_p = build_pead_features(panel)
    pead_cols = ["days_since_earnings", "pead_signal", "pead_quintile_rank"]
    full_feat = base_feat + pead_cols

    base = battery(panel_p, base_feat,  "BASELINE alpha158+5fund")
    pead = battery(panel_p, full_feat,  "CANDIDATE alpha158+5fund+3PEAD")

    log.info("\n══ PAIRED VERDICT ══")
    log.info("  metric              baseline   PEAD       Δ")
    log.info("  A/A mean IC         %+.4f    %+.4f   %+.4f",
             base["aa_mean"], pead["aa_mean"], pead["aa_mean"]-base["aa_mean"])
    log.info("  A/A std (3 seeds)   %.4f     %.4f    %+.4f",
             base["aa_std"], pead["aa_std"], pead["aa_std"]-base["aa_std"])
    log.info("  shuffle IC          %+.4f    %+.4f   %+.4f",
             base["shuffle_ic"], pead["shuffle_ic"], pead["shuffle_ic"]-base["shuffle_ic"])
    log.info("  time-shift +60d IC  %+.4f    %+.4f   %+.4f",
             base["timeshift_ic"], pead["timeshift_ic"], pead["timeshift_ic"]-base["timeshift_ic"])
    log.info("  REAL SIGNAL         %+.4f    %+.4f   %+.4f",
             base["real_signal"], pead["real_signal"], pead["real_signal"]-base["real_signal"])
    if pead["real_signal"] > base["real_signal"]:
        log.info("  ✓ PEAD adds %+.4f real signal — PROMOTE-CANDIDATE",
                 pead["real_signal"]-base["real_signal"])
    else:
        log.info("  ✗ PEAD does NOT add real signal — NO-GO (artifact growth)")

    Path("data/wf_pead_sanity.json").write_text(
        json.dumps({"baseline": base, "pead": pead}, indent=2))
    log.info("Saved → data/wf_pead_sanity.json")


if __name__ == "__main__":
    main()
