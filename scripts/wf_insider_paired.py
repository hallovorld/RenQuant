#!/usr/bin/env python
"""B-insider retest — E22 was 44% coverage, now 95%. Paired §5.2 sanity.

E22 (2026-05-02) closed insider features as -0.0008 contribution
(within noise) at 44% coverage. SEC IP throttle has since recovered
and Sunday retrain has populated more tickers — current coverage
83/98 (84.7%) for last 90d activity, with insider data spanning
2022-2026 for most tickers.

References (per CLAUDE.md §5.12):
- Jeng, Metrick & Zeckhauser 2003 "Estimating the Returns to Insider
  Trading" RFS — insider PURCHASES significantly outperform sales as
  predictors. Sales noisy due to liquidity / option-exercise / tax
  motivations.
- Cohen, Malloy & Pomorski 2012 "Decoding Inside Information" JF —
  routine vs opportunistic trades. Opportunistic insider buys carry
  the alpha.

Three new features added on top of the production 169-feat panel
(stacked on top of E47 PEAD + E49 SUE):
  insider_net_dollars_30d_z    — signed $ flow over 30d, robust z-scored
  insider_purchase_count_90d   — count of P transactions in 90d (rare = signal)
  insider_buy_ratio_90d        — buys / |total| dollar ratio in 90d

Output: data/wf_insider_paired.json
"""
from __future__ import annotations
import argparse, json, logging, time, sys
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from wf_pead_long_horizon import CUTS, LABEL  # reuse 7-cut WF protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-insider")

PARAMS = {"objective":"rank:pairwise","eta":0.05,"max_depth":5,"min_child_weight":50,
          "subsample":0.7,"colsample_bytree":0.7,"nthread":10,"verbosity":0,"seed":42}
INSIDER_COLS = ["insider_net_dollars_30d_z", "insider_purchase_count_90d",
                "insider_buy_ratio_90d"]


def cs_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def add_insider_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, date), aggregate insider activity over rolling windows."""
    insider_dir = REPO / "data" / "insider_trades"
    n_with_data = 0
    out_blocks = []
    panel_dates_global = sorted(panel["date"].unique())
    for tkr, g in panel.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        ip = insider_dir / f"{tkr}.parquet"
        if not ip.exists():
            for c in INSIDER_COLS: g[c] = np.nan
            out_blocks.append(g); continue
        n_with_data += 1
        ins = pd.read_parquet(ip)
        ins.index = pd.to_datetime(ins.index)
        ins = ins.sort_index()
        # Per-row aggregation over rolling 30d / 90d windows
        net_dollars_30d = np.zeros(len(g))
        purchase_count_90d = np.zeros(len(g))
        buy_ratio_90d = np.zeros(len(g))
        for i, d in enumerate(g["date"].values):
            d_ts = pd.Timestamp(d)
            # 30d window: signed dollars sum
            mask_30 = (ins.index >= d_ts - pd.Timedelta(days=30)) & (ins.index < d_ts)
            net_30 = float(ins.loc[mask_30, "dollars"].sum()) if mask_30.any() else 0.0
            net_dollars_30d[i] = net_30
            # 90d window: counts + ratio
            mask_90 = (ins.index >= d_ts - pd.Timedelta(days=90)) & (ins.index < d_ts)
            sub = ins.loc[mask_90]
            if len(sub) > 0:
                purchases = sub[sub["tx_code"] == "P"]
                purchase_count_90d[i] = float(len(purchases))
                buy_dollars = float(purchases["dollars"].sum()) if len(purchases) else 0.0
                total_abs = float(sub["dollars"].abs().sum())
                buy_ratio_90d[i] = buy_dollars / max(total_abs, 1e-9)
        g["insider_net_dollars_30d_z"] = net_dollars_30d
        g["insider_purchase_count_90d"] = purchase_count_90d
        g["insider_buy_ratio_90d"] = buy_ratio_90d
        out_blocks.append(g)

    log.info("  insider coverage: %d/%d tickers", n_with_data, panel["ticker"].nunique())
    out = pd.concat(out_blocks, ignore_index=True)
    # Robust z-score net_dollars_30d (heavy tail). Median + MAD.
    net = out["insider_net_dollars_30d_z"]
    med, mad = float(net.median()), float((net - net.median()).abs().median())
    scale = max(mad * 1.4826, 1.0)
    out["insider_net_dollars_30d_z"] = ((net - med) / scale).clip(-5, 5).fillna(0.0)
    # Cross-sectional median imputation per date for the count + ratio
    for c in ("insider_purchase_count_90d", "insider_buy_ratio_90d"):
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
    sh = [wf_xgb(panel, feat_cols, c, shuffle=True, seed=42) for c in CUTS]
    sh_mean = float(np.nanmean(sh))
    log.info("  shuffle    mean=%+.4f", sh_mean)
    return {
        "aa_mean":     float(np.mean(aa)),
        "aa_std":      float(np.std(aa)),
        "shuffle_ic":  sh_mean,
        "real_signal": float(np.mean(aa)) - sh_mean,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--out",   default="data/wf_insider_paired.json")
    args = p.parse_args()

    log.info("Loading 169-feat production panel + computing insider features...")
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    base_feat = [c for c in panel.columns if c not in excl]
    t0 = time.time()
    panel_p = add_insider_features(panel)
    log.info("  insider features added in %.1fs", time.time()-t0)
    full_feat = base_feat + INSIDER_COLS

    base = battery(panel_p, base_feat, "BASELINE 169-feat")
    cand = battery(panel_p, full_feat, "CANDIDATE 172-feat (+ 3 insider)")

    delta_real = cand["real_signal"] - base["real_signal"]
    log.info("\n══ PAIRED VERDICT ══")
    log.info("  metric              base      +insider      Δ")
    log.info("  A/A mean IC         %+.4f   %+.4f   %+.4f",
             base["aa_mean"], cand["aa_mean"], cand["aa_mean"]-base["aa_mean"])
    log.info("  shuffle IC          %+.4f   %+.4f   %+.4f",
             base["shuffle_ic"], cand["shuffle_ic"], cand["shuffle_ic"]-base["shuffle_ic"])
    log.info("  REAL SIGNAL         %+.4f   %+.4f   %+.4f",
             base["real_signal"], cand["real_signal"], delta_real)
    if delta_real > 0.005:
        log.info("  ✓ insider adds real signal — PROMOTE-CANDIDATE")
    elif delta_real > 0.0:
        log.info("  ⚠ marginal lift — does not clear +0.005 floor")
    else:
        log.info("  ✗ no real signal — NO-GO")

    Path(args.out).write_text(json.dumps({"baseline": base, "insider": cand,
                                            "delta_real": delta_real}, indent=2))
    log.info("Saved → %s", args.out)


if __name__ == "__main__":
    main()
