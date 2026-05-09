#!/usr/bin/env python
"""B2 — Per-sector model walk-forward IC test (Track 2 next-up by ROI).

Hypothesis: sector is a STATIC label (not slow-moving feature like
GMM regime probabilities), so a per-sector head AVOIDS the
stock-type identification artifact that killed E44 (regime-as-feature
real signal -0.013 after sanity battery).

Train one XGB rank:pairwise model per sector, on that sector's
ticker subset only. Aggregate per-cut IC by averaging across sectors
(cross-sector ensemble).

Pass gate (per CLAUDE.md ic-evaluation-methodology.md):
  Δmean IC > +0.005 (relaxed from +0.01 per user 2026-05-08)
  AND ≥5/7 cuts positive
  AND each sector net-positive (no sector dragging mean down)
  AND §5.2 sanity battery passes (separate run)

Output: data/wf_per_sector.json (per-sector + aggregate metrics)
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("wf-per-sector")

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


def cs_ic(p, a, d):
    df = pd.DataFrame({"p":p,"y":a,"date":d})
    ics = [spearmanr(g["p"], g["y"])[0] for _,g in df.groupby("date") if len(g)>=5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


def wf_xgb_subset(panel, feat_cols, cut, tickers):
    """Train XGB on subset of tickers, return (test_ic, n_train, n_test)."""
    tr_s,tr_e,te_s,te_e = cut
    sub = panel[panel["ticker"].isin(tickers)]
    tr = sub[(sub["date"]>=tr_s)&(sub["date"]<=tr_e)].dropna(subset=[LABEL])
    te = sub[(sub["date"]>=te_s)&(sub["date"]<=te_e)].dropna(subset=[LABEL])
    if len(tr)<500 or len(te)<50:
        return np.nan, len(tr), len(te)
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
    ic = cs_ic(booster.predict(xgb.DMatrix(Xte_n)), yte, te["date"].values)
    return ic, len(tr), len(te)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    p.add_argument("--config", default="backtesting/renquant_104/strategy_config.json")
    p.add_argument("--out",    default="data/wf_per_sector.json")
    p.add_argument("--min-tickers", type=int, default=5,
                   help="Skip sectors with fewer than N tickers (data sparse)")
    args = p.parse_args()

    log.info("Loading panel + sector map...")
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    cfg = json.loads(Path(args.config).read_text())
    sector_map = cfg.get("sector_map", {})
    log.info("rows=%d tickers=%d sectors=%d sectormap_size=%d",
             len(panel), panel["ticker"].nunique(),
             len(set(sector_map.values())), len(sector_map))

    # Group tickers by sector (only tickers that exist in BOTH the panel + sector_map)
    sector_to_tickers: dict[str, list[str]] = {}
    panel_tickers = set(panel["ticker"].unique())
    for tkr, sec in sector_map.items():
        if tkr in panel_tickers:
            sector_to_tickers.setdefault(sec, []).append(tkr)

    # Filter to sectors with enough tickers
    eligible = {s: ts for s, ts in sector_to_tickers.items() if len(ts) >= args.min_tickers}
    log.info("Eligible sectors (≥%d tickers): %d/%d",
             args.min_tickers, len(eligible), len(sector_to_tickers))
    for s, ts in sorted(eligible.items(), key=lambda x: -len(x[1])):
        log.info("  %-20s  %d tickers", s, len(ts))

    # Per-sector WF + aggregate
    per_sector_results: dict[str, list[float]] = {s: [] for s in eligible}
    cut_avg_ic: list[float] = []
    t0 = time.time()
    for ci, cut in enumerate(CUTS, 1):
        log.info("Cut %d/%d: train=[%s..%s] test=[%s..%s]",
                 ci, len(CUTS), cut[0], cut[1], cut[2], cut[3])
        cut_ics = []
        for sec, tickers in eligible.items():
            ic, n_tr, n_te = wf_xgb_subset(panel, feat_cols, cut, tickers)
            per_sector_results[sec].append(ic)
            if not np.isnan(ic):
                cut_ics.append(ic)
                log.info("    %-20s  IC=%+.4f  (train=%d test=%d)", sec, ic, n_tr, n_te)
            else:
                log.info("    %-20s  SKIP (insufficient data)", sec)
        avg = float(np.mean(cut_ics)) if cut_ics else np.nan
        cut_avg_ic.append(avg)
        log.info("  cut %d cross-sector mean IC = %+.4f (over %d sectors)", ci, avg, len(cut_ics))

    elapsed = time.time() - t0
    log.info("\n══ AGGREGATE (per-sector mean over 7 cuts, %.0fs) ══", elapsed)
    log.info("  %-20s  %8s  %8s  %4s  %s", "sector", "mean_IC", "std_IC", "pos", "per-cut")
    sector_summary = {}
    for sec, ics in sorted(per_sector_results.items()):
        valid = [x for x in ics if not np.isnan(x)]
        if not valid:
            continue
        m = float(np.mean(valid)); s = float(np.std(valid))
        pos = sum(1 for x in valid if x > 0)
        log.info("  %-20s  %+.4f  %.4f  %d/%d  [%s]",
                 sec, m, s, pos, len(valid),
                 ", ".join(f"{x:+.3f}" if not np.isnan(x) else "NA" for x in ics))
        sector_summary[sec] = {"mean": m, "std": s, "pos": f"{pos}/{len(valid)}",
                                "per_cut": [None if np.isnan(x) else float(x) for x in ics]}

    # Cross-sector aggregate
    valid_avg = [x for x in cut_avg_ic if not np.isnan(x)]
    cross_mean = float(np.mean(valid_avg)) if valid_avg else np.nan
    cross_std  = float(np.std(valid_avg))  if valid_avg else np.nan
    cross_pos  = sum(1 for x in valid_avg if x > 0)
    log.info("\n  CROSS-SECTOR cross-cut: mean=%+.4f std=%.4f pos=%d/%d",
             cross_mean, cross_std, cross_pos, len(valid_avg))
    log.info("  Production baseline (single global model): mean +0.067 std 0.074 pos 5/7")
    log.info("  Δ vs baseline: %+.4f  → %s",
             cross_mean - 0.067,
             "consider promote (Δ>0.005)" if (cross_mean - 0.067) > 0.005 else "below threshold")

    Path(args.out).write_text(json.dumps({
        "per_sector": sector_summary,
        "cross_sector_per_cut": cut_avg_ic,
        "cross_sector_aggregate": {
            "mean": cross_mean, "std": cross_std,
            "pos": f"{cross_pos}/{len(valid_avg)}",
        },
        "elapsed_sec": elapsed,
    }, indent=2, default=str))
    log.info("Saved → %s", args.out)


if __name__ == "__main__":
    main()
