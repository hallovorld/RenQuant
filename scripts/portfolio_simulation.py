#!/usr/bin/env python
"""Portfolio simulation: convert WF IC=+0.066 to actual Sharpe.

Tests both long-only (LO) top-decile and long-short (LS) decile on the
production baseline (R1K + alpha158 + 5-fund + XGB d=5 e=0.05 fwd_60d).

For each WF cut:
  1. Train XGB on train period
  2. For each rebalance date in test period (every 20 trading days):
     - Predict cross-sectional scores
     - Form portfolio: long top 10%, short bottom 10% (or long-only top 10%)
     - Equal-weight within bucket
  3. Hold 20 days, then rebalance
  4. Apply 10bp round-trip transaction cost on turnover
  5. Compute daily portfolio returns from actual ticker returns

Aggregate metrics:
  - Annualized return
  - Annualized vol
  - Sharpe (mean/std × √252)
  - Max drawdown
  - vs SPY benchmark Sharpe in same period
"""
from __future__ import annotations
import logging, json
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("portfolio-sim")

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
REBAL_DAYS = 20      # rebalance every ~month
HOLD_DAYS  = 20      # hold each portfolio 20 days before next rebalance
DECILE = 0.10        # top/bottom decile
TC_BPS = 10          # 10 bp round-trip per turnover unit


def daily_returns_from_close(ohlcv_dir: Path, ticker: str) -> pd.Series:
    """Return daily simple returns indexed by date for one ticker."""
    p = ohlcv_dir / ticker / "1d.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df["close"].pct_change()


def train_and_predict(panel: pd.DataFrame, feat_cols: list[str],
                       cut: tuple) -> pd.DataFrame:
    """Train XGB on cut's train period, predict on test period.
    Returns DataFrame with columns [date, ticker, pred]."""
    tr_s, tr_e, te_s, te_e = cut
    tr = panel[(panel["date"] >= tr_s) & (panel["date"] <= tr_e)].dropna(subset=[LABEL])
    te = panel[(panel["date"] >= te_s) & (panel["date"] <= te_e)]
    if len(tr) < 1000 or len(te) < 100:
        return pd.DataFrame()

    Xtr = tr[feat_cols].fillna(0).values.astype(np.float64)
    ytr = tr[LABEL].clip(-5, 5).values.astype(np.float64)
    Xte = te[feat_cols].fillna(0).values.astype(np.float64)

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    Xtr_n = ((Xtr - mu) / sd).clip(-5, 5)
    Xte_n = ((Xte - mu) / sd).clip(-5, 5)

    sort_idx = np.argsort(tr["date"].values)
    Xs = Xtr_n[sort_idx]; ys = ytr[sort_idx]; ds = tr["date"].values[sort_idx]
    _, gsz = np.unique(ds, return_counts=True)
    dtr = xgb.DMatrix(Xs, label=ys); dtr.set_group(gsz)
    booster = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)

    preds = booster.predict(xgb.DMatrix(Xte_n))
    out = te[["date", "ticker"]].copy()
    out["pred"] = preds
    return out


def simulate_portfolio(preds: pd.DataFrame, ohlcv_dir: Path,
                       mode: str = "long_short") -> pd.DataFrame:
    """Simulate portfolio: rebalance every REBAL_DAYS using top/bottom decile.

    Returns DataFrame with columns [date, daily_return, n_long, n_short, turnover, cost]
    """
    if preds.empty:
        return pd.DataFrame()

    preds = preds.sort_values("date")
    rebal_dates = sorted(preds["date"].unique())[::REBAL_DAYS]
    log.info("  Mode=%s, rebal_dates=%d, hold=%d days, decile=%.0f%%",
             mode, len(rebal_dates), HOLD_DAYS, DECILE*100)

    # Pre-load returns for all tickers in test
    all_tickers = sorted(preds["ticker"].unique())
    ret_panel = pd.DataFrame({t: daily_returns_from_close(ohlcv_dir, t) for t in all_tickers})

    # Build portfolio time series
    port_returns = []  # list of (date, return, n_long, n_short, turnover)
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for i, rd in enumerate(rebal_dates):
        # Get current rebalance prediction
        snap = preds[preds["date"] == rd]
        if len(snap) < 20: continue

        snap = snap.sort_values("pred", ascending=False)
        n_top = max(int(len(snap) * DECILE), 5)
        long_set  = set(snap.head(n_top)["ticker"].tolist())
        short_set = set(snap.tail(n_top)["ticker"].tolist()) if mode == "long_short" else set()

        # Turnover = changed positions / total positions
        long_change  = len(long_set ^ prev_long)
        short_change = len(short_set ^ prev_short)
        total_changes = long_change + short_change
        total_positions = max(len(long_set) + len(short_set), 1)
        turnover = total_changes / (2 * total_positions)  # 0-1 scale

        # Hold period dates: from rd to next rebalance
        next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else preds["date"].max()
        hold_dates = ret_panel.index[(ret_panel.index > rd) & (ret_panel.index <= next_rd)]
        if len(hold_dates) == 0: continue

        # Equal-weight within bucket; daily portfolio return:
        for hd in hold_dates:
            day_returns = ret_panel.loc[hd]
            long_mean  = day_returns[list(long_set)].mean() if long_set else 0.0
            short_mean = day_returns[list(short_set)].mean() if short_set else 0.0
            if mode == "long_short":
                gross = long_mean - short_mean
            else:  # long_only
                gross = long_mean
            cost = (turnover * TC_BPS / 10000) if hd == hold_dates[0] else 0.0
            port_returns.append({"date": hd, "ret": gross - cost,
                                  "n_long": len(long_set), "n_short": len(short_set),
                                  "turnover": turnover, "cost": cost})

        prev_long = long_set
        prev_short = short_set

    return pd.DataFrame(port_returns)


def metrics(returns: pd.Series, periods_per_year: float = 252) -> dict:
    """Annualized Sharpe, return, vol, max drawdown."""
    if len(returns) < 30:
        return {"sharpe": np.nan, "ann_return": np.nan, "ann_vol": np.nan, "max_dd": np.nan}
    mu = returns.mean() * periods_per_year
    sd = returns.std()  * np.sqrt(periods_per_year)
    sharpe = mu / sd if sd > 1e-9 else np.nan
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return {"sharpe": float(sharpe), "ann_return": float(mu),
            "ann_vol": float(sd), "max_dd": float(dd.min())}


def main():
    log.info("Loading R1K + fund baseline panel...")
    panel = pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    excl = {"ticker","date","split_label","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"}
    feat_cols = [c for c in panel.columns if c not in excl]

    ohlcv_dir = REPO / "data" / "ohlcv"

    # SPY benchmark
    spy_ret = daily_returns_from_close(ohlcv_dir, "SPY")

    all_returns_lo = []
    all_returns_ls = []
    cut_metrics = []

    for i, cut in enumerate(CUTS, 1):
        log.info("\n══ Cut %d: train [%s, %s] test [%s, %s] ══", i, *cut)
        preds = train_and_predict(panel, feat_cols, cut)
        if preds.empty:
            log.warning("Cut %d: no predictions", i); continue

        log.info("Simulating long-only top decile...")
        lo_returns = simulate_portfolio(preds, ohlcv_dir, mode="long_only")
        log.info("Simulating long-short decile...")
        ls_returns = simulate_portfolio(preds, ohlcv_dir, mode="long_short")

        # SPY for same test period
        te_s, te_e = cut[2], cut[3]
        spy_te = spy_ret[(spy_ret.index >= te_s) & (spy_ret.index <= te_e)]

        m_lo  = metrics(lo_returns.set_index("date")["ret"]) if not lo_returns.empty else {}
        m_ls  = metrics(ls_returns.set_index("date")["ret"]) if not ls_returns.empty else {}
        m_spy = metrics(spy_te)

        log.info("LO Sharpe=%.2f Ret=%.1f%% Vol=%.1f%% MaxDD=%.1f%%",
                 m_lo.get("sharpe", np.nan),
                 m_lo.get("ann_return", 0)*100, m_lo.get("ann_vol", 0)*100,
                 m_lo.get("max_dd", 0)*100)
        log.info("LS Sharpe=%.2f Ret=%.1f%% Vol=%.1f%% MaxDD=%.1f%%",
                 m_ls.get("sharpe", np.nan),
                 m_ls.get("ann_return", 0)*100, m_ls.get("ann_vol", 0)*100,
                 m_ls.get("max_dd", 0)*100)
        log.info("SPY Sharpe=%.2f Ret=%.1f%%", m_spy.get("sharpe", np.nan),
                 m_spy.get("ann_return", 0)*100)

        cut_metrics.append({"cut": i, "lo": m_lo, "ls": m_ls, "spy": m_spy})
        if not lo_returns.empty: all_returns_lo.append(lo_returns)
        if not ls_returns.empty: all_returns_ls.append(ls_returns)

    # Aggregate across cuts
    log.info("\n══ AGGREGATE METRICS (concatenated test periods) ══")
    if all_returns_lo:
        lo_concat = pd.concat(all_returns_lo).set_index("date")["ret"].sort_index()
        m = metrics(lo_concat)
        log.info("Long-only top decile (TC=10bp):")
        log.info("  Sharpe=%.2f  AnnRet=%.1f%%  AnnVol=%.1f%%  MaxDD=%.1f%%",
                 m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)
    if all_returns_ls:
        ls_concat = pd.concat(all_returns_ls).set_index("date")["ret"].sort_index()
        m = metrics(ls_concat)
        log.info("Long-short decile (TC=10bp):")
        log.info("  Sharpe=%.2f  AnnRet=%.1f%%  AnnVol=%.1f%%  MaxDD=%.1f%%",
                 m["sharpe"], m["ann_return"]*100, m["ann_vol"]*100, m["max_dd"]*100)

    # Per-cut breakdown
    log.info("\n══ PER-CUT SHARPE BREAKDOWN ══")
    log.info("%-3s %-15s %8s %8s %8s", "Cut", "Period", "LO Shp", "LS Shp", "SPY Shp")
    for cm in cut_metrics:
        log.info("%-3d %-15s %+.2f    %+.2f    %+.2f",
                 cm["cut"], f"Cut {cm['cut']}",
                 cm["lo"].get("sharpe", np.nan),
                 cm["ls"].get("sharpe", np.nan),
                 cm["spy"].get("sharpe", np.nan))

    out = REPO / "data" / "portfolio_sim_results.json"
    out.write_text(json.dumps(cut_metrics, indent=2, default=str))
    log.info("Saved per-cut metrics: %s", out)


if __name__ == "__main__":
    main()
