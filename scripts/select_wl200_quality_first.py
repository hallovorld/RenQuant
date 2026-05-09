#!/usr/bin/env python
"""P0 #4 — Quality-first watchlist expansion to wl200.

E26 wl183 NO-GO was bottom-up greedy. New approach: select by liquidity +
history + per-ticker Sharpe + sector diversity, then validate the FULL
list with the strategy's walk-forward.

Per literature:
- Grinold-Kahn 1999 §5: IR = IC × √breadth — but only when transfer
  coefficient holds (each new ticker contributes signal, not noise)
- Hou-Xue-Zhang 2020 RFS §6: replication failures concentrate in
  illiquid / small-cap / short-history tickers
- Markowitz 1952 + Sharpe 1964: sector diversification

Selection criteria (all must pass):
  1. Liquidity:      median 60d DV ≥ $50M (paper-tradable at $10k position size)
  2. History:        ≥ 2520 trading days (10y) of OHLCV
  3. Recent Sharpe:  ≥ +0.30 over last 27 months (filter persistent losers)
  4. Realized vol:   60d ann vol ≤ 80% (filter pump-and-dump)
  5. Sector cap:     max 30 per sector (sector_map fallback)

Output: data/wl200_quality_first.json with metadata + ticker list.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wl200")

REPO = Path(__file__).resolve().parent.parent

# Sector mapping — fallback when SEC industry codes aren't available
SECTOR_FALLBACK = {
    'AAPL':'Tech','MSFT':'Tech','NVDA':'Tech','AMZN':'Tech','GOOG':'Tech','META':'Tech',
    'AMD':'Tech','AVGO':'Tech','ANET':'Tech','ASML':'Tech','TSM':'Tech','MU':'Tech',
    'FTNT':'Tech','CRM':'Tech','PANW':'Tech','CRWD':'Tech','NET':'Tech','SNOW':'Tech',
    'DDOG':'Tech','ESTC':'Tech','HUBS':'Tech','MDB':'Tech','NOW':'Tech','PYPL':'Tech',
    'ADBE':'Tech','ADI':'Tech','AMAT':'Tech','APP':'Tech','MRVL':'Tech','QCOM':'Tech',
    'TXN':'Tech','TEAM':'Tech','WDC':'Tech','WDAY':'Tech','ZM':'Tech','ZS':'Tech',
    'SMCI':'Tech','GTLB':'Tech','LRCX':'Tech','MCHP':'Tech','MPWR':'Tech','ON':'Tech',
    'INTC':'Tech','INTU':'Tech','SOFI':'Fin','HOOD':'Fin','COIN':'Fin','PLTR':'Tech',
    'NFLX':'Tech','SPOT':'Tech','GOOGL':'Tech','RBLX':'Tech','UBER':'Tech','LYFT':'Tech',
    'AFRM':'Fin','SQ':'Fin','XYZ':'Fin',
    'XOM':'Energy','CVX':'Energy','OXY':'Energy','EOG':'Energy','SLB':'Energy',
    'JPM':'Fin','GS':'Fin','BLK':'Fin','MA':'Fin','V':'Fin','AXP':'Fin','SPGI':'Fin',
    'BAC':'Fin','C':'Fin','MS':'Fin','WFC':'Fin','SCHW':'Fin','PNC':'Fin',
    'JNJ':'Health','PFE':'Health','UNH':'Health','LLY':'Health','MRK':'Health','TMO':'Health',
    'ABBV':'Health','BMY':'Health','ABT':'Health','MDT':'Health','GILD':'Health','AMGN':'Health',
    'CVS':'Health','CI':'Health','HUM':'Health','REGN':'Health',
    'CAT':'Indust','HON':'Indust','RTX':'Indust','LMT':'Indust','BA':'Indust','GE':'Indust',
    'DE':'Indust','MMM':'Indust','EMR':'Indust','UNP':'Indust','UPS':'Indust','FDX':'Indust',
    'WMT':'Cons','COST':'Cons','HD':'Cons','MCD':'Cons','SBUX':'Cons','NKE':'Cons',
    'PG':'Cons','KO':'Cons','PEP':'Cons','TGT':'Cons','LOW':'Cons','TJX':'Cons',
    'TSLA':'Auto','F':'Auto','GM':'Auto','RIVN':'Auto','LCID':'Auto','NIO':'Auto',
    'GLD':'Commod','SLV':'Commod',
    'NEE':'Util','SO':'Util','DUK':'Util','D':'Util','AEP':'Util',
    'DLR':'REIT','EQIX':'REIT','AMT':'REIT','PSA':'REIT','SPG':'REIT','PLD':'REIT',
    'CMG':'Cons','VRT':'Indust','HPE':'Tech','CSCO':'Tech','ORCL':'Tech','HPQ':'Tech',
    'DELL':'Tech','IBM':'Tech',
}
DEFAULT_SECTOR = 'Other'


def evaluate_ticker(tkr: str, ohlcv_dir: Path) -> dict | None:
    """Compute per-ticker quality metrics. Returns None if ticker fails basic gates."""
    p = ohlcv_dir / tkr / "1d.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty or len(df) < 252:
        return None
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Trim to last 27 months for recent metrics
    last_27mo = df.tail(27 * 21)   # ~21 trading days/month

    # Liquidity: median 60d DV
    if "volume" in df.columns:
        dv = (df["close"] * df["volume"]).tail(60)
        med_dv = float(dv.median()) if len(dv) >= 30 else 0.0
    else:
        med_dv = 0.0

    # History
    n_days = len(df)

    # Realized 60d ann vol
    rets = df["close"].pct_change().tail(60)
    ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) >= 30 else float("inf")

    # 27-mo Sharpe (annualized, no risk-free rate adjustment)
    rets27 = last_27mo["close"].pct_change().dropna()
    if len(rets27) < 252:
        sharpe27 = 0.0
    else:
        ann_ret = rets27.mean() * 252
        ann_vol27 = rets27.std() * np.sqrt(252)
        sharpe27 = float(ann_ret / ann_vol27) if ann_vol27 > 0 else 0.0

    # 27-mo total return
    ret27 = float(last_27mo["close"].iloc[-1] / last_27mo["close"].iloc[0] - 1) if len(last_27mo) >= 2 else 0.0

    return {
        "ticker": tkr,
        "median_60d_dv": med_dv,
        "n_days": n_days,
        "ann_vol_60d": ann_vol,
        "sharpe_27mo": sharpe27,
        "return_27mo": ret27,
        "sector": SECTOR_FALLBACK.get(tkr, DEFAULT_SECTOR),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-size", type=int, default=200)
    ap.add_argument("--min-dv", type=float, default=50e6, help="median 60d DV ≥ $50M")
    ap.add_argument("--min-days", type=int, default=2520, help="≥ 10y history")
    ap.add_argument("--min-sharpe", type=float, default=0.30, help="≥ 0.30 27mo Sharpe")
    ap.add_argument("--max-vol", type=float, default=0.80, help="≤ 80% ann vol")
    ap.add_argument("--max-per-sector", type=int, default=30)
    ap.add_argument("--include-current", action="store_true",
                    help="Force-include current wl103 (default True)")
    ap.add_argument("--out", default=str(REPO / "data/wl200_quality_first.json"))
    args = ap.parse_args()

    ohlcv_dir = REPO / "data/ohlcv"
    candidates = sorted(p.name for p in ohlcv_dir.iterdir() if p.is_dir())
    log.info("Universe: %d tickers in OHLCV cache", len(candidates))

    cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.json").read_text())
    wl103 = cfg.get("watchlist", [])
    log.info("Current wl103: %d tickers", len(wl103))

    # Evaluate all candidates
    log.info("Evaluating %d candidates...", len(candidates))
    rows = []
    for i, tkr in enumerate(candidates):
        res = evaluate_ticker(tkr, ohlcv_dir)
        if res is not None:
            rows.append(res)
        if (i + 1) % 500 == 0:
            log.info("  progress %d/%d", i + 1, len(candidates))

    df = pd.DataFrame(rows)
    log.info("Evaluated %d eligible tickers", len(df))

    # Apply filters
    log.info("=== Filter cascade ===")
    n0 = len(df)
    df = df[df["n_days"] >= args.min_days]
    log.info("  history ≥ %d days: %d → %d", args.min_days, n0, len(df))
    n1 = len(df)
    df = df[df["median_60d_dv"] >= args.min_dv]
    log.info("  median DV ≥ $%.0fM: %d → %d", args.min_dv / 1e6, n1, len(df))
    n2 = len(df)
    df = df[df["ann_vol_60d"] <= args.max_vol]
    log.info("  ann vol ≤ %.0f%%: %d → %d", args.max_vol * 100, n2, len(df))
    n3 = len(df)
    df = df[df["sharpe_27mo"] >= args.min_sharpe]
    log.info("  27mo Sharpe ≥ %.2f: %d → %d", args.min_sharpe, n3, len(df))

    # Force-include current wl103 (regardless of filters), so backwards compat
    forced = pd.DataFrame(rows).query("ticker in @wl103")
    df = pd.concat([df, forced]).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    log.info("  + force-include wl103: → %d", len(df))

    # Composite score for ranking within remaining
    df["score"] = (
        0.4 * df["sharpe_27mo"].rank(pct=True) +
        0.3 * df["median_60d_dv"].rank(pct=True) +
        0.2 * df["return_27mo"].rank(pct=True) +
        0.1 * (1 - df["ann_vol_60d"].rank(pct=True))   # less vol better
    )

    # Sector cap
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    selected = []
    sector_count: dict[str, int] = {}
    for _, row in df.iterrows():
        sec = row["sector"]
        if sector_count.get(sec, 0) >= args.max_per_sector:
            continue
        selected.append(row.to_dict())
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(selected) >= args.target_size:
            break
    log.info("After sector cap (max %d/sector): %d selected", args.max_per_sector, len(selected))
    log.info("Sector breakdown: %s", sector_count)

    # Write output
    selected_tickers = [r["ticker"] for r in selected]
    delta_in = sorted(set(selected_tickers) - set(wl103))
    delta_out = sorted(set(wl103) - set(selected_tickers))
    log.info("New tickers (wl200 has, wl103 doesn't): %d", len(delta_in))
    log.info("Dropped (wl103 has, wl200 doesn't): %d", len(delta_out))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "watchlist": selected_tickers,
        "metadata": {
            "target_size": args.target_size,
            "min_median_dv_usd": args.min_dv,
            "min_history_days": args.min_days,
            "min_sharpe_27mo": args.min_sharpe,
            "max_ann_vol_60d": args.max_vol,
            "max_per_sector": args.max_per_sector,
            "n_selected": len(selected_tickers),
            "n_added_vs_wl103": len(delta_in),
            "n_dropped_vs_wl103": len(delta_out),
            "sector_breakdown": sector_count,
            "added_tickers": delta_in[:50],   # first 50 for readability
            "dropped_tickers": delta_out,
        },
        "per_ticker_metrics": selected,
    }, indent=2))
    log.info("Saved %s", out_path)
    log.info("")
    log.info("=== NEXT STEPS ===")
    log.info("1. Review the new tickers (%d) — sanity-check selection", len(delta_in))
    log.info("2. Build alpha158+fund panel for wl200 (~30 min)")
    log.info("3. Train + 7-cut walk-forward + §5.2 sanity")
    log.info("4. Compare wl200 WF Sharpe to wl103 baseline +0.40 (today's honest number)")


if __name__ == "__main__":
    main()
