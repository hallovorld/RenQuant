#!/usr/bin/env python
"""Deprecated standalone scorer for the alpha158+fund+XGB experiment.

End-to-end pipeline:
  1. Load production artifact (panel-ltr-prod-alpha158-fund-fwd60d.json)
  2. For each R1K ticker, compute alpha158 features from local OHLCV
  3. Look up latest SEC fundamentals (point-in-time)
  4. Apply z-score normalization using stored stats
  5. XGB inference → cross-sectional scores
  6. Build top-decile portfolio (long-only, 29 stocks, equal-weight)
  7. Log picks to data/production_runs/{date}.json for tracking

Direct execution is intentionally disabled by default. Live/paper trading must
go through ``python -m live.runner`` so decisions share InferencePipeline, QP
admission, risk gates, and decision_trace DB with sim/LEAN.

Usage:
    # Dry run, just show picks:
    python scripts/production_runner.py

    # Execute paper trade:
    python scripts/production_runner.py --execute --broker alpaca-paper

    # Show comparison to current production picks:
    python scripts/production_runner.py --compare-prod
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from datetime import datetime, date
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prod-runner")


def load_artifact(path: Path) -> dict:
    art = json.loads(path.read_text())
    booster = xgb.Booster()
    booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))
    art["_booster"] = booster
    art["_means"]   = np.array(art["feature_means"])
    art["_stds"]    = np.array(art["feature_stds"])

    # CRITICAL: the model was trained on already-z-scored features (panel build
    # applied per-feature z-score using train-only stats). compute_alpha158_at
    # at inference time produces RAW features. Need to apply panel z-score
    # FIRST before the artifact's per-cut normalization. Load panel z-score stats.
    panel_stats_path = REPO / "data" / "alpha158_qlib_dataset.stats.json"
    if panel_stats_path.exists():
        ps = json.loads(panel_stats_path.read_text())
        # Build map for alpha158 cols only (5 fund cols are in artifact stats only)
        ps_means = dict(zip(ps["feature_cols"], ps["feature_means"]))
        ps_stds  = dict(zip(ps["feature_cols"], ps["feature_stds"]))
        art["_panel_means"] = np.array([ps_means.get(c, 0.0) for c in art["feature_cols"]])
        art["_panel_stds"]  = np.array([ps_stds.get(c, 1.0)  for c in art["feature_cols"]])
        log.info("Loaded panel z-score stats from %s", panel_stats_path.name)
    else:
        art["_panel_means"] = np.zeros(len(art["feature_cols"]))
        art["_panel_stds"]  = np.ones(len(art["feature_cols"]))
        log.warning("No panel stats found — assuming raw features ≈ z-scored")

    log.info("Artifact loaded: %s, %d features, fingerprint=%s",
             art["kind"], len(art["feature_cols"]), art["config_fingerprint"])
    return art


def get_universe(strategy_dir: Path) -> list[str]:
    """Use the production watchlist from strategy_config.golden.json."""
    cfg = json.loads((strategy_dir / "strategy_config.golden.json").read_text())
    return list(cfg.get("watchlist", []))


def compute_features(ticker: str, today: pd.Timestamp,
                     fund_lookup: dict) -> dict | None:
    """Compute all 163 features for one ticker as of today."""
    from kernel.panel_pipeline.alpha158_features import compute_alpha158_at  # noqa
    ohlcv_path = REPO / "data" / "ohlcv" / ticker / "1d.parquet"
    if not ohlcv_path.exists():
        return None
    df = pd.read_parquet(ohlcv_path)
    df.index = pd.to_datetime(df.index)
    df = df[df.index <= today].sort_index()
    if len(df) < 70:
        return None
    feats = compute_alpha158_at(df, today)
    if not feats:
        return None
    # Add fundamentals
    fund = fund_lookup.get(ticker, {})
    for c in ["earnings_yield","book_to_price","gross_profitability","roe","asset_growth"]:
        feats[c] = fund.get(c, np.nan)
    return feats


def build_fund_lookup(today: pd.Timestamp) -> dict:
    """Latest available fund row per ticker as of today (point-in-time)."""
    fund = pd.read_parquet(REPO / "data" / "sec_fundamentals_daily.parquet")
    fund["date"] = pd.to_datetime(fund["date"])
    snap = fund[fund["date"] <= today].sort_values("date").groupby("ticker").tail(1)
    cols = ["earnings_yield","book_to_price","gross_profitability","roe","asset_growth"]
    return {row["ticker"]: {c: row[c] for c in cols if c in row}
            for _, row in snap.iterrows()}


def score_universe(art: dict, universe: list[str], today: pd.Timestamp) -> pd.DataFrame:
    """Run inference on universe; return DataFrame [ticker, pred] sorted high→low."""
    fund_lookup = build_fund_lookup(today)
    log.info("Fund coverage: %d / %d tickers have data as of %s",
             sum(1 for t in universe if t in fund_lookup), len(universe), today.date())

    rows = {}
    for t in universe:
        f = compute_features(t, today, fund_lookup)
        if f: rows[t] = f
    if not rows:
        log.error("No features computed for any ticker")
        return pd.DataFrame()
    log.info("Feature matrix: %d / %d tickers", len(rows), len(universe))

    X = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=art["feature_cols"])
    Xv = X.fillna(0).values.astype(np.float64)
    # Step 1: panel z-score (matches build_alpha158_qlib.py training-time normalization)
    Xv_panel = ((Xv - art["_panel_means"]) / art["_panel_stds"]).clip(-5, 5)
    # Step 2: artifact-level z-score (mostly identity since panel was already z-scored,
    # but kept for fund features which weren't in panel stats)
    Xn = ((Xv_panel - art["_means"]) / art["_stds"]).clip(-5, 5)
    preds = art["_booster"].predict(xgb.DMatrix(Xn))

    out = pd.DataFrame({"ticker": X.index, "pred": preds})
    return out.sort_values("pred", ascending=False).reset_index(drop=True)


def show_picks(scored: pd.DataFrame, top_n: int = 29):
    log.info("\n══ TOP %d LONG PICKS ══", top_n)
    for _, r in scored.head(top_n).iterrows():
        log.info("  %-6s  %+.4f", r["ticker"], r["pred"])
    log.info("\n══ Distribution ══")
    log.info("  min=%+.4f  med=%+.4f  max=%+.4f  spread=%.4f",
             scored["pred"].min(), scored["pred"].median(),
             scored["pred"].max(), scored["pred"].max() - scored["pred"].min())


def execute_alpaca(scored: pd.DataFrame, target_capital: float, broker: str,
                   dry_run: bool = True) -> None:
    """Execute long-only top-decile via Alpaca API. dry_run=True just prints."""
    top = scored.head(29)
    weight_each = 1.0 / len(top)  # equal weight
    target_dollar = target_capital * weight_each

    log.info("\n══ EXECUTION PLAN ══")
    log.info("Broker: %s, Target capital: $%.0f, %d positions, %.1f%% each (~$%.0f)",
             broker, target_capital, len(top), weight_each*100, target_dollar)

    if dry_run:
        log.info("DRY RUN — no orders placed. Use --execute to actually trade.")
        for _, r in top.iterrows():
            log.info("  WOULD BUY  %-6s  pred=%+.4f  target=$%.0f",
                     r["ticker"], r["pred"], target_dollar)
        return

    if broker != "alpaca-paper":
        log.error("Only --broker alpaca-paper supported for safety. Got: %s", broker)
        return

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
    except ImportError:
        log.error("alpaca-py not installed: pip install alpaca-py")
        return

    from dotenv import load_dotenv; load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    api_sec = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_sec:
        log.error("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
        return

    client = TradingClient(api_key, api_sec, paper=True)
    account = client.get_account()
    log.info("Alpaca paper account: equity=$%.0f buying_power=$%.0f",
             float(account.equity), float(account.buying_power))

    # Cancel any existing orders
    try:
        client.cancel_orders()
        log.info("Cancelled existing orders")
    except Exception as e:
        log.warning("Could not cancel orders: %s", e)

    n_filled = 0; n_failed = 0
    for _, r in top.iterrows():
        try:
            req = MarketOrderRequest(symbol=r["ticker"],
                                      notional=round(target_dollar, 2),
                                      side=OrderSide.BUY,
                                      time_in_force=TimeInForce.DAY)
            client.submit_order(req)
            n_filled += 1
        except Exception as e:
            log.warning("  %s order failed: %s", r["ticker"], str(e)[:80])
            n_failed += 1
    log.info("Submitted %d / %d orders (failed: %d)", n_filled, len(top), n_failed)


def save_run_log(scored: pd.DataFrame, today: pd.Timestamp,
                 art_fingerprint: str, broker: str | None) -> Path:
    out_dir = REPO / "data" / "production_runs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{today.date()}.json"
    out.write_text(json.dumps({
        "scoring_date": str(today.date()),
        "artifact_fingerprint": art_fingerprint,
        "n_scored": len(scored),
        "broker": broker,
        "top29": scored.head(29)[["ticker","pred"]].to_dict("records"),
        "pred_stats": {
            "min": float(scored["pred"].min()),
            "median": float(scored["pred"].median()),
            "max": float(scored["pred"].max()),
            "spread": float(scored["pred"].max() - scored["pred"].min()),
        },
    }, indent=2))
    log.info("Run log saved: %s", out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--artifact",
                   default=str(REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"))
    p.add_argument("--strategy-dir",
                   default=str(REPO / "backtesting" / "renquant_104"))
    p.add_argument("--date", help="YYYY-MM-DD; default = latest available")
    p.add_argument("--execute", action="store_true",
                   help="Deprecated direct execution path; disabled by default")
    p.add_argument("--allow-legacy-direct-execution", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--broker", choices=["alpaca-paper", "dry-run"], default="dry-run")
    p.add_argument("--capital", type=float, default=10000.0)
    args = p.parse_args()

    if args.execute and (
        not args.allow_legacy_direct_execution
        or os.getenv("RENQUANT_ALLOW_LEGACY_DIRECT_EXECUTION") != "1"
    ):
        p.error(
            "--execute is disabled for scripts/production_runner.py because it "
            "bypasses live.runner, InferencePipeline, QP admission, risk gates, "
            "and decision_trace DB. Use: python -m live.runner --strategy "
            "renquant_104 --broker alpaca-paper --once"
        )

    art = load_artifact(Path(args.artifact))
    universe = get_universe(Path(args.strategy_dir))
    log.info("Universe: %d tickers from %s", len(universe), Path(args.strategy_dir).name)

    if args.date:
        today = pd.Timestamp(args.date)
    else:
        # Use latest available date in OHLCV
        sample = pd.read_parquet(REPO / "data" / "ohlcv" / universe[0] / "1d.parquet")
        sample.index = pd.to_datetime(sample.index)
        today = sample.index.max()
    log.info("Scoring date: %s", today.date())

    scored = score_universe(art, universe, today)
    if scored.empty:
        log.error("No scores produced"); return

    show_picks(scored)
    save_run_log(scored, today, art["config_fingerprint"], args.broker if args.execute else None)

    if args.execute:
        execute_alpaca(scored, args.capital, args.broker, dry_run=False)
    elif args.broker == "alpaca-paper":
        execute_alpaca(scored, args.capital, args.broker, dry_run=True)


if __name__ == "__main__":
    main()
