#!/usr/bin/env python
"""Compute daily portfolio risk metrics from pipeline_runs history.

Target-critical: user goal is golden Sharpe=2.0 + APY=1.41. This script
computes the Sharpe / vol / drawdown / VaR / beta series so progress
toward the goal is measurable in the database.

Reads `pipeline_runs.portfolio_value` time series grouped by
`(run_type, strategy)`, builds a daily-return series, and writes
rolling metrics to `portfolio_daily_metrics`.

Usage::

    python scripts/compute_portfolio_metrics.py               # live DB
    python scripts/compute_portfolio_metrics.py --source sim  # sim DB
    python scripts/compute_portfolio_metrics.py --since 2024-01-01
"""
from __future__ import annotations

import argparse
import datetime
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("portfolio-metrics")

TRADING_DAYS_PER_YEAR = 252
# Risk-free rate. User's goal is Sharpe=2.0 absolute (likely raw Sharpe); we
# compute BOTH raw and excess-of-rf for flexibility. Default 0 = raw.
DEFAULT_RF_ANNUAL = 0.0


def _build_returns_df(rows: list, rf_annual: float) -> "pd.DataFrame":
    """Turn (run_date, portfolio_value) tuples into a returns frame.

    Collapses to one row per run_date (in case of multiple runs same day,
    take the LAST portfolio_value — latest snapshot wins).
    """
    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(rows, columns=["run_date", "portfolio_value"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    df = df.sort_values("run_date").groupby("run_date").last().reset_index()
    df["daily_return"] = df["portfolio_value"].pct_change()
    # Excess over risk-free (daily rf)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    df["excess_return"] = df["daily_return"] - rf_daily
    return df


def _rolling_sharpe(returns: "pd.Series", window: int) -> "pd.Series":
    """Annualized Sharpe over a rolling window."""
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=1)
    # Guard against sigma == 0
    sr_daily = mu / sigma.where(sigma > 0)
    return sr_daily * math.sqrt(TRADING_DAYS_PER_YEAR)


def _rolling_vol(returns: "pd.Series", window: int) -> "pd.Series":
    """Annualized stdev over a rolling window."""
    return returns.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _rolling_max_drawdown(portfolio: "pd.Series", window: int) -> "pd.Series":
    """Max peak-to-trough drawdown over a rolling window."""
    rolling_peak = portfolio.rolling(window, min_periods=1).max()
    dd = (portfolio - rolling_peak) / rolling_peak
    # Rolling MIN of the drawdown series = worst drawdown seen in window
    return dd.rolling(window, min_periods=1).min()


def _rolling_var(returns: "pd.Series", window: int, pct: float) -> "pd.Series":
    """Empirical VaR at percentile pct (e.g. 0.05 → 95% VaR)."""
    return returns.rolling(window).quantile(pct)


def _rolling_beta_spy(port_ret: "pd.Series", spy_ret: "pd.Series",
                      window: int) -> "pd.Series":
    """Rolling OLS beta of portfolio returns vs SPY returns."""
    cov = port_ret.rolling(window).cov(spy_ret)
    spy_var = spy_ret.rolling(window).var(ddof=1)
    return cov / spy_var.where(spy_var > 0)


def _load_spy_returns(cache_root: Path) -> "pd.Series":
    """Load SPY closes from parquet + compute daily returns."""
    import pandas as pd  # noqa: PLC0415
    path = cache_root / "SPY" / "1d.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df["close"].pct_change().rename("spy_ret")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant-104",
                   help="pipeline_runs.strategy filter (e.g. 'renquant-104'). "
                        "NOTE: this is the model_name config value (dash), "
                        "not the directory name (underscore).")
    p.add_argument("--strategy-dir", default="renquant_104",
                   help="Backtesting directory name for kernel imports.")
    p.add_argument("--source", choices=["live", "sim"], default="live")
    p.add_argument("--db", default=None, help="Explicit path; overrides --source.")
    p.add_argument("--broker", default="alpaca",
                   choices=["alpaca", "alpaca_paper", "paper", "ibkr"],
                   help="Broker tag for live runs.db lookup. 2026-05-09 fix: "
                        "live data lives in data/runs.{broker}.db, not data/runs.db.")
    p.add_argument("--since", type=lambda s: datetime.date.fromisoformat(s),
                   default=None)
    p.add_argument("--rf-annual", type=float, default=DEFAULT_RF_ANNUAL,
                   help="Annual risk-free rate for excess-return Sharpe.")
    p.add_argument("--cache-root", default="data/ohlcv")
    args = p.parse_args()

    if args.db is None:
        if args.source == "sim":
            args.db = "data/sim_runs.db"
        else:
            # 2026-05-09 audit fix: live data is broker-tagged
            # (kernel/state_paths.py runs_db_path convention). Pre-fix this
            # defaulted to data/runs.db which has 0 live rows after broker
            # isolation switch — silently producing empty portfolio_daily_metrics.
            sys.path.insert(0, str(REPO_ROOT / "backtesting" / args.strategy_dir))
            from kernel.state_paths import runs_db_path  # noqa: PLC0415
            args.db = str(runs_db_path("data/runs.db", args.broker).relative_to(REPO_ROOT)
                          if runs_db_path("data/runs.db", args.broker).is_absolute()
                          else runs_db_path("data/runs.db", args.broker))

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy_dir
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.persistence import get_connection, record_portfolio_metrics  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    db_path = REPO_ROOT / args.db
    conn = get_connection({"persistence": {"enabled": True,
                                             "db_path": str(db_path),
                                             "sim_db_path": str(db_path)}})
    if conn is None:
        log.error("Could not open DB at %s", db_path)
        sys.exit(1)

    # Pull (date, portfolio_value) for the requested (source, strategy)
    run_type = "sim" if args.source == "sim" else "live"
    q = """SELECT run_date, portfolio_value FROM pipeline_runs
             WHERE run_type = ? AND portfolio_value IS NOT NULL"""
    params: list = [run_type]
    if args.strategy:
        q += " AND strategy = ?"
        params.append(args.strategy)
    if args.since is not None:
        q += " AND run_date >= ?"
        params.append(args.since.isoformat())
    q += " ORDER BY run_date"
    rows = conn.execute(q, params).fetchall()

    if not rows:
        log.warning("No rows found for run_type=%s strategy=%s — nothing to compute",
                    run_type, args.strategy)
        return

    log.info("Loaded %d pipeline_runs rows", len(rows))

    df = _build_returns_df(rows, args.rf_annual)
    log.info("Returns: %d days from %s to %s",
             len(df), df["run_date"].iloc[0].date(), df["run_date"].iloc[-1].date())

    # Rolling metrics
    df["sharpe_21d"]   = _rolling_sharpe(df["excess_return"],   21)
    df["sharpe_63d"]   = _rolling_sharpe(df["excess_return"],   63)
    df["sharpe_252d"]  = _rolling_sharpe(df["excess_return"],  252)
    df["realized_vol_21d"]  = _rolling_vol(df["daily_return"], 21)
    df["realized_vol_252d"] = _rolling_vol(df["daily_return"], 252)
    df["max_drawdown_252d"] = _rolling_max_drawdown(df["portfolio_value"], 252)
    df["var_95_21d"]  = _rolling_var(df["daily_return"], 21, 0.05)
    df["var_99_21d"]  = _rolling_var(df["daily_return"], 21, 0.01)

    # Beta vs SPY (requires SPY returns)
    spy_ret = _load_spy_returns(REPO_ROOT / args.cache_root)
    if spy_ret is not None:
        spy_aligned = spy_ret.reindex(df["run_date"]).reset_index(drop=True)
        df["beta_spy_252d"] = _rolling_beta_spy(
            df["daily_return"].reset_index(drop=True),
            spy_aligned,
            252,
        )
    else:
        log.warning("SPY parquet missing — skipping beta")
        df["beta_spy_252d"] = None

    # Build payload for record_portfolio_metrics
    payload = [
        {
            "as_of_date":          r.run_date.date(),
            "run_type":            run_type,
            "strategy":            args.strategy,
            "portfolio_value":     r.portfolio_value,
            "daily_return":        r.daily_return,
            "sharpe_21d":          r.sharpe_21d,
            "sharpe_63d":          r.sharpe_63d,
            "sharpe_252d":         r.sharpe_252d,
            "realized_vol_21d":    r.realized_vol_21d,
            "realized_vol_252d":   r.realized_vol_252d,
            "max_drawdown_252d":   r.max_drawdown_252d,
            "var_95_21d":          r.var_95_21d,
            "var_99_21d":          r.var_99_21d,
            "beta_spy_252d":       r.beta_spy_252d,
        }
        for r in df.itertuples()
        if pd.notna(r.daily_return)
    ]
    n = record_portfolio_metrics(conn, payload)
    conn.commit()

    # Summary print: current vs goal
    last = df.iloc[-1]
    print(f"\n── Portfolio metrics summary ({run_type} / {args.strategy}) ──")
    print(f"  Latest date:       {last['run_date'].date()}")
    print(f"  Portfolio value:   ${last['portfolio_value']:,.0f}")
    print(f"  Sharpe (21d):      {last.get('sharpe_21d', float('nan')):.2f}")
    print(f"  Sharpe (63d):      {last.get('sharpe_63d', float('nan')):.2f}")
    print(f"  Sharpe (252d):     {last.get('sharpe_252d', float('nan')):.2f}  "
          f"← target: 2.0")
    print(f"  Vol (21d ann):     {last.get('realized_vol_21d', float('nan')):.2%}")
    print(f"  Max DD (252d):     {last.get('max_drawdown_252d', float('nan')):.2%}")
    print(f"  VaR 95% (21d):     {last.get('var_95_21d', float('nan')):.2%}")
    print(f"  Beta vs SPY:       {last.get('beta_spy_252d', float('nan')):.2f}")

    # Rough annualized return from portfolio_value
    first_val = df["portfolio_value"].iloc[0]
    last_val  = df["portfolio_value"].iloc[-1]
    days_elapsed = (df["run_date"].iloc[-1] - df["run_date"].iloc[0]).days
    if first_val and days_elapsed > 0:
        total_return = (last_val / first_val) - 1
        apy = (last_val / first_val) ** (365.25 / days_elapsed) - 1
        print(f"\n  Total return:      {total_return:+.2%}")
        print(f"  APY:               {apy:+.2%}  ← target: +41%")

    print(f"\n  {n} rows upserted.")


if __name__ == "__main__":
    main()
