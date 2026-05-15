#!/usr/bin/env python3
"""Pre-open cancel gate — cancels pending market orders when the
overnight ES futures move is severe.

Fires ~15 minutes before market open. Designed to prevent strategy
orders queued post-close from filling at catastrophic gap prices
(weekends, overnight macro events).

Scientific basis:
* French-Roll 1986 "Stock Return Variances: The Arrival of Information
  and the Reaction of Traders" — public info accumulates during
  closures; overnight return distribution is materially wider than
  intraday-equivalent.
* Bollerslev-Andersen-Diebold 2008 "Realized Beta" — overnight
  component contributes ~30% of total realized vol; risk management
  must size by full-period σ, not intraday-only.
* Shewhart 1931 statistical process control — 2σ control-limit is
  the canonical "alert threshold" for distinguishing common-cause
  variation from special-cause; corresponds to ~2.3% one-sided
  exceedance probability under normal-ish distributions.

Threshold: σ-normalized severity. The bar is "absolute overnight
move ≥ 2 × σ_60d_overnight". Default 2.0σ chosen per Shewhart; tunable
via --severity-threshold-sigma.

Data source: yfinance ES=F (E-mini S&P 500 futures) — free, no API
keys required. Falls back to ^GSPC daily bars if ES=F is unavailable.

Behavior:
1. Compute overnight ES move + 60d σ of historical overnight returns
2. If |severity| < threshold: PASS, no action
3. If |severity| ≥ threshold: cancel ALL pending market orders on the
   LIVE Alpaca account; ntfy alert; exit 0

Usage::

    python scripts/preopen_cancel_gate.py                       # default 2.0σ
    python scripts/preopen_cancel_gate.py --severity-threshold-sigma 1.5
    python scripts/preopen_cancel_gate.py --dry-run             # diagnose only
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("preopen-gate")


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def compute_overnight_severity(
    *,
    symbol: str = "ES=F",
    fallback_symbol: str = "^GSPC",
    lookback_days: int = 90,
    sigma_window: int = 60,
) -> dict:
    """Return {prior_close, current, current_pct, sigma_60d, severity}.

    severity = current_pct / sigma_60d where σ is the empirical stdev of
    historical overnight returns (close→open) over `sigma_window` bars.

    Raises ValueError if data is unavailable.
    """
    import numpy as np
    import yfinance as yf
    import pandas as pd

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=int(lookback_days * 1.7))

    def _fetch(sym: str):
        try:
            return yf.download(
                sym, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                progress=False, auto_adjust=False,
            )
        except Exception as exc:
            log.warning("yf.download(%s) failed: %s", sym, exc)
            return None

    df = _fetch(symbol)
    used = symbol
    if df is None or df.empty:
        log.warning("Primary symbol %s unavailable; falling back to %s",
                    symbol, fallback_symbol)
        df = _fetch(fallback_symbol)
        used = fallback_symbol
    if df is None or df.empty:
        raise ValueError(
            f"both {symbol} and {fallback_symbol} unavailable from yfinance"
        )

    df = df.sort_index()
    # Yahoo daily bars include Open / Close. Overnight return =
    # (today's Open - prior day's Close) / prior day's Close
    open_series  = df["Open"].astype(float)
    close_series = df["Close"].astype(float)
    if hasattr(open_series, "squeeze"):
        open_series  = open_series.squeeze()
        close_series = close_series.squeeze()
    prior_close = close_series.shift(1)
    overnight = (open_series - prior_close) / prior_close
    overnight = overnight.dropna()
    if len(overnight) < 10:
        raise ValueError(
            f"insufficient overnight history ({len(overnight)} obs)"
        )
    sigma_60d = float(overnight.tail(sigma_window).std())
    latest_close = float(close_series.iloc[-1])
    # "Current" — latest available print. For ES=F this is recent (futures
    # trade 23h/day); for ^GSPC this lags. Use the latest close as the
    # best-available "where we are now" proxy.
    severity = 0.0
    # Guard against degenerate σ (synthetic test data, or genuinely
    # zero-vol windows on holidays). Threshold 1e-8 is below any
    # realistic equity-vol scale (50 bps/day = 0.005 ≫ 1e-8) so
    # legitimate signal is never lost; eps-σ becomes treated as
    # "no information" rather than producing 1e18-scale severity.
    if sigma_60d > 1e-8:
        latest_overnight = float(overnight.iloc[-1])
        severity = latest_overnight / sigma_60d
        current_pct = latest_overnight
    else:
        current_pct = 0.0
    return {
        "source":      used,
        "prior_close": float(prior_close.iloc[-1]) if len(prior_close) else None,
        "latest":      latest_close,
        "current_pct": float(current_pct),
        "sigma_60d":   sigma_60d,
        "severity":    float(severity),
        "n_obs":       int(len(overnight)),
    }


def cancel_stale_market_orders(*, threshold_sigma: float, dry_run: bool) -> dict:
    """Cancel pending market orders if overnight severity ≥ threshold.

    Returns {metrics, cancelled, considered, action}.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderType

    metrics = compute_overnight_severity()
    sev = metrics["severity"]
    pct = metrics["current_pct"]
    sigma = metrics["sigma_60d"]
    log.info(
        "PREOPEN-GATE: %s overnight %+.3f%% (σ_60d=%.3f%%, "
        "severity=%+.2fσ, threshold=±%.1fσ, n_obs=%d)",
        metrics["source"], pct * 100, sigma * 100, sev,
        threshold_sigma, metrics["n_obs"],
    )

    if abs(sev) < threshold_sigma:
        log.info("PREOPEN-GATE: PASS — severity within ±%.1fσ. No action.",
                 threshold_sigma)
        return {"metrics": metrics, "cancelled": [],
                "considered": 0, "action": "pass"}

    client = TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=False,  # LIVE account
    )
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)
    orders = client.get_orders(filter=req)
    # Cancel only MARKET orders queued previously (not stop / limit / today's
    # newly-placed limit-style orders). The strategy uses market orders for
    # both buys and sells, so this targets exactly the daily-cron post-close
    # batch.
    pending_market = [
        o for o in orders
        if str(getattr(o, "order_type", "")) in ("OrderType.MARKET", "market")
    ]
    log.warning(
        "PREOPEN-GATE: TRIGGERED — severity=%+.2fσ ≥ ±%.1fσ; "
        "evaluating %d pending market order(s) for cancel.",
        sev, threshold_sigma, len(pending_market),
    )

    cancelled = []
    for o in pending_market:
        log.warning("  → CANCEL %s %s qty=%s (id=%s, intent=%s)",
                    o.side, o.symbol, o.qty, o.id,
                    getattr(o, "position_intent", "n/a"))
        if not dry_run:
            try:
                client.cancel_order_by_id(o.id)
                cancelled.append(o.symbol)
            except Exception as exc:
                log.error("  ! cancel failed for %s: %s", o.symbol, exc)

    if cancelled:
        try:
            msg = (
                f"PREOPEN-CANCEL: {metrics['source']} overnight "
                f"{pct*100:+.2f}% ({sev:+.1f}σ ≥ ±{threshold_sigma:.1f}σ) "
                f"→ cancelled {len(cancelled)} pending order(s): "
                f"{','.join(cancelled)}"
            )
            subprocess.run(
                ["curl", "-sf",
                 "-H", "Title: RenQuant 104 PREOPEN CANCEL",
                 "-H", "Priority: high",
                 "-d", msg, "https://ntfy.sh/renquant"],
                timeout=10, check=False,
            )
        except Exception:
            pass

    action = "dry-run" if dry_run else "cancelled"
    return {"metrics": metrics, "cancelled": cancelled,
            "considered": len(pending_market), "action": action}


def main() -> None:
    _load_env()
    p = argparse.ArgumentParser(
        description="Pre-open cancel gate for queued post-close market orders"
    )
    p.add_argument(
        "--severity-threshold-sigma", type=float, default=2.0,
        help="Cancel pending orders if |overnight σ-normalized return| ≥ "
             "this. Default 2.0 (Shewhart 2σ control limit, ~2.3%% "
             "one-sided exceedance).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute severity + list cancelables, but do NOT submit cancels.",
    )
    args = p.parse_args()
    result = cancel_stale_market_orders(
        threshold_sigma=args.severity_threshold_sigma,
        dry_run=args.dry_run,
    )
    log.info(
        "Done. Action=%s, cancelled=%s, considered=%d.",
        result["action"], result["cancelled"], result["considered"],
    )


if __name__ == "__main__":
    main()
