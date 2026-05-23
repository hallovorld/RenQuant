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

Threshold: sigma-normalized severity. The bar is "absolute overnight
move >= 2 x sigma_60d_overnight". Default 2.0 sigma chosen per Shewhart;
tunable via --severity-threshold-sigma.

Data source: yfinance ES=F 5-minute bars for the current pre-open move,
normalized by SPY cash-session close->open overnight sigma. This avoids
the old daily-bar bug where ES=F's futures-session open could be mistaken
for the current pre-open price.

Behavior:
1. Compute ES current move since the previous NYSE cash close
2. Normalize by 60d SPY cash overnight gap sigma
3. If |severity| < threshold: PASS, no action
4. If |severity| ≥ threshold: cancel ALL pending market orders on the
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


def _to_utc_timestamp(ts):
    import pandas as pd

    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _series(df, name: str):
    s = df[name]
    if hasattr(s, "squeeze"):
        s = s.squeeze()
    if getattr(s, "ndim", 1) > 1:
        s = s.iloc[:, 0]
    return s.astype(float).dropna()


def _previous_nyse_close(now_utc):
    import pandas as pd
    import pandas_market_calendars as mcal

    now_utc = _to_utc_timestamp(now_utc)
    cal = mcal.get_calendar("NYSE")
    start = (now_utc - pd.Timedelta(days=14)).date()
    end = now_utc.date()
    sched = cal.schedule(start, end)
    if sched.empty:
        raise ValueError("NYSE calendar returned no recent sessions")
    closes = sched["market_close"].map(_to_utc_timestamp)
    prior = closes[closes < now_utc]
    if prior.empty:
        raise ValueError("no prior NYSE cash close before current time")
    return prior.iloc[-1]


def _is_nyse_session_date(day=None) -> bool:
    import pandas as pd
    import pandas_market_calendars as mcal

    if day is None:
        target = pd.Timestamp.now(tz="America/New_York").date()
    else:
        target = day
    cal = mcal.get_calendar("NYSE")
    return not cal.schedule(target, target).empty


def _cash_overnight_sigma(
    yf,
    *,
    sigma_symbol: str,
    fallback_symbol: str,
    lookback_days: int,
    sigma_window: int,
    now_utc,
) -> tuple[float, int, str]:
    import pandas as pd

    end = _to_utc_timestamp(now_utc)
    start = end - pd.Timedelta(days=int(lookback_days * 1.7))

    def _fetch(sym: str):
        try:
            return yf.download(
                sym,
                start=start.date(),
                end=(end + pd.Timedelta(days=1)).date(),
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            log.warning("yf.download(%s daily) failed: %s", sym, exc)
            return None

    used = sigma_symbol
    df = _fetch(sigma_symbol)
    if df is None or df.empty:
        log.warning(
            "Sigma symbol %s unavailable; falling back to %s",
            sigma_symbol,
            fallback_symbol,
        )
        used = fallback_symbol
        df = _fetch(fallback_symbol)
    if df is None or df.empty:
        raise ValueError(
            f"both {sigma_symbol} and {fallback_symbol} unavailable from yfinance"
        )

    df = df.sort_index()
    opens = _series(df, "Open")
    closes = _series(df, "Close")
    prior_close = closes.shift(1)
    overnight = ((opens - prior_close) / prior_close).replace(
        [float("inf"), -float("inf")], pd.NA,
    )
    overnight = overnight.dropna()
    if len(overnight) < 10:
        raise ValueError(f"insufficient overnight sigma history ({len(overnight)} obs)")
    return float(overnight.tail(sigma_window).std()), int(len(overnight)), used


def _current_futures_move(
    yf,
    *,
    symbol: str,
    now_utc,
    max_stale_minutes: float,
) -> dict:
    import pandas as pd

    now_utc = _to_utc_timestamp(now_utc)
    prior_cash_close = _previous_nyse_close(now_utc)
    try:
        df = yf.download(
            symbol,
            period="10d",
            interval="5m",
            prepost=True,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        raise ValueError(f"yf.download({symbol} 5m) failed: {exc}") from exc
    if df is None or df.empty:
        raise ValueError(f"{symbol} 5m history unavailable from yfinance")

    close = _series(df.sort_index(), "Close")
    if close.empty:
        raise ValueError(f"{symbol} 5m close series unavailable")
    close.index = pd.DatetimeIndex([_to_utc_timestamp(ts) for ts in close.index])
    close = close[close.index <= now_utc]
    if close.empty:
        raise ValueError(f"{symbol} 5m history has no bars before now")

    latest_ts = close.index[-1]
    stale_minutes = (now_utc - latest_ts).total_seconds() / 60.0
    if stale_minutes > max_stale_minutes:
        raise ValueError(
            f"{symbol} latest 5m bar is stale ({stale_minutes:.1f} min old)"
        )

    ref = close[close.index <= prior_cash_close]
    if ref.empty:
        raise ValueError(f"{symbol} has no 5m bar before prior NYSE close")
    ref_ts = ref.index[-1]
    prior_price = float(ref.iloc[-1])
    latest = float(close.iloc[-1])
    if prior_price <= 0:
        raise ValueError(f"{symbol} invalid prior close proxy {prior_price}")
    return {
        "prior_close": prior_price,
        "latest": latest,
        "current_pct": (latest - prior_price) / prior_price,
        "prior_close_time": ref_ts.isoformat(),
        "latest_time": latest_ts.isoformat(),
        "stale_minutes": float(stale_minutes),
    }


def compute_overnight_severity(
    *,
    symbol: str = "ES=F",
    sigma_symbol: str = "SPY",
    fallback_sigma_symbol: str = "^GSPC",
    lookback_days: int = 90,
    sigma_window: int = 60,
    max_stale_minutes: float = 120.0,
    now=None,
) -> dict:
    """Return {prior_close, latest, current_pct, sigma_60d, severity}.

    The live move uses ES=F 5-minute futures from the prior NYSE cash close
    to the latest current print. The denominator is SPY cash close->open
    overnight sigma, so the alert threshold is tied to the gap distribution
    of the equities the orders actually trade.

    Raises ValueError if data is unavailable.
    """
    import yfinance as yf
    import pandas as pd

    now_utc = _to_utc_timestamp(now or pd.Timestamp.now(tz="UTC"))
    move = _current_futures_move(
        yf,
        symbol=symbol,
        now_utc=now_utc,
        max_stale_minutes=max_stale_minutes,
    )
    sigma_60d, n_obs, sigma_used = _cash_overnight_sigma(
        yf,
        sigma_symbol=sigma_symbol,
        fallback_symbol=fallback_sigma_symbol,
        lookback_days=lookback_days,
        sigma_window=sigma_window,
        now_utc=now_utc,
    )
    severity = 0.0
    # Guard against degenerate sigma (synthetic test data, or genuinely
    # zero-vol windows on holidays). Threshold 1e-8 is below any
    # realistic equity-vol scale (50 bps/day = 0.005 ≫ 1e-8) so
    # legitimate signal is never lost; eps-sigma becomes treated as
    # "no information" rather than producing 1e18-scale severity.
    if sigma_60d > 1e-8:
        severity = float(move["current_pct"]) / sigma_60d
    return {
        "source":      symbol,
        "sigma_source": sigma_used,
        "prior_close": float(move["prior_close"]),
        "latest":      float(move["latest"]),
        "current_pct": float(move["current_pct"] if sigma_60d > 1e-8 else 0.0),
        "sigma_60d":   sigma_60d,
        "severity":    float(severity),
        "n_obs":       int(n_obs),
        "prior_close_time": move["prior_close_time"],
        "latest_time": move["latest_time"],
        "stale_minutes": move["stale_minutes"],
    }


def cancel_stale_market_orders(*, threshold_sigma: float, dry_run: bool) -> dict:
    """Cancel pending market orders if overnight severity ≥ threshold.

    Returns {metrics, cancelled, considered, action}.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    try:
        metrics = compute_overnight_severity()
    except ValueError as exc:
        log.warning("PREOPEN-GATE: DATA-UNAVAILABLE — %s. No cancel.", exc)
        return {
            "metrics": {"error": str(exc)},
            "cancelled": [],
            "considered": 0,
            "action": "data-unavailable",
        }
    sev = metrics["severity"]
    pct = metrics["current_pct"]
    sigma = metrics["sigma_60d"]
    log.info(
        "PREOPEN-GATE: %s current-vs-cash-close %+.3f%% "
        "(%s sigma_60d=%.3f%%, severity=%+.2f sigma, "
        "threshold=+/-%.1f sigma, n_obs=%d, stale=%.1f min)",
        metrics["source"], pct * 100, metrics.get("sigma_source", "?"),
        sigma * 100, sev, threshold_sigma, metrics["n_obs"],
        metrics.get("stale_minutes", -1.0),
    )

    if abs(sev) < threshold_sigma:
        log.info("PREOPEN-GATE: PASS — severity within +/-%.1f sigma. No action.",
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
        "PREOPEN-GATE: TRIGGERED — severity=%+.2f sigma >= +/-%.1f sigma; "
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
                f"PREOPEN-CANCEL: {metrics['source']} current-vs-cash-close "
                f"{pct*100:+.2f}% ({sev:+.1f}σ >= +/-{threshold_sigma:.1f}σ) "
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
    p.add_argument(
        "--ignore-calendar", action="store_true",
        help="Run even when today is not an NYSE trading session.",
    )
    args = p.parse_args()
    if not args.ignore_calendar and not _is_nyse_session_date():
        log.info("NYSE closed today — skipping pre-open cancel gate.")
        return
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
