"""Live trading runner — renquant_103 (kernel-based pipeline).

Usage::

    python -m live.runner --strategy renquant_103 --broker paper --once
    python -m live.runner --strategy renquant_103 --broker alpaca-paper --once
    python -m live.runner --strategy renquant_103 --broker alpaca --once  # real money

Broker options: paper, alpaca, alpaca-paper, ibkr

Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables for Alpaca.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from .broker import BaseBroker
from .alpaca_broker import AlpacaBroker
from .ibkr_broker import IBKRBroker
from .paper_broker import PaperBroker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("live.runner")


# ── Kernel support ─────────────────────────────────────────────────────────────

_kernel_path_loaded: set[str] = set()


def _load_kernel(strategy_dir: Path) -> bool:
    """Add strategy_dir to sys.path so `from kernel.x import ...` works.

    Returns True if the kernel package exists in strategy_dir.
    """
    key = str(strategy_dir)
    if key in _kernel_path_loaded:
        return True
    if not (strategy_dir / "kernel" / "__init__.py").exists():
        return False
    if key not in sys.path:
        sys.path.insert(0, key)
    _kernel_path_loaded.add(key)
    return True


def _get_broker(broker_type: str, initial_cash: float = 100_000) -> BaseBroker:
    if broker_type == "paper":
        return PaperBroker(initial_cash=initial_cash)
    elif broker_type == "alpaca":
        return AlpacaBroker(paper=False)
    elif broker_type == "alpaca-paper":
        return AlpacaBroker(paper=True)
    elif broker_type == "ibkr":
        return IBKRBroker()
    else:
        raise ValueError(f"Unknown broker: {broker_type}")


def _log_trade(strategy_dir: Path, strategy_name: str, record: dict) -> None:
    """Append trade record to daily log file."""
    log_dir = REPO_ROOT / "live" / "logs" / strategy_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"

    entries = []
    if log_file.exists():
        entries = json.loads(log_file.read_text())
    entries.append(record)
    log_file.write_text(json.dumps(entries, indent=2, default=str))


def _build_relative_features(
    df_stock, df_spy, feature_columns, indicator_spec,
):
    """Compute indicators and relative features for a stock vs SPY.

    Used by scripts/recalibrate_scores.py for score calibration.
    kernel.indicators.compute_indicators is used so this has no common/ dependency.
    """
    import numpy as np
    import pandas as pd
    from kernel.indicators import compute_indicators

    df_stock = compute_indicators(df_stock, indicator_spec)
    df_spy = compute_indicators(df_spy, indicator_spec)

    common_idx = df_stock.index.intersection(df_spy.index)
    if len(common_idx) == 0:
        return None

    df_stock = df_stock.loc[common_idx]
    df_spy = df_spy.loc[common_idx]

    ratio_features = {"rsi", "adx"}
    diff_features = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}

    spy_close_full = df_spy["close"]
    spy_rets_full  = spy_close_full.pct_change()
    spy_ema50_full = spy_close_full.ewm(span=50, adjust=False).mean()

    # Audit fix LR-1 (Round 2 deep audit, 2026-04-25): pre-fix, live
    # runner / recalibrate_scores.py used a lag-1-autocorrelation
    # function under the misleading name `hurst_proxy`. After the TF-3
    # fix replaced the training-side `hurst_proxy` with the real R/S
    # Hurst exponent (kernel.regime.rolling_hurst), this code path
    # diverged — train fed real Hurst, live + recalibrate fed lag-1
    # autocorr. Calibrator was therefore fit on a different feature
    # distribution than the model expects → rank_score miscalibrated.
    # Now: use the same kernel.regime.rolling_hurst helper. Match
    # training/features.py's TF-3 fix.
    from kernel.regime import rolling_hurst as _rolling_hurst  # noqa: PLC0415

    spy_regime_features = {
        "spy_realized_vol": spy_rets_full.rolling(20).std() * np.sqrt(252),
        "spy_adx":   df_spy["adx"],
        "spy_trend": spy_close_full / spy_ema50_full.replace(0, np.nan),
        # Real Hurst exponent (R/S, 63-day window). Same as TF-3 fix.
        "hurst_proxy": _rolling_hurst(spy_rets_full, window=63),
    }

    result = pd.DataFrame(index=common_idx)
    result["close"] = df_stock["close"]
    for col in feature_columns:
        if col in ratio_features:
            if col in df_stock.columns and col in df_spy.columns:
                result[col] = df_stock[col] / df_spy[col].replace(0, np.nan)
        elif col in diff_features:
            if col in df_stock.columns and col in df_spy.columns:
                result[col] = df_stock[col] - df_spy[col]
        elif col in spy_regime_features:
            result[col] = spy_regime_features[col].reindex(common_idx)
        elif col in df_stock.columns:
            result[col] = df_stock[col]

    close = df_stock.loc[common_idx, "close"]
    spy_close = df_spy.loc[common_idx, "close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    result["trend"] = close / ema50
    result["trend_long"] = close / ema200
    rel_price = close / spy_close.replace(0, np.nan)
    result["rel_mom_20d"] = rel_price.pct_change(20)
    result["rel_mom_60d"] = rel_price.pct_change(60)

    return result.dropna()


def _load_strategy_multi(strategy_name: str) -> tuple[dict[str, Any], dict, Path]:
    """Load multi-stock strategy config and per-stock kernel model artifacts."""
    strategy_dir = REPO_ROOT / "backtesting" / strategy_name
    config_path = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    config = json.loads(config_path.read_text())

    use_kernel = _load_kernel(strategy_dir)
    if not use_kernel:
        log.error("Strategy %s does not have a kernel/ package", strategy_name)
        sys.exit(1)

    from kernel.pipeline.job_universe import UniverseContext, LoadUniverseJob  # noqa: PLC0415

    uctx = UniverseContext(config=config, strategy_dir=strategy_dir)
    LoadUniverseJob().run(uctx)
    models = uctx.loaded_models
    for ticker, reason in uctx.rejections:
        log.warning("%s %s, skipping", ticker, reason)

    log.info("Loaded models for %d/%d symbols: %s",
             len(models), len(config["watchlist"]), sorted(models.keys()))

    # ── MODEL SUMMARY TABLE ─────────────────────────────────────────────────
    log.info("─" * 62)
    log.info("MODEL SUMMARY  (%d loaded)", len(models))
    log.info("  %-6s  %-14s  %-12s  %-5s  %-10s  %s",
             "SYMBOL", "TYPE", "TRAINED", "SHARPE", "ROWS", "TRAIN END")
    for sym in sorted(models):
        meta = models[sym].get("_metadata", {})
        log.info("  %-6s  %-14s  %-12s  %-5.2f  %-10s  %s",
                 sym,
                 meta.get("best_approach", meta.get("policy_type", "?")),
                 meta.get("trained_date", "?"),
                 float(meta.get("sharpe", 0.0)),
                 str(meta.get("live_train_rows", "?")),
                 meta.get("live_train_end", "?"))
    log.info("─" * 62)

    config["_use_kernel"] = True
    config["_strategy_dir"] = str(strategy_dir)
    return config, models, strategy_dir


def _run_once_multi_pipeline(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
    sell_only: bool,
    use_intraday_prices: bool = False,
) -> None:
    """Create RunnerAdapter + InferencePipeline and execute one trading cycle."""
    _load_kernel(strategy_dir)  # ensure kernel/ is importable

    from kernel.pipeline import InferencePipeline, SellOnlyPipeline  # noqa: PLC0415
    from adapters.runner import RunnerAdapter                          # noqa: PLC0415

    run_mode = "sell-only" if sell_only else "full"
    if use_intraday_prices:
        run_mode += " (intraday)"
    sep = "=" * 62
    log.info(sep)
    label = strategy_dir.name.upper().replace("_", "-")
    log.info("%s  %s  [%s]", label, datetime.now().strftime("%Y-%m-%d %H:%M PT"), run_mode.upper())
    log.info(sep)

    adapter  = RunnerAdapter(
        config, models, broker, strategy_dir,
        sell_only=sell_only,
        use_intraday_prices=use_intraday_prices,
    )
    pipeline = SellOnlyPipeline() if sell_only else InferencePipeline()

    ctx = adapter.make_context()
    pipeline.run(ctx)
    adapter.commit(ctx)

    # ── Decision-level ntfy (mandatory — user requirement, 2026-04-23) ──
    # Every decision cycle — whether it placed orders or not — MUST
    # notify. This lets the user confirm the system is alive + made a
    # deliberate decision (vs silent hang / crash). Previously we only
    # notified on actual trades; per 2026-04-23 follow-up: notify always.
    _notify_decision(label, run_mode, ctx)


def _notify_decision(label: str, run_mode: str, ctx) -> None:
    """Fire ntfy on every decision cycle.

    - If orders/exits happened: Priority=high, body lists each action.
    - If zero-trade cycle: Priority=default, body summarises why
      (regime, transition state, candidate count, slot fill).

    Respects `RENQUANT_NTFY_TOPIC` env var (default 'renquant').
    Never raises — network failure logs WARNING but doesn't roll back
    the committed trade state.
    """
    import os, urllib.request  # noqa: PLC0415
    # IMPORTANT: read the BROKER-CONFIRMED order list (orders_placed),
    # populated by adapter.commit AFTER the duplicate-order guard and
    # broker submission. `ctx.orders` is merely the pipeline intent and
    # will include orders that the guard blocked (e.g. TSM on 2026-04-23
    # 21:22 — a pending pre-market order from 21:04 caused a repeat
    # pipeline cycle to emit a second BUY TSM which the guard correctly
    # skipped, but the old ntfy misreported it as a trade).
    orders         = list(getattr(ctx, "orders_placed",  []) or [])
    orders_skipped = list(getattr(ctx, "orders_skipped", []) or [])
    # Audit fix EXITS-FAIL (Round 2 deep audit, 2026-04-25): prefer
    # broker-confirmed `exits_placed`; fall back to `ctx.exits` for
    # adapters that haven't been migrated yet (LeanAdapter still uses
    # the bare list — sim/LEAN don't talk to a real broker so the
    # old form is correct there). Read `exits_failed` separately so
    # the operator sees a distinct "FAILED-EXIT" line on their phone
    # when a sell didn't actually go through.
    exits_placed_attr = getattr(ctx, "exits_placed", None)
    if exits_placed_attr is not None:
        exits = list(exits_placed_attr or [])
    else:
        exits = list(getattr(ctx, "exits", []) or [])
    exits_failed = list(getattr(ctx, "exits_failed", []) or [])
    regime         = getattr(ctx, "regime", None) or "?"
    conf   = getattr(ctx, "confidence", None)
    eq     = getattr(ctx, "portfolio_value", None)
    n_held = len(getattr(ctx, "holdings", {}) or {})

    # Rough "why" rollup for zero-trade summaries. Order the checks so
    # the MOST specific cause is surfaced first.
    def _why_no_trade() -> str:
        if getattr(ctx, "bear_only", False):
            return "bear_only"
        rs = getattr(ctx, "regime_state", None)
        if rs is not None and getattr(rs, "in_transition", False):
            return "transition_window"
        if getattr(ctx, "skip_buys", False):
            return "drawdown_halt"
        if getattr(ctx, "buy_blocked", False):
            return "buy_blocked"
        # Counter-based: if we had candidates but none were selected
        counters = getattr(ctx, "counters", {}) or {}
        if counters.get("defensive_non_bear_blocks", 0):
            return f"defensives_filtered({counters['defensive_non_bear_blocks']})"
        if counters.get("sector_blocks", 0):
            return f"sector_full({counters['sector_blocks']})"
        if counters.get("corr_blocks", 0):
            return f"correlation({counters['corr_blocks']})"
        if counters.get("blocked_wash", 0):
            return f"wash_sale({counters['blocked_wash']})"
        ranked = getattr(ctx, "ranked", None) or []
        if len(ranked) == 0:
            return "no_candidates"
        return "tier_threshold"

    parts: list[str] = []
    for o in orders:
        tkr    = o.get("ticker")  if isinstance(o, dict) else getattr(o, "ticker", "?")
        shares = o.get("shares")  if isinstance(o, dict) else getattr(o, "shares", "?")
        price  = o.get("price")   if isinstance(o, dict) else getattr(o, "price", 0.0)
        parts.append(f"BUY {tkr} x{shares} @ ${float(price):.2f}")
    for e in exits:
        # ctx.exits is list[(ticker, ExitSignal)] — unpack the tuple.
        # getattr(tuple, "ticker") always returned the default, so every
        # exit historically logged as "EXIT ? (?)" in ntfy.
        if isinstance(e, tuple) and len(e) == 2:
            tkr, sig = e
            reason = getattr(sig, "exit_type", getattr(sig, "reason", "sell"))
        else:
            tkr    = getattr(e, "ticker", "?")
            reason = getattr(e, "exit_type", getattr(e, "reason", "sell"))
        parts.append(f"EXIT {tkr} ({reason})")

    # EXITS-FAIL: surface failed sells distinctly so the operator
    # doesn't confuse a broker rejection with a successful exit. Each
    # entry is a dict with ticker / exit_type / qty / error.
    for fe in exits_failed:
        tkr   = fe.get("ticker", "?") if isinstance(fe, dict) else "?"
        rsn   = (fe.get("exit_type") or fe.get("reason") or "?") if isinstance(fe, dict) else "?"
        err   = (fe.get("error", "") if isinstance(fe, dict) else "")[:60]
        parts.append(f"FAILED-EXIT {tkr} ({rsn}: {err})")

    has_trade = bool(orders or exits)
    # If the guard blocked every intent (orders all skipped), the cycle
    # produced no real trade — surface the skip reason prominently so
    # the user doesn't mistake a blocked-duplicate for a fresh buy.
    if not has_trade:
        if orders_skipped:
            skip_parts = [
                f"{o.get('ticker', '?')} ({o.get('skip_reason', 'skipped')})"
                for o in orders_skipped
            ]
            parts.append("SKIPPED " + "; ".join(skip_parts))
        else:
            parts.append(f"no trade ({_why_no_trade()})")

    # Always append system state snapshot for audit visibility
    ctx_bits: list[str] = [f"regime={regime}"]
    if conf is not None:
        try:
            ctx_bits.append(f"conf={float(conf):.2f}")
        except (TypeError, ValueError):
            pass
    ctx_bits.append(f"held={n_held}")
    if eq is not None:
        try:
            ctx_bits.append(f"eq=${float(eq):,.0f}")
        except (TypeError, ValueError):
            pass
    parts.append(" ".join(ctx_bits))

    topic    = os.environ.get("RENQUANT_NTFY_TOPIC", "renquant")
    tag      = "TRADE" if has_trade else "DECISION"
    priority = "high"  if has_trade else "default"
    title    = f"{label} [{run_mode}] {tag}"
    body     = " | ".join(parts)
    url      = f"https://ntfy.sh/{topic}"
    try:
        req = urllib.request.Request(
            url, data=body.encode("utf-8"), method="POST",
            headers={"Title": title, "Priority": priority},
        )
        urllib.request.urlopen(req, timeout=5.0).read()
        log.info("ntfy sent: %s | %s", title, body)
    except Exception as exc:
        log.warning("ntfy publish FAILED (%s) — cycle still committed: %s",
                    exc, body)


def run_once_multi(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
    sell_only: bool = False,
    use_intraday_prices: bool = False,
) -> None:
    """Execute one multi-stock trading cycle via the kernel pipeline."""
    _run_once_multi_pipeline(
        config, models, broker, strategy_dir,
        sell_only, use_intraday_prices,
    )


def _is_multi_stock(strategy_name: str) -> bool:
    """Check if a strategy uses multi-stock watchlist config."""
    config_path = REPO_ROOT / "backtesting" / strategy_name / "strategy_config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())
    return "watchlist" in config


def main():
    parser = argparse.ArgumentParser(description="RenQuant live trading runner")
    parser.add_argument("--strategy", required=True, help="Strategy directory name")
    parser.add_argument("--broker", choices=["paper", "alpaca", "alpaca-paper", "ibkr"], default="paper")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--sell-only", action="store_true",
                        help="Process exits only — skip buy scan (for intraday runs)")
    parser.add_argument("--intraday", action="store_true",
                        help="Overlay latest Alpaca 5-min close onto today's bar "
                             "(only useful with --sell-only during market hours)")
    # Audit #84: 86400 (24h) was misleading — production scheduling lives
    # in macOS launchd, not in this loop. Default kept for back-compat but
    # the scheduled-mode path emits a warning when a user actually picks it.
    parser.add_argument("--interval", type=int, default=86400,
                        help="Seconds between runs in scheduled mode (default: 86400). "
                             "Production scheduling uses launchd; this loop is intended "
                             "only for ad-hoc tests.")
    args = parser.parse_args()

    config, models, strategy_dir = _load_strategy_multi(args.strategy)
    initial_cash = config.get("initial_cash", 100_000)
    broker = _get_broker(args.broker, initial_cash=initial_cash)
    broker.connect()
    run_fn = lambda: run_once_multi(
        config, models, broker, strategy_dir,
        sell_only=args.sell_only,
        use_intraday_prices=args.intraday,
    )

    try:
        if args.once:
            run_fn()
        else:
            log.warning("Scheduled mode invoked (interval=%ds). Production "
                        "scheduling should use macOS launchd; this loop is "
                        "for ad-hoc testing only (audit #84).", args.interval)
            while True:
                try:
                    run_fn()
                except Exception:
                    log.exception("Error in trading cycle")
                log.info("Sleeping %ds until next cycle...", args.interval)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
