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

    def _hurst_proxy(x):
        if len(x) <= 2 or np.std(x) == 0:
            return 0.0
        cc = np.corrcoef(x[:-1], x[1:])
        v = cc[0, 1]
        return float(v) if not np.isnan(v) else 0.0

    spy_regime_features = {
        "spy_realized_vol": spy_rets_full.rolling(20).std() * np.sqrt(252),
        "spy_adx":   df_spy["adx"],
        "spy_trend": spy_close_full / spy_ema50_full.replace(0, np.nan),
        "hurst_proxy": spy_rets_full.rolling(20).apply(_hurst_proxy, raw=True),
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

    # ── Trade-level ntfy (mandatory — user requirement, 2026-04-23) ─────
    # Any path that places orders MUST notify. Shell wrappers
    # (daily_104.sh / live_only_104.sh / intraday_sell_104.sh) send
    # their own wrapper-level ntfy, but direct `python -m live.runner`
    # invocations previously slipped through silently. This hook
    # guarantees every real trade surfaces to ntfy, regardless of how
    # the runner was triggered.
    _notify_trades(label, run_mode, ctx)


def _notify_trades(label: str, run_mode: str, ctx) -> None:
    """Fire ntfy if this cycle placed any buy/sell orders.

    Idempotent + silent on no-trades. Respects optional
    `RENQUANT_NTFY_TOPIC` env var (defaults to 'renquant').
    Never raises — network-safety wrapped.
    """
    import os, urllib.request, urllib.parse      # noqa: PLC0415
    orders = list(getattr(ctx, "orders", []) or [])
    exits  = list(getattr(ctx, "exits",  []) or [])
    if not orders and not exits:
        return   # silent when nothing happened

    parts: list[str] = []
    for o in orders:
        tkr    = o.get("ticker")       if isinstance(o, dict) else getattr(o, "ticker", "?")
        shares = o.get("shares")       if isinstance(o, dict) else getattr(o, "shares", "?")
        price  = o.get("price")        if isinstance(o, dict) else getattr(o, "price", 0.0)
        invest = o.get("invest") if isinstance(o, dict) else (shares * price if shares else 0)
        parts.append(f"BUY {tkr} x{shares} @ ${float(price):.2f}")
    for e in exits:
        tkr    = getattr(e, "ticker", "?")
        reason = getattr(e, "exit_type", getattr(e, "reason", "sell"))
        parts.append(f"EXIT {tkr} ({reason})")

    topic = os.environ.get("RENQUANT_NTFY_TOPIC", "renquant")
    title = f"{label} [{run_mode}] TRADE"
    body  = " | ".join(parts)
    url   = f"https://ntfy.sh/{topic}"
    try:
        data = body.encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Title": title, "Priority": "high"},
        )
        urllib.request.urlopen(req, timeout=5.0).read()
        log.info("ntfy sent: %s | %s", title, body)
    except Exception as exc:
        # Don't let a network blip suppress trade reporting — log loudly.
        log.warning("ntfy publish FAILED (%s) — trade still executed: %s",
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
    parser.add_argument("--interval", type=int, default=86400,
                        help="Seconds between runs in scheduled mode (default: 86400)")
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
            log.info("Starting scheduled mode (interval=%ds)", args.interval)
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
