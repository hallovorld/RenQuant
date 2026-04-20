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
from datetime import date, datetime
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
    staleness_days = int(config.get("model_staleness_days", 30))
    models_dir = strategy_dir / "models"

    use_kernel = _load_kernel(strategy_dir)
    if not use_kernel:
        log.error("Strategy %s does not have a kernel/ package", strategy_name)
        sys.exit(1)

    from kernel.models import load_artifact as _kernel_load_artifact  # noqa: PLC0415

    models = {}
    for symbol in config["watchlist"]:
        meta_path = models_dir / symbol / f"{symbol}-policy-metadata.json"
        if not meta_path.exists():
            log.warning("No model metadata for %s, skipping", symbol)
            continue

        metadata = json.loads(meta_path.read_text())

        # Check staleness
        trained_date = metadata.get("trained_date")
        if trained_date and staleness_days > 0:
            age = (date.today() - datetime.strptime(trained_date, "%Y-%m-%d").date()).days
            if age > staleness_days:
                log.warning("%s model is %d days old (limit=%d), skipping", symbol, age, staleness_days)
                continue

        # Reject below-floor models
        sharpe_floor = float(config.get("sharpe_floor", 0.8))
        model_sharpe = float(metadata.get("sharpe", 0.0))
        if sharpe_floor > 0 and model_sharpe < sharpe_floor:
            log.warning("%s sharpe=%.3f below floor=%.1f, skipping", symbol, model_sharpe, sharpe_floor)
            continue

        artifact = _kernel_load_artifact(models_dir / symbol, symbol)
        if artifact is None:
            log.warning("Kernel load failed for %s, skipping", symbol)
            continue
        models[symbol] = artifact

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
    return config, models, strategy_dir


def _run_once_multi_pipeline(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
    sell_only: bool,
) -> None:
    """Create a PipelineContext and run the 3-job pipeline."""
    _load_kernel(strategy_dir)  # ensure pipeline/ is importable

    from pipeline import Pipeline, PipelineContext          # noqa: PLC0415
    from pipeline.jobs.data import DataJob                  # noqa: PLC0415
    from pipeline.jobs.signals import SignalJob             # noqa: PLC0415
    from pipeline.jobs.execution import ExecutionJob        # noqa: PLC0415

    run_mode = "sell-only" if sell_only else "full"
    sep = "=" * 62
    log.info(sep)
    log.info("RENQUANT-103  %s  [%s]", datetime.now().strftime("%Y-%m-%d %H:%M PT"), run_mode.upper())
    log.info(sep)

    ctx = PipelineContext(
        config=config,
        strategy_dir=strategy_dir,
        sell_only=sell_only,
        broker=broker,
        models=models,
    )
    Pipeline([DataJob(), SignalJob(), ExecutionJob()]).run(ctx)


def run_once_multi(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
    sell_only: bool = False,
) -> None:
    """Execute one multi-stock trading cycle via the kernel pipeline."""
    _run_once_multi_pipeline(config, models, broker, strategy_dir, sell_only)


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
    parser.add_argument("--interval", type=int, default=86400,
                        help="Seconds between runs in scheduled mode (default: 86400)")
    args = parser.parse_args()

    config, models, strategy_dir = _load_strategy_multi(args.strategy)
    initial_cash = config.get("initial_cash", 100_000)
    broker = _get_broker(args.broker, initial_cash=initial_cash)
    broker.connect()
    run_fn = lambda: run_once_multi(config, models, broker, strategy_dir,
                                    sell_only=args.sell_only)

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
