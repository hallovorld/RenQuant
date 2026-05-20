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
    elif broker_type == "alpaca-shorts":
        # 2026-05-15: dedicated paper account for shorts feature testing.
        # Credentials in ALPACA_SHORTS_API_KEY / ALPACA_SHORTS_SECRET_KEY
        # (separate from ALPACA_API_KEY which the regular crons use for
        # the LIVE account). Distinct broker_name="alpaca-shorts" gives
        # state-file isolation: live_state.alpaca-shorts.json separate
        # from live_state.alpaca.json. No risk of position-tracking
        # collision with live trading.
        return AlpacaBroker(
            paper=True,
            env_prefix="ALPACA_SHORTS",
            label="alpaca-shorts",
        )
    elif broker_type == "ibkr":
        return IBKRBroker()
    elif broker_type == "readonly-alpaca":
        # 2026-05-19 user mandate: full-e2e shadow pipeline. Wraps real
        # AlpacaBroker so reads (account / holdings / quotes / fills) hit
        # LIVE alpaca API for ground-truth state, but writes (place_order /
        # cancel_order / place_stop_order) are swallowed locally with a
        # synthesised filled response. broker_name="alpaca_shadow" gives
        # state-file isolation: live_state.alpaca_shadow.json + runs_alpaca_shadow.db
        # never collide with prod live_state.alpaca.json. Pair with
        # `--strategy-config-name strategy_config.shadow.json` so the
        # panel scorer also swaps (HF PatchTST instead of XGB).
        from .broker_readonly import ReadOnlyBrokerWrapper  # noqa: PLC0415
        real = AlpacaBroker(paper=False)
        return ReadOnlyBrokerWrapper(real)
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


def _load_strategy_multi(
    strategy_name: str, broker_name: str | None = None,
    config_name: str = "strategy_config.json",
) -> tuple[dict[str, Any], dict, Path]:
    """Load multi-stock strategy config and per-stock kernel model artifacts."""
    strategy_dir = REPO_ROOT / "backtesting" / strategy_name
    config_path = strategy_dir / config_name
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    config = json.loads(config_path.read_text())

    use_kernel = _load_kernel(strategy_dir)
    if not use_kernel:
        log.error("Strategy %s does not have a kernel/ package", strategy_name)
        sys.exit(1)

    # renquant_104+ has job_universe.py (LoadUniverseJob handles admission).
    # renquant_103 predates that module — fall back to direct artifact loading.
    job_universe_path = strategy_dir / "kernel" / "pipeline" / "job_universe.py"
    if job_universe_path.exists():
        from kernel.pipeline.job_universe import UniverseContext, LoadUniverseJob  # noqa: PLC0415

        # 2026-04-27: pass broker tag so UniverseContext reads broker-isolated
        # live_state.{broker}.json (FilterUniverseFloorTask + FilterAutoDropTask).
        # broker_name comes from caller (main()): translation of args.broker
        # into a state-isolation tag. None falls back to legacy live_state.json.
        uctx = UniverseContext(
            config=config, strategy_dir=strategy_dir, broker_name=broker_name,
        )
        LoadUniverseJob().run(uctx)
        models = uctx.loaded_models
        for ticker, reason in uctx.rejections:
            log.warning("%s %s, skipping", ticker, reason)
    else:
        # Legacy loader for renquant_103-era kernels (no job_universe.py).
        import datetime as _dt  # noqa: PLC0415
        from kernel.models import load_artifact as _kernel_load_artifact  # noqa: PLC0415

        staleness_days = int(config.get("model_staleness_days", 30))
        sharpe_floor   = float(config.get("sharpe_floor", 0.8))
        models_dir     = strategy_dir / "models"
        models: dict   = {}
        for symbol in config["watchlist"]:
            meta_path = models_dir / symbol / f"{symbol}-policy-metadata.json"
            if not meta_path.exists():
                log.warning("%s no_artifact, skipping", symbol)
                continue
            metadata = json.loads(meta_path.read_text())
            trained_date = metadata.get("trained_date")
            if trained_date and staleness_days > 0:
                from datetime import date as _date  # noqa: PLC0415
                age = (_date.today() - _dt.datetime.strptime(trained_date, "%Y-%m-%d").date()).days
                if age > staleness_days:
                    log.warning("%s model is %d days old (limit=%d), skipping",
                                symbol, age, staleness_days)
                    continue
            model_sharpe = float(metadata.get("sharpe", 0.0))
            if sharpe_floor > 0 and model_sharpe < sharpe_floor:
                log.warning("%s sharpe_%.3f_below_%.1f, skipping",
                            symbol, model_sharpe, sharpe_floor)
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
    # 2026-05-19: shadow run uses ReadOnlyBrokerWrapper (broker_name=alpaca_shadow).
    # Prefix label with [SHADOW] so log + ntfy + state-file path all carry the
    # distinction. Hard isolation per user mandate "隔离干净".
    if getattr(broker, "broker_name", "") == "alpaca_shadow":
        label = f"[SHADOW]{label}"
    log.info("%s  %s  [%s]", label, datetime.now().strftime("%Y-%m-%d %H:%M PT"), run_mode.upper())
    log.info(sep)

    # Pre-flight smoke test (2026-04-28): catch model/config/state drift
    # BEFORE constructing the adapter or executing any pipeline. Each check
    # ≤ 1 second; total <= 5 seconds. HARD failures raise PreflightFailed
    # which we convert into a high-priority ntfy + abort. Refer to
    # backtesting/renquant_104/kernel/preflight.py for the full list.
    # Disable via config: live.preflight.enabled = false (default true).
    preflight_cfg = config.get("live", {}).get("preflight", {})
    if preflight_cfg.get("enabled", True):
        try:
            from kernel.preflight import run_preflight, PreflightFailed  # noqa: PLC0415
            run_preflight(
                config, broker=broker, strategy_dir=strategy_dir,
                broker_name=getattr(broker, "broker_name", None),
                strict=bool(preflight_cfg.get("strict", True)),
            )
        except PreflightFailed as exc:
            log.error("PRE-FLIGHT FAILED — aborting cron, no orders placed:\n%s", exc)
            try:
                import os as _os, urllib.request as _ureq  # noqa: PLC0415
                topic = _os.environ.get("RENQUANT_NTFY_TOPIC", "renquant")
                req = _ureq.Request(
                    f"https://ntfy.sh/{topic}",
                    data=str(exc).encode("utf-8"), method="POST",
                    headers={"Title": f"{label} [{run_mode}] PREFLIGHT-FAIL",
                             "Priority": "urgent"},
                )
                _ureq.urlopen(req, timeout=5.0).read()
            except Exception:
                pass
            raise SystemExit(2) from exc
        except ImportError:
            # preflight module not yet on PYTHONPATH (during a transitional
            # commit). Log and proceed — the legacy guards still apply.
            log.warning("preflight module not importable; proceeding without")
        except Exception as exc:
            # An unexpected check exception should not block cron; degrade
            # to soft warn and continue.
            log.warning("preflight raised unexpectedly: %s — proceeding", exc)

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
    #
    # EXCEPTION (2026-04-27): the every-30-min intraday sell-only cycle
    # fires up to 12× per day and is dominated by no-op cycles. Per user
    # follow-up that day, suppress ntfy on those when the cycle produced
    # nothing actionable — keep ntfy for actual trades + failed exits.
    silent_if_quiet = bool(sell_only and use_intraday_prices)
    _notify_decision(label, run_mode, ctx, silent_if_quiet=silent_if_quiet)


def _notify_decision(label: str, run_mode: str, ctx, silent_if_quiet: bool = False) -> None:
    """Fire ntfy on every decision cycle.

    - If orders/exits happened: Priority=high, body lists each action.
    - If zero-trade cycle: Priority=default, body summarises why
      (regime, transition state, candidate count, slot fill).

    Respects `RENQUANT_NTFY_TOPIC` env var (default 'renquant').
    Never raises — network failure logs WARNING but doesn't roll back
    the committed trade state.

    `silent_if_quiet=True` (used by the every-30-min intraday sell-only
    cycle) suppresses the ntfy entirely when the cycle had no real
    operation — no order placed, no exit executed, no failed exit, no
    unmanaged broker position to surface. Kept LOUD for trades + any
    failure mode the operator should see.
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
        # 2026-05-18: include explicit $ realized P/L when adapter has
        # stamped it on the ExitSignal (per user mandate; cost basis
        # from broker avg_entry_price / HoldingState.entry_price).
        if isinstance(e, tuple) and len(e) == 2:
            tkr, sig = e
            reason = getattr(sig, "exit_type", getattr(sig, "reason", "sell"))
        else:
            tkr    = getattr(e, "ticker", "?")
            reason = getattr(e, "exit_type", getattr(e, "reason", "sell"))
            sig = e
        pnl_d = getattr(sig, "realized_pnl_dollar", None)
        pnl_p = getattr(sig, "realized_pnl_pct", None)
        if pnl_d is not None and pnl_p is not None:
            parts.append(f"EXIT {tkr} ({reason}) P/L=${pnl_d:+.2f} ({pnl_p:+.2f}%)")
        else:
            parts.append(f"EXIT {tkr} ({reason})")

    # EXITS-FAIL: surface failed sells distinctly so the operator
    # doesn't confuse a broker rejection with a successful exit. Each
    # entry is a dict with ticker / exit_type / qty / error.
    for fe in exits_failed:
        tkr   = fe.get("ticker", "?") if isinstance(fe, dict) else "?"
        rsn   = (fe.get("exit_type") or fe.get("reason") or "?") if isinstance(fe, dict) else "?"
        err   = (fe.get("error", "") if isinstance(fe, dict) else "")[:60]
        parts.append(f"FAILED-EXIT {tkr} ({rsn}: {err})")

    # UNMANAGED-NTFY (Bug B fix 2026-04-25): surface broker positions held
    # outside the watchlist so the operator knows they exist. Pre-fix this
    # was log-only; the operator's phone never saw it and real positions
    # (e.g. BA in the audit) sat unmanaged with no stop-loss / trailing-stop.
    non_wl_holds = list(getattr(ctx, "non_wl_holds", []) or [])
    if non_wl_holds:
        parts.append(f"UNMANAGED {','.join(non_wl_holds)} (no model — manual exit)")

    # ROT-BLOCKED-NTFY (Bug L, 2026-04-25): surface rotation pairs that
    # find_rotation_pairs accepted but EmitRotationsTask later vetoed
    # (Kelly=0, bad price, insufficient cash). Pre-fix the operator only
    # saw the resulting buys/exits, so they could not tell whether the
    # system had ALSO wanted to swap (and what blocked the swap). Each
    # blocked entry is `{sell, buy, reason}`.
    rot_blocked = list(getattr(ctx, "rotations_blocked", []) or [])
    for rb in rot_blocked:
        if isinstance(rb, dict):
            sell_t = rb.get("sell", "?")
            buy_t  = rb.get("buy",  "?")
            reason = rb.get("reason", "?")
            parts.append(f"BLOCKED-ROTATION {sell_t}→{buy_t} ({reason})")

    has_trade = bool(orders or exits)
    # Anything the operator should see counts as "actionable" for the
    # silent-if-quiet gate. A bare-quiet cycle = no orders, no exits, no
    # failed exits, no unmanaged broker position, no rotation block, no
    # skipped intent. When silent_if_quiet=True (intraday sell-only),
    # those cycles return without sending ntfy.
    if silent_if_quiet and not (
        has_trade or exits_failed or non_wl_holds or rot_blocked or orders_skipped
    ):
        log.info("ntfy suppressed (silent intraday no-op): %s [%s]", label, run_mode)
        return
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

    # Always append system state snapshot for audit visibility.
    # 2026-05-15: expanded regime info — operator needs to see WHY
    # regime was chosen + WHETHER it just flipped. Three new fields:
    #   transition: cooldown active right after a regime switch (3-bar)
    #   hard_bear:  extreme vol/return forced BEAR (>0.35 ann_vol or
    #               -0.08 20d_ret per kernel.regime.detect_regime)
    #   hurst:      0.5=random, >0.65=trending (MOMENTUM), <0.52=mean-rev
    # Without these, "regime=BEAR" alone can't distinguish a transient
    # mis-fire (low conf) from a real bear (hard_bear=T + high conf).
    ctx_bits: list[str] = [f"regime={regime}"]
    if conf is not None:
        try:
            ctx_bits.append(f"conf={float(conf):.2f}")
        except (TypeError, ValueError):
            pass
    rs = getattr(ctx, "regime_state", None)
    if rs is not None:
        if getattr(rs, "in_transition", False):
            ctx_bits.append("transition=T")
        if getattr(rs, "hard_bear", False):
            ctx_bits.append("hard_bear=T")
        h = getattr(rs, "hurst", None)
        if h is not None:
            try:
                ctx_bits.append(f"hurst={float(h):.2f}")
            except (TypeError, ValueError):
                pass
        hr = getattr(rs, "hurst_regime", None)
        if hr and hr != "AMBIGUOUS":
            # Only surface non-default Hurst route to keep body terse
            ctx_bits.append(f"hurst_reg={hr[:3]}")
    ctx_bits.append(f"held={n_held}")
    if eq is not None:
        try:
            ctx_bits.append(f"eq=${float(eq):,.0f}")
        except (TypeError, ValueError):
            pass
    parts.append(" ".join(ctx_bits))

    # 2026-05-19 user mandate: surface shadow-model output in ntfy so
    # operator sees what the candidate (e.g. PatchTST) would have picked
    # vs. live XGB primary. Compact: one segment per shadow with top-3
    # picks, top-10 overlap, and Spearman vs primary.
    shadow_summary = list(getattr(ctx, "_shadow_summary", []) or [])
    for ss in shadow_summary:
        try:
            top3 = "/".join(ss.get("top3", [])[:3]) or "?"
            overlap = ss.get("top10_overlap", "?")
            n_cand = ss.get("n_candidates", "?")
            rho = ss.get("spearman_vs_primary", float("nan"))
            try:
                rho_str = f"{float(rho):+.2f}" if rho == rho else "n/a"  # NaN check
            except (TypeError, ValueError):
                rho_str = "n/a"
            parts.append(
                f"SHADOW[{ss.get('name','?')}] top3={top3} "
                f"top10∩prim={overlap}/10 ρ={rho_str} n={n_cand}"
            )
        except Exception as exc:
            log.warning("ntfy shadow segment failed: %s", exc)

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
    parser.add_argument("--broker", choices=["paper", "alpaca", "alpaca-paper", "alpaca-shorts", "ibkr", "readonly-alpaca"], default="paper")
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
    parser.add_argument("--strategy-config-name", default="strategy_config.json",
                        help="Side config filename inside the strategy dir "
                             "(default: strategy_config.json). Use this to test "
                             "alternate configs without touching the live one.")
    args = parser.parse_args()

    # 2026-04-27: thread broker tag through to UniverseContext for
    # broker-isolated live_state read.
    # 2026-05-19 ORDER-OF-OPS FIX: must construct broker BEFORE loading
    # the strategy so the broker's class-level broker_name (NOT the CLI arg)
    # threads through. e.g. --broker readonly-alpaca produces a wrapper
    # whose broker_name="alpaca_shadow" — using args.broker would feed the
    # un-wrapped CLI tag and break state-path isolation. Old order also
    # leaked the raw CLI arg into ALLOWED_BROKERS validation.
    initial_cash = 100_000  # placeholder; real value set after config load below
    broker = _get_broker(args.broker, initial_cash=initial_cash)
    config, models, strategy_dir = _load_strategy_multi(
        args.strategy, broker_name=broker.broker_name,
        config_name=args.strategy_config_name,
    )
    # Reconstruct broker with real initial_cash from config (PaperBroker
    # needs it; AlpacaBroker / wrapper ignore it but cheap to re-init).
    initial_cash = config.get("initial_cash", 100_000)
    if isinstance(broker, PaperBroker):
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
