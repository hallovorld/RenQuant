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
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from .broker import BaseBroker
from .alpaca_broker import AlpacaBroker
from .ibkr_broker import IBKRBroker
from .paper_broker import PaperBroker
from .alerts import AlertEvent, post_ntfy_alert, stable_alert_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("live.runner")


# ── Preflight / dry-run mode (GOAL-5 AC5, PR #565 codex CR) ─────────────────────
#
# `--broker readonly-alpaca --once` only constrains BROKER writes — the runner
# still opens/creates the runs DB, allocates a run id, persists live_state, runs
# the score-distribution DB writer, and emits an ntfy on every cycle. That is NOT
# a safe read-only operational probe. `--preflight` drives the SAME funnel to a
# decision line but GUARANTEES zero of: DB/state persistence, order placement,
# promotion, and notification — enforced structurally (not scattered ifs):
#   • persistence + orders + promotion — RunnerAdapter.commit() is the single
#     write chokepoint (every broker.place_order / record_* / save_live_state /
#     L6-sidecar / trade-log write lives there); dry-run never calls it, AND the
#     adapter opens NO runs DB (ctx._db is None ⇒ ScoreDistributionJob no-ops and
#     the data/runs_*.db file is never created), AND meta-label parquet capture is
#     forced off. commit() also refuses (notes the guard + returns) if ever
#     entered with a guard attached — defense in depth.
#   • notifications — the single ntfy send chokepoint (_post_ntfy_with_retries)
#     is intercepted by the active guard: any attempted send is recorded and
#     suppressed, so nothing leaves the process.
# A machine-readable `preflight_attestation:` line is emitted so the dawn shell
# guard can verify no-write/no-notify and fail closed otherwise. The guard's
# flags flip true iff a boundary is actually hit, so the attestation is
# self-attesting: a clean probe is `{...all false..., reached_decision:true}`.


class PreflightGuard:
    """Records whether any real side effect was reached during a dry-run.

    A clean preflight leaves every mutation flag False and reached_decision
    True. Any boundary that is actually hit flips its flag, which makes
    ``clean()`` False and is reflected verbatim in the attestation — so a
    miswired (or regressed) dry-run cannot silently claim to be side-effect
    free.
    """

    __slots__ = ("persisted", "notified", "promoted", "ordered", "reached_decision")

    def __init__(self) -> None:
        self.persisted = False
        self.notified = False
        self.promoted = False
        self.ordered = False
        self.reached_decision = False

    def note_persist(self) -> None:
        self.persisted = True

    def note_notify(self) -> None:
        self.notified = True

    def note_promote(self) -> None:
        self.promoted = True

    def note_order(self) -> None:
        self.ordered = True

    def note_decision(self) -> None:
        self.reached_decision = True

    def payload(self) -> dict[str, bool]:
        return {
            "persisted": self.persisted,
            "notified": self.notified,
            "promoted": self.promoted,
            "ordered": self.ordered,
            "reached_decision": self.reached_decision,
        }

    def clean(self) -> bool:
        """True iff a decision was reached and NO mutation boundary was hit."""
        return self.reached_decision and not (
            self.persisted or self.notified or self.promoted or self.ordered
        )


# Process-wide active guard. Single-threaded --once runner, so a module global
# is a safe way to thread the guard to the ntfy send chokepoint without touching
# every intermediate signature. Set/reset around the dry-run in
# _run_once_multi_pipeline (try/finally).
_ACTIVE_PREFLIGHT_GUARD: "PreflightGuard | None" = None


def _emit_preflight_attestation(guard: "PreflightGuard") -> None:
    """Print the machine-readable attestation line consumed by the shell guard."""
    line = "preflight_attestation: " + json.dumps(guard.payload())
    # stdout so the dawn wrapper's `> "$LOG"` capture gets it; also logged.
    print(line, flush=True)
    log.info(line)


_BUY_SIDE_PREFLIGHT_CHECKS = frozenset({
    "P-MODEL-ARTIFACT",
    "P-PANEL-CONTRACT",
    "P-WF-GATE",
    "P-REGIME-IC",
    "P-BEST-ITER",
    "P-CONFIG-FP",
    "P-WATCHLIST",
    "P-SECTOR-MAP",
    "P-CORR-METADATA",
    "P-FEATURE-COVER",
    "P-RUN-ID",
    "P-META-LABEL",
    "P-CALIBRATOR-HEALTH",
    "P-CALIBRATOR-FLAT-REGION",
})


def _failed_preflight_check_names(message: str) -> set[str]:
    """Extract failed preflight check names from PreflightFailed text."""
    names: set[str] = set()
    for line in str(message).splitlines():
        stripped = line.strip()
        if not stripped.startswith("✗"):
            continue
        match = re.search(r"\b(P-[A-Z0-9-]+)\b", stripped)
        if match:
            names.add(match.group(1))
    return names


def _is_buy_side_preflight_block(message: str) -> bool:
    """True when all failed preflight checks are model/buy admission gates."""
    failed = _failed_preflight_check_names(message)
    return bool(failed) and failed.issubset(_BUY_SIDE_PREFLIGHT_CHECKS)


def _preflight_alert_payload(label: str, run_mode: str, message: str) -> dict[str, Any]:
    """Classify a preflight failure into operator-alert severity.

    Model-contract failures intentionally block new buys and are usually
    resolved by retrain/promote. They should remain visible but not page like a
    broker/execution outage. Broker/preflight-code failures remain urgent.
    """
    import os as _os  # noqa: PLC0415

    failed = sorted(_failed_preflight_check_names(message))
    buy_side_block = _is_buy_side_preflight_block(message)
    tag = "BUY-BLOCKED" if buy_side_block else "PREFLIGHT-FAIL"
    if buy_side_block:
        body = (
            "No orders placed; full/buy blocked by model-contract preflight. "
            "Sell-only/risk exits should still be run by the daily wrapper.\n"
            f"{message}"
        )
        priority = "default"
        taxonomy = "DECISION"
        key_parts = ("preflight-buy-blocked", label, run_mode, ",".join(failed))
    else:
        body = str(message)
        priority = "urgent"
        taxonomy = "ACTION_REQUIRED"
        key_parts = (
            "preflight", label, run_mode,
            ",".join(failed), str(message)[:500],
        )
    try:
        cooldown = int(_os.environ.get("RENQUANT_PREFLIGHT_NTFY_COOLDOWN_SECONDS", "21600"))
    except ValueError:
        cooldown = 21600
    return {
        "title": f"{label} [{run_mode}] {tag}",
        "body": body,
        "priority": priority,
        "taxonomy": taxonomy,
        "key": stable_alert_key(*key_parts),
        "cooldown_seconds": max(0, cooldown),
    }


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
        # state-file isolation: live_state.alpaca_shadow.json + runs.alpaca_shadow.db
        # never collide with prod live_state.alpaca.json. For renquant_104
        # the CLI defaults this broker to strategy_config.shadow.json so the
        # panel scorer also swaps (HF PatchTST instead of XGB).
        #
        # 2026-07-27 shadow_blend rail: the wrapper's tag (and therefore
        # its state-file lane) is selected by RENQUANT_READONLY_TAG,
        # resolved inside ReadOnlyBrokerWrapper.__init__ — env threading
        # passes through the orchestrator live-bridge subprocess boundary
        # unchanged, so `--broker readonly-alpaca` stays the single CLI
        # broker type for every readonly shadow lane. Unset env =
        # legacy "alpaca_shadow" (byte-identical).
        from .broker_readonly import ReadOnlyBrokerWrapper  # noqa: PLC0415
        real = AlpacaBroker(paper=False)
        return ReadOnlyBrokerWrapper(real)
    else:
        raise ValueError(f"Unknown broker: {broker_type}")


def _log_trade(strategy_dir: Path, strategy_name: str, record: dict) -> None:
    """Append trade record to daily log file."""
    log_dir = REPO_ROOT / "live" / "logs" / strategy_name
    log_dir.mkdir(parents=True, exist_ok=True)
    from live.clock import trading_date  # noqa: PLC0415

    log_file = log_dir / f"{trading_date().isoformat()}.json"  # P0.3: exchange date

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


def _broker_held_tickers(broker: BaseBroker | None) -> set[str] | None:
    """Return currently held symbols from broker positions, or None if unavailable."""
    if broker is None:
        return None
    try:
        positions = broker.get_all_positions() or []
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "UNIVERSE-HELD-SOURCE: broker.get_all_positions failed before "
            "universe load (%s) — falling back to broker-isolated live_state",
            exc,
        )
        return None
    held: set[str] = set()
    for pos in positions:
        try:
            qty = float(pos.get("qty", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue
        sym = pos.get("symbol") if isinstance(pos, dict) else None
        if sym and np.isfinite(qty) and abs(qty) > 1e-9:
            held.add(str(sym))
    return held


def _load_strategy_multi(
    strategy_name: str, broker_name: str | None = None,
    config_name: str = "strategy_config.json",
    broker: BaseBroker | None = None,
    config_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict, Path]:
    """Load multi-stock strategy config and per-stock kernel model artifacts."""
    strategy_dir = REPO_ROOT / "backtesting" / strategy_name
    config_file = Path(config_path).expanduser() if config_path else strategy_dir / config_name
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    if not config_file.exists():
        log.error("Strategy config not found: %s", config_file)
        sys.exit(1)

    config = json.loads(config_file.read_text())
    config_name = config_file.name

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
            config=config,
            strategy_dir=strategy_dir,
            broker_name=broker_name,
            held_tickers=_broker_held_tickers(broker),
        )
        LoadUniverseJob().run(uctx)
        models = uctx.loaded_models
        config["_universe_rejections"] = dict(uctx.rejections)
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
                from live.clock import trading_date as _td  # noqa: PLC0415

                age = (_td() - _dt.datetime.strptime(trained_date, "%Y-%m-%d").date()).days  # P0.3
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
    config["_strategy_config_name"] = config_name
    config["_strategy_config_path"] = str(config_file)
    config.setdefault("_universe_rejections", {})
    return config, models, strategy_dir


def _run_once_multi_pipeline(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
    sell_only: bool,
    use_intraday_prices: bool = False,
    dry_run: bool = False,
) -> None:
    """Create RunnerAdapter + InferencePipeline and execute one trading cycle.

    ``dry_run=True`` (the ``--preflight`` probe) drives the funnel to the
    decision line but GUARANTEES no persistence / order / promotion / ntfy
    side effect and emits a ``preflight_attestation:`` line. See the
    ``PreflightGuard`` block at the top of this module for the enforcement
    contract.
    """
    global _ACTIVE_PREFLIGHT_GUARD  # noqa: PLW0603
    _load_kernel(strategy_dir)  # ensure kernel/ is importable

    from kernel.pipeline import InferencePipeline, SellOnlyPipeline  # noqa: PLC0415
    from adapters.runner import RunnerAdapter                          # noqa: PLC0415

    guard: PreflightGuard | None = PreflightGuard() if dry_run else None
    if dry_run:
        _ACTIVE_PREFLIGHT_GUARD = guard
    try:
        return _run_once_multi_pipeline_inner(
            config, models, broker, strategy_dir, sell_only,
            use_intraday_prices, dry_run, guard,
            InferencePipeline, SellOnlyPipeline, RunnerAdapter,
        )
    finally:
        if dry_run:
            _ACTIVE_PREFLIGHT_GUARD = None


# GOAL-9 fleet callsigns (operator-chosen scheme 2026-08-04: 功能缩写 —
# R=reversal, C=classifier, S=slow momentum, f=fast momentum; lowercase f
# deliberate to keep the fast/slow distinction visible in caps-only fonts).
# The PROD lane (broker "alpaca") is RS by composition; its title is not
# prefixed here (live titles keep their existing format).
LANE_CALLSIGNS = {
    "alpaca_shadow_blend": "RC",
    "alpaca_shadow_blend_mom": "RSs",
    "alpaca_shadow_blend_mom_fast": "Rf",
    "alpaca_shadow_blend_rb_mom": "RCS",
    "alpaca_shadow_blend_rb_fast": "RCf",
    # 2026-08-18 vol-window lane. The letters above name a SCORING
    # composition; this lane's scoring is the prod blend and what it varies is
    # the volatility gate (vol_window_license: inside ON ∧ ¬BEAR the
    # top-decile keeps buy admission), so it gets its own letter rather than a
    # composition string. Rename freely — nothing keys off the value, only off
    # a lane HAVING one (test_every_running_shadow_lane_has_a_callsign).
    "alpaca_shadow_vol_window": "V",
}


def _readonly_label_prefix(broker_name: str) -> str:
    """Log/ntfy title prefix for readonly shadow-lane brokers.

    2026-07-27 shadow_blend rail: generalized from the literal
    ``== "alpaca_shadow"`` check to any tag starting with "alpaca_shadow"
    (see live/broker_readonly.py tag parameterization). Contract:
    - legacy tag "alpaca_shadow" → "[READONLY]" EXACTLY (byte-identical
      legacy titles; existing consumers/tests match this literal);
    - any other alpaca_shadow* tag → "[READONLY][<TAG-UPPER>]" so blend vs
      legacy lanes are distinguishable in ntfy titles while STILL starting
      with the literal "[READONLY]" — _notify_decision's is_shadow check
      (label.startswith("[READONLY]")) keeps classifying both lanes as
      shadow/hypothetical;
    - non-shadow brokers → "" (no prefix).
    """
    if broker_name == "alpaca_shadow":
        return "[READONLY]"
    if broker_name.startswith("alpaca_shadow"):
        # 2026-08-04 operator directive ("简练,人话"): fleet CALLSIGNS instead
        # of the shouting full tag. The prefix MUST keep starting with the
        # literal "[READONLY]" — _notify_decision's is_shadow classification
        # depends on it (a shadow message must never classify as live).
        # Unknown future tags fall back to the full tag (never a bare
        # "[READONLY]", which would collide with the legacy lane's contract).
        # The fallback is the tag SHOUTED, e.g. "[READONLY][ALPACA_SHADOW_VOL_WINDOW]"
        # — 30 characters of title before the reader reaches a single decision.
        # It ran that way from 2026-08-18 to 2026-08-24 because adding a lane
        # silently degrades here instead of failing: `.get` with a permissive
        # default is invisible. Kept (a title must never be MISSING a lane
        # marker) but no longer relied upon — the callsign map is now asserted
        # to cover every lane daily_104.sh actually launches, so the fallback
        # is a backstop rather than the de-facto path for new lanes.
        return f"[READONLY][{LANE_CALLSIGNS.get(broker_name, broker_name.upper())}]"
    return ""


def _run_once_multi_pipeline_inner(
    config, models, broker, strategy_dir, sell_only,
    use_intraday_prices, dry_run, guard,
    InferencePipeline, SellOnlyPipeline, RunnerAdapter,
):  # noqa: ANN001, ANN201
    run_mode = "sell-only" if sell_only else "full"
    if use_intraday_prices:
        run_mode += " (intraday)"
    if dry_run:
        run_mode += " (preflight)"
    sep = "=" * 62
    log.info(sep)
    label = strategy_dir.name.upper().replace("_", "-")
    # 2026-05-19: shadow run uses ReadOnlyBrokerWrapper (broker_name=alpaca_shadow).
    # Prefix label so log + ntfy title carry the broker-mode distinction. Hard
    # isolation per user mandate "隔离干净" (actual state-file isolation is
    # driven by broker_name, threaded separately into _load_strategy_multi —
    # this label only affects the human-facing log/ntfy prefix).
    #
    # 2026-07-01 RENAMED from "[SHADOW]" to "[READONLY]" (operator incident:
    # a "[SHADOW]...BUY OXY" ntfy title was misread as "the shadow PatchTST
    # MODEL recommends OXY" — the [SHADOW] title actually only meant "this
    # run executed via the readonly/shadow BROKER", completely orthogonal to
    # which scoring model was primary; the decision was made by the
    # production XGB model, echoed through the readonly broker). The ntfy
    # BODY's per-model "SHADOW[name]"/"SHADOW-PICKS[name]" segments are an
    # unrelated concept (an alternate model's own view). Renaming the
    # title token removes the collision. Repo-wide grep on 2026-07-01 found
    # no consumer that pattern-matches the literal "[SHADOW]" title
    # substring outside this module's own `is_shadow` check below and this
    # module's own tests — safe to rename outright (Option A).
    #
    # 2026-07-27 shadow_blend rail: prefix generalized to ANY broker tag
    # starting with "alpaca_shadow" via _readonly_label_prefix — legacy tag
    # keeps the byte-identical "[READONLY]" title; other lanes (e.g.
    # alpaca_shadow_blend) get "[READONLY][ALPACA_SHADOW_BLEND]" so they
    # stay distinguishable in ntfy while is_shadow still matches.
    label = _readonly_label_prefix(getattr(broker, "broker_name", "")) + label
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
            # 2026-07-27 shadow_blend rail: startswith (not ==) so every
            # readonly shadow lane (alpaca_shadow, alpaca_shadow_blend, …)
            # gets the shadow preflight-strictness policy, mirroring the
            # legacy lane exactly. Tag prefix enforced by
            # live/broker_readonly.py validate_readonly_tag.
            is_shadow_broker = getattr(broker, "broker_name", "").startswith("alpaca_shadow")
            preflight_strict = bool(preflight_cfg.get("strict", True))
            if is_shadow_broker:
                preflight_strict = bool(preflight_cfg.get("shadow_strict", False))
            run_preflight(
                config, broker=broker, strategy_dir=strategy_dir,
                broker_name=getattr(broker, "broker_name", None),
                strict=preflight_strict,
                run_mode=run_mode,
            )
        except PreflightFailed as exc:
            if dry_run:
                # Probe path: a preflight-contract failure (buy-side model
                # block OR a hard broker/state failure) means the funnel did
                # NOT reach a normal decision line — exactly a daily-killer the
                # dawn probe must surface 8h early. Log it (analyzer + operator
                # see the P-* class), NEVER notify (a clean probe emits no
                # ntfy), emit the attestation with reached_decision:false so
                # the shell guard fails closed, and exit non-zero.
                buy_side = _is_buy_side_preflight_block(str(exc))
                log.error(
                    "PRE-FLIGHT FAILED (preflight probe, %s; no orders, no ntfy):\n%s",
                    "buy-side model-contract block" if buy_side else "hard failure",
                    exc,
                )
                _emit_preflight_attestation(guard)
                raise SystemExit(2) from exc
            import os as _os  # noqa: PLC0415
            suppress_preflight_ntfy = (
                _os.environ.get("RENQUANT_SUPPRESS_PREFLIGHT_NTFY") == "1"
            )
            log_fn = log.warning if suppress_preflight_ntfy else log.error
            log_fn("PRE-FLIGHT FAILED — aborting cron, no orders placed:\n%s", exc)
            if suppress_preflight_ntfy:
                log.info("preflight ntfy suppressed by RENQUANT_SUPPRESS_PREFLIGHT_NTFY")
            else:
                topic = _os.environ.get("RENQUANT_NTFY_TOPIC", "renquant")
                alert = _preflight_alert_payload(label, run_mode, str(exc))
                _post_ntfy_with_retries(
                    f"https://ntfy.sh/{topic}",
                    title=alert["title"],
                    body=alert["body"],
                    priority=alert["priority"],
                    taxonomy=alert["taxonomy"],
                    key=alert["key"],
                    cooldown_seconds=alert["cooldown_seconds"],
                )
            raise SystemExit(2) from exc
        except ImportError as exc:
            # Full/buy runs must never silently trade when the preflight
            # contract itself is unavailable. Sell-only risk exits may proceed.
            if dry_run:
                # A probe that cannot import its own preflight contract is
                # broken — fail closed (attestation reached_decision:false).
                log.error(
                    "P-PREFLIGHT-IMPORT preflight module not importable (preflight "
                    "probe) — no decision reached: %s", exc,
                )
                _emit_preflight_attestation(guard)
                raise SystemExit(2) from exc
            if sell_only:
                log.warning(
                    "preflight module not importable during sell-only; "
                    "proceeding so risk exits can run: %s",
                    exc,
                )
            else:
                log.error(
                    "P-PREFLIGHT-IMPORT preflight module not importable — aborting full/buy "
                    "run fail-closed: %s",
                    exc,
                )
                raise SystemExit(2) from exc
        except Exception as exc:
            # Full/buy runs must fail closed on broken preflight code. Sell-only
            # risk exits are still allowed because they reduce exposure.
            if dry_run:
                log.error(
                    "P-PREFLIGHT-EXCEPTION preflight raised unexpectedly (preflight "
                    "probe) — no decision reached: %s", exc,
                )
                _emit_preflight_attestation(guard)
                raise SystemExit(2) from exc
            if sell_only:
                log.warning(
                    "preflight raised unexpectedly during sell-only; "
                    "proceeding so risk exits can run: %s",
                    exc,
                )
            else:
                log.error(
                    "P-PREFLIGHT-EXCEPTION preflight raised unexpectedly — aborting full/buy "
                    "run fail-closed: %s",
                    exc,
                )
                raise SystemExit(2) from exc

    adapter  = RunnerAdapter(
        config, models, broker, strategy_dir,
        sell_only=sell_only,
        use_intraday_prices=use_intraday_prices,
        preflight=dry_run,
        preflight_guard=guard,
    )
    pipeline = SellOnlyPipeline() if sell_only else InferencePipeline()

    ctx = adapter.make_context()
    pipeline.run(ctx)

    if dry_run:
        # Reached the decision line. Skip commit() (the sole persistence /
        # order / promotion chokepoint) and the decision ntfy entirely — the
        # probe never mutates state or notifies. Emit the attestation for the
        # shell guard. The "DECISION" token keeps the dawn analyzer's
        # completion check satisfied without an ntfy.
        guard.note_decision()
        log.info(
            "%s [%s] PREFLIGHT-DECISION reached — no orders, no state persisted, "
            "no ntfy (dry-run probe)", label, run_mode.upper(),
        )
        _emit_preflight_attestation(guard)
        return

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


# ntfy's documented practical message-body limit is ~4096 bytes (the
# server-side default; see ntfy.sh docs). No truncation guard previously
# existed anywhere in this module — bodies just grew unboundedly with the
# number of orders/exits/shadow segments and risked a SILENT server-side
# cut mid-word (worse than an explicit, honest truncation marker). Added
# 2026-07-01 alongside the new per-shadow-model SHADOW-PICKS[...] segments,
# which can add several hundred bytes per configured shadow model. Budget
# leaves headroom under 4096 for title + ntfy's own framing overhead.
_NTFY_BODY_MAX_BYTES = 3800

#: How many BLOCKED-ROTATION pairs may reach the ntfy BODY. See the block that
#: uses it for why an unbounded list evicted the decision it was annotating.
_ROT_BLOCKED_NTFY_MAX = 3

#: How many action tokens may reach the ntfy TITLE before it summarises.
_TITLE_ACTION_MAX = 3


def _action_headline(
    orders: Iterable[Any],
    exits: Iterable[Any],
    orders_pending: Iterable[Any],
    exits_pending: Iterable[Any],
    exits_failed: Iterable[Any],
    *,
    max_items: int = _TITLE_ACTION_MAX,
) -> str:
    """Terse "what did it actually do" summary for the ntfy TITLE.

    WHY THE TITLE (2026-08-19). The title is the only part of a push
    notification that reliably survives on a phone lock screen; the body is
    collapsed. Prod titles were ``RENQUANT-104 [full] PENDING`` — a status
    word carrying no content — while the SHADOW lanes' titles said
    ``SHADOW-ACTION`` and their bodies were short enough to read whole. The
    operator could therefore read the shadow lanes and NOT read production,
    and asked, reasonably, whether prod was broken. It was not: the decision
    was in the body, behind ~3.3 KB of BLOCKED-ROTATION segments.

    Sizes are included because "BUY PANW" and "BUY PANW x300" are different
    events. Prices are NOT — they are in the body, and the title's budget is
    better spent on more tickers than on cents.
    """
    toks: list[str] = []
    for o in orders:
        tkr = o.get("ticker") if isinstance(o, dict) else getattr(o, "ticker", "?")
        sh = o.get("shares") if isinstance(o, dict) else getattr(o, "shares", "?")
        toks.append(f"BUY {tkr} x{sh}")
    for e in exits:
        tkr = e[0] if isinstance(e, tuple) and len(e) == 2 else getattr(e, "ticker", "?")
        toks.append(f"EXIT {tkr}")
    for o in orders_pending:
        if isinstance(o, dict):
            toks.append(f"BUY {o.get('ticker', '?')} x{o.get('shares', '?')}")
    for e in exits_pending:
        if isinstance(e, dict):
            toks.append(f"EXIT {e.get('ticker', '?')} x{e.get('qty', '?')}")
    # Failed exits go FIRST: a rejected sell is the one thing that may need a
    # human at the broker, so it must not be the token that gets summarised away.
    failed = [
        f"FAILED-EXIT {fe.get('ticker', '?')}" for fe in exits_failed if isinstance(fe, dict)
    ]
    toks = failed + toks
    if not toks:
        return ""
    head = ", ".join(toks[:max_items])
    if len(toks) > max_items:
        head += f" +{len(toks) - max_items}"
    return head


def _truncate_ntfy_body(body: str, max_bytes: int = _NTFY_BODY_MAX_BYTES) -> str:
    """Truncate ``body`` to a UTF-8-safe byte budget, appending a marker.

    Prefer an explicit, honest "...[truncated]" suffix over letting ntfy's
    own transport silently cut the message off mid-word.
    """
    encoded = body.encode("utf-8")
    if len(encoded) <= max_bytes:
        return body
    suffix = " …[truncated]"
    budget = max(max_bytes - len(suffix.encode("utf-8")), 0)
    truncated = encoded[:budget]
    while truncated:
        try:
            text = truncated.decode("utf-8")
            break
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    else:
        text = ""
    return text + suffix


def _post_ntfy_with_retries(
    url: str,
    *,
    title: str,
    body: str,
    priority: str,
    taxonomy: str = "INFO",
    key: str | None = None,
    cooldown_seconds: int = 0,
    force: bool = False,
) -> bool:
    """Best-effort ntfy publish with retry, curl fallback, and dedupe.

    Live trading state is already committed before notification, so this
    function must never raise. A transient ntfy/SSL failure is common enough
    that a single `urlopen(..., timeout=5)` is not acceptable for trade alerts.
    """
    # Dry-run safety net: this is the single ntfy send chokepoint. If a
    # preflight guard is active, record the attempt and suppress the send so
    # NOTHING leaves the process (defense in depth — the dry-run path already
    # skips the decision ntfy). A caught send flips guard.notified → the
    # attestation reports notified:true → the shell guard fails closed.
    _guard = _ACTIVE_PREFLIGHT_GUARD
    if _guard is not None:
        _guard.note_notify()
        log.info("PREFLIGHT: ntfy send suppressed (title=%r)", title)
        return False
    event = AlertEvent(
        taxonomy=taxonomy,
        title=title,
        body=body,
        key=key,
        priority=priority,
        cooldown_seconds=cooldown_seconds,
        force=force,
    )
    return post_ntfy_alert(url, event, logger=log)


#: Buy-leg reasons that count as a POST-SCORING signal decline, mirroring
#: `renquant_pipeline.kernel.pipeline.signal_direction`. Literals rather than an
#: import: this is the umbrella notification surface and must not reach across
#: the repo boundary into pipeline internals.
_ROTATION_SIGNAL_DECLINE_REASONS = frozenset({
    "nonpositive_expected_return_no_long",
    "negative_raw_signal_no_long",
})


def _no_trade_reason(ctx) -> str:
    """Pure rollup of "why no trade" for a decision cycle.

    Lifted from a closure inside ``_notify_decision`` 2026-06-01 so the
    priority contract (codex PR #48 review #2) can be regression-tested
    without constructing a full ntfy stack.

    Priority — surfaces the BINDING constraint, not the FIRST upstream
    drop. Old ordering put ``risk_gate_vol_dropped`` ahead of admission/QP
    and surfaced "no trade (vol_dropped(10))" even when 72 of 82
    candidates survived the vol gate and were blocked downstream by
    admission + QP infeasibility.
    """
    def _rotation_signal_block(c) -> "tuple[str, int] | None":
        """The DOMINANT buy-leg signal block, named exactly, with its own count.

        These are binding POST-scoring reasons: the candidates survived every
        gate, were scored, and a signal declined them. So they outrank the
        pre-scoring vol drop below.

        NAMED, NOT POOLED (2026-08-20, second pass). The first version of this
        helper returned one total and labelled it
        `rotation_nonpositive_expected_return(60)`. On that day's real payload
        only 13 of the 60 were nonpositive-expected-return; the other 47 were
        `negative_raw_signal` — a DIFFERENT gate, and 25 of those names had a
        POSITIVE expected return and were declined on panel score alone. A
        pooled label would have reproduced, one layer finer, the exact defect
        this function was being fixed for: a message naming a cause that is not
        the cause.

        Ties resolve by the FULL reason string, not its first character. The
        first version wrote `-ord(kv[0][0])`, which compares one character —
        and both current reasons start with "n", so equal counts fell back to
        dict insertion order and reversing `rotations_blocked` could change the
        notification. The PR claimed determinism it did not have
        [codex on RenQuant#599].

        EXACT ALLOWLIST, NOT SUBSTRING MATCHING [same review]. The first
        version tested `"expected_return" in reason`, which would classify a
        future `missing_expected_return` as an economic decline.

        This IS an enumerated allowlist, and those go stale — but the polarity
        is the safe one here, unlike orch#1013's admit predicate. There, an
        unlisted order type was silently DROPPED from a collection, so the
        default had to be "include". Here an unlisted reason merely fails to be
        ELEVATED above the vol-gate fall-through, so the default is "do not
        claim this is the cause" — which is what this function exists to
        guarantee. A genuinely new signal-decline reason must be added here as
        well as in `signal_direction.py`: a documented maintenance point, not a
        silent misclassification.
        """
        counts: dict[str, int] = {}
        for rb in (getattr(c, "rotations_blocked", []) or []):
            if not isinstance(rb, dict):
                continue
            reason = str(rb.get("reason", ""))
            if reason in _ROTATION_SIGNAL_DECLINE_REASONS:
                counts[reason] = counts.get(reason, 0) + 1
        if not counts:
            return None
        # Highest count wins; ties resolve by the FULL reason string, so the
        # output cannot depend on the order of `rotations_blocked`.
        top = max(counts.values())
        reason = min(r for r, n in counts.items() if n == top)
        return f"rotation_{reason}", top

    if getattr(ctx, "bear_only", False):
        return "bear_only"
    rs = getattr(ctx, "regime_state", None)
    if rs is not None and getattr(rs, "in_transition", False):
        return "transition_window"
    counters = getattr(ctx, "counters", {}) or {}
    specific_blocks = (
        # Earliest-stage fail-closed (no scorer / no calibration)
        ("panel_scoring_fail_closed", "panel_scoring_fail_closed"),
        ("qp_mu_contract_block", "qp_mu_contract_block"),
        # Mid-pipeline: regime admission cleared all buy candidates
        ("regime_admission_blocked", "regime_admission_blocked"),
        # Late-pipeline: QP solver could not produce orders (binding)
        ("qp_infeasible", "qp_infeasible"),
        ("qp_missing_solution", "qp_missing_solution"),
        ("qp_optimal_no_signal", "qp_optimal_no_signal"),
        ("qp_other_nonoptimal", "qp_other_nonoptimal"),
        # Pre-scoring drop — only the cause when nothing later applied.
        #
        # 2026-08-20: it applied anyway, through a hole. On that session the
        # message read `no trade (risk_gate_vol_dropped(30))` while the run's
        # own funnel said `verdict=ECONOMIC_NO_TRADE structural=False
        # candidates_final=84 buys=0` and 61 rotations were blocked
        # `nonpositive_expected_return_no_long`. 84 candidates WERE scored and
        # the model wanted none of them — the vol gate was not binding.
        #
        # The 2026-06-01 rewrite (see the docstring) fixed the ORDERING for
        # exactly this failure, but the rotation-side economic block was never
        # given a counter, so the loop fell through to the last entry and
        # blamed the vol gate again. The operator read that message and moved
        # to loosen a live risk limit for a reason that was not the cause.
        ("_rotation_signal", None),   # label comes from the helper, see above
        ("risk_gate_vol_dropped", "risk_gate_vol_dropped"),
    )
    for key, label in specific_blocks:
        if key == "_rotation_signal":
            hit = _rotation_signal_block(ctx)
            if hit:
                return f"{hit[0]}({hit[1]})"
            continue
        n = int(counters.get(key, 0) or 0)
        if n > 0:
            return f"{label}({n})"
    if counters.get("qp_blocked_buys", 0):
        if getattr(ctx, "skip_buys", False):
            return "drawdown_halt"
        if getattr(ctx, "buy_blocked", False):
            return "buy_blocked"
    qp_reasons = (
        ("qp_skipped_band", "qp_no_trade_band"),
        ("qp_delta_below_min_dw", "qp_delta_below_min_dw"),
        ("qp_zero_shares", "qp_zero_shares"),
        ("qp_no_buy_delta", "qp_no_buy_delta"),
        ("qp_not_selected", "qp_not_selected"),
    )
    qp_counts = [
        (int(counters.get(key, 0) or 0), label)
        for key, label in qp_reasons
        if int(counters.get(key, 0) or 0) > 0
    ]
    if qp_counts:
        n_qp, qp_label = max(qp_counts, key=lambda item: item[0])
        return f"{qp_label}({n_qp})"
    if getattr(ctx, "skip_buys", False):
        return "drawdown_halt"
    if getattr(ctx, "buy_blocked", False):
        return "buy_blocked"
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
    import os  # noqa: PLC0415
    # IMPORTANT: read the BROKER-CONFIRMED order list (orders_placed),
    # populated by adapter.commit AFTER the duplicate-order guard and
    # broker submission. `ctx.orders` is merely the pipeline intent and
    # will include orders that the guard blocked (e.g. TSM on 2026-04-23
    # 21:22 — a pending pre-market order from 21:04 caused a repeat
    # pipeline cycle to emit a second BUY TSM which the guard correctly
    # skipped, but the old ntfy misreported it as a trade).
    # is_shadow == "this cycle ran via the readonly/shadow BROKER" (no live
    # orders reach Alpaca). Renamed from "[SHADOW]" to "[READONLY]" 2026-07-01
    # to stop it colliding with the unrelated per-model "SHADOW[name]" /
    # "SHADOW-PICKS[name]" body segments below (an alternate scoring MODEL's
    # own view, orthogonal to broker mode — see module docstring incident).
    is_shadow      = label.startswith("[READONLY]")
    orders         = list(getattr(ctx, "orders_placed",  []) or [])
    orders_pending = list(getattr(ctx, "orders_pending", []) or [])
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
    exits_pending = list(getattr(ctx, "exits_pending", []) or [])
    exits_failed = list(getattr(ctx, "exits_failed", []) or [])
    regime         = getattr(ctx, "regime", None) or "?"
    conf   = getattr(ctx, "confidence", None)
    eq     = getattr(ctx, "portfolio_value", None)
    n_held = len(getattr(ctx, "holdings", {}) or {})

    # Module-level _why_no_trade is testable via direct import (see
    # tests/test_no_trade_priority.py). Closure removed 2026-06-01.

    def _why_no_trade() -> str:
        return _no_trade_reason(ctx)

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

    for o in orders_pending:
        if isinstance(o, dict):
            tkr = o.get("ticker", "?")
            shares = o.get("shares", "?")
            status = o.get("status", "pending")
            oid = o.get("order_id", "?")
            parts.append(f"PENDING-BUY {tkr} x{shares} ({status} {oid})")
    for e in exits_pending:
        if isinstance(e, dict):
            tkr = e.get("ticker", "?")
            qty = e.get("qty", "?")
            status = e.get("status", "pending")
            oid = e.get("order_id", "?")
            rsn = e.get("exit_type") or e.get("reason") or "sell"
            parts.append(f"PENDING-EXIT {tkr} x{qty} ({rsn}; {status} {oid})")

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
    #
    # CAPPED 2026-08-19. Bug L's fix was right and stays; what changed is the
    # VOLUME. On 2026-08-19 the live run emitted 60 BLOCKED-ROTATION segments
    # (~3.3 KB) against 2 real order segments, so the body hit
    # _NTFY_BODY_MAX_BYTES and the "…[truncated]" cut fell INSIDE the blocked
    # list — the trailing regime/equity context never reached the operator at
    # all. A diagnostic that evicts the decision it annotates is worse than no
    # diagnostic: the operator reported being unable to see what prod did.
    # The count is what carries the signal here, not each pair; the full list
    # stays in the run log, which is where a 60-item enumeration belongs.
    #
    # A PREFILTER ENTRY IS NOT A BLOCKED ROTATION (2026-08-20). The operator
    # read the day's message and asked, reasonably, why every rotation was
    # failing and why the sell leg was NULL:
    #
    #   BLOCKED-ROTATION None→APH (nonpositive_expected_return_no_long)
    #
    # Nothing about rotation had failed. `BuildPairsTask` declines a buy
    # CANDIDATE before any sell leg is chosen, and the producer writes
    # `sell=None` with `stage="prefilter"` deliberately — its own comment says
    # "no pair exists yet ... so monitors can tell the stages apart"
    # [renquant-pipeline kernel/pipeline/task_rotation.py]. The data was
    # correct and self-describing; this renderer ignored `stage` and invented a
    # rotation that never existed.
    #
    # The split that day was 60 prefilter + exactly ONE genuine paired block
    # (SPG→CRWD, correlation_guard, appended by ValidatePairsTask after all 60).
    # That one sat at position 61, inside the "+58 more" — so the flat list
    # spent all three visible slots on rotations that never happened while
    # hiding the only rotation that did. Splitting the kinds fixes both
    # directions, which is why `paired` keeps its own cap rather than sharing
    # one with the declines.
    #
    # Note `rb.get("sell", "?")` could never produce its own default: the key
    # is PRESENT with value None, so `.get` returns None and the "?" is dead
    # code. A default only fires on a missing key, never on a null value.
    rot_blocked = [rb for rb in (getattr(ctx, "rotations_blocked", []) or [])
                   if isinstance(rb, dict)]
    paired: list[dict] = []
    prefiltered: list[dict] = []
    for rb in rot_blocked:
        pre = (rb.get("stage") == "prefilter") or not rb.get("sell")
        (prefiltered if pre else paired).append(rb)

    for rb in paired[:_ROT_BLOCKED_NTFY_MAX]:
        parts.append(
            f"BLOCKED-ROTATION {rb.get('sell') or '?'}→{rb.get('buy') or '?'} "
            f"({rb.get('reason') or '?'})"
        )
    if len(paired) > _ROT_BLOCKED_NTFY_MAX:
        parts.append(
            f"BLOCKED-ROTATION +{len(paired) - _ROT_BLOCKED_NTFY_MAX} more "
            f"({len(paired)} total — see run log)"
        )

    if prefiltered:
        # ONE segment, not N. These are single tickers, so the count and the
        # per-reason split carry the signal; the enumeration belongs in the run
        # log. This also shrinks the body rather than growing it, which matters
        # — the 61-entry version blew past _NTFY_BODY_MAX_BYTES and truncated
        # away the regime/equity tail the operator actually reads.
        pf_counts: dict[str, int] = {}
        for rb in prefiltered:
            r = str(rb.get("reason") or "?")
            pf_counts[r] = pf_counts.get(r, 0) + 1
        # Count desc, then the FULL reason string asc — the same determinism
        # lesson as #599; sorting by count alone would let the payload order
        # decide what the operator sees.
        why = ", ".join(f"{r} {n}" for r, n in
                        sorted(pf_counts.items(), key=lambda kv: (-kv[1], kv[0])))
        names = [str(rb.get("buy") or "?") for rb in prefiltered[:_ROT_BLOCKED_NTFY_MAX]]
        more = len(prefiltered) - len(names)
        shown = ", ".join(names) + (f" +{more} more" if more > 0 else "")
        parts.append(f"DECLINED-BUY x{len(prefiltered)} ({why}) — {shown}")

    has_trade = bool(orders or exits)
    has_pending = bool(orders_pending or exits_pending)
    # Anything the operator should see counts as "actionable" for the
    # silent-if-quiet gate. A bare-quiet cycle = no orders, no exits, no
    # failed exits, no unmanaged broker position, no rotation block, no
    # skipped intent. When silent_if_quiet=True (intraday sell-only),
    # those cycles return without sending ntfy.
    if silent_if_quiet and not (
        has_trade or has_pending or exits_failed or non_wl_holds
        or rot_blocked or orders_skipped
    ):
        log.info("ntfy suppressed (silent intraday no-op): %s [%s]", label, run_mode)
        return
    # If the guard blocked every intent (orders all skipped), the cycle
    # produced no real trade — surface the skip reason prominently so
    # the user doesn't mistake a blocked-duplicate for a fresh buy.
    if not has_trade and not has_pending and not exits_failed:
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

    # Surface comparison-model output in ntfy so the operator sees what
    # the readonly candidate/baseline would have picked versus the active
    # primary. Compact: one segment per comparison with top-3
    # picks, top-10 overlap, and Spearman vs primary.
    shadow_summary = list(getattr(ctx, "_shadow_summary", []) or [])
    # in_primary_admitted overlay (2026-07-01): shadow_scoring.py leaves
    # top_picks[*]["in_primary_admitted"] as None because ApplyShadowScoringTask
    # runs inside PanelScoringJob (Phase 3), BEFORE RankingJob/SelectionJob
    # populate ctx.orders. By the time _notify_decision runs, the full
    # pipeline + adapter.commit HAVE run, so `orders` (broker-confirmed
    # orders_placed, extracted above) tells us which tickers primary
    # actually bought today. Compute that set once here.
    admitted_tickers = {
        (o.get("ticker") if isinstance(o, dict) else getattr(o, "ticker", None))
        for o in orders
    }
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
            # KEPT byte-for-byte (backward compat — may be parsed by log
            # tooling/dashboards downstream of this repo).
            parts.append(
                f"SHADOW[{ss.get('name','?')}] top3={top3} "
                f"top10∩prim={overlap}/10 ρ={rho_str} n={n_cand}"
            )
        except Exception as exc:
            log.warning("ntfy shadow segment failed: %s", exc)

        # NEW 2026-07-01 (operator incident: a "[SHADOW]...BUY OXY" ntfy was
        # misread as "the shadow PatchTST model recommends OXY" when it was
        # actually the primary XGB decision echoed through the readonly
        # broker; PatchTST's own view of OXY that day was rank 15/83,
        # z≈+0.88 — not a top pick). Additive line, separate from
        # SHADOW[...] above.
        #
        # 2026-07-01 ROUND 2 (Codex CHANGES_REQUESTED on umbrella PR #426 —
        # see doc/progress/2026-07-01-shadow-ntfy-top-picks.md addendum): a
        # raw rank is STILL not actionable if the artifact is stale or the
        # scored universe is a censored subset (real case: PatchTST observed
        # ~140d stale; rank 1 of an 83-name subset is not rank 1 of the
        # intended ~292-name watchlist). shadow_scoring.py now binds every
        # top_picks list to an `admission` verdict computed from the
        # artifact's own trained_date (or, since ROUND 3, a binding DATA
        # cutoff field when present — see shadow_scoring.py
        # `_compute_admission`) + n_scored/n_expected coverage. Fail closed
        # here too: a summary with no `admission` key at all (e.g. a stale
        # cached ctx from before this fix) is treated as NOT actionable,
        # never assumed safe.
        #
        # 2026-07-01 ROUND 3 (Codex review point 3, "stop calling the line
        # a recommendation or confidence"): the round-1 framing above this
        # comment used to describe this as a "genuine, actionable
        # recommendation with an HONEST confidence indicator" — that
        # framing is exactly what round 2/3 walked back. This is a RAW
        # RANK, gated on freshness/coverage; the trailing bracketed tag
        # below says so without using either word.
        #
        # 2026-07-01 ROUND 4 (Codex CHANGES_REQUESTED — scope narrowing,
        # see doc/progress/2026-07-01-shadow-ntfy-top-picks.md addendum
        # #4): shadow_scoring.py's ``_compute_admission`` now defaults
        # ``actionable`` to False regardless of the computed freshness/
        # coverage verdict (the thresholds are unvalidated operational
        # guesses pending a preregistered shadow evaluation), only flipping
        # to True behind an explicit opt-in config flag. Rendering here
        # trusts whatever ``admission["actionable"]`` says — this module
        # never recomputes the gate. BUT the diagnostic rank/z-score list
        # itself is now ALWAYS rendered (previously suppressed entirely in
        # the NOT ACTIONABLE branch): the PR's original intent was
        # observability ("want to know what shadow will do"), and with
        # ``actionable`` False by default for every cycle, fully
        # suppressing the list would make the whole feature permanently
        # dark. NOT ACTIONABLE still gets its own clearly-labeled line —
        # the ranks are diagnostic-only, never presented as a pick.
        try:
            picks = ss.get("top_picks") or []
            if picks:
                admission = ss.get("admission") or {}
                pick_actionable = bool(ss.get("admission")) and bool(admission.get("actionable"))
                verdict = admission.get("verdict", "unknown")
                run_id = admission.get("run_id", "?")
                n_scored = admission.get("n_scored", n_cand)
                n_expected = admission.get("n_expected")
                cov_str = f"{n_scored}/{n_expected}" if n_expected else f"{n_scored}/?"
                pick_strs = []
                for p in picks:
                    t = p.get("ticker", "?")
                    r = p.get("shadow_rank", "?")
                    z = p.get("shadow_zscore", float("nan"))
                    try:
                        z_str = f"{float(z):+.2f}" if z == z else "n/a"  # NaN check
                    except (TypeError, ValueError):
                        z_str = "n/a"
                    also_bought = ", ALSO-BOUGHT" if t in admitted_tickers else ""
                    pick_strs.append(f"{t}(rank {r}/{n_cand}, z={z_str}{also_bought})")
                if not pick_actionable:
                    if ss.get("admission"):
                        reasons = "; ".join(admission.get("reasons") or []) or "admission check failed"
                    else:
                        reasons = "no admission verdict computed"
                    parts.append(
                        f"SHADOW-PICKS[{ss.get('name', '?')}]: NOT ACTIONABLE "
                        f"({reasons}) [verdict={verdict} cov={cov_str} run={run_id}] "
                        + " ".join(pick_strs)
                        + " [diagnostic rank only, not actionable]"
                    )
                else:
                    parts.append(
                        f"SHADOW-PICKS[{ss.get('name', '?')}]: "
                        f"[{verdict} cov={cov_str} run={run_id}] "
                        + " ".join(pick_strs)
                        + " [raw rank (unvalidated, see freshness verdict)]"
                    )
        except Exception as exc:
            log.warning("ntfy shadow-picks segment failed: %s", exc)

    topic    = os.environ.get("RENQUANT_NTFY_TOPIC", "renquant")
    if is_shadow:
        # 2026-08-04 (93adb20) removed the body sentence "SHADOW/HYPOTHETICAL
        # (no live orders)" on the operator's "简练,人话" directive, reasoning
        # that "the title already carries [READONLY][<callsign>] AND the
        # SHADOW-* tag; the body is decisions + context only."
        #
        # RESTORED TERSELY 2026-08-19, because that reasoning assumed the
        # reader always sees the title. They do not. The operator received
        #     BUY NVDA x5 @ $217.56 | BUY VLO x4 @ $346.30 | regime=BULL_CALM …
        # and asked whether it was real money. It was a shadow lane. A phone
        # notification collapses the body and a copied body carries no title
        # at all, so for 15 days (2026-08-05 .. 2026-08-19, zero disclaimers
        # across all six shadow lanes) every shadow alert read like a fill.
        #
        # The directive was terseness, not the removal of the one token that
        # says "this is not real" — so this is 17 bytes instead of the old 33
        # and stays at parts[0], where it survives both truncation and a
        # body-only copy. Making a shadow alert indistinguishable from a live
        # one is the mirror of the 2026-08-05 conftest incident (a test paged
        # the operator for real): either way the pager stops meaning anything.
        parts.insert(0, "SHADOW — not real")
        tag = "SHADOW-ACTION" if (has_trade or has_pending) else "SHADOW-DECISION"
        priority = "default"
    else:
        if has_trade:
            tag = "TRADE"
        elif has_pending:
            tag = "PENDING"
        elif exits_failed:
            tag = "FAILED-EXIT"
        else:
            tag = "DECISION"
        priority = "urgent" if exits_failed else (
            "high" if (has_trade or has_pending) else "default"
        )
    _headline = _action_headline(
        orders, exits, orders_pending, exits_pending, exits_failed
    )
    # Don't say it twice: on a failed-exit-only cycle the tag is already
    # FAILED-EXIT, so the headline's leading "FAILED-EXIT AAPL" would render
    # as "FAILED-EXIT: FAILED-EXIT AAPL".
    if _headline.startswith(f"{tag} "):
        _headline = _headline[len(tag) + 1:]
    title    = f"{label} [{run_mode}] {tag}" + (f": {_headline}" if _headline else "")
    body     = _truncate_ntfy_body(" | ".join(parts))
    url      = f"https://ntfy.sh/{topic}"
    actionable = bool(
        has_trade or has_pending or exits_failed or non_wl_holds or rot_blocked
    )
    if actionable:
        taxonomy = "TRADE" if has_trade else "ACTION_REQUIRED"
        key = None
        cooldown = 0
        force = True
    else:
        why = "skipped_order" if orders_skipped else _why_no_trade()
        taxonomy = "DECISION"
        key = stable_alert_key(
            "decision", label, run_mode, regime, why,
            bool(getattr(ctx, "bear_only", False)),
            bool(getattr(ctx, "skip_buys", False)),
            bool(getattr(ctx, "buy_blocked", False)),
        )
        cooldown = int(os.environ.get("RENQUANT_DECISION_NTFY_COOLDOWN_SECONDS", "1800"))
        force = False
    _post_ntfy_with_retries(
        url,
        title=title,
        body=body,
        priority=priority,
        taxonomy=taxonomy,
        key=key,
        cooldown_seconds=cooldown,
        force=force,
    )


def run_once_multi(
    config: dict[str, Any],
    models: dict,
    broker: BaseBroker,
    strategy_dir: Path,
    sell_only: bool = False,
    use_intraday_prices: bool = False,
    dry_run: bool = False,
) -> None:
    """Execute one multi-stock trading cycle via the kernel pipeline."""
    _run_once_multi_pipeline(
        config, models, broker, strategy_dir,
        sell_only, use_intraday_prices, dry_run=dry_run,
    )


def _is_multi_stock(strategy_name: str) -> bool:
    """Check if a strategy uses multi-stock watchlist config."""
    config_path = REPO_ROOT / "backtesting" / strategy_name / "strategy_config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())
    return "watchlist" in config


def _resolve_strategy_config_name(
    strategy_name: str,
    broker_type: str,
    requested_config_name: str | None,
) -> str:
    """Resolve the strategy config file for CLI runs.

    Read-only Alpaca is the renquant_104 shadow e2e path. Letting it default
    to production config makes logs say "[READONLY]" (broker-mode prefix,
    renamed from "[SHADOW]" 2026-07-01) while scoring with prod XGB.
    Keep explicit overrides possible for read-only prod rehearsals.
    """
    if requested_config_name:
        return requested_config_name
    if broker_type == "readonly-alpaca" and strategy_name == "renquant_104":
        return "strategy_config.shadow.json"
    return "strategy_config.json"


def main():
    parser = argparse.ArgumentParser(description="RenQuant live trading runner")
    parser.add_argument("--strategy", required=True, help="Strategy directory name")
    parser.add_argument("--broker", choices=["paper", "alpaca", "alpaca-paper", "alpaca-shorts", "ibkr", "readonly-alpaca"], default="paper")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--preflight", action="store_true",
                        help="Read-only dry-run probe: drive the full funnel to a "
                             "decision line but place NO orders, persist NO DB/state, "
                             "promote nothing, and send NO notification. Emits a "
                             "machine-readable `preflight_attestation:` line. Implies "
                             "--once. Used by the dawn preflight guard.")
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
    parser.add_argument("--strategy-config-name", default=None,
                        help="Side config filename inside the strategy dir "
                             "(default: strategy_config.shadow.json for "
                             "renquant_104 readonly-alpaca shadow, otherwise "
                             "strategy_config.json). Use this to test alternate "
                             "configs without touching the live one.")
    parser.add_argument("--strategy-config-path", default=None,
                        help="Explicit strategy config JSON path. Runtime "
                             "state, data, and artifact paths still resolve "
                             "under backtesting/<strategy>; only the config "
                             "document is loaded from this path.")
    args = parser.parse_args()
    if args.strategy_config_path and args.strategy_config_name:
        parser.error("--strategy-config-path cannot be combined with --strategy-config-name")
    config_name = _resolve_strategy_config_name(
        args.strategy,
        args.broker,
        args.strategy_config_name,
    )
    config_path = Path(args.strategy_config_path).expanduser() if args.strategy_config_path else None
    if config_path is not None:
        config_name = config_path.name
    if args.strategy_config_name is None and config_name != "strategy_config.json":
        log.info(
            "Auto-selected %s for %s %s run",
            config_name,
            args.strategy,
            args.broker,
        )

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
    broker.connect()
    config, models, strategy_dir = _load_strategy_multi(
        args.strategy, broker_name=broker.broker_name,
        config_name=config_name,
        config_path=config_path,
        broker=broker,
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
        dry_run=args.preflight,
    )

    try:
        if args.once or args.preflight:
            # --preflight is a single-shot read-only probe (implies --once).
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
