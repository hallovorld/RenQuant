"""SimAdapter — bridges a simulated portfolio to InferencePipeline.

Mirror of LeanAdapter / RunnerAdapter, but for the notebook / backtest
simulation path. Instead of talking to LEAN or a broker, SimAdapter owns
the sim's mutable state (cash, holdings, HWM, regime) and emulates
broker actions (execute sell, record buy) inside `commit()`.

Usage::

    adapter = SimAdapter(
        config=config, strategy_dir=STRATEGY_DIR,
        ohlcv=ohlcv, spy_df=spy_df,
        sector_etf_map=SECTOR_ETF, initial_cash=100_000,
        panel_feature_frames=ff, panel_factor_frames=fac,
    )
    for today in bt_dates:
        ctx = adapter.make_context(today)
        InferencePipeline().run(ctx)
        adapter.commit(ctx)
    result = adapter.build_result()

This path runs the *exact same* Jobs and Tasks as LEAN + the live runner,
so any decision added to `InferencePipeline` is automatically picked up
by the notebook simulation too. No more drift.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adapters.panel_runtime import (
    describe_panel_frame_bundle,
    prepare_panel_runtime_frames,
)
from kernel.execution import (
    FeeConfig,
    SlippageConfig,
    T2CashQueue,
    compute_buy_fees,
    compute_sell_fees,
    slip_fill_price,
)
from kernel.decision_trace import (
    build_ticker_daily_state_rows,
    candidate_trace_pool,
    model_type_from_artifact,
    model_types_from_models,
    qp_trace_maps,
    selected_buy_tickers,
)
from kernel.pipeline.exit_params import apply_stop_loss_anchor_policy

log = logging.getLogger("adapters.sim")


_ALPHA158_SCORER_KINDS = {"panel_linear", "panel_ltr_xgboost"}
_HISTORY_SCORER_KINDS = {"hf_patchtst", "patchtst", "regime_router"}
_FORBIDDEN_HISTORY_COL_PREFIXES = ("fwd_",)
_FORBIDDEN_HISTORY_COLS = {"label", "split_label"}
_BUYING_POWER_SETTLED = "settled_cash"
_BUYING_POWER_NMBP = "non_marginable_buying_power"
_BUYING_POWER_ALIASES = {
    _BUYING_POWER_SETTLED: _BUYING_POWER_SETTLED,
    "settled": _BUYING_POWER_SETTLED,
    "cash": _BUYING_POWER_SETTLED,
    _BUYING_POWER_NMBP: _BUYING_POWER_NMBP,
    "cash_plus_unsettled": _BUYING_POWER_NMBP,
    "unsettled": _BUYING_POWER_NMBP,
}


def _normalize_buying_power_mode(raw: Any) -> str:
    mode = str(raw or _BUYING_POWER_NMBP).strip().lower()
    if mode not in _BUYING_POWER_ALIASES:
        raise ValueError(
            "execution.buying_power_mode must be one of "
            f"{sorted(_BUYING_POWER_ALIASES)}; got {raw!r}"
        )
    return _BUYING_POWER_ALIASES[mode]


def _artifact_kind(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    meta = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(meta, dict) and meta.get("kind"):
        return str(meta.get("kind"))
    if isinstance(payload, dict) and payload.get("kind"):
        return str(payload.get("kind"))
    return None


def _history_seq_len_from_artifact(path: Path) -> int | None:
    """Best-effort sequence length probe without loading a Torch checkpoint."""
    candidates = [
        path.with_name(path.name + ".metadata.json"),
        path.with_name(path.stem + "_metadata.json"),
        path.with_name(path.stem + "_summary.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        contract = payload.get("training_contract") or {}
        hparams = contract.get("hyperparameters") or {}
        raw = payload.get("seq_len") or hparams.get("seq_len")
        if raw:
            return int(raw)
    return None


def _model_type_from_artifact(model: Any) -> str | None:
    """Extract readable model type from dict/object artifacts for audit rows."""
    return model_type_from_artifact(model)


def _drop_inference_forbidden_cols(df: pd.DataFrame) -> pd.DataFrame:
    forbidden = [
        c for c in df.columns
        if c in _FORBIDDEN_HISTORY_COLS
        or any(str(c).startswith(prefix) for prefix in _FORBIDDEN_HISTORY_COL_PREFIXES)
    ]
    return df.drop(columns=forbidden) if forbidden else df


def _resolve_manifest_uri(manifest_path: Path, uri: str) -> Path:
    p = Path(uri)
    return p if p.is_absolute() else manifest_path.parent / p


def _annual_net_equity_curve(
    equity_df: pd.DataFrame,
    sells: list[dict[str, Any]],
    annual_tax_summary: dict[str, Any],
) -> pd.DataFrame:
    """Return an annual-net tax reporting equity curve.

    The simulator can either debit estimated event-level tax from cash on each
    sell (stress mode) or keep that estimate reporting-only (live-like mode).
    Performance reporting needs the complementary Schedule-D-style annual
    netting estimate: add back only tax dollars that were actually debited
    from the simulated cash path, then subtract the year's estimated net
    capital-gains tax on the final sim date for that calendar year. This does
    not alter the historical decision path.
    """
    if equity_df.empty or "portfolio" not in equity_df.columns:
        return pd.DataFrame(columns=list(equity_df.columns))

    out = equity_df.copy()
    values = pd.to_numeric(out["portfolio"], errors="coerce").astype(float).to_numpy()
    n = len(out)
    event_tax = np.zeros(n, dtype=float)
    annual_tax = np.zeros(n, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()

    for sell in sells:
        # Legacy trade logs did not store tax_cash_debited, so fall back to
        # tax for backward-compatible event-level stress reports.
        tax = _finite_float(
            sell.get("tax_cash_debited"),
            default=_finite_float(sell.get("tax"), default=0.0),
        )
        if tax <= 0.0:
            continue
        raw_date = sell.get("date") or sell.get("exit_date")
        if raw_date is None:
            continue
        try:
            d = pd.Timestamp(raw_date).normalize()
        except Exception:
            continue
        pos = int(dates.searchsorted(d, side="left"))
        if 0 <= pos < n:
            event_tax[pos] += float(tax)

    years = annual_tax_summary.get("years") or []
    for row in years:
        tax = _finite_float(row.get("estimated_tax"), default=0.0)
        year = row.get("year")
        try:
            year_i = int(year)
        except (TypeError, ValueError):
            continue
        if tax <= 0.0:
            continue
        positions = np.flatnonzero(dates.year == year_i)
        if len(positions):
            annual_tax[int(positions[-1])] += float(tax)

    out["portfolio"] = values + np.cumsum(event_tax) - np.cumsum(annual_tax)
    return out


def _finite_float(value: Any, *, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _tax_cash_debit_mode(config: dict | None) -> str:
    """Return how estimated capital-gains tax should affect sim cash.

    ``event_level`` preserves the legacy stress-test path by debiting every
    profitable sell immediately. ``reporting_only`` records the estimate on
    the trade row but leaves broker-like cash unchanged; annual-net reporting
    then applies the tax overlay separately.
    """
    tax_cfg = ((config or {}).get("tax") or {}) if isinstance(config, dict) else {}
    raw = str(tax_cfg.get("cash_debit_mode", "event_level") or "event_level").lower()
    aliases = {
        "event": "event_level",
        "immediate": "event_level",
        "stress": "event_level",
        "none": "reporting_only",
        "off": "reporting_only",
        "reporting": "reporting_only",
        "reporting-only": "reporting_only",
        "reporting_only": "reporting_only",
        "annual_net": "reporting_only",
        "event_level": "event_level",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"event_level", "reporting_only"}:
        log.warning(
            "Unknown tax.cash_debit_mode=%s; falling back to event_level", raw,
        )
        return "event_level"
    return mode


def _tax_cash_debit_amount(config: dict | None, tax: float) -> float:
    tax_f = _finite_float(tax, default=0.0)
    if tax_f <= 0.0:
        return 0.0
    if _tax_cash_debit_mode(config) == "reporting_only":
        return 0.0
    return tax_f


class SimAdapter:
    """Translate between a simulated portfolio and InferenceContext."""

    def __init__(
        self,
        *,
        config: dict,
        strategy_dir: Path,
        ohlcv: dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
        sector_etf_map: dict[str, str],
        initial_cash: float,
        fallback_corr: dict | None = None,
        panel_feature_frames: dict[str, pd.DataFrame] | None = None,
        panel_factor_frames: dict[str, pd.DataFrame] | None = None,
        backtest_start: "pd.Timestamp | str | None" = None,
        backtest_end: "pd.Timestamp | str | None" = None,
    ) -> None:
        # Walk-forward feature flag (Track P2, 2026-05-10). Config schema:
        #   walkforward.enabled:       bool, default false
        #   walkforward.manifest_path: str,  default "artifacts/walkforward_manifest.json"
        #                              (relative paths resolve under strategy_dir)
        #   walkforward.fail_on_no_model: bool, default true
        # When enabled=False (default), legacy static-model path runs and
        # `assert_no_leakage` checks model.trained_date < backtest_end so
        # the audit-2026-05-10 class of look-ahead leakage cannot recur.
        from kernel.regime import RegimeState  # noqa: PLC0415
        from kernel.config import REGIMES      # noqa: PLC0415

        self._config         = dict(config)
        self._config["_strategy_dir"] = str(strategy_dir)
        self._strategy_dir   = Path(strategy_dir)
        self._ohlcv          = ohlcv
        self._spy_df         = spy_df
        self._sector_etf_map = sector_etf_map
        # Stored for the legacy-path leakage assertion (see end of init).
        self._backtest_start: "pd.Timestamp | None" = (
            pd.Timestamp(backtest_start) if backtest_start is not None
            else pd.Timestamp(config.get("backtest_start"))
            if config.get("backtest_start") is not None else None
        )
        self._backtest_end: "pd.Timestamp | None" = (
            pd.Timestamp(backtest_end) if backtest_end is not None else None
        )
        self._universe_rejections: dict[str, str] = {}

        # ── Load per-ticker policy artifacts (same path as live/runner) ─────
        self._models = self._load_models()

        # ── Load regime, correlation, earnings artifacts ────────────────────
        self._gmm, self._earnings, self._corr = self._load_artifacts(fallback_corr)

        # ── Walk-forward loader OR legacy static panel scorer ───────────────
        # (Track P2, 2026-05-10) — exactly one path is taken; the other
        # attribute is None. `_get_panel_scorer_for_bar(today)` routes.
        self._walkforward_loader = self._try_load_walkforward_loader()
        if self._walkforward_loader is not None:
            self._panel_scorer = None
        else:
            self._panel_scorer = self._try_load_panel_scorer()
            # Legacy-path leakage guard — fires when prod model trained on
            # data inside the sim window. Skipped silently when
            # backtest_end is unknown (caller didn't pass it) OR when the
            # scorer's metadata lacks trained_date (older artifacts).
            self._assert_legacy_no_leakage()
        self._ngboost_head  = self._try_load_ngboost_head()
        # Inference-side panel runtime cache. Keeps side-data parquet loads
        # (fundamentals, earnings surprise, sentiment) out of the per-bar hot
        # path. Values are raw point-in-time inputs; scorer-specific
        # normalization still happens inside PanelScoringJob each bar.
        self._panel_runtime_cache: dict = {}
        self._alpha158_feature_cache: dict[str, pd.DataFrame] = {}
        self._panel_history_cache: pd.DataFrame | None = None
        self._panel_history_seq_len = int(
            self._config.get("ranking", {})
                        .get("panel_scoring", {})
                        .get("seq_len", 64)
        )
        if self._history_cache_required():
            self._panel_history_cache = self._load_panel_history_cache()

        # ── Panel feature/factor frames (audit P-1, 2026-04-24) ─────────────
        # Architecture symmetry with LeanAdapter / RunnerAdapter: if the
        # caller didn't pre-build panel frames AND panel scoring is enabled,
        # build them here via the shared panel-runtime frame helper
        # function the other adapters use. Pre-fix the caller had to know
        # to construct them manually (notebook cell 15 used to fail this);
        # now SimAdapter is self-sufficient.
        frames_required = self._panel_frames_required()
        if panel_feature_frames is not None and panel_factor_frames is not None:
            self._panel_feature_frames = panel_feature_frames
            self._panel_factor_frames  = panel_factor_frames
        elif not frames_required:
            self._panel_feature_frames = None
            self._panel_factor_frames = None
            self._panel_macro_frame = None
            self._panel_asset_embeddings = None
            log.info(
                "SimAdapter: skipped legacy panel frame prep for alpha158 scorer; "
                "ApplyScoresTask will build features from OHLCV"
            )
        elif self._panel_scorer is not None or self._walkforward_loader is not None:
            try:
                bundle = prepare_panel_runtime_frames(
                    config=self._config,
                    ohlcv=ohlcv,
                    spy_df=spy_df,
                )
                self._panel_feature_frames = bundle.feature_frames
                self._panel_factor_frames = bundle.factor_frames
                self._panel_macro_frame = bundle.macro_frame   # Bug #25
                self._panel_asset_embeddings = bundle.asset_embeddings  # T2-2
                n_ff, n_fac, macro_desc, n_emb = describe_panel_frame_bundle(bundle)
                log.info("SimAdapter: built panel frames internally "
                         "(feat=%d  factor=%d  macro=%s  emb=%d)",
                         n_ff, n_fac, macro_desc, n_emb)
            except Exception as exc:
                msg = (
                    "SimAdapter: panel frame prep failed while panel scoring "
                    f"is active; refusing to run a silent fallback sim: {exc}"
                )
                log.error(msg)
                raise RuntimeError(msg) from exc
        else:
            self._panel_feature_frames = panel_feature_frames
            self._panel_factor_frames  = panel_factor_frames
            self._panel_macro_frame    = None

        # ── Persistent sim state (emulates broker / LEAN Portfolio) ─────────
        self._cash           = float(initial_cash)
        self._initial_cash   = float(initial_cash)
        self._hwm            = float(initial_cash)
        self._skip_buys      = False
        self._holdings: dict[str, Any] = {}        # ticker → HoldingState
        self._pos_shares: dict[str, float] = {}    # ticker → shares count
        self._last_sell_date: dict[str, pd.Timestamp] = {}   # ticker → date
        # 2026-05-09 audit Phase 2.1: per-ticker realized $ P/L of the most-
        # recent FULL liquidation, mirroring runner's compute_recent_realized_pnl().
        # Used by WashSaleFilterTask via is_wash_sale_blocked_with_cost — gain
        # sales skip §1091 block, loss sales compute NPV deferred-tax cost.
        # Pre-fix sim left ctx.last_sell_pls as {} → wash-sale fell back to
        # binary block on every recent-sell ticker → sim diverged from live
        # (live had cost-aware logic active). Sim now mirrors live behavior.
        self._last_sell_pls: dict[str, float] = {}
        # G8 (2026-05-04): per-ticker date when a path-rule exit (trailing_stop /
        # stop_loss / single_day_loss / max_hold / gap_down) last fired. Distinct
        # from `_last_sell_date` (which tracks ANY sell for wash-sale on losses).
        # Read by PostStopCooldownFilterTask to block re-entry within
        # `risk.post_stop_cooldown.bars` of a stop event.
        self._last_stop_exit_date: dict[str, pd.Timestamp] = {}
        self._regime_state   = RegimeState()
        self._regime_counts  = {r: 0 for r in REGIMES}
        # Monitor: persist MonitorIdleStreakTask's streak counters across bars.
        self._monitor_state: dict = {}
        # Rotation V1 persistence gate (2026-04-24): list of per-bar
        # proposed (sell, buy) pair sets, oldest first. Only populated
        # when `rotation.persistence_bars > 0`. Capped at that window.
        self._rotation_proposals: list = []

        # ── Execution model (Track Batch A, 2026-05-10) ─────────────────────
        # Industry-grade fill model: commission schedule + slippage + T+N
        # settlement. Three independent components, all single-source-of-
        # truth per CLAUDE.md §5.13.5. Each is config-driven; defaults
        # match Alpaca/IBKR retail equity (Q4 2025 schedules).
        #
        # Config schema (under top-level `execution`):
        #   execution.enabled:             bool,  default True
        #   execution.sec_fee_rate:        float, default 27.0e-6
        #   execution.taf_per_share:       float, default 1.19e-4
        #   execution.commission_bps:      float, default 0.0  (Alpaca = 0)
        #   execution.half_spread_bps:     float, default 2.0  (liquid S&P)
        #   execution.impact_bps_per_adv:  float, default 0.0  (off)
        #   execution.t2_settlement_days:  int,   default 1 (SEC T+1)
        #   execution.buying_power_mode:   str,   default
        #     "non_marginable_buying_power". This mirrors live Alpaca's
        #     cash policy: settled cash plus executed-but-unsettled sell
        #     proceeds, without 2x/4x margin. Set "settled_cash" for a
        #     conservative cash-account sim where sale proceeds cannot fund
        #     buys until settlement.
        #   execution.legacy_no_fees:      bool,  default False — parity
        #     flag for tests that need byte-identical pre-execution-model
        #     behavior (skips slippage AND fees AND uses T+0). When True,
        #     overrides `enabled`.
        exec_cfg = self._config.get("execution", {}) or {}
        self._exec_legacy = bool(exec_cfg.get("legacy_no_fees", False))
        self._exec_enabled = bool(exec_cfg.get("enabled", True)) and not self._exec_legacy
        self._fee_cfg = FeeConfig(
            sec_fee_rate=float(exec_cfg.get("sec_fee_rate", 27.0e-6)),
            taf_per_share=float(exec_cfg.get("taf_per_share", 1.19e-4)),
            custom_bps=float(exec_cfg.get("commission_bps", 0.0)),
        )
        self._slip_cfg = SlippageConfig(
            half_spread_bps=float(exec_cfg.get("half_spread_bps", 2.0)),
            impact_bps_per_pct_adv=float(exec_cfg.get("impact_bps_per_adv", 0.0)),
        )
        self._t2_queue = T2CashQueue(
            settlement_days=int(exec_cfg.get("t2_settlement_days", 1)),
        )
        self._buying_power_mode = _normalize_buying_power_mode(
            exec_cfg.get("buying_power_mode", _BUYING_POWER_NMBP)
        )
        self._buying_power_remaining: float | None = None
        # Cumulative fee tracking for diagnostic / build_result consumers.
        self._total_fees: float = 0.0

        # SPY returns buffer (last 100) + previous close for daily return calc
        self._spy_returns: list[float] = []
        self._spy_prev_close: float | None = None

        # ── Logs collected across the run ───────────────────────────────────
        self._equity_curve: list[dict]  = []
        self._trade_log:    list[dict]  = []
        self._rotation_log: list[dict]  = []
        self._tax_cash_debited: float = 0.0

        # ── Meta-label snapshot logger (P4.1, 2026-05-11) ──────────────────
        # Owned by adapter so it persists across bars. Attached to ctx in
        # make_context(); the MetaLabelLoggingJob's SnapshotHoldingsTask
        # writes one row per held ticker per bar. Dumped on build_result()
        # to data/position_day_snapshots.parquet (path from config).
        # Disabled (None) when `meta_label_training.enabled` is false.
        ml_cfg = self._config.get("meta_label_training", {}) or {}
        if ml_cfg.get("enabled", False):
            from kernel.meta_label import SnapshotLogger  # noqa: PLC0415
            self._meta_label_logger = SnapshotLogger()
            self._meta_label_output_path = str(
                ml_cfg.get("output_path", "data/position_day_snapshots.parquet")
            )
        else:
            self._meta_label_logger = None
            self._meta_label_output_path = None

        # ── Meta-label veto predictor (P4.4, 2026-05-11) ───────────────────
        # Inference-time counterpart to the snapshot logger above.
        # Loads the XGBoost classifier trained by scripts/_meta_label_train.py
        # and exposes a `predictor(feats: dict) -> P(profitable_exit)`
        # callable that MetaLabelVetoTask queries to drop false-positive
        # path-rule exits. Disabled (None) when
        # config.ranking.meta_label.enabled is false or the artifact
        # is absent (§5.13.10 fallback).
        veto_cfg = (self._config.get("ranking") or {}).get("meta_label") or {}
        if veto_cfg.get("enabled", False):
            from kernel.meta_label.predictor import load_meta_label_predictor  # noqa: PLC0415
            art_path = veto_cfg.get(
                "artifact_path",
                "backtesting/renquant_104/artifacts/meta-label-exit.json",
            )
            art_resolved = Path(art_path)
            if not art_resolved.is_absolute():
                art_resolved = Path(self._strategy_dir).parent.parent / art_resolved
            self._meta_label_predictor = load_meta_label_predictor(art_resolved)
        else:
            self._meta_label_predictor = None

        # ── Optional SQLite decision-trace ──────────────────────────────────
        # sim writes to a SEPARATE DB (persistence.sim_db_path, default
        # data/sim_runs.db) so notebook experimentation doesn't pollute
        # the live decision-audit statistics in data/runs.db. The sim DB
        # is TRUNCATEd at the start of every run_backtest (sim.runner)
        # so only the most-recent notebook sim's rows remain.
        from kernel.persistence import get_connection  # noqa: PLC0415
        self._db = get_connection(
            config, strategy_dir=self._strategy_dir, role="sim",
        )

        # Feature cache optimization (2026-04-24): pre-compute per-ticker
        # full-range feature frames ONCE here instead of rebuilding per
        # bar in TickerSellJob/CandidateJob. 5-8x sim speedup on the
        # 570-bar 27-mo window × 42-ticker panel.
        #
        # ✅ Equivalence VERIFIED 2026-04-24: kernel.indicators.
        # build_spy_context_series replaced the scalar-broadcast
        # build_spy_context, which had been the lookahead source. Now
        # cached.loc[:t].iloc[-1] == build_feature_frame(ohlcv[:t]).iloc[-1]
        # for every bar t. See tests/test_feature_cache.py::TestEquivalence.
        #
        # Flag-gated: `sim.feature_cache_enabled: true` (default true —
        # 5-8x sim speedup). Set false to disable for debugging.
        self._feature_cache: dict = {}
        if config.get("sim", {}).get("feature_cache_enabled", True):
            self._build_feature_cache()
        if config.get("sim", {}).get("alpha158_feature_cache_enabled", True):
            self._build_alpha158_feature_cache()

        log.info(
            "SimAdapter init: models=%d  gmm=%s  corr=%s  earnings=%s  "
            "panel_scorer=%s  walkforward=%s  ngboost_head=%s  "
            "feature_cache=%d tickers  alpha158_cache=%d tickers",
            len(self._models), self._gmm is not None, bool(self._corr),
            bool(self._earnings), self._panel_scorer is not None,
            self._walkforward_loader is not None,
            self._ngboost_head is not None, len(self._feature_cache),
            len(self._alpha158_feature_cache),
        )

    def _build_feature_cache(self) -> None:
        """One-shot: build full-range feature frame per watchlist ticker.

        Uses the same feature-assembly contract the per-bar task would
        call, but precomputes the shared SPY indicator/context frames once
        per sim run. Per-bar tasks then slice by `today` instead of
        re-running the indicator pipeline 570×42 times.
        """
        from kernel.indicators import (  # noqa: PLC0415
            assemble_feature_frame_from_indicators,
            build_spy_context_series,
            compute_all,
        )

        spy_df = self._ohlcv.get("SPY")
        if spy_df is None:
            log.warning("Feature cache: SPY OHLCV missing — skipping build")
            return
        if self._backtest_end is not None:
            spy_df = spy_df.loc[:self._backtest_end]
        if spy_df.empty:
            log.warning("Feature cache: SPY OHLCV empty after sim-end clipping — skipping build")
            return

        spec    = self._config.get("indicator_spec", {})
        vol_win = int(self._config.get("regime", {}).get("vol_realized_window", 20))
        spy_ind = compute_all(spy_df, spec)
        if spy_ind is None or spy_ind.empty:
            log.warning("Feature cache: SPY indicators empty — skipping build")
            return
        spy_context = build_spy_context_series(spy_df, vol_window=vol_win)

        built = 0
        total = max(len(self._ohlcv) - 1, 0)
        for ticker, df in self._ohlcv.items():
            if ticker == "SPY" or df is None or df.empty:
                continue
            if self._backtest_end is not None:
                df = df.loc[:self._backtest_end]
                if df.empty:
                    continue
            stock_ind = compute_all(df, spec)
            if stock_ind is None or stock_ind.empty:
                continue
            frame = assemble_feature_frame_from_indicators(
                stock_ind, spy_ind, spy_context,
            )
            if frame is not None and not frame.empty:
                self._feature_cache[ticker] = frame
                built += 1
            if built and built % 25 == 0:
                log.info("Feature cache progress: %d/%d tickers", built, total)
        log.info("Feature cache built: %d/%d tickers", built, total)

    def _alpha158_cache_required(self) -> bool:
        """Whether this sim can use cached alpha158 scorer features."""
        if self._panel_scorer is not None:
            kind = (getattr(self._panel_scorer, "metadata", {}) or {}).get("kind")
            return kind in _ALPHA158_SCORER_KINDS
        if self._walkforward_loader is not None:
            for entry in self._walkforward_loader.entries:
                p = _resolve_manifest_uri(
                    self._walkforward_loader.manifest_path,
                    entry.artifact_uri,
                )
                if _artifact_kind(p) in _ALPHA158_SCORER_KINDS:
                    return True
        return False

    def _build_alpha158_feature_cache(self) -> None:
        """One-shot causal alpha158 feature cache for historical sims."""
        if not self._alpha158_cache_required():
            return
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_frame  # noqa: PLC0415

        built = 0
        for ticker, df in self._ohlcv.items():
            if ticker == "SPY" or df is None or df.empty:
                continue
            if self._backtest_end is not None:
                df = df.loc[:self._backtest_end]
                if df.empty:
                    continue
            frame = compute_alpha158_frame(df)
            if frame is not None and not frame.empty:
                self._alpha158_feature_cache[ticker] = frame
                built += 1
        log.info("Alpha158 feature cache built: %d/%d tickers", built, len(self._ohlcv))

    # ── Artifact loaders ────────────────────────────────────────────────────

    def _load_models(self) -> dict[str, dict]:
        """Run LoadUniverseJob: artifacts + staleness + (conditional) sharpe-floor."""
        from kernel.pipeline.job_universe import UniverseContext, LoadUniverseJob  # noqa: PLC0415
        uctx = UniverseContext(config=self._config, strategy_dir=self._strategy_dir)
        LoadUniverseJob().run(uctx)
        self._universe_rejections = dict(uctx.rejections)
        for ticker, reason in uctx.rejections:
            log.debug("SimAdapter: %s rejected — %s", ticker, reason)
        return uctx.loaded_models

    def _load_artifacts(self, fallback_corr):
        from kernel.regime import load_gmm_artifact  # noqa: PLC0415
        from kernel.walk_forward import (  # noqa: PLC0415
            assert_correlation_no_leakage,
            assert_gmm_no_leakage,
            parse_correlation_artifact,
        )
        artifacts_dir = self._strategy_dir / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir = self._strategy_dir
        regime_cfg = self._config.get("regime", {})

        # 2026-05-11 sim/prod isolation: defaults relocated to prod/.
        # Sim configs override these keys to sim/<file>.
        earnings_path = artifacts_dir / regime_cfg.get(
            "earnings_artifact", "prod/earnings-calendar.json",
        )
        earnings_cal = {}
        if earnings_path.exists():
            try:
                earnings_cal = json.loads(earnings_path.read_text())
            except Exception as exc:
                log.warning("earnings calendar load failed: %s", exc)

        gmm = load_gmm_artifact(
            artifacts_dir / regime_cfg.get("gmm_artifact", "prod/spy-gmm-regime.json"),
        )
        assert_gmm_no_leakage(
            gmm,
            self._config.get("backtest_start"),
            is_live_mode=False,
            context="SimAdapter gmm",
        )

        corr_path = artifacts_dir / regime_cfg.get(
            "correlation_artifact", "prod/watchlist-correlation.json",
        )
        # AUDIT 2026-05-10 §5.13.5 — correlation as-of-date leakage guard.
        # Unwraps v2 schema (matrix + as_of_date) or treats raw as v1.
        # Routes through kernel.walk_forward.correlation_guard.
        if corr_path.exists():
            raw = json.loads(corr_path.read_text())
            corr_dict, as_of_date = parse_correlation_artifact(raw)
            assert_correlation_no_leakage(
                as_of_date,
                self._config.get("backtest_start"),
                is_live_mode=False,
                allow_legacy_without_as_of=bool(
                    regime_cfg.get("allow_legacy_correlation_without_as_of", False)
                ),
                context=f"SimAdapter corr={corr_path.name}",
            )
        elif fallback_corr is not None:
            corr_dict, as_of_date = parse_correlation_artifact(fallback_corr)
            assert_correlation_no_leakage(
                as_of_date,
                self._config.get("backtest_start"),
                is_live_mode=False,
                allow_legacy_without_as_of=bool(
                    regime_cfg.get("allow_legacy_correlation_without_as_of", False)
                ),
                context="SimAdapter fallback corr",
            )
        else:
            corr_dict = {}
        return gmm, earnings_cal, corr_dict

    def _try_load_panel_scorer(self):
        panel_cfg = self._config.get("ranking", {}).get("panel_scoring", {})
        if not panel_cfg.get("enabled", False):
            return None
        path = Path(panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json"))
        if not path.is_absolute():
            path = self._strategy_dir / path
        if not path.exists():
            log.warning("SimAdapter: panel artifact not found at %s", path)
            return None
        try:
            from kernel.panel_pipeline import PanelScorer  # noqa: PLC0415
            return PanelScorer.load(path)
        except Exception as exc:
            log.warning("SimAdapter: panel scorer load failed — %s", exc)
            return None

    def _try_load_walkforward_loader(self):
        """Load the WalkForwardModelLoader iff walkforward.enabled=True.

        Track P2 (2026-05-10): only enters this path when the config
        opts in. Default is OFF — legacy sim runs continue working
        unchanged. When manifest_path doesn't exist and
        `fail_on_no_model=True` (the default), raise FileNotFoundError;
        with `fail_on_no_model=False` we log + return None and the
        legacy static path is used instead.
        """
        wf_cfg = self._config.get("walkforward", {})
        if not wf_cfg.get("enabled", False):
            return None
        manifest = Path(wf_cfg.get("manifest_path",
                                   "artifacts/walkforward_manifest.json"))
        if not manifest.is_absolute():
            manifest = self._strategy_dir / manifest
        fail_on_missing = bool(wf_cfg.get("fail_on_no_model", True))
        if not manifest.exists():
            msg = f"SimAdapter: walkforward manifest not found at {manifest}"
            if fail_on_missing:
                raise FileNotFoundError(msg)
            log.warning("%s — falling back to legacy static load", msg)
            return None
        try:
            from kernel.walk_forward.loader import WalkForwardModelLoader  # noqa: PLC0415
            loader = WalkForwardModelLoader(manifest)
        except Exception as exc:
            if fail_on_missing:
                raise
            log.warning("SimAdapter: walkforward loader init failed — %s", exc)
            return None
        if not loader.has_walkforward_model():
            msg = (f"SimAdapter: walkforward manifest at {manifest} "
                   f"has zero retrain entries")
            if fail_on_missing:
                raise ValueError(msg)
            log.warning("%s — falling back", msg)
            return None
        log.info("SimAdapter: walkforward enabled (manifest=%s)", manifest)
        return loader

    def _panel_frames_required(self) -> bool:
        """Whether legacy panel feature/factor frames are required for scoring."""
        if self._panel_scorer is not None:
            if getattr(self._panel_scorer, "requires_history", False):
                return False
            kind = (getattr(self._panel_scorer, "metadata", {}) or {}).get("kind")
            return kind not in _ALPHA158_SCORER_KINDS
        if self._walkforward_loader is not None:
            for entry in self._walkforward_loader.entries:
                p = _resolve_manifest_uri(
                    self._walkforward_loader.manifest_path,
                    entry.artifact_uri,
                )
                if _artifact_kind(p) not in _ALPHA158_SCORER_KINDS:
                    return True
            return False
        return False

    def _history_cache_required(self) -> bool:
        """Whether sim should preload the full panel history once."""
        if self._panel_scorer is not None and getattr(
            self._panel_scorer, "requires_history", False,
        ):
            self._panel_history_seq_len = int(
                getattr(self._panel_scorer, "seq_len", self._panel_history_seq_len)
            )
            return True
        panel_cfg = self._config.get("ranking", {}).get("panel_scoring", {}) or {}
        if str(panel_cfg.get("kind", "")).lower() in _HISTORY_SCORER_KINDS:
            return True
        shadows = panel_cfg.get("shadow_models", []) or []
        for shadow in shadows:
            if not isinstance(shadow, dict):
                continue
            if str(shadow.get("kind", "")).lower() in _HISTORY_SCORER_KINDS:
                self._panel_history_seq_len = max(
                    self._panel_history_seq_len,
                    int(shadow.get("seq_len", self._panel_history_seq_len)),
                )
                return True
        if self._walkforward_loader is not None:
            required = False
            for entry in self._walkforward_loader.entries:
                p = _resolve_manifest_uri(
                    self._walkforward_loader.manifest_path,
                    entry.artifact_uri,
                )
                kind = (_artifact_kind(p) or "").lower()
                if kind in _HISTORY_SCORER_KINDS or p.suffix == ".pt":
                    required = True
                    seq_len = _history_seq_len_from_artifact(p)
                    if seq_len:
                        self._panel_history_seq_len = max(
                            self._panel_history_seq_len, int(seq_len)
                        )
            if required:
                return True
        return False

    def _load_panel_history_cache(self) -> pd.DataFrame | None:
        """Load PatchTST history once per sim and strip label/forward columns."""
        panel_cfg = self._config.get("ranking", {}).get("panel_scoring", {}) or {}
        raw_path = panel_cfg.get(
            "panel_history_path",
            self._config.get("panel_history_path", "data/alpha158_291_fundamental_dataset.parquet"),
        )
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._strategy_dir.parent.parent / path
        try:
            hist = pd.read_parquet(path)
            hist = _drop_inference_forbidden_cols(hist)
            hist["date"] = pd.to_datetime(hist["date"])
            watchlist = set(self._config.get("watchlist", []) or [])
            if watchlist and "ticker" in hist.columns:
                hist = hist[hist["ticker"].isin(watchlist)]
            hist = hist.sort_values(["date", "ticker"]).reset_index(drop=True)
            log.info(
                "SimAdapter: loaded panel history cache %s (%d rows, %d tickers)",
                path, len(hist), hist["ticker"].nunique() if "ticker" in hist else 0,
            )
            return hist
        except Exception as exc:  # noqa: BLE001
            log.warning("SimAdapter: panel history cache load failed — %s", exc)
            return None

    def _assert_legacy_no_leakage(self) -> None:
        """Defense-in-depth: legacy static model trained_date < backtest_end.

        The 2026-05-10 audit class: prod model trained 2026-05-09 used in
        a sim covering 2024-01 → 2026-03. Pre-fix sim happily loaded that
        model and produced inflated metrics. Now we hard-fail at adapter
        construction time, before a single bar runs.

        Skipped only when:
          - panel scorer isn't loaded (legacy adapter without panel scoring)
          - caller didn't pass a sim anchor (older API surface)

        See `kernel.walk_forward.leakage_guard` for the canonical helper.
        """
        sim_first_bar = self._backtest_start or self._backtest_end
        if self._panel_scorer is None or sim_first_bar is None:
            return
        meta = getattr(self._panel_scorer, "metadata", {}) or {}
        trained_date = meta.get("trained_date")
        if not trained_date:
            raise ValueError(
                "Panel scorer artifact is missing trained_date metadata; "
                "historical sim cannot prove the static scorer is "
                "point-in-time. Retrain/restamp the artifact or use a "
                "walk-forward manifest with cutoff_date + lookahead_days "
                "before every simulated bar."
            )
        from kernel.walk_forward.leakage_guard import assert_no_leakage  # noqa: PLC0415
        # 2026-05-11 Round 3 audit (G4): thread lookahead_days from the
        # scorer's metadata so the legacy guard catches forward-label
        # bleed too (e.g. fwd_60d_excess-trained model would silently
        # pass `trained_date < backtest_end` even when bar-by-bar leak).
        contract = meta.get("training_contract") or {}
        split_ranges = (
            meta.get("split_date_ranges")
            or contract.get("split_date_ranges")
            or {}
        )
        validation_end = (
            (split_ranges.get("val") or {}).get("end")
            if isinstance(split_ranges, dict) else None
        )
        leakage_anchor = (
            meta.get("effective_selection_cutoff_date")
            or contract.get("effective_selection_cutoff_date")
            or validation_end
            or meta.get("effective_train_cutoff_date")
            or meta.get("train_cutoff_date")
            or meta.get("cutoff_date")
            or trained_date
        )
        lookahead = int(meta.get("lookahead_days", 0) or 0)
        assert_no_leakage(
            leakage_anchor,
            sim_first_bar,
            context=f"SimAdapter legacy load "
                    f"(anchor={leakage_anchor}, "
                    f"artifact={meta.get('feature_cols', ['?'])[:1]}…)",
            lookahead_days=lookahead,
        )

    def _get_panel_scorer_for_bar(self, today: pd.Timestamp):
        """Return the panel scorer to use for `today`'s bar.

        Walk-forward path: dispatch to `WalkForwardModelLoader.model_as_of(today)`,
        which returns the most-recent retrain whose cutoff_date < today.
        Legacy path: return the (already-leakage-checked) static scorer.

        Returns None when neither path is configured (panel scoring off).
        """
        if self._walkforward_loader is not None:
            return self._walkforward_loader.model_as_of(pd.Timestamp(today))
        return self._panel_scorer

    def _get_global_calibrator_for_bar(self, today: pd.Timestamp):
        """Return the point-in-time calibrator paired with today's WF scorer."""
        if self._walkforward_loader is None:
            return None
        gc_cfg = (
            self._config.get("ranking", {})
                        .get("panel_scoring", {})
                        .get("global_calibration", {})
        )
        if not gc_cfg.get("enabled", False):
            return None
        return self._walkforward_loader.calibrator_as_of(pd.Timestamp(today))

    def _try_load_ngboost_head(self):
        ngb_cfg = (self._config.get("ranking", {})
                              .get("panel_scoring", {})
                              .get("ngboost", {}))
        if not ngb_cfg.get("enabled", False):
            return None
        path = Path(ngb_cfg.get("artifact_path", "artifacts/prod/ngboost-head.alpha158_fund.json"))
        if not path.is_absolute():
            path = self._strategy_dir / path
        if not path.exists():
            log.warning("SimAdapter: ngboost artifact not found at %s", path)
            return None
        try:
            # 2026-05-09: polymorphic loader dispatches on artifact kind
            # field — handles both NGBoostHead and QuantileHead. The
            # production runner already uses this; sim was missing it.
            from training_panel.quantile_head import load_head_by_kind  # noqa: PLC0415
            return load_head_by_kind(path)
        except Exception as exc:
            log.warning("SimAdapter: ngboost head load failed — %s", exc)
            return None

    def _pending_settle_cash(self) -> float:
        queue = getattr(self, "_t2_queue", None)
        if queue is None:
            return 0.0
        pending = queue.pending_total()
        return pending if math.isfinite(pending) else 0.0

    def _available_buying_power(self) -> float:
        """Cash budget exposed to the decision tree for new long buys.

        ``settled_cash`` is conservative cash-account behavior. The default
        ``non_marginable_buying_power`` mirrors the live Alpaca broker path:
        executed sell proceeds replenish non-margin buying power before they
        have fully settled, while still avoiding 2x/4x margin buying power.
        """
        cash = float(getattr(self, "_cash", 0.0) or 0.0)
        if not math.isfinite(cash):
            return 0.0
        if (
            getattr(self, "_exec_enabled", False)
            and getattr(self, "_buying_power_mode", _BUYING_POWER_SETTLED)
            == _BUYING_POWER_NMBP
        ):
            cash += self._pending_settle_cash()
        return cash if math.isfinite(cash) else 0.0

    # ── Public entry points ─────────────────────────────────────────────────

    def make_context(self, today: pd.Timestamp):
        """Build InferenceContext from current sim state + today's bar."""
        from kernel.pipeline.context import InferenceContext  # noqa: PLC0415

        today_ts = pd.Timestamp(today)
        today_date = today_ts.date() if hasattr(today_ts, "date") else today_ts

        # ── Execution model: drain T+N queue at top of every bar ────────────
        # Per spec §4: first thing each bar, settle any pending sell
        # proceeds whose settle_date <= today. This is the SINGLE callsite
        # for drain() in the bar loop — keeps cash availability aligned
        # between live brokers and sim.
        # Defensive: hasattr guard for __new__-constructed test fixtures.
        if (getattr(self, "_exec_enabled", False)
                and hasattr(self, "_t2_queue")
                and self._t2_queue.settlement_days > 0):
            settled = self._t2_queue.drain(today_ts)
            if math.isfinite(settled) and settled > 0:
                self._cash += settled

        # 2026-05-14 Phase 2B: daily borrow cost on short positions.
        # Charges (|short_value| × borrow_rate / 252) per bar. Reads rate
        # from data/alpaca_borrow_status.json (ETB → cheap rate, HTB →
        # expensive rate). No-op when no shorts open or feature disabled.
        # Per 2026-05-14 research: all 103 watchlist names are ETB, so
        # real impact for current universe is < 0.5%/yr.
        self._charge_daily_borrow(today_ts)

        # Update SPY returns buffer
        if today_ts in self._spy_df.index:
            spy_close = float(self._spy_df.loc[today_ts, "close"])
            if self._spy_prev_close is not None and self._spy_prev_close > 0:
                self._spy_returns.append(spy_close / self._spy_prev_close - 1.0)
                if len(self._spy_returns) > 100:
                    self._spy_returns = self._spy_returns[-100:]
            self._spy_prev_close = spy_close

        # Prices for this bar — union of models + sector ETFs
        prices: dict[str, float] = {}
        for t in self._models:
            df = self._ohlcv.get(t)
            if df is not None and today_ts in df.index:
                prices[t] = float(df.loc[today_ts, "close"])
        for _sec, etf in self._sector_etf_map.items():
            df = self._ohlcv.get(etf)
            if df is not None and today_ts in df.index:
                prices[etf] = float(df.loc[today_ts, "close"])
        # Held-position prices (in case a holding isn't in _models — defensives)
        for t in self._holdings:
            df = self._ohlcv.get(t)
            if df is not None and today_ts in df.index:
                prices[t] = float(df.loc[today_ts, "close"])

        pv = self._portfolio_value(prices, today_ts=today_ts)
        pending_settle_cash = self._pending_settle_cash()
        buying_power_cash = self._available_buying_power()

        # last_sell_dates as date objects (pipeline expects datetime.date)
        last_sells_d: dict[str, datetime.date | None] = {}
        for sym, d in self._last_sell_date.items():
            last_sells_d[sym] = d.date() if hasattr(d, "date") else d
        # G8: same shape for stop-exit dates
        last_stops_d: dict[str, datetime.date | None] = {}
        for sym, d in self._last_stop_exit_date.items():
            last_stops_d[sym] = d.date() if hasattr(d, "date") else d

        # Truncated OHLCV: each ticker's DataFrame sliced to [:today_ts] so
        # no future bars are visible to the pipeline (replicates LEAN
        # "History(bars up to now)" semantics).
        truncated = {
            t: df.loc[:today_ts] for t, df in self._ohlcv.items()
        }

        ctx = InferenceContext(
            config           = self._config,
            today            = today_date,
            ohlcv            = truncated,
            spy_returns      = list(self._spy_returns),
            models           = self._models,
            gmm              = self._gmm,
            corr_matrix      = self._corr,
            earnings_calendar = self._earnings,
            holdings         = {t: self._holdings[t]
                                for t in list(self._holdings.keys())},
            last_sell_dates  = last_sells_d,
            # 2026-05-09 audit fix: propagate realized $ P/L for cost-aware
            # wash-sale. Pre-fix sim diverged from live (live had this; sim
            # didn't → wash-sale fell back to binary block in sim).
            last_sell_pls    = dict(self._last_sell_pls),
            last_stop_exit_dates = last_stops_d,
            portfolio_value  = pv,
            cash             = buying_power_cash,
            prices           = prices,
            hwm              = self._hwm,
            skip_buys        = self._skip_buys,
            regime_state     = self._regime_state,
            regime_counts    = self._regime_counts,
            feature_cache    = self._feature_cache,
        )
        ctx.settled_cash = self._cash
        ctx.pending_settle_cash = pending_settle_cash
        ctx.buying_power_mode = getattr(
            self, "_buying_power_mode", _BUYING_POWER_SETTLED,
        )
        ctx.run_id = f"{today_date.isoformat()}-sim-{uuid.uuid4().hex[:8]}"

        # Hand prior streak counters to MonitorIdleStreakTask; it writes back.
        ctx.monitor_state = dict(self._monitor_state)

        # P4.1 (2026-05-11) — attach meta-label snapshot logger if enabled.
        # None when config.meta_label_training.enabled is false → the
        # MetaLabelLoggingJob's should_skip handles the prod / no-train path.
        ctx.snapshot_logger = self._meta_label_logger

        # P4.4 (2026-05-11) — attach meta-label veto predictor if loaded.
        # None when config.ranking.meta_label.enabled is false or the
        # artifact was missing — MetaLabelVetoTask's fallback handles it.
        ctx._meta_label_predictor = self._meta_label_predictor  # noqa: SLF001

        # Rotation V1 persistence gate: hand over the last N bars' proposed
        # (sell, buy) pair sets. BuildPairsTask reads via rotation_cfg
        # passthrough. Adapter pushes this bar's proposals in commit().
        ctx.prior_rotation_proposals = list(self._rotation_proposals)

        # Rotation V4 (thesis_symmetric) needs the sim DB to look up
        # candidate scores on each held's entry date.
        if self._db is not None:
            ctx._db = self._db   # noqa: SLF001

        # Preload panel scoring artifacts so PanelScoringJob short-circuits
        # its LoadScorerTask / LoadNGBoostTask.
        # Track P2 (2026-05-10): per-bar lookup. Walk-forward picks the
        # latest retrain with cutoff_date < today so no future labels
        # leak into this bar's scoring.
        scorer_for_bar = self._get_panel_scorer_for_bar(today_ts)
        if scorer_for_bar is not None:
            ctx._panel_scorer = scorer_for_bar  # noqa: SLF001
        calibrator_for_bar = self._get_global_calibrator_for_bar(today_ts)
        if calibrator_for_bar is not None:
            ctx._global_calibrator = calibrator_for_bar  # noqa: SLF001
        if self._ngboost_head is not None:
            ctx._ngboost_head = self._ngboost_head  # noqa: SLF001
        ctx._panel_runtime_cache = self._panel_runtime_cache  # noqa: SLF001
        ctx._alpha158_feature_cache = self._alpha158_feature_cache  # noqa: SLF001
        if self._panel_history_cache is not None:
            past = self._panel_history_cache[
                self._panel_history_cache["date"] < today_ts
            ]
            recent_dates = sorted(past["date"].unique())[-self._panel_history_seq_len:]
            ctx._panel_history = past[past["date"].isin(recent_dates)]  # noqa: SLF001
        if self._panel_feature_frames is not None:
            # Slice feature/factor frames to today_ts too (no future leak)
            ctx._panel_feature_frames = {                              # noqa: SLF001
                t: df.loc[:today_ts] for t, df in self._panel_feature_frames.items()
            }
            if self._panel_factor_frames is not None:
                ctx._panel_factor_frames = {                            # noqa: SLF001
                    t: df.loc[:today_ts] for t, df in self._panel_factor_frames.items()
                }
        # Bug #25: also propagate macro frame, sliced to today_ts
        if getattr(self, "_panel_macro_frame", None) is not None:
            ctx._panel_macro_frame = self._panel_macro_frame.loc[:today_ts]  # noqa: SLF001

        return ctx

    def commit(self, ctx) -> None:  # noqa: ANN001
        """Apply pipeline outputs to sim state. Mirrors LeanAdapter.commit."""
        # ── Exits ───────────────────────────────────────────────────────────
        today_ts = pd.Timestamp(ctx.today)
        trade_events_this_bar: list[dict] = []
        len_trade_log_before = len(self._trade_log)
        # Phase 2B fix (2026-05-14): route qp_short_open exits OUTSIDE the
        # dedupe loop. When the QP closes a long AND opens a short on the
        # same ticker, _emit_qp_sell appends TWO ExitSignals (qp_close +
        # qp_short_open). The old per-ticker dedupe kept first-write-wins,
        # silently dropping the short-open. Now: collect short-opens
        # separately, dispatch them after the close-long step so the
        # ticker first goes long→0, then 0→short cleanly.
        short_opens: list = []
        regular_exits: list = []
        for ticker, sig in (ctx.exits or []):
            if str(getattr(sig, "exit_type", "")) == "qp_short_open":
                short_opens.append((ticker, sig))
            else:
                regular_exits.append((ticker, sig))

        # Dedupe ctx.exits per ticker: when TopUp/Trim's "already exiting"
        # guard misfired (pre-2026-04-24 tuple-attr bug) two exits could
        # be queued for the same ticker. Even after the guard fix, an
        # adversarial config could emit a stop_loss + a kelly_trim
        # simultaneously. Priority: full liquidation over partial trim,
        # earliest exit signal otherwise.
        exits_by_ticker: dict[str, tuple] = {}
        for ticker, sig in regular_exits:
            existing = exits_by_ticker.get(ticker)
            if existing is None:
                exits_by_ticker[ticker] = (ticker, sig)
                continue
            # Prefer the one that's a full exit
            ex_q = getattr(existing[1], "quantity", None)
            new_q = getattr(sig, "quantity", None)
            ex_full = ex_q is None or ex_q <= 0
            new_full = new_q is None or new_q <= 0
            if new_full and not ex_full:
                exits_by_ticker[ticker] = (ticker, sig)
            # Otherwise keep existing (first-write-wins)
        deduped_exits = list(exits_by_ticker.values())

        # Track which tickers need FULL liquidation vs partial trim. Only
        # full exits pop from holdings/pos_shares; partial trims update
        # share count in-place (see _apply_sell).
        # Bug 11 fix (2026-05-05): mirror _apply_sell's bug-8 partial-vs-
        # full logic exactly. Pre-fix, commit() used `q is None or q <= 0
        # or q >= cur` while _apply_sell switched to a strict isfinite
        # check. NaN sig.quantity → _apply_sell treated as FULL but
        # commit treated as PARTIAL → ticker fully liquidated in cash
        # but stayed in _holdings with shares=0 (ghost position). All
        # subsequent rebalances saw a phantom holding with no shares.
        import math as _math_q  # noqa: PLC0415
        full_exit_tickers: set[str] = set()
        for ticker, sig in deduped_exits:
            # 2026-05-14 Phase 2B: route qp_short_open to dedicated path
            # (creates a new HoldingState with negative shares; credits
            # short-sale proceeds to cash). Skips the close-existing-long
            # logic that the regular sell path uses.
            if str(getattr(sig, "exit_type", "")) == "qp_short_open":
                self._apply_short_open(ticker, sig, today_ts, ctx)
                continue
            q = getattr(sig, "quantity", None)
            cur = self._pos_shares.get(ticker, 0)
            is_finite_partial = (
                q is not None
                and isinstance(q, (int, float))
                and _math_q.isfinite(float(q))
                and q > 0
                and q < cur
            )
            if not is_finite_partial:
                full_exit_tickers.add(ticker)
            self._apply_sell(ticker, sig, today_ts, ctx)

        for ticker in full_exit_tickers:
            self._holdings.pop(ticker, None)
            self._pos_shares.pop(ticker, None)

        # Phase 2B: dispatch short-opens AFTER all long-closes settled.
        # This way a ticker that flipped from long to short cleanly
        # goes long→0→short.
        for ticker, sig in short_opens:
            self._apply_short_open(ticker, sig, today_ts, ctx)

        # Preserve updated sell_streak / HWM from pipeline's SellJob.
        # Exclude only FULL exits — partial trims keep the position open
        # with original entry_date / entry_price preserved.
        for ticker, hs in ctx.holdings.items():
            if ticker not in full_exit_tickers:
                self._holdings[ticker] = hs

        # ── Buys ────────────────────────────────────────────────────────────
        # Audit fix BUY-DEDUPE (Round 2 deep audit, 2026-04-25): exits
        # already de-duplicate by ticker (lines 400-414 above) — same
        # contract should hold for buys. Pre-fix, two independent jobs
        # nominating the same ticker (e.g. SizeAndEmit + a hypothetical
        # future TopUp buy emitting a stale order on the same bar)
        # would each call _apply_buy → cash debited twice, shares doubled,
        # phantom holding state. Belt-and-braces: even though current
        # pipeline order makes this hard to trigger, dedupe matches the
        # exit-side pattern (first-write-wins) and is cheap.
        seen_buy_tickers: set[str] = set()
        self._buying_power_remaining = self._available_buying_power()
        try:
            for order in ctx.orders:
                t = order.get("ticker") if isinstance(order, dict) else getattr(order, "ticker", None)
                if t is not None and t in seen_buy_tickers:
                    # Same ticker already booked this bar — skip the dup.
                    continue
                if t is not None:
                    seen_buy_tickers.add(t)
                self._apply_buy(order, today_ts, ctx)
        finally:
            self._buying_power_remaining = None

        # Collect trade events emitted this bar for the persistence trace
        trade_events_this_bar = self._trade_log[len_trade_log_before:]

        # ── Persist cross-bar state from pipeline ───────────────────────────
        self._regime_state  = ctx.regime_state
        self._regime_counts = ctx.regime_counts
        self._hwm           = ctx.hwm
        self._skip_buys     = ctx.skip_buys
        self._monitor_state = dict(getattr(ctx, "monitor_state", {}) or {})

        # Rotation V1 persistence gate (2026-04-24): push this bar's
        # proposed (sell, buy) pair set. Cap the history at 2× the
        # persistence_bars setting (or ≥ 10 if disabled) so memory stays
        # bounded. Only rotations that were actually proposed (pre-gate)
        # matter — post-gate filtering happens in task_rotation, so we
        # stamp ctx.rotations which reflects final pairs.
        pairs_this_bar: set[tuple[str, str]] = {
            (p.sell_ticker, p.buy_ticker) for p in getattr(ctx, "rotations", [])
        }
        persistence_n = int(self._config.get("rotation", {}).get("persistence_bars", 0))
        window = max(persistence_n * 2, 10)
        self._rotation_proposals.append(pairs_this_bar)
        if len(self._rotation_proposals) > window:
            self._rotation_proposals = self._rotation_proposals[-window:]

        # ── Equity curve entry ──────────────────────────────────────────────
        pv = self._portfolio_value(ctx.prices, today_ts=today_ts)
        self._equity_curve.append({
            "date": today_ts, "portfolio": pv, "regime": ctx.regime,
        })

        # ── SQLite decision trace ───────────────────────────────────────────
        if self._db is not None:
            from kernel.persistence import (  # noqa: PLC0415
                record_pipeline_run, record_candidate_scores, record_trades,
                record_ticker_daily_state,
            )
            from kernel.artifact_contract import build_run_bundle  # noqa: PLC0415
            run_bundle = build_run_bundle(
                self._config,
                self._strategy_dir,
                run_id=str(getattr(ctx, "run_id", "")),
                run_type="sim",
                ctx=ctx,
            )
            selected_tickers = selected_buy_tickers(trade_events_this_bar)
            run_id = record_pipeline_run(
                self._db,
                run_type        = "sim",
                run_date        = today_ts.date(),
                strategy        = str(self._config.get("model_name", "")),
                regime          = ctx.regime,
                confidence      = float(ctx.confidence) if ctx.confidence is not None else None,
                portfolio_value = pv,
                cash            = self._cash,
                n_candidates    = len(ctx.candidates),
                n_exits         = len(ctx.exits),
                n_rotations     = int(ctx.counters.get("rotations", 0)),  # ROT-COUNTER fix: emitted, not considered
                n_buys          = len(selected_tickers),
                buy_blocked     = bool(getattr(ctx, "buy_blocked", False)),
                skip_buys        = bool(getattr(ctx, "skip_buys", False)),
                bear_only        = bool(getattr(ctx, "bear_only", False)),
                counters         = getattr(ctx, "counters", {}) or {},
                run_bundle       = run_bundle,
                run_id          = getattr(ctx, "run_id", None),
            )
            blocked_map = getattr(ctx, "_blocked_by_ticker", None)
            sector_map = self._config.get("sector_map", {}) or {}
            model_types = model_types_from_models(self._models)
            panel_artifact = (
                self._config.get("ranking", {})
                            .get("panel_scoring", {})
                            .get("artifact_path")
            )
            qp_delta_by_ticker, qp_target_by_ticker, qp_status = qp_trace_maps(ctx)
            # 2026-05-04 user mandate ("rank_score need to be collected
            # properly for future fine tune"): persist the FULL pre-
            # veto candidate list so the candidate_scores table captures
            # the complete rank_score / mu / sigma distribution per
            # bar, not just the survivors. Vetoed rows are tagged via
            # blocked_map (veto:rank_score_below_floor / kelly_zero:*
            # / ngb_skipped:*), so SQL queries on blocked_by reveal
            # exactly where each ticker was filtered out.
            cand_pool = candidate_trace_pool(ctx)
            record_candidate_scores(
                self._db, run_id, cand_pool, ctx.holdings,
                selected_tickers=selected_tickers,
                blocked_map=blocked_map,
                sector_map=sector_map,
                model_types=model_types,
                panel_artifact=panel_artifact,
                qp_delta_by_ticker=qp_delta_by_ticker,
                qp_target_by_ticker=qp_target_by_ticker,
                qp_status=qp_status,
            )
            record_trades(self._db, run_id, trade_events_this_bar)
            tds_rows = build_ticker_daily_state_rows(
                config=self._config,
                ctx=ctx,
                selected_tickers=selected_tickers,
                blocked_map=blocked_map,
                model_types=model_types,
                universe_rejections=self._universe_rejections,
                model_keys=set(self._models or {}),
                portfolio_value=pv,
                sector_map=sector_map,
                qp_delta_by_ticker=qp_delta_by_ticker,
                qp_target_by_ticker=qp_target_by_ticker,
                qp_status=qp_status,
            )
            record_ticker_daily_state(
                self._db,
                run_date=today_ts.date(),
                rows=tds_rows,
                run_id=run_id,
            )

    # ── Sim-side execution primitives ───────────────────────────────────────

    def _apply_sell(self, ticker: str, sig, today_ts: pd.Timestamp, ctx) -> None:
        """Apply a sell — full liquidation (default) or partial when sig.quantity set.

        When sig.quantity is None or ≥ current shares, sells everything (caller's
        commit() then pops the ticker from holdings/pos_shares). When sig.quantity
        is a positive float < current shares, sells exactly that many shares and
        reduces _pos_shares in place; the caller then skips the pop step.
        """
        from kernel.portfolio import compute_trade_tax  # noqa: PLC0415
        if ticker not in self._holdings or ticker not in self._pos_shares:
            return
        hs = self._holdings[ticker]
        total_shares = self._pos_shares[ticker]

        # Bug 8 fix (2026-05-05): NaN/inf sig.quantity slipped through
        # both `<= 0` and `>= total_shares` (NaN comparisons return
        # False), then `sell_shares = float(NaN) = NaN` propagated
        # through apply_sell_lots → gross_pnl=NaN → cash=NaN. Once
        # _cash goes NaN, every subsequent _portfolio_value call returns
        # NaN and the equity curve is poisoned. Pre-fix the SAB-3 audit
        # caught this on the BUY side; sell side had the same hole.
        # Treat non-finite or non-positive quantity as full liquidation
        # (the caller's intent of "exit this position" without a partial
        # spec).
        import math as _math_q  # noqa: PLC0415
        req_qty = getattr(sig, "quantity", None)
        is_finite_partial = (
            req_qty is not None
            and isinstance(req_qty, (int, float))
            and _math_q.isfinite(float(req_qty))
            and req_qty > 0
            and req_qty < total_shares
        )
        if is_finite_partial:
            sell_shares = float(req_qty)
            is_partial  = True
        else:
            sell_shares = total_shares
            is_partial  = False

        price = ctx.prices.get(ticker)
        if price is None:
            df = self._ohlcv.get(ticker)
            if df is None or today_ts not in df.index:
                return
            price = float(df.loc[today_ts, "close"])

        # ── Execution model: apply slippage to fill price (Track Batch A) ───
        # Slippage adjustment happens BEFORE every downstream calc (gross_pnl,
        # tax, cash, trade log) so a single source-of-truth fill price
        # propagates everywhere. The market_price field is preserved in the
        # closure for reference but no longer used past this point.
        # Defensive: existing tests construct SimAdapter via __new__ (skipping
        # __init__) — preserve byte-identical legacy behavior when the
        # execution-model fields aren't initialized.
        if getattr(self, "_exec_enabled", False):
            slipped = slip_fill_price(
                market_price=price, side="sell", shares=sell_shares,
                adv_shares=None,
                cfg=self._slip_cfg,
            )
            if _math_q.isfinite(slipped) and slipped > 0:
                price = slipped

        hold_days = (today_ts.date() - hs.entry_date).days if hs.entry_date else 0

        # Bug 6 fix (2026-05-05 wl183 incident): under FIFO/HIFO lot
        # disposal, the realized cost basis of the disposed shares is NOT
        # the weighted-average entry_price. Pre-fix, gross_pnl was
        # sell_shares × (price − avg_entry), which under FIFO with rising
        # prices systematically UNDER-estimates true gain (oldest lots are
        # cheapest) → tax under-collected → cash inflated → APY/Sharpe
        # over-reported. Post-fix: dispose lots first, get the actual
        # cost basis from `apply_sell_lots`, compute gross_pnl off that.
        # Falls back to avg-cost when the holding has no lots (legacy
        # state with shares but lots not migrated yet — rare).
        from kernel.exits import apply_sell_lots, ensure_lots  # noqa: PLC0415
        ensure_lots(self._holdings[ticker])
        ja_cfg = (ctx.config.get("rotation", {}).get("joint_actions", {}) or {})
        lot_method = str(ja_cfg.get("qp_tax_lot_method", "fifo")).lower()
        had_lots = bool(self._holdings[ticker].lots)
        proceeds_basis, _ = apply_sell_lots(
            self._holdings[ticker], float(sell_shares), lot_method,
        )
        # 2026-05-10 audit fix (§5.13.11): when proceeds_basis is non-
        # positive (lots exhausted corner case — e.g. ensure_lots produced
        # zero-cost lots from corrupted state), the next line would emit
        # gross_pnl = sell_shares*price - 0 ≈ revenue, mis-classifying
        # the disposal as 100% gain and over-taxing. Fall back to the
        # weighted-avg entry_price so realized P&L tracks actual cost
        # basis even when the lot ledger is degenerate.
        if had_lots and _math_q.isfinite(proceeds_basis) and proceeds_basis > 0:
            gross_pnl = sell_shares * price - proceeds_basis
        else:
            # Legacy path or degenerate proceeds_basis: fall back to
            # avg-entry computation. entry_price already guarded finite
            # upstream in _apply_buy (SAB-2).
            _fallback_entry = float(getattr(hs, "entry_price", 0.0) or 0.0)
            if not _math_q.isfinite(_fallback_entry) or _fallback_entry <= 0:
                _fallback_entry = price  # last resort: treat as flat P&L
            gross_pnl = sell_shares * (price - _fallback_entry)
            # Stamp proceeds_basis so downstream pnl_pct computation
            # (had_lots branch below) sees a sane disposed basis.
            if not (_math_q.isfinite(proceeds_basis) and proceeds_basis > 0):
                proceeds_basis = sell_shares * _fallback_entry

        tax_cfg   = ctx.config.get("tax", {})
        tax = compute_trade_tax(
            gross_pnl, hold_days,
            float(tax_cfg.get("short_term_rate", 0.50)),
            float(tax_cfg.get("long_term_rate", 0.32)),
            int(tax_cfg.get("long_term_threshold_days", 365)),
        )
        tax_cash_debit_mode = _tax_cash_debit_mode(ctx.config if ctx is not None else {})
        tax_cash_debited = _tax_cash_debit_amount(
            ctx.config if ctx is not None else {}, tax,
        )
        # ── Execution model: sell fees + T+N settlement (Track Batch A) ────
        # Per §5.13.5 every fee dollar flows through compute_sell_fees;
        # per the spec, proceeds (notional - fees) are queued for T+N
        # settlement via the T2CashQueue, NOT credited immediately.
        # Estimated capital-gains tax is config-driven: event_level debits it
        # immediately as a liquidity stress test, reporting_only leaves broker
        # cash unchanged and records the estimate for annual-net reporting.
        # Defensive: __new__-constructed test fixtures bypass __init__,
        # so getattr() with explicit defaults keeps legacy semantics.
        _exec_on = getattr(self, "_exec_enabled", False)
        if _exec_on:
            sell_fees = compute_sell_fees(sell_shares, price, self._fee_cfg)
        else:
            sell_fees = {"sec_fee": 0.0, "taf": 0.0, "custom": 0.0, "total": 0.0}
        notional = sell_shares * price
        net_proceeds = notional - sell_fees["total"]
        if hasattr(self, "_total_fees"):
            self._total_fees += sell_fees["total"]

        # 2026-05-10 audit fix (§5.13.11): every arithmetic input to the
        # cash mutation must be finite, otherwise self._cash silently goes
        # NaN and every subsequent _portfolio_value emits NaN. Pre-fix the
        # guards covered the BUY path (SAB-3) and sig.quantity (Bug 8) but
        # NOT proceeds_basis / gross_pnl / tax post-computation. Raise so
        # the caller sees the diagnostic context instead of a silently
        # poisoned equity curve.
        if not (_math_q.isfinite(net_proceeds)
                and _math_q.isfinite(tax)
                and _math_q.isfinite(tax_cash_debited)
                and _math_q.isfinite(self._cash)):
            raise ValueError(
                f"SimAdapter._apply_sell cash NaN guard tripped: "
                f"ticker={ticker} today={today_ts} sell_shares={sell_shares} "
                f"price={price} proceeds_basis={proceeds_basis} "
                f"entry_price={getattr(hs, 'entry_price', None)} "
                f"gross_pnl={gross_pnl} tax={tax} "
                f"tax_cash_debited={tax_cash_debited} "
                f"net_proceeds={net_proceeds} "
                f"cash_before={self._cash}"
            )
        self._cash -= tax_cash_debited
        if not hasattr(self, "_tax_cash_debited"):
            self._tax_cash_debited = 0.0
        self._tax_cash_debited += float(tax_cash_debited)
        # T+0 legacy path: proceeds credited immediately. T+N path: queue
        # net proceeds for settlement; drain happens at top of next bar.
        # Defensive: hasattr guard so __new__ test fixtures stay byte-
        # identical to pre-execution-model legacy semantics.
        _t2_on = (
            _exec_on
            and hasattr(self, "_t2_queue")
            and self._t2_queue.settlement_days > 0
        )
        if _t2_on:
            self._t2_queue.add_pending(today_ts, net_proceeds)
        else:
            self._cash += net_proceeds
        if not _math_q.isfinite(self._cash):
            raise ValueError(
                f"SimAdapter._apply_sell cash became non-finite after mutation: "
                f"ticker={ticker} today={today_ts} sell_shares={sell_shares} "
                f"price={price} net_proceeds={net_proceeds} cash={self._cash}"
            )
        # Wash-sale clock: stamp ONLY on full liquidation. Partial trims
        # (Kelly rebalance) intentionally don't block subsequent top-ups —
        # otherwise the position can never grow back toward Kelly target.
        # Aligned with LeanAdapter + RunnerAdapter (2026-04-24 fix).
        if not is_partial:
            self._last_sell_date[ticker] = today_ts
            # 2026-05-09 audit fix: stamp realized $ P/L for cost-aware
            # wash-sale (mirrors runner's compute_recent_realized_pnl).
            # Use gross_pnl (pre-tax, since §1091 looks at the loss event,
            # not the after-tax net). Live runner uses pre-tax too.
            # Defensive: existing tests construct SimAdapter via __new__
            # which bypasses __init__ — ensure the attribute (mirrors
            # _last_stop_exit_date pattern below).
            if not hasattr(self, "_last_sell_pls"):
                self._last_sell_pls = {}
            try:
                self._last_sell_pls[ticker] = float(gross_pnl)
            except (TypeError, ValueError):
                pass    # leave None / unknown → binary fallback
        # G8 (2026-05-04): stamp post-stop blackout date on path-rule
        # exits regardless of partial/full. The blackout blocks
        # *re-entry* — even a partial Kelly trim that hits a
        # `single_day_loss` exit_type means timing is bad.
        from kernel.pipeline.task_post_stop_cooldown import (  # noqa: PLC0415
            DEFAULT_STOP_EXIT_TYPES,
        )
        if str(sig.exit_type) in DEFAULT_STOP_EXIT_TYPES:
            # Defensive: existing tests construct SimAdapter via a custom
            # __init__ that may skip our dict init. Ensure the attribute.
            if not hasattr(self, "_last_stop_exit_date"):
                self._last_stop_exit_date = {}
            self._last_stop_exit_date[ticker] = today_ts
        # 2026-05-04 audit P1-5: apply_sell_lots mutates `hs.lots` but
        # NOT `hs.entry_price`. Under HIFO disposal of a partial trim,
        # the highest-cost lot is gone first → the surviving weighted
        # avg drops. If we don't refresh entry_price here, downstream
        # trailing_stop / stop_loss / take_profit checks compare
        # current_price against a STALE pre-sell weighted avg → wrong
        # P&L sign on every check. The legacy comment said "Kelly trims
        # should not reset cost basis" — that's TRUE but the issue is
        # we're recomputing the basis FROM THE SURVIVING LOTS, which IS
        # the consistent post-sell basis (not a "reset"). Equally
        # important on full sells (lots empty → entry_price → 0).
        hs_after = self._holdings[ticker]
        hs_after.entry_price = hs_after.weighted_avg_entry_price()

        if is_partial:
            # Keep the position open with reduced share count.
            # entry_date stays — tenure tracks original acquisition.
            self._pos_shares[ticker] = total_shares - sell_shares
            self._holdings[ticker].shares = total_shares - sell_shares

        # 2026-05-04 audit Issue 27: NaN entry_price was truthy in Python
        # (`bool(float('nan'))` is True), so the `if hs.entry_price else 0.0`
        # branch did NOT short-circuit and `(price - NaN) / NaN = NaN`
        # propagated into trade_log, then into win_rate / avg_pnl / B2
        # holdout reports as NaN. Compute pnl_pct defensively: require
        # both finite price AND finite positive entry_price.
        #
        # Bug 7 fix (2026-05-05 wl183 incident, follow-on to bug 6): on
        # PARTIAL trims, hs.entry_price was already refreshed to the
        # SURVIVING-lot weighted avg (line 645), which is NOT the cost
        # basis of the DISPOSED shares. Using surviving-avg here reports
        # "unrealized P&L on the remaining position", not "realized P&L
        # of this trade". Result: partial-trim win/loss classification
        # was wrong — a Kelly trim at +30% on cheaper FIFO lots could
        # show pnl_pct ≈ 0% (vs the surviving expensive lots) instead
        # of the actual realized +30%. Corrupts win_rate + avg_pnl.
        # Fix: use the disposed cost basis (proceeds_basis / sell_shares)
        # which represents the *actual basis of what was sold*. Falls
        # back to surviving entry_price for legacy state without lots.
        import math as _math_pnl   # local import — module-level math not present
        if had_lots and proceeds_basis > 0 and sell_shares > 0:
            _disposed_basis = proceeds_basis / sell_shares
        else:
            _disposed_basis = float(getattr(hs, "entry_price", 0.0) or 0.0)
        if (_math_pnl.isfinite(price) and _math_pnl.isfinite(_disposed_basis)
                and _disposed_basis > 0):
            _pnl_pct = (price - _disposed_basis) / _disposed_basis
        else:
            _pnl_pct = 0.0
        regime_p = (ctx.config.get("regime_params", {}) or {}).get(
            getattr(ctx, "regime", None), {},
        ) if ctx is not None else {}
        applied_exit_p = getattr(sig, "exit_params", None)
        if isinstance(applied_exit_p, dict) and applied_exit_p:
            exit_p = dict(applied_exit_p)
        else:
            exit_p = dict(regime_p or {})
            entry_regime = getattr(hs, "entry_regime", None)
            entry_regime_p = (
                (ctx.config.get("regime_params", {}) or {}).get(entry_regime, {})
                if ctx is not None and entry_regime is not None else {}
            )
            if isinstance(entry_regime_p, dict) and "max_hold_days" in entry_regime_p:
                exit_p["max_hold_days"] = entry_regime_p["max_hold_days"]
                exit_p["max_hold_anchor_regime"] = entry_regime
            apply_stop_loss_anchor_policy(
                exit_p,
                config=(ctx.config if ctx is not None else {}),
                current_regime=getattr(ctx, "regime", None) if ctx is not None else None,
                entry_regime=entry_regime,
                entry_regime_params=entry_regime_p,
            )
        sig_reason = getattr(sig, "reason", None)
        source_job = str(getattr(sig, "source_job", None) or "TickerSellJob")
        source_task = str(getattr(sig, "source_task", None) or sig.exit_type or "sell")
        order_source = str(
            getattr(sig, "order_source", None) or f"{source_job}.{source_task}"
        )
        self._trade_log.append({
            "action":      "sell",
            "ticker":      ticker,
            "date":        today_ts,
            "price":       price,
            "shares":      sell_shares,
            "gross_pnl":   gross_pnl,
            "proceeds_basis": proceeds_basis,
            "net_pnl_after_tax": gross_pnl - tax,
            "pnl_pct":     _pnl_pct,
            "hold_days":   hold_days,
            "tax":         tax,
            "tax_cash_debited": tax_cash_debited,
            "tax_cash_debit_mode": tax_cash_debit_mode,
            "exit_reason": sig.exit_type,
            "partial":     is_partial,
            "regime":      getattr(ctx, "regime", None),
            "confidence":  getattr(ctx, "confidence", None),
            "exit_signal_reason": getattr(sig, "reason", None),
            "exit_stop_loss_pct": exit_p.get("stop_loss_pct"),
            "exit_stop_loss_anchor_policy": exit_p.get("stop_loss_anchor_policy"),
            "exit_stop_loss_anchor_regime": exit_p.get("stop_loss_anchor_regime"),
            "exit_stop_loss_current_regime": exit_p.get("stop_loss_current_regime"),
            "exit_stop_loss_current_pct": exit_p.get("stop_loss_current_pct"),
            "exit_stop_loss_entry_regime": exit_p.get("stop_loss_entry_regime"),
            "exit_stop_loss_entry_pct": exit_p.get("stop_loss_entry_pct"),
            "exit_stop_n_sigma": exit_p.get("stop_n_sigma"),
            "exit_take_profit_pct": exit_p.get("take_profit_pct"),
            "exit_stop_decay_days": exit_p.get("stop_decay_days"),
            "exit_stop_decay_floor": exit_p.get("stop_decay_floor"),
            "exit_max_single_day_loss_pct": exit_p.get("max_single_day_loss_pct"),
            "exit_sdl_n_sigma": exit_p.get("sdl_n_sigma"),
            "exit_sdl_skip_if_unrealized_above": exit_p.get(
                "sdl_skip_if_unrealized_above"
            ),
            "exit_trailing_stop_trigger_pct": exit_p.get("trailing_stop_trigger_pct"),
            "exit_trailing_stop_trail_pct": exit_p.get("trailing_stop_trail_pct"),
            "exit_atr_n_multiplier": exit_p.get("atr_n_multiplier"),
            "exit_max_hold_days": exit_p.get("max_hold_days"),
            "exit_max_hold_anchor_regime": exit_p.get("max_hold_anchor_regime"),
            "order_type":  f"SELL_{sig.exit_type}" if sig.exit_type else "SELL",
            "source":      str(getattr(sig, "source", None) or "ExitPipeline"),
            "source_job":  source_job,
            "source_task": source_task,
            "order_source": order_source,
            "attribution_version": "exit_decision_v1",
            "score_snapshot": {
                "rank_score": getattr(hs, "rank_score", None),
                "panel_score": getattr(hs, "panel_score", None),
                "mu": getattr(hs, "mu", None),
                "sigma": getattr(hs, "sigma", None),
                "kelly_target_pct": getattr(hs, "kelly_target_pct", None),
                "confidence": getattr(ctx, "confidence", None),
                "regime": getattr(ctx, "regime", None),
            },
            "decision_inputs": {
                "acceptance_reason": sig.exit_type or sig_reason,
                "exit_reason": sig.exit_type,
                "signal_reason": sig_reason,
                "partial": is_partial,
                "quantity": getattr(sig, "quantity", None),
                "hold_days": hold_days,
                "pnl_pct": _pnl_pct,
                "stop_loss_pct": exit_p.get("stop_loss_pct"),
                "stop_loss_anchor_policy": exit_p.get("stop_loss_anchor_policy"),
                "stop_loss_anchor_regime": exit_p.get("stop_loss_anchor_regime"),
                "stop_loss_current_regime": exit_p.get("stop_loss_current_regime"),
                "stop_loss_current_pct": exit_p.get("stop_loss_current_pct"),
                "stop_loss_entry_regime": exit_p.get("stop_loss_entry_regime"),
                "stop_loss_entry_pct": exit_p.get("stop_loss_entry_pct"),
                "stop_n_sigma": exit_p.get("stop_n_sigma"),
                "take_profit_pct": exit_p.get("take_profit_pct"),
                "stop_decay_days": exit_p.get("stop_decay_days"),
                "stop_decay_floor": exit_p.get("stop_decay_floor"),
                "max_single_day_loss_pct": exit_p.get("max_single_day_loss_pct"),
                "sdl_n_sigma": exit_p.get("sdl_n_sigma"),
                "sdl_skip_if_unrealized_above": exit_p.get(
                    "sdl_skip_if_unrealized_above"
                ),
                "trailing_stop_trigger_pct": exit_p.get("trailing_stop_trigger_pct"),
                "trailing_stop_trail_pct": exit_p.get("trailing_stop_trail_pct"),
                "atr_n_multiplier": exit_p.get("atr_n_multiplier"),
                "max_hold_days": exit_p.get("max_hold_days"),
                "max_hold_anchor_regime": exit_p.get("max_hold_anchor_regime"),
            },
        })

    def _apply_buy(self, order: dict, today_ts: pd.Timestamp, ctx) -> None:
        # Audit fix SAB-1..SAB-4 (Round 2 deep audit, 2026-04-25): pre-fix,
        # NaN price (data outage) or NaN shares could:
        #   - SAB-3: invest = NaN; `NaN > cash + 1e-6` False → buy "succeeds";
        #            self._cash -= NaN → cash permanently NaN.
        #   - SAB-2: old_entry * old_shares + NaN = NaN → new_entry NaN
        #            → future pnl_pct NaN.
        #   - SAB-1: max(hwm, NaN) = NaN → trailing stop dead for the position.
        # Now: explicit isfinite + > 0 guards on price/shares; reject the
        # order cleanly (log warning) on bad data.
        from kernel.exits import HoldingState, TaxLot, ensure_lots  # noqa: PLC0415
        import math
        ticker = order["ticker"]
        shares = order["shares"]
        price  = order["price"]
        if not (math.isfinite(price) and math.isfinite(shares)
                and price > 0 and shares > 0):
            log.warning(
                "SimAdapter: rejecting %s buy — bad price/shares (price=%s shares=%s)",
                ticker, price, shares,
            )
            return

        # ── Execution model: slippage + buy fees (Track Batch A) ────────────
        # Per §5.13.11 every monetary `>` is finite-guarded; per §5.13.5
        # routes through the single fee/slippage modules. When the
        # `legacy_no_fees` parity flag is on, both adjustments are skipped
        # for byte-identical pre-2026-05-10 behavior.
        # Defensive: __new__ test fixtures may bypass __init__ — preserve
        # legacy semantics when execution-model fields aren't initialized.
        if getattr(self, "_exec_enabled", False):
            fill_price = slip_fill_price(
                market_price=price, side="buy", shares=shares,
                adv_shares=None,  # retail order at < 0.1% ADV; impact off
                cfg=self._slip_cfg,
            )
            if not math.isfinite(fill_price) or fill_price <= 0:
                fill_price = price   # degenerate config → bail to market
            buy_fees = compute_buy_fees(shares, fill_price, self._fee_cfg)
        else:
            fill_price = price
            buy_fees = {"sec_fee": 0.0, "taf": 0.0, "custom": 0.0, "total": 0.0}

        invest = shares * fill_price + buy_fees["total"]
        if not math.isfinite(invest):
            log.warning("SimAdapter: %s buy invest non-finite — rejecting", ticker)
            return
        budget = getattr(self, "_buying_power_remaining", None)
        if budget is None or not math.isfinite(float(budget)):
            budget = self._available_buying_power()
        if invest > float(budget) + 1e-6:
            log.warning(
                "SimAdapter: insufficient buying power for %s "
                "(need %.2f, have %.2f; settled_cash=%.2f pending_settle=%.2f mode=%s)",
                ticker, invest, float(budget), self._cash,
                self._pending_settle_cash(),
                getattr(self, "_buying_power_mode", _BUYING_POWER_SETTLED),
            )
            return
        self._cash -= invest
        if getattr(self, "_buying_power_remaining", None) is not None:
            self._buying_power_remaining = float(budget) - invest
        if hasattr(self, "_total_fees"):
            self._total_fees += buy_fees["total"]
        # Lot cost basis records the post-slippage fill price; the buy
        # commission is treated as immediate cash drag (not capitalized).
        price = fill_price
        # If this ticker is already held (top-up path), increment shares
        # and adjust avg entry price. Otherwise fresh position.
        if ticker in self._holdings:
            old_shares = float(self._pos_shares.get(ticker, 0))
            new_shares = old_shares + shares
            old_entry  = self._holdings[ticker].entry_price
            # 2026-05-14 audit RED #A: short-cover P&L tax per §1233.
            # When old_shares < 0 (short position) and we BUY to cover,
            # the IRS §1233 says all P&L is short-term capital gains
            # regardless of holding period. Compute cover-side tax now
            # because the regular sell path (_apply_sell with tax) is
            # never invoked for shorts. Pre-fix this gave shorts a free
            # tax pass, inflating apparent v6 longshort_v1 result by an
            # estimated 2-5pt APY. Cover P&L = (short_entry_price -
            # cover_price) × covered_shares (positive when short
            # profited, taxed; negative = loss, no tax).
            if old_shares < 0 and math.isfinite(old_entry):
                covered_shares = min(shares, -old_shares)  # how many we actually closed
                if covered_shares > 0:
                    short_pnl = (old_entry - price) * covered_shares
                    if short_pnl > 0:
                        from kernel.portfolio import compute_trade_tax  # noqa: PLC0415
                        tax_cfg = ctx.config.get("tax", {}) if ctx is not None else {}
                        # §1233: shorts always short-term. hold_days=0
                        # forces ST rate.
                        short_tax = compute_trade_tax(
                            short_pnl, hold_days=0,
                            short_term_rate=float(tax_cfg.get("short_term_rate", 0.50)),
                            long_term_rate=float(tax_cfg.get("long_term_rate", 0.32)),
                            long_term_threshold_days=int(tax_cfg.get("long_term_threshold_days", 365)),
                        )
                        short_tax_cash_debited = _tax_cash_debit_amount(
                            ctx.config if ctx is not None else {}, short_tax,
                        )
                        if math.isfinite(short_tax_cash_debited) and short_tax_cash_debited > 0 \
                                and math.isfinite(self._cash):
                            self._cash -= short_tax_cash_debited
                            if not hasattr(self, "_tax_cash_debited"):
                                self._tax_cash_debited = 0.0
                            self._tax_cash_debited += float(short_tax_cash_debited)
                            log.info(
                                "SHORT_COVER_TAX %s short_pnl=$%.2f tax=$%.2f "
                                "tax_cash_debited=$%.2f cover_shares=%d",
                                ticker, short_pnl, short_tax,
                                short_tax_cash_debited, int(covered_shares),
                            )
                    # §1233(e) → §1091: short-cover loss creates a wash-sale
                    # window for subsequent LONG buys of substantially identical
                    # security within 30 days. Stamp _last_sell_date +
                    # _last_sell_pls so WashSaleFilterTask sees this exposure.
                    if short_pnl < 0:
                        if not hasattr(self, "_last_sell_date"):
                            self._last_sell_date = {}
                        self._last_sell_date[ticker] = today_ts
                        if not hasattr(self, "_last_sell_pls"):
                            self._last_sell_pls = {}
                        self._last_sell_pls[ticker] = float(short_pnl)
            # SAB-2 guard: if old_entry is NaN/inf (corrupted state), use
            # the current price as entry rather than corrupt new_entry.
            if not math.isfinite(old_entry):
                new_entry = price
            else:
                # Bug-bounty #3 fix: when covering a short (old_shares<0) or
                # crossing zero (short → long via over-cover), the avg-cost
                # formula above with signed shares produces a nonsensical
                # entry. Treat any cross-zero or short-cover as a new
                # position at the current fill price.
                if old_shares < 0 or new_shares <= 0:
                    new_entry = price
                else:
                    new_entry = (old_entry * old_shares + price * shares) / new_shares
            self._holdings[ticker].entry_price = new_entry
            cur_hwm = self._holdings[ticker].high_watermark
            if math.isfinite(cur_hwm):
                self._holdings[ticker].high_watermark = max(cur_hwm, price)
            else:
                # SAB-1 recovery: corrupted HWM → reset to price.
                self._holdings[ticker].high_watermark = price
            self._holdings[ticker].shares = new_shares
            self._pos_shares[ticker] = new_shares
            # G7: append a TaxLot for this top-up. Migrate any pre-G7
            # legacy state (lots empty but legacy avg/entry exist) first
            # so the existing position has a baseline lot.
            ensure_lots(self._holdings[ticker])
            self._holdings[ticker].lots.append(TaxLot(
                shares=float(shares), price=float(price), date=today_ts.date(),
            ))
        else:
            hs_new = HoldingState(
                entry_price    = price,
                entry_date     = today_ts.date(),
                high_watermark = price,
                prev_close     = price,
                shares         = shares,
                # Thesis-degradation baseline (Approach A) — snapshot
                # today's decision signals so future rotation checks
                # can compare today's scores to THESE fixed anchors.
                entry_rank_score       = order.get("rank_score"),
                entry_panel_score      = order.get("panel_score"),
                entry_kelly_target_pct = order.get("kelly_target_pct"),
                entry_regime           = order.get("regime"),
            )
            # G7: seed the lot list with this fresh acquisition.
            hs_new.lots.append(TaxLot(
                shares=float(shares), price=float(price), date=today_ts.date(),
            ))
            self._holdings[ticker] = hs_new
            self._pos_shares[ticker] = shares
        self._trade_log.append({
            "action":    "buy",
            "ticker":    ticker,
            "date":      today_ts,
            "price":     price,
            "shares":    shares,
            "invest":    invest,
            "regime":    order.get("regime") or getattr(ctx, "regime", None),
            "rank_score": order.get("rank_score"),
            "rs_score":  order.get("rs_score"),
            "sigma":     order.get("sigma"),
            "mu":        order.get("mu"),
            "sigma_mult": order.get("sigma_mult"),
            "source":     order.get("source"),
            "order_type": order.get("order_type"),
            "confidence": order.get("confidence"),
            "kelly_target_pct": order.get("kelly_target_pct"),
            "attribution_version": order.get("attribution_version"),
            "source_job": order.get("source_job"),
            "source_task": order.get("source_task"),
            "order_source": order.get("order_source"),
            "panel_score": order.get("panel_score"),
            "expected_return": order.get("expected_return"),
            "score_snapshot": order.get("score_snapshot"),
            "decision_inputs": order.get("decision_inputs"),
        })

    def _apply_short_open(self, ticker: str, sig, today_ts, ctx) -> None:
        """Phase 2B: open a new short position (negative shares).

        - Credits cash with short-sale proceeds (less commission/slippage).
        - Creates a HoldingState with shares = -|N|, entry_price = fill.
        - Daily borrow charge handled by _charge_daily_borrow.
        - Cover happens through _apply_buy when QP wants positive Δw on
          a ticker with shares < 0 (added in a follow-up).
        """
        from kernel.exits import HoldingState  # noqa: PLC0415
        import math as _math_q  # noqa: PLC0415
        shares = float(getattr(sig, "quantity", 0) or 0)
        if not _math_q.isfinite(shares) or shares <= 0:
            log.warning("SimAdapter._apply_short_open: %s bad quantity=%s, skipping",
                        ticker, shares)
            return
        # Mark price = today's close on the model OHLCV
        df = self._ohlcv.get(ticker)
        if df is None or today_ts not in df.index:
            log.warning("SimAdapter._apply_short_open: %s no price today, skipping", ticker)
            return
        price = float(df.loc[today_ts, "close"])
        if not _math_q.isfinite(price) or price <= 0:
            log.warning("SimAdapter._apply_short_open: %s bad price=%s, skipping",
                        ticker, price)
            return

        # Slippage + commission (treat like a sell)
        if getattr(self, "_exec_enabled", False):
            fill_price = slip_fill_price(
                market_price=price, side="sell", shares=shares,
                adv_shares=None, cfg=self._slip_cfg,
            )
            if not _math_q.isfinite(fill_price) or fill_price <= 0:
                fill_price = price
            fees = compute_sell_fees(shares, fill_price, self._fee_cfg)
        else:
            fill_price = price
            fees = {"sec_fee": 0.0, "taf": 0.0, "custom": 0.0, "total": 0.0}

        proceeds = shares * fill_price - fees["total"]
        if not _math_q.isfinite(proceeds):
            log.warning("SimAdapter._apply_short_open: %s non-finite proceeds, skipping", ticker)
            return

        # Credit cash with short proceeds (held as margin; daily borrow
        # cost handled by _charge_daily_borrow). T+N is ignored for
        # shorts in this MVP — Alpaca settles short proceeds T+1 anyway.
        self._cash += proceeds
        if hasattr(self, "_total_fees"):
            self._total_fees += fees["total"]

        # 2026-05-14 audit Bug B fix: log short opens in _trade_log so
        # downstream summaries (Avg P&L/trade, total trades, etc.) include
        # them. Pre-fix shorts were invisible to the trade-level analytics.
        self._trade_log.append({
            "action":      "short_open",
            "ticker":      ticker,
            "date":        today_ts,
            "price":       fill_price,
            "shares":      shares,         # positive magnitude (short side)
            "pnl_pct":     0.0,            # P&L realized only on cover
            "hold_days":   0,
            "tax":         0.0,
            "exit_reason": "short_open",
            "partial":     False,
        })

        # Create or update the HoldingState with NEGATIVE shares
        if ticker in self._holdings:
            existing = self._holdings[ticker]
            # Should only happen if a prior short increased magnitude.
            existing.shares = (existing.shares or 0) - shares
            # Average down entry on the short side
            existing.entry_price = fill_price  # simplistic; refine if needed
        else:
            self._holdings[ticker] = HoldingState(
                shares=-shares,
                entry_price=fill_price,
                entry_date=today_ts.date() if hasattr(today_ts, "date") else today_ts,
                high_watermark=fill_price,
            )
        self._pos_shares[ticker] = self._holdings[ticker].shares
        log.info("SHORT_OPEN %s shares=-%d px=%.2f proceeds=$%.0f",
                 ticker, int(shares), fill_price, proceeds)

    def _charge_daily_borrow(self, today_ts) -> None:
        """Phase 2B: daily borrow cost charge on short positions.

        Per Alpaca live API (2026-05-14 research, see
        `data/alpaca_borrow_status.json`):
          - easy_to_borrow=True (ETB): borrow_rate_etb (default 0.005/yr)
          - easy_to_borrow=False (HTB): borrow_rate_htb (default 0.05/yr)
          - shortable=False: cannot short — filtered upstream

        Charged daily: cost = |short_value| × borrow_rate / 252.

        No-op when no negative-share positions exist. Safe to call on
        every bar (cheap dict iteration).
        """
        if not getattr(self, "holdings", None):
            return
        # Lazy-load borrow status once
        if not hasattr(self, "_borrow_status_cache"):
            import json as _json  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415
            p = Path("data/alpaca_borrow_status.json")
            try:
                self._borrow_status_cache = (
                    _json.loads(p.read_text()).get("results", {})
                    if p.exists() else {}
                )
            except Exception:
                self._borrow_status_cache = {}
        bs = self._borrow_status_cache
        # Rates from config or defaults
        cfg = getattr(self, "_strategy_config", {}) or {}
        ls_cfg = cfg.get("long_short", {}) or {}
        rate_etb = float(ls_cfg.get("borrow_rate_etb", 0.005))
        rate_htb = float(ls_cfg.get("borrow_rate_htb", 0.05))
        total_charge = 0.0
        for ticker, hs in self.holdings.items():
            shares = float(getattr(hs, "shares", 0.0) or 0.0)
            if shares >= 0:
                continue
            # Need current price — use entry as fallback when not in self._ohlcv
            df = self._ohlcv.get(ticker) if hasattr(self, "_ohlcv") else None
            if df is not None and today_ts in df.index:
                price = float(df.loc[today_ts, "close"])
            else:
                price = float(getattr(hs, "entry_price", 0.0) or 0.0)
            if not math.isfinite(price) or price <= 0:
                continue
            short_value = abs(shares) * price
            info = bs.get(ticker, {})
            etb = info.get("easy_to_borrow", True)  # fail-open: assume ETB
            rate = rate_etb if etb else rate_htb
            daily_cost = short_value * rate / 252.0
            if math.isfinite(daily_cost) and daily_cost > 0:
                total_charge += daily_cost
        # Bug-bounty #4 fix: skip charge if would overdraw cash. Real
        # broker would margin-call, not let cash go negative. For sim
        # purposes, defer the charge to next bar when settlements arrive.
        if total_charge > 0 and math.isfinite(self._cash) \
                and total_charge < self._cash:
            self._cash -= total_charge
        elif total_charge > 0 and math.isfinite(self._cash):
            log.warning(
                "_charge_daily_borrow: skipping $%.2f charge (cash=$%.2f) "
                "to prevent overdraw — real broker would margin-call",
                total_charge, self._cash,
            )

    def _portfolio_value(self, prices: dict[str, float], today_ts=None) -> float:
        """Mark-to-market the held positions.

        Bug 25 fix (2026-04-24): when a holding has no price in the
        per-bar `prices` dict (delisted / suspended / new IPO not yet
        trading), we fall back to the last AVAILABLE close ON OR
        BEFORE today_ts — NOT `df.iloc[-1]` of the full ohlcv (which
        is the LAST historical bar = future data in a sim).

        Audit fix SA-1 (Round 9, 2026-04-25): pre-fix, NaN/inf in either
        `prices.get(t)` or the fallback close silently propagated into
        `total += shares * NaN = NaN`. Once corrupted, every subsequent
        `_portfolio_value` call returned NaN — equity curve filled with
        NaN, total_ret/APY came out NaN. Now: skip non-finite prices
        (treat as zero contribution) so a single bad bar doesn't poison
        the rest of the simulation.
        """
        import math
        # 2026-05-11 BUG #C fix: include T+N pending settlement balance.
        # Pre-fix, `_portfolio_value` returned cash + position MTM but
        # ignored sell proceeds sitting in `_t2_queue`. Result: on sell
        # day, shares drop but cash unchanged (proceeds queued) ⇒ NAV
        # phantom drop = sale amount. Two bars later when queue drains,
        # NAV phantom recovery = sale amount. With many sells this
        # inflates measured ann_vol by O(sale_size · √n_sells) — observed
        # 516% vol on W1_maxpos08 (45 sells × $5k positions).
        # Fix invariant (CLAUDE.md §5.3): NAV ≡ free_cash +
        # pending_settle + Σ(shares × price). All three legs are real
        # claim against the portfolio's economic value.
        total = self._cash
        if getattr(self, "_t2_queue", None) is not None:
            pending = self._t2_queue.pending_total()
            if math.isfinite(pending):
                total += pending
        for t, shares in self._pos_shares.items():
            p = prices.get(t)
            if p is None or not math.isfinite(p):
                df = self._ohlcv.get(t)
                if df is not None and not df.empty:
                    if today_ts is not None:
                        truncated = df.loc[:today_ts]
                        if not truncated.empty:
                            cand = float(truncated["close"].iloc[-1])
                            p = cand if math.isfinite(cand) else None
                        else:
                            p = None
                    else:
                        # No truncation hint — caller is responsible
                        # for not introducing lookahead.
                        cand = float(df["close"].iloc[-1])
                        p = cand if math.isfinite(cand) else None
            if p is not None and math.isfinite(p):
                total += shares * p
        return total

    # ── Summary accessors ───────────────────────────────────────────────────

    def build_result(self):
        """Return a SimResult equivalent to the legacy hand-written runner."""
        from sim.runner import SimResult  # noqa: PLC0415

        # P4.1 (2026-05-11) — flush per-day position-snapshot buffer to
        # parquet. No-op if meta_label_training was disabled (logger=None)
        # or the adapter was constructed via __new__() for testing
        # (no _meta_label_logger attr). Wrapped in try/except so a parquet
        # I/O hiccup doesn't kill the whole result builder — the sim
        # metrics are the primary output.
        _ml_log = getattr(self, "_meta_label_logger", None)
        _ml_out = getattr(self, "_meta_label_output_path", None)
        if _ml_log is not None and _ml_out:
            try:
                out_path = Path(_ml_out)
                if not out_path.is_absolute():
                    out_path = Path(self._strategy_dir).parent.parent / out_path
                _ml_log.dump_to_parquet(out_path)
                log.info(
                    "SimAdapter.build_result: dumped %d position-day snapshots → %s",
                    _ml_log.n_rows(), out_path,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "SimAdapter.build_result: snapshot dump failed (%s); "
                    "continuing with SimResult build", exc,
                )

        equity_df = pd.DataFrame(self._equity_curve).set_index("date") if self._equity_curve \
            else pd.DataFrame(columns=["portfolio", "regime"])
        final_val = float(equity_df["portfolio"].iloc[-1]) if not equity_df.empty else self._initial_cash
        total_ret = final_val / self._initial_cash - 1.0
        # 2026-05-10 audit fix: off-by-one. N equity points imply N-1
        # inter-day return periods over (N-1)/252 trading years. Pre-fix
        # used `len/252`, which over-counted year length by one bar and
        # disagreed with `risk_metrics.py:241` (which already uses N-1).
        # Pinned by tests/test_risk_metrics_extra.py::test_n_years_consistency.
        n_years = (len(equity_df) - 1) / 252 if len(equity_df) >= 2 else 0
        apy = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        sells = [t for t in self._trade_log if t["action"] == "sell"]
        wins  = [t for t in sells if t.get("pnl_pct", 0.0) > 0]
        win_rate  = len(wins) / max(1, len(sells))
        avg_hold  = sum(t.get("hold_days", 0) for t in sells) / len(sells) if sells else 0.0
        avg_pnl   = sum(t.get("pnl_pct",   0) for t in sells) / len(sells) if sells else 0.0
        total_tax = sum(_finite_float(t.get("tax"), default=0.0) for t in sells)
        tax_cash_debited = sum(
            _finite_float(
                t.get("tax_cash_debited"),
                default=_finite_float(t.get("tax"), default=0.0),
            )
            for t in sells
        )
        # Short-cover tax is recorded as a cash debit during _apply_buy because
        # the action that realizes short P&L is a buy-to-cover, not a sell.
        # Keep build_result compatible with __new__ fixtures and legacy logs by
        # taking the larger of explicit sell-row debits and the adapter counter.
        tax_cash_debited = max(
            float(tax_cash_debited),
            _finite_float(getattr(self, "_tax_cash_debited", 0.0), default=0.0),
        )
        exit_reasons = dict(Counter(t.get("exit_reason", "?") for t in sells))
        tax_cfg = (getattr(self, "_config", {}) or {}).get("tax", {}) or {}
        from kernel.portfolio import compute_annual_net_capital_gains_tax  # noqa: PLC0415
        annual_tax_summary = compute_annual_net_capital_gains_tax(
            sells,
            short_term_rate=float(tax_cfg.get("short_term_rate", 0.50)),
            long_term_rate=float(tax_cfg.get("long_term_rate", 0.32)),
            long_term_threshold_days=int(tax_cfg.get("long_term_threshold_days", 365)),
        )
        annual_net_tax = float(annual_tax_summary["total_estimated_tax"])
        annual_net_final_val = final_val + float(tax_cash_debited) - annual_net_tax
        annual_net_total_ret = annual_net_final_val / self._initial_cash - 1.0
        annual_net_apy = (
            (1 + annual_net_total_ret) ** (1 / n_years) - 1
            if n_years > 0 and annual_net_total_ret > -1
            else 0.0
        )
        annual_net_equity_df = _annual_net_equity_curve(
            equity_df, sells, annual_tax_summary,
        )

        # Rotation sell/buy pairs (same-day sell with exit_reason=rotation + same-day rotation buy)
        rotations: list[dict] = []
        for s in sells:
            if s.get("exit_reason") != "rotation":
                continue
            sd = s["date"].date() if hasattr(s["date"], "date") else s["date"]
            same_day_buys = [
                b for b in self._trade_log
                if b["action"] == "buy"
                and (b["date"].date() if hasattr(b["date"], "date") else b["date"]) == sd
            ]
            rotations.append({
                "date": sd, "sell": s["ticker"],
                "buy": same_day_buys[0]["ticker"] if same_day_buys else "?",
                "pnl_pct": s.get("pnl_pct", 0.0),
                "hold_days": s.get("hold_days", 0),
                "tax": s.get("tax", 0.0),
            })

        # Activity-monitoring stats: longest run of consecutive trading days
        # without any order (buy or sell). Computed post-hoc from the equity
        # curve + trade log so it always reflects the whole OOS window.
        trade_dates = {
            (t["date"].date() if hasattr(t["date"], "date") else t["date"])
            for t in self._trade_log
        }
        eq_dates = [
            (d.date() if hasattr(d, "date") else d) for d in equity_df.index
        ] if not equity_df.empty else []
        longest_streak = 0
        current_streak = 0
        first_trade: "str | None" = None
        last_activity: "str | None" = None
        for d in eq_dates:
            if d in trade_dates:
                current_streak = 0
                last_activity = str(d)
                if first_trade is None:
                    first_trade = str(d)
            else:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)

        # Risk-adjusted metrics (2026-05-02 §3 instrumentation). Computed
        # from the equity curve so they reflect the full OOS window. NaN
        # is propagated when there's insufficient data — caller (sim
        # runner / B2 hold-out) renders NaN as "—" rather than zero.
        from kernel.risk_metrics import (  # noqa: PLC0415
            alpha_vs_benchmark,
            beta_vs_benchmark,
            compute_risk_metrics,
            daily_returns_from_equity,
            information_ratio,
        )
        # 2026-05-10 (Track C5): risk-free rate is now config-driven. Default
        # 0.0 preserves byte-identical legacy behavior; production opts in
        # via cfg["performance"]["risk_free_rate_annual"]. include_geometric
        # adds Israelsen-2003 geometric Sharpe alongside the arithmetic form.
        _rf_annual = float(
            (self._config or {}).get("performance", {}).get(
                "risk_free_rate_annual", 0.0,
            )
        )
        if not equity_df.empty and "portfolio" in equity_df.columns:
            risk = compute_risk_metrics(
                equity_df["portfolio"],
                apy=apy,
                risk_free_rate=_rf_annual,
                include_geometric=True,
            )
        else:
            risk = {
                "sharpe":  float("nan"), "sortino": float("nan"),
                "calmar":  float("nan"), "max_dd":  float("nan"),
                "ann_vol": float("nan"),
                "sharpe_geometric": float("nan"),
            }
        if (not annual_net_equity_df.empty
                and "portfolio" in annual_net_equity_df.columns):
            annual_net_risk = compute_risk_metrics(
                annual_net_equity_df["portfolio"],
                apy=annual_net_apy,
                risk_free_rate=_rf_annual,
                include_geometric=True,
            )
        else:
            annual_net_risk = {
                "sharpe":  float("nan"), "sortino": float("nan"),
                "calmar":  float("nan"), "max_dd":  float("nan"),
                "ann_vol": float("nan"),
            }

        # 2026-05-10 audit (§5.13.4): single-Sharpe-without-falsifiability
        # is an unverified claim. Wire DSR (Bailey/Borwein/López de Prado
        # 2014, "The Deflated Sharpe Ratio") + PBO (Bailey/Borwein/López
        # de Prado/Zhu 2015 CSCV) and the benchmark triple β/α/IR (Sharpe
        # 1964 / Treynor-Black 1973) into every SimResult.
        perf_cfg = (self._config or {}).get("performance", {}) or {}
        n_trials = int(perf_cfg.get("n_trials", 1))
        dsr_val = float("nan")
        pbo_val = float("nan")
        if not equity_df.empty and "portfolio" in equity_df.columns:
            try:
                from kernel.metrics import compute_perf_triple  # noqa: PLC0415
                import numpy as _np  # noqa: PLC0415
                rets = daily_returns_from_equity(equity_df["portfolio"]).dropna()
                if len(rets) >= 2:
                    triple = compute_perf_triple(
                        returns=rets.to_numpy(dtype=float),
                        n_trials=n_trials,
                    )
                    dsr_val = float(triple["dsr"])
                    pbo_val = float(triple["pbo"])  # NaN in single-seed mode
            except Exception as _exc:  # noqa: BLE001
                # Falsifiability is opportunistic — never block sim emission
                # on scipy import errors / degenerate inputs. Log + emit NaN.
                log.warning("build_result: perf_triple computation failed: %s", _exc)

        # Benchmark-relative β / α / IR vs SPY. Compare daily returns to
        # SPY daily returns over the same date range.
        beta_spy = float("nan")
        alpha_spy = float("nan")
        ir_spy = float("nan")
        if (not equity_df.empty
                and "portfolio" in equity_df.columns
                and self._spy_df is not None
                and not self._spy_df.empty
                and "close" in self._spy_df.columns):
            port_rets = daily_returns_from_equity(equity_df["portfolio"])
            spy_aligned = self._spy_df["close"].reindex(equity_df.index)
            spy_rets = daily_returns_from_equity(spy_aligned)
            beta_spy = beta_vs_benchmark(port_rets, spy_rets)
            alpha_spy = alpha_vs_benchmark(port_rets, spy_rets, beta=beta_spy)
            ir_spy = information_ratio(port_rets, spy_rets)

        return SimResult(
            equity_df     = equity_df,
            trade_log     = self._trade_log,
            rotation_log  = self._rotation_log,
            final_value   = final_val,
            total_return  = total_ret,
            apy           = apy,
            win_rate      = win_rate,
            avg_hold      = avg_hold,
            avg_pnl       = avg_pnl,
            total_tax     = total_tax,
            exit_reasons  = exit_reasons,
            rotations     = rotations,
            event_level_tax_estimate      = float(total_tax),
            tax_cash_debited              = float(tax_cash_debited),
            tax_cash_debit_mode           = _tax_cash_debit_mode(
                getattr(self, "_config", {}),
            ),
            event_level_tax_debited       = float(tax_cash_debited),
            annual_net_tax_estimate       = annual_net_tax,
            tax_overstatement_vs_annual_net = float(total_tax) - annual_net_tax,
            annual_net_final_value_estimate = annual_net_final_val,
            annual_net_total_return_estimate = annual_net_total_ret,
            annual_net_apy_estimate       = annual_net_apy,
            annual_net_equity_df_estimate = annual_net_equity_df,
            annual_net_sharpe_estimate    = annual_net_risk["sharpe"],
            annual_net_sortino_estimate   = annual_net_risk["sortino"],
            annual_net_calmar_estimate    = annual_net_risk["calmar"],
            annual_net_max_dd_estimate    = annual_net_risk["max_dd"],
            annual_net_ann_vol_estimate   = annual_net_risk["ann_vol"],
            annual_net_tax_years          = annual_tax_summary["years"],
            longest_no_trade_streak     = longest_streak,
            longest_no_candidate_streak = int(self._monitor_state.get("no_candidate_streak", 0)),
            first_trade_date            = first_trade,
            last_activity_date          = last_activity,
            sharpe        = risk["sharpe"],
            sortino       = risk["sortino"],
            calmar        = risk["calmar"],
            max_dd        = risk["max_dd"],
            ann_vol       = risk["ann_vol"],
            dsr                       = dsr_val,
            pbo                       = pbo_val,
            n_trials                  = n_trials,
            beta_vs_spy               = beta_spy,
            alpha_vs_spy              = alpha_spy,
            information_ratio_vs_spy  = ir_spy,
        )
