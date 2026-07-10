"""RunnerAdapter — bridges live broker state → InferenceContext → order execution.

Can import kernel/ and common/ (runs on host, not in LEAN Docker).
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from adapters.panel_runtime import (
    attach_panel_runtime_frames,
    build_runtime_feature_cache,
    describe_panel_frame_bundle,
    prepare_panel_runtime_frames,
)
from kernel.decision_trace import (
    build_ticker_daily_state_rows,
    candidate_trace_pool,
    model_type_from_artifact as _shared_model_type_from_artifact,
    model_types_from_models,
    qp_trace_maps,
    selected_buy_tickers,
    trade_event_blocked_map,
    trade_event_tickers,
)
from kernel.pipeline.task_execution import (
    dedupe_exit_signals,
    is_full_liquidate_signal,
)
from kernel.trade_events import (
    build_buy_trade_event,
    build_sell_trade_event as build_sell_trade_event_for_db,
)

log = logging.getLogger("adapters.runner")


# ── Helpers ────────────────────────────────────────────────────────────────────

# ── Runner prep helpers — EXTRACTED to runner_prep.py ──────────────────
# (eng plan S2 item 5 decomposition slice 7, 2026-06-13.)
from adapters.runner_prep import (  # noqa: F401,E402
    _HWM_STALE_RATIO,
    _held_mark_ohlcv_frame,
    _parse_iso_dt,
    persisted_skip_buys,
    resolve_hwm,
)


# ── Live sell tax-lot accounting — EXTRACTED to runner_tax_lots.py ──────
# (eng plan S2 item 5 decomposition slice 6, 2026-06-13.)
from adapters.runner_tax_lots import (  # noqa: F401,E402
    apply_live_sell_lot_accounting,
    reconstruct_live_tax_lots_from_fills,
    sell_event_price,
    sell_event_realized_kwargs,
)


# ── Live trace builders — EXTRACTED to runner_trace.py ──────────────────
# (eng plan S2 item 5 decomposition slice 4, 2026-06-13.)
from adapters.runner_trace import (  # noqa: F401,E402
    _buy_attempt_event,
    _sell_attempt_event,
    live_execution_attempt_events,
    live_trace_selection_maps,
)


# ── Execution math — EXTRACTED to runner_execmath.py ────────────────────
# (eng plan S2 item 5 decomposition slice 5, 2026-06-13.)
from adapters.runner_execmath import (  # noqa: F401,E402
    broker_order_execution,
    cap_buy_order_to_cash,
    effective_live_holdings_after_orders,
    live_post_execution_snapshot,
    normalize_order_status,
    same_bar_sell_credit,
)

# ── L6 score-drift audit sidecar — commit() entry point ─────────────────
from adapters.runner_l6 import run_l6_score_audit_sidecar  # noqa: F401,E402


def _preopen_cancel_symbols(strategy_dir: Path, broker_name: str | None, today_str: str) -> set[str]:
    """Symbols whose queued orders were cancelled by the pre-open gate today."""
    if broker_name != "alpaca":
        return set()
    ledger = strategy_dir.parent.parent / "logs" / "alerts" / "preopen_cancel_ledger.jsonl"
    if not ledger.exists():
        return set()
    out: set[str] = set()
    try:
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("date") == today_str and row.get("broker") == broker_name:
                sym = str(row.get("symbol") or "").strip().upper()
                if sym:
                    out.add(sym)
    except Exception as exc:  # noqa: BLE001
        log.warning("preopen cancel ledger read failed: %s", exc)
    return out


def _build_tournament_shadow_ticker_scores(
    cand_pool: list[Any],
    blocked_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build the ``ticker_scores`` dict for tournament shadow admission.

    Sources:
      * ``cand_pool`` — CandidateResult objects from the pipeline (these
        tickers survived far enough to have scores recorded).
      * ``blocked_map`` — per-ticker block reasons; tickers blocked at the
        tournament gate carry ``model_signal:<signal>``.

    Tickers in ``cand_pool`` that were NOT blocked by the tournament gate
    implicitly had signal = "buy".
    """
    scores: dict[str, dict[str, Any]] = {}
    for cand in cand_pool:
        ticker = getattr(cand, "ticker", None)
        if not ticker:
            continue
        scores[ticker] = {
            "signal": "buy",  # survived ScoreBuyTask
            "raw_score": getattr(cand, "raw_score", None),
            "rank_score": getattr(cand, "rank_score", None),
        }
    # Overlay block reasons — tickers blocked at the tournament gate have
    # the signal embedded in the block reason (``model_signal:<signal>``).
    for ticker, reason in blocked_map.items():
        if ticker in scores:
            # Already in cand_pool with scores; check if the block was
            # downstream of the tournament gate (signal was still "buy").
            continue
        if reason.startswith("model_signal:"):
            signal = reason.split(":", 1)[1]
            scores[ticker] = {
                "signal": signal,
                "raw_score": None,
                "rank_score": None,
            }
        # Other block reasons (wash_sale, sector_cap, etc.) — the ticker
        # may not have been scored at all; leave absent from ticker_scores
        # so the shadow logger marks it as "no_model_data".
    return scores


def model_type_from_artifact(model: Any) -> str | None:
    """Extract the human model type from dict/object artifacts for DB audit rows."""
    return _shared_model_type_from_artifact(model)


class RunnerAdapter:
    """Translate between the live broker/state and InferenceContext.

    Usage::

        adapter = RunnerAdapter(config, models, broker, strategy_dir, sell_only)
        ctx = adapter.make_context()
        InferencePipeline().run(ctx)
        adapter.commit(ctx)
    """

    def __init__(
        self,
        config: dict,
        models: dict,
        broker: Any,
        strategy_dir: Path,
        sell_only: bool = False,
        use_intraday_prices: bool = False,
    ) -> None:
        self._config              = config
        self._models              = models
        self._broker              = broker
        self._strategy_dir        = strategy_dir
        self._sell_only           = sell_only
        self._use_intraday_prices = use_intraday_prices
        self._universe_rejections = dict(
            config.get("_universe_rejections") or {}
        )
        # S-FRAC stage 0 (design 2026-07-02 §2.2.2) stage-3 seam, filled
        # by sprint D2: the software-stop registry (loop-resident
        # protection for fractional quantities that cannot ride a broker
        # GTC stop) is constructed flag-gated BELOW, once _broker_name is
        # known (the registry state file is broker-isolated like every
        # other live state file). Default OFF ⇒ stays None ⇒
        # commit_contract.software_stops_armed() is False ⇒ any
        # fractional BUY intent is fail-closed at entry and Z9 routing
        # reports fractional holdings as unprotectable (loud, never a
        # truncated whole-share stop).
        self._software_stops = None

        # 2026-04-27: broker-isolated state. paper / alpaca-paper / alpaca
        # each get their own live_state.{broker}.json + runs.{broker}.db so
        # a paper smoke can never contaminate alpaca-live state. See
        # kernel/state_paths.py for the path convention.
        # 2026-04-28 self-audit (TEST-2 follow-up): require str to avoid
        # Mock objects in tests (or any non-str caller) tripping the
        # allowlist check inside state_paths._safe_broker.
        _bn = getattr(broker, "broker_name", None)
        self._broker_name: str | None = _bn if isinstance(_bn, str) else None

        # S-FRAC stage 3 (sprint D2): flag-gated software-stop registry.
        # execution.software_stops.enabled=false/absent ⇒ from_config
        # returns None and NOTHING changes (byte-inert; no file is ever
        # created). A registry that fails to construct is NOT armed —
        # fail-closed, fractional entries stay blocked by the stage-0
        # capability gate.
        try:
            # 2026-07-04: relocated to renquant_pipeline.software_stops
            # (renquant-pipeline#167) -- new capability logic belongs in
            # an owning repo, not the umbrella (RenQuant#440 review).
            from renquant_pipeline.software_stops import SoftwareStopRegistry  # noqa: PLC0415
            self._software_stops = SoftwareStopRegistry.from_config(
                config, broker_name=self._broker_name,
            )
        except Exception as exc:
            log.error(
                "software-stop registry construction FAILED: %s — layer "
                "NOT armed; fractional entries remain fail-closed by the "
                "stage-0 capability gate.", exc,
            )
            self._software_stops = None

        # Mutate config.persistence.db_path to broker-specific BEFORE
        # constructing the DB connection (kernel.persistence reads it).
        if self._broker_name:
            from kernel.state_paths import runs_db_path  # noqa: PLC0415
            persist_cfg = config.setdefault("persistence", {})
            base_db = persist_cfg.get("db_path", "data/runs.db")
            persist_cfg["db_path"] = str(runs_db_path(base_db, self._broker_name))

        from kernel.persistence import get_connection  # noqa: PLC0415
        self._db = get_connection(config, strategy_dir=strategy_dir)

        # ── Meta-label snapshot logger (P5, 2026-05-11) ────────────────────
        # Mirror of SimAdapter wiring. Owned by adapter so it persists
        # across bars. Attached to ctx in make_context(); MetaLabelLoggingJob
        # writes one row per held ticker per bar; dumped at runner
        # teardown. Disabled (None) when meta_label_training.enabled is
        # false — i.e. always disabled in prod, only ON during a
        # dedicated training data-capture run (intraday cron or research
        # batch).
        ml_train_cfg = config.get("meta_label_training", {}) or {}
        if ml_train_cfg.get("enabled", False):
            from kernel.meta_label import SnapshotLogger  # noqa: PLC0415
            self._meta_label_logger = SnapshotLogger()
            self._meta_label_output_path = str(
                ml_train_cfg.get("output_path", "data/position_day_snapshots.parquet")
            )
        else:
            self._meta_label_logger = None
            self._meta_label_output_path = None

        # ── Meta-label veto predictor (P5, 2026-05-11) ────────────────────
        # Loads the XGBoost classifier trained by scripts/_meta_label_train.py
        # and exposes a `predictor(feats: dict) -> P(profitable_exit)`
        # callable that MetaLabelVetoTask queries to drop false-positive
        # path-rule exits. This is the PROD deployment surface for the
        # meta-label mechanism — same artifact format / fallback contract
        # as SimAdapter so models trained in sim research deploy cleanly
        # to live without code change.
        veto_cfg = (config.get("ranking") or {}).get("meta_label") or {}
        if veto_cfg.get("enabled", False):
            from kernel.meta_label.predictor import load_meta_label_predictor  # noqa: PLC0415
            art_path = veto_cfg.get(
                "artifact_path",
                "backtesting/renquant_104/artifacts/meta-label-exit.json",
            )
            art_resolved = Path(art_path)
            if not art_resolved.is_absolute():
                art_resolved = Path(strategy_dir).parent.parent / art_resolved
            self._meta_label_predictor = load_meta_label_predictor(art_resolved)
        else:
            self._meta_label_predictor = None

    # ── make_context ───────────────────────────────────────────────────────────

    def make_context(self):  # noqa: ANN201
        """Build InferenceContext from broker, parquet cache, and live_state.json."""
        from kernel.pipeline.context import InferenceContext  # noqa: PLC0415
        from kernel.regime import RegimeState                  # noqa: PLC0415
        from kernel.config import REGIMES                      # noqa: PLC0415

        config  = self._config
        today   = datetime.date.today()
        broker  = self._broker

        # ── Load persisted live state ────────────────────────────────────────
        # Plan #144 (2026-04-26 round-7): db is canonical, JSON is cache.
        # Read JSON first (fast). On JSON missing/corrupt, fall back to
        # the latest live_state_snapshots row (per-bar mirror).
        # Per user spec: "live state json应该至少备份在db里" — db wins
        # on conflict between JSON and db.
        from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
        state_file, used_legacy = resolve_live_state_read(
            self._strategy_dir, self._broker_name,
        )
        if used_legacy:
            log.warning(
                "BROKER-ISOLATION: live_state.{broker}.json missing for "
                "broker=%s — reading legacy live_state.json (one-time "
                "migration fallback). Future writes go to broker-specific "
                "path. Verify this state belongs to broker '%s'.",
                self._broker_name, self._broker_name,
            )
        # state_store (S2 decomposition slice 1, 2026-06-13): JSON-first
        # load + RESTORE-FROM-DB fallback moved verbatim to
        # adapters/state_store.py.
        from adapters.state_store import load_live_state  # noqa: PLC0415

        state = load_live_state(state_file, config, self._strategy_dir)

        entry_dates     = state.get("entry_dates",     {})
        sell_streaks    = state.get("sell_streaks",    {})
        # Cross-DAY model-protection breach counter (pipeline #111). Mirrors
        # sell_streaks: persisted so the N-consecutive-day thesis-breach
        # debounce survives restarts. Missing key (older state) → {} → every
        # holding starts at 0 strikes, so the round-trip is back-compatible.
        protection_breaches = state.get("protection_breaches", {})
        last_sell_dates = state.get("last_sell_dates", {})
        # G8 (2026-05-04): per-ticker date when a path-rule exit
        # (trailing_stop / stop_loss / single_day_loss / max_hold / gap_down)
        # last fired. Persisted across runs so restart-after-stop honours
        # the cooldown. Read by PostStopCooldownFilterTask.
        last_stop_exit_dates = state.get("last_stop_exit_dates", {})
        position_hwm    = state.get("position_hwm",    {})
        # Thesis-degradation baselines (Approach A) — per-ticker
        # {rank_score, panel_score, kelly_target_pct} stamped at buy.
        entry_signals   = state.get("entry_signals",   {})
        # Z9 (2026-04-28): broker-side stop orders. Per-ticker
        # {order_id, stop_price, qty, stamped_at}. GTC at the broker
        # so they survive cron restarts. Local cache; broker.get_open_orders()
        # is the source of truth (we reconcile on every commit).
        stop_orders     = state.get("stop_orders",     {})
        # Runner-submitted SELL order ids, persisted across invocations so a
        # later reconciliation pass can tell its OWN exit fills apart from a
        # genuinely external/manual disposition. Maps
        # ``order_id -> {ticker, exit_type, qty, submitted_at}``. GC'd to a
        # 6-day window in commit() — a SEPARATE, shorter window than the
        # STATE-EXT-SELL fill lookback (widened to 45d by codex #428
        # review); this cache only affects the ATTRIBUTION log string
        # (z9_stop / runner_<exit_type> / external_or_manual), not the
        # wash-sale stamp date, so a >6-day-old runner-initiated sell just
        # falls back to "external_or_manual" in the log rather than
        # mis-stamping `last_sell_dates`.
        # 2026-06-03 HON incident: a runner single_day_loss sell filled, then
        # the next tick's reconciler mislabeled it source=external_or_manual
        # because only Z9 stop order_ids were tracked.
        recent_sell_orders = state.get("recent_sell_orders", {}) or {}
        hwm             = float(state.get("high_water_mark", 0.0))
        # Persisted RegimeState across live runs. Without this, each fresh
        # `daily_104.sh` invocation starts countdown=0 → CUSUM re-trips every
        # bar → `transition_window` stays True forever → buys perpetually
        # blocked. (Sim doesn't hit this because state lives in-process.)
        regime_persist  = state.get("regime_state", {}) or {}

        # ── Broker account ───────────────────────────────────────────────────
        account_value = broker.get_account_value()
        # 2026-05-04 audit Issue 36 fix: silent broker.get_cash() failure
        # used to default cash := account_value (= total NAV including
        # held positions). Downstream sizing then thought ALL of NAV was
        # liquid and could over-allocate. Fail-SAFE: fall back to ZERO
        # cash (the safest assumption — no fresh buys this bar) and log
        # loud so operator knows broker is partially down.
        try:
            cash = broker.get_cash()
        except Exception as _cash_exc:
            log.error(
                "runner: broker.get_cash() failed (%s: %s) — "
                "fail-SAFE setting cash=0 for this bar to prevent "
                "over-allocation. Pre-fix this defaulted to account_value "
                "(= total NAV) which silently allowed Kelly oversizing.",
                type(_cash_exc).__name__, _cash_exc,
            )
            cash = 0.0

        # Stale-HWM guard (see `resolve_hwm` docstring above). Snaps when
        # stored HWM is wildly above current equity, preserves normal
        # drawdowns otherwise.
        hwm, snapped = resolve_hwm(hwm, account_value)
        if snapped:
            log.warning("Stale HWM: snapped to current equity=%.2f "
                        "(stored HWM was stale; see adapters.runner.resolve_hwm)",
                        account_value)

        try:
            all_pos = broker.get_all_positions()
        except Exception:
            all_pos = []
        positions_cache = {p["symbol"]: p for p in all_pos}

        # Audit fix BROKER-PRECHECK (2026-04-26): pre-fetch broker's
        # currently open / pending orders ONCE per bar. Pre-fix, the
        # adapter called broker.get_open_orders() per-order at submit
        # time — N API calls per bar, AND the pipeline didn't know
        # which tickers were going to be rejected as duplicates BEFORE
        # sizing, so the cash budget assumed all orders fillable. The
        # e2e on 2026-04-26 03:20 showed 4 buys queued, 2 rejected as
        # duplicates → cash spent ≠ cash planned. Now: snapshot once,
        # inject into ctx, let upstream tasks (joint mode, selection,
        # rotation) skip these tickers BEFORE sizing.
        pending_broker_tickers: set[str] = set()
        try:
            pending_broker_tickers = set(broker.get_open_orders() or [])
            if pending_broker_tickers:
                log.info(
                    "BROKER-PRECHECK: %d pending order(s) at broker → "
                    "excluded from buy/rotate menus this bar: %s",
                    len(pending_broker_tickers),
                    sorted(pending_broker_tickers),
                )
        except Exception as exc:
            log.warning(
                "BROKER-PRECHECK: get_open_orders failed (%s) — "
                "duplicate-order guard falls back to per-order check at submit",
                exc,
            )

        # Audit fix ENTRY-DATE-FROM-FILLS (Round 4 deep audit, 2026-04-25):
        # Pre-fix, an inherited position with no entry_date in state got
        # stamped to TODAY (line ~191). Result: a position bought 60 days
        # ago was treated as fresh → min_hold_days=30 lockout started NOW
        # → user's old position couldn't be sold by the model for another
        # 30 days. Now: query broker fill history once per cycle, build a
        # ticker → earliest-BUY-fill-date map; use it as the seed for
        # missing entry_dates so hold tenure reflects actual cost-basis
        # tenure, not "first time the runner saw this position".
        first_fill_map: dict[str, datetime.date] = {}
        broker_fills: list[dict[str, Any]] = []
        try:
            fills = broker.get_filled_orders()
            broker_fills = list(fills or [])
            for f in fills or []:
                sym = f.get("symbol")
                if not sym or f.get("action") != "BUY":
                    continue
                fa = f.get("filled_at")
                if not fa:
                    continue
                try:
                    d = datetime.date.fromisoformat(str(fa)[:10])
                except (ValueError, TypeError):
                    continue
                # Earliest BUY for this symbol — only updated when no
                # SELL has happened in between (we don't currently track
                # the trip-lifecycle here; conservative: take the OLDEST
                # buy date so min_hold gives the position max benefit of
                # the doubt).
                if sym not in first_fill_map or d < first_fill_map[sym]:
                    first_fill_map[sym] = d
        except (AttributeError, NotImplementedError, Exception) as exc:
            log.info("ENTRY-DATE-FROM-FILLS: broker.get_filled_orders unavailable "
                     "(%s) — will fall back to sentinel for missing entry dates",
                     type(exc).__name__)
        live_tax_lots = reconstruct_live_tax_lots_from_fills(
            broker_fills,
            config=config,
        )

        # ── Holdings from live state + broker positions ─────────────────────
        from kernel.exits import HoldingState  # noqa: PLC0415

        from kernel.pipeline.task_benchmark_sleeve import (  # noqa: PLC0415
            benchmark_sleeve_ticker,
            decision_trace_tickers,
            is_benchmark_sleeve_enabled,
        )
        managed_symbols = list(config["watchlist"])
        sleeve_ticker = benchmark_sleeve_ticker(config)
        if is_benchmark_sleeve_enabled(config) and sleeve_ticker not in managed_symbols:
            managed_symbols.append(sleeve_ticker)
        held_set  = set(s for s in managed_symbols
                        if float(positions_cache.get(s, {}).get("qty", 0)) > 0)
        # Audit #59: log positions held outside the watchlist so the operator
        # knows they exist (the runner won't manage them — exits/buys only
        # apply to watchlist symbols — but silent invisibility is worse).
        non_wl_holds = [
            s for s, pos in positions_cache.items()
            if s not in held_set
            and s not in managed_symbols
            and float(pos.get("qty", 0)) > 0
        ]
        if non_wl_holds:
            log.warning("RunnerAdapter: %d position(s) held outside watchlist "
                        "(unmanaged): %s",
                        len(non_wl_holds), ", ".join(sorted(non_wl_holds)))
        # Audit fix UNMANAGED-NTFY (Round 4 deep audit, 2026-04-25): surface
        # non_wl_holds via ctx so live/runner.py::_notify_decision can include
        # an UNMANAGED line on the operator's phone — pre-fix, this was a
        # log-only warning that the user only saw if they opened the log file.
        # Real positions could sit in the broker for weeks with stop-loss /
        # trailing-stop never firing because the strategy doesn't know they
        # exist.
        self._non_wl_holds = list(sorted(non_wl_holds))
        holdings: dict[str, HoldingState] = {}
        for ticker in held_set:
            pos     = positions_cache.get(ticker, {})
            avg_cost = float(pos.get("avg_entry_price", 0.0))
            hwm_pos  = float(position_hwm.get(ticker, avg_cost))
            # entry_dates lookup with **persistent fallback**: if a position
            # is held but missing from entry_dates (e.g. inherited from
            # renquant_103 or manually added), stamp today and persist so
            # hold_days is measured from first-sighting, not from
            # today-minus-today (which made hold_days=0 forever → locked all
            # min_hold_days / rotation gates). Ideally seed from Alpaca's
            # fill timestamp on migration; today is the least-bad fallback.
            # Audit fix ENTRY-DATE-FROM-FILLS / ENTRY-DATE-BACKFILL
            # (Bug C extended, 2026-04-25): the broker's first BUY-fill
            # date is the AUTHORITATIVE entry date (cost-basis tenure
            # from Alpaca). Three cases:
            #   1. State has no entry_date → seed from broker fill if
            #      available, else use sentinel (31d ago).
            #   2. State has entry_date but broker shows OLDER first
            #      BUY → broker is correct (state was wrongly stamped
            #      "today" by a prior runner that didn't have ENTRY-DATE-
            #      FROM-FILLS). Override state with broker's earlier date.
            #      This unlocks min_hold_days / rotation tenure that the
            #      stale state was artificially extending.
            #   3. State has entry_date and it matches/predates broker →
            #      preserve (handles top-ups + cost-basis-fifo cases).
            broker_first = first_fill_map.get(ticker)
            if ticker not in entry_dates:
                if broker_first is not None:
                    entry_dates[ticker] = broker_first.isoformat()
                    log.info("ENTRY-DATE-SEED %s ← %s (broker fill history)",
                             ticker, broker_first.isoformat())
                else:
                    sentinel = today - datetime.timedelta(days=31)
                    entry_dates[ticker] = sentinel.isoformat()
                    log.warning("ENTRY-DATE-SEED %s ← %s (sentinel — broker had no "
                                "fill history; manual fix recommended)",
                                ticker, sentinel.isoformat())
            else:
                # Backfill: broker authority overrides stale state when older.
                if broker_first is not None:
                    try:
                        cur_entry = datetime.date.fromisoformat(entry_dates[ticker])
                    except (ValueError, TypeError):
                        cur_entry = today
                    if broker_first < cur_entry:
                        log.info("ENTRY-DATE-BACKFILL %s: state=%s → broker=%s "
                                 "(broker fill is older — stale state corrected)",
                                 ticker, entry_dates[ticker], broker_first.isoformat())
                        entry_dates[ticker] = broker_first.isoformat()
            entry_str = entry_dates[ticker]
            try:
                entry_dt = datetime.date.fromisoformat(entry_str)
            except ValueError:
                entry_dt = today
                entry_dates[ticker] = today.isoformat()
            # Audit fix QTY-NaN-HYDRATE (Round 2 deep audit, 2026-04-25):
            # broker NaN qty during a snapshot race would make
            # HoldingState.shares = NaN, then propagate into Kelly
            # current_pct calc. Downstream TopUp/Trim now have isfinite
            # guards (TU/TR-NaN) but cleaner to sanitize at hydration.
            import math as _math
            _qty_raw = float(pos.get("qty", 0))
            qty_held = _qty_raw if _math.isfinite(_qty_raw) else 0.0
            # Thesis-degradation baselines (Approach A) — hydrate from
            # persisted entry_signals. Missing keys → None, which the
            # rotation criterion treats as "no baseline, fall back to
            # legacy rule".
            es = entry_signals.get(ticker, {}) if isinstance(entry_signals.get(ticker, {}), dict) else {}
            holdings[ticker] = HoldingState(
                entry_price    = avg_cost,
                entry_date     = entry_dt,
                high_watermark = hwm_pos,
                sell_streak    = int(sell_streaks.get(ticker, 0)),
                shares         = qty_held,   # broker qty for Kelly top-up sizing
                entry_rank_score       = es.get("rank_score"),
                entry_panel_score      = es.get("panel_score"),
                entry_kelly_target_pct = es.get("kelly_target_pct"),
                entry_regime           = es.get("regime"),
            )
            # Cross-DAY model-protection breach counter. Set post-construction
            # (not a constructor kwarg) so a pin whose HoldingState predates the
            # field cannot TypeError in this hot path; pipeline reads it
            # getattr-safe (default 0).
            holdings[ticker].protection_breaches = int(protection_breaches.get(ticker, 0) or 0)
            lots = live_tax_lots.get(ticker)
            if lots:
                lot_qty = sum(float(getattr(L, "shares", 0.0) or 0.0) for L in lots)
                if abs(lot_qty - qty_held) <= max(0.01, abs(qty_held) * 1e-4):
                    holdings[ticker].lots = lots
                    holdings[ticker].entry_price = (
                        holdings[ticker].weighted_avg_entry_price()
                    )
                else:
                    log.warning(
                        "LIVE-TAX-LOTS: %s reconstructed lot qty %.4f != broker "
                        "qty %.4f; using broker avg_entry_price fallback",
                        ticker, lot_qty, qty_held,
                    )
            holdings[ticker].model_type = model_type_from_artifact(
                self._models.get(ticker)
            )
            sector = config.get("sector_map", {}).get(ticker)
            if isinstance(sector, str) and sector:
                holdings[ticker].sector = sector

        # ── Current prices from broker positions ────────────────────────────
        # 2026-05-09 audit fix (RU-PRICE-1): pre-fix `if qty > 0 and mkt > 0`
        # passed for micro-qty (e.g. 1e-7 fractional shares from a botched
        # broker fill) → `mkt / qty` produced an inflated price (e.g.
        # market_value=$100, qty=1e-6 → price=$100M/share). Guard with
        # isfinite + a 1-share floor so we treat sub-share dust as "no
        # trustworthy price" and fall back to OHLCV close below.
        #
        # Full daily runs must not mix real-time broker marks for held symbols
        # with daily OHLCV closes for candidates. Keep broker marks only for
        # sell-only/intraday risk checks or as an OHLCV-missing fallback.
        from adapters.runner_prices import compute_broker_mark_prices  # noqa: PLC0415
        prices, broker_mark_prices = compute_broker_mark_prices(
            positions_cache, sell_only=self._sell_only,
            use_intraday_prices=self._use_intraday_prices)

        # ── OHLCV from parquet cache ─────────────────────────────────────────
        import math as _math_p  # noqa: PLC0415  (price/close finiteness guards)
        from kernel.data import fetch_ohlcv  # noqa: PLC0415

        watchlist   = config["watchlist"]
        benchmark   = config.get("benchmark", "SPY")
        sector_etfs = set(config.get("sector_etf_map", {}).values())
        extra_symbols = []
        if is_benchmark_sleeve_enabled(config) and sleeve_ticker:
            extra_symbols.append(sleeve_ticker)
        # Parking-sleeve legs (st104 #39 follow-up): fetch daily OHLCV for
        # sleeve.spy_symbol / sleeve.sgov_symbol only when sleeve.enabled —
        # same conditional-coverage pattern as the benchmark sleeve above.
        from adapters.sleeve_prices import parking_sleeve_price_tickers  # noqa: PLC0415
        extra_symbols.extend(parking_sleeve_price_tickers(config))
        all_symbols = list(dict.fromkeys(
            watchlist
            + [benchmark]
            + sorted(sector_etfs)
            + sorted(held_set)
            + extra_symbols
        ))

        ohlcv: dict[str, Any] = {}
        for sym in all_symbols:
            try:
                df = fetch_ohlcv(sym)
                if not df.empty:
                    ohlcv[sym] = df
                    # Fill prices from OHLCV last close if broker didn't supply.
                    # 2026-05-09 audit fix (RU-PRICE-2): isfinite guard on
                    # the close value. Pre-fix, a NaN close in the last bar
                    # (data-feed glitch on suspended/halted ticker) silently
                    # propagated into ctx.prices → Kelly/HWM/QP all received
                    # NaN → cascade of silent failures.
                    if (not self._sell_only and not self._use_intraday_prices) or sym not in prices:
                        close_val = float(df["close"].iloc[-1])
                        if _math_p.isfinite(close_val) and close_val > 0:
                            prices[sym] = close_val
                        else:
                            log.warning(
                                "OHLCV close for %s is non-finite (%s) — "
                                "skipping price entry; downstream tasks will "
                                "see ticker as 'no price' (fail-safe).",
                                sym, close_val,
                            )
            except Exception as exc:
                log.warning("OHLCV fetch failed for %s: %s", sym, exc)
                if (self._sell_only or self._use_intraday_prices) and sym in held_set:
                    fallback_df = None
                    try:
                        from kernel.data import LocalStore  # noqa: PLC0415
                        fallback_df = LocalStore().load(sym)
                    except Exception as cache_exc:
                        log.warning(
                            "OHLCV cache fallback read failed for held %s: %s",
                            sym, cache_exc,
                        )
                    mark_px = broker_mark_prices.get(sym)
                    risk_df = _held_mark_ohlcv_frame(
                        sym,
                        today,
                        mark_px if mark_px is not None else prices.get(sym, 0.0),
                        fallback_df,
                    )
                    if risk_df is not None:
                        ohlcv[sym] = risk_df
                        prices[sym] = float(risk_df["close"].iloc[-1])
                        log.warning(
                            "SELL-ONLY-RISK-OHLCV-FALLBACK %s: using broker "
                            "mark %.4f as synthetic risk-only bar for %s; "
                            "not writing official OHLCV cache.",
                            sym, prices[sym], today.isoformat(),
                        )

        if not (self._sell_only or self._use_intraday_prices):
            for sym, px in broker_mark_prices.items():
                prices.setdefault(sym, px)

        # ── Intraday price overlay (for intraday sell-only checks) ──────────
        if self._use_intraday_prices:
            try:
                from kernel.data import fetch_intraday_bars  # noqa: PLC0415
                ibars = fetch_intraday_bars(
                    list(all_symbols),
                    timeframe="5Min",
                    start=datetime.datetime.combine(
                        today, datetime.datetime.min.time(),
                    ),
                )
                overlaid = 0
                for sym, idf in ibars.items():
                    if idf is None or idf.empty:
                        continue
                    latest_close = float(idf["close"].iloc[-1])
                    # 2026-05-09 audit fix (RU-INTRADAY-NaN): same guard as
                    # OHLCV daily path — non-finite intraday close (5Min bar
                    # gap on halted ticker) silently corrupted ctx.prices.
                    if not _math_p.isfinite(latest_close) or latest_close <= 0:
                        log.warning("Intraday close for %s non-finite (%s) — skipping",
                                     sym, latest_close)
                        continue
                    prices[sym] = latest_close
                    # Overwrite today's daily bar's close so kernel.exits sees the intraday level.
                    # Audit #58: copy the frame before mutating — fetch_ohlcv may
                    # return a cached reference that other downstream calls (sim,
                    # training, panel features) would see leak via the in-place
                    # write. The sliced copy in `ohlcv` is what the pipeline reads.
                    if sym in ohlcv and not ohlcv[sym].empty:
                        df = ohlcv[sym].copy()
                        last_day = df.index.max()
                        if last_day.date() == today:
                            df.at[last_day, "close"] = latest_close
                            ohlcv[sym] = df
                    overlaid += 1
                log.info("Intraday overlay: %d/%d symbols had fresh minute bars",
                         overlaid, len(all_symbols))
            except Exception as exc:
                log.warning("Intraday overlay failed — falling back to daily closes: %s", exc)

        spy_df = ohlcv.get(benchmark)
        spy_returns: list[float] = []
        if spy_df is not None:
            spy_close   = spy_df["close"].astype(float)
            spy_returns = list(spy_close.pct_change().dropna().values[-100:])

        # ── Load artifacts ───────────────────────────────────────────────────
        from adapters.runner_artifacts import load_context_artifacts  # noqa: PLC0415
        gmm, corr, earnings = load_context_artifacts(self._strategy_dir, config)
        # Convert last_sell_dates strings to date objects for kernel.selection guards
        last_sells_d: dict[str, datetime.date | None] = {}
        for sym, d_str in last_sell_dates.items():
            try:
                last_sells_d[sym] = datetime.date.fromisoformat(d_str)
            except (ValueError, TypeError):
                last_sells_d[sym] = None

        # 2026-05-09 cost-aware wash-sale: compute realized $ P/L per
        # ticker for the most-recent full liquidation in the last 30d.
        # Used by WashSaleFilterTask to skip block on GAIN sales (§1091
        # rule does not apply) and to compute NPV cost on loss sales.
        last_sells_pl: dict[str, float | None] = {}
        try:
            from kernel.realized_pnl import compute_recent_realized_pnl  # noqa: PLC0415
            last_sells_pl = compute_recent_realized_pnl(
                self._broker, days=int(config.get("wash_sale_days", 30)) + 5,
            )
            n_gains = sum(1 for v in last_sells_pl.values() if v is not None and v >= 0)
            n_losses = sum(1 for v in last_sells_pl.values() if v is not None and v < 0)
            log.info("realized_pnl: %d gains + %d losses in last %dd  (gains skip wash-sale block)",
                     n_gains, n_losses, int(config.get("wash_sale_days", 30)) + 5)
        except Exception as exc:
            log.warning("realized_pnl compute failed: %s — wash-sale falls back to binary", exc)
            last_sells_pl = {}
        # G8: same coercion for stop-exit dates
        last_stops_d: dict[str, datetime.date | None] = {}
        for sym, d_str in (last_stop_exit_dates or {}).items():
            try:
                last_stops_d[sym] = datetime.date.fromisoformat(d_str)
            except (ValueError, TypeError):
                last_stops_d[sym] = None

        # ── Persisted live state on context for commit() ─────────────────────
        self._state          = state
        self._entry_dates    = entry_dates
        self._entry_signals  = entry_signals   # Approach A — persisted per-ticker
        self._sell_streaks   = sell_streaks
        self._protection_breaches = protection_breaches
        self._last_sell_dates_str = last_sell_dates
        self._last_stop_exit_dates_str = dict(last_stop_exit_dates or {})
        self._position_hwm   = position_hwm
        self._stop_orders    = stop_orders     # Z9: per-ticker stop_order metadata
        self._recent_sell_orders = recent_sell_orders  # runner-submitted SELL order_ids
        self._positions_cache = positions_cache
        self._account_value  = account_value

        ctx = InferenceContext(
            config            = config,
            today             = today,
            run_timestamp     = datetime.datetime.now().astimezone(),
            broker_name       = self._broker_name,
            ohlcv             = ohlcv,
            spy_returns       = spy_returns,
            models            = self._models,
            gmm               = gmm,
            corr_matrix       = corr,
            earnings_calendar = earnings,
            holdings          = holdings,
            last_sell_dates   = last_sells_d,
            last_sell_pls     = last_sells_pl,
            last_stop_exit_dates = last_stops_d,
            portfolio_value   = account_value,
            cash              = cash,
            prices            = prices,
            hwm               = hwm,
            skip_buys         = persisted_skip_buys(state),
            regime_state      = RegimeState(
                regime        = regime_persist.get("regime",        "BULL_CALM"),
                confidence    = float(regime_persist.get("confidence",     0.5)),
                in_transition = bool(regime_persist.get("in_transition", False)),
                countdown     = int(regime_persist.get("countdown",          0)),
                cusum_pos     = float(regime_persist.get("cusum_pos",      0.0)),
                cusum_neg     = float(regime_persist.get("cusum_neg",      0.0)),
                # CUSUM-v2 Design C — restore wall-clock cooldown start.
                # Parse ISO string; None / missing → no cooldown active.
                cooldown_start = _parse_iso_dt(regime_persist.get("cooldown_start")),
            ),
            regime_counts     = {r: 0 for r in REGIMES},
            monitor_state     = dict(state.get("monitor_state", {}) or {}),
        )
        ctx.run_id = f"{today.isoformat()}-live-{uuid.uuid4().hex[:8]}"
        ctx.supports_short_open = False
        ctx._run_type = "live"  # noqa: SLF001

        # Bug 11 fix (2026-04-24): Rotation V4 (thesis_symmetric scoring
        # mode) needs ctx._db to look up candidate scores on each held's
        # entry date via lookup_candidate_scores_on_date. Previously this
        # was wired only on SimAdapter; without it, V4 silently no-ops on
        # the live runner path. RunnerAdapter writes to runs.db (live);
        # rotation V4 reads from there for entry-day score lookup.
        if self._db is not None:
            ctx._db = self._db   # noqa: SLF001

        # UNMANAGED-NTFY: pass through to ntfy decision-summary path.
        ctx.non_wl_holds = list(self._non_wl_holds)

        # BROKER-PRECHECK (2026-04-26): expose pending broker orders so
        # JointActionTask + SelectionJob + RotationJob can pre-filter
        # candidates BEFORE sizing.
        ctx.pending_broker_tickers = pending_broker_tickers

        # Rotation V1 persistence gate — live runner has no per-bar
        # state file pinned to rotation_proposals (yet); seed with empty
        # so the gate fails-closed when persistence_bars > 0. (When the
        # user enables persistence in production we'll wire it through
        # live_state.json — same pattern as monitor_state.)
        ctx.prior_rotation_proposals = list(
            state.get("rotation_proposals", []) or []
        )

        # Run-local feature cache (2026-05-25): live/shadow should use the
        # same causal feature surface as sim. The OHLCV has already passed the
        # live freshness guard; caching it for this one run avoids rebuilding
        # indicators separately in sell and candidate jobs.
        if config.get("live", {}).get("feature_cache_enabled", True):
            ctx.feature_cache = build_runtime_feature_cache(
                config=config,
                ohlcv=ohlcv,
            )
            log.info("Feature cache attached to live context: %d tickers",
                     len(ctx.feature_cache))

        # ── Panel scoring prep (optional) ────────────────────────────────────
        panel_cfg = config.get("ranking", {}).get("panel_scoring", {})
        if panel_cfg.get("enabled", False) and not self._sell_only:
            try:
                bundle = prepare_panel_runtime_frames(
                    config=config,
                    ohlcv=ohlcv,
                )
                attach_panel_runtime_frames(ctx, bundle)
                n_ff, n_fac, macro_desc, n_emb = describe_panel_frame_bundle(bundle)
                log.info("Panel frames prepared: feat=%d  factor=%d  macro=%s  emb=%d",
                         n_ff, n_fac, macro_desc, n_emb)
            except Exception as exc:
                msg = (
                    "Panel frame prep failed while panel_scoring.enabled=true; "
                    "aborting live inference instead of silently trading without "
                    f"panel scores: {exc}"
                )
                log.error(msg)
                raise RuntimeError(msg) from exc

        # ── P5 (2026-05-11) attach meta-label hooks ──────────────────────
        # snapshot_logger: None unless meta_label_training.enabled
        #   (training-data capture mode — typically OFF in prod)
        # _meta_label_predictor: None unless ranking.meta_label.enabled +
        #   artifact loaded. The MetaLabelVetoTask in pp_inference.py
        #   reads this to drop false-positive path-rule exits at live time.
        ctx.snapshot_logger = self._meta_label_logger
        ctx._meta_label_predictor = self._meta_label_predictor  # noqa: SLF001

        # S-FRAC stage 3 (sprint D2): expose the software-stop registry to
        # the pipeline — SellOnlyPipeline's SoftwareStopExitTask evaluates
        # it each intraday pass. None when execution.software_stops is
        # disabled (default) ⇒ the task no-ops (flag-off byte-inert).
        ctx.software_stops = self._software_stops

        return ctx

    # ── Z9: broker-side stop helpers ────────────────────────────────────────
    # Invariants:
    #   • stops live broker-side (GTC), not in our polling loop
    #   • single stop per ticker; stamped at BUY, replaced on TOPUP, cancelled
    #     on full SELL or external disposition (Z2 STATE-EXT-SELL)
    #   • new_stop_price ≤ existing_stop_price (never loosen on TOPUP)
    #   • disabled by default; enable via live.broker_side_stops.enabled=true
    #   • broker must support — silently skipped if broker.supports_broker_side_stops()=false

    def _z9_enabled(self, ctx) -> bool:  # noqa: ANN001
        """Delegate — moved to adapters/z9_stops.py (S2 slice 3, 2026-06-13)."""
        from adapters.z9_stops import z9_enabled  # noqa: PLC0415

        return z9_enabled(self._broker, ctx)

    @staticmethod
    def _z9_stop_pct(ctx) -> float:  # noqa: ANN001
        """Delegate — moved to adapters/z9_stops.py (S2 slice 3, 2026-06-13)."""
        from adapters.z9_stops import z9_stop_pct  # noqa: PLC0415

        return z9_stop_pct(ctx)

    def _override_no_trade_streak_from_broker(self, ctx) -> None:  # noqa: ANN001
        """Delegate — body moved verbatim to adapters/broker_sync.py
        (S2 decomposition slice 2, 2026-06-13)."""
        from adapters.broker_sync import override_no_trade_streak_from_broker  # noqa: PLC0415

        override_no_trade_streak_from_broker(self._broker, self._state, ctx)

    def _z9_place_or_replace_stop(
        self, ticker: str, qty: float, reference_price: float, today_str: str,
    ) -> None:
        """Delegate — moved to adapters/z9_stops.py (S2 slice 3, 2026-06-13).
        ``_last_ctx_stop_pct`` is the per-bar stop distance the caller set
        from _z9_stop_pct; passed through as ctx_pct.

        S-FRAC stage 0 (§2.2.2): the stage-3 software-stop seam is passed
        through so z9_stops routes protection per held quantity — the
        capability is re-evaluated at every placement against the CURRENT
        qty (restart-safe: nothing is cached across sessions)."""
        from adapters.z9_stops import place_or_replace_stop  # noqa: PLC0415

        place_or_replace_stop(
            self._broker, self._stop_orders, ticker, qty, reference_price,
            today_str, ctx_pct=getattr(self, "_last_ctx_stop_pct", 0.06),
            software_stops=getattr(self, "_software_stops", None),
        )

    def _z9_cancel_stop(self, ticker: str, reason: str = "") -> None:
        """Delegate — moved to adapters/z9_stops.py (S2 slice 3, 2026-06-13).

        S-FRAC stage 3: the software-stop registry is passed through so a
        full liquidation / GC also disarms the ticker's software stop."""
        from adapters.z9_stops import cancel_stop  # noqa: PLC0415

        cancel_stop(self._broker, self._stop_orders, ticker, reason,
                    software_stops=getattr(self, "_software_stops", None))

    # ── STATE-EXT-SELL fill attribution (issue #71 / audit #5) ────────────────

    # ── STATE-EXT-SELL fill attribution — EXTRACTED to runner_ext_sell.py ─
    # (eng plan S2 item 5 decomposition slice 8, 2026-06-13.)
    @staticmethod
    def _normalize_fill_record(f: dict) -> dict:
        from adapters.runner_ext_sell import normalize_fill_record  # noqa: PLC0415
        return normalize_fill_record(f)

    def _lookup_ext_sell_fills(self, ctx, disappeared: list[str]) -> dict[str, dict]:  # noqa: ANN001
        from adapters.runner_ext_sell import lookup_ext_sell_fills  # noqa: PLC0415
        return lookup_ext_sell_fills(self._broker, ctx, disappeared)

    @staticmethod
    def _bar_date(ctx) -> "datetime.date":  # noqa: ANN001
        from adapters.runner_ext_sell import bar_date  # noqa: PLC0415
        return bar_date(ctx)

    @staticmethod
    def _ext_sell_fill_date(fill: dict | None) -> "datetime.date | None":
        """Delegate — see adapters/runner_ext_sell.ext_sell_fill_date.

        2026-07-01 fix: the broker fill this lookup already returned is
        AUTHORITATIVE for the wash-sale stamp date, mirroring the
        ENTRY-DATE-FROM-FILLS principle used for entry_dates. Codex #428
        review: only a CONFIRMED SELL fill (side == "sell") qualifies,
        and the date is extracted via a TZ-aware America/New_York
        conversion, not first-10-chars string slicing."""
        from adapters.runner_ext_sell import ext_sell_fill_date  # noqa: PLC0415
        return ext_sell_fill_date(fill)

    @staticmethod
    def _ext_sell_stamp_decision(
        fill_date: "datetime.date | None", prior_stamp: str | None, today_str: str,
    ) -> "tuple[str, str]":
        """Delegate — see adapters/runner_ext_sell.ext_sell_stamp_decision.

        Codex #428 review ("ALSO reconsider"): choose between stamping the
        ACTUAL confirmed fill date, PRESERVING an existing older stamp
        (no confirmed fill within the lookback, but a value already on
        file — never overwrite known evidence with "today"), or the
        NO-FILL-FOUND fallback (today_str, only when there is truly no
        information at all)."""
        from adapters.runner_ext_sell import ext_sell_stamp_decision  # noqa: PLC0415
        return ext_sell_stamp_decision(fill_date, prior_stamp, today_str)

    def _gc_recent_sell_orders(self, ctx) -> dict:
        """Delegate — body moved verbatim to adapters/broker_sync.py
        (S2 decomposition slice 2, 2026-06-13)."""
        from adapters.broker_sync import gc_recent_sell_orders  # noqa: PLC0415

        kept = gc_recent_sell_orders(self._recent_sell_orders, self._bar_date(ctx))
        self._recent_sell_orders = kept
        return kept

    def _attribute_ext_sell(self, ticker: str, fills: dict[str, dict]) -> str:
        """Delegate — moved to adapters/runner_ext_sell.py (S2 slice 8)."""
        from adapters.runner_ext_sell import attribute_ext_sell  # noqa: PLC0415
        return attribute_ext_sell(self._stop_orders, self._recent_sell_orders,
                                  ticker, fills)

    # ── commit ─────────────────────────────────────────────────────────────────

    def commit(self, ctx) -> None:  # noqa: ANN001
        """Apply pipeline outputs: execute broker orders, update live_state.json."""
        broker        = self._broker
        today_str     = ctx.today.isoformat()
        pos_cache     = self._positions_cache

        # ── S-FRAC stage 0 commit quantity contract ─────────────────────────
        # Design: renquant-orchestrator doc/design/2026-07-02-s-frac-
        # fractional-v2.md §2.2. Single authority: adapters/commit_contract.
        #   * commit_path_fingerprint → stamped on ctx here, recorded in the
        #     run bundle (kernel/artifact_contract.build_run_bundle) — the
        #     §2.3 active-path liveness proof ("the live runner exercised
        #     the fractional-capable commit path" is a per-run recorded
        #     fact, not an assumption).
        #   * fractional_capability_gate → machine-verifiable preflight for
        #     execution.fractional_shares (the strategy#36 prose-gate
        #     blocker, closed). Default-OFF flag ⇒ gate trivially ok and
        #     the whole contract is inert (whole-share behavior is
        #     regression-pinned byte-identical).
        from adapters.commit_contract import (  # noqa: PLC0415
            commit_path_fingerprint,
            fmt_qty,
            fractional_capability_gate,
            fractional_entry_fail_closed_reason,
            normalize_fill_qty,
        )

        ctx.commit_path_fingerprint = commit_path_fingerprint()
        frac_gate = fractional_capability_gate(
            self._config, broker, getattr(self, "_software_stops", None),
        )
        if frac_gate["enabled"] and not frac_gate["ok"]:
            log.error(
                "S-FRAC capability gate FAILED — execution.fractional_shares "
                "is enabled but required capabilities are missing: %s. ALL "
                "BUY emission fail-closes this bar (exits are never blocked). "
                "The flag landed ahead of its dependencies; disable it or "
                "land the missing stage(s).",
                ",".join(frac_gate["missing"]),
            )

        # ── Apply exits ──────────────────────────────────────────────────────
        # Honours optional sig.quantity for partial sells (Kelly trim path).
        # When quantity is None or ≥ current qty → full liquidation (old
        # behaviour). When quantity is a positive float < current qty →
        # partial sell, position stays open with reduced shares; we keep
        # entry_dates / position_hwm / sell_streaks intact.
        #
        # Audit fix EXITS-FAIL (Round 2 deep audit, 2026-04-25): pre-fix,
        # broker.place_order failures inside the SELL branch logged with
        # log.error and `continue`d, but ctx.exits was the only list the
        # ntfy code read for "EXIT ticker (reason)" messages. Result: a
        # failed sell appeared on the operator's phone as if it had
        # succeeded — they thought the position closed, but it was still
        # held at the broker. Now: split ctx.exits → ctx.exits_placed
        # (broker-confirmed) and ctx.exits_failed (broker error). The
        # ntfy path will need to read exits_placed (analogous to the
        # orders_placed/orders_skipped split that already exists for
        # buys); falls back to ctx.exits when those fields aren't set.
        if not hasattr(ctx, "exits_placed"):
            ctx.exits_placed = []
        if not hasattr(ctx, "exits_pending"):
            ctx.exits_pending = []
        if not hasattr(ctx, "exits_failed"):
            ctx.exits_failed = []
        full_exit_tickers: set[str] = set()
        # Audit fix QTY-NaN (Round 2 deep audit, 2026-04-25): same NaN-
        # slip pattern as SE-1/TR-NaN/ROT-NaN-PRICE. Pre-fix, a broker
        # response with NaN qty (rare but possible during account
        # snapshot races) slipped past `qty <= 0` (NaN<=0 False), then
        # `sell_qty = abs(NaN) = NaN` was passed to broker.place_order
        # which crashed inside Alpaca's int(quantity). Now: skip with
        # a clear log on non-finite qty.
        import math as _math

        def _held_qty(t: str) -> float:
            pos = pos_cache.get(t, {})
            try:
                qty_f = float(pos.get("qty", 0))
            except (TypeError, ValueError):
                return 0.0
            return qty_f if _math.isfinite(qty_f) and qty_f > 0 else 0.0

        for ticker, sig in dedupe_exit_signals(ctx.exits, held_qty_for=_held_qty):
            pos = pos_cache.get(ticker, {})
            qty = float(pos.get("qty", 0))
            if not _math.isfinite(qty) or qty <= 0:
                if not _math.isfinite(qty):
                    log.warning("EXIT %s: broker qty=%s non-finite, skipping", ticker, qty)
                continue

            # Audit fix PLTR-AVAILABLE-QTY (2026-04-26 round-3, e2e finding):
            # use qty_available (= qty - held_for_orders) so we don't ask
            # the broker to sell shares locked in pending orders. Alpaca
            # rejects with available=0 in that case. Pre-fix, e2e round 3
            # saw PLTR sell fail this way. Falls back to qty when broker
            # doesn't expose qty_available.
            # Audit fix PLTR-AVAILABLE-QTY-V2 (2026-04-26 round-4):
            # `pos.get("qty_available", qty) or qty` collapses 0-available
            # back to qty (0 is falsy). Pre-fix bug — saw PLTR sell still
            # fail in e2e round 4 with available=0 because we fell back
            # to qty=5. Fix: use explicit None check.
            _qa_raw = pos.get("qty_available", None)
            qty_avail = qty if _qa_raw is None else float(_qa_raw)
            if not _math.isfinite(qty_avail) or qty_avail <= 0:
                # Audit fix LOG-FORMAT (2026-04-26 round-5): arg order
                # was swapped — log output showed "qty=PLTR" instead of
                # "qty=5". Now: ticker first, then qty_avail, then qty.
                log.warning(
                    "EXIT %s: qty_available=%s, qty=%s (likely held in "
                    "pending orders) — skipping. Cancel pending order first.",
                    ticker, qty_avail, qty,
                )
                if not hasattr(ctx, "exits_failed"):
                    ctx.exits_failed = []
                ctx.exits_failed.append({
                    "ticker": ticker, "qty": qty,
                    "exit_type": getattr(sig, "exit_type", ""),
                    "reason": getattr(sig, "reason", ""),
                    "error": f"qty_available={qty_avail}, all locked in pending orders",
                })
                continue

            req_qty = getattr(sig, "quantity", None)
            if (
                not is_full_liquidate_signal(sig, qty)
                and req_qty is not None
                and _math.isfinite(req_qty)
                and 0 < req_qty < qty_avail
            ):
                sell_qty   = float(req_qty)
                is_partial = True
            else:
                # Cap at qty_available (not qty) to avoid broker rejection.
                sell_qty   = abs(qty_avail)
                is_partial = (qty_avail < qty)

            try:
                result = broker.place_order(ticker, "SELL", sell_qty)
            except Exception as exc:
                log.error("SELL failed for %s: %s", ticker, exc)
                ctx.exits_failed.append({
                    "ticker":     ticker,
                    "exit_type":  getattr(sig, "exit_type", ""),
                    "reason":     getattr(sig, "reason", ""),
                    "qty":        sell_qty,
                    "is_partial": is_partial,
                    "error":      str(exc),
                })
                continue
            execution = broker_order_execution(
                result, requested_qty=sell_qty,
                fallback_price=ctx.prices.get(ticker, 0.0),
            )
            if execution["rejected"]:
                log.error(
                    "SELL rejected for %s: status=%s order_id=%s",
                    ticker, execution["status"], execution.get("order_id"),
                )
                ctx.exits_failed.append({
                    "ticker":     ticker,
                    "exit_type":  getattr(sig, "exit_type", ""),
                    "reason":     getattr(sig, "reason", ""),
                    "qty":        sell_qty,
                    "is_partial": is_partial,
                    "order_id":   execution.get("order_id"),
                    "status":     execution["status"],
                    "error":      f"broker_status:{execution['status']}",
                })
                continue
            # Record the runner-submitted SELL order_id (pending OR filled) so a
            # later reconciliation pass attributes the fill to the runner, not
            # external_or_manual (2026-06-03 HON single_day_loss incident).
            _submitted_oid = execution.get("order_id")
            if _submitted_oid:
                # Store the BAR DATE (date granularity) as a plain
                # YYYY-MM-DD string. ctx.today may be a datetime (datetime
                # is a subclass of date), so normalize to .date() first —
                # otherwise the stamp carries a time component that the
                # date-granularity GC can't parse on py3.10. Date
                # granularity is intentional: the GC's 6-day window vs the
                # 5-day fill lookback already carries 1 day of slack to
                # absorb any session that crosses midnight (codex #199
                # review).
                _now_iso = self._bar_date(ctx).isoformat()
                self._recent_sell_orders[str(_submitted_oid)] = {
                    "ticker":       ticker,
                    "exit_type":    getattr(sig, "exit_type", "") or "",
                    "qty":          float(sell_qty),
                    "submitted_at": _now_iso,
                }
            if execution["pending"]:
                pending = {
                    "ticker":     ticker,
                    "exit_type":  getattr(sig, "exit_type", ""),
                    "reason":     getattr(sig, "reason", ""),
                    "qty":        sell_qty,
                    "is_partial": is_partial,
                    "order_id":   execution.get("order_id"),
                    "status":     execution["status"],
                }
                ctx.exits_pending.append(pending)
                # fmt_qty (S-FRAC stage 0): byte-identical to the old %.0f
                # for whole shares; a fractional qty renders verbatim
                # instead of being display-rounded.
                log.warning(
                    "SELL pending at broker for %s: %s shares status=%s "
                    "order_id=%s; live_state/DB not mutated until fill.",
                    ticker, fmt_qty(sell_qty), execution["status"],
                    execution.get("order_id"),
                )
                continue

            sell_qty = float(execution["filled_qty"] or sell_qty)
            price = float(execution["filled_avg_price"] or ctx.prices.get(ticker, 0.0))
            is_partial = bool(execution["partial"] or sell_qty < qty - 1e-9)

            # Use HoldingState.entry_price as the running avg-cost fallback.
            hs = (ctx.holdings or {}).get(ticker)
            lot_accounted = apply_live_sell_lot_accounting(
                sig,
                hs,
                shares=float(sell_qty),
                price=float(price),
                today=ctx.today,
                config=self._config,
            )
            if not lot_accounted:
                # 2026-05-18: stamp P/L on the ExitSignal so live/runner.py's
                # _notify_decision can render explicit $ realized P/L in ntfy.
                # Fallback cost basis is broker avg_entry_price when fill
                # history cannot reconstruct tax lots.
                cost_basis = float(pos_cache.get(ticker, {}).get(
                    "avg_entry_price", 0.0
                ))
                if hs is not None and cost_basis <= 0:
                    cost_basis = float(getattr(hs, "entry_price", 0.0) or 0.0)
                if cost_basis > 0 and price > 0:
                    gain_per_share = price - cost_basis
                    gain_dollar = gain_per_share * sell_qty
                    gain_pct = (price / cost_basis - 1.0) * 100.0
                    try:
                        sig.realized_pnl_dollar = float(gain_dollar)
                        sig.realized_pnl_pct = float(gain_pct)
                        sig.cost_basis = float(cost_basis)
                        sig.sell_price = float(price)
                        sig.shares_sold = float(sell_qty)
                    except Exception:
                        pass
                else:
                    try:
                        sig.sell_price = float(price)
                        sig.shares_sold = float(sell_qty)
                    except Exception:
                        pass
            else:
                try:
                    sig.sell_price = float(price)
                    sig.shares_sold = float(sell_qty)
                except Exception:
                    pass

            ctx.exits_placed.append((ticker, sig))

            tag   = "TRIM" if is_partial else "SELL"
            pl_str = ""
            if getattr(sig, "realized_pnl_dollar", None) is not None:
                pl_str = (f"  P/L=${sig.realized_pnl_dollar:+.2f} "
                          f"({sig.realized_pnl_pct:+.2f}%)")
            log.info("%s  %s  [%s]  %s shares @ %.2f%s  %s",
                     tag, ticker, sig.exit_type, fmt_qty(sell_qty), price,
                     pl_str, sig.reason)

            # Wash-sale clock: stamp ONLY on full liquidation. Partial
            # trims (Kelly rebalance) intentionally don't block subsequent
            # top-ups — that would prevent the position from ever growing
            # back toward the Kelly target after an over-weight trim.
            if not is_partial:
                full_exit_tickers.add(ticker)
                self._last_sell_dates_str[ticker] = today_str
                self._entry_dates.pop(ticker, None)
                self._entry_signals.pop(ticker, None)   # Approach A cleanup
                self._sell_streaks.pop(ticker, None)
                self._protection_breaches.pop(ticker, None)
                self._position_hwm.pop(ticker, None)
                # Z9: cancel broker-side stop on full liquidation.
                if self._z9_enabled(ctx):
                    self._z9_cancel_stop(ticker, reason="full liquidation")
            # G8 (2026-05-04): stamp post-stop blackout on path-rule
            # exits regardless of partial/full. Distinct from wash-sale —
            # this fires even on small partial trims because the timing
            # signal (a stop tripped) invalidates re-entry.
            from kernel.pipeline.task_post_stop_cooldown import (  # noqa: PLC0415
                DEFAULT_STOP_EXIT_TYPES,
            )
            if str(getattr(sig, "exit_type", "")) in DEFAULT_STOP_EXIT_TYPES:
                self._last_stop_exit_dates_str[ticker] = today_str
            else:
                # TRIM (partial): replace stop with reduced qty at the same
                # stop_price (never loosens; see _z9_place_or_replace_stop).
                if self._z9_enabled(ctx):
                    held_now = (
                        broker.get_position(ticker)
                        if hasattr(broker, "get_position") else 0.0
                    )
                    if held_now > 0:
                        # Use the current price as reference; the helper
                        # min's against existing stop_price so the stop
                        # never moves up after a trim.
                        self._last_ctx_stop_pct = self._z9_stop_pct(ctx)
                        self._z9_place_or_replace_stop(
                            ticker, float(held_now), float(price), today_str,
                        )
                    else:
                        self._z9_cancel_stop(ticker, reason="trim → flat")
            regime_p = (self._config.get("regime_params", {}) or {}).get(
                ctx.regime, {},
            ) or {}
            sell_log_record = build_sell_trade_event_for_db(
                ticker=ticker,
                sig=sig,
                holding=hs,
                price=price,
                today=ctx.today,
                regime=getattr(ctx, "regime", None),
                confidence=getattr(ctx, "confidence", None),
                regime_params={
                    **regime_p,
                    "tax": self._config.get("tax", {}) or {},
                },
                config=self._config,
                **sell_event_realized_kwargs(sig, hs, today=ctx.today),
            )
            sell_log_record.update({
                "action":    "SELL",
                "symbol":    ticker,
                "exit_type": sig.exit_type,
                "reason":    sig.reason,
                "qty":       sell_qty,
                "partial":   is_partial,
            })
            self._log_trade(ctx, sell_log_record)

        # ── Apply buys ───────────────────────────────────────────────────────
        # Track BUYS as they actually execute vs what the pipeline merely
        # intended. `ctx.orders_placed` = filled/partially-filled at broker,
        # `ctx.orders_pending` = submitted but not filled yet, and
        # `ctx.orders_skipped` = blocked locally or rejected.
        if not hasattr(ctx, "orders_placed"):
            ctx.orders_placed = []
        if not hasattr(ctx, "orders_pending"):
            ctx.orders_pending = []
        if not hasattr(ctx, "orders_skipped"):
            ctx.orders_skipped = []
        import math
        try:
            buy_cash_remaining = float(ctx.cash)
        except (TypeError, ValueError):
            buy_cash_remaining = 0.0
        if not math.isfinite(buy_cash_remaining):
            buy_cash_remaining = 0.0
        sell_credit = same_bar_sell_credit(ctx)
        if sell_credit > 0:
            buy_cash_remaining += sell_credit
            log.info(
                "LIVE-SAME-BAR-SELL-CREDIT: buy budget credited by "
                "$%.2f from broker-confirmed exits",
                sell_credit,
            )
        if not self._sell_only:
            from kernel.pipeline.order_dedupe import (  # noqa: PLC0415
                dedupe_buy_orders_first_wins,
            )
            deduped_orders, skipped_duplicate_buys = (
                dedupe_buy_orders_first_wins(ctx.orders)
            )
            for order_intent in skipped_duplicate_buys:
                ticker = (
                    order_intent.get("ticker")
                    if isinstance(order_intent, dict) else
                    getattr(order_intent, "ticker", "?")
                )
                log.info("BUY skipped: duplicate same-bar buy intent for %s", ticker)
                if isinstance(order_intent, dict):
                    ctx.orders_skipped.append({
                        **order_intent,
                        "skip_reason": "duplicate_buy_intent",
                    })
            # S-FRAC v2 stage 2 (D7 gap inventory #1): the cash cap resizes
            # on the 6dp fractional grid ONLY when the capability gate is
            # fully satisfied (flag on AND capabilities present) — the same
            # source of truth the fail-closed entry check below consumes.
            # Flag off (all of today's production) ⇒ fractional=False ⇒
            # byte-identical legacy int truncation. Gate enabled-but-unsat
            # ⇒ every BUY fail-closes below regardless, so the value is
            # outcome-neutral there; False keeps it conservative.
            frac_cash_cap = bool(frac_gate["enabled"] and frac_gate["ok"])
            for order_intent in deduped_orders:
                order, budget_reason = cap_buy_order_to_cash(
                    order_intent, buy_cash_remaining,
                    fractional=frac_cash_cap,
                )
                if order is None:
                    log.info(
                        "BUY skipped: live cash budget rejected %s (%s)",
                        order_intent.get("ticker") if isinstance(order_intent, dict) else "?",
                        budget_reason,
                    )
                    if isinstance(order_intent, dict):
                        ctx.orders_skipped.append({
                            **order_intent,
                            "skip_reason": budget_reason or "cash_budget_rejected",
                        })
                    continue
                if budget_reason == "cash_budget_resized":
                    log.info(
                        "BUY resized by live cash budget: %s shares %s → %s",
                        order["ticker"], order.get("original_shares"), order["shares"],
                    )
                ticker = order["ticker"]
                shares = order["shares"]
                price  = order["price"]
                # S-FRAC stage 0 fail-closed entry (§2.2.2 + §2.2.3). Two
                # invariants, checked BEFORE any broker interaction:
                #   1. gate enabled-but-unsatisfied ⇒ NO buy is emitted at
                #      all (the config landed ahead of its dependencies);
                #   2. a fractional BUY intent never reaches the broker
                #      unless the flag is on AND the quantity is
                #      protectable (broker-side stop or an armed software-
                #      stop layer). With stage 3 absent this makes the
                #      stage-0 outage-window loss budget $0 by construction
                #      — no fractional position can come into existence.
                # Whole-share intents with the flag off (all of today's
                # production) take the `None` fast path — behavior
                # unchanged.
                frac_reason = fractional_entry_fail_closed_reason(
                    shares, frac_gate, broker=broker, symbol=ticker,
                    software_stops=getattr(self, "_software_stops", None),
                )
                if frac_reason is not None:
                    log.error(
                        "BUY fail-closed for %s (qty=%s): %s "
                        "[S-FRAC stage 0, design §2.2]",
                        ticker, fmt_qty(shares), frac_reason,
                    )
                    ctx.orders_skipped.append({
                        **order, "skip_reason": frac_reason,
                    })
                    continue
                # Duplicate-order guard
                try:
                    pending = broker.get_open_orders()
                    if ticker in pending:
                        log.info("BUY skipped: pending order exists for %s", ticker)
                        ctx.orders_skipped.append({
                            **order, "skip_reason": "pending_order_exists",
                        })
                        continue
                except Exception as exc:
                    log.error(
                        "BUY skipped: could not verify open orders for %s: %s",
                        ticker, exc,
                    )
                    ctx.orders_skipped.append({
                        **order,
                        "skip_reason": f"open_orders_check_failed:{type(exc).__name__}",
                    })
                    continue

                try:
                    result = broker.place_order(ticker, "BUY", shares)
                except Exception as exc:
                    log.error("BUY failed for %s: %s", ticker, exc)
                    ctx.orders_skipped.append({
                        **order, "skip_reason": f"broker_error:{type(exc).__name__}",
                    })
                    continue
                execution = broker_order_execution(
                    result, requested_qty=shares, fallback_price=price,
                )
                if execution["rejected"]:
                    log.error(
                        "BUY rejected for %s: status=%s order_id=%s",
                        ticker, execution["status"], execution.get("order_id"),
                    )
                    ctx.orders_skipped.append({
                        **order,
                        "skip_reason": f"broker_status:{execution['status']}",
                        "order_id": execution.get("order_id"),
                        "status": execution["status"],
                    })
                    continue

                submitted_notional = shares * price
                if execution["pending"]:
                    ctx.orders_pending.append({
                        **order,
                        "order_id": execution.get("order_id"),
                        "status": execution["status"],
                    })
                    buy_cash_remaining = max(buy_cash_remaining - submitted_notional, 0.0)
                    log.warning(
                        "BUY pending at broker for %s: %s shares status=%s "
                        "order_id=%s; entry state/DB not mutated until fill.",
                        ticker, fmt_qty(shares), execution["status"],
                        execution.get("order_id"),
                    )
                    continue

                # S-FRAC stage 0 (§2.2.1): broker filled_qty is authoritative
                # and preserved at float precision. This replaces the legacy
                # `int(execution["filled_qty"] or shares)` truncation — the
                # exact line Codex cited to block renquant-pipeline#153: a
                # broker fill of 0.435578 became 0 shares in orders_placed,
                # live_state, the trade journal, cash accounting, and the Z9
                # stop quantity. normalize_fill_qty snaps eps-integral fills
                # to int (the ONE sanctioned whole-share branch), so every
                # whole-share fill produces byte-identical order dicts /
                # journal rows / state JSON to the killed cast.
                shares = normalize_fill_qty(execution["filled_qty"], shares)
                price = float(execution["filled_avg_price"] or price)
                order = {**order, "shares": shares, "price": price}
                if execution.get("order_id") is not None:
                    order["order_id"] = execution.get("order_id")
                order["status"] = execution["status"]
                order["filled_qty"] = shares
                order["filled_avg_price"] = price
                ctx.orders_placed.append(order)

                invest = shares * price
                buy_cash_remaining = max(buy_cash_remaining - invest, 0.0)
                # Top-up detection: a buy on a ticker we already track is
                # an add-to-existing, not a fresh entry. Preserve entry_date,
                # entry_signals, sell_streaks, and last_sell_dates so the
                # original cost-basis tenure / wash-sale state stays intact.
                # HWM ratchets with current price (whichever is higher).
                is_topup = ticker in self._entry_dates
                action_tag = "TOPUP" if is_topup else "BUY"
                # fmt_qty: '5' for whole shares (byte-identical to the old
                # %d), full-precision float for a fractional fill (the old
                # %d silently truncated it in the log).
                log.info("%s  %s  %s shares @ %.2f  invest=$%.0f",
                         action_tag, ticker, fmt_qty(shares), price, invest)

                if not is_topup:
                    self._entry_dates[ticker]       = today_str
                    self._sell_streaks.pop(ticker, None)
                    self._protection_breaches.pop(ticker, None)
                    self._last_sell_dates_str.pop(ticker, None)
                    self._position_hwm[ticker]      = price
                    # Thesis-degradation baseline (Approach A) — stamp entry
                    # scores ONLY on a fresh buy (not a top-up to an already-
                    # held position). Persist in live_state.json so rotation
                    # checks on future bars see a fixed baseline.
                    self._entry_signals[ticker] = {
                        "rank_score":       order.get("rank_score"),
                        "panel_score":      order.get("panel_score"),
                        "kelly_target_pct": order.get("kelly_target_pct"),
                        "regime":           order.get("regime"),
                    }
                else:
                    # Top-up: only HWM may need to ratchet up.
                    self._position_hwm[ticker] = max(
                        float(self._position_hwm.get(ticker, 0.0)), price,
                    )
                # Z9 (2026-04-28): place / replace broker-side stop. Default
                # OFF; honors `live.broker_side_stops.enabled` config flag and
                # the broker's supports_broker_side_stops() capability.
                # On TOPUP: invariant is "never loosen" — handled by
                # _z9_place_or_replace_stop (it min's against existing stop).
                if self._z9_enabled(ctx):
                    self._last_ctx_stop_pct = self._z9_stop_pct(ctx)
                    # Total post-trade qty = previous + new shares.
                    held_now = (
                        broker.get_position(ticker)
                        if hasattr(broker, "get_position") else float(shares)
                    )
                    self._z9_place_or_replace_stop(
                        ticker, float(held_now), float(price), today_str,
                    )
                # Bug #22 fix (2026-04-26 round-7): defensive .get() on
                # order keys. The QP solver path (task_joint_qp.py) emits
                # order dicts WITHOUT rs_score / regime — they're produced
                # by SizeAndEmitTask but not by JointPortfolioQPTask.
                # Pre-fix the bare order["rs_score"] raised KeyError →
                # commit() crashed AFTER orders were submitted to Alpaca,
                # leaving live state inconsistent (orders filled but trade
                # log not written). Now: defensive get with safe defaults
                # so all order producers (selection / rotation / topup /
                # qp / future) are tolerated. rs_score is retired from
                # ranking math anyway (CLAUDE.md), so 0.0 is correct.
                # Trade-log distinguishes order provenance (2026-05-01 audit):
                # `order_type` carries through whatever the producer set
                # (TopUpHeldTask → "TOP_UP", SizeAndEmitTask → "NEW_BUY",
                # rotation/QP → respective tags). When absent, fall back to
                # the runner's own is_topup detection (handles legacy producers
                # that don't tag).
                fallback_type = "TOP_UP" if is_topup else "NEW_BUY"
                order_type    = order.get("order_type", fallback_type)
                buy_log_record = build_buy_trade_event(
                    order,
                    date=ctx.today,
                    default_regime=ctx.regime,
                    default_confidence=ctx.confidence,
                    default_acceptance_reason="live_buy",
                )
                buy_log_record.update({
                    "action":     "BUY",
                    "symbol":     ticker,
                    "shares":     shares,
                    "price":      price,
                    "invest":     invest,
                    "order_type": order_type,
                    "rank_score": order.get("rank_score", 0.0) or 0.0,
                    "rs_score":   order.get("rs_score",   0.0) or 0.0,
                    "regime":     order.get("regime",     ctx.regime),
                })
                self._log_trade(ctx, buy_log_record)

        # ── Persist updated sell streaks from SellJob ─────────────────────
        # Audit fix LS-HWM-1 (Round 2 deep audit, 2026-04-25): pre-fix,
        # this loop recomputed position_hwm from `ctx.prices[ticker]`
        # directly with `max(stored, price)`. That bypassed the EX-HWM
        # safety net living on hs.high_watermark — if ctx.prices[ticker]
        # was NaN/inf (one bad OHLCV bar), `max(stored, NaN) = NaN` and
        # the NaN got SERIALISED into live_state.json, surviving across
        # process restarts until the next compute_exits could recover it.
        # Now: prefer hs.high_watermark (already validated by compute_exits
        # via EX-HWM), fall back to a finite-checked max if hs is missing.
        import math
        for ticker, hs in ctx.holdings.items():
            if ticker in full_exit_tickers:
                continue
            self._sell_streaks[ticker] = hs.sell_streak
            self._protection_breaches[ticker] = int(getattr(hs, "protection_breaches", 0) or 0)
            # Prefer the validated HWM that compute_exits computed for
            # this bar; only fall back to a price-based max if hs is
            # somehow missing or non-finite.
            hs_hwm = getattr(hs, "high_watermark", None)
            if hs_hwm is not None and math.isfinite(hs_hwm):
                self._position_hwm[ticker] = float(hs_hwm)
            elif ticker in ctx.prices and math.isfinite(ctx.prices[ticker]):
                stored = float(self._position_hwm.get(ticker, 0.0))
                if not math.isfinite(stored):
                    stored = 0.0
                self._position_hwm[ticker] = max(stored, ctx.prices[ticker])

        # ── State garbage-collection (Bug A — stale entries) ──────────────
        # Audit fix STATE-GC (Round 4 deep audit, 2026-04-25): pre-fix,
        # `commit()` only added to live_state.json on buys/sells; it never
        # removed entries for tickers no longer held. Result: tickers like
        # XLU that were once held but later sold (manually or by a previous
        # version) remained in live_state forever — stale entry_date,
        # phantom position_hwm, ghost sell_streak — confusing the operator
        # and bloating state. Now: drop entries for tickers not in current
        # held_set, EXCEPT keep last_sell_dates entries inside the 30-day
        # wash-sale window (those are still load-bearing for future buys).
        #
        # Audit fix STATE-GC-NEWBUYS (Bug K2, 2026-04-25): pre-fix, ctx.holdings
        # was captured at start-of-bar (broker positions BEFORE today's buys
        # executed). New buys added entries to entry_dates via the buy loop,
        # then GC immediately dropped them because they weren't in
        # ctx.holdings. The state was self-correcting next iter (broker fills
        # would re-seed) but the immediate write was wrong. Fix: extend
        # currently_held with tickers from ctx.orders_placed (broker-confirmed
        # buys) so GC preserves them.
        currently_held = effective_live_holdings_after_orders(
            ctx.holdings.keys(),
            full_exit_tickers,
            getattr(ctx, "orders_placed", []) or [],
        )
        post_snapshot = live_post_execution_snapshot(ctx, broker, currently_held)

        # ── Manual / external disposition detection (Z2, 2026-04-28) ──────
        # Invariant: ANY position that disappears between bars must stamp
        # last_sell_dates, regardless of who sold it. Pre-fix, only sells
        # the runner itself executed got the wash-sale clock; manual sells
        # via the Alpaca app, broker-side liquidations (margin calls,
        # end-of-day flats), and IBKR-side closes were INVISIBLE to the
        # wash-sale guard — the bot could re-buy a hand-sold ticker the
        # next bar. NVTS post-mortem (2026-04-28): user manually exited
        # NVTS after the bot bought a parabolic top; with the old logic
        # NVTS could have been re-bought before the 30-day clock ran.
        # Heuristic: ticker was in entry_dates at start-of-bar AND is not
        # currently held AND wasn't stamped as a runner-sell today
        # → treat as external disposition, stamp today.
        #
        # 2026-05-17 Bug fix: EXCLUDE tickers with a pending broker order.
        # Pre-fix, a Sunday-afternoon BUY whose Alpaca order is still
        # `status=accepted` (not yet filled) shows the position as missing
        # at end-of-bar → STATE-EXT-SELL stamped wash-sale → next day's
        # fill couldn't re-enter even though it was the runner's own buy.
        # Today's HON (and 5/15's META) were both blocked this way.
        # Invariant: pending-at-broker ≠ externally-sold.
        pending_broker = set(getattr(ctx, "pending_broker_tickers", set()) or set())
        preopen_canceled = _preopen_cancel_symbols(
            self._strategy_dir, self._broker_name, today_str,
        )
        stale_canceled = [t for t in self._entry_dates
                          if t not in currently_held
                          and t not in pending_broker
                          and t in preopen_canceled
                          and self._last_sell_dates_str.get(t) != today_str]
        if stale_canceled:
            for t in stale_canceled:
                self._entry_dates.pop(t, None)
                self._entry_signals.pop(t, None)
                self._sell_streaks.pop(t, None)
                self._protection_breaches.pop(t, None)
                self._position_hwm.pop(t, None)
            log.warning(
                "STALE_STATE: %d ticker(s) missing from positions after "
                "pre-open cancelled order — clearing local entry state without "
                "wash-sale stamp: %s",
                len(stale_canceled), sorted(stale_canceled),
            )
        disappeared = [t for t in self._entry_dates
                       if t not in currently_held
                       and t not in pending_broker
                       and t not in preopen_canceled
                       and self._last_sell_dates_str.get(t) != today_str]
        skipped_pending = [t for t in self._entry_dates
                           if t not in currently_held
                           and t in pending_broker
                           and self._last_sell_dates_str.get(t) != today_str]
        if skipped_pending:
            log.info(
                "STATE-EXT-SELL: %d ticker(s) missing from positions but have "
                "pending broker orders — skipping wash-sale stamp (in-flight buy, "
                "not external sell): %s",
                len(skipped_pending), sorted(skipped_pending),
            )
        # Issue #71 / audit #5: STATE-EXT-SELL used to log only the ticker
        # name, leaving the operator unable to distinguish Z9 broker-side
        # stops from manual closes or corporate actions. Pre-fetch the recent
        # SELL-side broker fill history once so each disappeared ticker can
        # be attributed to a specific fill record (order_id, price, qty,
        # filled_at) and a source guess (z9_stop / external).
        ext_sell_fills = self._lookup_ext_sell_fills(ctx, disappeared)
        from adapters.runner_ext_sell import EXT_SELL_LOOKBACK_DAYS  # noqa: PLC0415
        for t in disappeared:
            attribution = self._attribute_ext_sell(t, ext_sell_fills)
            # 2026-07-01 fix (META incident) + codex #428 review follow-up:
            # this reconciliation may run DAYS after the ticker actually
            # left the book (e.g. a prior bar's GC/reconciliation step was
            # skipped by an unrelated pipeline failure, so `disappeared`
            # only fires here, later). Pre-fix, this always stamped
            # `today_str` — the date THIS code happens to run — discarding
            # the real fill date that `ext_sell_fills` (via
            # `_lookup_ext_sell_fills`, already fetched above, now with a
            # 45-day lookback — see EXT_SELL_LOOKBACK_DAYS) carries. That
            # silently EXTENDED the 30-day wash-sale block by however many
            # days late reconciliation ran. Confirmed live: META's
            # last_sell_dates was wrongly stamped 2026-06-26 (a
            # reconciliation run date) instead of the real broker SELL
            # fill on 2026-06-02 — a 24-day over-extension. Same authority
            # principle as ENTRY-DATE-FROM-FILLS for entry_dates: the
            # broker's fill timestamp is authoritative over "today".
            #
            # `_ext_sell_fill_date` only returns a date for a CONFIRMED
            # SELL fill (side == "sell", codex #428 review finding 2) with
            # a properly TZ-aware-parsed timestamp (finding 3) — an
            # ambiguous or BUY-side fill, or an unparseable/naive
            # timestamp, all yield None here rather than a guessed date.
            #
            # When no confirmed fill is found, `_ext_sell_stamp_decision`
            # (codex #428 review, "ALSO reconsider") PRESERVES an existing
            # older `last_sell_dates` value instead of overwriting it with
            # today — overwriting known evidence with "today" would
            # recreate the over-extension bug in a different form. Only
            # when there is truly no prior value at all does it fall back
            # to today_str (the conservative "block re-entry" choice for
            # a genuinely unknown-cause disappearance — corporate action,
            # account transfer, or a disposition the broker API can't
            # attribute to a dated fill within the lookback window).
            fill_date = self._ext_sell_fill_date(ext_sell_fills.get(t))
            prior_stamp = self._last_sell_dates_str.get(t)
            stamp_str, stamp_path = self._ext_sell_stamp_decision(
                fill_date, prior_stamp, today_str,
            )
            self._last_sell_dates_str[t] = stamp_str
            if stamp_path == "actual_fill":
                log.warning(
                    "STATE-EXT-SELL: %s disappeared from broker without runner "
                    "sell — stamping wash-sale clock from the ACTUAL broker fill "
                    "date %s (reconciliation ran %s) to prevent re-entry within "
                    "30d of the real sell (attribution: %s)",
                    t, stamp_str, today_str, attribution,
                )
            elif stamp_path == "unresolved_preserve":
                log.warning(
                    "STATE-EXT-SELL: %s disappeared from broker with NO "
                    "CONFIRMED SELL fill in the %dd lookback — UNRESOLVED "
                    "reconciliation; preserving the existing wash-sale stamp "
                    "%s rather than overwriting it with today (%s); recover "
                    "the real fill date from broker order history to confirm "
                    "or correct this (attribution: %s)",
                    t, EXT_SELL_LOOKBACK_DAYS, stamp_str, today_str, attribution,
                )
            else:
                log.warning(
                    "STATE-EXT-SELL: %s disappeared from broker without runner "
                    "sell, and NO broker SELL fill record was found — stamping "
                    "wash-sale clock as a NO-FILL-FOUND FALLBACK to today (%s) "
                    "(attribution: %s)",
                    t, today_str, attribution,
                )
            # Z9: cancel any orphan broker-side stop for this ticker.
            # The position is already gone; the stop on broker side is now
            # for 0 shares — Alpaca would auto-cancel, but be explicit.
            if self._z9_enabled(ctx):
                self._z9_cancel_stop(t, reason="external disposition")

        wash_sale_window_days = 30
        cutoff = ctx.today - datetime.timedelta(days=wash_sale_window_days)
        # 2026-05-17: preserve state for tickers with pending broker orders.
        # Same root as the STATE-EXT-SELL fix above: an in-flight buy is not
        # yet a position but its entry_date / entry_signal / position_hwm
        # are load-bearing for when it eventually fills (Monday open for
        # weekend-queued orders). Pre-fix, GC dropped them as "stale".
        held_or_pending = currently_held | pending_broker
        for store_name, store in (
            ("entry_dates",   self._entry_dates),
            ("entry_signals", self._entry_signals),
            ("sell_streaks",  self._sell_streaks),
            ("protection_breaches", self._protection_breaches),
            ("position_hwm",  self._position_hwm),
        ):
            stale = [t for t in store if t not in held_or_pending]
            for t in stale:
                store.pop(t, None)
            if stale:
                log.info("STATE-GC: dropped %d stale entries from %s: %s",
                         len(stale), store_name, ", ".join(sorted(stale)))
        # Z9: stop_orders GC. Orphan stops (no longer held) get cancelled
        # at the broker too — the position is gone so the stop is for 0
        # shares; Alpaca would no-op but be explicit.
        z9_stale = [t for t in self._stop_orders if t not in currently_held]
        for t in z9_stale:
            if self._z9_enabled(ctx):
                self._z9_cancel_stop(t, reason="stop_orders GC")
            else:
                self._stop_orders.pop(t, None)
        if z9_stale:
            log.info("STATE-GC: dropped %d stale stop_orders entries: %s",
                     len(z9_stale), ", ".join(sorted(z9_stale)))
        # S-FRAC stage 3: software-stop registry GC — entries whose
        # position is gone (external disposition / manual sell / a full
        # exit that bypassed the Z9 cancel path) are disarmed with an
        # audit reason. Flag-off (registry None) this is byte-inert; a
        # corrupt registry refuses the write and logs (never silently
        # mutated).
        if getattr(self, "_software_stops", None) is not None:
            try:
                sw_stale = self._software_stops.gc(currently_held)
            except Exception as exc:
                log.error("STATE-GC: software-stop registry GC failed: %s", exc)
                sw_stale = []
            if sw_stale:
                log.info(
                    "STATE-GC: disarmed %d stale software-stop entries: %s",
                    len(sw_stale), ", ".join(sorted(sw_stale)))
        # last_sell_dates: keep if within wash-sale window OR ticker still held.
        wash_stale = []
        for t, d_str in list(self._last_sell_dates_str.items()):
            if t in currently_held:
                continue
            try:
                d_obj = datetime.date.fromisoformat(d_str)
            except (ValueError, TypeError):
                wash_stale.append(t)
                continue
            if d_obj < cutoff:
                wash_stale.append(t)
        for t in wash_stale:
            self._last_sell_dates_str.pop(t, None)
        if wash_stale:
            log.info("STATE-GC: dropped %d expired wash-sale entries: %s",
                     len(wash_stale), ", ".join(sorted(wash_stale)))

        # ── Save live_state.json ──────────────────────────────────────────
        # Snapshot RegimeState (countdown / cusum / in_transition) so the
        # next live invocation resumes mid-cooldown instead of re-tripping
        # CUSUM from scratch. Without this, transition_window=True stays
        # stuck whenever SPY's 20-day window still differs from the 20-day
        # reference — which can last 20+ bars after a genuine regime shift.
        rs = getattr(ctx, "regime_state", None)
        regime_state_out = {
            "regime":        ctx.regime,
            "confidence":    round(ctx.confidence, 4),
            "in_transition": bool(getattr(rs, "in_transition", False)),
            "countdown":     int(getattr(rs, "countdown", 0)),
            "cusum_pos":     float(getattr(rs, "cusum_pos", 0.0)),
            "cusum_neg":     float(getattr(rs, "cusum_neg", 0.0)),
            # CUSUM-v2 Design C wall-clock cooldown start (ISO string or null).
            # Let intraday runs read elapsed time instead of ticking bar-count.
            "cooldown_start": (getattr(rs, "cooldown_start", None).isoformat()
                                if getattr(rs, "cooldown_start", None) is not None
                                else None),
        } if rs is not None else {}
        self._state.update({
            "regime":            ctx.regime,
            "regime_confidence": round(ctx.confidence, 4),
            "high_water_mark":   ctx.hwm,
            "skip_buys":         bool(ctx.skip_buys),
            "entry_dates":       self._entry_dates,
            "entry_signals":     self._entry_signals,   # Approach A
            "sell_streaks":      self._sell_streaks,
            "protection_breaches": self._protection_breaches,
            "last_sell_dates":   self._last_sell_dates_str,
            "last_stop_exit_dates": self._last_stop_exit_dates_str,
            "position_hwm":      self._position_hwm,
            "stop_orders":       self._stop_orders,    # Z9
            "recent_sell_orders": self._gc_recent_sell_orders(ctx),
            "regime_state":      regime_state_out,
            # MonitorIdleStreakTask counters — persisted across scheduled runs
            "monitor_state":     dict(getattr(ctx, "monitor_state", {}) or {}),
        })
        # 2026-05-20 fix: prefer broker-driven no_trade_streak when the broker
        # exposes get_filled_orders. Stateful counter has been bug-prone
        # (per-invocation vs per-day inflation, state-file corruption from
        # SIGKILL mid-write, …). Real source of truth = Alpaca's order book.
        try:
            self._override_no_trade_streak_from_broker(ctx)
        except Exception as exc:
            log.warning(
                "broker-driven no_trade_streak query failed (%s) — keeping "
                "stateful counter (value=%d). Counter is best-effort but may "
                "drift; investigate if persists.",
                exc,
                int(self._state.get("monitor_state", {}).get("no_trade_streak", 0)),
            )
        # Audit fix LS-ATOM (Round 2 deep audit, 2026-04-25): same atomic
        # write pattern as the parquet stores (DC-2-CACHE / FU-1 /
        # INT-ATOM / etc). Pre-fix, `write_text` opened the file in
        # truncate mode + wrote in-place. A SIGKILL or kernel panic
        # mid-write left a truncated/empty live_state.json on disk —
        # next live run loaded `{}` (default), losing all entry_dates,
        # position_hwm, sell_streaks, regime cooldown state. Wash-sale
        # guards then misfired, regime cooldowns reset, holding tenure
        # reset to today (corrupting tax classification + min_hold).
        # Now: write to .tmp + atomic rename, so a crash can leave the
        # .tmp half-written but the canonical file is still the prior
        # complete snapshot.
        from kernel.state_paths import live_state_path  # noqa: PLC0415
        # Always write to broker-specific path. Legacy live_state.json is
        # never overwritten — it stays as a frozen pre-isolation snapshot
        # for forensics until the operator manually retires it.
        state_file = live_state_path(self._strategy_dir, self._broker_name)
        from adapters.state_store import save_live_state_atomic  # noqa: PLC0415

        save_live_state_atomic(state_file, self._state, self._config)
        log.info("State saved → %s (atomic, broker=%s)",
                 state_file, self._broker_name)

        # ── Optional SQLite decision trace ────────────────────────────────
        if self._db is not None:
            from kernel.persistence import (  # noqa: PLC0415
                record_pipeline_run, record_candidate_scores, record_trades,
                record_live_state_snapshot, record_rotations,
                record_ticker_daily_state, validate_decision_trace_integrity,
            )
            from kernel.artifact_contract import build_run_bundle  # noqa: PLC0415
            # Reconstruct trade events from ctx (live path doesn't keep an
            # in-memory trade list — we synthesise from exits + orders).
            #
            # Audit fix EXITS-FAIL-DB (Round 4 deep audit, 2026-04-25):
            # pre-fix, this used `ctx.exits` (pipeline intent) instead of
            # `ctx.exits_placed` (broker-confirmed). Failed sells (caught
            # into `ctx.exits_failed` when broker rejected) silently were
            # written to `trades` table as successful — distorting PnL
            # analytics + n_exits count. Match the ntfy logic at
            # live/runner.py which already prefers exits_placed.
            # `*_placed` may legitimately be an empty list when every broker
            # attempt failed or was skipped. Do not fall back to pipeline
            # intent in that case; `trades` is an executed-trade table.
            if hasattr(ctx, "exits_placed"):
                exits_for_db = list(getattr(ctx, "exits_placed", []) or [])
            else:
                exits_for_db = list(ctx.exits or [])
            if hasattr(ctx, "orders_placed"):
                orders_for_db = list(getattr(ctx, "orders_placed", []) or [])
            else:
                orders_for_db = list(ctx.orders or [])
            pending_orders_for_trace = list(getattr(ctx, "orders_pending", []) or [])
            _, _, pending_tickers_for_trace = (
                live_trace_selection_maps(
                    [],
                    pending_orders_for_trace,
                    getattr(ctx, "_blocked_by_ticker", None) or {},
                )
            )
            if pending_tickers_for_trace:
                ctx.counters["broker_pending_submitted"] = (
                    ctx.counters.get("broker_pending_submitted", 0)
                    + len(pending_tickers_for_trace)
                )
            trade_events: list[dict] = []
            regime_p = (self._config.get("regime_params", {}) or {}).get(
                ctx.regime, {},
            ) or {}
            for t, sig in exits_for_db:
                hs    = ctx.holdings.get(t)
                price = sell_event_price(sig, ctx.prices.get(t, 0.0))
                trade_events.append(build_sell_trade_event_for_db(
                    ticker=t,
                    sig=sig,
                    holding=hs,
                    price=price,
                    today=ctx.today,
                    regime=getattr(ctx, "regime", None),
                    confidence=getattr(ctx, "confidence", None),
                    regime_params={**regime_p, "tax": self._config.get("tax", {}) or {}},
                    config=self._config,
                    **sell_event_realized_kwargs(sig, hs, today=ctx.today),
                ))
            for o in orders_for_db:
                trade_events.append(build_buy_trade_event(
                    o,
                    date=ctx.today,
                    default_regime=ctx.regime,
                    default_confidence=ctx.confidence,
                    default_acceptance_reason="live_buy",
                ))
            trade_events.extend(live_execution_attempt_events(ctx))
            run_bundle = build_run_bundle(
                self._config,
                self._strategy_dir,
                run_id=str(getattr(ctx, "run_id", "")),
                run_type="live",
                ctx=ctx,
                broker_mode=self._broker_name,
            )
            run_id = record_pipeline_run(
                self._db,
                run_type        = "live",
                run_date        = ctx.today,
                strategy        = str(self._config.get("model_name", "")),
                regime          = ctx.regime,
                confidence      = float(ctx.confidence) if ctx.confidence is not None else None,
                portfolio_value = post_snapshot["portfolio_value"],
                cash            = post_snapshot["cash"],
                n_candidates    = len(ctx.candidates),
                n_exits         = len(exits_for_db),
                # Audit fix ROT-COUNTER (Bug L, 2026-04-25): use EMITTED
                # rotations count (from EmitRotationsTask via counters dict),
                # not the considered count (len(ctx.rotations) before
                # Kelly/cash filters). Stops SQLite analytics from
                # double-counting rotations that were never executed.
                n_rotations     = int(ctx.counters.get("rotations", 0)),
                n_buys          = len(orders_for_db),
                buy_blocked     = bool(getattr(ctx, "buy_blocked", False)),
                skip_buys        = bool(getattr(ctx, "skip_buys", False)),
                bear_only        = bool(getattr(ctx, "bear_only", False)),
                counters         = getattr(ctx, "counters", {}) or {},
                run_bundle       = run_bundle,
                run_id          = getattr(ctx, "run_id", None),
            )
            selected_tickers, blocked_map, _pending_trace_tickers = (
                live_trace_selection_maps(
                    trade_events,
                    pending_orders_for_trace,
                    getattr(ctx, "_blocked_by_ticker", None) or {},
                )
            )
            for o in getattr(ctx, "orders_skipped", []) or []:
                if isinstance(o, dict) and o.get("ticker"):
                    blocked_map.setdefault(
                        o["ticker"], f"broker_skip:{o.get('skip_reason', 'skipped')}",
                    )
            blocked_map.update(trade_event_blocked_map(trade_events))
            # Audit fix DB-DECISION-FACTORS (2026-04-26 round-5): include
            # sector_map + model_types + panel_artifact path so post-hoc
            # analysis has the FULL decision context per (date, ticker).
            sector_map  = self._config.get("sector_map", {}) or {}
            model_types = model_types_from_models(self._models)
            panel_artifact = (
                self._config.get("ranking", {})
                            .get("panel_scoring", {})
                            .get("artifact_path")
            )
            qp_delta_by_ticker, qp_target_by_ticker, qp_status = qp_trace_maps(ctx)
            # 2026-05-04 user mandate ("rank_score need to be collected
            # properly for future fine tune"): persist the FULL pre-veto
            # candidate list so candidate_scores captures the complete
            # rank_score distribution per bar, not just survivors. The
            # snapshot is set by VetoWeakBuysTask before it filters
            # ctx.candidates. Vetoed rows are tagged via blocked_map
            # (veto:rank_score_below_floor / veto:rank_score_nan).
            cand_pool = candidate_trace_pool(ctx)
            from kernel.decision_trace import candidate_score_excluded_holding_tickers  # noqa: PLC0415
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
                excluded_holding_tickers=candidate_score_excluded_holding_tickers(self._config),
            )
            record_trades(self._db, run_id, trade_events)
            record_rotations(self._db, run_id, ctx)

            # ── L6 score-drift audit sidecar (eng plan §L6 audit sidecar).
            # Best-effort + AUDIT-ONLY: reads the candidate_scores just
            # written, appends PSI drift history, and folds the verdict into
            # the alert escalation book. Degrade-safe — never blocks the bar
            # (a missing L6 stack / table is a silent no-op).
            run_l6_score_audit_sidecar(self._db, run_id=run_id, run_date=ctx.today)

            # ── ticker_daily_state — every watchlist ticker, every bar ──
            # Per user spec round-5 (2026-04-26): write a row for EVERY
            # watchlist ticker at decision time, including those filtered
            # at universe / broker / no-model gates. Lets post-hoc
            # analysis answer "what did we KNOW about XYZ on this date
            # and WHY didn't we trade it?" — instead of just the cands.
            try:
                pf_value = float(ctx.portfolio_value) if ctx.portfolio_value else 0.0
                # Bug #20 fix (2026-04-26): pending_broker_tickers is a local
                # of make_context() (line 170), not visible in commit()'s
                # scope. It IS persisted onto ctx at line 478 — read from
                # there. Pre-fix, the bare-name reference raised NameError
                # → swallowed by the outer try/except → ticker_daily_state
                # silently dropped EVERY bar. Defensive default to set()
                # so a sell-only path that didn't run BROKER-PRECHECK still
                # writes the row (with pending_at_broker=0).
                pending_broker_tickers: set = set(
                    getattr(ctx, "pending_broker_tickers", None) or set()
                )
                tds_rows = build_ticker_daily_state_rows(
                    config=self._config,
                    ctx=ctx,
                    selected_tickers=selected_tickers,
                    blocked_map=blocked_map,
                    model_types=model_types,
                    universe_rejections=self._universe_rejections,
                    model_keys=set(self._models or {}),
                    pending_broker_tickers=pending_broker_tickers,
                    portfolio_value=pf_value,
                    sector_map=sector_map,
                    qp_delta_by_ticker=qp_delta_by_ticker,
                    qp_target_by_ticker=qp_target_by_ticker,
                    qp_status=qp_status,
                    extra_tickers=trade_event_tickers(trade_events),
                )
                n_tds = record_ticker_daily_state(
                    self._db, run_date=ctx.today, rows=tds_rows,
                    run_id=run_id,
                )
                log.info("ticker_daily_state: wrote %d row(s) for %s",
                         n_tds, ctx.today.isoformat())
                # Gate-verdict ledger (eng plan S2-PR4 / errata C):
                # best-effort append; never blocks the bar.
                try:
                    from kernel.persistence import record_gate_verdicts  # noqa: PLC0415

                    n_gv = record_gate_verdicts(
                        self._db, run_id=run_id, run_date=ctx.today,
                        registry=getattr(ctx, "gate_registry", None),
                    )
                    if n_gv:
                        log.info("gate_verdicts: wrote %d row(s)", n_gv)
                except Exception as exc:  # noqa: BLE001
                    log.warning("gate_verdicts write failed: %s", exc)
                # S5: cross-run decision ledger (renquant-orchestrator #133).
                # The gate_verdicts table above is per-run; this writes the
                # same verdicts to the append-only decision_ledger.db so
                # "why sell-only on date X?" is one SQL query across runs.
                try:
                    _dl_registry = getattr(ctx, "gate_registry", None)
                    if _dl_registry is not None:
                        from renquant_orchestrator.decision_ledger import (  # noqa: PLC0415
                            connect as _dl_connect,
                            write_verdicts as _dl_write,
                        )
                        _dl_rows = _dl_registry.ledger_rows(run_id=run_id)
                        if _dl_rows:
                            _dl_conn = _dl_connect()
                            try:
                                _dl_n = _dl_write(
                                    _dl_conn, run_id,
                                    ctx.today.isoformat(), _dl_rows,
                                )
                                if _dl_n:
                                    log.info("decision_ledger: persisted %d verdict(s)", _dl_n)
                            finally:
                                _dl_conn.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("decision_ledger write failed (non-fatal): %s", exc)
                # M5: tournament shadow admission logger (orch PR #395).
                # Logs both tournament and panel admission verdicts in
                # parallel so the two paths can be compared before retiring
                # the tournament gate.  Default OFF; fail-open.
                try:
                    _ts_cfg = self._config.get("tournament_shadow", {}) or {}
                    if _ts_cfg.get("enabled", False):
                        from renquant_orchestrator.tournament_shadow_admission import (  # noqa: PLC0415
                            log_shadow_admission,
                        )

                        _ts_scores = _build_tournament_shadow_ticker_scores(
                            cand_pool, blocked_map,
                        )
                        _ts_panel_cands = [
                            c.ticker for c in (ctx.candidates or [])
                        ]
                        _ts_panel_blocked = {
                            t: r for t, r in blocked_map.items()
                            if t not in set(_ts_panel_cands)
                        }
                        _ts_regime_params = getattr(ctx, "regime_params", None) or {}
                        _ts_min_score = float(
                            _ts_regime_params.get("min_model_score", 0.10),
                        )
                        _ts_bypass = bool(
                            self._config.get("ranking", {})
                                        .get("panel_scoring", {})
                                        .get("bypass_ticker_gate", False),
                        )
                        _ts_record = log_shadow_admission(
                            run_date=ctx.today,
                            watchlist=list(self._config.get("watchlist", [])),
                            ticker_scores=_ts_scores,
                            panel_candidates=_ts_panel_cands,
                            panel_blocked=_ts_panel_blocked,
                            min_model_score=_ts_min_score,
                            bypass_ticker_gate=_ts_bypass,
                            regime=getattr(ctx, "regime", None),
                            shadow_dir=_ts_cfg.get("shadow_dir"),
                            enabled=True,
                        )
                        if _ts_record is not None:
                            log.info(
                                "tournament_shadow: agreement=%.1f%% "
                                "(tourn_only=%d, panel_only=%d)",
                                _ts_record.agreement_rate * 100,
                                len(_ts_record.tournament_only),
                                len(_ts_record.panel_only),
                            )
                except Exception as exc:  # noqa: BLE001
                    log.warning("tournament_shadow write failed (non-fatal): %s", exc)
            except Exception as exc:
                # Diagnostic table — never block the bar on a write error.
                log.warning("ticker_daily_state write failed: %s", exc)
                if bool((self._config.get("persistence", {}) or {})
                        .get("strict_ticker_daily_state", True)):
                    raise

            # Plan S — append live_state snapshot. The JSON file is still
            # the source of truth (fast bootstrap + human edits); this row
            # is an append-only audit trail for "what was state X on date Y?"
            record_live_state_snapshot(
                self._db, run_id,
                run_date        = ctx.today,
                strategy        = str(self._config.get("model_name", "")),
                state           = self._state,
                cash            = post_snapshot["cash"],
                portfolio_value = post_snapshot["portfolio_value"],
                n_holdings      = int(post_snapshot["n_holdings"]),
            )
            validate_decision_trace_integrity(
                self._db,
                run_id,
                self._config,
                context="RunnerAdapter.commit",
            )

    # ── Trade log ─────────────────────────────────────────────────────────────

    def _log_trade(self, ctx, record: dict) -> None:
        import datetime as _dt
        strategy_name = self._config.get("model_name", "renquant_104")
        repo_root     = self._strategy_dir.parent.parent
        log_dir       = repo_root / "live" / "logs" / strategy_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{_dt.datetime.now().strftime('%Y-%m-%d')}.json"
        entries = json.loads(log_file.read_text()) if log_file.exists() else []
        record["timestamp"] = _dt.datetime.now().isoformat()
        entries.append(record)
        log_file.write_text(json.dumps(entries, indent=2, default=str))
