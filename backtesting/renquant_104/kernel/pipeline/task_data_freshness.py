"""DataFreshnessGateTask — refuse inference on stale market data.

2026-05-03 P0 incident: panel pipeline ingested OHLCV through Thursday close
only, then trained wl=183 + ran live inference on Sunday evening, submitting
6 orders to Alpaca that priced on Thursday closes 3 trading days behind the
last completed Friday session. Root cause was ``LocalStore.has_range``
silently accepting stale cache (``tolerance_days=5`` legacy default plus the
``end=None`` short-circuit that skipped the staleness check entirely).

This gate is the second line of defense — even if cache logic regresses
again, the inference pipeline will refuse to submit orders when ANY symbol's
OHLCV data does not include the last completed NYSE trading session.
When the adapter provides ``ctx.run_timestamp`` the gate is time-aware:
after today's NYSE close, today's session is required; before close,
yesterday's completed session is sufficient.

Invariant: no order leaves this pipeline based on market data older than
the most recent completed NYSE session.

Wired into ``InferencePipeline.run()`` BEFORE ``RegimeJob`` so the gate
fires before any decision logic touches stale data. Disable for backtests
or known-stale environments via ``config.data_freshness.enabled = false``.
"""
from __future__ import annotations

import datetime as _dt
import logging

import pandas as pd

from .pipeline import Task

log = logging.getLogger("kernel.pipeline.data_freshness")


class DataFreshnessGateTask(Task):
    """Hard-fail when ctx.ohlcv is older than last completed NYSE session.

    Reads:
      ctx.today, ctx.run_timestamp, ctx.ohlcv,
      ctx.config['data_freshness']['enabled'] (default True)
    Raises:
      RuntimeError when any symbol's max date is < last_completed_close.
    """

    def run(self, ctx) -> bool:
        cfg = (ctx.config or {}).get("data_freshness", {}) if hasattr(ctx, "config") else {}
        if cfg.get("enabled", True) is False:
            log.info("DataFreshnessGateTask: disabled via config — skipping")
            return True

        ohlcv = getattr(ctx, "ohlcv", None) or {}
        expected_symbols = self._expected_symbols(ctx, cfg)
        if not ohlcv:
            if expected_symbols:
                sample = ", ".join(sorted(expected_symbols)[:10])
                msg = (
                    "DataFreshnessGateTask: OHLCV missing — expected market "
                    f"data for {len(expected_symbols)} symbol(s). Sample: {sample}. "
                    "Refusing to submit any orders."
                )
                log.error(msg)
                raise RuntimeError(msg)
            # Empty ohlcv is a different bug class (missing fetch) — downstream
            # tasks (RegimeJob loading SPY etc) raise their own errors. This
            # gate's scope is STALENESS only. Log + continue keeps test stubs
            # working while production-real empty-ohlcv still fails fast.
            log.warning(
                "DataFreshnessGateTask: ctx.ohlcv is empty — staleness check "
                "skipped. Downstream tasks will fail if data really is missing."
            )
            return True

        ref_ts = self._ref_timestamp(ctx)
        ref_date = self._ref_date(getattr(ctx, "today", None), ref_ts)
        last_close = self._last_completed_nyse_close(ref_date, ref_ts)

        if last_close is None:
            log.warning(
                "DataFreshnessGateTask: no NYSE session in last 14 days "
                "before %s — degenerate calendar; skipping check.", ref_date
            )
            return True

        sell_only = self._is_sell_only(ctx)
        missing_expected = sorted(s for s in expected_symbols if s not in ohlcv)
        if missing_expected:
            sample = ", ".join(missing_expected[:10])
            n = len(missing_expected)
            msg = (
                f"DataFreshnessGateTask: OHLCV MISSING — {n} expected symbol(s) "
                f"are absent from ctx.ohlcv. Sample: {sample}"
                + (f" (+{n - 10} more)" if n > 10 else "")
                + ". Refusing to submit any orders."
            )
            log.error(msg)
            raise RuntimeError(msg)

        stale_syms: list[tuple[str, _dt.date]] = []
        symbols_to_check = (
            sorted(expected_symbols)
            if sell_only and expected_symbols else
            sorted(ohlcv)
        )
        for sym in symbols_to_check:
            df = ohlcv.get(sym)
            if df is None or len(df) == 0:
                stale_syms.append((sym, _dt.date(1970, 1, 1)))
                continue
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    max_d = df.index.max().date()
                else:
                    max_d = pd.to_datetime(df.index.max()).date()
            except Exception:
                stale_syms.append((sym, _dt.date(1970, 1, 1)))
                continue
            if max_d < last_close:
                stale_syms.append((sym, max_d))

        if stale_syms:
            sample = ", ".join(f"{s}@{d}" for s, d in stale_syms[:5])
            n = len(stale_syms)
            msg = (
                f"DataFreshnessGateTask: PANEL STALE — {n} symbol(s) lack "
                f"the last completed NYSE close ({last_close}). "
                f"Sample: {sample}"
                + (f" (+{n - 5} more)" if n > 5 else "")
                + ". Refusing to submit any orders. Run "
                "scripts/refresh_panel_ohlcv.py (or wait for the next "
                "post-close ingestion cron) and retry."
            )
            log.error(msg)
            raise RuntimeError(msg)

        log.info(
            "DataFreshnessGateTask: PASS  %d symbols, all ≥ %s",
            len(ohlcv), last_close,
        )
        return True

    @staticmethod
    def _expected_symbols(ctx, cfg: dict) -> set[str]:
        if cfg.get("require_expected_symbols", True) is False:
            return set()
        config = getattr(ctx, "config", {}) or {}
        explicit = cfg.get("expected_symbols")
        if explicit:
            return {str(s) for s in explicit if s}
        watchlist = list(config.get("watchlist", []) or [])
        holdings = list(getattr(ctx, "holdings", {}) or [])
        if DataFreshnessGateTask._is_sell_only(ctx):
            expected = {str(s) for s in holdings if s}
            benchmark = config.get("benchmark", "SPY")
            if benchmark:
                expected.add(str(benchmark))
            return expected
        if not watchlist and not holdings:
            return set()
        expected = {str(s) for s in watchlist + holdings if s}
        benchmark = config.get("benchmark", "SPY")
        if benchmark:
            expected.add(str(benchmark))
        for sym in (config.get("sector_etf_map", {}) or {}).values():
            if sym:
                expected.add(str(sym))
        return expected

    @staticmethod
    def _is_sell_only(ctx) -> bool:
        config = getattr(ctx, "config", {}) or {}
        mode = (
            getattr(ctx, "_run_mode", None)
            or config.get("_run_mode")
            or ""
        )
        return str(mode).strip().lower().replace("_", "-").startswith("sell-only")

    @staticmethod
    def _ref_date(today, ref_ts: pd.Timestamp | None = None) -> _dt.date:
        if today is None:
            if ref_ts is not None:
                return ref_ts.tz_convert("America/New_York").date()
            return _dt.date.today()
        if isinstance(today, _dt.datetime):
            return today.date()
        if isinstance(today, _dt.date):
            return today
        return pd.to_datetime(today).date()

    @staticmethod
    def _ref_timestamp(ctx) -> pd.Timestamp | None:
        for attr in ("run_timestamp", "now", "timestamp"):
            raw = getattr(ctx, attr, None)
            if raw is None:
                continue
            ts = pd.Timestamp(raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            return ts.tz_convert("UTC")
        return None

    @staticmethod
    def _last_completed_nyse_close(
        ref: _dt.date,
        now_ts: pd.Timestamp | None = None,
    ) -> _dt.date | None:
        try:
            import pandas_market_calendars as mcal  # noqa: PLC0415
        except ImportError:
            return ref - _dt.timedelta(days=2)

        cal = mcal.get_calendar("NYSE")
        sched = cal.schedule(
            start_date=ref - _dt.timedelta(days=14),
            end_date=ref,
        )
        if now_ts is not None:
            todays_session = sched[sched.index.date == ref]
            if not todays_session.empty:
                market_close = pd.Timestamp(todays_session["market_close"].iloc[-1])
                if market_close.tzinfo is None:
                    market_close = market_close.tz_localize("UTC")
                else:
                    market_close = market_close.tz_convert("UTC")
                if now_ts >= market_close:
                    return ref
        sched_before = sched[sched.index.date < ref]
        if sched_before.empty:
            return None
        return sched_before.index[-1].date()
