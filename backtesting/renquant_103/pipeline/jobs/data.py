"""DataJob — parallel OHLCV fetch and artifact loading.

Populates ctx: ohlcv, df_spy, gmm_artifact, corr_matrix, earnings_cal.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger("pipeline.data")

from ..context import PipelineContext
from ..pipeline import Job
from ..task import run_tasks


class DataJob(Job):
    """Load artifacts and fetch OHLCV for every symbol in parallel."""

    def run(self, ctx: PipelineContext) -> None:
        config       = ctx.config
        strategy_dir = ctx.strategy_dir
        regime_cfg   = config.get("regime", {})

        # ── Resolve artifact directory ─────────────────────────────────────────
        artifacts_dir = strategy_dir / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir = strategy_dir  # fallback for strategies without subdir

        # ── Load optional artifacts (fast I/O, sequential) ────────────────────
        gmm_path = artifacts_dir / regime_cfg.get("gmm_artifact", "spy-gmm-regime.json")
        if gmm_path.exists():
            try:
                ctx.gmm_artifact = json.loads(gmm_path.read_text())
            except Exception as exc:
                log.warning("Could not load GMM artifact: %s", exc)

        corr_path = artifacts_dir / regime_cfg.get("correlation_artifact", "watchlist-correlation.json")
        if corr_path.exists():
            try:
                ctx.corr_matrix = json.loads(corr_path.read_text())
            except Exception as exc:
                log.warning("Could not load correlation artifact: %s", exc)

        earn_path = artifacts_dir / "earnings-calendar.json"
        if earn_path.exists():
            try:
                ctx.earnings_cal = json.loads(earn_path.read_text())
            except Exception as exc:
                log.warning("Could not load earnings calendar: %s", exc)

        # ── Build symbol list (watchlist + benchmark + sector ETFs) ───────────
        watchlist    = config["watchlist"]
        benchmark    = config.get("benchmark", "SPY")
        sector_etfs  = set(config.get("sector_etf_map", {}).values())
        all_symbols  = list(dict.fromkeys(watchlist + [benchmark] + sorted(sector_etfs)))
        data_provider = config.get("data_src", "yfinance")

        # ── Parallel OHLCV fetch ───────────────────────────────────────────────
        from common.data import fetch_ohlcv  # common/ available on host

        def _fetch(symbol: str):
            df = fetch_ohlcv(symbol, provider=data_provider)
            if df.empty:
                return symbol, None
            df = _ensure_fresh(symbol, df, data_provider)
            return symbol, df

        tasks = [(sym, lambda s=sym: _fetch(s)) for sym in all_symbols]
        results = run_tasks(tasks, max_workers=8)

        for r in results:
            if r.error:
                log.warning("OHLCV fetch failed for %s: %s", r.name, r.error)
                continue
            symbol, df = r.result
            if df is None or df.empty:
                log.warning("No data for %s, skipping", symbol)
                continue
            ctx.ohlcv[symbol] = df

        if benchmark not in ctx.ohlcv:
            log.error("No benchmark data for %s — aborting", benchmark)
            raise RuntimeError(f"Missing benchmark data for {benchmark}")

        ctx.df_spy = ctx.ohlcv[benchmark]
        log.info("DataJob: loaded OHLCV for %d/%d symbols", len(ctx.ohlcv), len(all_symbols))


# ── Freshness guard ────────────────────────────────────────────────────────────

_MAX_AGE_DAYS = 5  # calendar days; accounts for weekends + 1 missed trading day


def _ensure_fresh(symbol: str, df, provider: str, max_age_days: int = _MAX_AGE_DAYS):
    """Return *df* refreshed if the last row is stale (> max_age_days old)."""
    last_date = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
    age = (date.today() - last_date).days
    if age <= max_age_days:
        return df

    log.warning(
        "OHLCV for %s is %d days old (last=%s) — refreshing",
        symbol, age, last_date,
    )
    try:
        from common.data import fetch_ohlcv, LocalStore
        fresh = fetch_ohlcv(symbol, cache=False, provider=provider)
        if not fresh.empty:
            LocalStore().save(fresh, symbol)
            log.info("OHLCV refreshed for %s — %d rows", symbol, len(fresh))
            return fresh
    except Exception as exc:
        log.warning("Refresh failed for %s (%s) — using stale data", symbol, exc)
    return df
