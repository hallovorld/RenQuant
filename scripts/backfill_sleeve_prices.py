#!/usr/bin/env python3
"""Warm-up backfill of parking-sleeve leg daily OHLCV (SGOV + SPY).

Why this exists
---------------
renquant-pipeline#185 implements parking-sleeve ``mode="live"`` (RS-1 SGOV
floor variant) and — correctly — FAIL-CLOSES on a missing SGOV price: no
buy is ever emitted without a positive price. Its data-availability
finding (2026-07-10, [VERIFIED]): SGOV daily bars exist in NEITHER OHLCV
store (`data/ohlcv/` has SPY+BIL-stale, no SGOV;
`backtesting/renquant_104/data/ohlcv/` has SPY only).

The daily fetch mechanism for the sleeve legs ALREADY EXISTS and is
umbrella-owned: ``adapters/sleeve_prices.parking_sleeve_price_tickers``
is wired into all four price paths (adapters/runner.py live OHLCV fetch,
adapters/lean.py History batch, main.py AddEquity, adapters/sim_price.py)
— but it is conditional on ``sleeve.enabled`` per renquant-strategy-104#39's
pinned coverage decision (a T-bill ETF must NOT join the watchlist; fetch
the sleeve tickers only when the sleeve is on). Consequence: SGOV bars
cannot exist BEFORE the enable flip, so the flip-day run would depend on
a cold-start remote fetch (10y history) in the middle of the live run —
one yfinance hiccup and mode=live fail-closes on day 1, and the RS-1 §4
pre-registered cash/SGOV/SPY comparison has no SGOV history to replay.

This script is the OPERATOR LANDING STEP that closes the gap: a one-shot,
idempotent warm-up of the sleeve-leg daily bars through the canonical
cache path (``kernel.data.fetch_ohlcv_incremental`` → same
``data/ohlcv/{SYMBOL}/1d.parquet`` store the runner reads). After the
flip, the existing conditional daily coverage keeps the bars fresh; this
script is NOT a recurring job.

Safety
------
* DRY-RUN BY DEFAULT — reports the resolved store, legs, and cache state;
  performs NO network call and NO write. Pass ``--write`` to fetch.
* Never touches the watchlist, strategy config, panel artifacts, or the
  model universe. The panel config fingerprint
  (``renquant_common.config_consistency._model_relevant_fields``) hashes
  watchlist / panel_ltr flags / sector maps only — neither the ``sleeve``
  config section nor OHLCV cache contents enter it, so backfilling SGOV
  bars cannot trip P-CONFIG-FP (pinned by tests/test_sleeve_prices.py).
* FAIL-CLOSED: refuses to run if the resolved SGOV leg is in the
  watchlist (st104#39 violation — fix the config, don't mask it), and
  exits non-zero if any leg is still missing/stale after ``--write``.

Usage
-----
  # Inspect (no writes, no network)
  .venv/bin/python scripts/backfill_sleeve_prices.py

  # Operator landing step: backfill into the canonical store
  .venv/bin/python scripts/backfill_sleeve_prices.py --write

  # Non-default store (e.g. the strategy-local store) or config
  .venv/bin/python scripts/backfill_sleeve_prices.py --write \
      --store-dir backtesting/renquant_104/data/ohlcv \
      --strategy-config backtesting/renquant_104/strategy_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

log = logging.getLogger("backfill_sleeve_prices")

DEFAULT_STRATEGY_CONFIG = _STRATEGY_DIR / "strategy_config.json"


def _load_config(path: Path) -> dict:
    """Load the strategy config; a missing/section-less config is fine.

    The deployed umbrella working config may predate the strategy-104
    ``sleeve`` section (#39/#44) — ``parking_sleeve_leg_tickers`` falls
    back to the pinned SPY/SGOV defaults, which mirror the pipeline
    task's own reads, so warm-up still resolves the right legs.
    """
    if not path.exists():
        log.warning("strategy config %s missing — using SPY/SGOV defaults", path)
        return {}
    return json.loads(path.read_text())


def _sgov_leg(config: dict) -> str:
    from adapters.sleeve_prices import parking_sleeve_config  # noqa: PLC0415

    sleeve = parking_sleeve_config(config)
    return str(sleeve.get("sgov_symbol", "SGOV")).strip().upper() or "SGOV"


def _leg_report(store, symbol: str) -> dict:
    """Cache state for one leg: exists / rows / span / session-fresh."""
    df = store.load(symbol)
    fresh = False
    try:
        fresh = store.has_range(symbol)  # NYSE-session-aware freshness
    except Exception as exc:  # pragma: no cover - calendar lib edge
        log.warning("has_range(%s) failed: %s", symbol, exc)
    return {
        "symbol": symbol,
        "path": str(store._path(symbol)),
        "exists": df is not None,
        "rows": 0 if df is None else int(len(df)),
        "first_bar": None if df is None else str(df.index.min().date()),
        "last_bar": None if df is None else str(df.index.max().date()),
        "fresh": bool(fresh),
    }


def _print_report(tag: str, rep: dict) -> None:
    state = (
        "[VERIFIED fresh]" if rep["fresh"]
        else ("[STALE]" if rep["exists"] else "[MISSING]")
    )
    print(
        f"{tag} {rep['symbol']}: {state} rows={rep['rows']} "
        f"span={rep['first_bar']}..{rep['last_bar']} path={rep['path']}"
    )


def run(argv: list[str] | None = None, fetch_fn=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strategy-config", type=Path, default=DEFAULT_STRATEGY_CONFIG,
        help="Strategy config to resolve sleeve.spy_symbol/sgov_symbol from "
             f"(default: {DEFAULT_STRATEGY_CONFIG})")
    parser.add_argument(
        "--store-dir", type=Path, default=None,
        help="OHLCV store directory (default: the repo-root-anchored "
             "data/ohlcv the daily runner reads)")
    parser.add_argument(
        "--timeout-sec", type=float, default=120.0,
        help="Per-symbol remote fetch timeout (default 120s; cold start "
             "pulls ~10y of daily bars)")
    parser.add_argument(
        "--write", action="store_true",
        help="Actually fetch + persist. Default is a dry-run report.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from adapters.sleeve_prices import parking_sleeve_leg_tickers  # noqa: PLC0415
    from kernel.data import LocalStore  # noqa: PLC0415

    config = _load_config(args.strategy_config)
    legs = parking_sleeve_leg_tickers(config)
    sgov = _sgov_leg(config)
    store = LocalStore(data_dir=args.store_dir) if args.store_dir else LocalStore()

    # FAIL-CLOSED guard: st104#39 pins that the T-bill leg never joins the
    # watchlist (it would enter panel scoring + cross-sectional admission
    # stats and change the panel config fingerprint). If it somehow has,
    # refuse — fix the config; a backfill must not mask the violation.
    watchlist = set(config.get("watchlist") or [])
    if sgov in watchlist:
        print(
            f"REFUSING: sleeve sgov_symbol {sgov} is in the watchlist — "
            "st104#39 violation; fix strategy config before backfilling.",
            file=sys.stderr,
        )
        return 2

    print(f"store: {store.data_dir}")
    print(f"legs (from {args.strategy_config}): {legs}")

    for symbol in legs:
        _print_report("before", _leg_report(store, symbol))

    if not args.write:
        print("DRY-RUN (default): no fetch performed. Re-run with --write "
              "to backfill through kernel.data.fetch_ohlcv_incremental.")
        return 0

    if fetch_fn is None:
        from kernel.data import fetch_ohlcv_incremental  # noqa: PLC0415

        def fetch_fn(symbol):  # noqa: F811 - default fetcher
            return fetch_ohlcv_incremental(
                symbol, store=store, timeout_sec=args.timeout_sec)

    failures = []
    for symbol in legs:
        try:
            fetch_fn(symbol)
        except Exception as exc:
            log.error("fetch failed for %s: %s", symbol, exc)
        rep = _leg_report(store, symbol)
        _print_report("after", rep)
        if not rep["fresh"]:
            failures.append(symbol)

    if failures:
        print(f"FAIL: legs still missing/stale after backfill: {failures}",
              file=sys.stderr)
        return 1
    print("OK: all sleeve legs present and session-fresh in "
          f"{store.data_dir} [VERIFIED]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
