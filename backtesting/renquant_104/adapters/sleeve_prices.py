"""Parking-sleeve daily price coverage — st104 #39 umbrella follow-up.

The S7 parking sleeve (renquant-pipeline #157, ``ParkingSleeveShadowTask``;
config keys defined inert in renquant-strategy-104 #39) needs daily prices
for its two legs, ``sleeve.spy_symbol`` and ``sleeve.sgov_symbol``. SPY is
already covered everywhere as the benchmark / a watchlist member; SGOV is
not fetched anywhere. Strategy-104 #39 pinned the coverage decision: a
T-bill ETF must NOT join the watchlist (it would enter panel scoring and
cross-sectional admission stats), so coverage is umbrella-owned and follows
the ``benchmark_sleeve`` precedent — subscribe/fetch the sleeve tickers
ONLY when the sleeve is enabled.

This module is the single umbrella implementation of "which extra tickers
does the parking sleeve need priced" — used by LEAN ``main.py``
(AddEquity), ``adapters/lean.py`` (History batch), ``adapters/runner.py``
(live OHLCV fetch), and ``adapters/sim_price.py`` (sim pricing universe).
One impl on purpose: hand-copying the config read into each call site is
how the calibrator-fingerprint triple-impl bug happened.

Import-routing note (why this lives in ``adapters/`` and not
``kernel/pipeline/``): on the multirepo daily path the orchestrator bridge
aliases every ``kernel.*`` module to the pinned ``renquant_pipeline``
checkout, so an umbrella-only helper added under ``kernel/pipeline/``
would be shadowed by (or collide with) the pipeline's own
``task_parking_sleeve`` module. ``adapters/*`` is umbrella-owned on every
path.

Contract with the pipeline task (renquant-pipeline
``kernel/pipeline/task_parking_sleeve.py``): the ``sleeve`` section shape,
the ``enabled`` gate, and the ``spy_symbol``/``sgov_symbol`` key names +
``"SPY"``/``"SGOV"`` defaults + strip/upper normalization below mirror
that task's reads exactly (and the defaults are additionally pinned by
renquant-strategy-104's config pin test). Flag absent/off ⇒ this module
returns nothing and every call site is byte-inert.
"""
from __future__ import annotations

from typing import Any


def parking_sleeve_config(obj: Any) -> dict:
    """Return ``config["sleeve"]`` from a config dict or ctx-like object."""
    cfg = obj if isinstance(obj, dict) else (getattr(obj, "config", None) or {})
    if not isinstance(cfg, dict):
        return {}
    sleeve = cfg.get("sleeve")
    return sleeve if isinstance(sleeve, dict) else {}


def is_parking_sleeve_enabled(obj: Any) -> bool:
    return bool(parking_sleeve_config(obj).get("enabled", False))


def parking_sleeve_price_tickers(obj: Any) -> list[str]:
    """Tickers the parking sleeve needs daily prices for, or ``[]``.

    ``[spy_symbol, sgov_symbol]`` (upcased, deduped) when
    ``sleeve.enabled`` is truthy; ``[]`` otherwise. Callers append the
    result to their price universe — every call site already dedupes, so
    overlap with watchlist/benchmark coverage is harmless.
    """
    sleeve = parking_sleeve_config(obj)
    if not sleeve.get("enabled", False):
        return []
    spy_symbol = str(sleeve.get("spy_symbol", "SPY")).strip().upper() or "SPY"
    sgov_symbol = str(sleeve.get("sgov_symbol", "SGOV")).strip().upper() or "SGOV"
    return list(dict.fromkeys([spy_symbol, sgov_symbol]))
