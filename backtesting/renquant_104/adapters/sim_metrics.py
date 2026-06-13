"""Sim reporting metrics & tax-debit helpers — sim.py decomposition slice 1.

EXTRACTED 2026-06-13 from adapters/sim.py (eng plan S2 item 5). Pure
reporting/stats math (annualized net equity curve, finite-guarded
aggregates, quantiles) and the tax cash-debit mode/amount config readers.
No SimAdapter state. Moved verbatim; re-exported from sim for back-compat.
These feed build_result() reporting, NOT the decision path — so they are
gated by unit tests + the line-faithful move (the DRPH replay checks
decision parity, which these do not touch).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


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


def _finite_attr_values(items: list[Any], attr: str) -> list[float]:
    vals: list[float] = []
    for item in items:
        value = _finite_float(getattr(item, attr, None), default=float("nan"))
        if math.isfinite(value):
            vals.append(value)
    return vals


def _mean_or_nan(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _quantile_or_nan(vals: list[float], q: float) -> float:
    return float(np.quantile(vals, q)) if vals else float("nan")


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
        "event_cash_debit": "event_level",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"event_level", "reporting_only"}:
        raise ValueError(
            f"Unknown tax.cash_debit_mode={raw!r}; expected event_level or "
            "reporting_only"
        )
    return mode


def _tax_cash_debit_amount(config: dict | None, tax: float) -> float:
    tax_f = _finite_float(tax, default=0.0)
    if tax_f <= 0.0:
        return 0.0
    if _tax_cash_debit_mode(config) == "reporting_only":
        return 0.0
    return tax_f
