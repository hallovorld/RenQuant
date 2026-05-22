#!/usr/bin/env python
"""Trade-ledger contract gates for renquant_104 WF acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeContractReport:
    passed: bool
    reason: str
    evidence: dict[str, Any]


def evaluate_trade_contract(
    round_trips: pd.DataFrame,
    *,
    require_entry_mu: bool = False,
    require_entry_sigma: bool = False,
    require_exit_regime: bool = True,
    require_exit_thresholds: bool = True,
) -> TradeContractReport:
    """Require round-trip ledgers to carry model and policy provenance.

    A profitable or losing trade cannot be audited scientifically if the
    durable ledger omits the model μ/σ used for sizing/QP, the exit-side
    regime, or the stop/trailing thresholds active at exit. Missing fields
    therefore fail the WF gate before any Sharpe metadata can be promoted.
    """
    if round_trips is None or round_trips.empty:
        return TradeContractReport(False, "no round-trip rows", {"n_rows": 0})
    df = round_trips.copy()
    status = (
        df["status"].astype(str).str.lower()
        if "status" in df.columns else
        pd.Series(["closed"] * len(df), index=df.index)
    )
    auditable = df[status.isin({"closed", "open"})]
    closed = df[status == "closed"]
    issues: list[str] = []
    evidence: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_auditable": int(len(auditable)),
        "n_closed": int(len(closed)),
    }

    if require_entry_mu:
        n = _missing_finite(auditable, "entry_mu")
        evidence["missing_entry_mu"] = n
        if n:
            issues.append(f"{n} auditable trade(s) missing finite entry_mu")
    if require_entry_sigma:
        n = _missing_finite(auditable, "entry_sigma")
        evidence["missing_entry_sigma"] = n
        if n:
            issues.append(f"{n} auditable trade(s) missing finite entry_sigma")
    if require_exit_regime:
        n = _missing_nonempty(closed, "exit_regime")
        evidence["missing_exit_regime"] = n
        if n:
            issues.append(f"{n} closed trade(s) missing exit_regime")
    if require_exit_thresholds:
        cols = [
            "exit_stop_loss_pct",
            "exit_max_single_day_loss_pct",
            "exit_sdl_n_sigma",
            "exit_trailing_stop_trigger_pct",
            "exit_trailing_stop_trail_pct",
            "exit_max_hold_days",
        ]
        missing_cols = [c for c in cols if c not in closed.columns]
        evidence["missing_exit_threshold_columns"] = missing_cols
        if missing_cols:
            issues.append("missing exit threshold columns: " + ",".join(missing_cols))

    if issues:
        return TradeContractReport(False, "; ".join(issues), evidence)
    return TradeContractReport(True, "trade ledger contract OK", evidence)


def _missing_finite(df: pd.DataFrame, col: str) -> int:
    if df.empty:
        return 0
    if col not in df.columns:
        return int(len(df))
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return int(s.isna().sum())


def _missing_nonempty(df: pd.DataFrame, col: str) -> int:
    if df.empty:
        return 0
    if col not in df.columns:
        return int(len(df))
    s = df[col]
    missing = s.isna() | (s.astype(str).str.strip() == "")
    return int(missing.sum())
