"""Tax computation utilities for after-tax return analysis."""

import pandas as pd

DEFAULT_TAX_CONFIG = {
    "short_term_rate": 0.50,
    "long_term_rate": 0.32,
    "long_term_threshold_days": 365,
}


def load_tax_config(strategy_config: dict) -> dict:
    tax = strategy_config.get("tax", {})
    return {
        "short_term_rate": tax.get("short_term_rate", DEFAULT_TAX_CONFIG["short_term_rate"]),
        "long_term_rate": tax.get("long_term_rate", DEFAULT_TAX_CONFIG["long_term_rate"]),
        "long_term_threshold_days": tax.get(
            "long_term_threshold_days", DEFAULT_TAX_CONFIG["long_term_threshold_days"]
        ),
    }


def tax_rate_for_holding(
    holding_days: int, short_term_rate: float,
    long_term_rate: float, long_term_threshold_days: int,
) -> float:
    if holding_days >= long_term_threshold_days:
        return long_term_rate
    return short_term_rate


def compute_trade_tax(
    pnl: float, holding_days: int,
    short_term_rate: float = 0.50,
    long_term_rate: float = 0.32,
    long_term_threshold_days: int = 365,
) -> float:
    if pnl <= 0:
        return 0.0
    rate = tax_rate_for_holding(holding_days, short_term_rate, long_term_rate,
                                long_term_threshold_days)
    return pnl * rate


def compute_after_tax_pnl(
    pnl: float, holding_days: int,
    short_term_rate: float = 0.50,
    long_term_rate: float = 0.32,
    long_term_threshold_days: int = 365,
) -> float:
    return pnl - compute_trade_tax(pnl, holding_days, short_term_rate,
                                   long_term_rate, long_term_threshold_days)


def add_tax_columns(trades_df: pd.DataFrame, tax_config: dict) -> pd.DataFrame:
    df = trades_df.copy()
    if df.empty:
        df["holding_days"] = []
        df["tax_rate"] = []
        df["tax"] = []
        df["after_tax_pnl"] = []
        return df

    short_rate = tax_config["short_term_rate"]
    long_rate  = tax_config["long_term_rate"]
    threshold  = tax_config["long_term_threshold_days"]

    df["holding_days"] = (df["exit_time"] - df["entry_time"]).dt.days
    df["tax_rate"] = df["holding_days"].apply(
        lambda d: tax_rate_for_holding(d, short_rate, long_rate, threshold)
    )
    df["tax"] = df.apply(
        lambda r: compute_trade_tax(r["pnl"], r["holding_days"],
                                    short_rate, long_rate, threshold),
        axis=1,
    )
    df["after_tax_pnl"] = df["pnl"] - df["tax"]
    return df


__all__ = [
    "load_tax_config", "tax_rate_for_holding", "compute_trade_tax",
    "compute_after_tax_pnl", "add_tax_columns",
]
