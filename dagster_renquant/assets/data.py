"""Data-tier assets: raw OHLCV + SEC fundamentals.

Bodies are validate-output stubs (not re-implementations). The point of the
Dagster layer is the dependency graph, not new compute paths.
"""

from __future__ import annotations

from dagster import LegacyFreshnessPolicy, asset

from dagster_renquant._paths import OHLCV_DIR, SEC_FUNDAMENTALS_PARQUET

# 1d after market close — a daily refresh ought to land within 24h.
DAILY_FRESHNESS = LegacyFreshnessPolicy(maximum_lag_minutes=24 * 60)
# 7d for SEC fundamentals — quarterly data; weekly refresh floor.
WEEKLY_FRESHNESS = LegacyFreshnessPolicy(maximum_lag_minutes=7 * 24 * 60)


@asset(
    legacy_freshness_policy=DAILY_FRESHNESS,
    description="Per-ticker daily OHLCV bars under data/ohlcv/<TICKER>/.",
    group_name="data",
)
def ohlcv_data() -> dict:
    """Validate that the OHLCV directory exists and is non-empty."""
    if not OHLCV_DIR.is_dir():
        raise FileNotFoundError(f"OHLCV directory missing: {OHLCV_DIR}")
    tickers = [p.name for p in OHLCV_DIR.iterdir() if p.is_dir()]
    if not tickers:
        raise RuntimeError(f"OHLCV directory empty: {OHLCV_DIR}")
    return {"path": str(OHLCV_DIR), "n_tickers": len(tickers)}


@asset(
    legacy_freshness_policy=WEEKLY_FRESHNESS,
    description="SEC fundamentals daily parquet (asset_growth, etc.).",
    group_name="data",
)
def sec_fundamentals() -> dict:
    """Validate that the SEC fundamentals parquet exists."""
    if not SEC_FUNDAMENTALS_PARQUET.is_file():
        raise FileNotFoundError(
            f"SEC fundamentals missing: {SEC_FUNDAMENTALS_PARQUET}"
        )
    size_mb = SEC_FUNDAMENTALS_PARQUET.stat().st_size / (1024 * 1024)
    return {"path": str(SEC_FUNDAMENTALS_PARQUET), "size_mb": round(size_mb, 2)}
