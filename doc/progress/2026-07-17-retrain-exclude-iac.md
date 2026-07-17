# Fix: exclude delisted IAC from the retrain freshness universe

STATUS: delivered
WHAT: weekly_wf_promote.sh passes --exclude-tickers (default IAC,
env-overridable via RENQUANT_RETRAIN_EXCLUDE_TICKERS) through the retrain
wrapper — the designed bridge for newly-delisted names not yet pruned from
the versioned universe inventory.
WHY/DIR: the 2026-07-17 VIX-anomaly-gated retrain failed fail-closed at
the zero-tolerance freshness gate on ONE name: IAC's bars ceased
2026-05-12 (293/294 siblings update daily — delisting-grade evidence).
The gate is correct; the universe declaration was stale.
EVIDENCE: freshness alert "1/294 panel tickers stale... Worst: IAC";
data/ohlcv/IAC/1d.parquet last row 2026-05-12.
NEXT: durable fix = the inventory generator learns delistings (prune IAC
from tier lists at regeneration); remove the default once landed.
