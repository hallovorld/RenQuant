#!/usr/bin/env bash
# Daily IV snapshot — captures one EOD chain snapshot per watchlist ticker.
#
# Roadmap C1 (2026-05-18): Alpaca Free Options API gives current snapshots
# only (no historical chains). To build a usable IV-feature panel for
# retrain, we accumulate one snapshot per trading day across 103 tickers.
# After ~6 months we'll have enough panel rows to attempt integration.
#
# Cadence rationale (§5.13.6): IV updates throughout the day; the EOD
# value captures the day's terminal market view. One snapshot per day
# adds one row per ticker — that's the maximum information per tick
# the panel can use (since panel features are daily).
#
# Schedule: weekdays 13:30 PT (16:30 ET) = 30min after market close.
# Skips weekends and holidays implicitly (the API returns empty/stale
# data on non-trading days; the per-symbol parquet's dedupe on as_of
# prevents bad rows from polluting history).

set -e
cd /Users/renhao/git/github/RenQuant
mkdir -p logs/iv_snapshot

LOG="logs/iv_snapshot/daily_$(date +%Y%m%d).log"
echo "=== $(date) — Daily IV snapshot starting ===" > "$LOG"

set -a
source .env
set +a

.venv/bin/python -u scripts/fetch_options_iv_alpaca.py 2>&1 | tee -a "$LOG"

echo "=== $(date) — Daily IV snapshot done ===" | tee -a "$LOG"
