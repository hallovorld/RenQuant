#!/usr/bin/env bash
# Daily news + sentiment refresh — pulls yesterday's Alpaca News then scores
# with FinBERT, appending to per-ticker parquets.
#
# Roadmap C5 step 3 (2026-05-18). Adds ~5-50 articles per ticker per day
# (varies with news flow); FinBERT scoring on the daily delta is ~5sec
# on MPS (M2 Pro).
#
# Cadence rationale (§5.13.6): sentiment features feed live inference
# at every cron firing. Per-day refresh adds yesterday's news_flow + the
# weekend backlog (Saturday/Sunday articles land in Monday's fetch).
# Sub-daily would burn API calls without changing the daily aggregate.
#
# Schedule: weekdays 06:30 PT (07:00 ET), before daily_104 (07:30 PT).
#
# Install:
#   cp scripts/launchd/com.renquant.daily-news-sentiment.plist \
#      ~/Library/LaunchAgents/
#   launchctl load \
#      ~/Library/LaunchAgents/com.renquant.daily-news-sentiment.plist

set -e
cd /Users/renhao/git/github/RenQuant
mkdir -p logs/news_daily

LOG="logs/news_daily/$(date +%Y-%m-%d).log"
echo "=== $(date) — Daily news+sentiment refresh starting ===" > "$LOG"

set -a
source .env
set +a

# 1. Fetch yesterday's news (and weekend backlog on Mondays)
echo "--- step 1: fetch Alpaca News ---" >> "$LOG"
.venv/bin/python -u scripts/fetch_news_alpaca.py 2>&1 | tee -a "$LOG"

# 2. Re-score with FinBERT (per-ticker parquet append; idempotent)
echo "--- step 2: FinBERT score ---" >> "$LOG"
.venv/bin/python -u scripts/score_news_finbert.py 2>&1 | tee -a "$LOG"

echo "=== $(date) — Daily news+sentiment done ===" | tee -a "$LOG"
