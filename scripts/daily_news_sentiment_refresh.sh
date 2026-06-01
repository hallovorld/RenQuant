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
REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
GITHUB_DIR="$(dirname "$REPO_DIR")"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
mkdir -p logs/news_daily

LOG="logs/news_daily/$(date +%Y-%m-%d).log"
echo "=== $(date) — Daily news+sentiment refresh starting ===" > "$LOG"

set -a
source .env
set +a

# 1. Fetch yesterday's news (and weekend backlog on Mondays)
echo "--- step 1: fetch Alpaca News ---" >> "$LOG"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-model renquant-base-data renquant-common):${PYTHONPATH:-}"
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.alpaca_news_refresh  # noqa: F401
PY
then
    "$PYTHON" -u -m renquant_base_data.alpaca_news_refresh \
        --strategy-config "$REPO_DIR/backtesting/renquant_104/strategy_config.json" \
        --data-dir "$REPO_DIR/data" \
        --json 2>&1 | tee -a "$LOG"
elif [ "${RQ_DAILY_NEWS_STRICT:-0}" = "1" ]; then
    echo "ERROR: renquant_base_data.alpaca_news_refresh unavailable and RQ_DAILY_NEWS_STRICT=1" \
        | tee -a "$LOG"
    exit 1
else
    echo "WARN: renquant_base_data.alpaca_news_refresh unavailable; falling back to umbrella script." \
        | tee -a "$LOG"
    "$PYTHON" -u scripts/fetch_news_alpaca.py 2>&1 | tee -a "$LOG"
fi

# 2. Re-score with FinBERT (per-ticker parquet append; idempotent)
echo "--- step 2: FinBERT score ---" >> "$LOG"
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_model_common.news_sentiment_finbert  # noqa: F401
PY
then
    "$PYTHON" -u -m renquant_model_common.news_sentiment_finbert \
        --data-dir "$REPO_DIR/data" \
        --json 2>&1 | tee -a "$LOG"
elif [ "${RQ_DAILY_NEWS_SENTIMENT_STRICT:-0}" = "1" ]; then
    echo "ERROR: renquant_model_common.news_sentiment_finbert unavailable and RQ_DAILY_NEWS_SENTIMENT_STRICT=1" \
        | tee -a "$LOG"
    exit 1
else
    echo "WARN: renquant_model_common.news_sentiment_finbert unavailable; falling back to umbrella script." \
        | tee -a "$LOG"
    "$PYTHON" -u scripts/score_news_finbert.py 2>&1 | tee -a "$LOG"
fi

echo "=== $(date) — Daily news+sentiment done ===" | tee -a "$LOG"
