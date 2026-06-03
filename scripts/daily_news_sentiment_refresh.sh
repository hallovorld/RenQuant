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

set -eo pipefail
REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
GITHUB_DIR="$(dirname "$REPO_DIR")"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
mkdir -p logs/news_daily

LOG="logs/news_daily/$(date +%Y-%m-%d).log"
echo "=== $(date) — Daily news+sentiment refresh starting ===" > "$LOG"

fail_missing_strategy_config() {
    local strict_env="$1"
    if renquant_strict_enabled "$strict_env"; then
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable and strict multirepo mode is enabled" \
            | tee -a "$LOG"
    else
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable; scheduled refresh defaults to fail-closed multirepo execution" \
            | tee -a "$LOG"
    fi
}

fail_multirepo_unavailable() {
    local module="$1" strict_env="$2"
    if renquant_strict_enabled "$strict_env"; then
        echo "ERROR: $module unavailable and strict multirepo mode is enabled" \
            | tee -a "$LOG"
    else
        echo "ERROR: $module unavailable; scheduled refresh defaults to fail-closed multirepo execution" \
            | tee -a "$LOG"
    fi
    exit 1
}

set -a
source .env
set +a

if ! STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    fail_missing_strategy_config RQ_DAILY_NEWS_STRICT
    exit 1
fi

# 1. Fetch yesterday's news (and weekend backlog on Mondays)
echo "--- step 1: fetch Alpaca News ---" >> "$LOG"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-model renquant-base-data renquant-common):${PYTHONPATH:-}"
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.alpaca_news_refresh  # noqa: F401
PY
then
    "$PYTHON" -u -m renquant_base_data.alpaca_news_refresh \
        --strategy-config "$STRATEGY_CONFIG" \
        --data-dir "$REPO_DIR/data" \
        --json 2>&1 | tee -a "$LOG"
else
    fail_multirepo_unavailable "renquant_base_data.alpaca_news_refresh" RQ_DAILY_NEWS_STRICT
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
else
    fail_multirepo_unavailable "renquant_model_common.news_sentiment_finbert" RQ_DAILY_NEWS_SENTIMENT_STRICT
fi

echo "=== $(date) — Daily news+sentiment done ===" | tee -a "$LOG"
